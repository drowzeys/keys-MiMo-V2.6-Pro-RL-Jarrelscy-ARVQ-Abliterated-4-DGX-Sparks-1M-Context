# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only MiMo-V2 MTP (Multi-Token Prediction) draft model.

Supports both MiMo-V2-Pro and MiMo-V2-Flash checkpoints.

Checkpoint weight layout (model.mtp.layers.{idx}.*):
  enorm            - RMSNorm for token embeddings
  hnorm            - RMSNorm for previous hidden states
  eh_proj          - ReplicatedLinear(hidden*2 -> hidden)
  input_layernorm  - pre-attention RMSNorm
  self_attn.*      - attention weights; format differs by variant:
                       Pro:   fused qkv_proj  [Q;K;V] concatenated
                       Flash: separate q_proj, k_proj, v_proj
  pre_mlp_layernorm - post-attention / pre-MLP RMSNorm
  mlp.*            - dense MLP (gate_proj / up_proj / down_proj)
  final_layernorm  - norm applied before logit computation
"""

from collections.abc import Iterable

import os

import torch
import torch.nn as nn

from vllm.logger import init_logger
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .mimo_v2 import MiMoV2Attention, MiMoV2MLP, _shard_fp8_qkv_proj
from .utils import _merge_multimodal_embeddings, maybe_prefix

# MiMo-V2 checkpoints contain multiple MTP layers, but vLLM currently supports
# only the first layer
# The Pro checkpoint ships three MTP layers (model.mtp.layers.0-2), one per
# draft step. MIMO_MTP_LAYERS>1 builds and loads them; the Step3.5 per-step
# proposer then runs layer i for draft step i.
logger = init_logger(__name__)

_MIMO_V2_PRO_NUM_MTP_LAYERS = int(os.environ.get("MIMO_MTP_LAYERS", "1"))
_MIMO_V2_FLASH_NUM_MTP_LAYERS = 1


class MiMoV2MTPLayer(nn.Module):
    """Single MTP predictor layer for MiMo-V2 (Pro and Flash).

    Mirrors the single-layer MiMo-V2 nextn reference implementation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        # Predictor head components
        self.enorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        # MTP uses the SWA attention configuration
        # implementation.
        swa_rope_theta = getattr(
            config,
            "swa_rope_theta",
            getattr(config, "rope_theta", 1000000),
        )
        sliding_window_size = getattr(config, "sliding_window_size", -1)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attn = MiMoV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.swa_num_attention_heads,
            num_kv_heads=config.swa_num_key_value_heads,
            head_dim=config.swa_head_dim,
            v_head_dim=getattr(config, "swa_v_head_dim", None),
            v_scale=getattr(config, "attention_value_scale", None),
            sliding_window_size=sliding_window_size,
            attention_bias=config.attention_bias,
            add_swa_attention_sink_bias=getattr(
                config, "add_swa_attention_sink_bias", False
            ),
            layer_id=0,
            rope_theta=swa_rope_theta,
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            quant_config=quant_config,
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            prefix=f"{prefix}.self_attn",
        )
        self.pre_mlp_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.mlp = MiMoV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Combine token embedding and previous hidden state
        h, _ = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
            )
        )

        # Transformer block with fused residual norms
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(positions=positions, hidden_states=h)
        h, residual = self.pre_mlp_layernorm(h, residual)
        h = self.mlp(h)
        h = h + residual

        return self.final_layernorm(h)


class _MiMoV2MTPLayers(nn.Module):
    """Thin wrapper so parameter paths match checkpoint: model.mtp.layers.*"""

    def __init__(
        self,
        config: PretrainedConfig,
        num_mtp_layers: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): MiMoV2MTPLayer(
                    config=config,
                    prefix=f"{prefix}.{i}",
                    quant_config=quant_config,
                )
                for i in range(num_mtp_layers)
            }
        )


class MiMoV2MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        spec_cfg = vllm_config.speculative_config
        assert spec_cfg is not None
        # model_config here is the target's; n_predict lives on the draft config.
        draft_hf = spec_cfg.draft_model_config.hf_config
        num_mtp_layers = getattr(draft_hf, "n_predict", None) or 1

        self.num_mtp_layers = num_mtp_layers
        logger.info("MiMo-V2 MTP: building %d MTP layer(s)", num_mtp_layers)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.mtp = _MiMoV2MTPLayers(
            config=config,
            num_mtp_layers=num_mtp_layers,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "mtp.layers"),
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.mtp.layers[str(current_step_idx)](
            inputs_embeds, positions, previous_hidden_states
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.logits_processor(lm_head, hidden_states)


class MiMoV2MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = MiMoV2MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # Flash format: separate projections → fused qkv_proj
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Pro QKV is grouped per KV head. TP4 owns two groups, so the FP8
        # scales cannot be chunked; shard them the same way as the target.
        pending_fp8_qkv: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only load MTP-related weights, shared embeddings, and lm_head
            if (
                "model.mtp" not in name
                and "model.embed_tokens" not in name
                and not name.startswith("lm_head")
            ):
                continue

            if name.endswith("qkv_proj.weight") or name.endswith(
                "qkv_proj.weight_scale_inv"
            ):
                if (
                    name.endswith("qkv_proj.weight")
                    and loaded_weight.dtype != torch.float8_e4m3fn
                ):
                    continue
                prefix, kind = name.rsplit(".", 1)
                weight_name = f"{prefix}.weight"
                if weight_name not in params_dict:
                    continue
                entry = pending_fp8_qkv.setdefault(prefix, {})
                entry[kind] = loaded_weight
                if "weight" not in entry or "weight_scale_inv" not in entry:
                    continue
                del pending_fp8_qkv[prefix]
                attn = self.get_submodule(prefix.rsplit(".", 1)[0])
                w_rank, s_rank = _shard_fp8_qkv_proj(
                    entry["weight"],
                    entry["weight_scale_inv"],
                    num_heads=attn.total_num_heads,
                    num_kv_heads=attn.total_num_kv_heads,
                    head_dim=attn.head_dim,
                    v_head_dim=attn.v_head_dim,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                )
                for tensor, param_name in (
                    (w_rank, weight_name),
                    (s_rank, f"{prefix}.weight_scale_inv"),
                ):
                    param = params_dict[param_name]
                    if tensor.shape[0] > param.shape[0]:
                        tensor = tensor[: param.shape[0]]
                    default_weight_loader(param, tensor)
                    loaded_params.add(param_name)
                continue

            # gate_proj/up_proj → gate_up_proj stacking (both formats);
            # Flash: q_proj/k_proj/v_proj → qkv_proj merging.
            stacked_matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)
                if (
                    name_rewritten.endswith(".bias")
                    and name_rewritten not in params_dict
                ):
                    continue
                if name_rewritten not in params_dict:
                    continue
                param = params_dict[name_rewritten]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_rewritten)
                stacked_matched = True
                break

            if stacked_matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            # attention_sink_bias is head-parallel; slice by tp
            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * heads_per_rank, heads_per_rank
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class MiMoV2OmniMTP(MiMoV2MTP, SupportsMultiModal):
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds


# ---------------------------------------------------------------------------
# Non-chain multi-layer MTP (SGLang multi_layer_eagle semantics for MiMo-V2).
#
# SGLang runs MiMo MTP layer k over the whole draft-extend span with the
# TARGET hidden states and unchanged positions; only the token ids rotate:
# layer k sees x_{i+1+k} at position i, with layer k-1's draft appended at
# each request's last valid slot. Every layer therefore fills its own KV for
# the full context. The Step3.5 proposer instead chains each layer's output
# hidden into the next and runs layers >0 on one token, which leaves their KV
# mostly empty. This replaces propose() for MiMo-V2 drafters only.
# ---------------------------------------------------------------------------
def _mimo_nonchain_propose(
    self,
    num_speculative_tokens,
    target_token_ids,
    target_positions,
    target_hidden_states,
    next_token_ids,
    token_indices_to_sample,
    common_attn_metadata,
    sampling_metadata,
    mm_embed_inputs=None,
    num_rejected_tokens_gpu=None,
    slot_mappings=None,
):
    from vllm.forward_context import set_forward_context

    self.num_speculative_tokens = num_speculative_tokens
    self._last_draft_probs = None
    num_tokens, token_indices_to_sample, common_attn_metadata = (
        self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
    )
    _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
        common_attn_metadata
    )
    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        self._determine_batch_execution_and_padding(num_tokens)
    )
    model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
        num_tokens, num_input_tokens, mm_embed_inputs
    )
    if model_kwargs.get("inputs_embeds") is not None:
        raise ValueError("MiMo non-chain MTP requires token-id inputs")
    input_ids = model_kwargs["input_ids"]
    slot_mapping = self._get_slot_mapping(
        slot_mapping_size, common_attn_metadata.slot_mapping
    )

    drafts, probs = [], []
    for k in range(num_speculative_tokens):
        model_kwargs["spec_step_idx"] = k
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping,
        ):
            ret = self.model(**model_kwargs)
        last_hidden = ret[0] if self.model_returns_tuple() else ret
        draft, draft_probs = self._sample_draft_tokens_for_step(
            last_hidden[token_indices_to_sample], sampling_metadata, spec_step_idx=k
        )
        drafts.append(draft)
        if draft_probs is not None:
            probs.append(draft_probs)
        if k + 1 < num_speculative_tokens:
            ids = input_ids[:num_tokens]
            ids[:-1] = ids[1:].clone()
            ids[token_indices_to_sample] = draft.to(ids.dtype)

    if probs:
        self._last_draft_probs = torch.stack(probs, dim=1).contiguous()
    return torch.stack(drafts, dim=1)


def _install_mimo_nonchain_propose() -> None:
    if os.environ.get("MIMO_MTP_NONCHAIN", "1") != "1":
        return
    from vllm.v1.spec_decode import step3p5 as _step3p5

    cls = _step3p5.Step3p5MTPProposer
    if getattr(cls, "_mimo_nonchain_installed", False):
        return
    original = cls.propose

    def propose(self, *args, **kwargs):
        hf = self.vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(hf, "model_type", None) == "mimo_v2_mtp":
            return _mimo_nonchain_propose(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    cls.propose = propose
    cls._mimo_nonchain_installed = True
    logger.info("MiMo-V2 MTP: non-chain multi-layer propose installed")


try:
    _install_mimo_nonchain_propose()
except Exception as exc:  # the API server process may lack worker modules
    logger.warning("MiMo-V2 MTP: non-chain propose not installed: %s", exc)
