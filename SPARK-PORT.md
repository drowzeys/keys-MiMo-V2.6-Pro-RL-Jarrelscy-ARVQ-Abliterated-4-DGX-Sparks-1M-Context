# Spark port notes

These are the fixes between Jarrelscy's `sm120` fork (`88c94233`) and the first successful token on GB10 / SM121. The fork is still the runtime. These notes are the delta.

## 1. Flash-attention package

The fork tree shadows the base vLLM install and ships a `vllm_flash_attn` stub that exports `compile_flash_attn_varlen_func_from_specs` but has no `layers/`, `cute/`, or `ops/`. Replacing the whole package with the base image's copy removes that symbol and model inspection fails. Leaving the stub makes `vllm.vllm_flash_attn.layers` fail after the weights load.

`serve/entrypoint.sh` keeps the fork's `vllm_flash_attn/__init__.py` and links only the missing subpackages from the base image. `cute/` is linked as a real directory of symlinks. A symlink of the `cute` directory itself makes the fork treat it as upstream flash-attn source and rewrite `sys.modules`.

## 2. MTP QKV scales

Pro QKV is stored grouped per KV head, packed as eight groups. The scale tensor is 216 rows. A plain TP4 split yields `[54, 48]`. The parameter is `[53, 48]`.

The target model already requantizes that layout in `_shard_fp8_qkv_proj`. The MTP loader has to call the same function. After that, every rank gets weight `[6784, 6144]` and scale `[53, 48]`.

The checkpoint contains MTP layers 0, 1, and 2. `MiMoV2MultiTokenPredictor` in the fork sets `num_mtp_layers = 1`, so extra speculative tokens replay layer 0. That is the configuration behind the MTP=2 champion number.

## 3. Fused SiLU op import

Loading the MTP draft imports the activation-quant fusion pass before `vllm._C` has registered `silu_and_mul_quant`. The op exists in this image. Import `vllm._C` at the top of `act_quant_fusion.py` before the fused-op table is built.

## 4. Memory

`--language-model-only` is required for the 1M text pool on this build. Without it, startup profiles one maximum-size video and the KV budget drops by several GiB per rank.

`--gpu-memory-utilization` stays at **0.85**. Do not raise it to buy context.

Expert parallelism is unsupported in this quant method. The serve is TP4 only. Expert tensors stay compressed at decode.

## 5. RoCE

`NCCL_NET=IB` on the 200G NIC. During one short generation each node's RoCE transmit counter moved by tens of mebibytes while the TCP counters on that NIC moved kilobytes. The GID index is per node: an index that is correct on three Sparks was wrong on the fourth.

## 6. All three MTP heads, non-chain (2026-09-26)

The checkpoint ships three MTP heads (`model.mtp.layers.0-2`), one per draft step. Xiaomi's recommended SGLang deploy runs them as multi-layer EAGLE. The fork hardcoded `num_mtp_layers = 1` and skipped loading heads 1-2 without a warning, so every draft step re-ran head 0.

The published image fixes this in three places:

- `mimo_v2_mtp.py` builds `n_predict` heads (`MIMO_MTP_LAYERS`, default 3 in the image). The count comes from `speculative_config.draft_model_config.hf_config`. Inside the drafter, `vllm_config.model_config` is the target's config, so reading it there silently yields 1.
- `speculative.py` routes multi-layer MiMo MTP to the per-step proposer.
- `mimo_v2_mtp.py` replaces that proposer's `propose()` for MiMo with **non-chain** semantics, which match SGLang's `multi_layer_eagle_worker_v2` for MiMo. Every head runs over the whole new-token span with the **target's** hidden states and the same positions. Only the token ids shift by one, with the previous head's draft in the last slot. So each head fills its own KV cache. The Step3.5 chain style, which feeds head k's output hidden into head k+1 on a single token, is wrong for MiMo.

Measured greedy per-position acceptance on prose:

| Target | pos 0 | pos 1 | pos 2 | tokens/pass (k=3) |
|---|---:|---:|---:|---:|
| Stock ARVQ `63430f7` | 0.75 | 0.30 | 0.07 | 2.12 |
| Abliterated `dealign-op` | 0.67 | 0.25 | 0.05 | 1.97 |

Heads 1-2 are weak on both trees. ARVQ quantization and the `o_proj` edits moved the hidden state the heads consume. On sampled prose, k=2 is still the champion until the heads are fine-tuned on-policy against this target, which is in progress.

## 7. Speed path (2026-09-26)

- **CUDA graphs.** torch.compile with `FULL_AND_PIECEWISE` works on this fork. The ablit capture hooks in `mimo_v2.py` used to do a file `open()` per layer per forward. That meant 140 syscalls per step and a graph break. They are now gated once at import (`MIMO_ABLIT_CAPTURE=1` re-enables them).
- **ARVQ prefill.** By default, prefill ran through the per-slot decode kernel, at 128 tok/s. `VLLM_ARVQ_GROUPED_PREFILL=1`, `_COMPACT_PREFILL=1` and `_SORT_NATIVE_PREFILL=1`, plus `--max-num-batched-tokens 5120`, give about 530 tok/s. 5120 is the largest chunk that stays under the grouped path's 1 GiB FP32 output bound. All four knobs are defaults in the image. The batched prefill in section 8 supersedes the grouped path.
- **Sliding window** is working on the 60 SWA layers (window 128), in both the KV cache (only the 10 full-attention layers hold per-token KV) and the Triton DiffKV kernel.
- **All-reduce** is PyNCCL over RoCE. It is about half of each decode pass. `NCCL_PROTO=Simple` changed nothing.

## 8. Expert-batched ARVQ prefill (2026-09-26)

Profile of an 11K-token prefill with the older grouped path (rank 0, 24.4 s of GPU time):
- **Native `hybrid_kernel` on hot routes: 7.1 s.** It runs 4 FP4 activation planes.
- **The grouped cold path: about 9.5 s.** It is a per-expert loop of about 43K dequant launches and 43K matmuls, and the CPU was launch-bound ("Command Buffer Full").
- **NCCL all-reduce: 4.5 s.**

v3 adds `nvfp4_arvq_batched_prefill.py` and `grouped.cu`:

- **Two kernels.** One gate/up and one down GEMM kernel covers all routes, cold (ARVQ 8+8 codebook, decoded in registers from a shared-memory LUT with the hardware FP4 convert) and hot (NVFP4).
- **Arithmetic.** FP16 tensor cores with FP32 accumulation. The down projection adds into an FP32 output with vector atomics. The grid is sliced into 512 columns for L2 reuse.
- **Gate.** `arvq_mlp` takes this path when `VLLM_ARVQ_BATCHED_PREFILL=1`, the chunk has at least `VLLM_ARVQ_BATCHED_MIN_TOKENS` tokens (default 32), and the stream is not capturing a CUDA graph. Decode and MTP steps keep the native kernels.
- **mcbook16 (v5) layers** are rejected by `supported()` and use the old paths. This checkpoint is v4 throughout.

Harness, per MoE layer on one TP4 rank:

| Layer | Tokens | Native | Grouped (old) | Batched (now) |
|---|---:|---:|---:|---:|
| 30 (all cold) | 5120 | 489 ms | 90 ms | 18.2 ms |
| 64 (291 cold, 93 hot) | 5120 | 623 ms | 147 ms | 18.9 ms |

Accuracy of the batched path:
- Cosine vs native is 0.9999996. Relative Frobenius error vs an FP32 reference is 1.709e-3, compared with 1.717e-3 for native, 1.735e-3 for grouped, and a 1.658e-3 bf16 floor.
- Greedy outputs on the cluster drift from the grouped build at tokens 31-53. That matches run-to-run drift within one boot (tokens 53-58), which comes from FP32 atomics and NCCL.

Cluster result: 9.5K-token prefill goes from 505 to 1,291 tok/s, and 38K-token from 527 to 1,038 tok/s. All-reduce is now the largest prefill cost.
