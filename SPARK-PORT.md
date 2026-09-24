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
