# MiMo-V2.6-Pro-RL — Jarrelscy ARVQ hybrid on 4 DGX Sparks, 1M context

Serving recipe for **[XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)** after **[Jarrelscy](https://huggingface.co/jarrelscy)**'s ARVQ / NVFP4 hybrid quantization, on **four NVIDIA DGX Spark (GB10)** nodes, tensor-parallel 4, **1,048,576-token context**.

This repository does not redistribute the weights or Jarrelscy's runtime. It records the Spark bring-up that those weights made possible, the champion draft setting, and two slots that are not finished.

## Credit

The quantization is Jarrelscy's. Official MiMo-V2.6-Pro images read the source MXFP4 / FP8 expert layout. They do not load this checkpoint. Jarrelscy's hybrid keeps a small hot-expert set in NVFP4 and the remaining routed experts in ARVQ codebooks, which is what fits the model across four 128 GB Sparks.

| Piece | Author | Where |
|---|---|---|
| ARVQ / NVFP4 hybrid checkpoint | Jarrelscy | [jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) @ `63430f7b9c1b13f4bfca9e3bc3969ec0115d1a88` |
| vLLM fork that loads `nvfp4_arvq_hybrid` | Jarrelscy | [jarrelscy/vllm-mimo-v26-arvq-sm120](https://github.com/jarrelscy/vllm-mimo-v26-arvq-sm120) @ `88c94233120247f275ec94baf21638321a930469` |
| Base model | Xiaomi MiMo | [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) |
| Four-Spark serve, MTP measurement, Spark port notes | Keys | this repo |

Jarrelscy marks full-model quality and SM120 / 1M serving as **unqualified**. The numbers below are a serving measurement on GB10 (SM121), not a quality claim.

## Champion

**MTP = 2** draft tokens, selected for single-stream prose. The draft is the checkpoint's own MTP stack (`model.mtp.layers.0`, `.1`, and `.2`). The fork as published loads one MTP layer and replays it, so "2" here means two speculative steps on that built-in head, not a second downloaded drafter.

| | |
|---|---|
| Nodes | 4× DGX Spark GB10, one GPU each, TP4, PP1 |
| Context | `1048576` |
| KV | BF16 (`--kv-cache-dtype` left at auto). FP8 KV is not used |
| GPU memory fraction | **0.85** (do not raise this on GB10) |
| Scheduler | `max-num-seqs 4`, `max-num-batched-tokens 2048`, chunked prefill, prefix caching |
| Execution | eager. CUDA graphs are off |
| Modalities | `--language-model-only` so the startup profile does not spend the KV budget on a video |
| Draft | `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` |
| Served name | `MiMo-V2.6-Pro-ARVQ` |

Measured KV pool on the champion boot: **2,248,773 tokens**. Weights about **73.2 GiB per rank**.

### Prose, 33K-token prompt, 512 new tokens, temperature 1.0

Single-stream mean of three stories (beekeeper, lighthouse, nurse):

| Draft tokens | Decode | Tokens per pass |
|---:|---:|---:|
| 1 | 17.4 tok/s | 1.58 |
| **2** | **18.6 tok/s** | **1.89** |
| 3 | 16.1 tok/s | 1.84 |

Same prompt, requests actually overlapped (`max_num_seqs 4`):

| Requests | 1 draft token | 2 draft tokens | 3 draft tokens |
|---:|---:|---:|---:|
| 2 | 29.6 tok/s | 27.9 tok/s | 23.4 tok/s |
| 4 | **45.0 tok/s** | 39.1 tok/s | 33.1 tok/s |

MTP=2 is the single-stream champion. MTP=1 is faster when four requests run together. A short prompt with one draft token, measured earlier on the eager server, was 24.5 tok/s. The 33K numbers above are the loaded figure.

## Bring-up

Build Jarrelscy's fork for the Spark (the image used here was compiled for `sm_121a` from that `sm120` tree). Put the checkpoint on storage the four ranks can read. Then, on each node:

```bash
# rank 0 is the API. ranks 1–3 are headless.
# GID is the IPv4 RoCE index for that node's HCA. It is not the same on every Spark.
export LM_ONLY=1
export MAXLEN=1048576
export SEQS=4
export SPEC='{"method":"mtp","num_speculative_tokens":2}'
export IMAGE=mimo26-arvq-spark:63430f7-sm121-v1
export MASTER_ADDR=10.0.0.1   # rank 0
export HOSTPATH=/path/to/MiMo-V2.6-Pro-RL-ARVQ-hybrid-63430f7
export GID_INDEX=3            # confirm with show_gids; one node in this cluster needed 7

bash serve/launch-rank.sh "$HEAD_IP" "$RANK" "$GID_INDEX" "$HOSTPATH" headless
# rank 0:
bash serve/launch-rank.sh "$HEAD_IP" 0 "$GID_INDEX" "$HOSTPATH" api
```

NCCL on this cluster used the 200G RoCE NIC (`NCCL_NET=IB`), not the TCP path on that same device. See [SPARK-PORT.md](SPARK-PORT.md) for the three loader fixes required before the first token.

## Reserved

- **[DFlash](DFLASH.md)** — weights are already in the checkpoint (`dflash/`). Spark test is in progress. No speed number yet.
- **[Abliteration](ABLITERATION.md)** — not applied to this Pro checkpoint. No refusal score is claimed here.

## License

Recipe text in this repo is MIT. The checkpoint and the vLLM fork keep their own licenses (the uploaded snapshot's card is MIT; the fork follows upstream vLLM).
