# keys-MiMo-V2.6-Pro-RL Jarrelscy ARVQ Abliterated — 4 DGX Sparks, 1M context

Serving recipe for the **abliterated** [Jarrelscy ARVQ / NVFP4 hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) of **[XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)** on **four NVIDIA DGX Spark (GB10)** nodes, tensor-parallel 4, **1,048,576-token context**.

Gated weights (automatic approval after terms): **[drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated)**.

The launcher enables MiMo tool calling on the server (`--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo`). Hermes then **executes** those calls (`write_file`, `terminal`, `execute_code`, `read_file`, …) so a prompt can write a project, run it, and iterate. See [HERMES.md](HERMES.md) and [`serve/verify-tools-and-build.sh`](serve/verify-tools-and-build.sh).

## Current status — 2026-09-26 UTC

- **Serving:** live on four Sparks with 1M context. MTP=2 over all three built-in heads, **CUDA graphs on**, four sequences, BF16 KV, GPU memory fraction 0.85. Image **v3**.
- **Prose: 20.4–20.8 tok/s single-stream** (was 18.6 eager). **Prefill: 1,038–1,291 tok/s** (was 128). 38K-token time to first token: **36.6 s** (was 302 s). The prefill gain comes from new batched ARVQ expert kernels in image v3.
- **✅ Tool-call loop FIXED (2026-09-26).** Hermes no longer loops forever on big tool batches. Truncated batches now return `finish_reason: "length"`, and the default output cap is 8192, up from 2048. See [HERMES.md](HERMES.md#fixed-2026-09-26-never-ending-tool-call-loop).
- **[Hermes and tool calling](HERMES.md):** server parsers on; Hermes `hermes-cli` / `hermes-telegram` execute `write_file`, `terminal`, `execute_code`. A prompt can scaffold a file, run it, and return the program output. `tool_use_enforcement: true` so the model calls tools instead of describing them.
- **[Abliteration](ABLITERATION.md):** live `dealign-op` tree. Thinking **off** **32/32** refusal and **22/22** cyber; thinking **on** 25/32 and 16/22 (visible content). Gated HF: [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated).
- **[DFlash](DFLASH.md):** measured and slower than MTP=2. MTP=2 remains the serving choice.
- **Vision:** the live serve is text-only. The separate Hermes vision endpoint was offline at verification.

Canonical snapshot: [serve/verification/current-status.json](serve/verification/current-status.json) (thinking-off 32/32 · 22/22, thinking-on 25/32 · 16/22, live `write_file`+`terminal` build stdout 42). The earlier [2026-09-25-status.json](serve/verification/2026-09-25-status.json) is the pre-champion tool-parser check on l68t.

## Credit

The quantization is Jarrelscy's. Official MiMo-V2.6-Pro images read the source MXFP4 / FP8 expert layout. They do not load this checkpoint. Jarrelscy's hybrid keeps a small hot-expert set in NVFP4 and the remaining routed experts in ARVQ codebooks, which is what fits the model across four 128 GB Sparks.

| Piece | Author | Where |
|---|---|---|
| ARVQ / NVFP4 hybrid checkpoint | Jarrelscy | [jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) @ `63430f7b9c1b13f4bfca9e3bc3969ec0115d1a88` |
| vLLM fork that loads `nvfp4_arvq_hybrid` | Jarrelscy | [jarrelscy/vllm-mimo-v26-arvq-sm120](https://github.com/jarrelscy/vllm-mimo-v26-arvq-sm120) @ `88c94233120247f275ec94baf21638321a930469` |
| Base model | Xiaomi MiMo | [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) |
| Abliterated ARVQ weights | Keys | [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated) |
| Four-Spark serve, MTP measurement, Spark port notes, tool/build loop | Keys | this repo |

Jarrelscy marks full-model quality and SM120 / 1M serving as **unqualified**. The numbers below are a serving measurement on GB10 (SM121), not a quality claim.

## Champion

**MTP = 2** draft tokens, selected for single-stream prose. The draft is the checkpoint's own MTP stack (`model.mtp.layers.0`, `.1`, `.2`). The image loads **all three heads** and runs them non-chain, the way Xiaomi's SGLang deploy does. The published fork loaded one head and replayed it. See [SPARK-PORT.md §6](SPARK-PORT.md#6-all-three-mtp-heads-non-chain-2026-09-26).

| | |
|---|---|
| Nodes | 4× DGX Spark GB10, one GPU each, TP4, PP1 |
| Context | `1048576` |
| KV | BF16 (`--kv-cache-dtype` left at auto). FP8 KV is not used |
| GPU memory fraction | **0.85** (do not raise this on GB10) |
| Scheduler | `max-num-seqs 4`, `max-num-batched-tokens 5120`, chunked prefill, prefix caching |
| Execution | torch.compile + CUDA graphs (`FULL_AND_PIECEWISE`). Compile cache persisted in `/var/tmp/mimo-arvq-vllm-cache` |
| ARVQ prefill | **expert-batched CUDA prefill** (v3: `VLLM_ARVQ_BATCHED_PREFILL=1`, chunks ≥32 tokens), fused decode activation pack (image defaults) |
| Modalities | `--language-model-only` so the startup profile does not spend the KV budget on a video |
| Draft | `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` |
| Served name | `MiMo-V2.6-Pro-ARVQ` |
| Tool calls | `--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo` (required; without these Hermes `tool_choice: auto` is HTTP 400) |
| Output cap | `--override-generation-config '{"max_new_tokens": 8192}'` (the checkpoint default of 2048 caused the tool loop) |
| Checkpoint | abliterated tree `…-ablit-dealign-op` / gated HF repo above |

Measured KV pool on the champion boot: about **2.07M tokens** (three MTP heads now hold KV). Weights about **73.2 GiB per rank**.

### Speed, current champion (2026-09-26)

Prose: 512 new tokens, temperature 1.0, top_p 0.95, thinking off. Single-stream mean of three stories (beekeeper, lighthouse, nurse), short prompt. Reproduce with [`serve/bench/speedbench.py`](serve/bench/speedbench.py).

| Draft tokens | Decode | Tokens per pass |
|---:|---:|---:|
| **2 (champion)** | **20.4 tok/s** | **1.82** |
| 3 | 18.6 tok/s | 1.86 |

Requests overlapped (`max_num_seqs 4`), MTP=2:

| Requests | Aggregate | Per request |
|---:|---:|---:|
| 1 | 20.4 tok/s | 20.4 tok/s |
| 2 | 31.1 tok/s | 15.6 tok/s |
| 4 | 43.5 tok/s | 10.9 tok/s |

Uncached prefill (nonce prompt, `max_tokens` 1):

| Prompt | Old eager recipe | Grouped prefill (superseded) | **Now: batched prefill** | Time to first token, now |
|---:|---:|---:|---:|---:|
| 9.5K tokens | 128 tok/s | 505 tok/s | **1,291 tok/s** | **7.4 s** |
| 38K tokens | 126 tok/s | 527 tok/s | **1,038 tok/s** | **36.6 s** |
| 152K tokens | — | — | 615 tok/s | 247 s |

Prefill slows as prompts grow because the 10 full-attention layers grow with context length. Decode is unchanged by v3.

MTP acceptance per draft position (sampled prose): 0.63 / 0.19. Heads 1-2 lost accuracy on this target because ARVQ quantization and the abliteration moved the hidden state they read. On-policy fine-tuning of the heads is in progress.

## Image

**`ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:latest`**, which is the same image as `:63430f7-sm121-v3`. It is public and needs no login. This is the only published image: older tags were removed, so you can't pull a slower build by mistake. `serve/launch-rank.sh` pins `:63430f7-sm121-v3`.

Digest `sha256:52cbb3b3b3bc902fa4bc5f4f59b5e82b660da9629d87ebc662add8319d4aca84`.

The image contains:

- **Jarrelscy's fork** compiled for GB10 (`sm_121a`), with the Spark loader fixes.
- **All three MTP heads**, run non-chain.
- **The tool-call loop fix.**
- **CUDA-graph-clean abliteration hooks.**
- **Expert-batched ARVQ prefill kernels** (`grouped.cu`, built during the image build). Per MoE layer at 5120 tokens they take 18.9 ms instead of 146.8 ms, with output cosine 0.9999996 vs native.

The recipe is [`serve/image/Dockerfile`](serve/image/Dockerfile), on top of [`Dockerfile.base`](serve/image/Dockerfile.base). It reproduces the published image file-for-file: all 17 overlaid files and env defaults were checked. The image does not contain the weights.

## Bring-up

Put the checkpoint on storage the four ranks can read. Then, on each node:

```bash
# rank 0 is the API. ranks 1–3 are headless.
# GID is the IPv4 RoCE index for that node's HCA. It is not the same on every Spark.
export LM_ONLY=1
export MAXLEN=1048576
export SEQS=4
export SPEC='{"method":"mtp","num_speculative_tokens":2}'
export IMAGE=ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v3   # the default
export MASTER_ADDR=10.0.0.1   # rank 0
export HOSTPATH=/path/to/MiMo-V2.6-Pro-RL-ARVQ-hybrid-ablit-dealign-op
# or: hf download drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated --local-dir "$HOSTPATH"
export GID_INDEX=3            # confirm with show_gids; one node in this cluster needed 7

bash serve/launch-rank.sh "$HEAD_IP" "$RANK" "$GID_INDEX" "$HOSTPATH" headless
# rank 0:
bash serve/launch-rank.sh "$HEAD_IP" 0 "$GID_INDEX" "$HOSTPATH" api
```

NCCL on this cluster used the 200G RoCE NIC (`NCCL_NET=IB`), not the TCP path on that same device. See [SPARK-PORT.md](SPARK-PORT.md) for the three loader fixes required before the first token.

## Integration and experiments

- **[Hermes](HERMES.md)** — parsers, Hermes execution, and build-from-prompt (`write_file` + `terminal`).
- **[DFlash](DFLASH.md)** — measured on the old eager build. 13.0 tok/s single-stream prose, 22.5 tok/s at four requests. Slower than MTP. Not the champion.
- **[Abliteration](ABLITERATION.md)** — live dealign-op: thinking-off 32/32 · 22/22; thinking-on 25/32 · 16/22.

## Build from a prompt

After the four ranks are up and Hermes points at `http://<rank-0>:8888/v1` model `MiMo-V2.6-Pro-ARVQ`:

```bash
# server + Hermes execution (write a file, run it, check output)
bash serve/verify-tools-and-build.sh http://127.0.0.1:8888/v1

# or a free-form build:
hermes chat -q "Create /tmp/demo/app.py that prints hello and run it. Use write_file then terminal." --oneshot --yolo
```

The model must emit tool calls. Hermes runs them on the host (`terminal.backend: local`). Do not leave `tool_use_enforcement` off for this checkpoint — MiMo otherwise narrates the build instead of calling tools.

## License

Recipe text in this repo is MIT. The checkpoint and the vLLM fork keep their own licenses (the uploaded snapshot's card is MIT; the fork follows upstream vLLM).
