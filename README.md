# keys-MiMo-V2.6-Pro-RL Jarrelscy ARVQ Abliterated — 4 DGX Sparks, 1M context

Serving recipe for the **abliterated** [Jarrelscy ARVQ / NVFP4 hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) of **[XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)** on **four NVIDIA DGX Spark (GB10)** nodes, tensor-parallel 4, **1,048,576-token context**.

Gated weights (automatic approval after terms): **[drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated)**.

The launcher enables MiMo tool calling on the server (`--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo`). Hermes then **executes** those calls (`write_file`, `terminal`, `execute_code`, `read_file`, …) so a prompt can write a project, run it, and iterate. See [HERMES.md](HERMES.md) and [`serve/verify-tools-and-build.sh`](serve/verify-tools-and-build.sh).

## Current status — 2026-09-25 UTC

- **Serving:** live on four Sparks with 1M context, MTP=2, four sequences, eager execution, BF16 KV, and GPU memory fraction 0.85.
- **[Hermes and tool calling](HERMES.md):** server parsers on; Hermes `hermes-cli` / `hermes-telegram` execute `write_file`, `terminal`, `execute_code`. A prompt can scaffold a file, run it, and return the program output. `tool_use_enforcement: true` so the model calls tools instead of describing them.
- **[Abliteration](ABLITERATION.md):** live `dealign-op` tree. Thinking **off** **32/32** refusal and **22/22** cyber; thinking **on** 25/32 and 16/22 (visible content). Gated HF: [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated).
- **[DFlash](DFLASH.md):** measured and slower than MTP=2. MTP=2 remains the serving choice.
- **Vision:** the live serve is text-only. The separate Hermes vision endpoint was offline at verification.

The [verification summary](serve/verification/2026-09-25-status.json) records the tool-call checks and the source hashes for the archived benchmark counts.

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
| Tool calls | `--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo` (required; without these Hermes `tool_choice: auto` is HTTP 400) |
| Checkpoint | abliterated tree `…-ablit-dealign-op` / gated HF repo above |

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

## Image

The Spark image is at:

`ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v1`

Digest `sha256:83c6bdd58e4d99521b2e4f2060225a34678b46fa8d069d1adcea9d4389fefe9e`.

It is Jarrelscy's fork compiled for GB10 (`sm_121a`), plus the Python fixes in [`serve/image/`](serve/image/). Those fixes were bind-mounted on the first successful serve and are now inside this tag. The image does not contain the weights. The package is private on push. Make it public once at [package settings](https://github.com/users/drowzeys/packages/container/package/mimo-v26-pro-arvq-spark/settings), or pull it with `gh auth token` while it stays private. Then run `serve/launch-rank.sh` with `IMAGE` set to that tag.

## Bring-up

Put the checkpoint on storage the four ranks can read. Then, on each node:

```bash
# rank 0 is the API. ranks 1–3 are headless.
# GID is the IPv4 RoCE index for that node's HCA. It is not the same on every Spark.
export LM_ONLY=1
export MAXLEN=1048576
export SEQS=4
export SPEC='{"method":"mtp","num_speculative_tokens":2}'
export IMAGE=ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v1
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
- **[DFlash](DFLASH.md)** — measured. 13.0 tok/s single-stream prose, 22.5 tok/s at four requests. Slower than MTP=2. Not the champion.
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
