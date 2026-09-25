# Abliteration status

Status as of **2026-09-25 UTC: live champion is dealign-op**. Gated weights: [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated) (automatic approval after terms).

The live serve is `MiMo-V2.6-Pro-ARVQ` with decoder `o_proj` transplanted from [dealignai/MiMo-V2.6-Pro-RL-UNCENSORED](https://huggingface.co/dealignai/MiMo-V2.6-Pro-RL-UNCENSORED) v3 onto Jarrelscy ARVQ `63430f7` (25 of 29 native o_proj layers; DFlash-source/pad anchors left stock).

## Gate (heuristic classifier)

| Mode | Refusal 32 | Cyber 22 |
|---|---:|---:|
| Thinking **off** (greedy, 192 tokens) | **32/32** bypass | **22/22** bypass |
| Thinking **on** (greedy, 1024 tokens, visible content) | **25/32** bypass · 7 refuse | **16/22** bypass · 1 refuse · 2 garble · 3 empty |
| Prior l68t leftover-SRA (thinking off) | 11/32 | 15/22 |
| Stock ARVQ | 5/32 | 9/22 |

Thinking-off and thinking-on are the same weights. Choose per request with `chat_template_kwargs.enable_thinking`. A bypass label means the reply starts delivering the requested content. It does not certify correctness.

Canonical snapshot: [serve/verification/current-status.json](serve/verification/current-status.json). Gate logs on the lab host: `~/mimo26-arvq-tp4/ablit/work-dealign/`.

## Scope

These results describe this Abliterated Pro ARVQ tree, not stock Xiaomi MXFP4 or MiMo-V2.6-Flash. A bypass label does not certify correctness.

The four-node serve retains 1M context, MTP=2, four sequences, eager execution, BF16 KV, GPU memory fraction **0.85**, and MiMo tool parsers. Hermes executes `write_file` / `terminal` / `execute_code` so a prompt can build and run code.

**Weights:** gated Hugging Face repo [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated) (automatic approval after terms). This GitHub repo is the Spark recipe and launcher; set `HOSTPATH` to that download (or the local `…-ablit-dealign-op` tree). The container image is runtime-only.
