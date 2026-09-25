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

The [verification summary](serve/verification/2026-09-25-status.json) still hashes the older l68t result files. New gate logs live under `~/mimo26-arvq-tp4/ablit/work-dealign/`.

## Scope

These results describe the local experimental Pro variant, not the untouched upstream checkpoint or MiMo-V2.6-Flash. They do not establish general model quality.

The tool-calling repair changed the server launch flags and Hermes configuration without editing checkpoint weights. The four-node serve retains 1M context, MTP=2, four sequences, eager execution, BF16 KV, and GPU memory fraction **0.85**.

This repository does not distribute the experimental weights. The launcher loads the checkpoint supplied through `HOSTPATH`; the published container image contains the runtime and loader fixes.
