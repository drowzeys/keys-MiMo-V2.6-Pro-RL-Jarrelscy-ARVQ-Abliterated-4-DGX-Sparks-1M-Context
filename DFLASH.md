# DFlash — reserved

Status: **in progress**. Not the champion. No Spark speed number yet.

Jarrelscy retained a DFlash draft inside the same checkpoint, next to the backbone and the MTP stack. It is not a separate download.

| | |
|---|---|
| Path | `dflash/` in [jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) |
| Class | `DFlashDraftModel` |
| Weights | `dflash/dflash_draft_model.safetensors` (~5.2 GiB) |
| Shape | 5 layers, hidden 6144, block size 8 |
| Target layers | 0, 15, 31, 47, 69 |
| Mask token | 151675 |

The serving fork already has a `method=dflash` path and maps `DFlashDraftModel` onto its DFlash loader. This cluster has not booted that path. The live champion remains MTP with two draft tokens.

When a Spark run is measured, replace this note with the same prose table used for MTP (33K prompt, 512 new tokens, temperature 1.0, single stream and concurrency 2 and 4) and say whether it replaced MTP=2.
