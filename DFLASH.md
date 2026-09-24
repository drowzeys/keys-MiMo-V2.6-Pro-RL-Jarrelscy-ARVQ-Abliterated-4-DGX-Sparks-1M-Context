# DFlash

Status: **measured**. It does not replace the MTP=2 champion.

Jarrelscy shipped the draft inside the same checkpoint, at `dflash/`. The Spark boot used that directory with `method=dflash` and **7** speculative tokens (block size 8). The mask embedding at token 151675 loaded. KV pool on that boot was **2,124,788** tokens, still above a 1M context. GPU memory fraction stayed at 0.85. Eager, BF16 KV, four sequences.

Same prose protocol as the MTP sweep: 33K-token prompt, 512 new tokens, temperature 1.0.

Single-stream mean of the three stories: **13.0 tok/s**. Tokens per pass averaged about **1.84**. One pass takes about 140 ms, and a 7-token draft only keeps roughly one extra token, so most of the block is rejected.

| Requests | Together | Each request | Tokens per pass |
|---:|---:|---:|---:|
| 1 | 12.4 tok/s decode | 12.4 | 1.79 |
| 2 | 17.3 tok/s | 8.9 | 1.84 |
| 4 | 22.5 tok/s | 5.8 | 1.77 |

All four requests ran together. Nothing waited in the queue.

| | Single stream | 2 requests | 4 requests |
|---|---:|---:|---:|
| MTP 1 | 17.4 | 29.6 | 45.0 |
| MTP 2 (champion) | 18.6 | 27.9 | 39.1 |
| DFlash, 7 tokens | 13.0 | 17.3 | 22.5 |

The live serve goes back to MTP=2 after this measurement.
