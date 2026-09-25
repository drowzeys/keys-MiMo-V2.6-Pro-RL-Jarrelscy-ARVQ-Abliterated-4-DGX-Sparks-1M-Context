# Abliteration status

Status as of **2026-09-25 UTC: experimental; the 32/32 target has not been reached**.

The local serve uses an experimental Pro ARVQ variant identified in the archived results as `work-r8t-hyb-agg-l68t`, under the served name `MiMo-V2.6-Pro-ARVQ`. The previous "not started" note was out of date.

## Existing local results

The archived result files were last modified on 2026-09-24 UTC. These are existing measurements, not a new evaluation performed during the tool-calling repair.

| Suite | Cases | Classifier label: bypass | Refusal | Garbled | Empty |
|---|---:|---:|---:|---:|---:|
| Refusal suite | 32 | **11** | 21 | 0 | 0 |
| Cyber suite | 22 | **15** | 7 | 0 | 0 |

The scorer uses text heuristics to label responses. A "bypass" label does not establish that a response is correct, complete, or effective. The four benign sanity prompts in the refusal result file also returned responses; those four cases are separate from the 32-case count.

The [verification summary](serve/verification/2026-09-25-status.json) includes these aggregate counts and SHA-256 hashes of the source result files. Benchmark prompts and generated responses are not included in that summary.

## Scope

These results describe the local experimental Pro variant, not the untouched upstream checkpoint or MiMo-V2.6-Flash. They do not establish general model quality.

The tool-calling repair changed the server launch flags and Hermes configuration without editing checkpoint weights. The four-node serve retains 1M context, MTP=2, four sequences, eager execution, BF16 KV, and GPU memory fraction **0.85**.

This repository does not distribute the experimental weights. The launcher loads the checkpoint supplied through `HOSTPATH`; the published container image contains the runtime and loader fixes.
