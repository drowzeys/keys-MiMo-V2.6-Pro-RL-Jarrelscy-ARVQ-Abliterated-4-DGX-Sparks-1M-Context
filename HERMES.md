# Hermes and tool calling

Status: **verified on 2026-09-25 UTC** with `MiMo-V2.6-Pro-ARVQ`.

## Server fix

The server answered normal requests but returned HTTP 400 when Hermes sent `tool_choice: "auto"`:

```text
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

[`serve/launch-rank.sh`](serve/launch-rank.sh) now supplies:

```bash
--enable-auto-tool-choice \
--tool-call-parser mimo \
--reasoning-parser mimo
```

The pinned runtime provides both `mimo` parsers. All four ranks were recreated with these flags. Existing checkpoint mounts, context length, draft settings, and memory settings were preserved. Recreate ranks with the updated launcher to apply the new arguments; restarting an old container alone reuses its old arguments.

## Hermes configuration

| Setting | Verified value |
|---|---|
| API base URL | `http://<rank-0-host>:8888/v1` — replace the host with your API node |
| Served model | `MiMo-V2.6-Pro-ARVQ` |
| Transport | `chat_completions` |
| Context length | `1048576` |
| Saved custom provider | `mimo26-pro`, selected as `custom:mimo26-pro` |
| Main chat template options | `thinking: false`, `enable_thinking: false` |
| CLI toolset | `hermes-cli` |
| Telegram toolset | `hermes-telegram` |

The default model and delegation route now use this Pro model. Delegation previously retained an old GLM model name. Text auxiliary tasks, including compression, approvals, and web extraction, now use `provider: main` instead of the offline Aeon endpoint. Standard toolsets were already enabled; the missing server flags prevented their calls from working.

The Hermes gateway was restarted after applying the configuration. Use `/new` in existing chats to start a session with the corrected settings.

## Verification

| Check | Result |
|---|---|
| Health and model discovery | API healthy; configured model present with 1M context |
| Nonstream tool call | `add_integers` returned structured arguments `{"a":19,"b":23}` with `finish_reason: "tool_calls"` |
| Tool-result round trip | After receiving the computed tool result, the model answered **42** |
| Streamed tool call | Valid call ID, function name, assembled JSON arguments, and tool-call finish reason |
| Hermes execution | Hermes called `read_file`, received a successful tool result, and returned the exact random token from the file |
| Gateway | Restarted and active |

The [verification summary](serve/verification/2026-09-25-status.json) contains the recorded results. These checks validate the tool-calling path; tools that depend on other services still require those services.

The MiMo serve uses `--language-model-only`. Hermes's separate vision endpoint was offline during verification, so image analysis was not verified. Server containers and the previous Hermes configuration were retained locally for rollback.
