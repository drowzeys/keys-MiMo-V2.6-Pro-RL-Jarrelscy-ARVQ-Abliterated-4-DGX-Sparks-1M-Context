# Hermes and tool calling

Status: **tool parsers on the live Abliterated serve; Hermes executes writes, shell, and code** so a prompt can build a project, not only describe one.

The checkpoint is the gated Abliterated tree. Thinking defaults **off** (`chat_template_kwargs.enable_thinking: false`). Turn thinking on per request when you want it.

## What “enabled” means

Two layers have to work together:

1. **vLLM** must accept `tool_choice: "auto"` and emit MiMo tool-call tokens. [`serve/launch-rank.sh`](serve/launch-rank.sh) always passes:

```bash
--enable-auto-tool-choice \
--tool-call-parser mimo \
--reasoning-parser mimo
```

Without those flags the API returns HTTP 400 (`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`). Recreate the four ranks after changing the launcher; a container restart keeps old argv.

2. **Hermes** must run the functions the model named. Use the full CLI / Telegram bundles, not a read-only subset:

| Surface | Toolset | Executes |
|---|---|---|
| CLI | `hermes-cli` | `write_file`, `patch`, `read_file`, `search_files`, `terminal`, `process_manage`, `execute_code`, `web_search`, `web_extract`, `delegate_task`, … |
| Telegram | `hermes-telegram` | same core, with terminal safety checks |

Set `agent.tool_use_enforcement: true` for this model. The auto list is GPT/Codex/Gemini/Grok only; MiMo otherwise narrates “I will write a file…” and never calls the tool.

## Build from a prompt

Hermes `terminal.backend` is `local` (cwd `/home/keyspark`). A coding prompt should:

1. `write_file` (or `patch`) the source
2. `terminal` or `execute_code` to run / test it
3. Read the output and iterate until it runs

Example:

```bash
hermes chat -q "Create /tmp/mimo-build-demo/hello.py that prints 42 and run it with python3. Use write_file then terminal. Reply with the program stdout only." --oneshot --yolo
```

`--yolo` skips dangerous-command approval prompts. Recipe check: [`serve/verify-tools-and-build.sh`](serve/verify-tools-and-build.sh).

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

The default model and delegation route use this Abliterated Pro checkpoint. Text auxiliary tasks (compression, approvals, web extraction) use `provider: main`. Restart `hermes-gateway` after config edits and use `/new` in existing chats.

## Verification

| Check | Result |
|---|---|
| Health and model discovery | API healthy; configured model present with 1M context |
| Nonstream tool call | `add_integers` returned structured arguments `{"a":19,"b":23}` with `finish_reason: "tool_calls"` |
| Tool-result round trip | After receiving the computed tool result, the model answered **42** |
| Streamed tool call | Valid call ID, function name, assembled JSON arguments, and tool-call finish reason |
| Hermes execution | Hermes called `read_file`, received a successful tool result, and returned the exact random token from the file |
| Build from prompt | 2026-09-25: Hermes `write_file` wrote `/tmp/mimo-ablit-build-demo/hello.py` (`print(42)`), `terminal` ran `python3` on it, stdout **42**. Three tool calls. |
| Gateway | Restarted and active |

Re-run `bash serve/verify-tools-and-build.sh` after a remount. Older [verification JSON](serve/verification/2026-09-25-status.json) covers the parser/round-trip checks; the build script is the live execute path.

The MiMo serve uses `--language-model-only`. Hermes's separate vision endpoint was offline during verification, so image analysis was not verified. Server containers and the previous Hermes configuration were retained locally for rollback.
