#!/usr/bin/env bash
# Verify MiMo tool parsers AND that Hermes actually executes a build-from-prompt.
# Usage: bash serve/verify-tools-and-build.sh [http://127.0.0.1:8888/v1]
set -euo pipefail
API="${1:-http://10.100.10.1:8888/v1}"
MODEL="${MODEL:-MiMo-V2.6-Pro-ARVQ}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/serve/verification/tools-and-build.json}"
DEMO=/tmp/mimo-ablit-build-demo
HERMES="${HERMES:-/home/keyspark/.hermes/hermes-agent/venv/bin/hermes}"

python3 - "$API" "$MODEL" "$OUT" "$DEMO" "$HERMES" << 'PY'
import json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

api, model, out_path, demo, hermes = sys.argv[1:6]
base = api.rstrip("/")
health = base[:-3] + "/health" if base.endswith("/v1") else base + "/health"
report = {"api": api, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

def post(path, body, timeout=180):
    req = urllib.request.Request(
        base + path if path.startswith("/") else path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

code = urllib.request.urlopen(health, timeout=8).status
assert code == 200, f"health {code}"
report["health"] = True

tools = [{
    "type": "function",
    "function": {
        "name": "add_integers",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}]
first = post("/chat/completions", {
    "model": model,
    "messages": [{"role": "user", "content": "Use the add_integers tool to add 19 and 23."}],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
})
msg = first["choices"][0]["message"]
tcs = msg.get("tool_calls") or []
assert tcs, f"no tool_calls: {msg}"
fn = tcs[0]["function"]
args = json.loads(fn["arguments"])
assert fn["name"] == "add_integers" and args.get("a") == 19 and args.get("b") == 23, args
report["nonstream_tool_call"] = {"name": fn["name"], "arguments": args, "finish": first["choices"][0].get("finish_reason")}

second = post("/chat/completions", {
    "model": model,
    "messages": [
        {"role": "user", "content": "Use the add_integers tool to add 19 and 23."},
        msg,
        {"role": "tool", "tool_call_id": tcs[0]["id"], "content": "42"},
    ],
    "tools": tools,
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
})
answer = (second["choices"][0]["message"].get("content") or "")
assert "42" in answer, answer
report["tool_result_roundtrip"] = {"assistant": answer[:240], "ok": True}

# Hermes executes write_file + terminal
shutil.rmtree(demo, ignore_errors=True)
Path(demo).mkdir(parents=True)
prompt = (
    f"Create {demo}/hello.py that prints exactly 42 with a trailing newline. "
    "Use the write_file tool, then the terminal tool to run: python3 "
    f"{demo}/hello.py . Reply with only the program stdout."
)
env = os.environ.copy()
env["HERMES_HOME"] = os.path.expanduser("~/.hermes")
cmd = [hermes, "chat", "-q", prompt, "--oneshot", "--yolo"]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
report["hermes"] = {
    "returncode": p.returncode,
    "stdout_tail": (p.stdout or "")[-800:],
    "stderr_tail": (p.stderr or "")[-400:],
}
hello = Path(demo) / "hello.py"
report["build"] = {
    "file_exists": hello.exists(),
    "file_text": hello.read_text() if hello.exists() else "",
}
ran = subprocess.run(["python3", str(hello)], capture_output=True, text=True, timeout=30) if hello.exists() else None
report["build"]["run_stdout"] = (ran.stdout if ran else "")
report["build"]["run_ok"] = bool(ran and ran.returncode == 0 and ran.stdout.strip() == "42")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("health", "nonstream_tool_call", "tool_result_roundtrip", "build")}, indent=2))
if not report["build"]["run_ok"]:
    raise SystemExit("build-from-prompt failed; see " + out_path)
print("OK tools+build ->", out_path)
PY
