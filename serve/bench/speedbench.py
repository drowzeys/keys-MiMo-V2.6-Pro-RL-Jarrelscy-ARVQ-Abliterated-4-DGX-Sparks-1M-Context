#!/usr/bin/env python3
"""Speed bench for the four-Spark MiMo-V2.6-Pro ARVQ serve.

prose:   three stories x 512 tokens, temperature 1.0, top_p 0.95, thinking off,
         one request at a time. Reports decode tok/s, tokens per pass, and the
         per-position MTP acceptance from /metrics.
conc:    the same stories sent 2 and 4 at a time; aggregate decode tok/s.
prefill: nonce-prefixed filler (nothing cached), max_tokens 1; TTFT and
         prompt tokens per second.

  python3 speedbench.py --base http://<rank-0>:8888 --out results.jsonl
"""
import argparse
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

PROMPTS = {
    "beekeeper": "Write a vivid literary short story (at least 700 words) about a beekeeper during a drought. No headings, flowing prose only.",
    "lighthouse": "Write a vivid literary short story (at least 700 words) about a lighthouse keeper in 1890s Norway. No headings, flowing prose only.",
    "nurse": "Write a vivid literary short story (at least 700 words) about a night-shift nurse in Lagos. No headings, flowing prose only.",
}
FILLER = (
    "The river bent twice before the mill, and the miller counted sacks by "
    "lamplight while the wheel turned slowly in the brown autumn water. "
)


def spec_counters(base):
    drafts, per_pos = 0.0, {}
    with urllib.request.urlopen(base + "/metrics", timeout=30) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_drafts_total{"):
                drafts += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
                pos = int(line.split('position="')[1].split('"')[0])
                per_pos[pos] = per_pos.get(pos, 0.0) + float(line.rsplit(" ", 1)[1])
    return drafts, per_pos


def stream(base, model, text, max_tokens, temperature, seed):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
    return t0, first, last, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="MiMo-V2.6-Pro-ARVQ")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="speedbench.jsonl")
    ap.add_argument("--conc", default="2,4")
    ap.add_argument("--prefill", default="8192,32768")
    a = ap.parse_args()
    out = open(a.out, "a")

    def emit(row):
        row = {"label": a.label, **row}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    stream(a.base, a.model, "Say hi.", 8, 0.0, 1)  # warm-up

    tps = []
    d0, p0 = spec_counters(a.base)
    for i, (name, text) in enumerate(PROMPTS.items()):
        dd0, pp0 = spec_counters(a.base)
        t0, first, last, usage = stream(a.base, a.model, text, 512, 1.0, 1729 + i)
        dd1, pp1 = spec_counters(a.base)
        n = usage.get("completion_tokens", 0)
        drafts = dd1 - dd0
        acc = sum(pp1.values()) - sum(pp0.values())
        tps.append((n - 1) / (last - first))
        emit({"kind": "prose", "task": name, "tokens": n, "ttft_s": round(first - t0, 3),
              "decode_tps": round(tps[-1], 2),
              "tokens_per_pass": round(1 + acc / drafts, 3) if drafts else 1.0})
    d1, p1 = spec_counters(a.base)
    drafts = d1 - d0
    per_pos = {k: round((p1.get(k, 0) - p0.get(k, 0)) / drafts, 3) for k in sorted(p1)} if drafts else {}
    emit({"kind": "prose_mean", "decode_tps": round(sum(tps) / len(tps), 2), "per_position_acceptance": per_pos})

    names = list(PROMPTS)
    for level in [int(x) for x in a.conc.split(",") if x]:
        texts = [PROMPTS[names[j % len(names)]] for j in range(level)]
        with ThreadPoolExecutor(level) as pool:
            t0 = time.perf_counter()
            res = list(pool.map(lambda jt: stream(a.base, a.model, jt[1], 512, 1.0, 99 + jt[0]), enumerate(texts)))
            wall = time.perf_counter() - t0
        tokens = sum(r[3].get("completion_tokens", 0) for r in res)
        firsts = [r[1] for r in res]
        lasts = [r[2] for r in res]
        agg = tokens / (max(lasts) - min(firsts))
        emit({"kind": "concurrent", "requests": level, "aggregate_tps": round(agg, 2),
              "per_request_tps": round(agg / level, 2), "wall_s": round(wall, 1)})

    for n in [int(x) for x in a.prefill.split(",") if x]:
        text = f"[{uuid.uuid4()}] " + FILLER * (n // 25) + "\nIn one word, what is counted?"
        t0, first, last, usage = stream(a.base, a.model, text, 1, 0.0, 1)
        p = usage.get("prompt_tokens", 0)
        ttft = (first or time.perf_counter()) - t0
        emit({"kind": "prefill", "prompt_tokens": p, "ttft_s": round(ttft, 2), "prefill_tps": round(p / ttft, 1)})


if __name__ == "__main__":
    main()
