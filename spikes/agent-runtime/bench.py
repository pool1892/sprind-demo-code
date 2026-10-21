"""Measure how slow the agents are through the command-line tools.

    python bench.py latency  [sandbox] [backend]   # each tier of each vendor, three tasks
    python bench.py parallel [sandbox] [backend]   # 4 and 8 calls at once
    python bench.py fast     [sandbox] [backend]   # the quickest settings for the live layer
    python bench.py report               # print the tables from results/*.json

Every call is appended to results/bench_calls.jsonl, so an interrupted run
loses nothing and `report` always shows what was actually measured.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import toy
from agent_runtime import run_agent

RESULTS = toy.HERE / "results"
CALLS = RESULTS / "bench_calls.jsonl"

TIERS = [("claude", "strong"), ("claude", "cheap"), ("codex", "strong"), ("codex", "cheap")]


def task(name: str) -> dict:
    if name == "trivial":
        return dict(prompt=toy.TRIVIAL_PROMPT)
    if name == "images8":
        return dict(prompt=toy.IMAGES_PROMPT, images=toy.eight_frames(), schema=toy.IMAGES_SCHEMA)
    if name == "session":
        return dict(prompt=toy.SESSION_PROMPT, tools=toy.tool_server(), schema=toy.SESSION_SCHEMA)
    raise ValueError(name)


CHECK = {"trivial": lambda r: isinstance(r, str) and "pong" in r.lower(), "images8": toy.check_images, "session": toy.check_session}


def one(group: str, backend: str, tier: str, name: str, sandbox: str, **extra) -> dict:
    model = toy.MODELS[(backend, tier)]
    res = run_agent(backend, model, toy.SYSTEM, sandbox=sandbox, timeout=600, **task(name), **extra)
    phases = res.meta.get("phases") or {}
    row = {
        "group": group, "backend": backend, "tier": tier, "model": res.model or model,
        "task": name, "sandbox": sandbox, "effort": extra.get("effort"),
        "env": extra.get("_env"),
        "seconds": res.seconds,
        "ready_s": phases.get("ready_s"), "answer_s": phases.get("answer_s"),
        "api_s": (res.meta.get("api_ms") or 0) / 1000 or None,  # claude only
        "ok": res.ok, "correct": bool(res.ok and CHECK[name](res.result)),
        "error": res.error, "error_detail": res.error_detail[:200],
        "tool_calls": len(res.meta.get("tool_log", [])), "turns": res.meta.get("num_turns"),
        **{k: res.usage.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")},
        "rate_limit": (res.meta.get("rate_limit") or {}).get("unifiedWindows"),
        "at": time.strftime("%H:%M:%S"),
    }
    RESULTS.mkdir(exist_ok=True)
    with open(CALLS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(
        f"{group:9} {backend:6} {tier:6} {name:8} {row['seconds']:7.2f}s ok={row['ok']} "
        f"correct={row['correct']} err={row['error']}", flush=True,
    )
    return row


def latency(sandbox: str, only: str | None = None) -> None:
    for backend, tier in TIERS:
        if only and backend != only:
            continue
        for name, reps in (("trivial", 2), ("images8", 2), ("session", 1)):
            for _ in range(reps):
                one("latency", backend, tier, name, sandbox)


def parallel(sandbox: str, only: str | None = None) -> None:
    plan = [  # (backend, tier, how many at once)
        ("claude", "cheap", 4), ("claude", "cheap", 8), ("claude", "strong", 4),
        ("codex", "cheap", 4), ("codex", "cheap", 8), ("codex", "strong", 4),
    ]
    for backend, tier, n in plan:
        if only and backend != only:
            continue
        group = f"par{n}"
        started = time.perf_counter()
        with ThreadPoolExecutor(n) as pool:
            rows = list(pool.map(lambda _: one(group, backend, tier, "trivial", sandbox), range(n)))
        wall = time.perf_counter() - started
        summary = {
            "group": "parallel_summary", "backend": backend, "tier": tier, "n": n, "sandbox": sandbox,
            "batch_seconds": round(wall, 2), "calls_per_minute": round(60 * n / wall, 1),
            "errors": [r["error"] for r in rows if not r["ok"]],
            "slowest": max(r["seconds"] for r in rows), "fastest": min(r["seconds"] for r in rows),
        }
        with open(CALLS, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
        print(summary, flush=True)


def fast(sandbox: str, only: str | None = None) -> None:
    """Settings that trade thinking for speed."""
    if only in (None, "claude"):
        one("fast", "claude", "cheap", "trivial", sandbox, _env={"MAX_THINKING_TOKENS": "0"})
        one("fast", "claude", "cheap", "images8", sandbox, _env={"MAX_THINKING_TOKENS": "0"})
        one("fast", "claude", "strong", "images8", sandbox, effort="low")
    if only in (None, "codex"):
        one("fast", "codex", "cheap", "images8", sandbox, effort="low")


def report() -> None:
    rows = [json.loads(line) for line in CALLS.read_text().splitlines() if line.strip()]
    calls = [r for r in rows if r["group"] != "parallel_summary"]

    def med(values):
        values = [v for v in values if v is not None]
        return f"{statistics.median(values):.1f}" if values else "-"

    print("\nLatency, one call at a time (median seconds; n = calls)")
    print("| backend | model | task | n | wall | until ready | CLI overhead | input tok | output tok | correct |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for backend, tier in TIERS:
        for name in ("trivial", "images8", "session"):
            sel = [r for r in calls if r["group"] == "latency" and (r["backend"], r["tier"], r["task"]) == (backend, tier, name)]
            if not sel:
                continue
            overhead = [r["seconds"] - r["api_s"] for r in sel if r.get("api_s")]
            print(
                f"| {backend} | {sel[0]['model']} | {name} | {len(sel)} | {med([r['seconds'] for r in sel])} | "
                f"{med([r['ready_s'] for r in sel])} | {med(overhead)} | {med([r['input_tokens'] for r in sel])} | "
                f"{med([r['output_tokens'] for r in sel])} | {sum(r['correct'] for r in sel)}/{len(sel)} |"
            )
    print("\nFast settings")
    print("| backend | model | task | setting | wall | output tok | correct |")
    print("|---|---|---|---|---|---|---|")
    for r in calls:
        if r["group"] == "fast":
            setting = r["effort"] and f"effort {r['effort']}" or json.dumps(r["env"])
            print(f"| {r['backend']} | {r['model']} | {r['task']} | {setting} | {r['seconds']:.1f} | {r['output_tokens']} | {r['correct']} |")
    print("\nParallel trivial calls")
    print("| backend | model | at once | batch wall | calls/min | fastest | slowest | errors |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["group"] == "parallel_summary":
            print(
                f"| {r['backend']} | {toy.MODELS[(r['backend'], r['tier'])]} | {r['n']} | {r['batch_seconds']} | "
                f"{r['calls_per_minute']} | {r['fastest']:.1f} | {r['slowest']:.1f} | {r['errors'] or 'none'} |"
            )


if __name__ == "__main__":
    mode = sys.argv[1]
    box = sys.argv[2] if len(sys.argv) > 2 else "bwrap"
    backend_filter = sys.argv[3] if len(sys.argv) > 3 else None
    if mode == "report":
        report()
    else:
        {"latency": latency, "parallel": parallel, "fast": fast}[mode](box, backend_filter)
