"""Run the three reference tasks once on one backend and print what came back.

    python demo.py claude haiku [none|bwrap|docker] [effort]
    python demo.py codex gpt-6-luna
"""

from __future__ import annotations

import json
import sys

import toy
from agent_runtime import run_agent


def show(label: str, res, correct=None) -> None:
    calls = [c["name"].split("__")[-1] for c in res.tool_calls]
    print(f"--- {label}: ok={res.ok} correct={correct} seconds={res.seconds} model={res.model}")
    print(f"    result: {json.dumps(res.result)[:400]}")
    print(f"    usage:  {json.dumps(res.usage)}")
    print(f"    tool calls: {calls}")
    if not res.ok:
        print(f"    error: {res.error}: {res.error_detail[:400]}")
    print(f"    run dir: {res.run_dir}")


def main() -> None:
    backend, model = sys.argv[1], sys.argv[2]
    sandbox = sys.argv[3] if len(sys.argv) > 3 else "none"
    effort = sys.argv[4] if len(sys.argv) > 4 else None
    common = dict(sandbox=sandbox, effort=effort)

    res = run_agent(backend, model, toy.SYSTEM, toy.TRIVIAL_PROMPT, **common)
    show("trivial", res)

    res = run_agent(
        backend, model, toy.SYSTEM, toy.IMAGES_PROMPT,
        images=toy.eight_frames(), schema=toy.IMAGES_SCHEMA, **common,
    )
    show("eight images", res, toy.check_images(res.result))

    res = run_agent(
        backend, model, toy.SYSTEM, toy.SESSION_PROMPT,
        tools=toy.tool_server(), schema=toy.SESSION_SCHEMA, **common,
    )
    show("tool session", res, toy.check_session(res.result))


if __name__ == "__main__":
    main()
