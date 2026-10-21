"""How fast is the live layer if the CLI process is kept running?

`run_agent` starts a new `claude` process for every call and pays about a
second of start-up and shut-down each time. A live monitor watches one episode
and is asked the same question again and again, so it can keep one process and
one conversation open: each new frame is one more user message on stdin. This
script measures that: one `claude -p` process in streaming mode, a few turns
with one camera frame each, seconds per turn.

The conversation grows with every turn, which is what a monitor wants (it has
seen the earlier frames) and what the prompt cache makes cheap.

    python warm_session.py [model] [turns]      # default: haiku, 4 turns

Claude only. The codex equivalent would be `codex app-server`; not tried.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time

import toy
from agent_runtime import HOME, _claude_argv, _cli_env, claude_bin

SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "integer"}, "stop": {"type": "boolean"}},
    "required": ["code", "stop"],
    "additionalProperties": False,
}
SYSTEM = (
    "You monitor a robot arm through one camera frame at a time. For each frame report the "
    "3-digit code printed in it and whether the attempt should be stopped."
)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "haiku"
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    argv = _claude_argv(claude_bin(), model, SYSTEM, None, None, SCHEMA, None)
    env = dict(_cli_env("claude", str(HOME)), MAX_THINKING_TOKENS="0")
    started = time.perf_counter()
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True, cwd=toy.PROBE / "ws")
    truth = toy.frame_codes()["e09"]
    rows = []
    for turn in range(turns):
        frame = toy.WORKSPACE / "frames" / "e09" / f"{turn}.png"
        message = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(frame.read_bytes()).decode()}},
                    {"type": "text", "text": f"Frame at t={turn}s."},
                ],
            },
        }
        sent = time.perf_counter()
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        result = None
        for line in proc.stdout:
            event = json.loads(line)
            if event.get("type") == "result":
                result = event
                break
        seconds = time.perf_counter() - sent
        usage = result.get("usage", {})
        rows.append(
            {
                "turn": turn,
                "seconds": round(seconds, 2),
                "since_process_start": round(time.perf_counter() - started, 2),
                "api_ms": result.get("duration_api_ms"),
                "correct": (result.get("structured_output") or {}).get("code") == truth[str(turn)],
                "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
                "cached_input_tokens": usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens"),
            }
        )
        print(rows[-1], flush=True)
    proc.stdin.close()
    proc.wait(timeout=30)
    out = toy.HERE / "results" / "warm_session.json"
    out.write_text(json.dumps({"model": model, "turns": rows}, indent=1))


if __name__ == "__main__":
    main()
