"""The loop's cost totals sum every model start: a replaced start's tokens, time and tool
calls count (fleetlab.loop.costs). No model, no simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import SessionRecord
from fleetlab.loop.costs import read_sessions, session_costs, start_tool_calls


def _start(root: Path, name: str, calls: int, usage: dict[str, int], ok: bool) -> dict[str, Any]:
    run_dir = root / name
    log = run_dir / "box" / "toolbox" / "tool_calls.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("".join(json.dumps({"seq": i + 1}) + "\n" for i in range(calls)))
    return {
        "ok": ok,
        "error": None if ok else "timeout",
        "detail": "",
        "run_dir": str(run_dir),
        "seconds": 10.0 if ok else 4.0,
        "usage": usage,
        "runtime_meta": {},
        "answer_present": ok,
    }


def test_a_replaced_start_counts_in_every_total(tmp_path: Path) -> None:
    lost = _start(tmp_path, "a", 5, {"input_tokens": 100, "output_tokens": 7}, ok=False)
    kept = _start(tmp_path, "b", 3, {"input_tokens": 50, "cached_input_tokens": 40}, ok=True)
    session = SessionRecord(
        number=1,
        state="done",
        export="e",
        run_dir=kept["run_dir"],
        ok=True,
        error=None,
        seconds=14.0,
        usage=kept["usage"],  # the record keeps only the last start's usage ...
        queued=[],
        submitted=["decision-1"],
        report=None,
        tool_calls=3,  # ... and its tool calls
        attempts=[lost, kept],
        rerun=True,
    )
    costs = session_costs([session], "gpt-6-astra")
    assert (costs["model_starts"], costs["failed_starts"]) == (2, 1)
    assert costs["model_seconds"] == 14.0
    assert costs["tokens"]["input_tokens"] == 150 and costs["tokens"]["output_tokens"] == 7
    assert costs["tokens"]["cached_input_tokens"] == 40
    assert costs["tool_calls"] == 8  # 5 of the replaced start and 3 of the kept one
    assert costs["api_equivalent_usd"]["status"] == "bounded_api_equivalent"
    assert costs["per_session"][0]["tool_calls"] == 8
    assert start_tool_calls(tmp_path / "missing") == 0

    sessions = tmp_path / "analysis" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "session-01.json").write_text(json.dumps(session.__dict__))
    assert [s.number for s in read_sessions(tmp_path / "analysis", [1])] == [1]
    assert read_sessions(tmp_path / "analysis", [2]) == []
