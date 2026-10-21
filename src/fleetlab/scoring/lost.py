"""Consume the runtime's lost_twice and per-start attempts receipts (entry 11)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fleetlab.agents.run import AgentResult
from fleetlab.baselines import models as baseline_runtime

# The writer keeps this symbol when the implementation moves to agents.run.
# Its runtime namespace accommodates both a definition and a non-reexported import.
_lost_runtime_answer = cast(
    Callable[[AgentResult], bool], vars(baseline_runtime)["lost_runtime_answer"]
)


def exhausted_call(record: dict[str, Any]) -> bool:
    if "lost_twice" in record:
        if type(record["lost_twice"]) is not bool:
            raise ValueError("invalid lost_twice flag")
        return record["lost_twice"]  # the session runtime decides whether tools left an answer
    starts = record.get("attempts")
    if starts is None:
        return False
    if not isinstance(starts, list) or not 1 <= len(starts) <= 2:
        raise ValueError("invalid attempts: require one or two recorded starts")
    lost = []
    for start in starts:
        if (
            not isinstance(start, dict)
            or type(start.get("ok")) is not bool
            or type(start.get("answer_present")) is not bool
            or (start.get("error") is not None and not isinstance(start["error"], str))
            or not isinstance(start.get("detail"), str)
            or not isinstance(start.get("seconds"), (int, float))
        ):
            raise ValueError("invalid runtime attempt receipt")
        lost.append(
            _lost_runtime_answer(
                AgentResult(
                    ok=start["ok"],
                    result={} if start["answer_present"] else None,
                    error=start["error"],
                    error_detail=start["detail"],
                    seconds=start["seconds"],
                    usage=start.get("usage", {}),
                )
            )
        )
    return len(starts) == 2 and all(lost)


def run_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in (root / "analysis.json", root / "run.json", root.parent / "run.json"):
        if not path.is_file():
            continue
        raw = json.loads(path.read_text())
        if path == root.parent / "run.json" and (
            not raw.get("analysis") or Path(raw["analysis"]).resolve() != root.resolve()
        ):
            continue
        records.append((path, raw))
    return records


def analysis_lost(root: Path) -> bool:
    records = run_records(root)
    records.extend(
        (path, json.loads(path.read_text()))
        for path in sorted((root / "sessions").glob("session-*.json"))
    )
    return any(raw.get("lost_twice") is True for _, raw in records)
