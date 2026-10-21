"""Require the oracle's full second-execution coverage before valuing a world."""

from typing import Any


def replay_problem(verification: dict[str, Any], attempts: int, command: str) -> str | None:
    coverage = verification.get("coverage", [])
    if not isinstance(coverage, list):
        coverage = []
    rows = [r for r in coverage if isinstance(r, dict) and r.get("window") == "scoring"]
    if len(rows) == 1:
        row = rows[0]
        if (
            type(row.get("attempts")) is int
            and row["attempts"] == attempts
            and type(row.get("attempts_executed_twice")) is int
            and row["attempts_executed_twice"] == attempts
            and row.get("missing") == []
            and attempts > 0
        ):
            return None
    return "scoring window lacks full replay_verification coverage; produce it with: " + command
