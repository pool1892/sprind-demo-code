"""Claim 6: ledger-backed costs, including failed attempts and cached inputs."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import Analysis
from fleetlab.toolserver.files import read_lines

TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


def analysis_ledgers(analysis: Analysis, override: Path | None = None) -> list[Path]:
    """Follow runtime records, without assuming that calls used the shared run root."""
    if override is not None:
        return [override]
    paths: set[Path] = set()
    for file in (analysis.root / "sessions").glob("session-*.json"):
        record = json.loads(file.read_text())
        if directory := record.get("run_dir"):
            path = Path(directory).expanduser().resolve().parent / "ledger.jsonl"
            if path.is_file():
                paths.add(path)
    if paths:
        return sorted(paths)
    adjacent = analysis.root.parent / "agent-runs" / "ledger.jsonl"
    if adjacent.is_file():
        return [adjacent]
    root = Path(os.environ.get("ROBOTS_AGENT_ROOT", f"/var/tmp/robots-agents-{os.getuid()}"))
    return [root / "ledger.jsonl"]


def score_accounting(
    analysis: Analysis,
    attempts: int,
    ledger: Path | Sequence[Path],
    prices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The denominator is distinct attempts available for the analysis, not scoring attempts.

    `prices` maps exact ledger model names to USD per million input/cached-input/output
    tokens, with a `source` and `as_of`. No undocumented live price is guessed.
    """
    sessions = [
        json.loads(p.read_text())
        for p in sorted((analysis.root / "sessions").glob("session-*.json"))
    ]
    expected = {str(Path(s["run_dir"]).resolve()) for s in sessions if s.get("run_dir")}
    prefix = analysis.analysis_id + "/session-"
    paths = [ledger] if isinstance(ledger, Path) else list(ledger)
    rows = [
        r
        for path in paths
        for r in read_lines(path)
        if str(r.get("label", "")).startswith(prefix)
        or str(Path(r.get("run_dir") or "/missing").resolve()) in expected
    ]
    # A ledger may have been assembled from overlapping files. Refuse conflicting duplicates.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("run_dir") or row.get("run"))
        if key in unique and unique[key] != row:
            raise ValueError(f"conflicting ledger rows: {key}")
        unique[key] = row
    rows = list(unique.values())
    held = {str(Path(r.get("run_dir") or "/missing").resolve()) for r in rows}
    gaps = sorted(expected - held)
    problem = (
        "no matching runtime ledger rows"
        if not rows
        else (
            "session run directories absent from ledger"
            if gaps
            else ("no attempts in denominator" if attempts <= 0 else None)
        )
    )
    if rows and not sessions:
        problem = "runtime session records absent"
    for number, session in enumerate(sessions, start=1):
        expected_calls = max(1, len(session.get("attempts", [])))
        if expected_calls > 1:
            label = f"{analysis.analysis_id}/session-{session.get('number', number):02d}"
            if sum(r.get("label") == label for r in rows) < expected_calls:
                problem = "failed retry calls absent from ledger"
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("backend")),
                str(row.get("model_reported") or row.get("model")),
                str(row.get("service_tier") or "default"),
            )
        ].append(row)
    tiers = []
    for (backend, model, tier), records in sorted(groups.items()):
        usage: dict[str, Any] = {}
        for field in TOKEN_FIELDS:
            values = [r.get("usage", {}).get(field) for r in records]
            usage[field] = sum(values) if all(isinstance(v, (int, float)) for v in values) else None
        seconds = [r.get("seconds") for r in records]
        wall = (
            sum(float(v) for v in seconds if v is not None)
            if all(isinstance(v, (int, float)) for v in seconds)
            else None
        )
        costs: list[float] = []
        sources: list[Any] = []
        for row in records:
            u = row.get("usage", {})
            cost = u.get("cost_usd_list_price")
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
                sources.append("runtime CLI list-price equivalent")
                continue
            price = (prices or {}).get(model) or (prices or {}).get(str(row.get("model")))
            if price is None or not price.get("source") or not price.get("as_of"):
                break
            if not all(isinstance(u.get(k), (int, float)) for k in TOKEN_FIELDS[:3]):
                break
            incoming, cached, outgoing = (float(u[k]) for k in TOKEN_FIELDS[:3])
            if cached > incoming or min(incoming, cached, outgoing) < 0:
                raise ValueError("invalid ledger token accounting")
            rates = [float(price[k]) for k in ("input", "cached_input", "output")]
            if any(not math.isfinite(r) or r < 0 for r in rates):
                raise ValueError("API-equivalent token rates must be finite and non-negative")
            costs.append(
                (
                    (incoming - cached) * float(price["input"])
                    + cached * float(price["cached_input"])
                    + outgoing * float(price["output"])
                )
                / 1_000_000
            )
            sources.append(price)
        equivalent = sum(costs) if len(costs) == len(records) else None
        factor = 1000 / attempts if attempts > 0 and problem is None else None
        tiers.append(
            {
                "backend": backend,
                "model": model,
                "service_tier": tier,
                "calls": len(records),
                "failed_calls": sum(not r.get("ok") for r in records),
                "tokens": usage,
                "wall_seconds": wall,
                "api_price_equivalent_usd": equivalent,
                "per_thousand_attempts": {
                    **{
                        k: None if v is None or factor is None else v * factor
                        for k, v in usage.items()
                    },
                    "wall_seconds": None if wall is None or factor is None else wall * factor,
                    "api_price_equivalent_usd": None
                    if equivalent is None or factor is None
                    else equivalent * factor,
                },
                "price_sources": sources,
                "price_missing": None
                if equivalent is not None
                else "no complete reported cost or sourced price schedule",
            }
        )
    return {
        "attempts_analysed": attempts,
        "ledger": str(paths[0]) if len(paths) == 1 else [str(p) for p in paths],
        "ledger_calls": len(rows),
        "missing_session_runs": gaps,
        "missing": problem,
        "tiers": tiers,
        "wall_time_basis": (
            "sum of CLI process wall times, including failed attempts; "
            "excludes experiment queue time"
        ),
    }
