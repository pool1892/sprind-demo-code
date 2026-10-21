"""The cost record of agent sessions: every model start counted, failed and replaced ones too.

A session's record keeps the usage and the tool calls of its *last* start only
(`SessionRecord.usage`, `SessionRecord.tool_calls`), while `SessionRecord.attempts`
keeps a receipt for every start. A total must sum the starts: a start lost to the
software around the model and replaced (design entry 11) still spent tokens, time and
tool calls. `session_costs` is that sum and the one helper for it; the driver, and any
lane that reports agent costs, use it.

Tokens and seconds come from each start's receipt; tool calls from each start's own
tool log (`<run_dir>/box/toolbox/tool_calls.jsonl`), so a replaced start's calls count.
A start without a usage record (an interrupted dispatch) is counted and never guessed:
it adds nothing to the sums, which become lower bounds (`usage_complete` false, with
`usage_note`), and the price is marked a lower bound with its partial figure. A
difference between two cost records (`cost_delta`) is unavailable when either side's
usage is incomplete: a lower bound minus a lower bound is no difference.

What the engine simulated is read from its run directories (`simulated_jobs`): the
finished jobs, and the engine's incidents (lost assignments that are run again): an
assignment whose execution had started counts as a confirmed failed simulation, one
whose start was never confirmed is labelled apart.

Probes the agent composes (docs/design/loop.md, section 10) are costed apart from the
experiments' task attempts everywhere: `probe_costs` per probe (charged as the probe
outcome says: requested while queued, executed and reserved once settled, nothing when
refused before it ran), and `queue_costs` with the probe runs' simulations
(`simulated_probes` of a queue record, `simulated_jobs` of the probe run: probe
episodes, chain-sample replays, partial jobs and unconfirmed assignments).

Names for other lanes: `session_costs`, `cost_delta`, `USAGE_FIGURES`,
`experiment_costs`, `probe_costs`, `queue_costs`, `simulated_jobs`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import SessionRecord

__all__ = [
    "TOKEN_KEYS",
    "USAGE_FIGURES",
    "cost_delta",
    "experiment_costs",
    "probe_costs",
    "queue_costs",
    "read_sessions",
    "session_costs",
    "simulated_jobs",
    "start_tool_calls",
]

TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
# The figures a start without a usage record leaves a lower bound (`cost_delta`).
USAGE_FIGURES = frozenset(
    {"model_seconds", *TOKEN_KEYS, "cost_usd_list_price", "api_equivalent_usd"}
)


def start_tool_calls(run_dir: str | Path) -> int:
    """The tool calls one CLI start made, from its own call log (0 when it has none)."""
    from fleetlab.agents.run import call_log

    log = call_log(Path(run_dir))
    if not log.is_file():
        return 0
    return sum(1 for line in log.read_text().splitlines() if line.strip())


def _tokens(starts: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total = dict.fromkeys(TOKEN_KEYS, 0)
    for start in starts:
        used = start.get("usage") or {}
        for key in TOKEN_KEYS:
            total[key] += int(used.get(key) or 0)
    return total


def _has_usage(start: Mapping[str, Any]) -> bool:
    meta = start.get("runtime_meta") or {}
    return bool(start.get("usage")) and not meta.get("interrupted_without_result")


def session_costs(sessions: Sequence[SessionRecord], model: str | None = None) -> dict[str, Any]:
    """The summed cost of `sessions` over every start: model starts and failed starts,
    model wall time, tokens, tool calls, the CLI's reported list price (claude) and, for
    a priced model, the API-equivalent bound (`integration.accounting.price_equivalent`,
    Fast rates; not a bill). Starts without a usage record make the usage figures lower
    bounds: `usage_complete` false, `usage_note` why, prices marked `lower_bound`."""
    from fleetlab.integration.accounting import PRICES, price_equivalent

    starts = [start for session in sessions for start in session.attempts]
    tokens = _tokens(starts)
    missing = sum(not _has_usage(s) for s in starts)
    complete = missing == 0
    note = (
        None
        if complete
        else f"{missing} of {len(starts)} model starts have no usage record: tokens, model "
        "seconds and prices are lower bounds"
    )
    per_session = [
        {
            "number": session.number,
            "state": session.state,
            "starts": len(session.attempts),
            "seconds": sum(float(a.get("seconds") or 0) for a in session.attempts),
            "tokens": _tokens(session.attempts),
            "usage_complete": all(_has_usage(a) for a in session.attempts),
            "tool_calls": sum(start_tool_calls(a["run_dir"]) for a in session.attempts),
        }
        for session in sessions
    ]
    listed = [
        price
        for start in starts
        if isinstance(price := (start.get("usage") or {}).get("cost_usd_list_price"), int | float)
    ]
    list_status = (
        "unavailable"
        if not listed
        else "complete"
        if complete and len(listed) == len(starts)
        else "lower_bound"
    )
    prices = PRICES.get(model) if model else None
    api: dict[str, Any] = (
        price_equivalent(tokens, prices)
        if prices is not None
        else {"status": "unavailable", "reason": "no reviewed price for this model"}
    )
    if not complete and prices is not None:
        api = {"status": "lower_bound", "reason": note, "partial": api}
    return {
        "sessions": len(sessions),
        "model_starts": len(starts),
        "failed_starts": sum(not s.get("ok") for s in starts),
        "starts_without_usage": missing,
        "usage_complete": complete,
        "usage_note": note,
        "model_seconds": sum(float(s.get("seconds") or 0) for s in starts),
        "tokens": tokens,
        "tool_calls": sum(p["tool_calls"] for p in per_session),
        "cost_usd_list_price": round(sum(listed), 6) if listed else None,
        "cost_usd_list_price_status": list_status,
        "api_equivalent_usd": api,
        "per_session": per_session,
    }


def _figure(costs: Mapping[str, Any], key: str) -> float | None:
    value: Any
    if key in TOKEN_KEYS:
        value = (costs.get("tokens") or {}).get(key)
    elif key == "api_equivalent_usd":
        api = costs.get("api_equivalent_usd") or {}
        value = api.get("low_usd") if api.get("status") == "bounded_api_equivalent" else None
    else:
        value = costs.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def cost_delta(first: Mapping[str, Any], second: Mapping[str, Any], key: str) -> dict[str, Any]:
    """`second - first` for one figure of two cost records (`session_costs` and the like).
    A usage figure is unavailable when either side's usage is incomplete; any figure is
    unavailable when either side lacks it."""
    a, b = _figure(first, key), _figure(second, key)
    row: dict[str, Any] = {"first": a, "second": b}
    partial = [
        name
        for name, costs in (("first", first), ("second", second))
        if key in USAGE_FIGURES and not costs.get("usage_complete", True)
    ]
    if partial:
        row.update(
            difference=None,
            status="unavailable",
            reason=f"usage incomplete in {partial}: a lower bound minus a lower bound",
        )
    elif a is None or b is None:
        row.update(difference=None, status="unavailable", reason=f"{key} not reported")
    else:
        row.update(difference=b - a, status="complete")
    return row


def simulated_jobs(run_dir: Path) -> dict[str, Any]:
    """What the engine simulated in one run directory: finished run jobs and replays (the
    chain sample), the jobs left partial by a stopped run, and from the engine's incidents
    the lost assignments run again: `failed_confirmed` had started executing (or finished
    and reported late), `unconfirmed` were assigned without a confirmed start, and
    `not_delivered` never reached a worker."""
    from fleetlab.log import run_status

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_jobs": 0,
        "replays": 0,
        "partial_jobs": 0,
        "failed_confirmed": 0,
        "unconfirmed": 0,
        "not_delivered": 0,
    }
    if not run_dir.is_dir():
        return out
    status = run_status(run_dir)
    replays = sum("." in k for k in status.done)  # a replay's key is <episode>.<mode>.<hash>
    out.update(run_jobs=len(status.done) - replays, replays=replays)
    out["partial_jobs"] = len(status.partial)
    record = run_dir / "run.json"
    incidents: list[dict[str, Any]] = []
    if record.is_file():
        incidents = (json.loads(record.read_text()).get("extra") or {}).get("incidents") or []
    for incident in incidents:
        for state in (incident.get("assignment_status") or {}).values():
            if state in ("execution_started", "completed_report_late"):
                out["failed_confirmed"] += 1
            elif state == "not_delivered":
                out["not_delivered"] += 1
            else:
                out["unconfirmed"] += 1
    out["incidents"] = len(incidents)
    # Every confirmed simulation: the finished jobs and the lost ones that had started.
    out["simulated_confirmed"] = len(status.done) + out["failed_confirmed"]
    return out


def experiment_costs(
    analysis: Any, first_session: int = 1, last_session: int | None = None
) -> dict[str, Any]:
    """The experiments the agent requested in sessions `first_session` to `last_session` of
    an analysis (`agents.sessions.Analysis`): per experiment its arms, robots, status and
    the attempts simulated per arm. A refused request costs nothing; a request whose
    simulation failed has no result and is reported as pending, its partial run counted
    by the queue record (`HeavyQueue`)."""
    results = analysis.results_by_id()
    experiments = []
    for queued in analysis.requests():
        if queued.session < first_session or (
            last_session is not None and queued.session > last_session
        ):
            continue
        result = results.get(queued.experiment_id)
        arms = [len(a.episode_ids) for a in result.arms] if result is not None else []
        experiments.append(
            {
                "experiment_id": queued.experiment_id,
                "session": queued.session,
                "arms": len(queued.request.arms),
                "actions": [
                    [a.model_dump(mode="json") for a in arm.actions] for arm in queued.request.arms
                ],
                "robots": list(queued.request.robots),
                "episodes_per_arm": queued.request.episodes_per_arm,
                "status": result.status if result is not None else "pending",
                "message": result.message if result is not None else None,
                "attempts_per_arm": arms,
                "attempts": sum(arms),
            }
        )
    return {
        "experiments": experiments,
        "requests": len(experiments),
        "requests_refused": sum(e["status"] == "refused" for e in experiments),
        "arm_attempts": sum(e["attempts"] for e in experiments),
    }


def probe_costs(
    analysis: Any, first_session: int = 1, last_session: int | None = None
) -> dict[str, Any]:
    """The probes the agent requested in sessions `first_session` to `last_session` of an
    analysis, apart from the experiments: per probe its robots, repetitions, length,
    status and the attempts it charged (`contracts.probes.ProbeOutcome`: requested while
    queued; executed and reserved once settled, never mixed; none when refused)."""
    outcomes = analysis.probe_outcomes()
    probes = []
    for queued in analysis.probe_requests():
        if queued.session < first_session or (
            last_session is not None and queued.session > last_session
        ):
            continue
        outcome = outcomes.get(queued.probe_id)
        probes.append(
            {
                "probe_id": queued.probe_id,
                "session": queued.session,
                "robots": list(queued.request.robots),
                "repeats": queued.request.repeats,
                "steps_per_episode": queued.request.program.steps,
                "status": outcome.status if outcome is not None else "queued",
                "attempts_executed": outcome.attempts_executed if outcome is not None else 0,
                "attempts_reserved": outcome.attempts_reserved if outcome is not None else 0,
                "attempts_charged": (
                    outcome.charged if outcome is not None else queued.request.attempts
                ),
            }
        )
    return {
        "probes": probes,
        "probe_requests": len(probes),
        "probe_requests_refused": sum(p["status"] == "refused" for p in probes),
        "probe_attempts": sum(p["attempts_charged"] for p in probes),
        "probe_attempts_executed": sum(p["attempts_executed"] for p in probes),
        "probe_attempts_reserved": sum(p["attempts_reserved"] for p in probes),
    }


def queue_costs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The heavy-queue side of experiments (`HeavyQueue.records`, `HeavyRoundEngine`):
    every simulated job, the engine's chain-sample replays and its confirmed failed
    simulations included, unconfirmed assignments apart, simulation and waiting time.
    The probe runs' simulations (`simulated_probes`) are summed apart (`*_probe_*`)."""

    def total(key: str, part: str = "simulated") -> int:
        return sum(int((r.get(part) or {}).get(key) or 0) for r in records)

    probes = {
        "simulated_probe_attempts": total("run_jobs", "simulated_probes"),
        "simulated_probe_replays": total("replays", "simulated_probes"),
        "simulated_probe_partial": total("partial_jobs", "simulated_probes"),
        "simulated_probe_unconfirmed": total("unconfirmed", "simulated_probes"),
        "simulated_probe_jobs": total("simulated_confirmed", "simulated_probes"),
    }
    return {
        **probes,
        "queues": len(records),
        "failed_queues": sum(r.get("state") == "failed" for r in records),
        "simulated_arm_attempts": total("run_jobs"),
        "simulated_replays": total("replays"),
        "simulated_failed_confirmed": total("failed_confirmed"),
        "simulated_partial": total("partial_jobs"),
        "simulated_unconfirmed": total("unconfirmed"),
        "simulated_jobs": total("simulated_confirmed"),
        "simulation_seconds": sum(
            float((r.get("timing") or {}).get("simulation_seconds") or 0) for r in records
        ),
        "queue_wait_seconds": sum(
            float((r.get("timing") or {}).get("queue_seconds") or 0) for r in records
        ),
        "wall_seconds": sum(float(r.get("wall_seconds") or 0) for r in records),
    }


def read_sessions(analysis_root: Path, numbers: Iterable[int] | None = None) -> list[SessionRecord]:
    """The stored session records of an analysis (optionally only some), for a cost total."""
    wanted = set(numbers) if numbers is not None else None
    records = []
    for path in sorted((analysis_root / "sessions").glob("session-*.json")):
        record = SessionRecord(**json.loads(path.read_text()))
        if wanted is None or record.number in wanted:
            records.append(record)
    return records
