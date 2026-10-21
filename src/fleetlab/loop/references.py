"""The two reference methods on the common reserved evaluation (the plan's comparisons).

    runs = run_references(round1_export, paths.root / "references")
    ledgers = {r.state: reference_ledger(r, world.world_id, k, layout.adoption_day(1))
               for r in runs if r.ok}
    evaluate(world, layout, {"original": l0, "round1": l1, "round2": l2, **ledgers}, ...)

The references, reused as they are and never retuned:

- `classical`: the fixed statistical procedure, one pass, logs only
  (`fleetlab.baselines.classical.run`, the `classical-frozen-5` lineage): it reads the
  export's visible tables and runs no experiment.
- `build_rule`: the rule that analyses nothing (`fleetlab.baselines.build_rule`): it
  reads the change log, which tasks each site runs, the catalogue and the recorded
  conditions, and no outcome.

Each reads round 1's export (the export round 1's agent read first: the world's observed
history), and its ranked list becomes a state by the one convention of the one-look plan
(`fleetlab.loop.state.one_look_ledger`): adoptable actions only, cut at the final state's
number of changes, adopted on round 1's adoption day. The build rule's ranked list is the
actions it names (at most three by its own rule); its format padding never enters.

A reference that fails is recorded as failed and has no state: missing, never zero.

Costs, in the shared cost format of `fleetlab.loop.costs` (lane driver): a reference runs
no model session, so the session figures are zero and complete (`usage_complete` true);
prices are unavailable (no model); it runs no experiment, so it simulates nothing before
the evaluation; its own computation is the wall time of the method in this process
(`compute_seconds`, measured; `None` when it was not measured, never guessed). The
attempts of the evaluation are the panel's, counted by the evaluation for every state.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import Action
from fleetlab.loop.state import Ledger, one_look_ledger

__all__ = [
    "REFERENCES",
    "ReferenceRun",
    "reference_costs",
    "reference_ledger",
    "run_reference",
    "run_references",
]

# The references, by the name of the state each becomes in the evaluation.
REFERENCES = ("classical", "build_rule")
LINEAGE = {"classical": "classical-frozen-5", "build_rule": "build-rule-v1"}
ROUND1_INPUTS = (
    "data_dictionary.md",
    "log/description.json",
    "log/episodes.parquet",
    "log/change_log.parquet",
    "log/maintenance.parquet",
    "log/health.parquet",
    "log/catalogue.parquet",
)


@dataclass(frozen=True)
class ReferenceRun:
    """One reference method on one export: its ranked actions, or why it has none."""

    state: str
    ok: bool
    ranked: tuple[Action, ...]
    answer_dir: Path | None
    compute_seconds: float | None
    record: dict[str, Any] = field(default_factory=dict)


def _hashes(export: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((export / name).read_bytes()).hexdigest()
        for name in ROUND1_INPUTS
        if (export / name).is_file()
    }


def run_reference(
    state: str, export: Path, out: Path, *, bootstraps: int = 200, seed: int = 0
) -> ReferenceRun:
    """Run one reference on `export`; its answer goes to `out/answer`, its record to
    `out/reference.json`. `out` must be new. A failure is recorded, not raised."""
    if state not in REFERENCES:
        raise ValueError(f"no reference {state!r}; the references are {list(REFERENCES)}")
    export = Path(export).expanduser().resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "state": state,
        "method": state,
        "lineage": LINEAGE[state],
        "export": str(export),
        "input_sha256": _hashes(export),
        "python": platform.python_version(),
        "reads": "round 1's export only; no experiment, no model",
    }
    started = time.perf_counter()
    try:
        if state == "classical":
            from fleetlab.baselines.classical import run

            answer = run(export, bootstraps=bootstraps, seed=seed)
            seconds = time.perf_counter() - started
            answer.write(out / "answer")
            ranked = tuple(r.action for r in answer.fix_set.actions)
            record.update(bootstraps=bootstraps, seed=seed, causes=len(answer.diagnosis.causes))
        else:
            from fleetlab.baselines.build_rule import PADDING, build_rule, write_rule

            rule = build_rule(export)
            seconds = time.perf_counter() - started
            write_rule(out / "answer", rule)
            ranked = tuple(r.action for r in rule.ranked.actions if r.rationale != PADDING)
            record.update(rule=rule.metadata.get("rule"))
    except Exception as error:  # recorded as missing, never as an empty answer
        record.update(
            status="failed",
            error=type(error).__name__,
            detail=traceback.format_exc(limit=8),
            # Not a computation of the method's answer: its cost stays unknown.
            compute_seconds=None,
            seconds_until_failure=round(time.perf_counter() - started, 3),
        )
        (out / "reference.json").write_text(json.dumps(record, indent=1) + "\n")
        return ReferenceRun(state, False, (), None, None, record)
    record.update(
        status="done",
        compute_seconds=round(seconds, 3),
        ranked=[a.model_dump(mode="json") for a in ranked],
    )
    (out / "reference.json").write_text(json.dumps(record, indent=1) + "\n")
    return ReferenceRun(state, True, ranked, out / "answer", seconds, record)


def run_references(
    export: Path, out: Path, states: Sequence[str] = REFERENCES, **kwargs: Any
) -> list[ReferenceRun]:
    """Every reference on `export`, each into `out/<state>/`."""
    return [run_reference(state, export, Path(out) / state, **kwargs) for state in states]


def reference_ledger(run: ReferenceRun, world_id: str, k: int, day: int) -> Ledger:
    """The reference's state: its ranked list cut at `k` changes (the final state's
    number of changes, `changes_in_force`) and adopted on `day` (round 1's adoption
    day), by the one-look plan's convention. Refuses a failed reference."""
    if not run.ok:
        raise ValueError(f"reference {run.state} failed; it has no state (missing, not zero)")
    return one_look_ledger(world_id, list(run.ranked), k, day)


def reference_costs(run: ReferenceRun, ledger: Ledger | None = None) -> dict[str, Any]:
    """The reference's costs in the shared format of `fleetlab.loop.costs`: the session
    figures (none: zero and complete), the experiments (none) and its own computation."""
    tokens = dict.fromkeys(
        ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"), 0
    )
    return {
        "method": run.state,
        "status": run.record.get("status", "failed" if not run.ok else "done"),
        "sessions": 0,
        "model_starts": 0,
        "failed_starts": 0,
        "starts_without_usage": 0,
        "usage_complete": True,
        "usage_note": "no model: the method is a computation over the export",
        "model_seconds": 0.0,
        "tokens": tokens,
        "tool_calls": 0,
        "cost_usd_list_price": None,
        "cost_usd_list_price_status": "unavailable",
        "api_equivalent_usd": {"status": "unavailable", "reason": "no model"},
        "per_session": [],
        "experiments": {"experiments": [], "requests": 0, "requests_refused": 0, "arm_attempts": 0},
        "simulated_jobs_before_evaluation": 0,
        "compute_seconds": None if run.compute_seconds is None else round(run.compute_seconds, 3),
        "compute_seconds_status": "measured" if run.compute_seconds is not None else "unknown",
        "changes": None if ledger is None else len(ledger.in_force()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the references on round 1's export and write their states and costs:

        uv run --locked python -m fleetlab.loop.references --world ROOT --loop DIR \\
            [--export DIR/rounds/r1/exports/start] (--k K | --final-ledger LEDGER.json)

    Writes `DIR/references/<state>/` (answer, reference.json), the ledgers
    `DIR/ledgers/reference-<state>.json` and `DIR/references/costs.json`. The states are
    then evaluated beside the agent's (`python -m fleetlab.loop.evaluate --state
    classical=DIR/ledgers/reference-classical.json ...`). CPU only.
    """
    import argparse

    from fleetlab.loop.layout import LoopLayout
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.state import changes_in_force
    from fleetlab.loop.world import LoopWorld

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.references")
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--export", type=Path, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--k", type=int)
    group.add_argument("--final-ledger", type=Path)
    args = parser.parse_args(argv)
    paths = LoopPaths(args.loop.expanduser().resolve())
    world = LoopWorld.load(args.world)
    layout = LoopLayout.model_validate_json(paths.layout.read_text())
    export = args.export or paths.exports(1) / "start"
    k = args.k if args.k is not None else changes_in_force(Ledger.load(args.final_ledger))
    runs = run_references(export, paths.root / "references")
    costs = {}
    for run in runs:
        ledger = None
        if run.ok:
            ledger = reference_ledger(run, world.world_id, k, layout.adoption_day(1))
            target = paths.ledgers / f"reference-{run.state}.json"
            paths.ledgers.mkdir(parents=True, exist_ok=True)
            target.write_text(ledger.model_dump_json(indent=1) + "\n")
        costs[run.state] = reference_costs(run, ledger)
    summary = {"k": k, "export": str(export), "costs": costs}
    (paths.root / "references" / "costs.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if all(run.ok for run in runs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
