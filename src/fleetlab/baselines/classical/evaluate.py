"""OFFLINE development evaluation, deliberately separate from the blind runtime.

Reads persisted answers and the hidden scoring oracle. Never calls analyse/run,
never writes a source world. Accepts any explicitly supplied world root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import Action
from fleetlab.contracts.tools import Diagnosis, RankedFixSet


def _check_answer_world(answer: Path, evidence: dict[str, Any], root: Path, world_id: str) -> None:
    if evidence["world_id"] == world_id:
        return
    from fleetlab.scoring.provenance import producer_binding

    name = "classical statistics"
    if evidence.get("condition") == "experiments":
        name += " with experiments"
    binding = producer_binding(name, answer, world_id=world_id, world_root=root)
    if binding.get("binding_missing"):
        raise ValueError("answer world binding unavailable: " + binding["binding_missing"])


def evaluate(world_root: Path, answer: Path, output: Path) -> dict[str, Any]:
    # These imports belong ONLY to evaluation, never to the analysis entry point.
    from fleetlab.oracle.table import build_table
    from fleetlab.oracle.world import OracleLayout, all_runs, load_world
    from fleetlab.scoring import ScoringInputs, score
    from fleetlab.scoring.measured import read_measured

    if output.exists():
        raise FileExistsError(output)
    world = load_world(world_root)
    outbox = answer / "outbox" if (answer / "outbox").is_dir() else answer
    diagnosis = Diagnosis.model_validate_json((outbox / "diagnosis.json").read_text())
    ranked = RankedFixSet.model_validate_json((outbox / "fix_set.json").read_text())
    evidence = json.loads((answer / "evidence.json").read_text())
    _check_answer_world(answer, evidence, world_root, world.world_id)
    if world.fault_table is None:
        raise ValueError("development truth is missing")
    layout = OracleLayout(world_root / "oracle", world.world_id)
    built = build_table(
        world.scoring(), world.excluded, all_runs(world_root, layout), world.world_hash
    )
    if built.problems():
        raise ValueError(f"oracle validation failed: {built.problems()}")
    measured = read_measured(layout.root / "measured_causes.json", world.world_id, world.world_hash)
    report = score(
        ScoringInputs(
            fault_table=world.fault_table,
            measured_causes=measured,
            oracle=built.table,
            truth=world.truth(),
            current=world.current(),
            diagnosis=diagnosis,
            fix_set=ranked,
            catalogue=world.catalogue(),
            seed=world.spec.seed,
            draws=200,
            confounders=[(p[0], p[1]) for p in world.spec.patterns.confounders],
        )
    )
    at_three = next(
        b for b in report.claim3.budgets if b.budget.kind == "count" and b.budget.amount == 3
    )
    from fleetlab.scoring.ranking import spearman

    informative_keys = {
        Action.model_validate(r["action"]).model_dump_json()
        for r in evidence["actions"]
        if r.get("informative_interval")
    }
    informative = [
        r
        for r in report.claim2.actions
        if r.true_points is not None and r.action.model_dump_json() in informative_keys
    ]
    informative_covered = sum(bool(r.covered) for r in informative)
    informative_rho = spearman(
        [r.expected_gain for r in informative],
        [float(r.true_points) for r in informative if r.true_points is not None],
    )
    claim1 = report.claim1 if measured is not None else report.claim1_curve
    assert claim1 is not None
    confounders = claim1.confounders
    summary = {
        "world": world.world_id,
        "claim1_basis": "measured causes" if measured is not None else "descriptive curves",
        "claim1_missing": claim1.missing,
        "scoring_episodes": report.scoring_episodes,
        "condition": evidence.get("condition", "logs"),
        "completed_experiments": evidence.get("completed_requests", 0),
        "oracle_proxy_lookups": evidence.get("oracle_lookups", 0),
        "mae_points": report.claim2.mae_points.value,
        "calibration_slope": report.claim2.slope.value,
        "decoy_rejection": claim1.decoy_rejection.value,
        "confounders_resolved": sum(r.resolved is True for r in confounders),
        "confounders_total": sum(r.resolved is not None for r in confounders),
        "informative_actions": len(informative),
        "informative_intervals_covered": informative_covered,
        "informative_interval_coverage": informative_covered / len(informative)
        if informative
        else None,
        "informative_rank_correlation": informative_rho,
        "warnings": len(diagnosis.warnings),
        "warning_scoring": (
            "Not implemented in this stage-2 scorer; emitted warnings "
            "are retained in the diagnosis."
        ),
        "budgets": [
            {
                "kind": b.budget.kind,
                "amount": b.budget.amount,
                "share": b.agent.share.value,
                "points": b.agent.points,
                "rescued": b.agent.gain.rescued if b.agent.gain else None,
                "broken": b.agent.gain.broken if b.agent.gain else None,
            }
            for b in report.claim3.budgets
        ],
        "recall": claim1.recall.value,
        "precision": claim1.precision.value,
        "detectable": claim1.detectable,
        "hits": claim1.hits,
        "false_positives": claim1.false_positives,
        "rank_correlation": report.claim2.spearman.value,
        "ranked_actions": report.claim2.valued,
        "interval_coverage": report.claim2.coverage.value,
        "k3_share": at_three.agent.share.value,
        "k3_points": at_three.agent.points,
        "k3_rescued": at_three.agent.gain.rescued if at_three.agent.gain else None,
        "k3_broken": at_three.agent.gain.broken if at_three.agent.gain else None,
        "achievable_k3_points": at_three.achievable.points,
        "chance_mean_share": at_three.random.mean_share.value,
        "chance_p5_share": at_three.random.p5_share.value,
        "chance_p95_share": at_three.random.p95_share.value,
        "chance_valued_draws": at_three.random.valued,
        "chance_requested_draws": at_three.random.draws,
        "chance_skipped_fresh_replay": at_three.random.skipped_fresh_replay,
        "chance_skipped_error": at_three.random.skipped_error,
        "needs_fresh_replay": len(report.needs_fresh_replay),
        "world_hash": world.world_hash,
        "note": "Stage-2 scorer, including its detectable-cause denominator and conditional "
        "random baseline (draws needing fresh replays are omitted and counted). "
        "This evaluation ran no simulations; experiment provenance is in answer evidence. "
        "Traceability has no MCP call log for direct computation.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "score.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def oracle_probe(world_root: Path, answer_root: Path, output: Path) -> dict[str, Any]:
    """Optimistic, noise-free diagnostic of the fixed eight-request rule.

    The scoring-window oracle is NOT experiment-window data. No queue records,
    arm episodes or purported real experimental results are fabricated.
    """
    from fleetlab.baselines.classical.data import load_export
    from fleetlab.baselines.classical.experiments import choose, read_answer, revise
    from fleetlab.contracts.actions import Budget, FixSet
    from fleetlab.oracle.table import build_table
    from fleetlab.oracle.world import OracleLayout, all_runs, load_world
    from fleetlab.scoring.valuation import value_or_missing

    if output.exists():
        raise FileExistsError(output)
    original = read_answer(answer_root)
    world = load_world(world_root)
    _check_answer_world(answer_root, original.evidence, world_root, world.world_id)
    layout = OracleLayout(world_root / "oracle", world.world_id)
    built = build_table(
        world.scoring(), world.excluded, all_runs(world_root, layout), world.world_hash
    )
    if built.problems():
        raise ValueError(built.problems())
    estimates = []
    for action in choose(original):
        valued = value_or_missing(
            FixSet(actions=[action], budget=Budget(kind="count", amount=1)),
            built.table,
            world.truth(),
            world.current(),
        )
        if valued.gain is None:
            raise ValueError(f"oracle proxy needs a fresh replay: {valued.missing}")
        points = 100 * valued.gain.net / len(world.scoring())
        estimates.append(
            dict(
                action=action.model_dump(mode="json"),
                gain_points=points,
                low=points,
                high=points,
                confidence=float(points > 0),
                source="NOISE-FREE SCORING-WINDOW ORACLE PROXY, NOT A/B DATA",
            )
        )
    data = load_export(world_root / "export")
    revised = revise(original, estimates, data.costs, condition="oracle-proxy")
    revised.write(output / "answer")
    return evaluate(world_root, output / "answer", output / "score")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_root", type=Path)
    parser.add_argument("answer", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--oracle-proxy",
        action="store_true",
        help="noise-free eight-request diagnostic; never a real A/B condition",
    )
    args = parser.parse_args()
    operation = oracle_probe if args.oracle_proxy else evaluate
    print(json.dumps(operation(args.world_root, args.answer, args.out), indent=2))


if __name__ == "__main__":
    main()
