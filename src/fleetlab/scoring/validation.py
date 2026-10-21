"""Constructed oracle-informed answers, solely for checking the scorer itself."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.actions import COSTS, Budget, FixSet
from fleetlab.contracts.tools import Cause, Diagnosis, Interval, RankedAction, RankedFixSet
from fleetlab.oracle.table import build_table
from fleetlab.oracle.valuation import best_removal, cheapest_fix, removal_options, value_or_missing
from fleetlab.oracle.world import OracleLayout, all_runs, load_world
from fleetlab.scoring.measured import measured_table, read_measured
from fleetlab.scoring.ranking import single


def constructed_answer(
    root: Path,
    base: Path,
    *,
    empty: bool = False,
    oracle_dir: Path | None = None,
    extra_runs: Sequence[Path] = (),
) -> Analysis:
    world = load_world(root)
    assert world.fault_table is not None
    analysis = Analysis.create(world.world_id, "constructed scorer check; hidden truth", base=base)
    (analysis.root / "run.json").write_text(
        json.dumps(
            {
                "condition": "logs alone",
                "repeat": 1,
                "issued_day": world.spec.windows.observed.last,
                "world_hash": world.world_hash,
                "synthetic": True,
            }
        )
    )
    measured = read_measured(
        (oracle_dir or root / "oracle") / "measured_causes.json", world.world_id, world.world_hash
    )
    if measured is None or not measured.complete or measured.problems:
        raise ValueError("constructed ceiling requires complete oracle measured causes")
    _, units = measured_table(measured, world.fault_table)
    counted = {f for unit in units for f in unit.fault_ids}
    diagnosis = Diagnosis(
        causes=[]
        if empty
        else [
            Cause(
                target=f.target,
                direction=f.directions[0],
                confidence=1,
                evidence=[],
                failure_mode="Constructed from hidden truth for scorer verification",
            )
            for f in world.fault_table.faults
            if f.fault_id in counted
        ],
        mode_summaries=[],
        warnings=[],
        retrain=[],
    )
    (analysis.outbox / "diagnosis.json").write_text(diagnosis.model_dump_json(indent=2))
    if empty:
        answer = FixSet(actions=[], budget=Budget(kind="count", amount=3))
        (analysis.outbox / "fix_set.json").write_text(answer.model_dump_json(indent=2))
        return analysis
    table = build_table(
        world.scoring(),
        world.excluded,
        all_runs(root, OracleLayout(oracle_dir or root / "oracle", world.world_id), extra_runs),
        world.world_hash,
    ).table
    options, needs = removal_options(table)
    if needs:
        raise ValueError("constructed ceiling requires complete oracle removals")
    _, best = best_removal(options, {f: 1 for f in table.fault_ids() if f not in table.excluded}, 3)
    by_id = {f.fault_id: f for f in world.spec.fault_schedule}
    selected = [
        fix[0] for f in best if (fix := cheapest_fix(by_id[f], world.truth(), COSTS)) is not None
    ]
    actions: list[RankedAction] = []
    for action in [*selected, *world.catalogue()]:
        if action in [r.action for r in actions]:
            continue
        valued = value_or_missing(single(action), table, world.truth(), world.current())
        if valued.gain is None or not action.has_replay:
            continue
        gain = 100 * valued.gain.net / len(table.identities)
        actions.append(
            RankedAction(
                action=action,
                expected_gain=gain,
                uncertainty=Interval(low=gain, high=gain),
                rationale="Hidden-truth scorer check",
            )
        )
        if len(actions) >= 12:
            break
    # Preserve the optimal 3-set; expected gains still come from each singleton.
    ranked = RankedFixSet(actions=actions)
    (analysis.outbox / "fix_set.json").write_text(ranked.model_dump_json(indent=2))
    return analysis


def validate_world(
    root: Path,
    out: Path,
    *,
    oracle_dir: Path | None = None,
    extra_runs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Ceiling, floor and replay coverage audit; no claim to validate a chance level."""
    import json

    import numpy as np

    from fleetlab.contracts.actions import SCORING_BUDGETS, cut
    from fleetlab.contracts.oracle import Gain, MissingReplays, value_fix_set
    from fleetlab.scoring.__main__ import score_world, write_score
    from fleetlab.scoring.pooling import pool, pool_markdown

    out.mkdir(parents=True, exist_ok=False)
    ceiling = constructed_answer(
        root, out / "answers", oracle_dir=oracle_dir, extra_runs=extra_runs
    )
    empty = constructed_answer(root, out / "answers", empty=True)
    upper = score_world(
        root,
        ceiling.root,
        method="constructed ceiling",
        oracle_dir=oracle_dir,
        extra_runs=extra_runs,
    )
    lower = score_world(
        root, empty.root, method="constructed empty", oracle_dir=oracle_dir, extra_runs=extra_runs
    )
    write_score(upper, out / "ceiling")
    write_score(lower, out / "empty")
    assert upper.claim1.recall.value == 1
    assert upper.claim2.spearman.value is not None and abs(upper.claim2.spearman.value - 1) < 1e-12
    assert upper.claim2.mae_points.value == 0
    assert (
        upper.headline.rows[0].share.value is not None
        and abs(upper.headline.rows[0].share.value - 1) < 1e-12
    )
    assert lower.claim1.recall.value == 0
    assert lower.headline.rows[0].share.value == 0
    assert all(b.agent.gain == Gain() for b in lower.claim3.budgets)
    world = load_world(root)
    table = build_table(
        world.scoring(),
        world.excluded,
        all_runs(root, OracleLayout(oracle_dir or root / "oracle", world.world_id), extra_runs),
        world.world_hash,
    ).table
    rng = np.random.default_rng(world.spec.seed)
    draws: list[list[Gain | None]] = [[] for _ in SCORING_BUDGETS]
    for _ in range(200):
        catalogue = world.catalogue()
        shuffled = [catalogue[int(i)] for i in rng.permutation(len(catalogue))]
        for b, budget in enumerate(SCORING_BUDGETS):
            try:
                gain = value_fix_set(cut(shuffled, budget), table, world.truth(), world.current())
            except (MissingReplays, ValueError, KeyError):
                gain = None
            draws[b].append(gain)
    checks = []
    for b, scored_budget in enumerate(upper.claim3.budgets):
        values = [100 * g.net / len(table.identities) for g in draws[b] if g is not None]
        checks.append(
            {
                "budget": scored_budget.label,
                "status": "verified" if len(values) == 200 else "unavailable",
                "valued": len(values),
                "draws": 200,
                "mean_points": sum(values) / len(values) if len(values) == 200 else None,
                "meaning": "direct value_fix_set coverage audit; "
                "does not validate chance percentiles",
            }
        )
    roster = out / "constructed-world-roster.json"
    roster.write_text(
        json.dumps({"worlds": [{"world_id": world.world_id, "world_hash": world.world_hash}]})
    )
    pooled = pool([upper, lower], roster=roster)
    (out / "comparison.json").write_text(json.dumps(pooled, indent=2) + "\n")
    (out / "comparison.md").write_text(pool_markdown(pooled))
    result = {
        "world_id": world.world_id,
        "ceiling_recall": upper.claim1.recall.value,
        "ceiling_spearman": upper.claim2.spearman.value,
        "ceiling_share_k3": upper.headline.rows[0].share.value,
        "empty_recall": lower.claim1.recall.value,
        "empty_share_k3": lower.headline.rows[0].share.value,
        "random": checks,
        "chance_population_validated": False,
        "synthetic_answers": True,
        "oracle_dir": str(oracle_dir or root / "oracle"),
        "extra_runs": [str(p) for p in extra_runs],
        "meaning": "stored real oracle records; constructed answers; no agent-performance evidence",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--extra-run", action="append", type=Path, default=[])
    args = parser.parse_args()
    result = validate_world(
        args.world_root.expanduser(),
        args.out.expanduser(),
        oracle_dir=None if args.oracle_dir is None else args.oracle_dir.expanduser(),
        extra_runs=[p.expanduser() for p in args.extra_run],
    )
    print(json.dumps(result, indent=2))
    return 0 if all(r["status"] == "verified" for r in result["random"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
