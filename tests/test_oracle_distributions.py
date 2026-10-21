"""Full population bounds, binomial percentile uncertainty and any-window requests."""

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom
from test_oracle import CAMERA, SCORING, toy_table, toy_world
from test_oracle_exact import entries

from fleetlab.contracts.actions import Budget, FixSet
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import OracleRow, ReplayNeed, fault_list_hash, value_fix_set
from fleetlab.oracle.chance import chance_intervals, percentile_ranks
from fleetlab.oracle.exact import exact_chance
from fleetlab.oracle.fresh import fresh_jobs, read_requests
from fleetlab.oracle.steps import Oracle
from fleetlab.oracle.valuation import DrawValuer


def test_exact_bounds_cover_every_completion_and_account_for_every_set() -> None:
    table, truth = toy_table(), toy_world().truth()
    budget = Budget(kind="count", amount=2)
    key = (SCORING[4].episode_id, fault_list_hash([SCORING[4].faults[-1]]))
    del table._rows[key]
    result = exact_chance(DrawValuer(table, truth, None), entries(), budget)
    assert result.missing is None and result.unvalued_sets > 0 and result.bounds
    assert result.sets == sum(w for _, w in result.histogram) + result.unvalued_sets == 6
    assert result.mean_points is result.p5_points is result.p95_points is None
    for success in (False, True):
        table._rows[key] = success
        sets = itertools.combinations(entries(), 2)
        nets = [
            value_fix_set(FixSet(actions=list(a), budget=budget), table, truth).net for a in sets
        ]
        assert Counter(nets).total() == result.sets
        pts = 100 * np.array(nets) / len(table.identities)
        actual = dict(
            mean_points=float(pts.mean()),
            p5_points=float(np.percentile(pts, 5)),
            p95_points=float(np.percentile(pts, 95)),
        )
        for field, value in actual.items():
            lo, hi = result.bounds[field]
            assert lo <= value <= hi


def test_exact_unresolvable_action_is_unavailable_not_unknown_mass() -> None:
    from fleetlab.contracts.actions import Action
    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.vocabulary import ConfigKey, Verb

    action = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.software_version),
        value="new",
    )
    result = exact_chance(
        DrawValuer(toy_table(), toy_world().truth(), None), [action], Budget(kind="count", amount=1)
    )
    assert result.missing and "cannot resolve" in result.missing
    assert result.bounds is None and result.histogram == []


@pytest.mark.parametrize("n", [1, 10, 200, 1000])
def test_percentile_ranks_have_distribution_free_coverage(n: int) -> None:
    low, high, coverage = percentile_ranks(n)
    assert 0 <= low < high <= n + 1
    assert coverage >= 0.95
    assert coverage == pytest.approx(binom.cdf(high - 1, n, 0.95) - binom.cdf(low - 1, n, 0.95))
    if n == 200:
        assert (low, high) == (184, 197)


def test_cost_percentile_interval_contains_sample_order_statistics_and_missing_range() -> None:
    table, truth = toy_table(), toy_world().truth()
    budgets = [Budget(kind="cost", amount=0.6)]
    complete = chance_intervals(
        table, truth, None, entries(), seed=7, achievable={}, draws=200, budgets=budgets
    )[0]
    assert complete.p95_interval_order_stats == (184, 197)
    assert complete.p95_interval_coverage and complete.p95_interval_coverage >= 0.95
    assert complete.p95_points_interval
    assert complete.p95_points is not None
    assert complete.p95_points_interval[0] <= complete.p95_points <= complete.p95_points_interval[1]
    table._rows.clear()
    missing = chance_intervals(
        table, truth, None, entries(), seed=7, achievable={}, draws=200, budgets=budgets
    )[0]
    assert missing.p95_points_interval
    assert missing.p95_points_interval[0] <= complete.p95_points_interval[0]
    assert missing.p95_points_interval[1] >= complete.p95_points_interval[1]


def test_request_envelope_is_the_scorers_record_format(tmp_path: Path) -> None:
    path = tmp_path / "score.json"
    path.write_text(
        json.dumps(
            {
                "needs_fresh_replay": [
                    {
                        "episode_id": SCORING[0].episode_id,
                        "fault_ids": [],
                        "faults": [],
                        "needed_for": ["claim 5 cited example"],
                    }
                ]
            }
        )
    )
    assert read_requests([path]) == [ReplayNeed(SCORING[0].episode_id, ())]
    path.write_text(
        json.dumps([{"episode_id": SCORING[0].episode_id, "fault_ids": ["f01"], "faults": []}])
    )
    with pytest.raises(ValueError, match="fault_ids differ"):
        read_requests([path])


def test_fresh_refuses_foreign_episode_and_altered_retained_fault() -> None:
    identity = SCORING[4]
    changed = CAMERA.model_copy(update={"level": 100})
    jobs, refused = fresh_jobs(
        {identity.episode_id: identity},
        [ReplayNeed("e999999", ()), ReplayNeed(identity.episode_id, (changed,))],
    )
    assert jobs == [] and len(refused) == 2


def test_observed_example_request_end_to_end(tmp_path: Path) -> None:
    from fleetlab.engine import software
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.oracle import toy

    root = tmp_path / "w99"
    manifest = toy.write_world(root, software.current("none"))
    config = RunConfig(
        out=str(root / "runs" / "toy"),
        batch_size=4,
        episodes_per_robot_day=4,
        **toy.engine_overrides(),
    )
    assert run(config, manifest).record.chain_ok is True
    oracle = Oracle.open(root, registry=toy.REGISTRY)
    observed: EpisodeIdentity = next(i for i in oracle.world.window("observed") if i.faults)
    need = ReplayNeed(observed.episode_id, ())
    assert oracle.table(all_windows=True).table.success(observed.episode_id, ()) is None
    plan = oracle.step_fresh_plan("examples", [], requests=[need])
    assert (plan["jobs"], plan["proof"], plan["refused"]) == (1, 1, [])
    info = oracle.step_fresh_run("examples", here=True, overrides=toy.engine_overrides())
    assert info["chain_ok"] is True and info["proof"]["identical"] == 1
    value = oracle.step_fresh_value("examples")[0]
    assert isinstance(value["success"], bool) and value["missing"] is None
    assert oracle.table(all_windows=True).table.success(observed.episode_id, ()) == value["success"]
    assert observed not in oracle.table().table.identities
    assert oracle.step_fresh_plan("examples", [], requests=[need])["jobs"] == 1


def test_causes_need_no_connection_to_marked_pairs() -> None:
    from dataclasses import replace

    from fleetlab.oracle.report import compute_report
    from fleetlab.oracle.table import TableBuild
    from fleetlab.oracle.world import Cause

    world = toy_world()
    causes = tuple(
        Cause(
            unit=unit,
            fault_ids=ids,
            roles=("effective",),
            detectable=False,
            qualified_by=(),
            expected_points=None,
            counts=False,
        )
        for unit, ids in [("f01", ("f01",)), ("f03", ("f03",)), ("f01+f02", ("f01", "f02"))]
    )
    patterns = world.spec.patterns.model_copy(update={"interactions": [("f01", "f03")]})
    world = replace(world, causes=causes, spec=world.spec.model_copy(update={"patterns": patterns}))
    assert ("f01", "f03") in world.spec.patterns.interactions
    report, _ = compute_report(world, TableBuild(toy_table(), []), boot=0, draws=0)
    assert [c.item for c in report.causes] == ["f01+f02"]
    assert {"f01", "f03", "f01+f02"} <= {d.item for d in report.detectability}
    assert report.causes[0].net == 3


@pytest.mark.parametrize(
    "kind,bounded,failed", [("count", False, True), ("count", True, False), ("cost", False, False)]
)
def test_required_count_floor_failure_and_bounded_success(
    kind: str, bounded: bool, failed: bool
) -> None:
    from fleetlab.oracle.chance import Chance
    from fleetlab.oracle.exact import ExactChance
    from fleetlab.oracle.report import compute_report, report_failed
    from fleetlab.oracle.table import TableBuild

    report, _ = compute_report(toy_world(), TableBuild(toy_table(), []), boot=0, draws=0)
    report.draws = 200
    report.chance = [
        Chance(
            budget="k = 3" if kind == "count" else "4 h",
            draws=200,
            valued=200,
            skipped_fresh_replay=0,
            skipped_error=0,
            exact=ExactChance(
                method="uniform count sets",
                missing=None if bounded else "counting limit",
                bounds={"p95_points": (0, 1)} if bounded else None,
            ),
        )
    ]
    assert report_failed(report) is failed


def test_fault_free_cited_outcome_can_be_supplied_by_its_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine import software
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.oracle import table as reader
    from fleetlab.oracle import toy
    from fleetlab.oracle.plan import read_manifest

    root = tmp_path / "w99"
    manifest = toy.write_world(root, software.current("none"))
    world_run = root / "runs" / "toy"
    config = RunConfig(
        out=str(world_run), batch_size=4, episodes_per_robot_day=4, **toy.engine_overrides()
    )
    assert run(config, manifest).record.chain_ok is True
    oracle = Oracle.open(root, registry=toy.REGISTRY)
    identity = next(i for i in oracle.world.window("observed") if not i.faults)
    original = reader.rows_of_run

    def missing_outcome(
        run_dir: Path, identities: dict[str, EpisodeIdentity], world_hash: str | None = None
    ) -> tuple[list[OracleRow], reader.RunRows]:
        rows, info = original(run_dir, identities, world_hash)
        if run_dir == world_run:
            rows = [r for r in rows if r.episode_id != identity.episode_id]
            info.rows = len(rows)
        return rows, info

    monkeypatch.setattr(reader, "rows_of_run", missing_outcome)
    assert oracle.table(all_windows=True).table.success(identity.episode_id, ()) is None
    request = ReplayNeed(identity.episode_id, ())
    planned = oracle.step_fresh_plan("healthy", [], requests=[request])
    assert planned["refused"] == []
    assert (planned["jobs"], planned["proof"]) == (0, 1)
    jobs = read_manifest(oracle.layout.fresh("healthy") / "jobs.jsonl")
    assert len(jobs) == 1 and not jobs[0].remove and not jobs[0].add
    info = oracle.step_fresh_run("healthy", here=True, overrides=toy.engine_overrides())
    assert info["chain_ok"] is True and info["proof"]["identical"] == 1
    assert isinstance(oracle.step_fresh_value("healthy")[0]["success"], bool)
    again = oracle.step_fresh_plan("healthy", [], requests=[request])
    assert (again["jobs"], again["proof"]) == (0, 1)
