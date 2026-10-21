"""Design power guarantee, ordinary incomplete reports and surviving review mutants."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
from scipy.stats import binom
from test_oracle import CAMERA, SCORING, toy_table, toy_world

from fleetlab.contracts.actions import CONFIG_CHANGE_KIND
from fleetlab.contracts.faults import FaultSpec, Target
from fleetlab.contracts.oracle import OracleRow, OracleTable, ReplayNeed, fault_list_hash
from fleetlab.contracts.vocabulary import ConfigKey
from fleetlab.oracle.chance import Chance
from fleetlab.oracle.exact import ExactChance
from fleetlab.oracle.gains import LOW, Estimate
from fleetlab.oracle.mechanism import MechanismTable, mechanism_table
from fleetlab.oracle.plan import proof_jobs, read_manifest
from fleetlab.oracle.power import (
    BOOTSTRAP_ERROR,
    POWER,
    ROUNDING_GUARD_POINTS,
    SAMPLE_ERROR,
    sufficient_power_bound,
)
from fleetlab.oracle.report import (
    ADDITIVE_RELATIVE_TOLERANCE,
    additive_numbers,
    compute_report,
    report_status,
)
from fleetlab.oracle.steps import Oracle
from fleetlab.oracle.table import TableBuild
from fleetlab.oracle.world import Cause, OracleLayout


@pytest.mark.parametrize("deviations", [0, 1, 3, 7])
def test_power_bound_does_not_use_observed_spread_or_harm(deviations: int) -> None:
    term = Estimate(
        item="pair",
        item_kind="interaction",
        episodes=28,
        net=-deviations,
        detail={"summed_harm_net": 2 * deviations, "excess_se_points": deviations / 10},
    )
    detail = additive_numbers(term, 336)
    expected = sufficient_power_bound(336, 28, 1000, ADDITIVE_RELATIVE_TOLERANCE)
    assert detail["sufficient_excess_power_bound_points"] == expected.points
    assert expected.points is not None and detail["excess_power_guarantee"] == POWER
    assert detail["min_reliable_excess_points"] is None


@pytest.mark.parametrize("n,m,boot", [(336, 28, 1000), (336, 112, 1000), (28, 28, 100)])
def test_design_bound_satisfies_each_analytic_failure_budget(n: int, m: int, boot: int) -> None:
    bound = sufficient_power_bound(n, m, boot, ADDITIVE_RELATIVE_TOLERANCE)
    assert bound.points is not None and bound.sample_net is not None
    assert bound.bootstrap_tail_probability is not None
    c, p, d = bound.sample_net, bound.bootstrap_tail_probability, bound.points * n / 100
    variance = 4 * m - d * d / m
    assert variance / (variance + (d - c) ** 2) <= SAMPLE_ERROR
    gap = c - ADDITIVE_RELATIVE_TOLERANCE * max(0, m - c) - ROUNDING_GUARD_POINTS * n / 100
    tail = math.exp(-gap * gap / (8 * m + (2 / 3) * (2 + 2 * m / n) * gap))
    assert tail <= p * (1 + 1e-12)
    rank = math.floor((boot - 1) * LOW / 100) + 1
    assert binom.sf(rank - 1, boot, p) <= BOOTSTRAP_ERROR
    assert pytest.approx(POWER) == 1 - SAMPLE_ERROR - BOOTSTRAP_ERROR


def inert_change() -> FaultSpec:
    return FaultSpec(
        fault_id="f90",
        kind=CONFIG_CHANGE_KIND,
        schedule=CAMERA.schedule,
        target=Target(scope="r01", config_key=ConfigKey.software_version),
        nominal="old",
        level=1,
        params={"value": "new"},
    )


def test_stored_outcome_contradicting_mechanism_equivalence_is_refused() -> None:
    table = mechanism_table(toy_table())
    identity = SCORING[0]
    before = table.success(identity.episode_id, identity.faults)
    assert before is not None
    added = (*identity.faults, inert_change())
    table.add(
        OracleRow(
            episode_id=identity.episode_id, active_faults=fault_list_hash(added), success=not before
        )
    )
    with pytest.raises(ValueError, match="no-mechanism addition changed the outcome"):
        table.success(identity.episode_id, added)


def test_original_config_change_is_kept_even_with_a_known_empty_outcome() -> None:
    fault = inert_change()
    identity = SCORING[0].model_copy(update={"faults": [fault]})
    table = MechanismTable(
        [identity],
        [
            OracleRow(
                episode_id=identity.episode_id,
                active_faults=fault_list_hash([fault]),
                success=False,
            ),
            OracleRow(
                episode_id=identity.episode_id, active_faults=fault_list_hash([]), success=True
            ),
        ],
    )
    assert table.success(identity.episode_id, [fault]) is False
    assert table.success(identity.episode_id, []) is True


def test_fault_free_requests_outside_proof_sample_are_all_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = tuple(
        SCORING[0].model_copy(update={"episode_id": f"e{900000 + i:06d}", "faults": []})
        for i in range(41)
    )
    sampled = {j.identity.episode_id for j in proof_jobs(identities)}
    assert len(sampled) < len(identities)
    world = replace(toy_world(), identities=identities)
    oracle = Oracle(tmp_path, world, OracleLayout(tmp_path / "oracle", world.world_id))
    monkeypatch.setattr(Oracle, "table", lambda *a, **k: TableBuild(OracleTable(identities), []))
    requests = [ReplayNeed(i.episode_id, ()) for i in identities]
    record = oracle.step_fresh_plan("outside-sample", [], requests=requests)
    assert (record["jobs"], record["proof"], record["refused"]) == (0, 41, [])
    jobs = read_manifest(oracle.layout.fresh("outside-sample") / "jobs.jsonl")
    assert {j.identity.episode_id for j in jobs} == {i.episode_id for i in identities}
    assert len({j.key for j in jobs}) == len(jobs) == 41


@pytest.mark.parametrize("missing_exact", [None, "cannot resolve catalogue action"])
def test_unavailable_required_exact_floor_is_refused_not_incomplete(
    missing_exact: str | None,
) -> None:
    report, _ = compute_report(toy_world(), TableBuild(toy_table(), []), boot=0, draws=0)
    report.problems = []
    report.draws = 200
    report.chance = [
        Chance(
            budget="k = 3",
            draws=200,
            valued=200,
            skipped_fresh_replay=0,
            skipped_error=0,
            exact=None
            if missing_exact is None
            else ExactChance(method="count", missing=missing_exact),
        )
    ]
    assert report_status(report) == 1


def test_detectability_missing_only_for_replays_makes_report_incomplete() -> None:
    table = toy_table()
    # Preserve every as-run row, remove only the replay outcomes.
    table._rows = {
        (i.episode_id, fault_list_hash(i.faults)): bool(table.success(i.episode_id, i.faults))
        for i in table.identities
    }
    cause = Cause(
        unit="f01",
        fault_ids=("f01",),
        roles=("effective",),
        detectable=True,
        qualified_by=("scoring",),
        expected_points=10,
        counts=True,
    )
    world = replace(toy_world(), causes=(cause,))
    report, _ = compute_report(world, TableBuild(table, []), boot=0, draws=0)
    assert report.detectability[0].verdict == "missing"
    effect = report.effect("f01")
    assert effect is not None and effect.missing
    assert report_status(report) == 2
    assert all(p.startswith("incomplete ") for p in report.problems)
