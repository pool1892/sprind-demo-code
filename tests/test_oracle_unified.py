"""Regressions for the unified mechanism rule and the last two vendor reviews."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from test_oracle import CAMERA, SCORING, toy_table, toy_world

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import CONFIG_CHANGE_KIND, Action, Budget, FixSet, resolve
from fleetlab.contracts.faults import FaultSpec, Target
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import (
    Gain,
    OracleRow,
    OracleTable,
    ReplayNeed,
    fault_list_hash,
    value_fix_set,
)
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey, Verb
from fleetlab.oracle.exact import ExactChance, _finish, exact_chance
from fleetlab.oracle.fresh import needs_of_fix_sets, value_all
from fleetlab.oracle.mechanism import MechanismTable, mechanism_table
from fleetlab.oracle.steps import Oracle
from fleetlab.oracle.table import TableBuild
from fleetlab.oracle.valuation import DrawValuer, value_or_missing
from fleetlab.oracle.world import OracleLayout


def test_engine_hooks_never_read_no_mechanism_keys() -> None:
    from fleetlab.faults.config_change import NO_MECHANISM, ConfigChange

    class Unreadable:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"a no-mechanism hook read {name}")

        def __getitem__(self, key: object) -> Any:
            raise AssertionError(f"a no-mechanism hook read {key}")

    fault_type = ConfigChange()
    for key in NO_MECHANISM:
        spec = FaultSpec(
            fault_id="f90",
            kind=CONFIG_CHANGE_KIND,
            schedule=CAMERA.schedule,
            target=Target(scope="r01", config_key=key),
            level=1,
            nominal="old",
            params={"value": "new"},
        )
        state = fault_type.new_episode(spec, SCORING[0])
        sentinel: Any = Unreadable()
        assert fault_type.cell_hook(sentinel, state) is None
        for hook in (
            fault_type.observation_hook,
            fault_type.action_hook,
            fault_type.policy_input_hook,
            fault_type.data_hook,
        ):
            assert hook(sentinel, state) is sentinel


def test_all_fix_set_consumers_share_the_mechanism_rule() -> None:
    raw, world = toy_table(), toy_world()
    inert = Action(
        verb=Verb.set,
        target=Target(scope="r02", config_key=ConfigKey.software_version),
        value="new",
    )
    remove = Action(verb=Verb.recalibrate, target=CAMERA.target)

    def current(identity: EpisodeIdentity) -> dict[ConfigKey | ConditionKey, ConfigValue]:
        return {ConfigKey.software_version: "old"}

    fixes = FixSet(actions=[inert, remove], budget=Budget(kind="count", amount=2))
    table = mechanism_table(raw)
    expected = value_fix_set(
        FixSet(actions=[remove], budget=Budget(kind="count", amount=1)), raw, world.truth()
    )
    assert value_fix_set(fixes, table, world.truth(), current) == expected
    assert value_or_missing(fixes, raw, world.truth(), current).gain == expected
    assert DrawValuer(raw, world.truth(), current).value(fixes.actions) == expected
    assert value_all([("agent", fixes)], raw, world.truth(), current)[0]["net"] == expected.net
    assert needs_of_fix_sets([("baseline", fixes)], raw, world.truth(), current).unique() == []
    change = resolve(
        inert, SCORING[4], SCORING[4].faults, world.truth(), current(SCORING[4])
    ).added[0]
    assert table.success(SCORING[4].episode_id, [*SCORING[4].faults, change]) is False
    # An original fault of this kind remains in force even on an inert key.
    identity = SCORING[0].model_copy(update={"faults": [change]})
    original = MechanismTable(
        [identity],
        [
            OracleRow(
                episode_id=identity.episode_id,
                active_faults=fault_list_hash([change]),
                success=False,
            )
        ],
    )
    assert original.success(identity.episode_id, ()) is None
    assert original.success(identity.episode_id, [change]) is False


@pytest.mark.parametrize("net", [0, 1])
def test_independent_statistic_stays_a_value_with_unknown_mass(net: int) -> None:
    table = toy_table()
    result = _finish(
        ExactChance(method="test", unvalued_sets=1),
        Counter({net: 1000}),
        DrawValuer(table, toy_world().truth(), None),
    )
    assert result.mean_points is None and result.bounds and "mean_points" in result.bounds
    assert result.p5_points == result.p95_points == round(100 * net / 12, 3)
    assert "p5_points" not in result.bounds and "p95_points" not in result.bounds


def test_no_mechanism_fix_needs_no_outcome_to_prove_zero_gain() -> None:
    table, world = toy_table(), toy_world()
    table._rows.clear()
    action = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.software_version),
        value="new",
    )

    def current(identity: EpisodeIdentity) -> dict[ConfigKey | ConditionKey, ConfigValue]:
        return {ConfigKey.software_version: "old"}

    fix_set = FixSet(actions=[action], budget=Budget(kind="count", amount=1))
    assert value_fix_set(fix_set, mechanism_table(table), world.truth(), current) == Gain()
    assert value_or_missing(fix_set, table, world.truth(), current).gain == Gain()
    assert DrawValuer(table, world.truth(), current).value([action]) == Gain()
    assert value_all([("noop", fix_set)], table, world.truth(), current)[0]["net"] == 0
    assert needs_of_fix_sets([("noop", fix_set)], table, world.truth(), current).unique() == []
    # This proves an unchanged outcome, never invents the missing healthy outcome.
    assert mechanism_table(table).success(SCORING[0].episode_id, ()) is None


def test_exact_net_high_counts_failures_and_unknowns_not_successes() -> None:
    table = OracleTable(SCORING[:4])
    for identity, success in zip(SCORING[:3], [False, False, True], strict=True):
        table.add(
            OracleRow(
                episode_id=identity.episode_id,
                active_faults=fault_list_hash(identity.faults),
                success=success,
            )
        )
    result = _finish(
        ExactChance(method="test", unvalued_sets=1),
        Counter(),
        DrawValuer(table, toy_world().truth(), None),
    )
    assert result.net_low == -2 and result.net_high == 3
    assert result.bounds and result.bounds["mean_points"] == (-50, 75)


def test_upper_order_statistic_uses_exact_one_based_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.oracle import chance

    monkeypatch.setattr(
        chance, "random_draws", lambda *a, **kw: [[Gain()] * 196 + [Gain.of(1, 0)] * 4]
    )
    # The positive atom starts exactly at rank 197, not rank 196.
    row = chance.chance_intervals(
        toy_table(),
        toy_world().truth(),
        None,
        [Action(verb=Verb.recalibrate, target=CAMERA.target)],
        seed=7,
        achievable={},
        budgets=[Budget(kind="cost", amount=4)],
    )[0]
    assert row.p95_points_interval == (0, 8.334)


def test_known_requested_outcome_plans_no_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = Oracle(tmp_path, toy_world(), OracleLayout(tmp_path))
    monkeypatch.setattr(Oracle, "table", lambda *a, **k: TableBuild(toy_table(), []))
    request = ReplayNeed(SCORING[4].episode_id, ())
    record = oracle.step_fresh_plan("new-request", [], requests=[request])
    assert (record["jobs"], record["proof"], record["needs"]) == (0, 0, {})
    assert (oracle.layout.fresh("new-request") / "jobs.jsonl").read_text() == ""


def test_enumeration_limit_bounds_all_unvisited_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_oracle_exact import entries

    from fleetlab.oracle import exact

    raw = DrawValuer(toy_table(), toy_world().truth(), None)
    complete = exact_chance(raw, entries(), Budget(kind="count", amount=2))
    monkeypatch.setattr(exact, "EXACT_STATE_LIMIT", 2)
    bounded = exact_chance(raw, entries(), Budget(kind="count", amount=2))
    assert bounded.missing is None and bounded.counting_limited
    assert bounded.uncomputed_sets > 0 and bounded.uncomputed_sets <= bounded.unvalued_sets
    assert (
        bounded.sets
        == complete.sets
        == sum(w for _, w in bounded.histogram) + bounded.unvalued_sets
    )
    assert bounded.bounds
    for key in ("mean_points", "p5_points", "p95_points"):
        value = getattr(complete, key)
        if key in bounded.bounds:
            lo, hi = bounded.bounds[key]
            assert value is not None and lo <= value <= hi
        else:
            assert getattr(bounded, key) == value


def test_additive_report_scales_the_band_by_the_actual_window() -> None:
    from test_oracle_review import pair_world

    from fleetlab.oracle.report import ADDITIVE_RELATIVE_TOLERANCE, compute_report
    from fleetlab.oracle.world import OracleWorld

    world, table = pair_world([(False, False, False, True)] * 28)
    world = OracleWorld(
        spec=world.spec, identities=world.identities, additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(table, []), boot=100, draws=0)
    term = report.interactions[0]
    detail = term.detail
    assert report.scoring_episodes == 28
    assert (
        detail["additive_tolerance_points"]
        == 100 * ADDITIVE_RELATIVE_TOLERANCE * detail["summed_harm_net"] / 28
        == 20
    )
    assert detail["excess_net"] == -28 and detail["excess_points"] == -100
    assert (
        detail["excess_low"]
        == detail["excess_high"]
        == detail["max_excess_not_excluded_points"]
        == -100
    )
    check = next(p for p in report.patterns if p.pattern == "additive_pair")
    assert check.verdict == "below the sum"
    assert "band -20.000 to +20.000 points" in check.text
    assert sum("Additivity rests" in note for note in report.notes) == 1


def test_report_sources_are_recovered_without_runs_option(tmp_path: Path) -> None:
    import json

    from fleetlab.engine import software
    from fleetlab.oracle import toy

    root = tmp_path / "world"
    toy.write_world(root, software.current("none"))
    directory, extra = tmp_path / "oracle", tmp_path / "external-run"
    oracle = Oracle.open(root, directory, [extra], registry=toy.REGISTRY)
    oracle.step_plan()
    reopened = Oracle.open(root, directory, registry=toy.REGISTRY)
    assert extra in reopened.runs()
    # Reports written before sources.json also carry all required paths.
    (directory / "sources.json").unlink()
    (directory / "oracle.json").write_text(
        json.dumps({"world_hash": oracle.world.world_hash, "runs": [{"run_dir": str(extra)}]})
    )
    assert extra in Oracle.open(root, directory, registry=toy.REGISTRY).runs()


def test_paired_spread_is_diagnostic_not_a_power_threshold() -> None:
    from test_oracle_review import pair_world

    from fleetlab.oracle.report import compute_report
    from fleetlab.oracle.world import OracleWorld

    world, table = pair_world([(False, True, True, True)] * 7 + [(False, False, True, True)] * 21)
    world = OracleWorld(
        spec=world.spec, identities=world.identities, additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(table, []), boot=0, draws=0)
    detail = report.interactions[0].detail
    assert detail["excess_net"] == 7 and detail["summed_harm_net"] == 21
    assert detail["excess_se_points"] == pytest.approx(100 / 12)
    assert detail["min_reliable_excess_points"] is None
    assert detail["reliable_excess_power"] is None
    assert "withdrawn" in detail["reliable_excess_method"]
    assert detail["sufficient_excess_power_bound_points"] is None  # boot=0
    assert "bootstrap" in detail["excess_power_unavailable"]
    assert detail["max_excess_not_excluded_points"] is None


@pytest.mark.parametrize("case", ["missing", "fewer_rows", "fewer_draws", "less_bootstrap"])
def test_complete_report_cannot_be_downgraded(tmp_path: Path, case: str) -> None:
    from fleetlab.oracle.report import ReportRefused, compute_report, write_report

    report, episodes = compute_report(toy_world(), TableBuild(toy_table(), []), boot=50, draws=0)
    report.problems = []
    report.chance = []
    write_report(tmp_path, report, episodes)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    candidate = report.model_copy(deep=True)
    if case == "missing":
        candidate.all_removed_missing = 1
    elif case == "fewer_rows":
        candidate.rows -= 1
    elif case == "fewer_draws":
        report.draws = 200
        write_report(tmp_path, report, episodes)
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    else:
        candidate.boot = 0
    with pytest.raises(ReportRefused, match="complete report preserved"):
        write_report(tmp_path, candidate, [])
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
