"""Regressions for the second vendor review; tables and execution mocks only."""

from pathlib import Path

import pytest
from test_oracle import CAMERA, SCORING, toy_table, toy_world, world_spec
from test_oracle_review import pair_world

from fleetlab.contracts.actions import Action, Budget
from fleetlab.contracts.oracle import OracleTable
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.contracts.vocabulary import Verb
from fleetlab.oracle.gains import Estimate, Gains
from fleetlab.oracle.report import (
    PatternCheck,
    compute_report,
    pattern_checks,
    summary_markdown,
)
from fleetlab.oracle.table import TableBuild
from fleetlab.oracle.world import OracleWorld


def additive_check(estimate: Estimate) -> PatternCheck:
    world = OracleWorld(spec=world_spec(), identities=(), additive_pairs=(("f01", "f02"),))
    return next(
        c
        for c in pattern_checks(
            world,
            {
                "fault_effects": [],
                "interactions": [estimate],
                "by_day": [],
                "scoring_episodes": 100,
            },
        )
        if c.pattern == "additive_pair"
    )


@pytest.mark.parametrize(
    ("low", "high", "summed", "harm_low", "verdict"),
    [
        (-1, 1, 10, 2, "not contradicted"),
        (-1.01, 1, 10, 2, "not contradicted"),
        (-1, 1.01, 10, 2, "not contradicted"),
        (1.01, 1.5, 10, 2, "below the sum"),  # joint harm 30 must not enlarge the band
        (-1, 1, -10, -12, "not contradicted"),
        (0, 0, 0, 0, "not contradicted"),
        (0, 0, 10, 0, "not contradicted"),
        (0, 0, 0, 2, "not contradicted"),
        (-0.8, -0.2, 10, 2, "not contradicted"),  # excludes zero but remains inside the band
        (-2, -1.01, 10, 2, "contradicted"),
        (None, None, 10, None, "not contradicted"),
    ],
)
def test_additive_only_contradicts_outside_the_band(
    low: float | None, high: float | None, summed: int, harm_low: float | None, verdict: str
) -> None:
    estimate = Estimate(
        item="f01+f02",
        item_kind="interaction",
        net=-1 if high is not None and high < 0 else 0,
        points=0,
        low=low,
        high=high,
        episodes=28,
        detail={
            "joint_net": 30,
            "alone_net": [15, 15],
            "summed_harm_net": summed,
            "summed_harm_low": harm_low,
            "summed_harm_high": 12,
        },
    )
    assert additive_check(estimate).verdict == verdict
    if summed < 0:
        from fleetlab.oracle.report import additive_verdict

        assert additive_verdict(estimate, 100)[1] == 0


def test_interaction_interval_ending_at_zero_is_not_confirmed() -> None:
    world, table = pair_world([(False, True, True, True)] * 3 + [(True, True, True, True)] * 25)
    estimate = Gains(table, boot=1000, seed=7).interaction("f01", "f02", True)
    assert estimate.net == -3 and estimate.high == 0
    check = next(
        c
        for c in pattern_checks(
            world, {"fault_effects": [], "interactions": [estimate], "by_day": []}
        )
        if c.pattern == "interaction"
    )
    assert check.verdict == "not confirmed"


def test_standalone_harms_keep_fault_order_and_bootstrap_sum() -> None:
    _, table = pair_world([(False, False, True, True)] * 5 + [(False, True, False, True)] * 13)
    gains = Gains(table, boot=500, seed=7)
    estimate = gains.interaction("f01", "f02", True)
    assert estimate.detail["standalone_harm_net"] == [5, 13]
    assert estimate.detail["summed_harm_net"] == 18
    assert estimate.detail["summed_harm_low"] == 100
    assert estimate.detail["summed_harm_high"] == 100


def test_additive_positive_harm_interval_cannot_override_zero_measured_harm() -> None:
    world, table = pair_world([(False, False, True, True), (True, True, False, False)])
    world = OracleWorld(
        spec=world_spec(), identities=world.identities, additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(table, []), boot=1, seed=4, draws=0)
    term = report.interactions[0]
    assert term.detail["summed_harm_net"] == 0 and term.detail["summed_harm_low"] > 0
    assert (
        next(c for c in report.patterns if c.pattern == "additive_pair").verdict
        == "not contradicted"
    )


def test_missing_fault_free_row_keeps_zero_gain_but_nulls_both_policy_counts() -> None:
    identity = SCORING[0].model_copy(update={"faults": []})
    world = OracleWorld(spec=world_spec(), identities=(identity,))
    table = OracleTable([identity])
    report, _ = compute_report(world, TableBuild(table, []), boot=0, draws=0)
    assert report.all_faults.net == 0 and report.all_faults.missing is None
    assert report.still_failing is None and report.policy_failures_on_healthy_robot is None
    assert report.all_removed_missing == 1 and report.problems
    from fleetlab.oracle.fresh import needs_of_oracle

    assert needs_of_oracle(table).unique()[0].episode_id == identity.episode_id


def test_marked_pairs_without_carriers_have_estimate_rows() -> None:
    world = OracleWorld(
        spec=world_spec(), identities=tuple(SCORING), additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(toy_table(), []), boot=0, draws=0)
    estimate = next(e for e in report.interactions if e.item == "f01+f02")
    assert estimate.net is None and estimate.missing == "no episode carries both"
    assert estimate.detail["marked"] is True


def test_cached_files_without_manifest_cannot_acquire_origin(tmp_path: Path) -> None:
    from fleetlab.oracle.run import RunRefused, run_here

    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "cached.parquet").write_text("unknown origin")
    with pytest.raises(RunRefused, match="origin"):
        run_here(
            directory,
            [Job(identity=SCORING[0], mode=JobMode.replay)],
            [],
            4,
            manifest_file=tmp_path / "absent.jsonl",
            world_hash=None,
        )
    assert not (tmp_path / "absent.jsonl").exists()
    assert not (directory / "oracle-origin.json").exists()


def test_unknown_baseline_allows_both_sides_of_chance_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.oracle import chance

    table = OracleTable(SCORING[:1])
    monkeypatch.setattr(chance, "random_draws", lambda *a, **k: [["fresh"]])
    result = chance.chance_intervals(
        table,
        toy_world().truth(),
        None,
        [Action(verb=Verb.recalibrate, target=CAMERA.target)],
        seed=7,
        achievable={},
        draws=1,
        budgets=[Budget(kind="count", amount=1)],
    )[0]
    assert result.bounds is not None
    assert result.bounds["mean_points"] == (-100, 100)


@pytest.mark.parametrize("case", ["problems", "fault_effect", "chance", "healthy", "complete"])
def test_report_command_fails_for_refused_or_incomplete_results(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.oracle.__main__ import main
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.world import OracleLayout

    oracle = Oracle(tmp_path, toy_world(), OracleLayout(tmp_path))
    report, _ = compute_report(toy_world(), TableBuild(toy_table(), []), boot=50, draws=2)
    report.problems = []
    report.chance = []
    if case == "problems":
        report.problems = ["run refused: no origin"]
    elif case == "fault_effect":
        report.fault_effects[0].missing = "MissingReplays"
    elif case == "chance":
        from fleetlab.oracle.chance import Chance

        report.chance = [
            Chance(
                budget="4 h",
                draws=2,
                valued=1,
                skipped_fresh_replay=1,
                skipped_error=0,
                missing="draws need replays",
            )
        ]
    elif case == "healthy":
        report.all_removed_missing = 1
    oracle.layout.summary.write_text("test report")
    monkeypatch.setattr(Oracle, "open", lambda *a, **k: oracle)
    monkeypatch.setattr(Oracle, "step_report", lambda *a, **k: report)
    assert main(["report", str(tmp_path), "--no-truth-table"]) == (
        0 if case == "complete" else 1 if case == "problems" else 2
    )


def test_bootstrap_off_text_does_not_repeat_interval_unavailable() -> None:
    report, _ = compute_report(toy_world(), TableBuild(toy_table(), []), boot=0, draws=0)
    assert "interval unavailable, interval unavailable" not in summary_markdown(report)


def test_additive_bootstrap_off_reports_uncertainty_without_none_endpoints() -> None:
    world, table = pair_world([(False, False, True, True)] * 28)
    world = OracleWorld(
        spec=world_spec(), identities=world.identities, additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(table, []), boot=0, draws=0)
    check = next(c for c in report.patterns if c.pattern == "additive_pair")
    assert check.verdict == "not contradicted"
    assert "summed harm interval unavailable" in check.text
    assert "None" not in check.text
