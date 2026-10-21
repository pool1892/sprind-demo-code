"""The skills' helpers on the skills' toy fleet (tests/skills_fleet.py): they find what was planted.

The fleet has a gripper cause on one robot, a harmful configuration change at
one site logged on the same day as a harmless software update, a drifting
camera on another robot and a harmless calibration change. Each helper runs
its SKILL.md block through the same `LogReader` the tool server uses.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import skills_fleet as sf
from scipy.stats import norm
from skills_support import Exports, helpers

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.tools import Interval, RankedAction
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.log import LogReader

triage = helpers("triage")
cards = helpers("episode-cards")
modes = helpers("failure-modes")
scope = helpers("scope")
from_logs = helpers("sensitivity-from-logs")
experiment = helpers("sensitivity-by-experiment")
rank = helpers("rank-actions")


@pytest.fixture(scope="module")
def reader(skill_exports: Exports) -> Any:
    with LogReader(skill_exports.fleet_export, materialize=True, lock=True) as opened:
        yield opened


def by_level(rows: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, Any]]:
    return {r["level"]: r for r in rows if r["dimension"] == dimension}


def test_fill_changes_only_params_and_input(reader: Any) -> None:
    sql = from_logs.blocks()["gains"]
    filled = from_logs.fill(
        sql, {"project_to_day": 9}, [("a", "fleet", 0, 1.0), ("b's", "fleet", 1, 0.5)]
    )
    assert "9 AS project_to_day" in filled and "1.96 AS z" in filled
    assert "('b''s', 'fleet', 1, 0.5)" in filled
    assert "everything at" not in filled  # the default input is gone
    assert filled.count("-- helper: gains") == 1
    with pytest.raises(KeyError, match="no parameters"):
        from_logs.fill(sql, {"unknown": 1})
    with pytest.raises(ValueError, match="at least one row"):
        from_logs.fill(sql, None, [])
    assert from_logs.fill(sql) == sql.rstrip("\n")


def test_triage_flags_the_robot_with_the_planted_cause(reader: Any) -> None:
    rows = triage.outcomes(reader)
    fleet = by_level(rows, "fleet")["fleet"]
    assert fleet["n"] == sf.DAYS * sf.EPISODES_PER_DAY * 6
    robots = by_level(rows, "robot")
    worst = max(robots.values(), key=lambda r: r["points_below_expected"])
    assert worst["level"] == sf.GRIPPER_ROBOT and worst["stands_out"] == "worse"
    assert worst["drops_per_ep"] > 0 and all(
        r["drops_per_ep"] == 0 for k, r in robots.items() if k != sf.GRIPPER_ROBOT
    )
    for row in rows:
        assert row["rate_low"] <= row["rate"] <= row["rate_high"]
    # The task mix: the first site does the hard tasks, so its expectation is lower.
    sites = by_level(rows, "site")
    assert sites["s1"]["task_expected"] < sites["s2"]["task_expected"]
    # Points below expectation over all robots roughly add up to the loss against tasks.
    assert len(by_level(rows, "task")) == len(sf.TASKS)
    assert len(by_level(rows, "day")) == sf.DAYS


def test_the_sample_is_stratified_and_reproducible(reader: Any) -> None:
    first = triage.sample(reader, failures_per_robot=2, successes_per_robot=1)
    again = triage.sample(reader, failures_per_robot=2, successes_per_robot=1)
    other = triage.sample(reader, failures_per_robot=2, successes_per_robot=1, salt="b")
    assert first == again and first != other
    per_robot: dict[tuple[str, str], int] = {}
    for row in first:
        key = (row["robot_id"], row["outcome"])
        per_robot[key] = per_robot.get(key, 0) + 1
    assert all(per_robot[(r, "failure")] == 2 for r in sf.SITES["s1"] + sf.SITES["s2"])
    assert all(per_robot[(r, "success")] == 1 for r in sf.SITES["s1"] + sf.SITES["s2"])
    days = [
        r["day"] for r in first if r["robot_id"] == sf.GRIPPER_ROBOT and r["outcome"] == "failure"
    ]
    assert len(set(days)) == len(days), "picks spread over the days"
    assert len({row["episode_id"] for row in first}) == len(first)


def planned_by_id() -> dict[str, sf.Planned]:
    return {p.identity.identity.episode_id: p for p in sf.plan()}


def test_profile_reads_the_phase_and_picks_frames(reader: Any) -> None:
    planned = planned_by_id()
    chosen = [e for e, p in planned.items() if p.failure in ("drop", "miss")][:12]
    chosen += [e for e, p in planned.items() if p.identity.record_video][:6]
    rows = cards.profile(reader, chosen)
    assert sorted(r["episode_id"] for r in rows) == sorted(set(chosen))
    for row in rows:
        failure = planned[row["episode_id"]].failure
        if failure == "drop":
            assert (
                row["end_phase"] == "carry: lost the object"
                and row["collapse_step"] == sf.DROP_STEP
            )
        elif failure == "miss":
            assert row["end_phase"] == "grasp: closed on nothing"
        elif failure is None:
            assert row["end_phase"] == "completed" and row["reopen_step"] is not None
        logged = reader.query(
            "SELECT step FROM steps WHERE episode_id = ? AND frame_index IS NOT NULL",
            [row["episode_id"]],
        )["step"].to_pylist()
        assert set(row["frame_steps"]) <= set(logged) and len(row["frame_steps"]) <= 8
        assert row["has_video"] == bool(logged) and bool(row["frame_steps"]) == bool(logged)


def test_signatures_count_the_planted_mode_where_it_is(reader: Any) -> None:
    rows = {r["signature"]: r for r in modes.signatures(reader)}
    lost = rows["lost the object"]
    planned = planned_by_id()
    assert lost["failures"] == sum(p.failure == "drop" for p in planned.values())
    assert lost["by_robot"].startswith(f"{sf.GRIPPER_ROBOT} ")
    assert rows["collision"]["by_site"].startswith(f"{sf.DELAY_SITE} ")
    assert math.isclose(sum(r["share_of_failures"] for r in rows.values()), 1.0, abs_tol=0.01)


def test_cards_check_maps_modes_to_signatures(reader: Any) -> None:
    planned = planned_by_id()
    drops = [e for e, p in planned.items() if p.failure == "drop"][:3]
    misses = [e for e, p in planned.items() if p.failure == "miss"][:2]
    given = [(e, "object lost in the carry") for e in drops] + [(e, "missed grasp") for e in misses]
    rows = modes.cards_check(reader, given)
    assert {(r["mode"], r["signature"], r["cards"]) for r in rows} == {
        ("object lost in the carry", "lost the object", 3),
        ("missed grasp", "closed on nothing", 2),
    }


def test_breakdown_puts_a_signature_on_its_robot(reader: Any) -> None:
    rows = scope.breakdown(reader, signature="lost the object")
    robots = {r["level"]: r for r in rows if r["dimension"] == "robot"}
    assert robots[sf.GRIPPER_ROBOT]["share"] > 0.3
    assert all(r["hits"] == 0 for k, r in robots.items() if k != sf.GRIPPER_ROBOT)
    assert robots[sf.GRIPPER_ROBOT]["days"] == list(range(sf.DAYS))


def test_changes_show_the_confounded_pair(reader: Any) -> None:
    rows = scope.changes(reader, days_around=3)
    pair = [r for r in rows if r["day"] == sf.DELAY_DAY and r["scope"] == sf.DELAY_SITE]
    assert {r["what"] for r in pair} == {"control.action_delay_steps", "software.version"}
    assert pair[0]["did"] == pair[1]["did"] < 0, "the log cannot tell them apart"
    assert pair[0]["did"] < -0.1 and pair[0]["did_low"] < pair[0]["did"] < pair[0]["did_high"]
    assert all(r["same_day_overlapping"] for r in pair)
    benign = next(r for r in rows if r["scope"] == sf.BENIGN_ROBOT and r["kind"] == "change")
    assert benign["same_day_overlapping"] == "" and benign["did_low"] < 0 < benign["did_high"]
    maintenance = [r for r in rows if r["kind"] == "maintenance"]
    assert {r["what"] for r in maintenance} == {"gripper clean", "arm service"}


def test_health_puts_the_planted_readings_first(reader: Any) -> None:
    rows = scope.health(reader)
    top = {(r["robot_id"], r["component"], r["metric"]) for r in rows[:2]}
    assert top == {
        (sf.GRIPPER_ROBOT, "gripper", "grip_force"),
        (sf.DRIFT_ROBOT, "scene_camera", "calibration_residual"),
    }
    drift = next(
        r for r in rows if r["robot_id"] == sf.DRIFT_ROBOT and r["metric"] == "calibration_residual"
    )
    assert drift["slope_vs_fleet"] > 0.3 and drift["corr_with_failure"] > 0.5


def test_trend_finds_the_drift(reader: Any) -> None:
    rows = scope.trend(reader)
    robots = [r for r in rows if r["dimension"] == "robot"]
    # Against the rest of the fleet the site's step looks like a slope on its robots;
    # against the other robots of its own site only the drifting robot slopes.
    assert robots[0]["level"] == sf.DRIFT_ROBOT
    assert robots == sorted(robots, key=lambda r: r["within_site_slope"])
    drift = robots[0]
    assert drift["within_site_slope"] < 0 and drift["crossing_day"] is not None
    assert len(drift["daily_gap"]) == len(drift["daily_within_site_gap"]) == sf.DAYS
    site = next(r for r in rows if r["level"] == sf.DELAY_SITE)
    assert site["within_site_slope"] is None and site["slope_per_day"] < 0


def test_excluded_episodes_stay_out_of_the_rates(reader: Any) -> None:
    last_day = reader.query(
        "SELECT list(episode_id) AS ids FROM episodes WHERE day = ?", [sf.DAYS - 1]
    ).to_pylist()[0]["ids"]
    excluded = {"excluded": [(e,) for e in last_day]}
    fleet = by_level(triage.run(reader, "outcomes", None, None, excluded), "fleet")["fleet"]
    assert fleet["n"] == (sf.DAYS - 1) * sf.EPISODES_PER_DAY * 6
    assert len(by_level(triage.run(reader, "outcomes", None, None, excluded), "day")) == sf.DAYS - 1
    trend = scope.run(reader, "trend", None, None, excluded)
    assert all(len(r["daily_gap"]) == sf.DAYS - 1 for r in trend)
    breakdown = scope.run(reader, "breakdown", None, None, excluded)
    assert breakdown[0]["n"] == fleet["n"]
    gains = from_logs.gains(reader, candidates()[:1])
    fewer = from_logs.run(reader, "gains", None, [tuple(candidates()[0])], excluded)
    assert fewer[0]["n"] == gains[0]["n"] - sf.EPISODES_PER_DAY
    with pytest.raises(KeyError, match="no inputs"):
        triage.fill(triage.blocks()["outcomes"], None, None, {"picked": [("x",)]})
    with pytest.raises(KeyError, match="takes no input"):
        experiment.fill(experiment.blocks()["power"], None, [(1,)])


def test_changes_take_a_range_of_days(reader: Any) -> None:
    early = scope.changes(reader, to_day=1)
    assert {r["day"] for r in early} == {1}
    assert all(r["n_after"] == sf.EPISODES_PER_DAY for r in early)  # only day 1 after day 1


def test_signatures_fall_back_on_the_fingers_without_events(skill_exports: Exports) -> None:
    with LogReader(skill_exports.fleet_export, materialize=True) as unlocked:
        with_events = {r["signature"]: r["failures"] for r in modes.signatures(unlocked)}
        unlocked.con.execute("DELETE FROM events")
        without = {r["signature"]: r["failures"] for r in modes.signatures(unlocked)}
    planned = planned_by_id()
    assert with_events["lost the object"] == without["lost the object"]
    assert with_events["closed on nothing"] == without["closed on nothing"]
    assert without["lost the object"] == sum(p.failure == "drop" for p in planned.values())
    assert "collision" not in without  # an event the fingers cannot see


def true_gain_points(scope_id: str, from_day: int) -> float:
    """Points of fleet success the planted cause costs on its scope, from the planned odds."""
    planned = sf.plan()
    affected = []
    for p in planned:
        identity = p.identity.identity
        covered = scope_id in (identity.robot_id, identity.site_id)
        if covered and identity.day >= from_day:
            healthy = sf.TASK_BASE[p.task]
            affected.append(
                healthy - sf.success_probability(identity.robot_id, identity.day, p.task)
            )
    covered_share = sum(
        scope_id in (p.identity.identity.robot_id, p.identity.identity.site_id) for p in planned
    ) / len(planned)
    return 100 * float(np.mean(affected)) * covered_share


def candidates() -> list[Any]:
    c = from_logs.Candidate
    return [
        c("gripper", sf.GRIPPER_ROBOT, 0, 1.0),
        c("delay", sf.DELAY_SITE, sf.DELAY_DAY, 1.0),
        c("camera", sf.DRIFT_ROBOT, 0, 1.0),
        c("benign calibration", sf.BENIGN_ROBOT, 1, 1.0),
    ]


def test_gains_cover_the_planted_effects(reader: Any) -> None:
    rows = {r["label"]: r for r in from_logs.gains(reader, candidates(), project_to_day=7)}
    for label, scope_id, day in (
        ("gripper", sf.GRIPPER_ROBOT, 0),
        ("delay", sf.DELAY_SITE, sf.DELAY_DAY),
    ):
        row = rows[label]
        truth = true_gain_points(scope_id, day)
        assert row["gain_low"] <= truth <= row["gain_high"], (label, truth, row)
        assert row["gain_low"] > 0
    assert rows["delay"]["share"] == 0.5 and rows["gripper"]["share"] == pytest.approx(
        1 / 6, abs=1e-3
    )
    # The drift projects to a larger loss than it shows on average.
    assert rows["camera"]["slope_per_day"] > 0
    assert rows["camera"]["projected_gain_points"] > rows["camera"]["gain_points"]
    # The harmless change sits inside the harmful site: its failures are shared, and
    # against the other robots of its site it loses little.
    assert rows["benign calibration"]["shared_failures"] > 0
    assert rows["gripper"]["shared_failures"] == 0
    assert rows["benign calibration"]["within_site_gap"] < 0.15 < rows["gripper"]["within_site_gap"]
    assert rows["delay"]["within_site_gap"] is None


def test_gains_without_excluding_suspects_compare_with_everyone(reader: Any) -> None:
    clean = {r["label"]: r for r in from_logs.gains(reader, candidates())}
    plain = {r["label"]: r for r in from_logs.gains(reader, candidates(), exclude_suspects=False)}
    assert plain["delay"]["n_comparison"] < clean["delay"]["n_comparison"] or (
        plain["delay"]["expected_rate"] < clean["delay"]["expected_rate"]
    )


def reference_power(p0: float, p1: float, n: int) -> float:
    pbar = (p0 + p1) / 2
    x = (abs(p1 - p0) * math.sqrt(n) - 1.96 * math.sqrt(2 * pbar * (1 - pbar))) / math.sqrt(
        p0 * (1 - p0) + p1 * (1 - p1)
    )
    return float(norm.cdf(x))


@pytest.mark.parametrize(("p0", "p1"), [(0.6, 0.85), (0.4, 0.6), (0.85, 0.95), (0.3, 0.9)])
def test_power_matches_the_textbook(reader: Any, p0: float, p1: float) -> None:
    row = experiment.power(reader, p0, p1)
    pbar = (p0 + p1) / 2
    needed = (
        1.96 * math.sqrt(2 * pbar * (1 - pbar)) + 0.8416 * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    ) ** 2 / (p1 - p0) ** 2
    assert row["n_per_arm_needed"] == math.ceil(needed)
    assert row["power_at_max"] == pytest.approx(reference_power(p0, p1, 60), abs=0.005)
    detectable = row["smallest_detectable_at_max"]
    if p0 + detectable < 0.999:
        assert reference_power(p0, p0 + detectable, 60) >= 0.795
        assert reference_power(p0, p0 + detectable - 0.01, 60) < 0.805


def test_readout_compares_arms(reader: Any) -> None:
    ids = reader.query(
        "SELECT robot_id, list(episode_id ORDER BY episode_id) AS ids FROM episodes "
        "WHERE experiment_id IS NULL GROUP BY 1"
    ).to_pylist()
    episodes = {r["robot_id"]: r["ids"] for r in ids}
    rows = experiment.readout(reader, {0: episodes["r01"], 1: episodes[sf.GRIPPER_ROBOT]})
    control, treated = rows
    rates = by_level(triage.outcomes(reader), "robot")
    assert (
        control["rate"] == rates["r01"]["rate"]
        and treated["rate"] == rates[sf.GRIPPER_ROBOT]["rate"]
    )
    assert treated["diff"] == pytest.approx(treated["rate"] - control["rate"], abs=1e-3)
    assert treated["diff_low"] < treated["diff"] < treated["diff_high"] < 0
    assert control["diff"] == 0 and treated["z_statistic"] < -1.96
    shared = experiment.readout(
        reader, {0: episodes["r01"], 1: episodes[sf.GRIPPER_ROBOT]}, share=0.25
    )[1]
    assert shared["gain_points"] == pytest.approx(100 * treated["diff"] * 0.25, abs=0.01)
    assert shared["gain_low"] == pytest.approx(100 * treated["diff_low"] * 0.25, abs=0.05)
    assert shared["gain_high"] == pytest.approx(100 * treated["diff_high"] * 0.25, abs=0.05)
    with pytest.raises(ValueError, match="arm 0"):
        experiment.readout(reader, {1: episodes["r01"]})


def ranked(
    verb: Verb, scope_id: str, gain: float, low: float, high: float, **thing: Any
) -> RankedAction:
    value = thing.pop("value", None)
    return RankedAction(
        action=Action(verb=verb, target=Target(scope=scope_id, **thing), value=value),
        expected_gain=gain,
        uncertainty=Interval(low=low, high=high),
        rationale="test",
    )


def test_cut_takes_counts_and_hours_down_the_list(reader: Any) -> None:
    key = ConfigKey.control_action_delay_steps
    actions = [
        ranked(Verb.replace_, sf.GRIPPER_ROBOT, 9.0, 5.0, 11.0, component=Component.gripper),
        ranked(Verb.revert, sf.DELAY_SITE, 7.0, 5.0, 13.0, config_key=key),
        ranked(Verb.recalibrate, sf.DRIFT_ROBOT, 3.0, 1.0, 6.0, component=Component.scene_camera),
        ranked(Verb.service, sf.GRIPPER_ROBOT, 2.5, 0.0, 6.0, component=Component.gripper),
        ranked(Verb.set, sf.DELAY_SITE, 2.0, 0.0, 9.0, config_key=key, value=0),
        ranked(Verb.clean, sf.GRIPPER_ROBOT, 0.5, 0.0, 2.0, component=Component.gripper),
    ]
    rows = rank.cut(reader, actions)
    assert [r["rank"] for r in rows] == list(range(1, 7))
    assert [r["hours"] for r in rows] == [8.0, 0.1, 0.5, 3.0, 0.1, 0.25]
    # A round adopts the top `adopt` actions, three by default; hours are information only.
    assert [r["adopted"] for r in rows] == [True, True, True, False, False, False]
    assert [r["cum_hours"] for r in rows] == [8.0, 8.1, 8.6, 11.6, 11.7, 11.95]
    assert rows[4]["value"] == "0"
    assert all(r["checks"] == "" for r in rows[:3])  # no list length is required any more
    assert [r["adopted"] for r in rank.cut(reader, actions, adopt=1)] == [True] + [False] * 5
    # None is a decision too.
    assert not any(r["adopted"] for r in rank.cut(reader, actions, adopt=0))
    four = rank.cut(reader, actions, adopt=4)
    assert [r["adopted"] for r in four] == [True] * 4 + [False] * 2
    assert "more changes adopted than a round allows" in four[3]["checks"]
    assert "more changes adopted" not in four[2]["checks"]
    # Adopting an action whose interval reaches zero asks for a reason.
    assert "say why in its rationale" in four[3]["checks"]


def test_cut_reports_what_is_wrong_with_a_list(reader: Any) -> None:
    good = ranked(Verb.recalibrate, sf.DRIFT_ROBOT, 3.0, 1.0, 6.0, component=Component.scene_camera)
    outside = ranked(Verb.clean, sf.DRIFT_ROBOT, 4.0, 0.0, 2.0, component=Component.scene_camera)
    rows = rank.cut(reader, [good, *[good] * 9, outside])
    assert "listed twice" not in rows[0]["checks"]  # the first time it is fine
    assert all("the same action is listed twice" in r["checks"] for r in rows[1:10])
    last = rows[-1]["checks"]
    assert (
        "expected gain outside its interval" in last and "larger gain than the action above" in last
    )
    rows = rank.run(reader, "cut", None, [(1, "fly", "fleet", None, None, 1.0, 0.0, 2.0)])
    assert "not in the catalogue" in rows[0]["checks"]
    at_a_site = [(1, "recalibrate", sf.DELAY_SITE, "scene_camera", None, 1.0, 0.0, 2.0)]
    assert "not in the catalogue" in rank.run(reader, "cut", None, at_a_site)[0]["checks"]
    no_class = [(1, "route_to_teleop", "fleet", None, "juggle", 0.0, 0.0, 0.0)]
    assert "not in the catalogue" in rank.run(reader, "cut", None, no_class)[0]["checks"]
    clean_arm = [(1, "clean", sf.GRIPPER_ROBOT, "arm", None, 1.0, 0.0, 2.0)]
    assert "not in the catalogue" in rank.run(reader, "cut", None, clean_arm)[0]["checks"]
    rows = rank.run(
        reader, "cut", None, [(2, "revert", "fleet", "software.version", None, 1.0, 0.0, 2.0)]
    )
    assert "ranks must run" in rows[0]["checks"]


def test_the_command_line_prints_rows(
    skill_exports: Exports, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        triage.main([str(skill_exports.fleet_export), "outcomes", "--params", '{"from_day": 1}'])
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    first = json.loads(lines[0])
    assert first["dimension"] == "fleet" and first["n"] == (sf.DAYS - 1) * sf.EPISODES_PER_DAY * 6
    assert triage.main([str(skill_exports.fleet_export), "sample", "--sql"]) == 0
    assert capsys.readouterr().out.startswith("-- helper: sample")


def test_cut_takes_values_as_text_and_leaves_training_data_uncosted(reader: Any) -> None:
    version = ConfigKey.software_version
    actions = [
        ranked(Verb.set, sf.DELAY_SITE, 3.0, 1.0, 5.0, config_key=version, value="1.4.0"),
        ranked(
            Verb.set,
            sf.DELAY_SITE,
            2.0,
            0.0,
            4.0,
            config_key=ConfigKey.control_action_delay_steps,
            value=0,
        ),
        ranked(Verb.route_to_teleop, "fleet", 0.0, 0.0, 0.0, value="stack"),
        ranked(Verb.request_training_data, "fleet", 0.0, 0.0, 0.0, value="drops in the carry"),
    ]
    rows = rank.cut(reader, actions)
    assert [r["value"] for r in rows] == ["1.4.0", "0", "stack", "drops in the carry"]
    assert [r["thing"] for r in rows[2:]] == [None, None]
    assert rows[2]["hours"] == 0.0 and rows[3]["hours"] is None
    assert rows[3]["cum_hours"] == rows[2]["cum_hours"] == 0.2
    assert "not costed" in rows[3]["checks"] and "not in the catalogue" not in rows[3]["checks"]


def test_cut_adopts_the_first_distinct_actions(reader: Any) -> None:
    key = ConfigKey.control_action_delay_steps
    replace = ranked(Verb.replace_, sf.GRIPPER_ROBOT, 9.0, 5.0, 11.0, component=Component.gripper)
    actions = [
        replace,
        ranked(Verb.revert, sf.DELAY_SITE, 7.0, 5.0, 13.0, config_key=key),
        replace,  # a repeat is skipped, and counts toward nothing
        ranked(Verb.service, sf.GRIPPER_ROBOT, 3.0, 0.0, 6.0, component=Component.gripper),
        ranked(Verb.replace_, sf.DRIFT_ROBOT, 2.5, 0.0, 6.0, component=Component.arm),
        ranked(Verb.request_training_data, "fleet", 0.0, 0.0, 0.0, value="drops"),
    ]
    rows = rank.cut(reader, actions)
    assert [r["adopted"] for r in rows] == [True, True, False, True, False, False]
    assert "listed twice" in rows[2]["checks"]
    # What has no simulated effect cannot be adopted.
    rows = rank.cut(reader, [actions[0], actions[5]], adopt=2)
    assert "no simulated effect: cannot be adopted" in rows[1]["checks"]
    assert "cannot be adopted" not in rank.cut(reader, [actions[0], actions[5]])[0]["checks"]


def test_set_gain_combines_parts_without_adding_single_gains(reader: Any) -> None:
    # Two changes on one site (a part), one change and an undo on a robot elsewhere.
    rows = rank.set_gain(
        reader,
        [
            ("site", "adopt", "first", 0.25, 0.55, 6.0, 2.0, 9.0),
            ("site", "adopt", "second", 0.25, 0.55, 4.0, 0.5, 7.0),
            ("robot", "replace", "L1 by another value", 0.1, 0.4, 3.0, 1.0, 4.5),
        ],
    )
    by_part = {r["part"]: r for r in rows}
    assert [r["part"] for r in rows] == ["robot", "site", "set"]
    site, robot, whole = by_part["site"], by_part["robot"], by_part["set"]
    ceiling = 100 * 0.25 * 0.45
    assert site["ceiling"] == pytest.approx(ceiling, abs=0.01)
    # Independent removal: more than the best single, less than the sum.
    combined = ceiling * (1 - (1 - 6.0 / ceiling) * (1 - 4.0 / ceiling))
    assert site["gain"] == pytest.approx(combined, abs=0.01)
    assert 6.0 < site["gain"] < 10.0
    assert site["low"] == 2.0 and site["high"] == pytest.approx(ceiling, abs=0.01)
    assert robot["gain"] == 3.0 and robot["low"] == 1.0 and robot["high"] == 4.5
    assert robot["how"] == "one change" and robot["checks"] == site["checks"] == ""
    # Across parts the gains add and the half-widths combine as independent errors.
    mid = combined + 3.0
    assert whole["gain"] == pytest.approx(mid, abs=0.01)
    low = mid - math.sqrt((combined - 2.0) ** 2 + 2.0**2)
    high = mid + math.sqrt((ceiling - combined) ** 2 + 1.5**2)
    assert low - 0.01 <= whole["low"] <= low and high <= whole["high"] <= high + 0.01
    assert whole["low"] > 2.0 + 1.0 and whole["high"] < ceiling + 4.5  # narrower than the ends
    assert whole["changes_in_part"] == 3
    # What the block reports: a gain above what the part loses, an unknown kind.
    bad = rank.set_gain(reader, [("robot", "keep", "x", 0.1, 0.9, 2.0, 1.0, 3.0)])
    assert "above what the part loses" in bad[0]["checks"] and "kind must be" in bad[0]["checks"]
    assert bad[0]["gain"] == pytest.approx(1.0, abs=0.01)  # held to the part's ceiling
    assert bad[1]["checks"] == "see the parts"


def test_set_gain_lets_possible_harms_add_up(reader: Any) -> None:
    # The first case (review of 63736f8): two changes on the same robots, each with a
    # positive point gain but an interval reaching below zero. They may harm different
    # successful attempts, so together they may cost both low ends.
    rows = rank.set_gain(
        reader,
        [
            ("site", "adopt", "first", 1.0, 0.8, 4.0, -5.0, 8.0),
            ("site", "adopt", "second", 1.0, 0.8, 4.0, -5.0, 8.0),
        ],
    )
    part, whole = rows
    assert part["gain"] == pytest.approx(20 * (1 - 0.8 * 0.8), abs=0.01)  # 7.2
    assert part["low"] == -10.0 and part["high"] == 16.0
    assert "their harms can add up" in part["checks"]
    assert (whole["low"], whole["gain"], whole["high"]) == (-10.0, part["gain"], 16.0)
    assert "the set may cost points" in whole["checks"]
    # The second case: two undos that each cost points. The harms add at the low end and
    # may be the same attempts at the high end; the range never reaches zero.
    rows = rank.set_gain(
        reader,
        [
            ("site", "undo", "L1", 1.0, 0.8, -2.0, -4.0, -1.0),
            ("site", "undo", "L2", 1.0, 0.8, -2.0, -4.0, -1.0),
        ],
    )
    part, whole = rows
    assert (part["low"], part["gain"], part["high"]) == (-8.0, -4.0, -1.0)
    assert "a change expected to cost points" in part["checks"]
    assert (whole["low"], whole["gain"], whole["high"]) == (-8.0, -4.0, -1.0)
    assert "the set may cost points" in whole["checks"]
    # The harms cannot exceed the part's successes: 100 x 0.1 x 0.2 = 2 points.
    rows = rank.set_gain(
        reader,
        [
            ("robot", "adopt", "a", 0.1, 0.2, 1.0, -5.0, 3.0),
            ("robot", "adopt", "b", 0.1, 0.2, 1.0, -5.0, 3.0),
        ],
    )
    assert rows[0]["low"] == -2.0
    # The third case (review of 31becba): the point estimate and the high end obey the same
    # cap. With share 1 and rate 0.1 only 10 points of success can be lost.
    rows = rank.set_gain(
        reader,
        [
            ("site", "adopt", "a", 1.0, 0.1, -8.0, -9.0, -7.0),
            ("site", "adopt", "b", 1.0, 0.1, -8.0, -9.0, -7.0),
        ],
    )
    part, whole = rows
    assert (part["low"], part["gain"], part["high"]) == (-10.0, -10.0, -7.0)
    assert "harms beyond what the part can lose" in part["checks"]
    assert (whole["low"], whole["gain"], whole["high"]) == (-10.0, -10.0, -7.0)
    # A single change keeps its own interval; it is warned about when it reaches below zero.
    single = rank.set_gain(reader, [("robot", "adopt", "a", 0.5, 0.6, 3.0, -1.0, 6.0)])[0]
    assert (single["low"], single["gain"], single["high"]) == (-1.0, 3.0, 6.0)
    assert "its interval reaches below zero" in single["checks"]


@pytest.mark.parametrize(("arms", "left"), [(2, 480), (4, 200)])
def test_power_states_the_cost_of_every_arm(reader: Any, arms: int, left: int) -> None:
    row = experiment.power(reader, 0.6, 0.85, arms=arms, attempts_left=left, max_per_arm=40)
    assert row["arms"] == arms
    assert row["attempts_needed"] == arms * row["n_per_arm_needed"]
    assert row["attempts_at_max"] == arms * 40
    assert row["attempts_left_after"] == left - arms * 40


def test_arm_episodes_stay_out_and_read_out_by_arm(reader: Any) -> None:
    total = reader.query("SELECT count(*) AS n FROM episodes").to_pylist()[0]["n"]
    fleet = by_level(triage.outcomes(reader), "fleet")["fleet"]
    assert total == fleet["n"] + 2 * sf.ARM_EPISODES
    assert {r["n"] for r in scope.breakdown(reader) if r["dimension"] == "fleet"} == {fleet["n"]}
    arms = sf.arm_episode_ids()
    share = round(1 / 6, 3)
    rows = experiment.readout(reader, arms, share=share)
    assert [r["n"] for r in rows] == [sf.ARM_EPISODES, sf.ARM_EPISODES]
    assert rows[1]["gain_points"] == pytest.approx(100 * rows[1]["diff"] * share, abs=0.01)
    marked = reader.query(
        "SELECT DISTINCT experiment_id FROM episodes WHERE experiment_id IS NOT NULL"
    ).to_pylist()
    assert marked == [{"experiment_id": sf.EXPERIMENT_ID}]


def test_on_real_episodes_every_success_reads_as_held(mini_export: Any) -> None:
    # The settled finger opening keeps every held object, bowl rims included,
    # above `empty_below`, and still finds grasps that closed on nothing.
    with LogReader(mini_export, materialize=True, lock=True) as real:
        rows = cards.profile(real, real.episode_ids())
        signatures = {r["signature"]: r["failures"] for r in modes.signatures(real)}
    empty = float(cards.blocks()["profile"].split(" AS empty_below")[0].rsplit(" ", 1)[1])
    successes = [r for r in rows if r["outcome"] == "success"]
    assert successes and all(r["held_ratio"] >= empty for r in successes)
    assert all(r["end_phase"] == "completed" for r in successes)
    phases = {r["end_phase"] for r in rows if r["outcome"] == "failure"}
    assert "grasp: closed on nothing" in phases
    assert signatures.get("closed on nothing", 0) >= 1
    nothing = [r for r in rows if r["end_phase"] == "grasp: closed on nothing"]
    assert all(r["held_ratio"] < empty for r in nothing)
