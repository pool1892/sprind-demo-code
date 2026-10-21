"""Public capacity arithmetic on synthetic rows and actions; no engine imports."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from fleetlab.baselines.classical.capacity import select_feasible
from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.identity import ARM_SLOTS
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb


def robots(n: int = 8) -> list[dict[str, str]]:
    return [
        {"robot_id": f"r{i:02d}", "site_id": "s1" if i <= n // 2 else "s2"} for i in range(1, n + 1)
    ]


def setting(scope: str, value: int = 0) -> Action:
    return Action(
        verb=Verb.set,
        target=Target(scope=scope, config_key=ConfigKey.control_action_delay_steps),
        value=value,
    )


def repair(robot: str, verb: Verb = Verb.recalibrate) -> Action:
    return Action(verb=verb, target=Target(scope=robot, component=Component.scene_camera))


def usage(report: dict[str, Any]) -> dict[tuple[str, int], int]:
    return {(row["robot_id"], row["day_index"]): row["episodes"] for row in report["usage"]}


def test_reserved_fleet_and_four_site_conditions_skip_second_same_robot_and_fill_eight() -> None:
    fleet = setting("fleet")
    conditions = [
        Action(
            verb=Verb.change_condition,
            target=Target(scope="s1", condition_key=key),
            value="nominal",
        )
        for key in ConditionKey
    ]
    first = repair("r01")
    skipped = repair("r01", Verb.clean)
    other = [repair("r02"), repair("r05")]
    candidates = [fleet, *conditions, first, skipped, *other]
    chosen, report = select_feasible(candidates, list(reversed(robots())))
    assert chosen == [fleet, *conditions, first, *other]
    assert len({a.model_dump_json() for a in chosen}) == 8
    assert report["episodes_per_arm"] == 60 and report["days"] == 2
    assert report["total_arm_episodes"] == 960 and report["shortfall"] == 0
    assert [(r["candidate_index"], r["reason"]) for r in report["rejected"]] == [(6, "capacity")]
    assert report["rejected"][0]["overloaded"] == [
        {"robot_id": "r01", "day_index": 0, "used": 132, "requested": 60, "projected": 192},
        {"robot_id": "r01", "day_index": 1, "used": 124, "requested": 60, "projected": 184},
    ]
    expected = {
        ("r01", 0): 132,
        ("r01", 1): 124,
        ("r02", 0): 132,
        ("r02", 1): 124,
        ("r03", 0): 72,
        ("r03", 1): 64,
        ("r04", 0): 72,
        ("r04", 1): 64,
        ("r05", 0): 68,
        ("r05", 1): 66,
        ("r06", 0): 8,
        ("r06", 1): 6,
        ("r07", 0): 8,
        ("r07", 1): 6,
        ("r08", 0): 8,
        ("r08", 1): 6,
    }
    assert usage(report) == expected
    assert all(count <= ARM_SLOTS for count in usage(report).values())
    assert json.loads(json.dumps(report)) == report


def test_fleet_site_and_robot_scopes_are_accumulated_exactly() -> None:
    candidates = [setting("fleet"), setting("s1"), repair("r02")]
    chosen, report = select_feasible(candidates, robots(2), episodes_per_arm=5)
    assert chosen == candidates
    assert usage(report) == {("r01", 0): 10, ("r01", 1): 6, ("r02", 0): 8, ("r02", 1): 6}
    assert [r["robots"] for r in report["selected"]] == [["r01", "r02"], ["r01"], ["r02"]]


@pytest.mark.parametrize("n, days", [(1, 1), (5, 2), (60, 2), (17, 3)])
def test_custom_counts_match_independent_per_arm_enumeration(n: int, days: int) -> None:
    rows = [{"robot_id": robot, "site_id": "s1"} for robot in ("r03", "r01", "r02")]
    chosen, report = select_feasible([setting("s1")], rows, episodes_per_arm=n, days=days)
    assert len(chosen) == 1
    expected: Counter[tuple[str, int]] = Counter()
    ordered = sorted(row["robot_id"] for row in rows)
    for _arm in range(2):
        for slot in range(n):
            expected[ordered[slot % 3], (slot // 3) % days] += 1
    assert usage(report) == {(r, d): expected[r, d] for r in ordered for d in range(days)}
    assert sum(usage(report).values()) == 2 * n


def test_exact_capacity_and_impossible_single_request_never_reduce_n() -> None:
    rows = [{"robot_id": "r01", "site_id": "s1"}]
    chosen, report = select_feasible([repair("r01")], rows, episodes_per_arm=80, days=1)
    assert len(chosen) == 1 and usage(report) == {("r01", 0): ARM_SLOTS}
    chosen, report = select_feasible([repair("r01")], rows, episodes_per_arm=81, days=1)
    assert chosen == [] and usage(report) == {("r01", 0): 0}
    assert report["episodes_per_arm"] == 81
    assert report["rejected"][0]["overloaded"] == [
        {"robot_id": "r01", "day_index": 0, "used": 0, "requested": 162, "projected": 162}
    ]


def test_failed_request_does_not_consume_capacity_before_later_feasible_scope() -> None:
    candidates = [
        repair("r01"),
        repair("r01", Verb.clean),
        repair("r01", Verb.replace_),
        repair("r05"),
    ]
    chosen, report = select_feasible(candidates, robots())
    assert chosen == [*candidates[:2], candidates[3]]
    assert usage(report)["r01", 0] == usage(report)["r01", 1] == 120
    assert usage(report)["r05", 0] == usage(report)["r05", 1] == 60
    assert report["shortfall"] == 5


def test_duplicates_and_repeated_visible_rows_do_not_consume_capacity() -> None:
    first, second = repair("r01"), repair("r01", Verb.clean)
    rows = robots()
    chosen, report = select_feasible([first, first, second], [*rows, *rows])
    assert chosen == [first, second]
    assert report["rejected"][0]["reason"] == "duplicate"
    assert usage(report)["r01", 0] == usage(report)["r01", 1] == 120


def test_selection_and_report_are_stable_under_visible_row_permutation() -> None:
    candidates = [setting("fleet"), setting("s1"), repair("r01"), repair("r02")]
    expected = select_feasible(candidates, robots(), episodes_per_arm=15)
    assert select_feasible(candidates, list(reversed(robots())), episodes_per_arm=15) == expected
    assert expected[0] == candidates


def test_request_limit_stops_selection_and_reports_remaining_candidates() -> None:
    candidates = [repair(f"r{i:02d}") for i in range(1, 9)]
    chosen, report = select_feasible(candidates, robots(), request_limit=3)
    assert chosen == candidates[:3] and report["total_arm_episodes"] == 360
    assert [r["reason"] for r in report["rejected"]] == ["request_limit"] * 5
    chosen, report = select_feasible(candidates, robots(), request_limit=0)
    assert chosen == [] and sum(usage(report).values()) == 0


def test_missing_scope_is_rejected_without_capacity_charge() -> None:
    candidates = [repair("r99"), setting("s9"), repair("r01")]
    chosen, report = select_feasible(candidates, robots())
    assert chosen == candidates[2:]
    assert [r["reason"] for r in report["rejected"]] == ["no_covered_robots"] * 2


def test_conflicting_robot_site_rows_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="conflicting site"):
        select_feasible(
            [repair("r01")],
            [
                {"robot_id": "r01", "site_id": "s1"},
                {"robot_id": "r01", "site_id": "s2"},
            ],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes_per_arm": 0},
        {"episodes_per_arm": True},
        {"days": 0},
        {"days": -1},
        {"request_limit": -1},
        {"request_limit": 1.5},
    ],
)
def test_invalid_count_arguments_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        select_feasible([repair("r01")], robots(), **kwargs)
