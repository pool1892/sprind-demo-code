"""Blind selection of A/B requests within the public robot/day arm capacity.

The caller supplies candidates in priority order and visible robot/site rows.
Both arms contribute one episode per slot. Robots must also be sorted in the
actual requests: this helper uses that order to predict their allocation.
This is a fresh queue's allocation, without earlier experiment reservations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fleetlab.baselines.classical.protocol import EXPERIMENT_LENGTH
from fleetlab.contracts.actions import Action
from fleetlab.contracts.identity import ARM_SLOTS
from fleetlab.contracts.ids import IdKind, check_neutral, scope_covers


def select_feasible(
    candidates: Sequence[Action],
    robot_rows: Sequence[Mapping[str, str]],
    *,
    episodes_per_arm: int = 60,
    days: int = EXPERIMENT_LENGTH,
    request_limit: int = 8,
) -> tuple[list[Action], dict[str, Any]]:
    """Keep the first feasible distinct actions; never reduce their requested n.

    Rows name ``robot_id`` and ``site_id``. Repeated identical rows are allowed
    so callers may use visible episode rows; conflicting sites are refused.
    The report is JSON-compatible and exposes selected allocations, all
    skipped candidates, final usage, and any shortfall from the request limit.
    Explicit n/day/limit overrides are honored; the caller enforces any
    experiment-design budget in addition to this physical capacity check.
    """
    for name, value, minimum in (
        ("episodes_per_arm", episodes_per_arm, 1),
        ("days", days, 1),
        ("request_limit", request_limit, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    robots: dict[str, str] = {}
    for row in robot_rows:
        robot = check_neutral(IdKind.robot, row["robot_id"])
        site = check_neutral(IdKind.site, row["site_id"])
        if robot in robots and robots[robot] != site:
            raise ValueError(f"robot {robot} has conflicting site rows")
        robots[robot] = site
    usage = {(robot, day): 0 for robot in sorted(robots) for day in range(days)}
    chosen: list[Action] = []
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, action in enumerate(candidates):
        record: dict[str, Any] = {
            "candidate_index": index,
            "action": action.model_dump(mode="json"),
        }
        key = action.model_dump_json()
        if key in seen:
            rejected.append(dict(record, reason="duplicate"))
            continue
        seen.add(key)
        if len(chosen) >= request_limit:
            rejected.append(dict(record, reason="request_limit"))
            continue
        covered = sorted(
            robot
            for robot, site in robots.items()
            if scope_covers(action.target.scope, robot, site)
        )
        if not covered:
            rejected.append(dict(record, reason="no_covered_robots"))
            continue
        demand = {(robot, day): 0 for robot in covered for day in range(days)}
        for slot in range(episodes_per_arm):
            robot = covered[slot % len(covered)]
            day = (slot // len(covered)) % days
            demand[robot, day] += 2
        overloaded = [
            {
                "robot_id": robot,
                "day_index": day,
                "used": usage[robot, day],
                "requested": count,
                "projected": usage[robot, day] + count,
            }
            for (robot, day), count in demand.items()
            if usage[robot, day] + count > ARM_SLOTS
        ]
        if overloaded:
            rejected.append(dict(record, reason="capacity", overloaded=overloaded))
            continue
        for cell, count in demand.items():
            usage[cell] += count
        chosen.append(action)
        selected.append(
            dict(
                record,
                robots=covered,
                allocation=[
                    {"robot_id": robot, "day_index": day, "episodes": count}
                    for (robot, day), count in demand.items()
                ],
            )
        )
    return chosen, {
        "episodes_per_arm": episodes_per_arm,
        "days": days,
        "arms": 2,
        "capacity_per_robot_day": ARM_SLOTS,
        "request_limit": request_limit,
        "selected": selected,
        "rejected": rejected,
        "shortfall": request_limit - len(chosen),
        "total_arm_episodes": 2 * episodes_per_arm * len(chosen),
        "usage": [
            {"robot_id": robot, "day_index": day, "episodes": count, "remaining": ARM_SLOTS - count}
            for (robot, day), count in usage.items()
        ],
    }
