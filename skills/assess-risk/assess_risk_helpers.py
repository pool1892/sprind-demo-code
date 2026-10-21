"""Telemetry signs of a running attempt, for the supervisor's harness.

The supervisor (a vision-language model) reads frames well and long tables
of numbers badly. A harness that builds its context can call `signs` on the
telemetry window and put the result beside the frames, as a second witness
to what the frames show. Only the visible telemetry fields are used, the
same as the `steps` table: `step`, `gripper_qpos`, `joint_vel`, `eef_pos`.

The thresholds are shares of the window's own range or general physical
scales, never values of a particular fleet: the finger opening is judged
against the widest opening in the window (or `open_width` when the harness
knows it), stillness against `still_speed` in radians per second.

    from assess_risk_helpers import signs
    signs(context.telemetry_window)   # StepTelemetry objects or dictionaries
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

CLOSED_BELOW = 0.85  # fingers closed: opening below this share of the open width
# Fingers met, nothing between them. Measured on 416 recorded X-VLA episodes (skills/README.md):
# with nothing between them the fingers settle near 0.013 of the open width, a bowl held by its
# rim at 0.04 or more, other objects at 0.4 or more.
EMPTY_BELOW = 0.025
SETTLE_STEPS = 25  # the fingers creep on this long after closing when nothing is between them
STILL_SPEED = 0.05  # joint speed norm in rad/s below which the arm is still
REVERSAL_SHARE = 0.2  # a direction change counts above this share of the peak velocity


def _field(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def _vector(row: Any, name: str) -> list[float]:
    return [float(v) for v in _field(row, name)]


def _grasp_start(closed: Sequence[bool], settle_steps: int) -> int | None:
    """The first index of the first closed stretch of `settle_steps`, else of any closed step."""
    first = None
    run_start = None
    for i, is_closed in enumerate(closed):
        if is_closed:
            if run_start is None:
                run_start = i
                first = i if first is None else first
            if i - run_start + 1 >= settle_steps:
                return run_start
        else:
            run_start = None
    return first


def signs(
    window: Sequence[Any],
    *,
    open_width: float | None = None,
    still_speed: float = STILL_SPEED,
    settle_steps: int = SETTLE_STEPS,
) -> dict[str, Any]:
    """What the telemetry window says about the attempt.

    Returns the steps covered, whether the gripper is closed now and holds
    something, the settled finger opening (None while the fingers are still
    settling), whether it closed on nothing, the step at which a held object
    was lost, the longest run of still steps, the number of direction
    reversals of the end effector, and the height gained since the gripper
    closed, with a phase hint from the gripper's history. Whether anything is
    held is judged only once the fingers have settled (`settle_steps` after
    closing): before that a closing on nothing looks like a grip.
    """
    if not window:
        raise ValueError("an empty telemetry window says nothing")
    steps = [int(_field(row, "step")) for row in window]
    widths = [sum(abs(q) for q in _vector(row, "gripper_qpos")) for row in window]
    reference = open_width if open_width is not None else max(widths)
    opening = [w / reference if reference > 0 else 0.0 for w in widths]
    speeds = [math.sqrt(sum(v * v for v in _vector(row, "joint_vel"))) for row in window]
    heights = [_vector(row, "eef_pos")[2] for row in window]

    closed = [o < CLOSED_BELOW for o in opening]
    close_index = _grasp_start(closed, settle_steps)
    settled = lost_at = None
    if close_index is not None:
        start = close_index + settle_steps
        reading = [o for o in opening[start : start + 11] if o < CLOSED_BELOW]
        settled = statistics.median(reading) if reading else None
        reopen = next((i for i in range(start, len(opening)) if not closed[i]), len(opening))
        if settled is not None and settled >= EMPTY_BELOW:
            lost_at = next(
                (steps[i] for i in range(start + 11, reopen) if opening[i] < EMPTY_BELOW), None
            )
    closed_now = closed[-1]
    on_nothing = settled is not None and settled < EMPTY_BELOW

    longest = run = 0
    for speed in speeds:
        run = run + 1 if speed < still_speed else 0
        longest = max(longest, run)

    reversals = 0
    xs = [_vector(row, "eef_pos")[0] for row in window]
    deltas = [b - a for a, b in pairwise(xs)]
    peak = max((abs(d) for d in deltas), default=0.0)
    previous = 0.0
    for delta in deltas:
        if abs(delta) > REVERSAL_SHARE * peak > 0:
            if previous and math.copysign(1, delta) != math.copysign(1, previous):
                reversals += 1
            previous = delta

    lift = None if close_index is None else max(heights[close_index:]) - heights[close_index]
    if lost_at is not None:
        phase = "carry: lost the object"
    elif on_nothing:
        phase = "grasp: closed on nothing"
    elif closed_now and settled is None:
        phase = "grasp: closing, not settled yet"
    elif closed_now:
        phase = "lift or carry: holding"
    elif close_index is not None:
        phase = "place or retreat: opened again"
    else:
        phase = "approach: gripper open"
    return {
        "from_step": steps[0],
        "to_step": steps[-1],
        "phase_hint": phase,
        "closed_now": closed_now,
        "holding_now": closed_now
        and settled is not None
        and settled >= EMPTY_BELOW
        and lost_at is None
        and opening[-1] >= EMPTY_BELOW,
        "settled_opening": None if settled is None else round(settled, 4),
        "closed_on_nothing": on_nothing,
        "lost_object_at": lost_at,
        "longest_still_run": longest,
        "reversals": reversals,
        "lift_m": None if lift is None else round(lift, 4),
    }
