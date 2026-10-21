"""The fleet's success check, version 1: placed (contracts, section 5).

An episode succeeds at the first control step at which the object is placed:
LIBERO's own check holds, no body of the robot (arm links, gripper parts)
touches the object, and the object is at rest, slower than
`PLACED_THRESHOLDS["linear_m_s"]` and `["angular_rad_s"]`. It is a function of
one simulator state, so an original and its replay get the same verdict bit
for bit. The verdict is done when placed holds ("placed"), or `tail_steps`
after LIBERO's check first held without placed ("LIBERO only"); the rollout
ends the episode at the step limit itself ("step limit"). The check's state is
LIBERO's first-fire step or none, carried through a snapshot.

Everything comes from `contracts.env.PrivilegedState` and the task's
`TaskGoal`: `libero_success`, the contacts by body name (the adapter names an
object's bodies by the object's name, as the goal does) and the object's
free-joint velocities. The environment hands the check the task's goal at
every reset (`SuccessCheck.reset(goal)`).

Ported from spikes/success/defs.py (definition d, there with the object also
touching its target, which the review dropped); the spike's tail: placed held
a median of 3 steps after LIBERO's check for the expert on the bowl and 15 on
the basket (at most 12 and 44), 6 and 8 for X-VLA with a scripted release (at
most 25).
"""

from __future__ import annotations

import json

import numpy as np

from fleetlab.contracts.env import (
    PLACED_THRESHOLDS,
    REASON_LIBERO_ONLY,
    REASON_PLACED,
    PrivilegedState,
    SuccessVerdict,
    TaskGoal,
)

ROBOT_PREFIXES = ("robot0_", "gripper0_")  # robosuite's names of the arm's and gripper's bodies


def is_robot(body: str) -> bool:
    return body.startswith(ROBOT_PREFIXES)


def robot_touches(state: PrivilegedState, body: str) -> bool:
    """Whether any body of the robot is in contact with `body`."""
    return any(
        (a == body and is_robot(b)) or (b == body and is_robot(a)) for a, b in state.contacts
    )


def at_rest(state: PrivilegedState, body: str) -> bool:
    velocity = state.object_velocities.get(body)
    if velocity is None:
        return False
    return bool(
        np.linalg.norm(velocity[:3]) < PLACED_THRESHOLDS["linear_m_s"]
        and np.linalg.norm(velocity[3:]) < PLACED_THRESHOLDS["angular_rad_s"]
    )


class FleetCheck:
    """Placed: LIBERO's check, the robot clear of the object, the object at rest."""

    needs_contacts_only_when_libero = True  # the adapter may skip contacts and velocities before
    reports_placed = True  # its success step is the step the object was found placed

    def __init__(self, goal: TaskGoal | None = None, tail_steps: int | None = None) -> None:
        self.goal = goal
        self.tail_steps = int(PLACED_THRESHOLDS["tail_steps"] if tail_steps is None else tail_steps)
        self.libero_first: int | None = None

    def reset(self, goal: TaskGoal) -> None:
        self.goal = goal
        self.libero_first = None

    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict:
        if self.goal is None:
            raise RuntimeError("the placed check needs the task's goal (reset(goal))")
        if state.libero_success:
            if self.libero_first is None:
                self.libero_first = step
            body = self.goal.object_body
            if not robot_touches(state, body) and at_rest(state, body):
                return SuccessVerdict(True, True, REASON_PLACED)
        if self.libero_first is None:
            return SuccessVerdict(False, False, "LIBERO's check has not held")
        if step - self.libero_first >= self.tail_steps:
            return SuccessVerdict(False, True, REASON_LIBERO_ONLY)
        return SuccessVerdict(
            False, False, f"LIBERO's check first held at step {self.libero_first}; not placed"
        )

    def snapshot(self) -> bytes:
        return json.dumps(self.libero_first).encode()

    def restore(self, blob: bytes) -> None:
        self.libero_first = json.loads(blob)
