"""The host-fixed delivery check of a job, and LIBERO's goal predicates on the loaded scene.

Delivered: every requested goal holds at the end, by LIBERO's own
predicate functions on the loaded scene (`On` requires contact, so poses alone do not
decide it); no robot body touches an object the goals name; those of them that move freely
are at rest; the gripper is open. It is evaluated from privileged state at the end of a run,
and its numbers are `contracts.jobs.DELIVERY`, in one place:

- at rest: slower than `DELIVERY.linear_m_s` (1 cm/s) and `DELIVERY.angular_rad_s`
  (0.15 rad/s), the success spike's thresholds, on the object's free-joint velocity;
- open: the last gripper command was open (negative, or none yet) and the fingers are at
  least `DELIVERY.gripper_open_m` (6 cm) apart.

The objects a goal names are its arguments' owners: an object or a fixture by its own
name, a region (a site such as `flat_stove_1_cook_region`) by the object that owns it, and
a region of the table by nothing, since everything rests on the table. A fixture such as
the stove has no free joint and is checked for contact only.

`delivery_verdict` and the helpers are pure functions of a `DeliveryState`, so the tests
construct states; `SceneGoals` reads a loaded LIBERO scene and imports LIBERO when built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.jobs import DELIVERY, DeliveryVerdict, RequestedGoal
from fleetlab.sim.libero_success import is_robot

__all__ = [
    "TABLE",
    "DeliveryState",
    "SceneGoals",
    "at_rest",
    "delivery_verdict",
    "gripper_state",
    "named_objects",
    "robot_touching",
    "scene_owner",
    "speeds",
]

TABLE = "main_table"


def robot_touching(contacts: Sequence[tuple[str, str]], body: str) -> bool:
    """Whether a body of the robot (arm links, gripper parts) is in contact with `body`."""
    return any((a == body and is_robot(b)) or (b == body and is_robot(a)) for a, b in contacts)


def speeds(velocity: npt.ArrayLike) -> tuple[float, float]:
    """Linear (m/s) and angular (rad/s) speed of a free joint's six velocity values."""
    v = np.asarray(velocity, np.float64).reshape(-1)
    return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:6]))


def at_rest(velocity: npt.ArrayLike) -> bool:
    linear, angular = speeds(velocity)
    return linear < DELIVERY.linear_m_s and angular < DELIVERY.angular_rad_s


def gripper_state(opening: float, command: float | None) -> tuple[bool, bool]:
    """(commanded open, open): the last command was open and the fingers are apart."""
    commanded_open = command is None or command < 0
    return commanded_open, bool(commanded_open and opening >= DELIVERY.gripper_open_m)


def named_objects(
    goals: Sequence[RequestedGoal], owner: Callable[[str], str | None]
) -> tuple[str, ...]:
    """The bodies the goals name, in order of first mention, the table never."""
    out: list[str] = []
    for goal in goals:
        for name in goal.objects:
            body = owner(name)
            if body is not None and body != TABLE and body not in out:
                out.append(body)
    return tuple(out)


@dataclass(frozen=True)
class DeliveryState:
    """What the delivery check reads at one instant: built from privileged state in a run,
    by hand in the tests."""

    goals_hold: tuple[bool, ...]  # per requested goal, LIBERO's predicate at this instant
    contacts: tuple[tuple[str, str], ...]  # body pairs in contact; objects by object name
    velocities: Mapping[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    gripper_opening: float = 0.0  # metres between the fingers
    gripper_command: float | None = None  # the last command (-1 open, 1 closed), or none


def delivery_verdict(
    state: DeliveryState, goals: Sequence[RequestedGoal], named: Sequence[str]
) -> DeliveryVerdict:
    """The host's delivery check on one instant, with what failed in words."""
    if len(state.goals_hold) != len(goals):
        raise ValueError(f"{len(state.goals_hold)} goal values for {len(goals)} goals")
    touching = [n for n in named if robot_touching(state.contacts, n)]
    moving = [n for n in named if n in state.velocities and not at_rest(state.velocities[n])]
    commanded_open, is_open = gripper_state(state.gripper_opening, state.gripper_command)
    failed: list[str] = []
    for i, (goal, holds) in enumerate(zip(goals, state.goals_hold, strict=True)):
        if not holds:
            failed.append(f"goal {i} ({goal.text}) does not hold")
    failed += [f"the robot touches {n}" for n in touching]
    for n in moving:
        linear, angular = speeds(state.velocities[n])
        failed.append(f"{n} is moving ({linear:.3f} m/s, {angular:.3f} rad/s)")
    if not is_open:
        how = "commanded open" if commanded_open else "commanded closed"
        failed.append(f"the gripper is not open ({how}, {state.gripper_opening:.3f} m apart)")
    return DeliveryVerdict(
        delivered=not failed,
        goals_hold=list(state.goals_hold),
        robot_touches=touching,
        not_at_rest=moving,
        gripper_commanded_open=commanded_open,
        gripper_opening_m=float(state.gripper_opening),
        reason="delivered" if not failed else "; ".join(failed),
    )


def scene_owner(rs: Any, name: str) -> str | None:
    """The body that owns a LIBERO scene name: itself, a region's object, or none (the table)."""
    sites = getattr(rs, "object_sites_dict", {}) or {}
    if name in sites:
        parent = sites[name].parent_name
        return None if parent is None or parent == TABLE else str(parent)
    return None if name == TABLE else name


class SceneGoals:
    """LIBERO's own predicates of the requested goals on one loaded scene, per instant.

    Unary predicates (`Turnon`, `Open`) take one object state, binary ones (`On`, `In`)
    two, as the BDDL goals do (`envs/predicates`); `On` asks LIBERO's `check_ontop`, which
    needs contact. Every name must be in the scene's `object_states_dict`.
    """

    def __init__(self, rs: Any, goals: Sequence[RequestedGoal]) -> None:
        from libero.libero.envs.predicates import eval_predicate_fn

        states = rs.object_states_dict
        missing = sorted({o for g in goals for o in g.objects if o not in states})
        if missing:
            raise ValueError(f"the loaded scene has no {', '.join(missing)}")
        self._evaluate = eval_predicate_fn
        self._calls = [(g.predicate.value.lower(), [states[o] for o in g.objects]) for g in goals]
        self.goals = tuple(goals)
        self.named = named_objects(goals, lambda name: scene_owner(rs, name))

    def holds(self) -> tuple[bool, ...]:
        return tuple(bool(self._evaluate(name, *args)) for name, args in self._calls)
