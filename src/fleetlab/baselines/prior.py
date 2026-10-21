"""The unchanged stage-2 fixed prior, shared without importing the scorer."""

from __future__ import annotations

from collections.abc import Sequence

from fleetlab.contracts.actions import Action
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.vocabulary import Component, Verb

# The prior, most plausible first: cameras out of calibration, dirty optics, a worn
# gripper, then the arm and the controller.
PRIOR: tuple[tuple[Verb, Component], ...] = (
    (Verb.recalibrate, Component.scene_camera),
    (Verb.recalibrate, Component.wrist_camera),
    (Verb.clean, Component.scene_camera),
    (Verb.clean, Component.wrist_camera),
    (Verb.clean, Component.gripper),
    (Verb.service, Component.gripper),
    (Verb.service, Component.arm),
    (Verb.service, Component.controller),
)


def trivial_no_data_ranker(fleet: Fleet, catalogue: Sequence[Action]) -> list[Action]:
    """The catalogue ranked by the fixed prior, each prior entry over the robots in id order,
    then the rest of the catalogue in its own order.

    Reads only the static structure (the fleet's robots and the catalogue), never a
    log. Only catalogue entries are proposed, and only complete ones (every value
    filled), so that the list can be cut into fix sets.
    """
    entries = [a for a in catalogue if a.is_complete]
    robots = sorted(r.id for r in fleet.robots())
    by_key = {(a.verb, a.target.component, a.target.scope): a for a in entries}
    ranked: list[Action] = []
    seen: set[str] = set()
    for verb, component in PRIOR:
        for robot in robots:
            action = by_key.get((verb, component, robot))
            if action is not None:
                ranked.append(action)
                seen.add(action.model_dump_json())
    for action in entries:
        key = action.model_dump_json()
        if key not in seen:
            ranked.append(action)
            seen.add(key)
    return ranked
