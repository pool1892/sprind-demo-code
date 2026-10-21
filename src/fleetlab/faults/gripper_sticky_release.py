"""A gripper that lets go late: a physical fault, as an action hook on the gripper command.

Residue on the jaws, a dry guide or a sticking valve holds the fingers shut
for a while after the robot commands them open: for `level` control steps
after the commanded gripper changes from closing to opening, the executed
command keeps the last closing value; a new closing command ends the hold.
The arm moves on as commanded, so a robot that has set an object down and
lifts away may carry it off again or drag it out of place. Under LIBERO's
own check this would be invisible (the check fires while the object is
still held); under the fleet's check, which wants the object let go and at
rest, it costs success. The commanded action in the log opens
on time; the gripper's measured opening shows the hold. `clean` (the jaws),
`service` or `replace` of the gripper removes it. A command of exactly 0
("hold the opening", which a teleoperator may send and X-VLA does not)
counts as not closing, so it starts the hold like an opening command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    DosePoint,
    NoParams,
    component_truth,
    default_points,
)

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7


@dataclass
class StickyState(FaultState):
    steps: int = 0
    closing: float | None = None  # the last closing command, while the fingers are shut
    held: int = 0  # steps the opening has been held back so far


@register("gripper_sticky_release")
class GripperStickyRelease(FaultTypeBase):
    """The fingers open `level` control steps after the command to open (action hook)."""

    kind = "gripper_sticky_release"
    fault_class = FaultClass.physical
    Params = NoParams
    dose_levels = (0.0, 5.0, 10.0, 20.0, 40.0)
    truth = component_truth({Verb.clean, Verb.service, Verb.replace_})
    calibration: tuple[DosePoint, ...] = default_points(
        (5.0, 10.0, 20.0, 40.0), Component.gripper.value, unit=" steps late open"
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.gripper:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        steps = round(spec.level)
        if steps != spec.level:
            raise ValueError(f"fault {spec.fault_id}: the hold is a whole number of steps")
        return StickyState(spec=spec, episode_id=identity.episode_id, steps=steps)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, StickyState)
        command = float(action[6])
        if command > 0.0:  # closing: remember it, the fingers are (getting) shut
            state.closing, state.held = command, 0
            return action
        if state.closing is None or state.held >= state.steps:
            state.closing = None
            return action
        state.held += 1
        out = np.array(action, copy=True)
        out[6] = state.closing
        result: Action7 = out
        return result
