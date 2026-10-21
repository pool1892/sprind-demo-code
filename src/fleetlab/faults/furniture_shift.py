"""The table, with everything on it, shifted within installation tolerance: a site fault
on the `placement` condition, as a cell, an observation and an action hook.

The robot and the scene camera stand where they always stood; the table, the
furniture and the objects stand `level` centimetres away from where the cell
was set up, in the table's plane. In the simulator the table cannot be moved
under the recorded start states, so the equivalent is done: the robot's base
and the scene camera move the other way in the world (cell hook), and the
robot keeps working in its own frame, the frame it believes the world is in:
it reports its end effector's position shifted back (observation hook), and
an absolute target pose it is sent lands where that frame puts it (action
hook). The policy therefore sees and acts exactly as if the table had moved.
The wrist camera moves with the robot. In relative control a command is a
change of pose and needs no shift.

The experimental design proposed moving the fixtures through LIBERO's placement
sampler. That cannot move the objects that stand on or in a fixture (their
poses come from the recorded start state after the reset), so a bowl in a
moved drawer would start inside its wall; this mechanism avoids that.
`change_condition placement` back to the nominal value removes the fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.vocabulary import ConditionKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import (
    SCENE_CAMERA,
    DosePoint,
    FaultParams,
    ModelPatch,
    condition_truth,
    default_points,
    handle_env,
    mj_id,
    with_eef_pos,
)
from fleetlab.faults.command import CURRENT_COMMAND

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Observation, Undo
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7

# The value of the `placement` condition: the table's offset [x, y] in centimetres.
NOMINAL_PLACEMENT: list[float] = [0.0, 0.0]


class ShiftParams(FaultParams):
    """The direction of the shift in the table's plane, in degrees.

    0 is straight away from the robot (+x), 90 to the robot's left (+y).
    """

    direction_deg: float = 90.0


@dataclass
class ShiftState(FaultState):
    offset: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))


@register("furniture_shift")
class FurnitureShift(FaultTypeBase):
    """The table and everything on it shifted relative to robot and scene camera.

    Level: centimetres. Absolute control shifts the commanded positions.
    """

    kind = "furniture_shift"
    fault_class = FaultClass.site
    Params = ShiftParams
    dose_levels = (0.0, 1.0, 2.0, 4.0)
    truth = condition_truth()
    calibration: tuple[DosePoint, ...] = default_points(
        (1.0, 2.0, 4.0),
        ConditionKey.placement.value,
        nominal=NOMINAL_PLACEMENT,
        unit=" cm sideways",
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.condition_key is not ConditionKey.placement:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        dx, dy = shift_cm(spec)
        return ShiftState(
            spec=spec,
            episode_id=identity.episode_id,
            offset=np.array([dx, dy, 0.0]) / 100.0,
        )

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        assert isinstance(state, ShiftState)
        return move_robot_and_camera(handle, -state.offset)

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, ShiftState)
        pos = np.asarray(observation["robot_state"]["eef"]["pos"], dtype=np.float64)
        return with_eef_pos(observation, pos + state.offset)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, ShiftState)
        command = CURRENT_COMMAND.get()
        if command.control_mode is ControlMode.relative or command.world_frame:
            return action
        out = np.array(action, copy=True)
        out[:3] = out[:3] - state.offset.astype(out.dtype)
        result: Action7 = out
        return result

    def to_policy_frame(
        self, action: Action7, state: FaultState, control_mode: ControlMode
    ) -> Action7:
        """Express a world target in the translated command frame, before buffering."""
        assert isinstance(state, ShiftState)
        if control_mode is ControlMode.relative or not np.any(state.offset):
            return action
        out: Action7 = np.array(action, copy=True)
        out[:3] += state.offset.astype(out.dtype)
        return out

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        dx, dy = shift_cm(spec)
        return [dx, dy]


def shift_cm(spec: FaultSpec) -> tuple[float, float]:
    """The table's offset in centimetres, [x, y]."""
    angle = np.deg2rad(ShiftParams.model_validate(spec.params).direction_deg)
    return float(spec.level * np.cos(angle)), float(spec.level * np.sin(angle))


def move_robot_and_camera(handle: CellHandle, offset: npt.NDArray[np.float64]) -> Undo | None:
    """Move the robot's base and the scene camera by `offset` metres in the world."""
    if not np.any(offset):
        return None
    model = handle.model
    env: Any = handle_env(handle)
    base = mj_id(model, "body", env.robots[0].robot_model.root_body)
    if int(model.body_parentid[base]) != 0:
        raise RuntimeError("the robot's base is expected to hang in the world frame")
    camera = mj_id(model, "camera", SCENE_CAMERA)
    if int(model.cam_bodyid[camera]) != 0:
        raise RuntimeError("the scene camera is expected to hang in the world frame")
    patch = ModelPatch()
    patch.set(model, "body_pos", base, model.body_pos[base] + offset)
    patch.set(model, "cam_pos", camera, model.cam_pos[camera] + offset)
    return patch.undo
