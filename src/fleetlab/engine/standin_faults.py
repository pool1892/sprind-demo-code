"""The two calibrated faults of stage 1, the stand-ins the engine's measurements used.

`fleetlab.faults` now has both as `camera_pose_offset` (pan, level in degrees)
and `action_latency` (level in steps); on the faulted mix their episodes are
bit-identical to these (engine README, `mix_real` against `mix_a`).

Ported from spikes/xvla-speed/faults.py onto `contracts.faults.FaultTypeBase`,
with the same arithmetic: the scene camera turned about the vertical axis
(cell hook on the live MuJoCo model, undone after the episode), and commanded
actions executed n control steps late (action hook; until the line has
filled, the first command of the episode is executed).

They register under kinds of their own (`standin_camera_pan`,
`standin_action_delay`) in a registry of their own, `STANDIN`, so that they
never clash with the kinds of `fleetlab.faults`. A run uses
them with `RunConfig(fault_registry="fleetlab.engine.standin_faults:STANDIN")`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from fleetlab.contracts.env import CellHandle, Undo
from fleetlab.contracts.faults import (
    Direction,
    FaultSpec,
    FaultState,
    FaultTypeBase,
    TruthMapping,
)
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.ids import ScopeKind
from fleetlab.contracts.policy import Action7
from fleetlab.contracts.vocabulary import FaultClass, TargetKind, Verb
from fleetlab.faults import FaultRegistry

SCENE_CAMERA = "agentview"  # MuJoCo's name of the camera behind the scene image


class CameraPanParams(BaseModel):
    degrees: float


class DelayParams(BaseModel):
    steps: int


class CameraPan(FaultTypeBase):
    """The fixed scene camera turned about the vertical axis, as if its mount had been knocked."""

    kind = "standin_camera_pan"
    fault_class = FaultClass.sensor
    Params = CameraPanParams
    dose_levels = (3.0, 6.0, 9.0)
    truth = TruthMapping(
        target_kind=TargetKind.component,
        fixing_verbs=frozenset({Verb.recalibrate}),
        scopes=frozenset({ScopeKind.robot}),
        writes_change=False,
        directions=frozenset({Direction.degraded}),
    )

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        import mujoco

        model = handle.model
        degrees = float(state.spec.params["degrees"])  # type: ignore[arg-type]
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, SCENE_CAMERA)
        if cam < 0 or model.cam_bodyid[cam] != 0:
            raise RuntimeError("the scene camera is expected to hang in the world frame")
        saved = model.cam_quat[cam].copy()
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), np.deg2rad(degrees))
        new = np.zeros(4)
        mujoco.mju_mulQuat(new, turn, model.cam_quat[cam])
        model.cam_quat[cam] = new

        def undo() -> None:
            model.cam_quat[cam] = saved

        return undo


@dataclass
class DelayState(FaultState):
    line: deque[Action7] = field(default_factory=deque)


class ActionDelay(FaultTypeBase):
    """Every commanded action is executed `steps` control steps late."""

    kind = "standin_action_delay"
    fault_class = FaultClass.software
    Params = DelayParams
    dose_levels = (14.0, 17.0, 20.0)
    truth = TruthMapping(
        target_kind=TargetKind.config_key,
        fixing_verbs=frozenset({Verb.revert, Verb.set}),
        scopes=frozenset({ScopeKind.robot, ScopeKind.site, ScopeKind.fleet}),
        writes_change=True,
        directions=frozenset({Direction.misconfigured, Direction.changed}),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return DelayState(spec=spec, episode_id=identity.episode_id)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, DelayState)
        state.line.append(np.asarray(action, np.float32).copy())
        if len(state.line) > int(state.spec.params["steps"]):  # type: ignore[arg-type]
            return state.line.popleft()
        return state.line[0]


STANDIN = FaultRegistry()
STANDIN.add(CameraPan())
STANDIN.add(ActionDelay())
