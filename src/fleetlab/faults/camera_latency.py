"""A camera's frames arrive late: a sensor fault, as an observation hook with a frame buffer.

The images the policy and the log receive are `level` control steps old,
while the robot's joint and gripper readings are current: a camera behind a
saturated USB hub or a congested network link, or a driver that queues
frames instead of dropping them, delivers what it saw a while ago. Until
`level` frames have passed, the episode's first frame is repeated (the
camera has nothing older). The buffered frames are episode state, so a
snapshot carries them and an episode replays bit for bit. The commands are
executed on time; only the picture is old, which tells it apart from a
delay in the command buffer. `service` (cable, hub, driver) or `replace`
removes it. X-VLA reads the cameras once per chunk (every 30 steps), so a
lag shorter than a few steps hardly matters and one longer than a chunk
means the policy never sees a frame newer than its previous call.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import PIXEL_KEYS
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    CAMERAS,
    DosePoint,
    NoParams,
    camera_of,
    component_truth,
    default_points,
    with_image,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation
    from fleetlab.contracts.identity import EpisodeIdentity


@dataclass
class LagState(FaultState):
    camera_key: str = ""
    steps: int = 0
    frames: deque[npt.NDArray[np.uint8]] = field(default_factory=deque)


@register("camera_latency")
class CameraLatency(FaultTypeBase):
    """A camera's images are `level` control steps old (observation hook). Level: steps."""

    kind = "camera_latency"
    fault_class = FaultClass.sensor
    Params = NoParams
    dose_levels = (0.0, 6.0, 12.0, 24.0, 36.0)
    truth = component_truth({Verb.service, Verb.replace_})
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (6.0, 12.0, 24.0, 36.0), Component.scene_camera.value, unit=" steps scene lag"
        ),
        *default_points((12.0,), Component.wrist_camera.value, unit=" steps wrist lag"),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        steps = round(spec.level)
        if steps != spec.level:
            raise ValueError(f"fault {spec.fault_id}: a camera's lag is a whole number of steps")
        key = PIXEL_KEYS[CAMERAS[camera_of(spec)]]
        return LagState(spec=spec, episode_id=identity.episode_id, camera_key=key, steps=steps)

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, LagState)
        state.frames.append(np.array(observation["pixels"][state.camera_key], copy=True))
        out = state.frames.popleft() if len(state.frames) > state.steps else state.frames[0]
        return with_image(observation, state.camera_key, np.array(out, copy=True))
