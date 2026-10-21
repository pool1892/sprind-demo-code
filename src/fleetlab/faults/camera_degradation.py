"""A camera's image blurred, noisy or wrongly exposed: a sensor fault, as an observation hook.

The policy and the log see the degraded image; the simulator renders as
before. Sensor noise is drawn per step from the episode's own generator for
this fault (`episode_rng`), so an episode replays bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import PIXEL_KEYS
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    CAMERAS,
    DosePoint,
    FaultParams,
    camera_of,
    component_truth,
    default_points,
    episode_rng,
    with_image,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation
    from fleetlab.contracts.identity import EpisodeIdentity

# The image size the levels refer to (LeRobot's default render size).
REFERENCE_IMAGE_SIZE = 360


class DegradationParams(FaultParams):
    """Which degradation. Level: blur in pixels (Gaussian sigma, at a 360-pixel image),
    noise in 8-bit levels (standard deviation), exposure in stops (`darker` for under-exposed).
    """

    mode: Literal["blur", "noise", "exposure"] = "blur"
    darker: bool = False


@dataclass
class DegradationState(FaultState):
    camera_key: str = ""
    params: DegradationParams = field(default_factory=DegradationParams)
    rng: np.random.Generator | None = None


@register("camera_degradation")
class CameraDegradation(FaultTypeBase):
    """A camera's image blurred, noisy or wrongly exposed (observation hook).

    The policy and the log see the degraded image; the simulator renders as
    before. Sensor noise is drawn per step from the episode's own generator
    for this fault (`episode_rng`), so an episode replays bit for bit.
    """

    kind = "camera_degradation"
    fault_class = FaultClass.sensor
    Params = DegradationParams
    dose_levels = (0.0, 1.0, 2.0, 4.0)
    truth = component_truth({Verb.clean, Verb.replace_})
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (1.0, 2.0, 4.0), Component.scene_camera.value, {"mode": "blur"}, unit=" px blur"
        ),
        *default_points(
            (10.0, 25.0), Component.scene_camera.value, {"mode": "noise"}, unit=" noise"
        ),
        *default_points(
            (1.0, 2.0),
            Component.scene_camera.value,
            {"mode": "exposure", "darker": True},
            unit=" stops darker",
        ),
        *default_points(
            (1.0,), Component.scene_camera.value, {"mode": "exposure"}, unit=" stop brighter"
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        params = DegradationParams.model_validate(spec.params)
        rng = episode_rng(identity, spec, "image-noise") if params.mode == "noise" else None
        return DegradationState(
            spec=spec,
            episode_id=identity.episode_id,
            camera_key=PIXEL_KEYS[CAMERAS[camera_of(spec)]],
            params=params,
            rng=rng,
        )

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, DegradationState)
        image = observation["pixels"][state.camera_key]
        level = state.spec.level
        if state.params.mode == "blur":
            out = blur(image, level * image.shape[0] / REFERENCE_IMAGE_SIZE)
        elif state.params.mode == "exposure":
            gain = 2.0 ** (-level if state.params.darker else level)
            out = _to_uint8(image.astype(np.float32) * np.float32(gain))
        else:
            assert state.rng is not None
            noise = state.rng.standard_normal(image.shape, dtype=np.float32)
            out = _to_uint8(image.astype(np.float32) + noise * np.float32(level))
        return with_image(observation, state.camera_key, out)


def _to_uint8(values: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def blur(image: npt.NDArray[np.uint8], sigma: float) -> npt.NDArray[np.uint8]:
    """A Gaussian blur of an 8-bit image, edges mirrored.

    OpenCV's 8-bit Gaussian blur is bit-exact by design, so the result does
    not depend on its thread count. OpenCV comes with LeRobot.
    """
    if sigma <= 0:
        return image.copy()
    import cv2

    out: Any = cv2.GaussianBlur(
        np.ascontiguousarray(image),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.asarray(out, dtype=np.uint8)
