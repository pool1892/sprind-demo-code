"""Dirt on a camera's lens: a sensor fault, as an observation hook.

A film of dust and grime on the front glass scatters light. Where it lies,
the image is blurred and washed out toward a bright haze (veiling glare);
the film is densest in one patch, a smear, and thinner elsewhere. The level
is the film's density, 0 for a clean lens to 1 for one nobody has wiped for
weeks: at a pixel, the image is blended with its blurred and hazed version by
`level * map(pixel)`, where the map is 1 at the smear's centre and falls to
`floor` far from it. Dust settles a little every day, so this is the
library's natural drift: a level that rises day by day until the lens is
cleaned. `clean` removes it, the cheapest fix there is; `replace` does too.

The policy and the log see the dirty image; the simulator renders as before.
Nothing is drawn at random, so an episode replays bit for bit. The health
checks do not see it: the calibration residual measures where the camera
points, not how clear its image is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydantic import Field

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
    with_image,
)
from fleetlab.faults.camera_degradation import REFERENCE_IMAGE_SIZE, blur

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation
    from fleetlab.contracts.identity import EpisodeIdentity

# The grey the scattered light washes the image toward (8-bit), and how far, at full density.
HAZE_GREY = 205.0


class FilmParams(FaultParams):
    """Where the film lies and what it does at full density.

    `centre` is the smear's centre as fractions of the image's width and
    height (of the image as delivered), `radius` its spread as a fraction of
    the image's size (a Gaussian's sigma), `floor` the film's density far
    from it relative to the centre. `blur_px` is the blur at full density at
    a 360-pixel image, `haze` how far the image is washed toward the haze.
    """

    centre: list[float] = Field(default_factory=lambda: [0.5, 0.5], min_length=2, max_length=2)
    radius: float = Field(default=0.3, gt=0.0, le=2.0)
    floor: float = Field(default=0.3, ge=0.0, le=1.0)
    blur_px: float = Field(default=10.0, gt=0.0, le=40.0)
    haze: float = Field(default=0.35, ge=0.0, le=1.0)


@dataclass
class FilmState(FaultState):
    camera_key: str = ""


@register("lens_contamination")
class LensContamination(FaultTypeBase):
    """A dirty camera lens: blur and haze, densest in a smear (observation hook).

    Level: the film's density, 0 to 1.
    """

    kind = "lens_contamination"
    fault_class = FaultClass.sensor
    Params = FilmParams
    dose_levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    truth = component_truth({Verb.clean, Verb.replace_})
    calibration: tuple[DosePoint, ...] = (
        *default_points((0.25, 0.5, 0.75, 1.0), Component.scene_camera.value, unit=" scene film"),
        *default_points((0.5, 1.0), Component.wrist_camera.value, unit=" wrist film"),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        FilmParams.model_validate(spec.params)
        if spec.level > 1.0:
            raise ValueError(f"fault {spec.fault_id}: the film's density is at most 1")
        key = PIXEL_KEYS[CAMERAS[camera_of(spec)]]
        return FilmState(spec=spec, episode_id=identity.episode_id, camera_key=key)

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, FilmState)
        params = FilmParams.model_validate(state.spec.params)
        image = observation["pixels"][state.camera_key]
        out = dirty(image, state.spec.level, params)
        return with_image(observation, state.camera_key, out)


@lru_cache(maxsize=16)
def density_map(
    height: int, width: int, cx: float, cy: float, radius: float, floor: float
) -> npt.NDArray[np.float32]:
    """The film's relative density at every pixel: 1 at the centre, `floor` far away."""
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height - cy
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width - cx
    d2 = ys[:, None] ** 2 + xs[None, :] ** 2
    bump = np.exp(-d2 / (2.0 * radius * radius))
    out: npt.NDArray[np.float32] = (floor + (1.0 - floor) * bump).astype(np.float32)
    out.setflags(write=False)
    return out


def dirty(image: npt.NDArray[np.uint8], level: float, params: FilmParams) -> npt.NDArray[np.uint8]:
    """The image through a lens with a film of this density."""
    if level <= 0.0:
        return image.copy()
    height, width = image.shape[:2]
    cx, cy = params.centre
    weight = np.float32(level) * density_map(height, width, cx, cy, params.radius, params.floor)
    weight = np.clip(weight, 0.0, 1.0)[..., None]
    sharp = image.astype(np.float32)
    soft = blur(image, params.blur_px * height / REFERENCE_IMAGE_SIZE).astype(np.float32)
    haze = np.float32(params.haze)
    veiled = soft * (np.float32(1.0) - haze) + haze * np.float32(HAZE_GREY)
    mixed = sharp + weight * (veiled - sharp)
    out: npt.NDArray[np.uint8] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return out
