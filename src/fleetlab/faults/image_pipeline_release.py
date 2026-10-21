"""A software release that changes how camera images are prepared: a software fault,
as an observation hook on both cameras.

The change log shows a `Change` of `software.version` and nothing else, as
for every release (contracts, section 3); what the new version does to the
images stays hidden in it. Three regressions a fleet meets in practice:

- `channel_swap`: the red and blue channels exchanged, the classic slip
  between a library that hands out BGR and a model that expects RGB;
- `crop`: `level` percent of the width and of the height cut away, half on
  each side, and the rest scaled back to the full size, as when a new camera
  driver defaults to another region of interest: the policy sees the cell
  zoomed in;
- `jpeg`: the frames JPEG-compressed at quality `100 - level`, as when a
  release compresses the camera stream to save bandwidth.

The policy and the log see the prepared images; the simulator renders as
before. OpenCV's 8-bit resize and its JPEG codec are deterministic, so an
episode replays bit for bit. `revert` or `set` of `software.version` back to
the old release removes it. The other software kind, `software_release`,
carries faults of other types; this one brings a mechanism of its own,
which only software has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.env import PIXEL_KEYS, Camera
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import (
    SOFTWARE,
    DosePoint,
    FaultParams,
    config_key_of,
    config_truth,
    with_image,
)
from fleetlab.faults.software_release import VERSION_PATTERN

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation
    from fleetlab.contracts.identity import EpisodeIdentity

OLD_VERSION = "2.4.1"


Stream = Literal["scene", "wrist"]


def _both() -> list[Stream]:
    return ["scene", "wrist"]


class PipelineParams(FaultParams):
    """The release the robots now run and what it does to the images.

    Level: 1 for `channel_swap`; the percent of each side cropped away for
    `crop` (below 50); 100 minus the JPEG quality for `jpeg` (below 100).
    """

    version: Annotated[str, Field(pattern=VERSION_PATTERN)]
    mode: Literal["channel_swap", "crop", "jpeg"]
    cameras: list[Stream] = Field(default_factory=_both, min_length=1)


def _point(label: str, level: float, mode: str) -> DosePoint:
    return DosePoint(
        label,
        level,
        {"version": "2.6.0", "mode": mode},
        ConfigKey.software_version.value,
        OLD_VERSION,
    )


@register("image_pipeline_release")
class ImagePipelineRelease(FaultTypeBase):
    """A release whose image preparation swaps channels, crops or compresses (observation hook)."""

    kind = "image_pipeline_release"
    fault_class = FaultClass.software
    Params = PipelineParams
    dose_levels = (0.0, 1.0, 10.0, 20.0, 70.0, 90.0)
    truth = config_truth(SOFTWARE)
    calibration: tuple[DosePoint, ...] = (
        _point("red and blue swapped", 1.0, "channel_swap"),
        _point("10% cropped", 10.0, "crop"),
        _point("20% cropped", 20.0, "crop"),
        _point("jpeg quality 30", 70.0, "jpeg"),
        _point("jpeg quality 10", 90.0, "jpeg"),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        config_key_of(spec, {ConfigKey.software_version})
        params = PipelineParams.model_validate(spec.params)
        limit = {"channel_swap": 1.0, "crop": 49.0, "jpeg": 99.0}[params.mode]
        if spec.level > limit:
            raise ValueError(f"fault {spec.fault_id}: {params.mode} goes up to {limit:g}")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        params = PipelineParams.model_validate(state.spec.params)
        for camera in params.cameras:
            key = PIXEL_KEYS[Camera(camera)]
            image = prepare(observation["pixels"][key], params.mode, state.spec.level)
            observation = with_image(observation, key, image)
        return observation

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return PipelineParams.model_validate(spec.params).version


def prepare(image: npt.NDArray[np.uint8], mode: str, level: float) -> npt.NDArray[np.uint8]:
    """An image as the release prepares it."""
    import cv2

    if level <= 0.0:
        return image.copy()
    if mode == "channel_swap":
        return np.ascontiguousarray(image[..., ::-1])
    if mode == "crop":
        height, width = image.shape[:2]
        dy = round(height * level / 200.0)  # half of the cropped share on each side
        dx = round(width * level / 200.0)
        inner = np.ascontiguousarray(image[dy : height - dy, dx : width - dx])
        out: Any = cv2.resize(inner, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.asarray(out, dtype=np.uint8)
    quality = round(100.0 - level)
    bgr = np.ascontiguousarray(image[..., ::-1])
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded: Any = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(np.asarray(decoded, dtype=np.uint8)[..., ::-1])
