"""A camera's calibration record rewritten: a configuration change with no mechanism.

`camera.scene.calibration` and `camera.wrist.calibration` hold the pose the
fleet believes its camera has. Someone (or an automatic check) writes a new
value, a few millimetres and tenths of a degree from the old one, and the
change log shows it with its true old and new values, like any other change.
The policy reads images, not the believed pose, so nothing an episode does
changes: every episode is its healthy twin bit for bit (the type implements
no hook at all). It is the library's inert confounder entry: a
change in the change log on the same day as a fault that does
something, which an analysis must not blame. `set` or `revert` of the key
removes it, to no effect. Its verdict is "inert", not "decoy": nothing is
measured, there is nothing to measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import DosePoint, ValueParams, config_key_of, config_truth

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity

CALIBRATION_KEYS = frozenset(
    {ConfigKey.camera_scene_calibration, ConfigKey.camera_wrist_calibration}
)
# The poses the world generator writes as nominal (x, y, z in m; roll, pitch, yaw in degrees).
NOMINAL_SCENE_CALIBRATION: list[float] = [0.66, 0.0, 1.61, 0.0, 33.0, 90.0]
NOMINAL_WRIST_CALIBRATION: list[float] = [0.0, 0.0, 0.05, 0.0, 0.0, 180.0]


@register("calibration_update")
class CalibrationUpdate(FaultTypeBase):
    """A camera's believed pose rewritten in the configuration; no hook (inert).

    Level 1 while the new record is in force; `params.value` is the new pose.
    """

    kind = "calibration_update"
    fault_class = FaultClass.configuration
    Params = ValueParams
    dose_levels = (0.0, 1.0)
    truth = config_truth()
    calibration: tuple[DosePoint, ...] = (
        DosePoint(
            "scene calibration rewritten",
            1.0,
            {"value": [0.662, -0.003, 1.608, 0.0, 33.4, 90.2]},
            ConfigKey.camera_scene_calibration.value,
            NOMINAL_SCENE_CALIBRATION,
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        config_key_of(spec, set(CALIBRATION_KEYS))
        ValueParams.model_validate(spec.params)
        return FaultState(spec=spec, episode_id=identity.episode_id)

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return ValueParams.model_validate(spec.params).value
