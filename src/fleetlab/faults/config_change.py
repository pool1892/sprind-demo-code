"""A configuration value an action set: the generic kind of the changes `set` adds
(contracts, section 8; `contracts.actions.CONFIG_CHANGE_KIND`).

`params = {"value": v}`, level 1, `nominal` the value it replaced, the target
naming the key at the action's scope. It dispatches on the key to the type
whose hooks that key works through, with the same level and params a fault of
that type would carry, so a `set` to a fault's own value reproduces that
fault's episodes exactly:

- `policy.sampling_steps`, `policy.actions_executed_per_chunk`,
  `policy.image_views`, `policy.precision`: `policy_knob` (policy input hook);
- `control.mode`: `control_mode_change` (observation and action hooks);
- `control.action_delay_steps`: `action_latency` with that many steps;
- `camera.scene.calibration`, `camera.wrist.calibration`: nothing. The key
  holds the pose the fleet believes its camera has; the policy reads images,
  not the belief, so setting it moves nothing;
- `software.version`: nothing. A release means something only through what a
  `software_release` fault says it carries; a version set by an action is
  taken to carry nothing.

A value equal to the nominal one is no change and runs no hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import CONFIG_CHANGE_KIND
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._carrier import CarrierBase
from fleetlab.faults._common import ValueParams, config_truth
from fleetlab.faults.policy_knob import POLICY_FIELDS

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity

# The keys whose change does nothing to an episode.
NO_MECHANISM = frozenset(
    {
        ConfigKey.camera_scene_calibration,
        ConfigKey.camera_wrist_calibration,
        ConfigKey.software_version,
    }
)


@register(CONFIG_CHANGE_KIND)
class ConfigChange(CarrierBase):
    """A configuration key set to `params.value` by an action (dispatches on the key)."""

    kind = CONFIG_CHANGE_KIND
    fault_class = FaultClass.configuration
    Params = ValueParams
    dose_levels = (0.0, 1.0)
    truth = config_truth()

    def inner_specs(self, spec: FaultSpec, identity: EpisodeIdentity) -> list[FaultSpec]:
        key = spec.target.config_key
        if key is None:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) names no configuration key")
        value = ValueParams.model_validate(spec.params).value
        if key in NO_MECHANISM or value == spec.nominal:
            return []
        if key in POLICY_FIELDS:
            return [_as(spec, "policy_knob", 1.0, {"value": value})]
        if key is ConfigKey.control_mode:
            return [_as(spec, "control_mode_change", 1.0, {"value": value})]
        if key is ConfigKey.control_action_delay_steps:
            if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{key.value} takes a number of steps, not {value!r}")
            return [_as(spec, "action_latency", float(value), {})] if value > 0 else []
        raise ValueError(f"no mechanism is defined for {key.value}")

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return ValueParams.model_validate(spec.params).value


def _as(spec: FaultSpec, kind: str, level: float, params: dict[str, ConfigValue]) -> FaultSpec:
    return spec.model_copy(update={"kind": kind, "level": level, "params": params})
