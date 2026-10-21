"""The arm controller switched to the other control mode: a configuration fault,
as a policy input hook.

The fault writes a `Change` of `control.mode`; `set` or `revert` removes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.identity import PolicySpec
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import DosePoint, ValueParams, config_key_of, config_truth

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity


@register("control_mode_change")
class ControlModeChange(FaultTypeBase):
    """The robot's controller runs in the other control mode (policy input hook).

    The effective spec carries the changed mode, and the engine sets the
    environment's arm controller from the effective spec (`run_job`, step 2,
    on stage2/engine). The policy does not adapt: X-VLA's adapter turns every
    chunk into absolute target poses whatever the mode, so a controller
    switched to relative control reads each target pose as a change of pose
    (clipped to [-1, 1] and scaled to 5 cm and 0.5 rad per step), the
    misconfigured controller as a fleet would see it. A policy whose adapter
    converts its output to the effective mode would not notice the fault.
    Level 1 while the change is in force; `params.value` is the new mode.
    """

    kind = "control_mode_change"
    fault_class = FaultClass.configuration
    Params = ValueParams
    dose_levels = (0.0, 1.0)
    truth = config_truth()
    calibration: tuple[DosePoint, ...] = (
        DosePoint("relative", 1.0, {"value": "relative"}, ConfigKey.control_mode.value, "absolute"),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        config_key_of(spec, {ConfigKey.control_mode})
        ControlMode(str(ValueParams.model_validate(spec.params).value))
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def policy_input_hook(self, policy_spec: PolicySpec, state: FaultState) -> PolicySpec:
        mode = ControlMode(str(ValueParams.model_validate(state.spec.params).value))
        return PolicySpec.model_validate({**policy_spec.model_dump(), "control_mode": mode})

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return ValueParams.model_validate(spec.params).value
