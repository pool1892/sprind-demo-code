"""One of the policy's settings changed in the configuration: a configuration fault,
as a policy input hook.

Sampling steps, actions executed per chunk, image views or precision. The
engine runs the policy with the effective spec the hook returns and records
it on the truth side. The fault writes a `Change` of its key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.identity import PolicySpec
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import DosePoint, ValueParams, config_key_of, config_truth

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity


# The configuration key of each policy setting -> its field in `PolicySpec` and its type.
POLICY_FIELDS: dict[ConfigKey, tuple[str, type]] = {
    ConfigKey.policy_sampling_steps: ("sampling_steps", int),
    ConfigKey.policy_actions_executed_per_chunk: ("actions_executed_per_chunk", int),
    ConfigKey.policy_image_views: ("image_views", int),
    ConfigKey.policy_precision: ("precision", str),
    # How the rollout cuts each chunk, like the executed count (not a calibrated fault).
    ConfigKey.control_latency_compensation_steps: ("latency_compensation_steps", int),
}


def apply_policy_setting(policy_spec: PolicySpec, key: ConfigKey, value: ConfigValue) -> PolicySpec:
    """The spec with one setting changed, validated like any spec."""
    name, kind = POLICY_FIELDS[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ValueError(f"{key.value} takes a {kind.__name__}, not {value!r}")
    return PolicySpec.model_validate({**policy_spec.model_dump(), name: value})


@register("policy_knob")
class PolicyKnob(FaultTypeBase):
    """One of the policy's settings changed in the configuration (policy input hook).

    Level 1 while the change is in force; `params.value` is the new value
    and the target's key says which setting. The stage-1 measurements on 100
    healthy episodes (spikes/xvla-speed section 6): 5 sampling steps 98, 10
    actions per chunk 99, 3 image views 98, against 100 with the stored values.
    """

    kind = "policy_knob"
    fault_class = FaultClass.configuration
    Params = ValueParams
    dose_levels = (0.0, 1.0)
    truth = config_truth()
    calibration: tuple[DosePoint, ...] = (
        DosePoint("5 sampling steps", 1.0, {"value": 5}, ConfigKey.policy_sampling_steps.value, 10),
        DosePoint("2 sampling steps", 1.0, {"value": 2}, ConfigKey.policy_sampling_steps.value, 10),
        DosePoint("1 sampling step", 1.0, {"value": 1}, ConfigKey.policy_sampling_steps.value, 10),
        DosePoint(
            "10 actions per chunk",
            1.0,
            {"value": 10},
            ConfigKey.policy_actions_executed_per_chunk.value,
            30,
        ),
        DosePoint("3 image views", 1.0, {"value": 3}, ConfigKey.policy_image_views.value, 4),
        # 2: scene and wrist without an empty view; 1: the scene camera alone.
        DosePoint("2 image views", 1.0, {"value": 2}, ConfigKey.policy_image_views.value, 4),
        DosePoint("1 image view", 1.0, {"value": 1}, ConfigKey.policy_image_views.value, 4),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        key = config_key_of(spec, set(POLICY_FIELDS))
        apply_policy_setting(identity.policy, key, ValueParams.model_validate(spec.params).value)
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def policy_input_hook(self, policy_spec: PolicySpec, state: FaultState) -> PolicySpec:
        key = state.spec.target.config_key
        assert key is not None
        return apply_policy_setting(
            policy_spec, key, ValueParams.model_validate(state.spec.params).value
        )

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return ValueParams.model_validate(spec.params).value


#
