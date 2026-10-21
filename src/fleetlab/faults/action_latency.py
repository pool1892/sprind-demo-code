"""Commands executed late: a configuration fault, as an action hook with a delay line.

`control.action_delay_steps` is the number of control steps a command waits
in the command buffer. The fault writes a `Change` of that key with its true
old and new values, like a benign change; `set` or `revert` removes it. The
walking skeleton's fault (spikes/skeleton/faults.py); the experimental design calls it
`action_delay`, a name the contracts' vocabulary already spells
(`control.action_delay_steps`), so the kind is `action_latency`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConfigKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import DosePoint, NoParams, config_key_of, config_truth, default_points

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7


@dataclass
class LatencyState(FaultState):
    steps: int = 0
    line: deque[Any] = field(default_factory=deque)


@register("action_latency")
class ActionLatency(FaultTypeBase):
    """Every commanded action is executed `level` control steps late (action hook).

    Until the delay line has filled, the episode's first command is executed,
    as in the walking skeleton. Measured on X-VLA in stage 1: up to 14 steps
    nothing, 17 steps 71 to 72.5% against 97% healthy.
    """

    kind = "action_latency"
    fault_class = FaultClass.configuration
    Params = NoParams
    dose_levels = (0.0, 8.0, 14.0, 17.0, 20.0)
    truth = config_truth()
    calibration: tuple[DosePoint, ...] = default_points(
        (8.0, 14.0, 17.0, 20.0),
        ConfigKey.control_action_delay_steps.value,
        nominal=0,
        unit=" steps",
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        config_key_of(spec, {ConfigKey.control_action_delay_steps})
        steps = round(spec.level)
        if steps != spec.level:
            raise ValueError(f"fault {spec.fault_id}: a delay is a whole number of steps")
        return LatencyState(spec=spec, episode_id=identity.episode_id, steps=steps)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, LatencyState)
        state.line.append(action)
        if len(state.line) > state.steps:
            out: Action7 = state.line.popleft()
            return out
        first: Action7 = state.line[0]
        return first

    @staticmethod
    def new_value(spec: FaultSpec) -> ConfigValue:
        return round(spec.level)


#
