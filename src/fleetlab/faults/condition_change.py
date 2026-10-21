"""A site condition an action changed: the generic kind of the changes `change_condition`
adds (contracts, section 8; `contracts.actions.CONDITION_CHANGE_KIND`).

`params = {"value": v}`, level 1, `nominal` the value it replaced, the target
naming the condition at the site. The value of each condition is defined by
the fault type that acts on it, measured against the cell as LIBERO sets it
up, and the dispatch calls that type's mechanism with the value itself, so a
change to a fault's own value (`condition_value`) reproduces the fault
exactly:

- `lighting`: stops, signed (`illumination_change`), or [stops, warmth] with
  the lamps' colour too (`light_color_shift`; warmth signed, positive warmer);
- `placement`: the table's offset [x, y] in centimetres (`furniture_shift`),
  or [x, y, z] with its height too (`table_height_change`; z up);
- `object_batch`: [mass doublings, friction halvings] (`object_batch_variant`);
- `clutter`: the blend of walls and floor toward another colour, 0 to 1
  (`backdrop_change`, provisionally on this condition).

A value equal to the nominal one is no change and runs no hook; the shorter of
the two is padded with zeros for that comparison (`same_condition`).
`contracts.actions.resolve` compares an action's value with a fault's nominal
literally, so a world should write a condition's nominal in the form its
faults' values take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import CONDITION_CHANGE_KIND
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConditionKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import ValueParams, condition_truth
from fleetlab.faults.backdrop_change import BACKDROP_CONDITION, BackdropParams, repaint
from fleetlab.faults.furniture_shift import FurnitureShift, ShiftState, move_robot_and_camera
from fleetlab.faults.illumination_change import change_light
from fleetlab.faults.light_color_shift import change_colour
from fleetlab.faults.object_batch_variant import change_batch

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Observation, Undo
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7, ControlMode


@dataclass
class ConditionState(FaultState):
    key: ConditionKey | None = None  # None: the value is the nominal one, nothing happens
    value: Any = None
    shift: ShiftState | None = None


def _numbers(key: ConditionKey, value: ConfigValue, *counts: int) -> list[float]:
    """The numbers of a condition's value, which must have one of the allowed `counts`."""
    values: list[object] = list(value) if isinstance(value, list) else [value]
    numbers = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    if len(values) not in counts or len(numbers) != len(values):
        allowed = " or ".join(str(c) for c in counts)
        raise ValueError(f"the {key.value} condition takes {allowed} number(s), not {value!r}")
    return [float(v) for v in numbers]


def same_condition(value: ConfigValue, nominal: ConfigValue) -> bool:
    """Whether a value is the nominal one, the shorter padded with zeros: [0, 0] is
    [0, 0, 0] for `placement` and 0.0 is [0.0, 0.0] for `lighting`, so a value written
    in the older form and one in the extended form mean the same."""

    def numbers(v: ConfigValue) -> list[float] | None:
        values: list[object] = list(v) if isinstance(v, list) else [v]
        if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in values):
            return None
        return [float(x) for x in values]  # type: ignore[arg-type]

    a, b = numbers(value), numbers(nominal)
    if a is None or b is None:
        return value == nominal
    width = max(len(a), len(b))
    return a + [0.0] * (width - len(a)) == b + [0.0] * (width - len(b))


@register(CONDITION_CHANGE_KIND)
class ConditionChange(FaultTypeBase):
    """A site condition changed to `params.value` by an action (dispatches on the condition)."""

    kind = CONDITION_CHANGE_KIND
    fault_class = FaultClass.site
    Params = ValueParams
    dose_levels = (0.0, 1.0)
    truth = condition_truth()

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        key = spec.target.condition_key
        if key is None:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) names no condition")
        value = ValueParams.model_validate(spec.params).value
        state = ConditionState(spec=spec, episode_id=identity.episode_id)
        if same_condition(value, spec.nominal):
            return state
        if key is ConditionKey.lighting:
            stops, warmth = [*_numbers(key, value, 1, 2), 0.0][:2]
            state.value = [stops, warmth]
        elif key is ConditionKey.object_batch:
            state.value = _numbers(key, value, 2)
        elif key is ConditionKey.placement:
            dx, dy, dz = [*_numbers(key, value, 2, 3), 0.0][:3]
            state.value = [dx, dy, dz]
            state.shift = ShiftState(
                spec=spec,
                episode_id=identity.episode_id,
                offset=np.array([dx, dy, dz]) / 100.0,
            )
        elif key is BACKDROP_CONDITION:
            blend = _numbers(key, value, 1)[0]
            if not 0.0 <= blend <= 1.0:
                raise ValueError(f"the {key.value} condition is a blend from 0 to 1, not {blend}")
            state.value = blend
        else:
            raise ValueError(f"no mechanism is defined for the {key.value} condition")
        state.key = key
        return state

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        assert isinstance(state, ConditionState)
        if state.key is ConditionKey.lighting:
            stops, warmth = state.value
            undos = [u for u in (change_light(handle, stops), change_colour(handle, warmth)) if u]
            if not undos:
                return None

            def undo_both() -> None:
                for undo in reversed(undos):
                    undo()

            return undo_both
        if state.key is ConditionKey.object_batch:
            return change_batch(handle, state.value[0], state.value[1])
        if state.key is ConditionKey.placement:
            assert state.shift is not None
            return move_robot_and_camera(handle, -state.shift.offset)
        if state.key is BACKDROP_CONDITION:
            backdrop = BackdropParams()
            return repaint(handle, list(backdrop.surfaces), list(backdrop.tint), state.value)
        return None

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        return ValueParams.model_validate(spec.params).value

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, ConditionState)
        if state.shift is None:
            return observation
        return FurnitureShift().observation_hook(observation, state.shift)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, ConditionState)
        if state.shift is None:
            return action
        return FurnitureShift().action_hook(action, state.shift)

    def to_policy_frame(
        self, action: Action7, state: FaultState, control_mode: ControlMode
    ) -> Action7:
        assert isinstance(state, ConditionState)
        if state.shift is None:
            return action
        return FurnitureShift().to_policy_frame(action, state.shift, control_mode)
