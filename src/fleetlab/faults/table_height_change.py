"""The table, with everything on it, lower or higher than the cell was set up for:
a site fault on the `placement` condition, as a cell, an observation and an action hook.

A table whose levelling feet were turned, or a replacement table of another
height, stands `level` centimetres lower (or higher, `raised`) than the
robot and the scene camera expect. The mechanism is `furniture_shift`'s in
the vertical: the robot's base and the scene camera move the other way in
the world (cell hook), the robot reports its end effector in its own frame
(observation hook), and absolute targets land in that frame (action hook),
so the policy sees and reaches exactly as if the table had moved. A lower
table puts every object out of the reach the policy learned by that much
in height; a higher one brings the arm closer to the table top.

The value of the `placement` condition becomes [x, y, z] in centimetres
(z up); `condition_change` takes two or three numbers, so a value the world
generator wrote as [x, y] still means the same. `change_condition placement`
back to the nominal value removes the fault.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState
from fleetlab.contracts.vocabulary import ConditionKey
from fleetlab.faults import register
from fleetlab.faults._common import DosePoint, FaultParams, default_points
from fleetlab.faults.furniture_shift import NOMINAL_PLACEMENT, FurnitureShift, ShiftState

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity


class HeightParams(FaultParams):
    """Whether the table stands higher (`raised`) or lower than set up."""

    raised: bool = False


@register("table_height_change")
class TableHeightChange(FurnitureShift):
    """The table lower (or higher) by `level` centimetres. Level: centimetres."""

    kind = "table_height_change"
    Params = HeightParams  # type: ignore[assignment]
    dose_levels = (0.0, 1.0, 2.0, 4.0)
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (1.0, 2.0, 4.0),
            ConditionKey.placement.value,
            nominal=NOMINAL_PLACEMENT,
            unit=" cm lower",
        ),
        *default_points(
            (2.0,),
            ConditionKey.placement.value,
            {"raised": True},
            NOMINAL_PLACEMENT,
            unit=" cm higher",
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.condition_key is not ConditionKey.placement:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 10.0:
            raise ValueError(f"fault {spec.fault_id}: {spec.level} cm is beyond any tolerance")
        return ShiftState(
            spec=spec,
            episode_id=identity.episode_id,
            offset=np.array([0.0, 0.0, height_cm(spec)]) / 100.0,
        )

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        return [0.0, 0.0, height_cm(spec)]


def height_cm(spec: FaultSpec) -> float:
    """The table's height against the cell as set up, in centimetres (negative: lower)."""
    raised = HeightParams.model_validate(spec.params).raised
    return float(spec.level if raised else -spec.level)
