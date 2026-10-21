"""The site's lamps replaced by lamps of another colour: a site fault on the `lighting`
condition, as a cell hook on the lights.

The cell was set up under daylight-white lamps (about 6500 K). Warm lamps
(about 3000 K, a common LED or halogen replacement) light it yellow-orange,
cool ones (about 10000 K) blue. Every light of the scene and the renderer's
headlight have their ambient, diffuse and specular colour multiplied per
channel by `1 + level * (tint - 1)`, where the tint is the lamp's colour
relative to the old one at the same luminance; the level is the share of
the way to that lamp, 1 for the full replacement. The brightness stays
about the same, so this is the colour half of what `illumination_change`
does with brightness, and a candidate decoy like it, which the
calibration checks.

The value of the `lighting` condition becomes [stops, warmth], warmth signed
(positive warmer, negative cooler); `condition_change` takes a number
(stops) or the pair. `change_condition lighting` back to the nominal value
removes the fault.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConditionKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import (
    DosePoint,
    FaultParams,
    ModelPatch,
    condition_truth,
    default_points,
)
from fleetlab.faults.illumination_change import NOMINAL_LIGHTING

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo
    from fleetlab.contracts.identity import EpisodeIdentity

# The lamps' colours against the cell's daylight-white ones, normalised to the same
# luminance (sRGB of a black body against D65, rounded): warm about 3000 K, cool 10000 K.
_LUMA = np.array([0.299, 0.587, 0.114])
_WARM = np.array([1.0, 0.71, 0.42])
_COOL = np.array([0.80, 0.86, 1.0])
WARM_TINT = _WARM / float(_WARM @ _LUMA)
COOL_TINT = _COOL / float(_COOL @ _LUMA)


class ColourParams(FaultParams):
    """Whether the new lamps are cooler (`cooler`) than the old ones; warmer by default."""

    cooler: bool = False


@register("light_color_shift")
class LightColorShift(FaultTypeBase):
    """The lights' colour moved toward warm (or cool) lamps by the level (cell hook).

    Level: the share of the way to the new lamps' colour, 0 to 1.
    """

    kind = "light_color_shift"
    fault_class = FaultClass.site
    Params = ColourParams
    dose_levels = (0.0, 0.5, 1.0)
    truth = condition_truth()
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (0.5, 1.0), ConditionKey.lighting.value, nominal=NOMINAL_LIGHTING, unit=" warmer"
        ),
        *default_points(
            (1.0,),
            ConditionKey.lighting.value,
            {"cooler": True},
            NOMINAL_LIGHTING,
            unit=" cooler",
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.condition_key is not ConditionKey.lighting:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 1.0:
            raise ValueError(f"fault {spec.fault_id}: the share of the way is at most 1")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        return change_colour(handle, warmth(state.spec))

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        return [0.0, warmth(spec)]


def warmth(spec: FaultSpec) -> float:
    """The signed share of the way to warm (positive) or cool (negative) lamps."""
    cooler = ColourParams.model_validate(spec.params).cooler
    return float(-spec.level if cooler else spec.level)


def change_colour(handle: CellHandle, signed: float) -> Undo | None:
    """Tint every light and the headlight; return the exact undo, or None."""
    if signed == 0.0:
        return None
    tint = WARM_TINT if signed > 0 else COOL_TINT
    factor = 1.0 + abs(signed) * (tint - 1.0)
    model = handle.model
    patch = ModelPatch()
    for name in ("light_ambient", "light_diffuse", "light_specular"):
        patch.set(model, name, slice(None), getattr(model, name) * factor[None, :])
    headlight = model.vis.headlight
    for name in ("ambient", "diffuse", "specular"):
        patch.set(headlight, name, slice(None), getattr(headlight, name) * factor)
    return patch.undo
