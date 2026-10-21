"""The site's light dimmer or brighter: a site fault on the `lighting` condition, as a cell hook.

Every light of the scene and the renderer's headlight change their ambient,
diffuse and specular intensity by `2 ** level` (brighter) or `2 ** -level`
(dimmer): the level is in photographic stops. Expected to be a decoy for
X-VLA, which the calibration checks.
`change_condition lighting` back to the nominal value removes it.
The experimental design calls the type `lighting`, which is the condition's own name,
so the kind is `illumination_change`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo
    from fleetlab.contracts.identity import EpisodeIdentity

# The value of the `lighting` condition: stops against the cell as set up (signed).
NOMINAL_LIGHTING = 0.0


class LightParams(FaultParams):
    brighter: bool = False


@register("illumination_change")
class IlluminationChange(FaultTypeBase):
    """The lights dimmed or brightened by `level` stops (cell hook)."""

    kind = "illumination_change"
    fault_class = FaultClass.site
    Params = LightParams
    dose_levels = (0.0, 0.5, 1.0, 2.0)
    truth = condition_truth()
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (1.0, 2.0), ConditionKey.lighting.value, nominal=NOMINAL_LIGHTING, unit=" stops dimmer"
        ),
        *default_points(
            (1.0,),
            ConditionKey.lighting.value,
            {"brighter": True},
            NOMINAL_LIGHTING,
            unit=" stop brighter",
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.condition_key is not ConditionKey.lighting:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        stops = self.condition_value(state.spec)
        assert isinstance(stops, float)
        return change_light(handle, stops)

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        brighter = LightParams.model_validate(spec.params).brighter
        return float(spec.level if brighter else -spec.level)


def change_light(handle: CellHandle, stops: float) -> Undo | None:
    """Scale every light and the headlight by `2 ** stops`; return the exact undo, or None."""
    if stops == 0.0:
        return None
    model = handle.model
    factor = 2.0**stops
    patch = ModelPatch()
    for name in ("light_ambient", "light_diffuse", "light_specular"):
        patch.set(model, name, slice(None), getattr(model, name) * factor)
    headlight = model.vis.headlight
    for name in ("ambient", "diffuse", "specular"):
        patch.set(headlight, name, slice(None), getattr(headlight, name) * factor)
    return patch.undo
