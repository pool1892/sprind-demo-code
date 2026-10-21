"""The work cell's surroundings repainted: a site fault, as a cell hook on textures.

The walls and the floor (or, with `surfaces`, the table top) take another
colour at the same brightness: each texel's luminance is kept and its hue
replaced by `tint`, blended with the original by the level (0 to 1). The
changed textures are uploaded to the renderer, and uploaded again on undo
(spikes/replay section 2). In LIBERO's table scenes the walls fill about a
quarter of the scene camera's image and the floor under 1%; the table top
fills about half of both cameras' images. Expected to be a decoy for X-VLA
with the walls and floor, which the calibration checks.

The vocabulary has no condition for the surroundings. Until it has one,
the fault sits on the `clutter` condition, the nearest: what
stands or hangs around the work that has nothing to do with it.
The experimental design calls the type `background`; the kind is `backdrop_change`,
a word the export's tables and skills do not use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConditionKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import (
    SURFACE_MATERIALS,
    DosePoint,
    FaultParams,
    ModelPatch,
    condition_truth,
    default_points,
    mj_id,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo
    from fleetlab.contracts.identity import EpisodeIdentity

# The condition the fault sits on, and its value: the blend toward the new colour.
BACKDROP_CONDITION = ConditionKey.clutter
NOMINAL_BACKDROP = 0.0
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


Surface = Literal["walls", "floor", "table"]
DEFAULT_SURFACES: tuple[Surface, ...] = ("walls", "floor")


class BackdropParams(FaultParams):
    """Which surfaces change and to which hue (RGB, 0 to 1)."""

    surfaces: list[Surface] = Field(default_factory=lambda: list(DEFAULT_SURFACES), min_length=1)
    tint: list[float] = Field(
        default_factory=lambda: [0.25, 0.45, 0.85], min_length=3, max_length=3
    )


@register("backdrop_change")
class BackdropChange(FaultTypeBase):
    """Walls and floor (by default) in another colour, blended by the level (cell hook)."""

    kind = "backdrop_change"
    fault_class = FaultClass.site
    Params = BackdropParams
    dose_levels = (0.0, 0.5, 1.0)
    truth = condition_truth()
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (0.5, 1.0), BACKDROP_CONDITION.value, nominal=NOMINAL_BACKDROP, unit=" walls and floor"
        ),
        *default_points(
            (1.0,),
            BACKDROP_CONDITION.value,
            {"surfaces": ["table"]},
            NOMINAL_BACKDROP,
            unit=" table top",
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.condition_key is not BACKDROP_CONDITION:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 1.0:
            raise ValueError(f"fault {spec.fault_id}: the blend is at most 1, not {spec.level}")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        params = BackdropParams.model_validate(state.spec.params)
        return repaint(handle, params.surfaces, params.tint, state.spec.level)

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        return float(spec.level)


def surface_textures(model: Any, surfaces: Sequence[str]) -> list[int]:
    """The texture ids of the given surfaces' materials, each once, in order."""
    out: list[int] = []
    for surface in surfaces:
        for material in SURFACE_MATERIALS[surface]:
            row = model.mat_texid[mj_id(model, "material", material)]
            for tex in np.atleast_1d(row):
                if int(tex) >= 0 and int(tex) not in out:
                    out.append(int(tex))
    return out


def repaint(
    handle: CellHandle, surfaces: Sequence[str], tint: Sequence[float], blend: float
) -> Undo | None:
    """Blend the surfaces' textures toward `tint` at their own luminance; return the exact undo."""
    if blend == 0.0:
        return None
    model = handle.model
    patch = ModelPatch(handle.upload_texture)
    hue = np.asarray(tint, dtype=np.float32)
    hue = hue / max(float(hue @ _LUMA), 1e-3)  # a tint of luminance one
    for tex in surface_textures(model, surfaces):
        channels = int(model.tex_nchannel[tex])
        if channels < 3:
            continue
        start = int(model.tex_adr[tex])
        size = int(model.tex_width[tex]) * int(model.tex_height[tex]) * channels
        texels = model.tex_data[start : start + size].reshape(-1, channels).astype(np.float32)
        luma = texels[:, :3] @ _LUMA
        target = np.clip(luma[:, None] * hue[None, :], 0.0, 255.0)
        texels[:, :3] = (1.0 - blend) * texels[:, :3] + blend * target
        new = np.clip(np.rint(texels), 0, 255).astype(np.uint8).reshape(-1)
        patch.set(model, "tex_data", slice(start, start + size), new)
        patch.texture(tex)
    return None if patch.empty else patch.undo
