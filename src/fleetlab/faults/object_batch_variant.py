"""A batch of objects heavier or slipperier than the one the cell was set up with:
a physical fault on the site's `object_batch` condition, as a cell hook.

Every movable object of the scene changes: its mass and inertia double
`level * params.mass` times, its sliding friction halves `level *
params.friction` times. LIBERO's objects are light (5.6 g for a bowl, 10 to
32 g for the cartons), so the levels are doublings. A slipperier object gets
priority in its contacts, because MuJoCo otherwise uses the larger friction
of the two touching geoms and the gripper pads' 2.0 would hide any change
(spikes/teleop-expert, which found the same for the pads). The constants
MuJoCo derives from the masses for its contact solver (`body_invweight0`,
`dof_invweight0`, `dof_M0`, `body_subtreemass`) are scaled with them, which
for a free body is exactly what `mj_setConst` computes, without
`mj_setConst`'s rounding changes everywhere else. `change_condition
object_batch` back to the nominal batch removes the fault. The experimental
design calls the type `object_batch`, which is the condition's own name, so the
kind is `object_batch_variant`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import ConditionKey, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import (
    DEGRADED,
    SITE,
    DosePoint,
    FaultParams,
    ModelPatch,
    condition_truth,
    default_points,
    handle_env,
    mj_id,
    subtree,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo
    from fleetlab.contracts.identity import EpisodeIdentity

# The value of the `object_batch` condition: [mass doublings, friction halvings].
NOMINAL_BATCH: list[float] = [0.0, 0.0]


class BatchParams(FaultParams):
    """How the level splits between mass (doublings) and friction (halvings).

    `prioritise` gives the objects' geoms contact priority even when their
    friction does not change: the calibration's control point, which measures
    what the priority alone does (the objects' solref and solimp then govern
    their contacts, the pads' 0.01 and 0.5 among them).
    """

    mass: float = Field(default=1.0, ge=0.0)
    friction: float = Field(default=0.0, ge=0.0)
    prioritise: bool = False


@register("object_batch_variant")
class ObjectBatchVariant(FaultTypeBase):
    """Heavier or slipperier objects at a site (cell hook on every movable object).

    Level: doublings of mass and halvings of friction, weighted by the params
    (default: heavier only; level 4 makes every object 16 times as heavy).
    """

    kind = "object_batch_variant"
    fault_class = FaultClass.physical
    Params = BatchParams
    dose_levels = (0.0, 2.0, 4.0, 6.0)
    # A physical fault is diagnosed as degraded; a batch also "changed".
    truth = condition_truth(DEGRADED | SITE)
    calibration: tuple[DosePoint, ...] = (
        *default_points(
            (2.0, 4.0, 6.0),
            ConditionKey.object_batch.value,
            nominal=NOMINAL_BATCH,
            unit=" mass doublings",
        ),
        DosePoint(
            "2 friction halvings",
            2.0,
            {"mass": 0.0, "friction": 1.0},
            ConditionKey.object_batch.value,
            NOMINAL_BATCH,
        ),
        DosePoint(
            "contact priority only (control)",
            1.0,
            {"mass": 0.0, "friction": 0.0, "prioritise": True},
            ConditionKey.object_batch.value,
            NOMINAL_BATCH,
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.condition_key is not ConditionKey.object_batch:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        params = BatchParams.model_validate(state.spec.params)
        return change_batch(
            handle,
            state.spec.level * params.mass,
            state.spec.level * params.friction,
            prioritise=params.prioritise,
        )

    @staticmethod
    def condition_value(spec: FaultSpec) -> ConfigValue:
        params = BatchParams.model_validate(spec.params)
        return [spec.level * params.mass, spec.level * params.friction]


def movable_roots(handle: CellHandle) -> list[int]:
    """The root bodies of the scene's movable objects (LIBERO's `objects_dict`)."""
    env: Any = handle_env(handle)
    return [mj_id(handle.model, "body", obj.root_body) for obj in env.objects_dict.values()]


def change_batch(
    handle: CellHandle, mass_doublings: float, friction_halvings: float, prioritise: bool = False
) -> Undo | None:
    """Scale every movable object's mass and friction; return the exact undo, or None.

    A friction change gives the objects' geoms contact priority, and so does
    `prioritise` alone.
    """
    if mass_doublings == 0.0 and friction_halvings == 0.0 and not prioritise:
        return None
    model = handle.model
    patch = ModelPatch()
    factor = 2.0**mass_doublings
    slip = 2.0**-friction_halvings
    priority = int(np.max(model.geom_priority)) + 1 if model.ngeom else 1
    added: dict[int, float] = {}  # extra mass in the subtree of each ancestor of an object
    for root in movable_roots(handle):
        bodies = subtree(model, root)
        if factor != 1.0:
            extra = float(sum(model.body_mass[b] for b in bodies)) * (factor - 1.0)
            ancestor = root
            while ancestor != 0:
                ancestor = int(model.body_parentid[ancestor])
                added[ancestor] = added.get(ancestor, 0.0) + extra
            for body in bodies:
                patch.set(model, "body_mass", body, model.body_mass[body] * factor)
                patch.set(model, "body_inertia", body, model.body_inertia[body] * factor)
                patch.set(model, "body_invweight0", body, model.body_invweight0[body] / factor)
                patch.set(model, "body_subtreemass", body, model.body_subtreemass[body] * factor)
            for dof in range(int(model.nv)):
                if int(model.dof_bodyid[dof]) in bodies:
                    patch.set(model, "dof_invweight0", dof, model.dof_invweight0[dof] / factor)
                    patch.set(model, "dof_M0", dof, model.dof_M0[dof] * factor)
        if slip != 1.0 or prioritise:
            for geom in range(int(model.ngeom)):
                touches = model.geom_contype[geom] or model.geom_conaffinity[geom]
                if int(model.geom_bodyid[geom]) in bodies and touches:
                    friction = model.geom_friction[geom].copy()
                    friction[0] *= slip
                    patch.set(model, "geom_friction", geom, friction)
                    patch.set(model, "geom_priority", geom, priority)
    for ancestor, extra in added.items():
        patch.set(model, "body_subtreemass", ancestor, model.body_subtreemass[ancestor] + extra)
    return patch.undo
