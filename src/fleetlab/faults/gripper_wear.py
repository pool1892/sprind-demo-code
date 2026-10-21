"""A worn gripper: a physical fault, as a cell hook on the gripper's pads and fingers.

Two mechanisms, both scaled by the level (the share of the gripper's wear
allowance used up, 0 to 1):

- The finger pads lose grip: their sliding friction falls by `friction_loss`
  at full wear (2.0 when new; MuJoCo uses the larger friction of two touching
  geoms, so the pads govern down to the object's own, 0.95 for LIBERO's bowls).
- The fingers no longer close fully: the closed position of each finger's
  position actuator moves outward, so the closed gripper leaves a gap of
  `closed_gap_mm` at full wear (`fingers="one"`: one finger takes it all, a
  bent finger). The fingers press with a stiffness of 1000 N/m, so on a thin
  part such as a bowl's 2.6 mm rim the grip force falls with the gap and is
  gone once the gap reaches the rim; on a thick object nothing changes until
  the gap is wider than the object (spikes/teleop-expert, surprise 7).

robosuite builds the gripper controller at every reset from the actuator's
control range, so the cell hook, which runs before the reset, takes effect for
the episode. `replace` or `service` of the gripper removes the fault.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    FINGER_ACTUATORS,
    PAD_GEOMS,
    DosePoint,
    FaultParams,
    ModelPatch,
    component_truth,
    default_points,
    mj_id,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo
    from fleetlab.contracts.identity import EpisodeIdentity


class WearParams(FaultParams):
    """What full wear (level 1) does: pad friction lost, closed gap, which fingers."""

    friction_loss: float = Field(default=0.5, ge=0.0, lt=1.0)
    closed_gap_mm: float = Field(default=3.0, ge=0.0, le=40.0)
    fingers: Literal["both", "one"] = "both"


@register("gripper_wear")
class GripperWear(FaultTypeBase):
    """Worn finger pads and fingers that no longer close fully (cell hook).

    Level: the share of the wear allowance used up, 0 to 1. With the default
    params, level 0.5 leaves a closed gap of 1.5 mm and three quarters of the
    pads' friction.
    """

    kind = "gripper_wear"
    fault_class = FaultClass.physical
    Params = WearParams
    dose_levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    truth = component_truth({Verb.replace_, Verb.service})
    calibration: tuple[DosePoint, ...] = default_points(
        (0.25, 0.5, 0.75, 1.0), Component.gripper.value, unit=" wear"
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.gripper:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 1.0:
            raise ValueError(f"fault {spec.fault_id}: wear is at most 1, not {spec.level}")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        params = WearParams.model_validate(state.spec.params)
        model = handle.model
        wear = state.spec.level
        patch = ModelPatch()
        for name in PAD_GEOMS:
            geom = mj_id(model, "geom", name)
            friction = model.geom_friction[geom].copy()
            friction[0] *= 1.0 - params.friction_loss * wear
            patch.set(model, "geom_friction", geom, friction)
        gap = params.closed_gap_mm / 1000.0 * wear
        shares = (0.5, 0.5) if params.fingers == "both" else (1.0, 0.0)
        for name, share in zip(FINGER_ACTUATORS, shares, strict=True):
            if share == 0.0:
                continue
            actuator = mj_id(model, "actuator", name)
            low, high = (float(x) for x in model.actuator_ctrlrange[actuator])
            # A finger closes toward the end of its range nearer zero.
            closes_low = abs(low) <= abs(high)
            new = (low + share * gap, high) if closes_low else (low, high - share * gap)
            patch.set(model, "actuator_ctrlrange", actuator, new)
        return patch.undo

    @staticmethod
    def health_shift(spec: FaultSpec) -> dict[str, float]:
        """The share of its grip force the gripper loses on the fleet's reference grip check.

        The check grips a 4 mm test plate: the force falls with the closed gap
        (1000 N/m on each finger) and with the pads' friction. Noise and
        benign variation are the world generator's to add.
        """
        params = WearParams.model_validate(spec.params)
        plate = 4.0
        gap = params.closed_gap_mm * spec.level
        squeeze = max(0.0, plate - gap) / plate
        friction = 1.0 - params.friction_loss * spec.level
        return {"grip_force_loss": float(1.0 - squeeze * friction)}
