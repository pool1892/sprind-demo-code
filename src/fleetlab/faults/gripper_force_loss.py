"""A gripper that has lost force: a physical fault, as a cell hook on the finger actuators.

The gripper's drive has weakened (a worn gear train, a motor that runs hot
and is derated, a tired spring): each finger's position servo is softer and
its force limit lower, both by the level, the share of the force lost (0 to
1). LIBERO's Panda fingers are position servos of 1000 N/m with a limit of
20 N, on slide joints with heavy damping (100 N s/m) and 1 N of friction of
their own. The servo must overcome that friction, so a softer one closes
more slowly and stalls further from closed: on nothing, the closed gripper
leaves about 1 mm healthy, 3.3 mm at a loss of 0.5, and at 0.9 it takes
seconds to close at all (measured in `make check-sim`). On a bowl's 2.6 mm
rim it presses with less force, or, once the stall is wider than the rim,
not at all; on a thick object the lower force limit sets the grip. A heavy
or slippery object therefore slides out sooner: this is one half of the
library's interaction with `object_batch_variant`. Unlike `gripper_wear`,
nothing is out of shape: the pads keep their friction and the fingers their
range, and the grip check reads the lost force.

MuJoCo reads the actuators at every step, so the cell hook, which runs before
the reset, takes effect for the episode. `replace` or `service` of the
gripper removes the fault.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    FINGER_ACTUATORS,
    DosePoint,
    ModelPatch,
    NoParams,
    component_truth,
    default_points,
    mj_id,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo
    from fleetlab.contracts.identity import EpisodeIdentity


@register("gripper_force_loss")
class GripperForceLoss(FaultTypeBase):
    """The fingers' servo stiffness and force limit scaled by 1 - level (cell hook).

    Level: the share of the gripper's force lost, 0 to 1 (below 1: a gripper
    with no force at all is a dead gripper, not a weak one).
    """

    kind = "gripper_force_loss"
    fault_class = FaultClass.physical
    Params = NoParams
    dose_levels = (0.0, 0.25, 0.5, 0.75)
    truth = component_truth({Verb.replace_, Verb.service})
    calibration: tuple[DosePoint, ...] = default_points(
        (0.25, 0.5, 0.75), Component.gripper.value, unit=" force lost"
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.gripper:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level >= 1.0:
            raise ValueError(f"fault {spec.fault_id}: the share of force lost is below 1")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        keep = 1.0 - state.spec.level
        model = handle.model
        patch = ModelPatch()
        for name in FINGER_ACTUATORS:
            actuator = mj_id(model, "actuator", name)
            gain = model.actuator_gainprm[actuator].copy()
            bias = model.actuator_biasprm[actuator].copy()
            gain[0] *= keep  # force = kp * ctrl - kp * qpos: both terms scale
            bias[1] *= keep
            patch.set(model, "actuator_gainprm", actuator, gain)
            patch.set(model, "actuator_biasprm", actuator, bias)
            limit = model.actuator_forcerange[actuator] * keep
            patch.set(model, "actuator_forcerange", actuator, limit)
        return patch.undo

    @staticmethod
    def health_shift(spec: FaultSpec) -> dict[str, float]:
        """The share of its grip force the gripper loses on the fleet's grip check."""
        return {"grip_force_loss": float(spec.level)}
