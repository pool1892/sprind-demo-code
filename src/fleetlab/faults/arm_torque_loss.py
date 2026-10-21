"""Arm drives that deliver less torque than commanded: a physical fault, as a cell hook.

The arm's joint drives have weakened (motors run hot and are derated, a
magnet has lost strength, a gearbox has lost efficiency): every joint
delivers `1 - level` of the torque the controller commands. The controller
does not know: its gravity compensation and its pull toward the target fall
short together, so the arm sags under its own weight and holds short of its
target and low, most where the arm reaches out. The commanded target poses
in the log are what they were; the measured end-effector pose falls short of
them, which is the signature in the logs. `service` or `replace` of the arm
removes the fault; `recalibrate` does not, because nothing is out of
calibration. Measured on LIBERO (`libero_spatial` task 4): with the hand
held at its start pose it settles 1.0, 2.5, 6.0 and 9.8 mm low (and about as
far short toward the base) with 0.05, 0.1, 0.2 and 0.3 of the torque lost;
on a reach 15 cm forward and 10 cm down, 3, 8 and 13 mm low with 0.1, 0.2
and 0.3, and 9 cm with 0.5, where the arm cannot hold the reach at all (the
independent review's probe). The calibration runs 0.1 to 0.4: a millimetre
at 0.05 is below the policy's own scatter, and 0.5 is total.

The torque is scaled through each arm motor's gain (`actuator_gainprm`), which
MuJoCo applies at every step, so the cell hook takes effect for the episode;
robosuite's torque limits are unchanged. The fleet's health checks have no
reading for it (`arm.joint_torque_rms` varies benignly only): a real fleet
would see the drives work harder, and that is left out of the health model
on purpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    ARM_MOTORS,
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


@register("arm_torque_loss")
class ArmTorqueLoss(FaultTypeBase):
    """Every arm joint delivers `1 - level` of its commanded torque (cell hook).

    Level: the share of torque lost, 0 to 0.9.
    """

    kind = "arm_torque_loss"
    fault_class = FaultClass.physical
    Params = NoParams
    dose_levels = (0.0, 0.1, 0.2, 0.3, 0.4)
    truth = component_truth({Verb.service, Verb.replace_})
    calibration: tuple[DosePoint, ...] = default_points(
        (0.1, 0.2, 0.3, 0.4), Component.arm.value, unit=" torque lost"
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.arm:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 0.9:
            raise ValueError(f"fault {spec.fault_id}: at most 0.9 of the torque is lost")
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        keep = 1.0 - state.spec.level
        model = handle.model
        patch = ModelPatch()
        for name in ARM_MOTORS:
            motor = mj_id(model, "actuator", name)
            gain = model.actuator_gainprm[motor].copy()
            gain[0] *= keep
            patch.set(model, "actuator_gainprm", motor, gain)
        return patch.undo
