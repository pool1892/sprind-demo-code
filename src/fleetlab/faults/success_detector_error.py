"""The fleet's success detector records the wrong outcome: a data fault, as a data hook.

In a share of episodes (the level, 0 to 1) the detector on the robot's
controller gets the outcome wrong when the episode ends: it misses a success
(`mode="missed"`, the default), or it records a success on a verdict that
ends the episode as a failure (`mode="false_success"`), or either
(`"both"`). Today the fleet's check ends an episode as a failure only when it
gives up on a placement that LIBERO's own check had seen ("LIBERO only"); any
other done failure a later check gives would be flipped the same way.
The data hook changes only the `success` of a verdict that is `done` and
never `done` itself (contracts, section 7, `check_data_hook`), so the episode
ends on the same step and the robot does exactly what it does without the
fault: the oracle gain is zero by construction and the fault is scored on
the logged outcome. Which episodes are
affected is drawn once per episode from the episode's own draw for this
fault, so the episodes affected at a lower level are among those affected at
a higher one. `service` or `replace` of the controller removes the fault.

A false success on an attempt that failed in some other way (dropped, never
grasped, out of time) would need a verdict that is done there, which the
check does not give before the step limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fleetlab.contracts.env import SuccessVerdict
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    DosePoint,
    FaultParams,
    component_truth,
    default_points,
    episode_uniform,
)

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity

MISSED_REASON = "the goal was not registered"
FALSE_REASON = "the goal was registered"


class DetectorParams(FaultParams):
    mode: Literal["missed", "false_success", "both"] = "missed"


@dataclass
class DetectorState(FaultState):
    wrong: bool = False
    mode: str = "missed"


@register("success_detector_error")
class SuccessDetectorError(FaultTypeBase):
    """A share of episodes ends with the wrong outcome logged (data hook). Level: that share."""

    kind = "success_detector_error"
    fault_class = FaultClass.data
    Params = DetectorParams
    dose_levels = (0.0, 0.1, 0.2, 0.4)
    truth = component_truth({Verb.service, Verb.replace_})
    calibration: tuple[DosePoint, ...] = (
        *default_points((0.2, 0.4), Component.controller.value, unit=" missed"),
        *default_points(
            (0.5,), Component.controller.value, {"mode": "false_success"}, unit=" false successes"
        ),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.controller:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 1.0:
            raise ValueError(f"fault {spec.fault_id}: a share is at most 1, not {spec.level}")
        wrong = episode_uniform(identity, spec, "detector") < spec.level
        mode = DetectorParams.model_validate(spec.params).mode
        return DetectorState(spec=spec, episode_id=identity.episode_id, wrong=wrong, mode=mode)

    def data_hook(self, verdict: SuccessVerdict, state: FaultState) -> SuccessVerdict:
        assert isinstance(state, DetectorState)
        if not (state.wrong and verdict.done):
            return verdict
        if verdict.success and state.mode in ("missed", "both"):
            return SuccessVerdict(success=False, done=True, reason=MISSED_REASON)
        if not verdict.success and state.mode in ("false_success", "both"):
            return SuccessVerdict(success=True, done=True, reason=FALSE_REASON)
        return verdict
