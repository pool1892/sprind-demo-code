"""The fleet's success detector reports successes that did not happen: a data fault,
as a data hook with an observation listener.

The detector on the robot's controller registers the goal as reached on a
share of the attempts that failed (the level, 0 to 1), when the robot went
through a placement: it closed the gripper and opened it again, as it does
when it sets an object down. A detector fooled by the motion of a placement
(a misaligned goal region, a weight sensor that reads the gripper's push, a
camera check that sees the object near the goal) is the kind of false
positive a fleet meets; an attempt that never closed the gripper is not
registered. The robot does exactly what it does without the fault, and the
episode ends on the same step (the data hook changes only the `success` of a
done verdict, contracts section 7): the oracle gain is zero by construction
and the fault is scored on the logged outcome. Its effect on the logged
success rate is the level times the share of eligible failures, so it shows
only where robots fail: it is calibrated on
top of a turned scene camera, and in a healthy cell it is all but invisible.

The observation hook only listens (it returns the observation unchanged):
it follows the measured finger opening to see a close and a reopen. Which
failures are misreported is drawn once per episode, so those at a lower
level are among those at a higher one. `service` or `replace` of the
controller removes the fault. This is the false-positive detector
that the miss-only `success_detector_error` left open. The two
should not sit on one robot's controller: a success the other misses
becomes a done failure, which this one would flip back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.contracts.env import SuccessVerdict
from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    DosePoint,
    NoParams,
    component_truth,
    episode_uniform,
)
from fleetlab.faults.success_detector_error import FALSE_REASON

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation
    from fleetlab.contracts.identity import EpisodeIdentity

# How far the finger opening (both fingers' travel added, m; about 0.078 fully open) must
# fall below the episode's first reading to count as a close, and rise again above its
# lowest since then to count as a reopen. A 5 cm carton held leaves about 0.05.
CLOSE_TRAVEL = 0.015
REOPEN_TRAVEL = 0.01
BASE = ("camera_pose_offset", "6 deg scene pan")


@dataclass
class FalsePositiveState(FaultState):
    chosen: bool = False  # this episode's failure, if eligible, is misreported
    first: float | None = None  # the finger opening at the episode's first step
    lowest: float | None = None  # the lowest opening since the gripper closed
    reopened: bool = False  # closed and opened again: a placement went through
    flipped: bool = False


@register("success_false_positive")
class SuccessFalsePositive(FaultTypeBase):
    """A share of failed placements logged as successes (observation listener and data hook).

    Level: the share of eligible failed attempts misreported, 0 to 1.
    """

    kind = "success_false_positive"
    fault_class = FaultClass.data
    Params = NoParams
    dose_levels = (0.0, 0.25, 0.5, 1.0)
    truth = component_truth({Verb.service, Verb.replace_})
    calibration: tuple[DosePoint, ...] = tuple(
        DosePoint(
            f"{level:g} on a 6 deg camera",
            level,
            {},
            Component.controller.value,
            None,
            (BASE,),
        )
        for level in (0.25, 0.5, 1.0)
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.controller:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 1.0:
            raise ValueError(f"fault {spec.fault_id}: a share is at most 1, not {spec.level}")
        chosen = episode_uniform(identity, spec, "false-positive") < spec.level
        return FalsePositiveState(spec=spec, episode_id=identity.episode_id, chosen=chosen)

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, FalsePositiveState)
        qpos = observation["robot_state"]["gripper"]["qpos"]
        opening = abs(float(qpos[0])) + abs(float(qpos[1]))
        if state.first is None:
            state.first = opening
        if state.lowest is None:
            if opening < state.first - CLOSE_TRAVEL:
                state.lowest = opening
        elif opening > state.lowest + REOPEN_TRAVEL:
            state.reopened = True
        else:
            state.lowest = min(state.lowest, opening)
        return observation

    def data_hook(self, verdict: SuccessVerdict, state: FaultState) -> SuccessVerdict:
        assert isinstance(state, FalsePositiveState)
        if verdict.done and not verdict.success and state.chosen and state.reopened:
            state.flipped = True
            return SuccessVerdict(success=True, done=True, reason=FALSE_REASON)
        return verdict
