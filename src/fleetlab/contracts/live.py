"""The live layer: risk estimator, supervisor and its decisions (docs/design/contracts.md,
section 14).

The robot holds still while the supervisor decides, counted as time, not as a
failed attempt (`hold_action`). Every intervention is physical, in the live
simulation: `retry` is a scripted recovery primitive (open the gripper, lift,
return to a pre-grasp pose, hand control back to the policy); `stop` ends the
attempt safely; `hand_over` creates a "continue" job for the teleoperator from
the current snapshot. A snapshot is never used to put the world back for the
robot. The decision trace is recorded on the truth side (`DecisionRecord`),
and a supervised episode replays in "scripted" mode from it.

Version 1 decisions are continue, retry, stop and hand_over; stage 5 adds
`recover` (release and retreat) and `degrade` (shorter executed chunks).

Choices where the note is silent: the rollout makes one risk estimator per
episode, so a stateful estimator starts fresh; `SupervisorContext` holds the
recent frames per camera, oldest first, and the operator queue as the number
of waiting hand-overs and of free operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import Field

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.env import Camera
from fleetlab.contracts.policy import ACTION_DIM, Action7, ControlMode


class Decision(StrEnum):
    """What the supervisor decides for a running episode, including stage 5."""

    continue_ = "continue"
    retry = "retry"
    stop = "stop"
    hand_over = "hand_over"
    recover = "recover"
    degrade = "degrade"


DECISION_DESCRIPTIONS: dict[Decision, str] = {
    Decision.continue_: "let the robot carry on",
    Decision.retry: "open the gripper, lift, return to a pre-grasp pose, hand back to the policy",
    Decision.stop: "end the attempt safely",
    Decision.hand_over: "hand the attempt to a teleoperator from where it stands",
    Decision.recover: "release an object already over its target, then retreat vertically",
    Decision.degrade: "discard queued actions and replan after at most four policy actions",
}

# Names reserved by the original contract and implemented in stage 5.
STAGE5_DECISIONS: tuple[str, ...] = ("recover", "degrade")


@dataclass(frozen=True)
class StepTelemetry:
    """One step as the fleet logs it: the visible `steps` columns, nothing from the truth side."""

    step: int
    time_s: float
    joint_pos: npt.NDArray[np.float64]
    joint_vel: npt.NDArray[np.float64]
    gripper_qpos: npt.NDArray[np.float64]
    eef_pos: npt.NDArray[np.float64]
    eef_quat: npt.NDArray[np.float64]
    commanded_action: npt.NDArray[np.float32]
    chunk_index: int
    chunk_step: int


@runtime_checkable
class RiskEstimator(Protocol):
    """Reads the telemetry step by step and estimates the chance of failure."""

    def update(self, telemetry: StepTelemetry) -> float:
        """The probability that the episode fails within the estimator's horizon."""
        ...


@dataclass(frozen=True)
class OperatorQueue:
    """The teleoperation queue as the supervisor sees it."""

    waiting: int
    free_operators: int


@dataclass(frozen=True)
class SupervisorContext:
    """What the supervisor gets when the risk estimator fires."""

    recent_frames: dict[Camera, list[npt.NDArray[np.uint8]]]
    telemetry_window: list[StepTelemetry]
    risk: float
    task_text: str
    options: list[Decision]
    operator_queue: OperatorQueue = field(default_factory=lambda: OperatorQueue(0, 0))
    # Physical control ticks excluding decision holds; defaults keep external adapters valid.
    active_step: int = 0
    previous_decisions: tuple[Decision, ...] = ()
    frame_steps: tuple[int, ...] = ()
    # Visible cycle budget; None means an external caller has not supplied it.
    remaining_active_steps: int | None = None


class SupervisorDecision(NamedTuple):
    """A decision with its rationale, the time the call took, and the model's own time.

    `decision_ms` is measured through the subscription route; `model_ms` is the
    model's own response time, the production estimate.
    """

    decision: Decision
    rationale: str
    decision_ms: float
    model_ms: float


@runtime_checkable
class Supervisor(Protocol):
    def decide(self, context: SupervisorContext) -> SupervisorDecision: ...


class SupervisorCallFailure(RuntimeError):
    """A measured call failure for the engine to record and physically apply as fallback.

    None is an unavailable model timer, never a fabricated zero. `called=False`
    means a disabled route or failed preflight prevented a model invocation.
    """

    def __init__(
        self, error: str, decision_ms: float, model_ms: float | None = None, *, called: bool = True
    ) -> None:
        super().__init__(error)
        self.decision_ms = decision_ms
        self.model_ms = model_ms
        self.called = called


class DecisionRecord(ContractModel):
    """One entry of a supervised episode's decision trace (truth side, `decisions` table)."""

    step: int = Field(ge=0)
    decision: Decision
    hold_steps: int = Field(ge=0)  # steps the robot held still while the supervisor decided
    decision_ms: float = Field(ge=0.0)
    model_ms: float | None = Field(ge=0.0)
    rationale: str = ""
    call_status: Literal["ok", "failed", "skipped", "rule"] = "ok"
    error: str | None = None
    # New retries replay the exact additional active budget granted at primitive start.
    retry_budget_steps: int | None = Field(default=None, ge=1)


def hold_action(control_mode: ControlMode, last_commanded: Action7) -> Action7:
    """The action that holds the robot still, per control mode, gripper command unchanged.

    In absolute control the last commanded pose is repeated; in relative
    control the motion is zero. The adapter's `NO_OP_ACTION` (gripper open) is
    not a hold: it would drop a held object.
    """
    last = np.asarray(last_commanded, dtype=np.float32)
    if last.shape != (ACTION_DIM,):
        raise ValueError(f"an action has {ACTION_DIM} entries, not shape {last.shape}")
    if control_mode is ControlMode.absolute:
        return last.copy()
    hold = np.zeros(ACTION_DIM, dtype=np.float32)
    hold[-1] = last[-1]
    return hold
