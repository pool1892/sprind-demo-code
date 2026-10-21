"""Probes the analysis agent composes: a short scripted motion run on a robot without the
policy, with every control step's command and measured response recorded
(docs/design/loop.md, section 10).

The agent writes a probe from a small public vocabulary of motion primitives and asks
for it with the tool `request_probe` (`ProbeRequest`): its own name for the probe, the
question it answers, what the agent expects to see, the robots, the repetitions per
robot, and the program, a list of segments played in order. Each segment is one
primitive:

- `hold`: keep the current target for `duration_steps`;
- `step`: jump the target on `axis` to `amplitude_m` (an offset from the start
  position) at the segment's first step and keep it there;
- `ramp`: move the target on `axis` in a straight line from where it is to
  `amplitude_m`, reaching it at the segment's last step;
- `sine`: oscillate the target on `axis` around where it is, with amplitude
  `amplitude_m` and period `period_steps`; after the segment it is back where it was;
- `open_gripper`, `close_gripper`: command the gripper open or closed from the
  segment's first step on, the arm holding its target.

Targets are end-effector positions: an offset from the position the robot reports when
the probe starts, along the axes x, y and z of the frame in which the robot reports its
end effector (z points up), with the orientation held as it was at the start. The arm
takes end-effector targets (operational-space control), so a probe commands no joint
positions; it records the measured joint positions and velocities beside the measured
end effector. The gripper starts open. Every number is bounded (`PROBE_BOUNDS`): the
target never leaves a box of 5 cm around the start position on any axis, it moves no
faster than 0.5 m/s in a ramp or a sine, a sine is at most 5 Hz, a segment lasts at most
10 s and a probe at most 20 s. A probe touches nothing on purpose: it moves the hand a
few centimetres in free space and opens or closes the empty gripper.

Each probe episode is one robot attempt, charged to the round's experiment budget like
an experiment's episodes (refused requests are free). Probes run after the session
ends, like experiments, on the robot as it currently is, with every change the fleet
has adopted in force. The commands travel the same command path as the policy's, so the
recorded response is whatever the robot's controller, its hardware and the path between
them make of a known command. The results are visible tables: `probe_episodes` (one row
per probe episode, with its probe id), `probe_segments` (the program as it ran) and
`probe_steps` (one row per control step: the commanded target and gripper command, and
the measured end-effector position and orientation, joint positions and velocities, and
gripper opening). A probe has no task and no outcome; it never appears among the
episodes of the regular work or of experiments.

`program_trajectory` is the one interpretation of a program: the robot's player and the
validation use it, and the agent can reproduce every commanded target from the
`probe_segments` rows.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from pydantic import Field, StringConstraints, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.ids import ID_PATTERNS, EpisodeId, IdKind, RobotId, check_neutral

__all__ = [
    "PROBE_BOUNDS",
    "PROBE_CONTROL_HZ",
    "PROBE_MESSAGES",
    "PROBE_NAME_PATTERN",
    "PROBE_STATUS_MESSAGE",
    "ProbeAxis",
    "ProbeBounds",
    "ProbeId",
    "ProbeOutcome",
    "ProbePrimitive",
    "ProbeProgram",
    "ProbeReceipt",
    "ProbeRequest",
    "ProbeResult",
    "ProbeSegment",
    "QueuedProbe",
    "Trajectory",
    "program_trajectory",
]

# Control steps per second: a probe's durations and periods count these.
PROBE_CONTROL_HZ = 20


class ProbeBounds(NamedTuple):
    """The public safety bounds of a probe."""

    max_offset_m: float = 0.05  # the target stays within this of the start, per axis
    max_speed_m_s: float = 0.5  # the fastest a ramp or a sine moves the target
    min_period_steps: int = 4  # a sine's shortest period: 5 Hz
    max_segment_steps: int = 200  # 10 s
    max_probe_steps: int = 400  # 20 s for the whole program
    max_segments: int = 12
    max_robots: int = 4  # robots per request
    max_repeats: int = 5  # probe episodes per robot per request
    max_text: int = 600  # characters of the purpose and of the expectation


PROBE_BOUNDS = ProbeBounds()

PROBE_NAME_PATTERN = r"^[a-z][a-z0-9-]{2,39}$"
ProbeId = Annotated[
    str,
    StringConstraints(pattern=ID_PATTERNS[IdKind.probe]),
    Field(description="a probe, as request_probe numbered it"),
]


class ProbePrimitive(StrEnum):
    """The motion primitives a probe's program is made of."""

    hold = "hold"
    step = "step"
    ramp = "ramp"
    sine = "sine"
    open_gripper = "open_gripper"
    close_gripper = "close_gripper"


class ProbeAxis(StrEnum):
    """An axis of the frame the robot reports its end effector in (z up)."""

    x = "x"
    y = "y"
    z = "z"


_MOVES = frozenset({ProbePrimitive.step, ProbePrimitive.ramp, ProbePrimitive.sine})
_AXES = {ProbeAxis.x: 0, ProbeAxis.y: 1, ProbeAxis.z: 2}
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0


class ProbeSegment(ContractModel):
    """One primitive of a probe's program; fields a primitive does not use are null."""

    primitive: ProbePrimitive
    axis: ProbeAxis | None  # step, ramp and sine; null otherwise
    amplitude_m: float | None  # step, ramp: the offset reached; sine: the amplitude
    duration_steps: int = Field(ge=1, le=PROBE_BOUNDS.max_segment_steps)
    period_steps: int | None  # sine only, in control steps

    @model_validator(mode="after")
    def _fields_fit(self) -> ProbeSegment:
        b = PROBE_BOUNDS
        moves = self.primitive in _MOVES
        if moves and (self.axis is None or self.amplitude_m is None):
            raise ValueError(f"{self.primitive.value} needs an axis and an amplitude_m")
        if not moves and (self.axis is not None or self.amplitude_m is not None):
            raise ValueError(f"{self.primitive.value} takes no axis and no amplitude_m (null)")
        if self.amplitude_m is not None and (
            not math.isfinite(self.amplitude_m) or abs(self.amplitude_m) > b.max_offset_m
        ):
            raise ValueError(f"amplitude_m is at most {b.max_offset_m} m in size")
        if self.primitive is ProbePrimitive.sine:
            if self.period_steps is None:
                raise ValueError("a sine needs period_steps")
            if not b.min_period_steps <= self.period_steps <= b.max_segment_steps:
                raise ValueError(
                    f"period_steps is between {b.min_period_steps} and {b.max_segment_steps}"
                )
            assert self.amplitude_m is not None
            speed = 2 * math.pi * abs(self.amplitude_m) * PROBE_CONTROL_HZ / self.period_steps
            if speed > b.max_speed_m_s + 1e-12:
                raise ValueError(
                    f"this sine moves the target at up to {speed:.2f} m/s; at most "
                    f"{b.max_speed_m_s} m/s (a smaller amplitude or a longer period)"
                )
        elif self.period_steps is not None:
            raise ValueError(f"{self.primitive.value} takes no period_steps (null)")
        return self


class Trajectory(NamedTuple):
    """A program played out: per commanded step its segment, target offset and gripper."""

    segment: npt.NDArray[np.int32]  # (n,)
    offset: npt.NDArray[np.float64]  # (n, 3) metres from the start position
    gripper: npt.NDArray[np.float32]  # (n,) -1 open, 1 closed
    first_step: tuple[int, ...]  # each segment's first step
    base: tuple[tuple[float, float, float], ...]  # each segment's held offset at its start


def program_trajectory(segments: list[ProbeSegment] | tuple[ProbeSegment, ...]) -> Trajectory:
    """The commanded target offsets and gripper commands of a program, step by step.

    Step k of the probe commands `start + offset[k]` with the gripper at `gripper[k]`;
    the program has `sum(duration_steps)` commanded steps. Pure arithmetic in float64,
    the same in every process.
    """
    offset = np.zeros(3, np.float64)
    gripper = GRIPPER_OPEN
    seg_out: list[int] = []
    off_out: list[npt.NDArray[np.float64]] = []
    grip_out: list[float] = []
    first: list[int] = []
    bases: list[tuple[float, float, float]] = []
    for index, s in enumerate(segments):
        first.append(len(seg_out))
        bases.append((float(offset[0]), float(offset[1]), float(offset[2])))
        n = s.duration_steps
        if s.primitive is ProbePrimitive.open_gripper:
            gripper = GRIPPER_OPEN
        elif s.primitive is ProbePrimitive.close_gripper:
            gripper = GRIPPER_CLOSED
        axis = _AXES[s.axis] if s.axis is not None else 0
        base = float(offset[axis])
        for k in range(n):
            current = offset.copy()
            if s.primitive is ProbePrimitive.step:
                assert s.amplitude_m is not None
                current[axis] = s.amplitude_m
            elif s.primitive is ProbePrimitive.ramp:
                assert s.amplitude_m is not None
                current[axis] = base + (s.amplitude_m - base) * (k + 1) / n
            elif s.primitive is ProbePrimitive.sine:
                assert s.amplitude_m is not None and s.period_steps is not None
                current[axis] = base + s.amplitude_m * math.sin(2 * math.pi * k / s.period_steps)
            seg_out.append(index)
            off_out.append(current)
            grip_out.append(gripper)
        if s.primitive is ProbePrimitive.step or s.primitive is ProbePrimitive.ramp:
            assert s.amplitude_m is not None
            offset[axis] = s.amplitude_m
    return Trajectory(
        segment=np.asarray(seg_out, np.int32),
        offset=np.asarray(off_out, np.float64).reshape(-1, 3),
        gripper=np.asarray(grip_out, np.float32),
        first_step=tuple(first),
        base=tuple(bases),
    )


class ProbeProgram(ContractModel):
    """The motion a probe episode plays: its segments, in order (part of the episode's
    identity, so that a probe episode replays like any other)."""

    segments: list[ProbeSegment] = Field(min_length=1, max_length=PROBE_BOUNDS.max_segments)

    @model_validator(mode="after")
    def _bounded(self) -> ProbeProgram:
        b = PROBE_BOUNDS
        total = self.steps
        if total > b.max_probe_steps:
            raise ValueError(f"the program lasts {total} steps; at most {b.max_probe_steps}")
        trajectory = program_trajectory(self.segments)
        worst = float(np.abs(trajectory.offset).max(initial=0.0))
        if worst > b.max_offset_m + 1e-9:
            raise ValueError(
                f"the target reaches {worst:.3f} m from the start position; at most "
                f"{b.max_offset_m} m on any axis"
            )
        for index, s in enumerate(self.segments):
            if s.primitive is ProbePrimitive.ramp:
                assert s.axis is not None and s.amplitude_m is not None
                base = trajectory.base[index][_AXES[s.axis]]
                speed = abs(s.amplitude_m - base) * PROBE_CONTROL_HZ / s.duration_steps
                if speed > b.max_speed_m_s + 1e-12:
                    raise ValueError(
                        f"segment {index}: this ramp moves the target at {speed:.2f} m/s; at "
                        f"most {b.max_speed_m_s} m/s (a longer duration)"
                    )
        return self

    @property
    def steps(self) -> int:
        """Commanded control steps of one probe episode."""
        return sum(s.duration_steps for s in self.segments)


class ProbeRequest(ContractModel):
    """A probe the agent composed (`request_probe`): its name, what it is for, what the
    agent expects, where it runs and how often, and the program."""

    name: Annotated[str, StringConstraints(pattern=PROBE_NAME_PATTERN)]
    purpose: str = Field(max_length=PROBE_BOUNDS.max_text)  # the question it answers
    expectation: str = Field(max_length=PROBE_BOUNDS.max_text)  # what would answer it
    robots: list[RobotId] = Field(min_length=1, max_length=PROBE_BOUNDS.max_robots)
    repeats: int = Field(ge=1, le=PROBE_BOUNDS.max_repeats)  # probe episodes per robot
    segments: list[ProbeSegment] = Field(min_length=1, max_length=PROBE_BOUNDS.max_segments)

    @model_validator(mode="after")
    def _valid(self) -> ProbeRequest:
        if len(set(self.robots)) != len(self.robots):
            raise ValueError("a request names each robot once")
        ProbeProgram(segments=list(self.segments))  # the program's own bounds
        return self

    @property
    def program(self) -> ProbeProgram:
        return ProbeProgram(segments=list(self.segments))

    @property
    def attempts(self) -> int:
        """What the request costs: one robot attempt per probe episode."""
        return len(self.robots) * self.repeats


class ProbeReceipt(ContractModel):
    """The answer to `request_probe`: accepted or not and why, what it costs, and the
    round's budget after it, probe attempts and task attempts apart."""

    accepted: bool
    probe_id: str  # empty when refused
    problems: list[str]
    attempts: int = Field(ge=0)  # probe attempts this request is charged: robots x repeats
    steps_per_episode: int = Field(ge=0)  # commanded control steps of one probe episode
    probe_attempts_used: int = Field(ge=0)  # the round's probe attempts, this request included
    task_attempts_used: int = Field(ge=0)  # the round's experiment-arm attempts
    attempts_left: int = Field(ge=0)  # what is left of the round's attempts

    @model_validator(mode="after")
    def _consistent(self) -> ProbeReceipt:
        if self.accepted:
            check_neutral(IdKind.probe, self.probe_id)
        elif self.probe_id or self.attempts:
            raise ValueError("a refused probe has no id and costs nothing")
        return self


# The fixed public messages of a probe's outcome, by status; nothing else reaches the
# agent. "done": every robot x repeat episode ran and is shown, each charged. "refused":
# the probe was turned down before anything ran (a robot it cannot be played on); it is
# free. "failed": the run broke off and the whole request stays charged, in two parts that
# are never mixed: `attempts_executed`, the episodes the run left a record of (finished or
# partly written; their records are withheld, never exported or named), and
# `attempts_reserved`, the rest, of which the runner cannot confirm whether they ran (a
# worker lost in the middle of a motion leaves no record): kept, never refunded, and never
# counted as executed.
PROBE_MESSAGES: dict[str, str] = {
    "done": "The probe ran; its episodes are in probe_episodes and probe_steps.",
    "probe_not_runnable": "The probe could not be run on these robots; its budget was returned.",
    "execution_failed": (
        "The runner could not complete the probe. Its attempts stay charged: "
        "attempts_executed ran and left records, which are withheld; attempts_reserved "
        "may or may not have run."
    ),
}
PROBE_STATUS_MESSAGE: dict[str, str] = {
    "done": "done",
    "refused": "probe_not_runnable",
    "failed": "execution_failed",
}


class ProbeOutcome(ContractModel):
    """What the runtime records of a probe it ran, refused or could not finish
    (`results/probes.jsonl`): ids and a fixed public message, never the agent's words."""

    probe_id: ProbeId
    status: Literal["done", "refused", "failed"]
    episode_ids: list[EpisodeId]
    message: str
    # Known executed work: probe episodes the run left a record of. All of them when done,
    # none when refused; when failed, those finished or partly written before it broke off.
    attempts_executed: int = Field(ge=0)
    # Unknown work, kept charged: when failed, the rest of the request, which the runner
    # cannot confirm either way; zero otherwise. Never counted as executed.
    attempts_reserved: int = Field(ge=0)

    @model_validator(mode="after")
    def _public(self) -> ProbeOutcome:
        if self.message != PROBE_STATUS_MESSAGE[self.status]:
            raise ValueError("a probe outcome carries its status's fixed public message")
        if (self.status == "done") != bool(self.episode_ids):
            raise ValueError("a probe that ran has episodes; a refused or failed one none")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("a probe's episodes are distinct")
        if self.status == "done" and self.attempts_executed != len(self.episode_ids):
            raise ValueError("a done probe executed exactly its episodes")
        if self.status != "failed" and self.attempts_reserved:
            raise ValueError("only a failed probe keeps attempts reserved")
        if self.status == "refused" and self.attempts_executed:
            raise ValueError("a refused probe executed nothing")
        return self

    @property
    def charged(self) -> int:
        """The attempts this outcome charges to the budget: executed and reserved."""
        return self.attempts_executed + self.attempts_reserved


class QueuedProbe(ContractModel):
    """One accepted probe request as the runtime hands it to the engine."""

    probe_id: ProbeId
    session: int = Field(ge=1)
    request: ProbeRequest


class ProbeResult(ContractModel):
    """What became of a requested probe (`probe_results`): the agent's own definition
    and the probe episodes, whose telemetry is in the tables."""

    probe_id: ProbeId
    status: Literal["queued", "done", "refused", "failed"]
    name: str
    purpose: str
    expectation: str
    episode_ids: list[EpisodeId]
    message: str
    attempts_charged: int  # what it costs: requested while queued, else executed + reserved
    attempts_executed: int  # known executed work (episodes the run left a record of)
    attempts_reserved: int  # a failed probe's unknown rest, kept charged
