"""What the engine hands to the episode log: one `EpisodeRecord` per job, written by `fleetlab.log`.

The rollout produces an `EpisodeRecord`: the visible per-step telemetry
(after the observation hook, the command before the action hook), the chain
(as rendered, the executed action), the events and decisions, the verdicts,
the video frames. `LogSink` turns it into the calls of the log's
`EpisodeWriter`, which owns the files, their layout and their
order (the chain last, so that a job whose chain exists is done).

Visible rows are a deterministic function of the identity and the
simulation: no measured time goes into them. Measured times go into the
job's `stats`, which end up in `run.json` (hidden).

Choices where the note is silent:

- `episodes.started_at` is the log writer's, from the identity and the world's
  episodes per robot and day (`contracts.identity.started_at_for`).
- Steps rows are one per state, 0 to the last step; the last row has no
  commanded action. Hold steps (the robot held still while a supervisor
  decided) and the recovery primitive's steps have no chunk index.
- Events, version 1: `grasp` and `release` when the commanded gripper changes
  sign, and `supervisor_call`, `handover`, `retry` from the decisions. Only a
  run job writes events and decisions (the log refuses them for others).
- The fixture hash (section 4) has no column in the chain table; it goes into
  the job's stats (`run.json`), and `chain.compare` checks it when both sides
  have it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import Camera
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
from fleetlab.contracts.live import DecisionRecord
from fleetlab.contracts.log import EventKind, Outcome
from fleetlab.contracts.rollout import Job, JobMode, SnapshotRef
from fleetlab.engine.chain import Chain

CONTROL_HZ = 20
FRAME_EVERY = 4  # every fourth control step: 5 frames per second


@dataclass
class StepRows:
    """The visible per-step telemetry of one job, column by column."""

    step: list[int] = field(default_factory=list)
    joint_pos: list[npt.NDArray[np.float64]] = field(default_factory=list)
    joint_vel: list[npt.NDArray[np.float64]] = field(default_factory=list)
    gripper_qpos: list[npt.NDArray[np.float64]] = field(default_factory=list)
    eef_pos: list[npt.NDArray[np.float64]] = field(default_factory=list)
    eef_quat: list[npt.NDArray[np.float64]] = field(default_factory=list)
    commanded: list[npt.NDArray[np.float32] | None] = field(default_factory=list)
    chunk_index: list[int | None] = field(default_factory=list)
    chunk_step: list[int | None] = field(default_factory=list)
    frame_index: list[int | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.step)

    def row(self, i: int) -> dict[str, Any]:
        """Row i as the log's `steps` table takes it (without `frame_index`)."""
        return {
            "step": self.step[i],
            "time_s": self.step[i] / CONTROL_HZ,
            "joint_pos": self.joint_pos[i],
            "joint_vel": self.joint_vel[i],
            "gripper_qpos": self.gripper_qpos[i],
            "eef_pos": self.eef_pos[i],
            "eef_quat": self.eef_quat[i],
            "commanded_action": self.commanded[i],
            "chunk_index": self.chunk_index[i],
            "chunk_step": self.chunk_step[i],
        }


@dataclass
class EpisodeRecord:
    """Everything one job produced, for the sink."""

    job: Job
    effective_faults: list[FaultSpec]
    effective_spec: PolicySpec
    in_worker: bool
    steps: StepRows
    chain: Chain
    true_success: bool
    true_reason: str
    visible_success: bool
    libero_first_success_step: int | None
    success_step: int | None = None  # the step the check ended the episode a success
    placed_step: int | None = None  # the same, when the check is the placed check
    events: list[tuple[int, EventKind, str]] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    frames: list[tuple[int, npt.NDArray[np.uint8], npt.NDArray[np.uint8]]] = field(
        default_factory=list
    )
    snapshots: list[SnapshotRef] = field(default_factory=list)
    spawned: list[Job] = field(default_factory=list)
    ended_by: str = ""  # "check", "limit", "stop", "hand_over"
    stats: dict[str, Any] = field(default_factory=dict)  # measured, hidden: run.json only

    @property
    def key(self) -> str:
        return self.job.key

    @property
    def identity(self) -> EpisodeIdentity:
        return self.job.identity

    @property
    def episode_id(self) -> str:
        return self.job.identity.episode_id

    @property
    def visible(self) -> bool:
        """Only run jobs write to the visible log."""
        return self.job.mode is JobMode.run

    @property
    def n_steps(self) -> int:
        """The last step reached (control steps from the reset)."""
        return self.chain.steps[-1] if self.chain.steps else 0


def summary(record: EpisodeRecord) -> dict[str, Any]:
    """The small JSON the main process keeps per job (hidden: it ends up in run.json)."""
    return {
        "key": record.key,
        "episode_id": record.episode_id,
        "mode": record.job.mode.value,
        "success": record.true_success,
        "visible_success": record.visible_success,
        "steps": record.n_steps,
        "ended_by": record.ended_by,
        "fixture_hash": record.chain.fixture_hash,
        "libero_first_success_step": record.libero_first_success_step,
        "success_step": record.success_step,
        "spawned": [j.model_dump(mode="json") for j in record.spawned],
        "snapshots": [s.model_dump(mode="json") for s in record.snapshots],
        **record.stats,
    }


@dataclass
class LogSink:
    """Writes each job through `fleetlab.log.EpisodeWriter` into an opened run directory.

    `episodes_per_robot_day` sets the visible start times (`started_at_for`).
    The step the check fired ("placed") and LIBERO's first fire go to the
    writer for every job mode once its `finish` takes them
    (`job_outcomes`); until then they reach the truth side through the
    verdicts of run jobs and `run.json`.
    """

    run_dir: Path
    episodes_per_robot_day: int

    def write(self, record: EpisodeRecord, task_text: str) -> dict[str, Any]:
        from fleetlab.log import EpisodeWriter

        writer = EpisodeWriter(
            self.run_dir,
            record.job,
            task_text=task_text,
            in_worker=record.in_worker,
            episodes_per_robot_day=self.episodes_per_robot_day,
        )
        if record.visible:
            frames = {step: (scene, wrist) for step, scene, wrist in record.frames}
            for i in range(len(record.steps)):
                images = frames.get(record.steps.step[i])
                if record.steps.frame_index[i] is None:
                    images = None
                given = (
                    None if images is None else {Camera.scene: images[0], Camera.wrist: images[1]}
                )
                writer.add_step(record.steps.row(i), given)
            for step, kind, detail in record.events:
                writer.add_event(step, kind, detail)
            for decision in record.decisions:
                writer.add_decision(decision)
            writer.set_effective_policy(record.effective_spec)
            writer.set_verdict(
                record.true_success,
                record.true_reason,
                record.libero_first_success_step,
                success_step=record.success_step,
                placed_step=record.placed_step,
            )
        chain = record.chain
        writer.set_fixture_hash(chain.fixture_hash)
        for i, step in enumerate(chain.steps):
            executed = chain.executed[i]
            writer.add_chain(
                step,
                chain.state[i],
                {Camera.scene: chain.scene[i], Camera.wrist: chain.wrist[i]},
                None if np.isnan(executed).all() else executed,
            )
        if record.visible:
            handed = record.ended_by == "hand_over"  # a fleet knows an operator took over
            writer.finish(outcome=Outcome.handed_over if handed else record.visible_success)
        else:
            writer.finish(
                success=record.true_success,
                steps=record.n_steps,
                placed_step=record.placed_step,
                libero_step=record.libero_first_success_step,
            )
        return summary(record)


@dataclass
class MemorySink:
    """Keeps every record. For tests and single-process checks."""

    records: dict[str, EpisodeRecord] = field(default_factory=dict)

    def write(self, record: EpisodeRecord, task_text: str) -> dict[str, Any]:
        self.records[record.key] = record
        return summary(record)
