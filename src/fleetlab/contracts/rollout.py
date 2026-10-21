"""The rollout engine's jobs, snapshots and run layout (docs/design/contracts.md, section 10).

A run is one invocation of the engine on a manifest of jobs; a job is one
episode, or one replay of it, with its identity. Modes:

- "run": the episode as the world schedules it.
- "replay": the same identity with faults removed (`remove`) or changes added
  (`add`), from the reset or from a snapshot (the counterfactual replay that
  scores an intervention).
- "continue": a hand-over continuation for the teleoperator, from a snapshot.
  Its own step budget `max_steps` may go beyond the suite's limit (the
  teleoperator needed up to 600 steps plus settling; the simulator's horizon
  is raised to match), and it ends when the in-worker policy reports `done`.
- "scripted": a supervised episode replayed with its recorded decisions
  (`decision_trace`), so that its chain can be proved like any other.

Snapshots serve the counterfactual replay and the hand-over only; they are
never used to put the world back for the robot itself (section 14). The engine
proves snapshot exactness like replay: continuing from a snapshot gives the
chain of the uninterrupted episode.

Choices this module makes where the note is silent:

- `Job.key` names a job's files: the episode id for a "run" job, the episode
  id with the mode and a hash of what differs for the others (`max_steps`
  enters the hash only when it is set, so the keys of jobs without it stay).
- `Job.max_steps` is allowed for "scripted" jobs too, whose replayed
  continuation may need the same budget; "run" and "replay" jobs keep the
  identity's `max_steps`.
- A job is done when its chain exists, and for a "run" job also its steps
  file (`RunLayout.is_done`); replays write nothing to the visible log.
- `RunRecord` is the schema of `run.json`, with `chain_ok` the run-level flag
  of section 4 and free-form `extra` for timings, waits, memory and workers.
- `chain_sample` picks the unmodified replays that prove a run: every tenth
  episode, or more often so that at least 20 are picked. A supervised episode's
  sample replay is a "scripted" job from its recorded decision trace.
- Snapshots lie at `truth/snapshots/<episode>/<key>.<step>.pkl` (the engine's
  convention), and a job spawned from one at run time is persisted next to it
  as `<key>.<step>.continue.json` (`RunLayout.continuation_file`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, JsonValue, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.env import Camera, EnvSnapshot
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.identity import EpisodeIdentity, SoftwareVersion
from fleetlab.contracts.ids import EpisodeId, FaultId
from fleetlab.contracts.live import Decision, DecisionRecord, StepTelemetry
from fleetlab.contracts.log import RUN_RECORD, SCHEMA, SNAPSHOT_DIR, VIDEO_PATH
from fleetlab.contracts.policy import Action7

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation
    from fleetlab.contracts.policy import Policy

__all__ = [
    "CHAIN_SAMPLE_MIN",
    "Decision",
    "DecisionRecord",
    "Job",
    "JobMode",
    "RunLayout",
    "RunRecord",
    "Snapshot",
    "SnapshotRef",
    "SupervisorHook",
    "SupervisorSpec",
    "Takeover",
    "chain_sample",
]


class JobMode(StrEnum):
    run = "run"
    replay = "replay"
    continue_ = "continue"
    scripted = "scripted"


class SnapshotRef(ContractModel):
    """Where a stored snapshot lies: an episode, a step, a path relative to the run directory."""

    episode_id: EpisodeId
    step: int = Field(ge=0)
    path: str


class SupervisorSpec(ContractModel):
    """Which supervisor watches a job, and its settings."""

    name: str
    params: dict[str, JsonValue] = Field(default_factory=dict)


class Job(ContractModel):
    """One episode or one replay, with everything the engine needs to run it."""

    identity: EpisodeIdentity
    mode: JobMode = JobMode.run
    remove: list[FaultId] = Field(default_factory=list)
    add: list[FaultSpec] = Field(default_factory=list)
    from_snapshot: SnapshotRef | None = None
    record_video: bool = False
    supervisor: SupervisorSpec | None = None
    decision_trace: list[DecisionRecord] | None = None
    max_steps: int | None = Field(default=None, ge=1)  # a continuation's own step budget
    # Where a run job saves snapshots: the restore check, the oracle's counterfactuals.
    snapshot_steps: list[int] = Field(default_factory=list)
    # Persisted operator admission and human parameters; absent for the expert alone.
    teleoperator_params: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _fits_mode(self) -> Job:
        """Check what the mode allows.

        Beyond these checks, one rule of version 1 lives in the engine because
        it needs the fault registry: a "replay" or "continue" job from a
        snapshot may not `remove` or `add` a fault whose type defines a cell
        hook. A snapshot carries the scene as the reset's cell hooks left it
        (only the fixture bodies are restored, section 5), so such a change
        would not take effect; the engine refuses the job.
        """
        unknown = sorted(set(self.remove) - self.identity.fault_ids)
        if unknown:
            raise ValueError(f"remove names faults the episode does not have: {unknown}")
        clash = sorted({f.fault_id for f in self.add} & self.identity.fault_ids)
        if clash:
            raise ValueError(f"added changes reuse fault ids of the episode: {clash}")
        if self.mode is JobMode.run and (self.remove or self.add or self.from_snapshot):
            raise ValueError("a run job runs the episode as scheduled, from its reset")
        if self.mode is JobMode.continue_ and self.from_snapshot is None:
            raise ValueError("a continue job starts from a snapshot")
        if self.teleoperator_params is not None and self.mode is not JobMode.continue_:
            raise ValueError("only a continue job carries teleoperator_params")
        if self.mode is JobMode.scripted:
            if self.from_snapshot is not None and self.from_snapshot.step > 0:
                raise ValueError("scripted decision replay must start from reset, not mid-attempt")
            if not self.decision_trace:
                raise ValueError("a scripted job replays a decision trace")
            if self.supervisor is not None:
                raise ValueError("a scripted job takes its decisions from the trace")
        elif self.decision_trace is not None:
            raise ValueError("only a scripted job carries a decision trace")
        if self.max_steps is not None and self.mode in (JobMode.run, JobMode.replay):
            raise ValueError(
                f"a {self.mode.value} job runs to the identity's max_steps; only continue "
                "and scripted jobs have their own budget"
            )
        if self.snapshot_steps:
            if self.mode is not JobMode.run:
                raise ValueError("only a run job saves snapshots (snapshot_steps)")
            steps = self.snapshot_steps
            if len(set(steps)) != len(steps):
                raise ValueError("snapshot_steps names each step once")
            outside = [s for s in steps if not 0 <= s < self.identity.max_steps]
            if outside:
                raise ValueError(f"snapshot steps {outside} are outside the episode's steps")
        return self

    @property
    def step_budget(self) -> int:
        """The steps this job may run: its own `max_steps`, else the identity's."""
        return self.identity.max_steps if self.max_steps is None else self.max_steps

    @property
    def effective_faults(self) -> list[FaultSpec]:
        """The faults in force: the identity's minus those removed, plus those added."""
        removed = set(self.remove)
        return [f for f in self.identity.faults if f.fault_id not in removed] + list(self.add)

    @property
    def key(self) -> str:
        """A name for the job's files, unique within a run."""
        if self.mode is JobMode.run:
            return self.identity.episode_id
        differs = self.model_dump(
            mode="json", include={"remove", "add", "from_snapshot", "decision_trace"}
        )
        if self.max_steps is not None:
            differs["max_steps"] = self.max_steps
        if self.teleoperator_params is not None:
            differs["teleoperator_params"] = self.teleoperator_params
        return f"{self.identity.episode_id}.{self.mode.value}.{hash_json(differs)}"


@dataclass(frozen=True)
class Snapshot:
    """The environment snapshot plus the rollout's own state at one step.

    The arm controller's goal is not saved (`contracts.env.EnvSnapshot`).
    `fault_states` holds each active fault's `FaultType.snapshot`, by fault id;
    `success_check` the check's `SuccessCheck.snapshot`.
    """

    env: EnvSnapshot
    queued_actions: list[Action7]
    chunk_index: int
    step: int
    fault_states: dict[str, bytes]
    success_check: bytes
    last_commanded: Action7 | None = None
    # Active ticks left, not a physical end-step: decision/operator holds consume none.
    # None identifies legacy snapshots; the engine derives their budget from the check clock.
    remaining_active_steps: int | None = None
    retry_used: bool = False
    degraded: bool = False


class SupervisorHook(Protocol):
    """Called every step after the risk estimator fires."""

    def __call__(
        self, step: int, observation: Observation, telemetry_window: Sequence[StepTelemetry]
    ) -> Decision: ...


class Takeover(Protocol):
    """Turns a snapshot and the teleoperator into a "continue" job."""

    def __call__(self, snapshot: SnapshotRef, teleoperator: Policy) -> Job: ...


class RunRecord(ContractModel):
    """`run.json`, hidden: what the run ran on and whether its chain sample held."""

    software: SoftwareVersion
    batch_size: int = Field(ge=1)
    replay_rules: list[str]  # names from identity.REPLAY_RULES the engine kept
    chain_sample: list[EpisodeId] = Field(default_factory=list)
    chain_ok: bool | None = None  # None until the sample has been replayed
    extra: dict[str, JsonValue] = Field(default_factory=dict)  # timings, waits, memory, workers


CHAIN_SAMPLE_MIN = 20
CONTINUATION_SUFFIX = ".continue.json"


def chain_sample(episode_ids: Sequence[str]) -> list[str]:
    """The episodes replayed unmodified to prove a run: every tenth, and at least 20.

    Taken in the given (manifest) order at a fixed stride; all of them when
    there are 20 or fewer.
    """
    n = len(episode_ids)
    if n <= CHAIN_SAMPLE_MIN:
        return list(episode_ids)
    stride = max(1, min(10, n // CHAIN_SAMPLE_MIN))
    return list(episode_ids[::stride])


@dataclass(frozen=True)
class RunLayout:
    """The paths of a run directory (section 11)."""

    root: Path

    @property
    def log_dir(self) -> Path:
        return self.root / "log"

    @property
    def truth_dir(self) -> Path:
        return self.root / "truth"

    @property
    def run_record(self) -> Path:
        return self.root / RUN_RECORD

    @property
    def snapshots_dir(self) -> Path:
        return self.root / SNAPSHOT_DIR

    @property
    def description(self) -> Path:
        return self.log_dir / "description.json"

    def table(self, name: str, key: str | None = None) -> Path:
        """The file of a table of `contracts.log.SCHEMA`.

        A table stored one file per episode (`steps`) needs the episode id, one
        stored per job (`chain`) the job's key.
        """
        spec = SCHEMA[name]
        if spec.file_key is None:
            return self.root / spec.path
        if key is None:
            raise ValueError(f"table {name} is stored one file per {spec.file_key}")
        return self.root / spec.path.format(**{spec.file_key: key})

    def video(self, episode_id: str, camera: Camera) -> Path:
        return self.root / VIDEO_PATH.format(episode_id=episode_id, camera=camera.value)

    def chain(self, job: Job) -> Path:
        return self.table("chain", job.key)

    def snapshot_file(self, episode_id: str, key: str, step: int) -> Path:
        """Where the engine stores the snapshot of job `key` at `step`:
        `truth/snapshots/<episode>/<key>.<step>.pkl`."""
        return self.snapshots_dir / episode_id / f"{key}.{step}.pkl"

    def continuation_file(self, episode_id: str, key: str, step: int) -> Path:
        """Where a job spawned from that snapshot (a hand-over's continuation) is persisted
        the moment it is created: `truth/snapshots/<episode>/<key>.<step>.continue.json`,
        next to the snapshot, so that a resumed run finds and finishes it."""
        return self.snapshots_dir / episode_id / f"{key}.{step}.continue.json"

    def continuation_files(self) -> list[Path]:
        """Every persisted continuation job of the run, in a fixed order."""
        return sorted(self.snapshots_dir.glob(f"*/*{CONTINUATION_SUFFIX}"))

    def continuations(self) -> list[Job]:
        """The persisted continuation jobs, read and validated."""
        return [Job.model_validate_json(p.read_text()) for p in self.continuation_files()]

    def is_done(self, job: Job) -> bool:
        """A job is done when its chain exists, and for a run job its steps file too."""
        if not self.chain(job).is_file():
            return False
        return job.mode is not JobMode.run or self.table("steps", job.identity.episode_id).is_file()
