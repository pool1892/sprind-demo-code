"""Episode identity, seeds, noise and the replay rules (docs/design/contracts.md, section 4).

An episode is its identity: two runs with the same identity produce the same
states and images, bit for bit, on the same machine, under the seven replay
rules (`REPLAY_RULES`). Not part of the identity, and shown not to matter: the
worker process, what ran before or beside it, the order of episodes, whether
video is written, the other rows of a call.

Seeds come from `seeds_for`, sampling noise from `noise_for` and the visible
start time from `started_at_for`, all pure functions that must never change
between worlds. The neutral identifiers of
principle 6 live in `contracts.ids` and are re-exported here.

Choices this module makes where the note is silent:

- `seeds_for` hashes the fields with SHA-256 (as `contracts.hashing` does)
  under a scheme name and a purpose; the scene seed is below 2**32 because
  LIBERO hands it to numpy's legacy generator, the noise seed below 2**63
  (it is stored as int64).
- `noise_for` draws standard normal noise from numpy's PCG64 generator seeded
  with `SeedSequence([scheme, noise_seed, chunk_index, sample])`, in float32,
  and casts to the requested dtype afterwards. `sample` numbers extra draws for
  the same chunk (section 14, an action-spread signal); the executed call uses
  sample 0.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.day_layout import (
    ARM_SLOTS,
    ARM_SPACING_S,
    ARM_START_S,
    LAST_START_S,
    SHIFT_LENGTH_S,
    SHIFT_START_S,
)
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.ids import (
    EpisodeId,
    ExperimentId,
    FaultId,
    IdKind,
    RobotId,
    Scope,
    SiteId,
    WorldId,
    check_neutral,
    id_number,
    is_neutral,
    neutral_id,
    which_kind,
)
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.probes import ProbeProgram
from fleetlab.contracts.version import CONTRACTS_VERSION

__all__ = [
    "ARM_SLOTS",
    "ARM_SPACING_S",
    "ARM_START_S",
    "LAST_START_S",
    "NOISE_SEED_LIMIT",
    "REPLAY_RULES",
    "SCENE_SEED_LIMIT",
    "SHIFT_LENGTH_S",
    "SHIFT_START_S",
    "WORLD_SEED_LIMIT",
    "EpisodeId",
    "EpisodeIdentity",
    "ExperimentId",
    "FaultId",
    "IdKind",
    "PolicySpec",
    "ReplayRule",
    "RobotId",
    "Scope",
    "SiteId",
    "SoftwareVersion",
    "Task",
    "WorldId",
    "check_neutral",
    "id_number",
    "is_neutral",
    "neutral_id",
    "noise_for",
    "seeds_for",
    "started_at_for",
    "which_kind",
]


class Task(NamedTuple):
    """A LIBERO task: the suite and the task's index in it."""

    suite: str
    task_index: int


class PolicySpec(ContractModel):
    """Which policy, in which revision and with which settings, an episode ran.

    The identity holds the nominal spec; a configuration fault's effective spec
    (after the policy input hook) is recorded on the truth side. The control
    mode has its home here. `sampling_steps` is None for a policy that does not
    sample (ACT, the teleoperator).

    `latency_compensation_steps` (configuration key
    `control.latency_compensation_steps`) is how many actions at the head of
    every chunk after an episode's first the rollout skips before it executes
    the next `actions_executed_per_chunk`: with commands that reach the robot
    d steps late, a chunk computed while the robot moves arrives with d stale
    actions, and skipping them makes each executed action land at the time
    its chunk meant it for (the delay compensation of real-time chunking).
    The first chunk is cut as before, because the robot waits for it. At
    least one action of a chunk is always kept. The setting therefore also
    shortens how often the robot replans (30 - d actions of X-VLA's 30). It
    is left out of the serialized form at 0, so that every identity, manifest
    and world written before it existed keeps its bytes and hashes.
    """

    name: str
    revision: str
    precision: str
    control_mode: ControlMode
    actions_executed_per_chunk: int = Field(ge=1)
    sampling_steps: int | None = Field(ge=1)
    image_views: int = Field(ge=1)
    latency_compensation_steps: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)


class SoftwareVersion(ContractModel):
    """The software an episode ran on. Exactness holds on one machine with this software."""

    lock_hash: str  # hash of uv.lock
    libero_patch_set: str  # which of our LIBERO patches are applied
    mujoco_version: str
    miopen_db_hash: str  # the pinned convolution database (replay rule 7)
    contracts_version: str = CONTRACTS_VERSION


# LIBERO hands the scene seed to `np.random.seed`, which takes values below 2**32.
SCENE_SEED_LIMIT = 2**32
# The noise seed only seeds a SeedSequence; it is stored as int64.
NOISE_SEED_LIMIT = 2**63
# World seeds also fit signed int64; hashing JSON integers preserves every bit.
WORLD_SEED_LIMIT = 2**63
_SEED_SCHEME = "fleetlab.seeds.v1"
_NOISE_SCHEME = 1


# The simulated working day (`day_layout`): a shift from 08:00 for eight hours, then
# 160 experiment-arm slots per robot and day from 16:00, the last start 23:59:59.


def started_at_for(identity: EpisodeIdentity, episodes_per_robot_day: int = 14) -> str:
    """The simulated time of day an episode starts, as "HH:MM:SS"; a pure function.

    The day's shift starts at 08:00 and the robot's `episodes_per_robot_day`
    as-run episodes follow at equal intervals over eight hours, by
    `episode_index` (whole seconds, rounded down): with 14 episodes, index 0
    starts at 08:00:00 and index 13 at 15:25:42. Indices beyond the as-run
    count are experiment-arm episodes: they start at 16:00:00 and
    follow every three minutes, whatever the as-run spacing, because a fleet
    running tests after hours packs them densely. That gives 160 arm slots per
    robot and day, the last at 23:57:00. Two runs of one identity get the same
    time.

    The log writer takes `episodes.started_at` from it, as
    `datetime.time.fromisoformat(started_at_for(identity, n))` with the world's
    `episodes_per_robot_day`. Raises ValueError only for an index
    whose time would pass 23:59:59.
    """
    n = episodes_per_robot_day
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"episodes_per_robot_day must be a positive integer, not {n!r}")
    index = identity.episode_index
    if index < n:
        seconds = SHIFT_START_S + index * SHIFT_LENGTH_S // n
    else:
        seconds = ARM_START_S + (index - n) * ARM_SPACING_S
    if seconds > LAST_START_S:
        raise ValueError(
            f"episode_index {index} would start after 23:59:59: a robot's day holds {n} "
            f"as-run episodes and {ARM_SLOTS} experiment-arm episodes"
        )
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def seeds_for(world_seed: int, robot_id: str, day: int, episode_index: int) -> tuple[int, int]:
    """The scene seed and the noise seed of an episode, as a pure function of its fields.

    Each seed comes from SHA-256 over `scheme NUL purpose NUL json([world_seed,
    robot_id, day, episode_index])`, with purpose "scene" or "noise", so the two
    are independent. No global random generator is involved: the same fields
    give the same seeds in any process, in any order. `episode_index` numbers
    the robot's episodes within the day, so no two episodes of a world share
    the fields.

    Collisions: the noise seed has 63 bits, so two episodes of any realistic
    world never share it. The scene seed has 32 bits (LIBERO's limit), so two
    episodes share one with probability 2**-32 per pair, about one pair in a
    world of 90,000 episodes; that only correlates the furniture placement of
    two episodes that differ in robot, day or index.
    """
    for name, value in (("world_seed", world_seed), ("day", day), ("episode_index", episode_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, not {value!r}")
    if world_seed >= WORLD_SEED_LIMIT:
        raise ValueError("world_seed must fit a non-negative signed int64")
    check_neutral(IdKind.robot, robot_id)
    fields = json.dumps([world_seed, robot_id, day, episode_index], separators=(",", ":"))

    def digest(purpose: str) -> bytes:
        return hashlib.sha256(f"{_SEED_SCHEME}\0{purpose}\0{fields}".encode()).digest()

    scene = int.from_bytes(digest("scene")[:4], "big")
    noise = int.from_bytes(digest("noise")[:8], "big") >> 1
    return scene, noise


class EpisodeIdentity(ContractModel):
    """Everything that fixes an episode. Lives on the truth side (`identities` table).

    Two runs with equal identities are the same episode. `episode_index`
    numbers the robot's episodes within the day; `scene_seed` and `noise_seed`
    come from `seeds_for`. `policy` is the nominal spec. `row` is None for a
    row-independent policy and fixed otherwise. `faults` are the faults in
    force in the episode (`contracts.faults.active_faults`), each with the
    level of its day: the validator refuses a fault at level zero, one not
    scheduled on the day, or one whose scope misses the robot, so that the
    oracle never toggles a fault that was not in force. `experiment_id` names
    the experiment an episode is an arm of (the visible `episodes.experiment_id`).
    `probe` is the program of a probe episode (`contracts.probes`): a scripted motion
    the robot plays without the policy. It is left out of the serialized form when
    absent, so that every identity written before it existed keeps its bytes and hashes.
    """

    world_id: WorldId
    episode_id: EpisodeId
    robot_id: RobotId
    site_id: SiteId
    day: int = Field(ge=0)
    episode_index: int = Field(ge=0)
    suite: str
    task_index: int = Field(ge=0)
    start_state: int = Field(ge=0)
    scene_seed: int = Field(ge=0, lt=SCENE_SEED_LIMIT)
    noise_seed: int = Field(ge=0, lt=NOISE_SEED_LIMIT)
    policy: PolicySpec
    batch_size: int = Field(ge=1)
    row: int | None = Field(ge=0)
    image_size: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    faults: list[FaultSpec] = Field(default_factory=list)
    software: SoftwareVersion
    experiment_id: ExperimentId | None = None  # the experiment this episode is an arm of
    probe: ProbeProgram | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _consistent(self) -> EpisodeIdentity:
        if self.row is not None and self.row >= self.batch_size:
            raise ValueError(f"row {self.row} is outside a batch of {self.batch_size}")
        ids = [f.fault_id for f in self.faults]
        if len(ids) != len(set(ids)):
            raise ValueError("an episode lists each fault once")
        outside = [
            f.fault_id for f in self.faults if not f.target.covers(self.robot_id, self.site_id)
        ]
        if outside:
            raise ValueError(f"faults {outside} do not cover robot {self.robot_id}")
        idle = [f.fault_id for f in self.faults if not f.is_fault]
        if idle:
            raise ValueError(f"faults {idle} are at level zero; an identity lists faults in force")
        off = [f.fault_id for f in self.faults if not f.schedule.active_on(self.day)]
        if off:
            raise ValueError(f"faults {off} are not scheduled on day {self.day}")
        return self

    @property
    def task(self) -> Task:
        return Task(self.suite, self.task_index)

    @property
    def fault_ids(self) -> frozenset[str]:
        return frozenset(f.fault_id for f in self.faults)

    def check_seeds(self, world_seed: int) -> None:
        """Raise ValueError unless the seeds are the ones `seeds_for` gives this episode."""
        expected = seeds_for(world_seed, self.robot_id, self.day, self.episode_index)
        if (self.scene_seed, self.noise_seed) != expected:
            raise ValueError(
                f"episode {self.episode_id}: seeds are not derived from the world seed"
            )


def noise_for(
    identity: EpisodeIdentity,
    chunk_index: int,
    shape: tuple[int, ...],
    dtype: str | np.dtype[Any] = "float32",
    *,
    sample: int = 0,
) -> npt.NDArray[Any]:
    """The sampling noise of one row for one policy call: standard normal, from the identity.

    Drawn on the CPU with numpy's PCG64 generator, seeded by
    `SeedSequence([1, noise_seed, chunk_index, sample])`, as float32, then cast
    to `dtype` (a policy's `noise_dtype`). The same identity, chunk and sample
    give the same bits in any process. This function must never change between
    worlds; a change is a new scheme number and a contracts version bump.
    """
    for name, value in (("chunk_index", chunk_index), ("sample", sample)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, not {value!r}")
    entropy = [_NOISE_SCHEME, identity.noise_seed, chunk_index, sample]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    values = generator.standard_normal(shape, dtype=np.float32)
    return values.astype(np.dtype(dtype), copy=False)


class ReplayRule(NamedTuple):
    """One of the rules the rollout keeps so that an episode replays bit for bit."""

    name: str
    description: str
    source: str


# The seven replay rules (replay and xvla-speed spikes). The engine reports
# which it kept, by name, in run.json.
REPLAY_RULES: tuple[ReplayRule, ...] = (
    ReplayRule(
        "scene_seed_before_reset",
        "the scene seed is set directly before the reset, the recorded start state after it",
        "spikes/replay; LiberoEnv.reset",
    ),
    ReplayRule(
        "noise_from_rollout",
        "the policy's sampling noise is drawn from noise_seed by the rollout and handed to the "
        "policy; a policy never draws its own",
        "spikes/replay",
    ),
    ReplayRule(
        "batch_size_fixed",
        "the batch size is fixed for a run and recorded; the row is free for a policy whose "
        "row_independent flag is verified, fixed otherwise",
        "spikes/xvla-speed section 3",
    ),
    ReplayRule(
        "rollout_owns_queue",
        "the rollout owns the action queue; the environment never resets by itself",
        "spikes/replay",
    ),
    ReplayRule(
        "full_snapshot_restore",
        "a mid-episode restore uses a full snapshot: integration state with the solver warm "
        "start, gripper controller state, fixture poses, sensor clocks, and the observation "
        "delivered at that step; the restore ends with mj_forward and does not render",
        "spikes/replay section 3; spikes/teleop-expert (absolute control)",
    ),
    ReplayRule(
        "one_render_context",
        "one rendering context is current at a time; each environment makes its own current "
        "before every render",
        "LiberoEnv._make_current",
    ),
    ReplayRule(
        "pinned_conv_database",
        "convolution algorithms come from the pinned MIOpen database with cudnn.benchmark off; "
        "the database hash is part of the software version; the database is keyed per images "
        "per call (batch size times real image views) and precision, and a pair it does not "
        "hold is refused, so a fault that changes the image views needs its own entry",
        "spikes/xvla-speed section 4",
    ),
)
