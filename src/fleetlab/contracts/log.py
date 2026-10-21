"""The episode log, version 1: every stored column, its type and its visibility
(docs/design/contracts.md, section 11).

One run directory per run, two trees: `log/` holds what the agents may see,
`truth/` what they must not, and `run.json` (hidden) the run's own record.
Every column is declared once in `SCHEMA` with its Arrow type, a visibility
tag and a one-line description. The visible export and the data dictionary
are built from the columns tagged visible (`visible_schema`,
`allowed_columns`), never by removing hidden ones.

Everything visible is a deterministic function of the identities, so two runs
of the same world give byte-identical visible tables: times in visible tables
are simulated, and measured times (inference waits, wall clock, decision
latency) live in `run.json` or hidden tables only. Each time column says which
kind it is (`ColumnSpec.clock`), and `schema_problems` refuses a visible
measured time.

`validate_log_tree` checks a run's `log/` tree before anything is exported:
only known files, only allowed columns, no column that looks like a measured
time, and neutral identifiers in file names and id columns.

Choices this module makes where the note is silent:

- `steps` carries `episode_id` as its first column, so that the per-episode
  files can be queried as one table.
- `started_at` is the simulated time of day (Arrow time64, microseconds), the
  value of `contracts.identity.started_at_for`, so that two runs
  of one identity write the same time.
- `change_log.old` and `.new` hold the value as JSON text, since a value can
  be a number, a string, a flag or a list.
- The hidden table `verdicts` records, per episode, the fleet check's verdict
  before the data hook with its reason (`contracts.env.VERDICT_REASONS`), the
  step the fleet's check fired, and LIBERO's first-fire step (section 5).
- `chain` is stored as NumPy arrays (`.npz`), one per column, under the job's
  key (`contracts.rollout.Job.key`: the episode id for a run, its own key for a
  replay); the row of step 0 holds the hashes after the reset and NaN as its
  executed action. `fixture_hash` is a per-file column (`ColumnSpec.per_file`):
  a 0-d array with `SimEnv.fixture_hash` as hex, after the reset or the
  restore.
- `faults_active.params` and `.nominal` hold JSON text; `nominal` is empty for
  a fault on a component.
- The visible table `costs` holds `contracts.actions.COSTS` (`costs_rows`):
  what an operator knows a repair costs. The visible table `catalogue` holds
  every valid action of the world (`contracts.actions.catalogue_rows`), so
  that an agent queries it instead of reading it whole.
  `episodes.experiment_id` names the experiment an episode was an arm of
  (empty otherwise), so a later session keeps those episodes out of the
  observed rates.

`job_outcomes` records every job of any mode in one place, one
row per job: the "continue", "scripted" and snapshot-replay jobs have no other
table, and the live layer's scoring reads them here. `removed` is a JSON list
of fault ids, `added` a JSON list of `FaultSpec`s in pydantic's JSON form.
`replays` stays as the oracle's view: the replays from the reset, keyed by
episode and effective fault list.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.contracts._base import Visibility, field_visibility
from fleetlab.contracts.env import Camera
from fleetlab.contracts.ids import IdKind, check_scope, is_neutral
from fleetlab.contracts.model import MODEL_CLASSES

__all__ = [
    "JOB_TABLE_NAMES",
    "PROBE_TABLE_NAMES",
    "SCHEMA",
    "ColumnSpec",
    "EventKind",
    "Outcome",
    "TableSpec",
    "Visibility",
    "allowed_columns",
    "arrow_schema",
    "data_dictionary_markdown",
    "export_table_markdown",
    "log_tree_problems",
    "schema_problems",
    "validate_log_tree",
    "visible_schema",
    "visible_table_names",
]


class EventKind(StrEnum):
    """The kinds of event in the `events` table."""

    grasp = "grasp"
    release = "release"
    drop = "drop"
    collision = "collision"
    near_miss = "near_miss"
    stall = "stall"
    supervisor_call = "supervisor_call"
    handover = "handover"
    retry = "retry"


class Outcome(StrEnum):
    """An episode's visible outcome: by the fleet's success check, or handed over.

    `handed_over` is a third value because a fleet knows an operator took over;
    the continuation's result is truth only (`job_outcomes`).
    """

    success = "success"
    failure = "failure"
    handed_over = "handed_over"


V = Visibility.visible
H = Visibility.hidden
Clock = Literal["simulated", "measured"]


@dataclass(frozen=True)
class ColumnSpec:
    """One stored column: name, Arrow type, visibility tag, description, a time's clock."""

    name: str
    type: Any  # a pyarrow DataType
    visibility: Visibility
    description: str
    clock: Clock | None = None
    id_kind: IdKind | Literal["scope"] | None = None  # values must be neutral ids of this kind
    per_file: bool = False  # one value per file (a 0-d array in an npz), not one per row


@dataclass(frozen=True)
class TableSpec:
    """One stored table: where it lives, how it is stored, what it holds.

    `source` says who writes it: "run" for the tables every run directory holds,
    "export" for the visible tables only an export holds, built by its caller from other
    files (the probe tables of `contracts.probes`); an export without them is complete.
    """

    name: str
    tree: Literal["log", "truth"]
    path: str  # relative to the run directory; `{episode_id}` or `{job_key}` for one file each
    storage: Literal["parquet", "npz"]
    description: str
    columns: tuple[ColumnSpec, ...]
    source: Literal["run", "export"] = "run"

    @property
    def per_episode(self) -> bool:
        return "{episode_id}" in self.path

    @property
    def file_key(self) -> str | None:
        """The placeholder of a table stored as one file per episode or job, or None."""
        for key in ("episode_id", "job_key"):
            if "{" + key + "}" in self.path:
                return key
        return None

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"table {self.name} has no column {name!r}")


def _vector(n: int, kind: Any = None) -> Any:
    return pa.list_(kind if kind is not None else pa.float64(), n)


_EPISODE_ID_V = ColumnSpec("episode_id", pa.string(), V, "the episode", id_kind=IdKind.episode)
_EPISODE_ID_H = ColumnSpec("episode_id", pa.string(), H, "the episode", id_kind=IdKind.episode)

_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "episodes",
        "log",
        "log/episodes.parquet",
        "parquet",
        "one row per episode: who ran which task when, and how it ended",
        (
            _EPISODE_ID_V,
            ColumnSpec("robot_id", pa.string(), V, "the robot that ran it", id_kind=IdKind.robot),
            ColumnSpec("site_id", pa.string(), V, "the robot's site", id_kind=IdKind.site),
            ColumnSpec("day", pa.int32(), V, "simulated day, from 0"),
            ColumnSpec("task_text", pa.string(), V, "the task as given to the robot"),
            ColumnSpec(
                "started_at",
                pa.time64("us"),
                V,
                "time of day the episode started",
                clock="simulated",
            ),
            ColumnSpec("steps", pa.int32(), V, "control steps run (20 per second)"),
            ColumnSpec(
                "outcome", pa.string(), V, "success, failure, or handed_over to an operator"
            ),
            ColumnSpec(
                "duration_s", pa.float64(), V, "length of the episode in seconds", clock="simulated"
            ),
            ColumnSpec("video_scene", pa.string(), V, "scene camera video, relative to log/"),
            ColumnSpec("video_wrist", pa.string(), V, "wrist camera video, relative to log/"),
            ColumnSpec(
                "experiment_id",
                pa.string(),
                V,
                "the experiment the episode was an arm of, empty for regular work",
                id_kind=IdKind.experiment,
            ),
        ),
    ),
    TableSpec(
        "steps",
        "log",
        "log/steps/{episode_id}.parquet",
        "parquet",
        "one row per control step of one episode: the robot's telemetry",
        (
            _EPISODE_ID_V,
            ColumnSpec("step", pa.int32(), V, "control step, from 0"),
            ColumnSpec(
                "time_s", pa.float64(), V, "seconds since the episode started", clock="simulated"
            ),
            ColumnSpec("joint_pos", _vector(7), V, "joint positions in radians"),
            ColumnSpec("joint_vel", _vector(7), V, "joint velocities in radians per second"),
            ColumnSpec("gripper_qpos", _vector(2), V, "finger positions in metres"),
            ColumnSpec("eef_pos", _vector(3), V, "end effector position in metres"),
            ColumnSpec("eef_quat", _vector(4), V, "end effector orientation as a quaternion"),
            ColumnSpec(
                "commanded_action",
                _vector(7, pa.float32()),
                V,
                "the action commanded: position, rotation, gripper; empty in the last row",
            ),
            ColumnSpec("chunk_index", pa.int32(), V, "which policy call the action came from"),
            ColumnSpec("chunk_step", pa.int32(), V, "the action's place within its call"),
            ColumnSpec(
                "frame_index", pa.int32(), V, "the step's frame in the videos, empty if it has none"
            ),
        ),
    ),
    TableSpec(
        "events",
        "log",
        "log/events.parquet",
        "parquet",
        "notable moments within episodes, as the fleet's event detection reports them",
        (
            _EPISODE_ID_V,
            ColumnSpec("step", pa.int32(), V, "control step of the event"),
            ColumnSpec("kind", pa.string(), V, "the kind of event"),
            ColumnSpec("detail", pa.string(), V, "what the detector saw, in words"),
        ),
    ),
    TableSpec(
        "health",
        "log",
        "log/health.parquet",
        "parquet",
        "periodic component checks, one row per reading",
        (
            ColumnSpec("day", pa.int32(), V, "simulated day of the check"),
            ColumnSpec("robot_id", pa.string(), V, "the robot", id_kind=IdKind.robot),
            ColumnSpec("component", pa.string(), V, "the component checked"),
            ColumnSpec("metric", pa.string(), V, "what was measured"),
            ColumnSpec("value", pa.float64(), V, "the reading"),
        ),
    ),
    TableSpec(
        "change_log",
        "log",
        "log/change_log.parquet",
        "parquet",
        "changes of configuration values, as the fleet recorded them",
        (
            ColumnSpec("day", pa.int32(), V, "simulated day of the change"),
            ColumnSpec("scope", pa.string(), V, "fleet, a site or a robot", id_kind="scope"),
            ColumnSpec("key", pa.string(), V, "the configuration key"),
            ColumnSpec("old", pa.string(), V, "the value before, as JSON"),
            ColumnSpec("new", pa.string(), V, "the value after, as JSON"),
            ColumnSpec("author", pa.string(), V, "who made it: ops, vendor or auto"),
        ),
    ),
    TableSpec(
        "maintenance",
        "log",
        "log/maintenance.parquet",
        "parquet",
        "maintenance carried out on robots",
        (
            ColumnSpec("day", pa.int32(), V, "simulated day of the work"),
            ColumnSpec("robot_id", pa.string(), V, "the robot", id_kind=IdKind.robot),
            ColumnSpec("component", pa.string(), V, "the component worked on"),
            ColumnSpec("verb", pa.string(), V, "what was done"),
            ColumnSpec("note", pa.string(), V, "the technician's note"),
        ),
    ),
    TableSpec(
        "costs",
        "log",
        "log/costs.parquet",
        "parquet",
        "what each action costs in operator hours, parts included",
        (
            ColumnSpec("verb", pa.string(), V, "the action's verb"),
            ColumnSpec(
                "target", pa.string(), V, "the component, or the kind of key; empty if none"
            ),
            ColumnSpec("hours", pa.float64(), V, "operator hours, empty if not costed"),
        ),
    ),
    TableSpec(
        "catalogue",
        "log",
        "log/catalogue.parquet",
        "parquet",
        "every action the fleet can take, one row per valid verb and target, with its cost",
        (
            ColumnSpec("verb", pa.string(), V, "the action's verb"),
            ColumnSpec("scope", pa.string(), V, "fleet, a site or a robot", id_kind="scope"),
            ColumnSpec("component", pa.string(), V, "the component it acts on, if any"),
            ColumnSpec("config_key", pa.string(), V, "the configuration key, if any"),
            ColumnSpec("condition_key", pa.string(), V, "the site condition, if any"),
            ColumnSpec(
                "value", pa.string(), V, "a task class for route_to_teleop; empty: you choose it"
            ),
            ColumnSpec("hours", pa.float64(), V, "operator hours, empty if not costed"),
        ),
    ),
    TableSpec(
        "identities",
        "truth",
        "truth/identities.parquet",
        "parquet",
        "everything that fixes each episode (section 4)",
        (
            _EPISODE_ID_H,
            ColumnSpec("world_id", pa.string(), H, "the world"),
            ColumnSpec("robot_id", pa.string(), H, "the robot"),
            ColumnSpec("site_id", pa.string(), H, "the robot's site"),
            ColumnSpec("day", pa.int32(), H, "simulated day"),
            ColumnSpec("episode_index", pa.int32(), H, "the robot's episode number within the day"),
            ColumnSpec("suite", pa.string(), H, "LIBERO suite"),
            ColumnSpec("task_index", pa.int32(), H, "LIBERO task within the suite"),
            ColumnSpec("start_state", pa.int32(), H, "recorded initial state"),
            ColumnSpec("scene_seed", pa.int64(), H, "seed set before the reset"),
            ColumnSpec("noise_seed", pa.int64(), H, "seed of the policy's sampling noise"),
            ColumnSpec("batch_size", pa.int32(), H, "inference batch size of the run"),
            ColumnSpec("row", pa.int32(), H, "row in the batch, empty when free"),
            ColumnSpec("image_size", pa.int32(), H, "rendered image size in pixels"),
            ColumnSpec("max_steps", pa.int32(), H, "step limit"),
            ColumnSpec("policy_name", pa.string(), H, "nominal policy"),
            ColumnSpec("policy_revision", pa.string(), H, "nominal policy revision"),
            ColumnSpec("in_worker", pa.bool_(), H, "whether the policy ran in the worker"),
            ColumnSpec("identity_json", pa.string(), H, "the full EpisodeIdentity as JSON"),
        ),
    ),
    TableSpec(
        "faults_active",
        "truth",
        "truth/faults_active.parquet",
        "parquet",
        "the faults active in each episode, one row per fault",
        (
            _EPISODE_ID_H,
            ColumnSpec("fault_id", pa.string(), H, "the fault"),
            ColumnSpec("kind", pa.string(), H, "fault kind"),
            ColumnSpec("level", pa.float64(), H, "level on the episode's day"),
            ColumnSpec("target_scope", pa.string(), H, "where the fault sits"),
            ColumnSpec("target_component", pa.string(), H, "component it changes, if any"),
            ColumnSpec("target_config_key", pa.string(), H, "configuration key it changes, if any"),
            ColumnSpec("target_condition_key", pa.string(), H, "condition it changes, if any"),
            ColumnSpec("onset_day", pa.int32(), H, "first day in force"),
            ColumnSpec("duration_days", pa.int32(), H, "days in force, empty for open-ended"),
            ColumnSpec("drift_shape", pa.string(), H, "linear or step, empty without drift"),
            ColumnSpec("drift_rate", pa.float64(), H, "drift rate, empty without drift"),
            ColumnSpec("params", pa.string(), H, "the fault's parameters as JSON"),
            ColumnSpec(
                "nominal", pa.string(), H, "the value it replaced as JSON, empty for a component"
            ),
        ),
    ),
    TableSpec(
        "effective_policy",
        "truth",
        "truth/effective_policy.parquet",
        "parquet",
        "the policy spec each episode ran with, after the policy input hook",
        (
            _EPISODE_ID_H,
            ColumnSpec("name", pa.string(), H, "policy"),
            ColumnSpec("revision", pa.string(), H, "policy revision"),
            ColumnSpec("precision", pa.string(), H, "numeric precision"),
            ColumnSpec("control_mode", pa.string(), H, "absolute or relative"),
            ColumnSpec("actions_executed_per_chunk", pa.int32(), H, "actions executed per call"),
            ColumnSpec("sampling_steps", pa.int32(), H, "denoising steps, empty if none"),
            ColumnSpec("image_views", pa.int32(), H, "camera views fed to the policy"),
        ),
    ),
    TableSpec(
        "verdicts",
        "truth",
        "truth/verdicts.parquet",
        "parquet",
        "each episode's true outcome, LIBERO's own check and the fleet check's step",
        (
            _EPISODE_ID_H,
            ColumnSpec(
                "true_success", pa.bool_(), H, "the fleet check's verdict before the data hook"
            ),
            ColumnSpec(
                "true_reason", pa.string(), H, "why it ended, one of contracts.env.VERDICT_REASONS"
            ),
            ColumnSpec(
                "libero_first_success_step",
                pa.int32(),
                H,
                "first step LIBERO's own check fired, empty if never",
            ),
            ColumnSpec(
                "success_step", pa.int32(), H, "the step the fleet's check fired, empty if never"
            ),
        ),
    ),
    TableSpec(
        "chain",
        "truth",
        "truth/chain/{job_key}.npz",
        "npz",
        "the hash chain that proves a replay exact; row 0 is the state after the reset",
        (
            ColumnSpec(
                "fixture_hash",
                pa.string(),
                H,
                "SimEnv.fixture_hash after the reset or the restore, one per file",
                per_file=True,
            ),
            ColumnSpec("step", pa.int32(), H, "control step, 0 after the reset"),
            ColumnSpec("state_hash", pa.string(), H, "hash of mj_getState(mjSTATE_INTEGRATION)"),
            ColumnSpec("scene_image_hash", pa.string(), H, "hash of the scene image as rendered"),
            ColumnSpec("wrist_image_hash", pa.string(), H, "hash of the wrist image as rendered"),
            ColumnSpec(
                "executed_action",
                _vector(7, pa.float32()),
                H,
                "the action after the action hook, NaN in row 0",
            ),
        ),
    ),
    TableSpec(
        "replays",
        "truth",
        "truth/replays.parquet",
        "parquet",
        "the oracle rows: an episode's outcome under an effective fault list",
        (
            _EPISODE_ID_H,
            ColumnSpec("active_faults", pa.string(), H, "hash of the effective fault list"),
            ColumnSpec("success", pa.bool_(), H, "outcome of the replay"),
        ),
    ),
    TableSpec(
        "job_outcomes",
        "truth",
        "truth/job_outcomes.parquet",
        "parquet",
        "the outcome of every job of any mode: run, replay, continue, scripted",
        (
            ColumnSpec("job_key", pa.string(), H, "the job's key (contracts.rollout.Job.key)"),
            _EPISODE_ID_H,
            ColumnSpec("mode", pa.string(), H, "run, replay, continue or scripted"),
            ColumnSpec("removed", pa.string(), H, "ids of the faults removed, as a JSON list"),
            ColumnSpec("added", pa.string(), H, "the changes added, as a JSON list of FaultSpecs"),
            ColumnSpec(
                "from_snapshot_step", pa.int32(), H, "step of the snapshot it started from, if any"
            ),
            ColumnSpec("success", pa.bool_(), H, "the job's true outcome"),
            ColumnSpec("steps", pa.int32(), H, "control steps run, the episode's last step"),
            ColumnSpec(
                "placed_step",
                pa.int32(),
                H,
                "first step the fleet's check found the object placed, empty if never",
            ),
            ColumnSpec(
                "libero_step", pa.int32(), H, "first step LIBERO's own check fired, empty if never"
            ),
        ),
    ),
    TableSpec(
        "oracle",
        "truth",
        "truth/oracle.parquet",
        "parquet",
        "gains derived from the oracle rows",
        (
            ColumnSpec("item", pa.string(), H, "a fault id, a pair, or a fix set's name"),
            ColumnSpec(
                "item_kind", pa.string(), H, "fault_effect, interaction, fix_set or achievable"
            ),
            ColumnSpec("rescued", pa.int32(), H, "episodes turned from failure to success"),
            ColumnSpec("broken", pa.int32(), H, "episodes turned from success to failure"),
            ColumnSpec("net", pa.int32(), H, "rescued minus broken"),
        ),
    ),
    TableSpec(
        "decisions",
        "truth",
        "truth/decisions.parquet",
        "parquet",
        "the supervisor's decision trace",
        (
            _EPISODE_ID_H,
            ColumnSpec("step", pa.int32(), H, "step of the decision"),
            ColumnSpec("decision", pa.string(), H, "the decision"),
            ColumnSpec("hold_steps", pa.int32(), H, "steps held still while deciding"),
            ColumnSpec("rationale", pa.string(), H, "supervisor's reason for the decision"),
            ColumnSpec(
                "call_status", pa.string(), H, "ok, failed, skipped route, or deterministic rule"
            ),
            ColumnSpec("error", pa.string(), H, "call failure, if any"),
            ColumnSpec(
                "retry_budget_steps",
                pa.int32(),
                H,
                "retry's own active budget from primitive start",
            ),
            ColumnSpec(
                "decision_ms",
                pa.float64(),
                H,
                "time of the call, subscription route",
                clock="measured",
            ),
            ColumnSpec(
                "model_ms", pa.float64(), H, "the model's own response time", clock="measured"
            ),
        ),
    ),
)

# Probes the agent composes (`contracts.probes`): built by the loop's export from the
# probe runs, never by a run itself (`TableSpec.source`).
#
# The timing of `probe_steps` (as of b38534e), which the engine's rollout keeps and the
# export copies row for row from the run's `steps`: row k is one instant, k control steps
# (k / 20 s) after the probe started, and `time_s` is that instant. At it the robot's state
# is measured first (`eef_pos`, `eef_quat`, `joint_pos`, `joint_vel`, `gripper_opening`,
# as the robot reports them), and then the row's command is issued (`target_pos`,
# `gripper_command`): the target as the probe's controller submitted it, before anything
# on the command path acts on it. The command acts on the arm during the next control
# step, so with nothing on the path a command shows first in the measurement of row k + 1.
# The last row, at step N (N / 20 s, N the program's commands), is the state after the
# N-th command; it has no command. Nothing in the table is a time of the command path.
_PROBE_ID = ColumnSpec("probe_id", pa.string(), V, "the probe, as request_probe numbered it")
_PROBE_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "probe_episodes",
        "log",
        "log/probe_episodes.parquet",
        "parquet",
        "one row per probe episode: a probe's scripted motion played once on one robot, "
        "without the policy and without a task",
        (
            dataclasses.replace(_PROBE_ID, id_kind=IdKind.probe),
            _EPISODE_ID_V,
            ColumnSpec(
                "robot_id", pa.string(), V, "the robot that played it", id_kind=IdKind.robot
            ),
            ColumnSpec("site_id", pa.string(), V, "the robot's site", id_kind=IdKind.site),
            ColumnSpec("day", pa.int32(), V, "simulated day, from 0"),
            ColumnSpec("repeat", pa.int32(), V, "which repetition on this robot, from 0"),
            ColumnSpec("steps", pa.int32(), V, "control steps commanded (20 per second)"),
        ),
        source="export",
    ),
    TableSpec(
        "probe_segments",
        "log",
        "log/probe_segments.parquet",
        "parquet",
        "the program of each probe as it ran, one row per segment",
        (
            dataclasses.replace(_PROBE_ID, id_kind=IdKind.probe),
            ColumnSpec("segment", pa.int32(), V, "the segment's place in the program, from 0"),
            ColumnSpec(
                "primitive", pa.string(), V, "hold, step, ramp, sine, open_gripper or close_gripper"
            ),
            ColumnSpec("axis", pa.string(), V, "x, y or z of the end effector, empty if none"),
            ColumnSpec(
                "amplitude_m", pa.float64(), V, "step and ramp: offset reached; sine: amplitude"
            ),
            ColumnSpec("duration_steps", pa.int32(), V, "the segment's control steps"),
            ColumnSpec("period_steps", pa.int32(), V, "a sine's period in control steps"),
            ColumnSpec("first_step", pa.int32(), V, "the probe step the segment starts at"),
        ),
        source="export",
    ),
    TableSpec(
        "probe_steps",
        "log",
        "log/probe_steps/{episode_id}.parquet",
        "parquet",
        "one row per control step of one probe episode: the command and the measured response",
        (
            _EPISODE_ID_V,
            ColumnSpec(
                "step", pa.int32(), V, "row k: the state after k commands, then command k (from 0)"
            ),
            ColumnSpec(
                "time_s",
                pa.float64(),
                V,
                "step / 20: when the row's state was measured and its command issued",
                clock="simulated",
            ),
            ColumnSpec(
                "segment", pa.int32(), V, "the segment issuing the row's command, empty in the last"
            ),
            ColumnSpec(
                "target_pos",
                _vector(3, pa.float32()),
                V,
                "end effector position issued at this row in metres, as the probe submitted it",
            ),
            ColumnSpec(
                "gripper_command",
                pa.float32(),
                V,
                "gripper command issued at this row, as submitted: -1 open, 1 closed",
            ),
            ColumnSpec(
                "eef_pos", _vector(3), V, "end effector position measured at this row, in metres"
            ),
            ColumnSpec("eef_quat", _vector(4), V, "end effector orientation measured at this row"),
            ColumnSpec("joint_pos", _vector(7), V, "joint positions measured at this row, radians"),
            ColumnSpec(
                "joint_vel", _vector(7), V, "joint velocities measured at this row, radians/s"
            ),
            ColumnSpec(
                "gripper_opening", pa.float64(), V, "finger opening measured at this row, metres"
            ),
        ),
        source="export",
    ),
)
_TABLES += _PROBE_TABLES

# Jobs (`contracts.jobs`, docs/design/jobs.md): goal-suite skills chained into
# one job in one continuous simulator state, run by `fleetlab.jobs` and visible only in a
# job run's tree (`TableSpec.source`). Row k of a per-step table is one instant, k control
# steps after the reset: the state is measured first and then the row's command is issued,
# as in `steps`; the last row is the state after the last command and has no command. A
# requested goal is an index into the request's goals; scene names are LIBERO's.
_JOB_ID = ColumnSpec("job_id", pa.string(), V, "the job, as the runtime numbered it (j0001)")
_CONTROL_STEP = ColumnSpec("step", pa.int32(), V, "control step from the reset (20 per second)")
_JOB_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "job_episodes",
        "log",
        "log/job_episodes.parquet",
        "parquet",
        "one row per run of a job's program from one start: how it ended and whether the "
        "host's delivery check found the job delivered",
        (
            _JOB_ID,
            _EPISODE_ID_V,
            ColumnSpec("program_digest", pa.string(), V, "SHA-256 of the program the run played"),
            ColumnSpec(
                "base_task", pa.int32(), V, "the goal-suite task whose scene was loaded (0 to 9)"
            ),
            ColumnSpec(
                "init_state", pa.int32(), V, "the stored initial state the run started from"
            ),
            ColumnSpec(
                "control_steps", pa.int32(), V, "control steps from the reset to the end of the run"
            ),
            ColumnSpec(
                "tail_steps",
                pa.int32(),
                V,
                "control steps the policy went on after the program, until delivered or 60",
            ),
            ColumnSpec(
                "delivered", pa.bool_(), V, "the host's delivery check at the end of the run"
            ),
            ColumnSpec(
                "delivery_detail", pa.string(), V, "what the delivery check found, in words"
            ),
            ColumnSpec(
                "goals_held_at_end",
                pa.list_(pa.bool_()),
                V,
                "per requested goal, in order, whether it held at the end",
            ),
            ColumnSpec(
                "robot_touching_at_end",
                pa.list_(pa.string()),
                V,
                "objects the goals name that a robot body touched at the end",
            ),
            ColumnSpec(
                "moving_at_end",
                pa.list_(pa.string()),
                V,
                "objects the goals name that were not at rest at the end",
            ),
            ColumnSpec(
                "gripper_opening_at_end", pa.float64(), V, "finger opening at the end, in metres"
            ),
            ColumnSpec(
                "order_kept", pa.bool_(), V, "whether every order rule held on this run's trace"
            ),
            ColumnSpec("order_detail", pa.string(), V, "each order rule's verdict, in words"),
            ColumnSpec(
                "start_eef_pos",
                _vector(3),
                V,
                "end effector position reported when the job started, in metres",
            ),
            ColumnSpec(
                "start_eef_quat",
                _vector(4),
                V,
                "end effector orientation reported when the job started, as a quaternion",
            ),
            ColumnSpec(
                "video_scene",
                pa.string(),
                V,
                "scene camera video, relative to log/, empty without video",
            ),
            ColumnSpec(
                "video_wrist",
                pa.string(),
                V,
                "wrist camera video, relative to log/, empty without video",
            ),
        ),
        source="export",
    ),
    TableSpec(
        "job_step_events",
        "log",
        "log/job_step_events.parquet",
        "parquet",
        "one row per program step of each run, and one for the delivery tail if it ran: "
        "where it began and ended and how",
        (
            _EPISODE_ID_V,
            ColumnSpec(
                "program_step",
                pa.int32(),
                V,
                "the step's place in the program, from 0; empty for the delivery tail",
            ),
            ColumnSpec("kind", pa.string(), V, "policy, motion, wait, home, or tail"),
            ColumnSpec(
                "start_step",
                pa.int32(),
                V,
                "control step at which the step began, empty if it was skipped",
            ),
            ColumnSpec(
                "end_step", pa.int32(), V, "control step at which it ended, empty if skipped"
            ),
            ColumnSpec("steps_used", pa.int32(), V, "control steps it ran"),
            ColumnSpec(
                "outcome",
                pa.string(),
                V,
                "reached, timeout or skipped; for the tail, reached when delivered within it",
            ),
            ColumnSpec(
                "instruction", pa.string(), V, "the instruction of a policy step or the tail"
            ),
        ),
        source="export",
    ),
    TableSpec(
        "object_steps",
        "log",
        "log/object_steps/{episode_id}.parquet",
        "parquet",
        "one row per control step and movable object of one run: pose, speed and contacts",
        (
            _EPISODE_ID_V,
            _CONTROL_STEP,
            ColumnSpec("object", pa.string(), V, "the object, as the scene names it"),
            ColumnSpec("position", _vector(3), V, "position in the world frame, in metres"),
            ColumnSpec("quaternion", _vector(4), V, "orientation as a quaternion (w, x, y, z)"),
            ColumnSpec("linear_speed", pa.float64(), V, "speed in metres per second"),
            ColumnSpec("angular_speed", pa.float64(), V, "rotation speed in radians per second"),
            ColumnSpec(
                "in_contact_with_robot", pa.bool_(), V, "whether a robot body touches the object"
            ),
            ColumnSpec(
                "in_contact_with",
                pa.list_(pa.string()),
                V,
                "every body touching the object, robot bodies by their own names",
            ),
        ),
        source="export",
    ),
    TableSpec(
        "goal_steps",
        "log",
        "log/goal_steps/{episode_id}.parquet",
        "parquet",
        "one row per control step and requested goal of one run: whether LIBERO's own "
        "predicate of the goal held",
        (
            _EPISODE_ID_V,
            _CONTROL_STEP,
            ColumnSpec("goal", pa.int32(), V, "the requested goal, as an index from 0"),
            ColumnSpec("holds", pa.bool_(), V, "whether the goal held at this step"),
        ),
        source="export",
    ),
    TableSpec(
        "robot_steps",
        "log",
        "log/robot_steps/{episode_id}.parquet",
        "parquet",
        "one row per control step of one run: the hand, the gripper and the command issued",
        (
            _EPISODE_ID_V,
            _CONTROL_STEP,
            ColumnSpec("eef_pos", _vector(3), V, "end effector position in metres"),
            ColumnSpec("eef_quat", _vector(4), V, "end effector orientation as a quaternion"),
            ColumnSpec("gripper_opening", pa.float64(), V, "finger opening in metres"),
            ColumnSpec(
                "commanded_action",
                _vector(7, pa.float32()),
                V,
                "the action issued at this step (position, rotation, gripper); empty in last row",
            ),
            ColumnSpec(
                "gripper_command",
                pa.float32(),
                V,
                "gripper command issued at this step: -1 open, 1 closed; empty in the last row",
            ),
            ColumnSpec(
                "instruction",
                pa.string(),
                V,
                "the instruction in force, empty during motion, wait and home steps",
            ),
            ColumnSpec(
                "program_step",
                pa.int32(),
                V,
                "the program step issuing the command, empty in the tail and the last row",
            ),
            ColumnSpec("phase", pa.string(), V, "program, tail, or end for the last row"),
            ColumnSpec(
                "frame_index", pa.int32(), V, "the step's frame in the videos, empty if it has none"
            ),
        ),
        source="export",
    ),
)
_TABLES += _JOB_TABLES

SCHEMA: dict[str, TableSpec] = {table.name: table for table in _TABLES}
# The tables only an export holds, by the package that builds them (`TableSpec.source`).
PROBE_TABLE_NAMES: tuple[str, ...] = tuple(t.name for t in _PROBE_TABLES)
JOB_TABLE_NAMES: tuple[str, ...] = tuple(t.name for t in _JOB_TABLES)

# Files in log/ that are not tables: the system description and the videos.
DESCRIPTION_FILE = "log/description.json"
VIDEO_PATH = "log/video/{episode_id}.{camera}.mp4"
# Directories and files of the truth side that are not tables.
SNAPSHOT_DIR = "truth/snapshots"
RUN_RECORD = "run.json"

# Column names that read like measured time; refused in any visible file.
MEASURED_TIME_NAME = re.compile(r"(_ms$|_ns$|latency|wall|elapsed|wait)")


def schema_problems(schema: dict[str, TableSpec] | None = None) -> list[str]:
    """What is wrong with a schema: tags, descriptions, trees, clocks. Empty when sound."""
    problems: list[str] = []
    for table in (schema or SCHEMA).values():
        names = [c.name for c in table.columns]
        if len(names) != len(set(names)):
            problems.append(f"{table.name}: a column name repeats")
        if not table.description.strip():
            problems.append(f"{table.name}: no description")
        for column in table.columns:
            where = f"{table.name}.{column.name}"
            if not isinstance(column.visibility, Visibility):
                problems.append(f"{where}: no visibility tag")
            if not column.description.strip() or "\n" in column.description:
                problems.append(f"{where}: needs a one-line description")
            if not isinstance(column.type, pa.DataType):
                problems.append(f"{where}: type is not an Arrow type")
            expected = Visibility.visible if table.tree == "log" else Visibility.hidden
            if column.visibility is not expected:
                problems.append(f"{where}: tagged {column.visibility} in the {table.tree} tree")
            if column.visibility is Visibility.visible and column.clock == "measured":
                problems.append(f"{where}: a measured time in a visible table")
            if column.visibility is Visibility.visible and MEASURED_TIME_NAME.search(column.name):
                problems.append(f"{where}: reads like a measured time")
    return problems


def visible_schema() -> dict[str, TableSpec]:
    """The tables and columns the agents may see, built from the columns tagged visible."""
    out: dict[str, TableSpec] = {}
    for table in SCHEMA.values():
        columns = tuple(c for c in table.columns if c.visibility is Visibility.visible)
        if columns:
            out[table.name] = TableSpec(
                table.name,
                table.tree,
                table.path,
                table.storage,
                table.description,
                columns,
                table.source,
            )
    return out


def visible_table_names() -> tuple[str, ...]:
    """The names of the tables the agents may query."""
    return tuple(visible_schema())


def allowed_columns(table: str) -> tuple[str, ...]:
    """The columns of `table` that may be written to the visible log (empty if none)."""
    spec = visible_schema().get(table)
    return () if spec is None else tuple(c.name for c in spec.columns)


def arrow_schema(table: str, visible_only: bool = False) -> Any:
    """The Arrow schema of a table: all its columns, or only the visible ones.

    A table in `log/` holds only visible columns, so both give the same there;
    `visible_only=True` for a truth table raises, since it has none.
    """
    if visible_only:
        if table not in visible_schema():
            raise ValueError(f"table {table!r} has no visible columns; it lives on the truth side")
        spec = visible_schema()[table]
    else:
        spec = SCHEMA[table]
    return pa.schema([pa.field(c.name, c.type) for c in spec.columns])


def type_name(kind: Any) -> str:
    """A column type as the data dictionary shows it: a vector as `double[7]` or `float[7]`.

    Arrow writes a fixed-size list as `fixed_size_list<item: double>[7]`, which
    matches neither what DuckDB shows (`DOUBLE[]`) nor how a reader thinks of
    it; other types keep Arrow's name.
    """
    if pa.types.is_fixed_size_list(kind):
        return f"{kind.value_type}[{kind.list_size}]"
    return str(kind)


def _table_markdown(table: TableSpec) -> list[str]:
    path = table.path.removeprefix("log/")
    lines = [f"## {table.name}", "", f"`{path}`: {table.description}.", ""]
    lines += ["| column | type | description |", "|---|---|---|"]
    lines += [f"| {c.name} | {type_name(c.type)} | {c.description} |" for c in table.columns]
    lines.append("")
    return lines


def export_table_markdown(tables: tuple[str, ...] = PROBE_TABLE_NAMES) -> str:
    """The dictionary's sections of tables only an export holds (`TableSpec.source`), for an
    export that holds them: by default the probe tables, which the loop's exports add; a job
    run's tree passes `JOB_TABLE_NAMES`. The tables of every run are
    `data_dictionary_markdown`."""
    lines: list[str] = []
    for table in visible_schema().values():
        if table.source == "export" and table.name in tables:
            lines += _table_markdown(table)
    return "\n".join(lines)


def data_dictionary_markdown() -> str:
    """The data dictionary the agents receive, generated from the visible part of the schema.

    The tables only an export holds are not listed: an export that holds them adds their
    sections (`export_table_markdown`).
    """
    lines = ["# Data dictionary", ""]
    for table in visible_schema().values():
        if table.source == "run":
            lines += _table_markdown(table)
    lines += [
        "## Other files",
        "",
        "`description.json`: the system description: vocabulary, fleet structure, "
        "configuration, logs and costs; the action catalogue is the table `catalogue`.",
        f"`video/<episode_id>.<camera>.mp4` for camera in {', '.join(c.value for c in Camera)}: "
        "the episode's video; `steps.frame_index` says which steps have a frame.",
        "",
    ]
    return "\n".join(lines)


def _path_pattern(template: str) -> re.Pattern[str]:
    escaped = re.escape(template)
    escaped = escaped.replace(re.escape("{episode_id}"), r"(?P<episode_id>[^/]+)")
    escaped = escaped.replace(re.escape("{camera}"), r"(?P<camera>[^/.]+)")
    return re.compile(f"^{escaped}$")


def _value_problem(kind: IdKind | Literal["scope"], value: Any) -> bool:
    if value is None:
        return False
    if kind == "scope":
        try:
            check_scope(value)
        except ValueError:
            return True
        return False
    return not is_neutral(kind, value)


def log_tree_problems(run_dir: Path | str) -> list[str]:
    """Everything in a run's `log/` tree that the export must refuse; empty when clean."""
    root = Path(run_dir)
    log_dir = root / "log"
    if not log_dir.is_dir():
        return [f"{log_dir} does not exist"]
    tables = [(t, _path_pattern(t.path)) for t in visible_schema().values()]
    video = _path_pattern(VIDEO_PATH)
    problems: list[str] = []
    for path in sorted(p for p in log_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == DESCRIPTION_FILE:
            problems += _description_problems(path)
            continue
        match = video.match(relative)
        if match:
            if not is_neutral(IdKind.episode, match["episode_id"]):
                problems.append(f"{relative}: episode id in the name is not neutral")
            if match["camera"] not in {c.value for c in Camera}:
                problems.append(f"{relative}: unknown camera")
            continue
        found = [(t, m) for t, pattern in tables if (m := pattern.match(relative))]
        if not found:
            problems.append(f"{relative}: not a file of the visible log")
            continue
        table, m = found[0]
        if table.per_episode and not is_neutral(IdKind.episode, m["episode_id"]):
            problems.append(f"{relative}: episode id in the name is not neutral")
        problems += _table_problems(path, relative, table)
    return problems


def _table_problems(path: Path, relative: str, table: TableSpec) -> list[str]:
    problems: list[str] = []
    try:
        names = list(pq.read_schema(path).names)
    except Exception as error:  # pyarrow raises several types for an unreadable file
        return [f"{relative}: unreadable as Parquet ({error})"]
    allowed = set(allowed_columns(table.name))
    for name in names:
        if name not in allowed:
            problems.append(f"{relative}: column {name!r} is not on the visible allow-list")
        if MEASURED_TIME_NAME.search(name):
            problems.append(f"{relative}: column {name!r} reads like a measured time")
    checked = [c for c in table.columns if c.id_kind is not None and c.name in names]
    if checked:
        data = pq.read_table(path, columns=[c.name for c in checked])
        for column in checked:
            assert column.id_kind is not None
            bad = sorted(
                {
                    str(v)
                    for v in data.column(column.name).to_pylist()
                    if _value_problem(column.id_kind, v)
                }
            )
            if bad:
                problems.append(f"{relative}: {column.name} values not neutral: {bad[:5]}")
    return problems


def _description_problems(path: Path) -> list[str]:
    """The description is built from visible fields; check that no hidden field name appears."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return [f"{DESCRIPTION_FILE}: unreadable ({error})"]
    hidden = {
        name
        for model in MODEL_CLASSES
        for name in model.model_fields
        if field_visibility(model, name) is Visibility.hidden
    }
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            found.update(k for k in value if k in hidden)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return [f"{DESCRIPTION_FILE}: hidden field {name!r}" for name in sorted(found)]


class LogLeakError(ValueError):
    """A run's visible log holds something the agents must not see."""


def validate_log_tree(run_dir: Path | str) -> None:
    """Raise `LogLeakError` listing every problem in the run's `log/` tree."""
    problems = log_tree_problems(run_dir)
    if problems:
        raise LogLeakError(f"{len(problems)} problems in {run_dir}/log:\n" + "\n".join(problems))
