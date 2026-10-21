"""The run writer: one run directory in the layout of docs/design/contracts.md, section 11.

    with RunWriter(run_dir, record, episodes_per_robot_day=n) as run:  # before the workers
        run.write_model(fleet, task_classes)     # description.json, change_log, maintenance
        for job in run.pending(jobs):            # what is left after a crash
            episode = run.episode(job, task_text=...)  # or EpisodeWriter(run_dir, job, ...)
            ...  episode.add_step(telemetry, frames)   # per state; frames upright, both cameras
            ...  episode.add_chain(step, state_hash, image_hashes, executed_action)
            episode.set_verdict(...)
            episode.finish(outcome=...)          # a replay: finish(success=...)
        run.finalize()                           # run-level tables from the finished jobs

An episode writer buffers one job: its steps, its chain, its video frames and
its rows for the run-level tables. `finish` writes the videos, the steps file
and the job's staged rows (`truth/parts/<job key>.json`), and the chain last:
a job whose chain and staged rows exist is complete (`RunWriter.is_done`).
Every file goes through a temporary name and a rename. `finalize` builds the
run-level tables from the staged rows of the complete jobs, sorted by episode,
so that they do not depend on the order in which jobs finished; it can run
again at any time.

Resuming: opening a `RunWriter` on an existing run checks that the run record
agrees (software, batch size, replay rules, episodes per robot and day; the
rest of the record, `extra` included, is kept from the file), deletes what a
killed writer left in the staging directory and the files of jobs without a
chain, and holds a lock so that no second writer opens the same run. The
engine then runs `run.pending(jobs)` and finalizes.

What a job writes depends on its mode. A "run" job writes the visible log
(steps, video, its episodes and events rows) and its hidden rows (identity,
faults, effective policy, verdict, decisions) and chain. Every other mode
writes no visible file: a "replay" from the reset adds one `replays` row (the
outcome under its effective fault list); "continue", "scripted" and replays
from a snapshot write their chain only. Every job adds one row to
`job_outcomes`.
"""

from __future__ import annotations

import datetime
import fcntl
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from fleetlab.contracts.actions import catalogue_rows, costs_rows
from fleetlab.contracts.env import VERDICT_REASONS, Camera
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, started_at_for
from fleetlab.contracts.ids import IdKind, is_neutral
from fleetlab.contracts.live import DecisionRecord, StepTelemetry
from fleetlab.contracts.log import (
    RUN_RECORD,
    SCHEMA,
    VIDEO_PATH,
    EventKind,
    Outcome,
    schema_problems,
)
from fleetlab.contracts.model import Fleet, describe_for_agents
from fleetlab.contracts.oracle import fault_list_hash
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, RunRecord
from fleetlab.log._files import (
    CANARY_METADATA_KEY,
    LOCK_FILE,
    MERGED_FILE,
    PARTS_DIR,
    STAGING_DIR,
    clear_staging,
    plant_canary,
    read_canary,
    staged,
    write_parquet,
    write_text,
)
from fleetlab.log._rows import (
    EPISODE_TABLES,
    RUN_TABLES,
    RowError,
    Rows,
    action,
    check_keys,
    column_names,
    decode_part,
    encode_part,
    file_columns,
    pylist,
    sort_table,
    table_columns,
    to_arrow,
    vector,
)
from fleetlab.video import write_mp4

__all__ = [
    "EpisodeFiles",
    "EpisodeWriter",
    "ResumeMismatch",
    "RowError",
    "RunLocked",
    "RunStatus",
    "RunWriter",
    "SchemaError",
    "VideoSettings",
    "run_status",
]

_HASH = re.compile(r"^[0-9a-f]{16}$")
# The key under which the run record keeps the world's episodes per robot and day.
EPISODES_PER_ROBOT_DAY = "episodes_per_robot_day"


class SchemaError(RuntimeError):
    """`contracts.log.SCHEMA` has problems; nothing is written until they are fixed."""


class ResumeMismatch(RuntimeError):
    """An existing run directory was made with other software or another batch size."""


class RunLocked(RuntimeError):
    """Another writer holds the run directory."""


@dataclass(frozen=True)
class VideoSettings:
    """How episode videos are encoded: libx264 at this rate and quality, one thread, no metadata.

    The rate only sets playback speed: which steps have a frame is the
    caller's choice and `steps.frame_index` records it.
    """

    fps: int = 5
    crf: int = 23
    preset: str = "veryfast"


@dataclass(frozen=True)
class EpisodeFiles:
    """What `EpisodeWriter.finish` wrote: paths and sizes by kind of file."""

    key: str
    paths: dict[str, Path]
    sizes: dict[str, int]


@dataclass(frozen=True)
class RunStatus:
    """Where a run directory stands.

    `done`: job keys with a chain and staged rows (and a steps file for a run
    job). `partial`: keys with some files but no chain, left by a killed
    writer or still being written. `pending`: of the jobs asked about, those
    not done. `merged`: the keys the run-level tables were last built from.
    """

    done: tuple[str, ...]
    partial: tuple[str, ...]
    pending: tuple[str, ...] = ()
    merged: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()  # run-level files that do not exist yet
    staging_files: int = 0

    @property
    def finalized(self) -> bool:
        """Complete and consistent: nothing partial, every table built from every finished job."""
        return (
            not self.partial
            and not self.missing
            and self.staging_files == 0
            and set(self.done) == set(self.merged)
        )


def _key_of_chain(path: Path) -> str:
    return path.name.removesuffix(".npz")


def _is_run_key(key: str) -> bool:
    return is_neutral(IdKind.episode, key)


def _run_level_files() -> list[str]:
    return [SCHEMA[t].path for t in (*EPISODE_TABLES, *RUN_TABLES)]


def run_status(run_dir: Path | str, jobs: Iterable[Job] | None = None) -> RunStatus:
    """The status of a run directory, read from its files alone."""
    root = Path(run_dir)
    chains = {_key_of_chain(p) for p in (root / "truth" / "chain").glob("*.npz")}
    parts = {p.stem for p in (root / PARTS_DIR).glob("*.json") if p.name != Path(MERGED_FILE).name}
    steps = {p.stem for p in (root / "log" / "steps").glob("*.parquet")}
    videos = {p.name.split(".")[0] for p in (root / "log" / "video").glob("*.mp4")}
    done = {k for k in chains & parts if not _is_run_key(k) or k in steps}
    touched = chains | parts | steps | videos
    partial = touched - done
    # A video or steps file belongs to a run job; it is partial only if that job is not done.
    pending: tuple[str, ...] = ()
    if jobs is not None:
        pending = tuple(j.key for j in jobs if j.key not in done)
    merged: tuple[str, ...] = ()
    merged_path = root / MERGED_FILE
    if merged_path.is_file():
        merged = tuple(json.loads(merged_path.read_text()))
    missing = tuple(p for p in _run_level_files() if not (root / p).is_file())
    staging = root / STAGING_DIR
    staging_files = sum(1 for p in staging.rglob("*") if p.is_file()) if staging.is_dir() else 0
    return RunStatus(
        done=tuple(sorted(done)),
        partial=tuple(sorted(partial)),
        pending=pending,
        merged=merged,
        missing=missing,
        staging_files=staging_files,
    )


def _part_path(root: Path, key: str) -> Path:
    return root / PARTS_DIR / f"{key}.json"


def _canary_metadata(root: Path, table: str) -> dict[bytes, bytes] | None:
    """The canary goes into the metadata of every hidden table."""
    if SCHEMA[table].tree != "truth":
        return None
    token = read_canary(root)
    return {CANARY_METADATA_KEY: token.encode()} if token else None


def _write_table(root: Path, table: str, data: pa.Table) -> int:
    path = root / SCHEMA[table].path
    return write_parquet(sort_table(table, data), path, root, _canary_metadata(root, table))


def _select(table: str, derived: Mapping[str, Any]) -> dict[str, Any]:
    """The row of `table` from a dictionary of derived values: exactly the declared columns."""
    names = column_names(table)
    missing = [n for n in names if n not in derived]
    if missing:
        raise RowError(f"the writer cannot derive {missing} of table {table}")
    return {n: derived[n] for n in names}


def _json_value(value: Any) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"), sort_keys=True)


def identity_row(identity: EpisodeIdentity, in_worker: bool) -> dict[str, Any]:
    """The `identities` row of an episode."""
    data = identity.model_dump(mode="json")
    derived = {
        **{k: v for k, v in data.items() if not isinstance(v, dict | list)},
        "policy_name": identity.policy.name,
        "policy_revision": identity.policy.revision,
        "in_worker": in_worker,
        "identity_json": identity.model_dump_json(),
    }
    return _select("identities", derived)


def fault_rows(identity: EpisodeIdentity) -> list[dict[str, Any]]:
    """The `faults_active` rows of an episode, one per fault in its identity."""
    rows = []
    for fault in identity.faults:
        drift = fault.schedule.drift
        derived = {
            "episode_id": identity.episode_id,
            "fault_id": fault.fault_id,
            "kind": fault.kind,
            "level": fault.level,
            "target_scope": fault.target.scope,
            "target_component": _enum(fault.target.component),
            "target_config_key": _enum(fault.target.config_key),
            "target_condition_key": _enum(fault.target.condition_key),
            "onset_day": fault.schedule.onset_day,
            "duration_days": fault.schedule.duration_days,
            "drift_shape": None if drift is None else drift.shape,
            "drift_rate": None if drift is None else drift.rate,
            "params": _json_value(fault.params),
            "nominal": _json_value(fault.nominal),
        }
        rows.append(_select("faults_active", derived))
    return rows


def policy_row(episode_id: str, spec: PolicySpec) -> dict[str, Any]:
    """The `effective_policy` row of an episode."""
    return _select("effective_policy", {"episode_id": episode_id, **spec.model_dump(mode="json")})


def _enum(value: Any) -> Any:
    return None if value is None else getattr(value, "value", value)


def _telemetry(telemetry: StepTelemetry | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(telemetry, StepTelemetry):
        source: dict[str, Any] = {
            "step": telemetry.step,
            "time_s": telemetry.time_s,
            "joint_pos": telemetry.joint_pos,
            "joint_vel": telemetry.joint_vel,
            "gripper_qpos": telemetry.gripper_qpos,
            "eef_pos": telemetry.eef_pos,
            "eef_quat": telemetry.eef_quat,
            "commanded_action": telemetry.commanded_action,
            "chunk_index": telemetry.chunk_index,
            "chunk_step": telemetry.chunk_step,
        }
    else:
        source = dict(telemetry)
    lengths = {"joint_pos": 7, "joint_vel": 7, "gripper_qpos": 2, "eef_pos": 3, "eef_quat": 4}
    for name, n in lengths.items():
        if name in source:
            source[name] = vector(source[name], n)
    if "commanded_action" in source:
        source["commanded_action"] = action(source["commanded_action"])
    for name in ("step", "chunk_index", "chunk_step"):
        if isinstance(source.get(name), np.integer):
            source[name] = int(source[name])
    if isinstance(source.get("time_s"), np.floating):
        source["time_s"] = float(source["time_s"])
    return source


def _nan_action() -> list[float]:
    return [float("nan")] * 7


class EpisodeWriter:
    """Buffers one job's rows, frames and chain, and writes them in `finish`.

    Can be built in a worker process from the run directory and the job; the
    run directory must have been opened by a `RunWriter` first.
    """

    def __init__(
        self,
        run_dir: Path | str,
        job: Job,
        *,
        task_text: str,
        episodes_per_robot_day: int,
        in_worker: bool = False,
        video: VideoSettings | None = None,
    ) -> None:
        self.layout = RunLayout(Path(run_dir))
        if not self.layout.run_record.is_file():
            raise FileNotFoundError(
                f"{self.layout.root} has no {RUN_RECORD}: open it with RunWriter"
            )
        record = RunRecord.model_validate_json(self.layout.run_record.read_text())
        if record.extra.get(EPISODES_PER_ROBOT_DAY) != episodes_per_robot_day:
            raise ResumeMismatch(
                f"{self.layout.root} runs {record.extra.get(EPISODES_PER_ROBOT_DAY)!r} episodes "
                f"per robot and day, not {episodes_per_robot_day!r}"
            )
        self.episodes_per_robot_day = episodes_per_robot_day
        # The simulated start time comes from the identity (contracts.identity.started_at_for).
        self.started_at = datetime.time.fromisoformat(
            started_at_for(job.identity, episodes_per_robot_day)
        )
        self.job = job
        self.identity = job.identity
        self.episode_id = job.identity.episode_id
        self.key = job.key
        self.task_text = task_text
        self.in_worker = in_worker
        self.video = video or VideoSettings()
        self._steps: list[dict[str, Any]] = []
        self._chain: list[dict[str, Any]] = []
        self._frames: dict[Camera, list[npt.NDArray[np.uint8]]] = {c: [] for c in Camera}
        self._events: list[dict[str, Any]] = []
        self._decisions: list[dict[str, Any]] = []
        self._policy: PolicySpec | None = None
        self._verdict: dict[str, Any] | None = None
        self._placed_step: int | None = None
        self._fixture_hash: str | None = None
        self._finished = False

    @property
    def visible(self) -> bool:
        """Whether this job writes the visible log: only a "run" job does."""
        return self.job.mode is JobMode.run

    @property
    def first_chain_step(self) -> int:
        snapshot = self.job.from_snapshot
        return 0 if snapshot is None else snapshot.step

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError(f"job {self.key} is finished")

    def _check_visible(self, what: str) -> None:
        if not self.visible:
            raise RowError(f"a {self.job.mode.value} job writes nothing visible ({what})")

    def _check_run(self, what: str) -> None:
        if not self.visible:
            raise RowError(f"{what} belongs to the run job, not to a {self.job.mode.value} job")

    def add_step(
        self,
        telemetry: StepTelemetry | Mapping[str, Any],
        frames: Mapping[Camera, npt.NDArray[np.uint8]] | None = None,
    ) -> int | None:
        """One row of `steps`: the state after `step` actions and the action commanded there.

        `frames`, if given, are both cameras' images at this step, upright (flip
        LIBERO's rendered images first); the step gets the next frame index,
        which is returned. A commanded action that is None or all NaN is stored
        empty, as in the last row of an episode.
        """
        self._check_open()
        self._check_visible("steps")
        row = _telemetry(telemetry)
        if "frame_index" in row:
            raise RowError("frame_index is assigned by the writer from the frames given")
        row.setdefault("episode_id", self.episode_id)
        if row["episode_id"] != self.episode_id:
            raise RowError(f"a row of episode {row['episode_id']} given to {self.episode_id}")
        expected = len(self._steps)
        if row.get("step") != expected:
            raise RowError(f"step {row.get('step')} given where step {expected} is due")
        row["frame_index"] = None if frames is None else self._add_frames(frames)
        check_keys("steps", row)
        self._steps.append(row)
        frame_index: int | None = row["frame_index"]
        return frame_index

    def _add_frames(self, frames: Mapping[Camera, npt.NDArray[np.uint8]]) -> int:
        by_camera = {Camera(k): v for k, v in frames.items()}
        if set(by_camera) != set(Camera):
            raise RowError(f"frames for {sorted(by_camera)} given; every step needs all cameras")
        checked = {}
        for camera, image in by_camera.items():
            array = np.asarray(image)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise RowError(f"{camera.value} frame: expected height x width x 3 uint8")
            if array.shape[0] % 2 or array.shape[1] % 2:
                raise RowError(f"{camera.value} frame: H.264 needs an even height and width")
            earlier = self._frames[camera]
            if earlier and earlier[0].shape != array.shape:
                raise RowError(f"{camera.value} frame changed size within the episode")
            checked[camera] = np.array(array, copy=True)
        index = len(self._frames[Camera.scene])
        for camera, array in checked.items():
            self._frames[camera].append(array)
        return index

    def set_fixture_hash(self, fixture_hash: str) -> None:
        """`SimEnv.fixture_hash` after the reset or the restore, as hex: one per chain file."""
        self._check_open()
        if not isinstance(fixture_hash, str) or not _HASH.match(fixture_hash):
            raise RowError(f"{fixture_hash!r} is not a hash of contracts.hashing (16 hex digits)")
        self._fixture_hash = fixture_hash

    def add_chain(
        self,
        step: int,
        state_hash: str,
        image_hashes: Mapping[Camera, str],
        executed_action: npt.NDArray[np.floating[Any]] | None,
    ) -> None:
        """One row of the chain: hashes of state `step` and the executed action that led to it.

        The first row of a job, the state after the reset (step 0) or the state
        restored from a snapshot (the snapshot's step), has no executed action:
        pass None, it is stored as NaN. Comparing a continuation's chain with the
        uninterrupted episode's therefore skips the action of that first row.
        """
        self._check_open()
        hashes = {Camera(k): v for k, v in image_hashes.items()}
        if set(hashes) != set(Camera):
            raise RowError(f"image hashes for {sorted(hashes)}; the chain needs every camera")
        for value in (state_hash, *hashes.values()):
            if not isinstance(value, str) or not _HASH.match(value):
                raise RowError(f"{value!r} is not a hash of contracts.hashing (16 hex digits)")
        expected = self._chain[-1]["step"] + 1 if self._chain else self.first_chain_step
        if step != expected:
            raise RowError(f"chain step {step} given where step {expected} is due")
        executed = action(executed_action)
        if not self._chain and executed is not None:
            raise RowError(
                f"chain row {step} is the job's first state (after the reset or the restore): "
                "it has no executed action"
            )
        row = {
            "step": int(step),
            "state_hash": state_hash,
            "scene_image_hash": hashes[Camera.scene],
            "wrist_image_hash": hashes[Camera.wrist],
            "executed_action": _nan_action() if executed is None else executed,
        }
        check_keys("chain", row)
        self._chain.append(row)

    def add_event(self, step: int, kind: EventKind | str, detail: str) -> None:
        """One row of `events` for this episode."""
        self._check_open()
        self._check_visible("events")
        row = {
            "episode_id": self.episode_id,
            "step": int(step),
            "kind": EventKind(kind).value,
            "detail": detail,
        }
        check_keys("events", row)
        self._events.append(row)

    def add_decision(self, record: DecisionRecord) -> None:
        """One row of `decisions`: the supervisor's trace of this episode."""
        self._check_open()
        self._check_run("a decision")
        row = {"episode_id": self.episode_id, **record.model_dump(mode="json")}
        check_keys("decisions", row)
        self._decisions.append(row)

    def set_effective_policy(self, spec: PolicySpec) -> None:
        """The policy spec after the policy input hook; without it, the identity's nominal spec."""
        self._check_open()
        self._check_run("the effective policy")
        self._policy = spec

    def set_verdict(
        self,
        true_success: bool,
        true_reason: str,
        libero_first_success_step: int | None,
        success_step: int | None = None,
        *,
        placed_step: int | None = None,
    ) -> None:
        """The fleet check's verdict before the data hook, LIBERO's first-fire step, the step
        the fleet's check fired, and (for `job_outcomes`) the first step it found the object
        placed."""
        self._check_open()
        self._check_run("the verdict")
        if true_reason not in VERDICT_REASONS:
            raise RowError(f"verdict reason {true_reason!r} is not one of {list(VERDICT_REASONS)}")
        derived = {
            "episode_id": self.episode_id,
            "true_success": bool(true_success),
            "true_reason": true_reason,
            "libero_first_success_step": libero_first_success_step,
            "success_step": success_step,
        }
        self._verdict = _select("verdicts", derived)
        check_keys("verdicts", self._verdict)
        self._placed_step = placed_step

    def add(self, table: str, row: Mapping[str, Any]) -> None:
        """A row of any table this job writes, by name, checked against the schema.

        `identities`, `faults_active`, `episodes`, `replays` and `job_outcomes`
        are derived from the job and `finish`; rows for them, or for a
        run-scoped table, are refused.
        """
        if table == "steps":
            self.add_step(row)
        elif table == "chain":
            self._add_chain_row(row)
        elif table == "events":
            check_keys("events", {"episode_id": self.episode_id, **row})
            self.add_event(row["step"], row["kind"], row["detail"])
        elif table == "decisions":
            data = {k: v for k, v in row.items() if k != "episode_id"}
            self._check_episode(row)
            self.add_decision(DecisionRecord.model_validate(data))
        elif table == "effective_policy":
            self._check_episode(row)
            data = {k: v for k, v in row.items() if k != "episode_id"}
            check_keys("effective_policy", {"episode_id": self.episode_id, **data})
            self.set_effective_policy(PolicySpec.model_validate(data))
        elif table == "verdicts":
            self._check_episode(row)
            check_keys("verdicts", {"episode_id": self.episode_id, **row})
            self.set_verdict(
                row["true_success"],
                row["true_reason"],
                row["libero_first_success_step"],
                row["success_step"],
            )
        elif table in SCHEMA:
            raise RowError(f"table {table} is not written row by row by an episode writer")
        else:
            raise RowError(f"no table {table!r} in contracts.log.SCHEMA")

    def _check_episode(self, row: Mapping[str, Any]) -> None:
        if row.get("episode_id", self.episode_id) != self.episode_id:
            raise RowError(f"a row of episode {row['episode_id']} given to {self.episode_id}")

    def _add_chain_row(self, row: Mapping[str, Any]) -> None:
        check_keys("chain", row)
        self.add_chain(
            row["step"],
            row["state_hash"],
            {Camera.scene: row["scene_image_hash"], Camera.wrist: row["wrist_image_hash"]},
            row["executed_action"],
        )

    def finish(
        self,
        *,
        outcome: Outcome | bool | str | None = None,
        success: bool | None = None,
        steps: int | None = None,
        placed_step: int | None = None,
        libero_step: int | None = None,
    ) -> EpisodeFiles:
        """Write the job's files; the chain last, which marks the job done.

        A run job needs `outcome` (the fleet check after the data hook, as the
        fleet saw it) and a verdict (`set_verdict`); its start time comes from
        its identity, its length from its last step. Any other job needs
        `success` (its true outcome) and may give `placed_step` and
        `libero_step`; `steps` defaults to the last step of its chain. Every
        job adds one row to `job_outcomes`.
        """
        self._check_open()
        if not self._chain:
            raise RowError(f"job {self.key} has no chain rows")
        root = self.layout.root
        paths: dict[str, Path] = {}
        sizes: dict[str, int] = {}
        if self.visible:
            given = {
                "success": success,
                "steps": steps,
                "placed_step": placed_step,
                "libero_step": libero_step,
            }
            named = sorted(k for k, v in given.items() if v is not None)
            if named:
                raise RowError(f"a run job takes {named} from its steps and verdict, not finish()")
            tables = self._finish_run(outcome, paths, sizes)
            assert self._verdict is not None
            outcome_row = self._outcome_row(
                self._verdict["true_success"],
                tables["episodes"][0]["steps"],
                self._placed_step,
                self._verdict["libero_first_success_step"],
            )
        else:
            if outcome is not None:
                raise RowError(f"a {self.job.mode.value} job has no visible outcome: use success=")
            if success is None:
                raise RowError(f"a {self.job.mode.value} job needs success=")
            last = self._chain[-1]["step"]
            if steps is not None and steps != last:
                raise RowError(f"steps={steps} but the chain ends at step {last}")
            tables = {}
            if self.job.mode is JobMode.replay and self.job.from_snapshot is None:
                row = {
                    "episode_id": self.episode_id,
                    "active_faults": fault_list_hash(self.job.effective_faults),
                    "success": bool(success),
                }
                tables["replays"] = [row]
            outcome_row = self._outcome_row(bool(success), last, placed_step, libero_step)
        if self._fixture_hash is None:
            raise RowError(f"job {self.key} has no fixture hash: call set_fixture_hash")
        tables["job_outcomes"] = [outcome_row]
        validated = {t: pylist(to_arrow(t, rows)) for t, rows in tables.items()}
        part: dict[str, Any] = {
            "key": self.key,
            "mode": self.job.mode.value,
            "episode_id": self.episode_id,
            "tables": validated,
            "canary": read_canary(root),
        }
        part_path = _part_path(root, self.key)
        sizes["part"] = write_text(encode_part(part), part_path, root)
        paths["part"] = part_path
        chain_path = self.layout.chain(self.job)
        sizes["chain"] = self._write_chain(chain_path)
        paths["chain"] = chain_path
        self._finished = True
        self._frames = {c: [] for c in Camera}
        return EpisodeFiles(self.key, paths, sizes)

    def _outcome_row(
        self, success: bool, steps: int, placed_step: int | None, libero_step: int | None
    ) -> dict[str, Any]:
        snapshot = self.job.from_snapshot
        derived = {
            "job_key": self.key,
            "episode_id": self.episode_id,
            "mode": self.job.mode.value,
            "removed": _json_value(list(self.job.remove)),
            "added": _json_value([f.model_dump(mode="json") for f in self.job.add]),
            "from_snapshot_step": None if snapshot is None else snapshot.step,
            "success": bool(success),
            "steps": int(steps),
            "placed_step": placed_step,
            "libero_step": libero_step,
        }
        return _select("job_outcomes", derived)

    def _finish_run(
        self,
        outcome: Outcome | bool | str | None,
        paths: dict[str, Path],
        sizes: dict[str, int],
    ) -> dict[str, list[dict[str, Any]]]:
        if outcome is None:
            raise RowError("a run job needs outcome=")
        if isinstance(outcome, bool):
            outcome = Outcome.success if outcome else Outcome.failure
        outcome = Outcome(outcome)
        if self._verdict is None:
            raise RowError(f"episode {self.episode_id} has no verdict: call set_verdict")
        if not self._steps:
            raise RowError(f"episode {self.episode_id} has no steps")
        step_numbers = [r["step"] for r in self._steps]
        chain_numbers = [r["step"] for r in self._chain]
        if step_numbers != chain_numbers:
            raise RowError(
                f"episode {self.episode_id}: {len(step_numbers)} steps but chain rows for "
                f"{len(chain_numbers)}; both hold one row per state"
            )
        last = self._steps[-1]
        for kind, rows in (("event", self._events), ("decision", self._decisions)):
            late = [r["step"] for r in rows if r["step"] > last["step"]]
            if late:
                raise RowError(f"{kind} at step {late[0]} after the episode's last step")
        has_video = bool(self._frames[Camera.scene])
        if has_video != self.job.record_video:
            raise RowError(
                f"episode {self.episode_id}: record_video is {self.job.record_video} but "
                f"{len(self._frames[Camera.scene])} frames were given"
            )
        root = self.layout.root
        video: dict[Camera, str | None] = {c: None for c in Camera}
        if has_video:
            for camera in Camera:
                path = self.layout.video(self.episode_id, camera)
                with staged(path, root) as tmp:
                    write_mp4(
                        tmp,
                        self._frames[camera],
                        self.video.fps,
                        crf=self.video.crf,
                        preset=self.video.preset,
                        repeatable=True,
                    )
                paths[f"video_{camera.value}"] = path
                sizes[f"video_{camera.value}"] = path.stat().st_size
                relative = VIDEO_PATH.format(episode_id=self.episode_id, camera=camera.value)
                video[camera] = relative.removeprefix("log/")
        steps_path = self.layout.table("steps", self.episode_id)
        sizes["steps"] = write_parquet(to_arrow("steps", self._steps), steps_path, root)
        paths["steps"] = steps_path
        episode = {
            "episode_id": self.episode_id,
            "robot_id": self.identity.robot_id,
            "site_id": self.identity.site_id,
            "day": self.identity.day,
            "task_text": self.task_text,
            "started_at": self.started_at,
            "steps": last["step"],
            "outcome": outcome.value,
            "duration_s": last["time_s"],
            "video_scene": video[Camera.scene],
            "video_wrist": video[Camera.wrist],
            "experiment_id": self.identity.experiment_id,
        }
        policy = self._policy if self._policy is not None else self.identity.policy
        return {
            "episodes": [_select("episodes", episode)],
            "events": self._events,
            "identities": [identity_row(self.identity, self.in_worker)],
            "faults_active": fault_rows(self.identity),
            "effective_policy": [policy_row(self.episode_id, policy)],
            "verdicts": [self._verdict],
            "decisions": self._decisions,
        }

    def _write_chain(self, path: Path) -> int:
        data = to_arrow("chain", self._chain)
        arrays: dict[str, Any] = {}
        for column in table_columns("chain"):
            values = data.column(column.name)
            if pa.types.is_fixed_size_list(column.type):
                width = column.type.list_size
                flat = values.combine_chunks().flatten().to_numpy(zero_copy_only=False)
                dtype = column.type.value_type.to_pandas_dtype()
                arrays[column.name] = flat.astype(dtype).reshape(-1, width)
            elif pa.types.is_string(column.type):
                arrays[column.name] = np.array(values.to_pylist(), dtype=np.str_)
            else:
                arrays[column.name] = values.to_numpy().astype(column.type.to_pandas_dtype())
        for column in file_columns("chain"):  # the fixture hash, a 0-d array
            arrays[column.name] = np.array(self._fixture_hash, dtype=np.str_)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        root = self.layout.root
        with staged(path, root) as tmp:
            tmp.write_bytes(buffer.getvalue())
        return path.stat().st_size


@dataclass
class _Lock:
    handle: IO[str] | None = field(default=None)


class RunWriter:
    """Opens (or resumes) a run directory and writes its run-level files.

    Hold it for the whole run, in the engine's main process, and open it
    before any worker starts: on a resume it deletes the files of unfinished
    jobs. Episode writers can be made here (`episode`) or in workers
    (`EpisodeWriter(run_dir, job, ...)`).
    """

    def __init__(
        self,
        run_dir: Path | str,
        record: RunRecord,
        *,
        episodes_per_robot_day: int,
        video: VideoSettings | None = None,
    ) -> None:
        """Open or resume `run_dir`.

        `episodes_per_robot_day` is the world's (`WorldSpec`); the episodes'
        start times derive from it (`contracts.identity.started_at_for`), so it
        is kept in the run record and a resume with another value is refused.
        """
        problems = schema_problems()
        if problems:
            raise SchemaError("contracts.log.SCHEMA has problems:\n" + "\n".join(problems))
        self.layout = RunLayout(Path(run_dir))
        root = self.layout.root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = _Lock()
        self._acquire()
        self.episodes_per_robot_day = episodes_per_robot_day
        try:
            self._open(record, video)
        except BaseException:
            self.close()
            raise

    def _open(self, record: RunRecord, video: VideoSettings | None) -> None:
        root = self.layout.root
        self.video = video or VideoSettings()
        self.cleared_files = clear_staging(root)
        for sub in ("log/steps", "log/video", "truth/chain", PARTS_DIR):
            (root / sub).mkdir(parents=True, exist_ok=True)
        n = self.episodes_per_robot_day
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"episodes_per_robot_day must be a positive integer, not {n!r}")
        given = record.extra.get(EPISODES_PER_ROBOT_DAY, n)
        if given != n:
            raise ValueError(f"the record says {given!r} episodes per robot and day, not {n}")
        record = RunRecord.model_validate(
            {**record.model_dump(mode="json"), "extra": {**record.extra, EPISODES_PER_ROBOT_DAY: n}}
        )
        self.resumed = self.layout.run_record.is_file()
        if self.resumed:
            # The fields that fix what the run's episodes are must agree; the rest of the
            # record (chain sample, chain_ok, extra) is kept from the file.
            existing = RunRecord.model_validate_json(self.layout.run_record.read_text())
            checks = {
                "software": (existing.software, record.software),
                "batch_size": (existing.batch_size, record.batch_size),
                "replay_rules": (existing.replay_rules, record.replay_rules),
                EPISODES_PER_ROBOT_DAY: (existing.extra.get(EPISODES_PER_ROBOT_DAY), n),
            }
            for name, (was, now) in checks.items():
                if was != now:
                    raise ResumeMismatch(f"{root} was run with {name} {was!r}, not {now!r}")
            self.record = existing
        else:
            self.record = record
            self._write_record()
        self.canary = plant_canary(root)
        self.discarded: tuple[str, ...] = self.discard_partial() if self.resumed else ()

    # Lifetime.

    def _acquire(self) -> None:
        handle = open(self.layout.root / LOCK_FILE, "a")  # noqa: SIM115 (held until close)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RunLocked(f"another writer holds {self.layout.root}") from None
        self._lock.handle = handle

    def close(self) -> None:
        """Release the run directory."""
        if self._lock.handle is not None:
            fcntl.flock(self._lock.handle, fcntl.LOCK_UN)
            self._lock.handle.close()
            self._lock.handle = None

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Jobs.

    def episode(self, job: Job, *, task_text: str, in_worker: bool = False) -> EpisodeWriter:
        """An episode writer for `job` in this run."""
        return EpisodeWriter(
            self.layout.root,
            job,
            task_text=task_text,
            episodes_per_robot_day=self.episodes_per_robot_day,
            in_worker=in_worker,
            video=self.video,
        )

    def is_done(self, job: Job) -> bool:
        """Done: the chain and the staged rows exist, and for a run job the steps file."""
        return self.layout.is_done(job) and _part_path(self.layout.root, job.key).is_file()

    def pending(self, jobs: Iterable[Job]) -> list[Job]:
        """The jobs still to run, in the given order."""
        return [job for job in jobs if not self.is_done(job)]

    def status(self, jobs: Iterable[Job] | None = None) -> RunStatus:
        return run_status(self.layout.root, jobs)

    def discard_partial(self) -> tuple[str, ...]:
        """Delete the files of jobs that have no chain; returns their keys.

        Called when a run is resumed, before any worker starts. Never call it
        while jobs are being written.
        """
        root = self.layout.root
        partial = run_status(root).partial
        for key in partial:
            _part_path(root, key).unlink(missing_ok=True)
            self.layout.table("chain", key).unlink(missing_ok=True)
            if _is_run_key(key):
                self.layout.table("steps", key).unlink(missing_ok=True)
                for camera in Camera:
                    self.layout.video(key, camera).unlink(missing_ok=True)
        return partial

    # Run-level files.

    def write_model(self, fleet: Fleet, task_classes: Sequence[str] = ()) -> None:
        """`log/description.json`, `change_log`, `maintenance` and `catalogue` from the model.

        The description is `contracts.model.describe_for_agents(fleet, True,
        task_classes)`, with the world spec's task classes; cutting the logs to
        the observed window is the caller's job.
        """
        root = self.layout.root
        description = describe_for_agents(fleet, True, task_classes)
        write_text(json.dumps(description, indent=1) + "\n", self.layout.description, root)
        self.write_table("change_log", change_log_rows(fleet))
        self.write_table("maintenance", maintenance_rows(fleet))
        self.write_table("catalogue", catalogue_rows(fleet, task_classes))

    def write_table(self, table: str, rows: Rows) -> None:
        """Write a run-scoped table whole (health, change_log, maintenance, oracle); replaces it."""
        if table not in RUN_TABLES:
            raise RowError(
                f"table {table} is written by the episode writers; "
                f"run-scoped tables are {list(RUN_TABLES)}"
            )
        root = self.layout.root
        data = to_arrow(table, rows)
        write_parquet(data, root / SCHEMA[table].path, root, _canary_metadata(root, table))

    def update_record(self, **fields: Any) -> RunRecord:
        """Change fields of the run record (chain_sample, chain_ok, extra) and rewrite run.json.

        `extra` is merged into the record's. What fixes the run's episodes
        (software, batch size, replay rules, episodes per robot and day) cannot change.
        """
        fixed = sorted({"software", "batch_size", "replay_rules"} & set(fields))
        extra = dict(fields.pop("extra", {}))
        if EPISODES_PER_ROBOT_DAY in extra:
            fixed.append(EPISODES_PER_ROBOT_DAY)
        if fixed:
            raise ValueError(f"{fixed} fix the run's episodes and cannot change")
        fields["extra"] = {**self.record.extra, **extra}
        self.record = RunRecord.model_validate({**self.record.model_dump(mode="json"), **fields})
        self._write_record()
        return self.record

    def _write_record(self) -> None:
        text = self.record.model_dump_json(indent=1) + "\n"
        write_text(text, self.layout.run_record, self.layout.root)

    def finalize(self) -> RunStatus:
        """Build the run-level tables from the staged rows of every finished job.

        Rows are sorted by episode (and step), so the tables do not depend on
        the order jobs finished in. Run-scoped tables that were never written
        are written empty, so that every table exists. Safe to call again.
        """
        root = self.layout.root
        status = self.status()
        rows: dict[str, list[dict[str, Any]]] = {t: [] for t in EPISODE_TABLES}
        for key in status.done:
            part = decode_part(_part_path(root, key).read_text())
            for table, table_rows in part["tables"].items():
                rows[table].extend(table_rows)
        for table in EPISODE_TABLES:
            _write_table(root, table, to_arrow(table, rows[table]))
        self.write_table("costs", costs_rows())  # the action costs, from the contracts
        for table in RUN_TABLES:
            if not (root / SCHEMA[table].path).is_file():
                self.write_table(table, [])
        write_text(json.dumps(list(status.done)), root / MERGED_FILE, root)
        return self.status()


def change_log_rows(fleet: Fleet) -> list[dict[str, Any]]:
    """The `change_log` rows of a fleet's change log, values as JSON text."""
    return [
        {
            "day": change.day,
            "scope": change.scope,
            "key": change.key.value,
            "old": json.dumps(change.old, separators=(",", ":")),
            "new": json.dumps(change.new, separators=(",", ":")),
            "author": change.author.value,
        }
        for change in fleet.change_log
    ]


def maintenance_rows(fleet: Fleet) -> list[dict[str, Any]]:
    """The `maintenance` rows of a fleet's maintenance log."""
    return [
        {
            "day": entry.day,
            "robot_id": entry.robot_id,
            "component": entry.component.value,
            "verb": entry.verb.value,
            "note": entry.note,
        }
        for entry in fleet.maintenance_log
    ]
