"""A job run's records as the five job tables (`contracts.log`, `JOB_TABLE_NAMES`).

A job run directory holds, beside `run.json` (hidden: the batch, the software, timings):

- `log/job_episodes.parquet` and `log/job_step_events.parquet`, one row per run of a
  program and per program step of it, rewritten after every episode so that a killed run
  leaves consistent tables;
- `log/object_steps/<episode_id>.parquet`, `log/goal_steps/...`, `log/robot_steps/...`,
  one file per episode, row k the state k control steps after the reset;
- `log/video/<episode_id>.<camera>.mp4` (`contracts.log.VIDEO_PATH`), with every
  `frame_every`-th state and the last, `robot_steps.frame_index` naming the frames;
- `episodes.jsonl` (one `JobEpisode` per line) and `batch_result.json`;
- `truth/jobs/<episode_id>.npz`: the executed actions and the hash of MuJoCo's state after
  the reset and after every step, which `fleetlab.jobs.replay` checks a re-execution
  against, and `truth/jobs/<episode_id>.json`: the episode's full identity.

The visible tree passes `contracts.log.log_tree_problems`, and `fleetlab.log.LogReader`
reads a run directory as it is. `write_job_tables` merges several runs into one tree, for
`fleetlab.log.export_runs(extra_tables=...)`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.contracts.env import Camera
from fleetlab.contracts.jobs import JobEpisode, JobProgram, JobStepKind, StartIdentity
from fleetlab.contracts.log import JOB_TABLE_NAMES, SCHEMA, VIDEO_PATH
from fleetlab.log._files import staged, write_parquet, write_text
from fleetlab.log._rows import to_arrow

__all__ = [
    "BATCH_RESULT",
    "EPISODES_FILE",
    "RUN_RECORD",
    "TRUTH_DIR",
    "EpisodeRecorder",
    "episode_problems",
    "episode_row",
    "read_episodes",
    "step_event_rows",
    "table_path",
    "truth_paths",
    "video_relative",
    "write_episode_tables",
    "write_job_tables",
    "write_run_tables",
]

EPISODES_FILE = "episodes.jsonl"
BATCH_RESULT = "batch_result.json"
RUN_RECORD = "run.json"
TRUTH_DIR = "truth/jobs"
PER_EPISODE = ("object_steps", "goal_steps", "robot_steps")
RUN_LEVEL = ("job_episodes", "job_step_events")


def table_path(root: Path, table: str, episode_id: str | None = None) -> Path:
    """Where a job table's file lies under a run directory or an export."""
    template = SCHEMA[table].path
    if "{episode_id}" in template:
        if episode_id is None:
            raise ValueError(f"{table} is stored per episode")
        return root / template.format(episode_id=episode_id)
    return root / template


def video_relative(episode_id: str, camera: Camera) -> str:
    """A video's path relative to `log/`, as `job_episodes.video_<camera>` holds it."""
    return VIDEO_PATH.format(episode_id=episode_id, camera=camera.value).removeprefix("log/")


def truth_paths(root: Path, episode_id: str) -> tuple[Path, Path]:
    """The episode's replay record (`.npz`) and its identity (`.json`)."""
    return root / TRUTH_DIR / f"{episode_id}.npz", root / TRUTH_DIR / f"{episode_id}.json"


def _vec(value: npt.ArrayLike) -> list[float]:
    return [float(x) for x in np.asarray(value, np.float64).reshape(-1)]


class EpisodeRecorder:
    """The rows of one episode's per-step tables, built while it runs.

    `observe` adds row k, the state after k commands; `command` fills in the command
    issued at the latest row; `finish` marks the last row, which has no command.
    """

    def __init__(self, episode_id: str, objects: Sequence[str], n_goals: int) -> None:
        self.episode_id = episode_id
        self.objects = tuple(objects)
        self.n_goals = n_goals
        self.robot: list[dict[str, Any]] = []
        self.object: list[dict[str, Any]] = []
        self.goal: list[dict[str, Any]] = []

    def observe(
        self,
        step: int,
        *,
        eef_pos: npt.ArrayLike,
        eef_quat: npt.ArrayLike,
        gripper_opening: float,
        poses: Mapping[str, npt.ArrayLike],
        velocities: Mapping[str, npt.ArrayLike],
        contacts: Sequence[tuple[str, str]],
        goals_hold: Sequence[bool],
        frame_index: int | None,
    ) -> None:
        from fleetlab.jobs.check import robot_touching, speeds

        if self.robot and self.robot[-1]["phase"] is None:
            raise ValueError(f"step {step}: the previous row has no command")
        if len(goals_hold) != self.n_goals:
            raise ValueError(f"{len(goals_hold)} goal values for {self.n_goals} goals")
        self.robot.append(
            {
                "episode_id": self.episode_id,
                "step": step,
                "eef_pos": _vec(eef_pos),
                "eef_quat": _vec(eef_quat),
                "gripper_opening": float(gripper_opening),
                "commanded_action": None,
                "gripper_command": None,
                "instruction": None,
                "program_step": None,
                "phase": None,
                "frame_index": frame_index,
            }
        )
        for name in self.objects:
            pose = np.asarray(poses[name], np.float64)
            linear, angular = speeds(velocities[name])
            touching = sorted(
                {b for a, b in contacts if a == name} | {a for a, b in contacts if b == name}
            )
            self.object.append(
                {
                    "episode_id": self.episode_id,
                    "step": step,
                    "object": name,
                    "position": _vec(pose[:3]),
                    "quaternion": _vec(pose[3:7]),
                    "linear_speed": linear,
                    "angular_speed": angular,
                    "in_contact_with_robot": robot_touching(contacts, name),
                    "in_contact_with": touching,
                }
            )
        for i, holds in enumerate(goals_hold):
            self.goal.append(
                {"episode_id": self.episode_id, "step": step, "goal": i, "holds": bool(holds)}
            )

    def command(
        self,
        action: npt.ArrayLike,
        *,
        instruction: str | None,
        program_step: int | None,
        phase: str,
    ) -> None:
        """The command issued at the latest row: the action, and what issued it."""
        row = self.robot[-1]
        if row["phase"] is not None:
            raise ValueError(f"step {row['step']} has its command already")
        values = [float(x) for x in np.asarray(action, np.float32).reshape(-1)]
        if len(values) != 7:
            raise ValueError(f"an action has 7 values, not {len(values)}")
        row.update(
            commanded_action=values,
            gripper_command=values[6],
            instruction=instruction,
            program_step=program_step,
            phase=phase,
        )

    def finish(self) -> None:
        self.robot[-1]["phase"] = "end"

    def tables(self) -> dict[str, pa.Table]:
        if not self.robot or self.robot[-1]["phase"] != "end":
            raise ValueError("finish the episode first")
        return {
            "robot_steps": to_arrow("robot_steps", self.robot),
            "object_steps": to_arrow("object_steps", self.object),
            "goal_steps": to_arrow("goal_steps", self.goal),
        }


def episode_row(
    episode: JobEpisode, videos: Mapping[Camera, str | None], order_detail: str
) -> dict[str, Any]:
    """The `job_episodes` row of one episode."""
    verdict = episode.delivery
    return {
        "job_id": episode.job_id,
        "episode_id": episode.episode_id,
        "program_digest": episode.program_digest,
        "base_task": episode.start.base_task,
        "init_state": episode.start.init_state,
        "control_steps": episode.control_steps,
        "tail_steps": episode.tail_steps,
        "delivered": episode.delivered,
        "delivery_detail": verdict.reason,
        "goals_held_at_end": list(verdict.goals_hold),
        "robot_touching_at_end": list(verdict.robot_touches),
        "moving_at_end": list(verdict.not_at_rest),
        "gripper_opening_at_end": verdict.gripper_opening_m,
        "order_kept": episode.order_kept,
        "order_detail": order_detail,
        "start_eef_pos": episode.start_eef_pos,
        "start_eef_quat": episode.start_eef_quat,
        "video_scene": videos.get(Camera.scene),
        "video_wrist": videos.get(Camera.wrist),
    }


def step_event_rows(
    episode: JobEpisode, program: JobProgram, tail_instruction: str | None
) -> list[dict[str, Any]]:
    """The `job_step_events` rows of one episode: its program steps, then the tail if it ran."""
    rows = []
    for result in episode.program_steps:
        step = program.steps[result.step]
        rows.append(
            {
                "episode_id": episode.episode_id,
                "program_step": result.step,
                "kind": result.kind.value,
                "start_step": result.start_control_step,
                "end_step": result.end_control_step,
                "steps_used": result.steps_used,
                "outcome": result.outcome.value,
                "instruction": step.instruction if step.kind is JobStepKind.policy else None,
            }
        )
    if episode.tail_steps:
        end = episode.control_steps
        rows.append(
            {
                "episode_id": episode.episode_id,
                "program_step": None,
                "kind": "tail",
                "start_step": end - episode.tail_steps,
                "end_step": end,
                "steps_used": episode.tail_steps,
                "outcome": "reached" if episode.delivered else "timeout",
                "instruction": tail_instruction,
            }
        )
    return rows


def write_episode_tables(root: Path, episode_id: str, tables: Mapping[str, pa.Table]) -> None:
    for name in PER_EPISODE:
        write_parquet(tables[name], table_path(root, name, episode_id), root)


def write_run_tables(
    root: Path, episodes: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> None:
    """Rewrite the two run-level tables with every row so far, in episode order."""
    ordered = sorted(episodes, key=lambda r: (r["job_id"], r["episode_id"]))
    by_episode = sorted(events, key=lambda r: r["episode_id"])  # stable: program order kept
    write_parquet(to_arrow("job_episodes", ordered), table_path(root, "job_episodes"), root)
    write_parquet(
        to_arrow("job_step_events", by_episode), table_path(root, "job_step_events"), root
    )


def read_episodes(root: Path) -> list[JobEpisode]:
    """The `JobEpisode` records of a run directory, in the order they ran."""
    path = root / EPISODES_FILE
    if not path.is_file():
        return []
    return [JobEpisode.model_validate_json(line) for line in path.read_text().splitlines() if line]


def append_episode(root: Path, episode: JobEpisode) -> None:
    with (root / EPISODES_FILE).open("a") as f:
        f.write(episode.model_dump_json() + "\n")


def write_json(root: Path, relative: str, value: Any) -> None:
    write_text(
        json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", root / relative, root
    )


def _read(path: Path) -> pa.Table:
    return pq.read_table(path)


def write_job_tables(stage: Path, run_dirs: Sequence[Path]) -> list[str]:
    """Merge job runs into one tree under `stage` (a staging export or a fresh directory).

    The run-level tables are concatenated in episode order; the per-episode files and the
    videos are copied. Two runs with the same episode id are refused. Returns the episode
    ids, in order.
    """
    stage = Path(stage)
    episodes: list[pa.Table] = []
    events: list[pa.Table] = []
    seen: dict[str, Path] = {}
    for run in map(Path, run_dirs):
        path = table_path(run, "job_episodes")
        if not path.is_file():
            continue
        table = to_arrow("job_episodes", _read(path))
        for episode_id in table["episode_id"].to_pylist():
            if episode_id in seen:
                raise ValueError(f"episode {episode_id} is in {seen[episode_id]} and in {run}")
            seen[episode_id] = run
            for name in PER_EPISODE:
                source = table_path(run, name, episode_id)
                if source.is_file():
                    target = table_path(stage, name, episode_id)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            for camera in Camera:
                source = run / "log" / video_relative(episode_id, camera)
                if source.is_file():
                    target = stage / "log" / video_relative(episode_id, camera)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        episodes.append(table)
        events_path = table_path(run, "job_step_events")
        if events_path.is_file():
            events.append(to_arrow("job_step_events", _read(events_path)))
    if episodes:
        merged = pa.concat_tables(episodes).sort_by(
            [("job_id", "ascending"), ("episode_id", "ascending")]
        )
        _write_plain(merged, table_path(stage, "job_episodes"))
    if events:
        merged_events = pa.concat_tables(events)
        order = np.argsort(np.asarray(merged_events["episode_id"].to_pylist()), kind="stable")
        _write_plain(merged_events.take(pa.array(order)), table_path(stage, "job_step_events"))
    return sorted(seen)


def _write_plain(data: pa.Table, path: Path) -> None:
    """A visible table as `fleetlab.log` writes it (zstd, no stored schema), without staging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, path, compression="zstd", store_schema=False)


def job_table_names() -> tuple[str, ...]:
    return JOB_TABLE_NAMES


def save_replay_record(
    root: Path,
    episode_id: str,
    executed: npt.NDArray[np.float32],
    state_hashes: Sequence[str],
    identity_json: str,
) -> None:
    """The truth a replay needs: actions as they reached the simulator, state hashes."""
    npz, ident = truth_paths(root, episode_id)
    if len(state_hashes) != len(executed) + 1:
        raise ValueError("one state hash after the reset and one after every action")
    with staged(npz, root) as tmp, tmp.open("wb") as f:
        np.savez_compressed(
            f,
            executed=np.asarray(executed, np.float32).reshape(-1, 7),
            state_hash=np.asarray(list(state_hashes)),
        )
    write_text(identity_json + "\n", ident, root)


def load_replay_record(
    root: Path, episode_id: str
) -> tuple[npt.NDArray[np.float32], list[str], str]:
    npz, ident = truth_paths(root, episode_id)
    with np.load(npz) as data:
        executed = np.asarray(data["executed"], np.float32)
        hashes = [str(h) for h in data["state_hash"]]
    return executed, hashes, ident.read_text().strip()


def recorded_episode_ids(root: Path) -> set[str]:
    """The episodes the run's `job_episodes` table holds."""
    path = table_path(root, "job_episodes")
    if not path.is_file():
        return set()
    return set(pq.read_table(path, columns=["episode_id"])["episode_id"].to_pylist())


def episode_problems(
    root: Path,
    episode: JobEpisode,
    *,
    job_id: str,
    program_digest: str,
    start: StartIdentity,
    record_video: bool,
    in_table: bool,
) -> list[str]:
    """Why a recorded episode cannot stand for this job's run from this start (a resume)."""
    eid = episode.episode_id
    problems = []
    if episode.job_id != job_id:
        problems.append(f"{eid} belongs to {episode.job_id}, not {job_id}")
    if episode.program_digest != program_digest:
        problems.append(f"{eid} ran another program ({episode.program_digest[:12]})")
    if episode.start != start:
        problems.append(f"{eid} ran from another start (initial state {episode.start.init_state})")
    for name in PER_EPISODE:
        if not table_path(root, name, eid).is_file():
            problems.append(f"{eid} has no {name} file")
    npz, ident = truth_paths(root, eid)
    if not (npz.is_file() and ident.is_file()):
        problems.append(f"{eid} has no replay record")
    else:
        executed, hashes, _ = load_replay_record(root, eid)
        if (
            len(executed) != episode.control_steps
            or len(hashes) != len(executed) + 1
            or hashes[-1] != episode.end_state_hash
        ):
            problems.append(f"{eid}'s replay record does not match the episode")
    if record_video:
        for camera in Camera:
            if not (root / "log" / video_relative(eid, camera)).is_file():
                problems.append(f"{eid} has no {camera.value} video")
    if not in_table:
        problems.append(f"{eid} is not in job_episodes")
    return problems
