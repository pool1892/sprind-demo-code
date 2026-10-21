"""The export: the agent-visible tree the tool server mounts (docs/design/contracts.md, 11 and 12).

    report = export_runs([run_a, run_b], out_dir, fleet=fleet)

builds

    out_dir/log/episodes.parquet, steps/<episode_id>.parquet, events.parquet, health.parquet,
                change_log.parquet, maintenance.parquet, description.json,
                video/<episode_id>.scene.mp4, video/<episode_id>.wrist.mp4
    out_dir/data_dictionary.md

from the runs' `log/` trees and nothing else of them. Every table is rebuilt
from the visible schema's allow-list (`contracts.log.allowed_columns`), cast to
its declared types and written anew, so no column, metadata or row of another
episode can ride along; the description comes from
`contracts.model.describe_for_agents` when the fleet is given. The export
refuses (`ExportRefused`), and writes nothing, when:

- the schema has problems, among them a measured time in a visible table;
- a run is not finalized (unfinished jobs, tables not built from every job);
- a run's `log/` tree fails `contracts.log.log_tree_problems`: a file that is
  not part of the visible log, a column off the allow-list or named like a
  measured time, an identifier that is not neutral, a hidden field in the
  description;
- an episode id appears in two runs, a steps or video file is missing, or
  frames are logged without a video or a video without frames;
- a video carries a creation or modification time;
- the runs disagree on a fleet-level table (health, change and maintenance
  logs, description) and no fleet is given;
- the leak audit (`leaks.audit_export`, on by default) finds a truth term or
  a canary.

The export is built in a temporary directory next to `out_dir` and renamed
into place when every check has passed.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.contracts.actions import catalogue_rows, costs_rows
from fleetlab.contracts.env import Camera
from fleetlab.contracts.ids import IdKind, is_neutral
from fleetlab.contracts.log import (
    DESCRIPTION_FILE,
    SCHEMA,
    VIDEO_PATH,
    allowed_columns,
    log_tree_problems,
    schema_problems,
    visible_schema,
)
from fleetlab.contracts.model import Fleet, describe_for_agents
from fleetlab.log._rows import RowError, sort_table, to_arrow
from fleetlab.log.dictionary import DICTIONARY_FILE, data_dictionary
from fleetlab.log.leaks import LeakReport, audit_export
from fleetlab.log.writer import change_log_rows, maintenance_rows, run_status

__all__ = ["ExportRefused", "ExportReport", "export_problems", "export_runs"]

# Tables that belong to the fleet, not to episodes: one source for the whole export.
FLEET_TABLES = ("health", "change_log", "maintenance", "catalogue")
# The fleet-level tables a given fleet supplies (`export_runs(fleet=...)`).
_FROM_FLEET = ("change_log", "maintenance", "catalogue")


class ExportRefused(ValueError):
    """The runs cannot be exported; the message lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"export refused, {len(problems)} problems:\n" + "\n".join(problems))


@dataclass(frozen=True)
class ExportReport:
    out_dir: Path
    episodes: int
    step_rows: int
    sizes: dict[str, int]
    leaks: LeakReport | None = field(default=None)


def _visible(path: Path, table: str) -> pa.Table:
    """A visible table read through its allow-list and cast to its declared types."""
    try:
        data = pq.read_table(path, columns=list(allowed_columns(table)))
        return to_arrow(table, data.replace_schema_metadata(None))
    except (OSError, pa.ArrowException, RowError) as error:
        raise ExportRefused([f"{path}: {error}"]) from error


def _write(data: pa.Table, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, path, compression="zstd", store_schema=False)
    return path.stat().st_size


def _symlinks(root: Path) -> list[str]:
    return [f"{p}: symbolic link" for p in root.rglob("*") if p.is_symlink()]


def _one_source(tables: Sequence[tuple[Path, pa.Table]], what: str) -> pa.Table | None:
    """The one non-empty version of a fleet-level table the runs agree on, or None."""
    filled = [(run, t) for run, t in tables if t.num_rows]
    if not filled:
        return None
    first_run, first = filled[0]
    for run, other in filled[1:]:
        if not other.equals(first):
            raise ExportRefused([f"runs {first_run} and {run} disagree on {what}; pass the fleet"])
    return first


def mp4_times(path: Path) -> list[int]:
    """The creation and modification times in an MP4's movie, track and media headers."""
    times: list[int] = []
    containers = {b"moov", b"trak", b"mdia"}
    headers = {b"mvhd", b"tkhd", b"mdhd"}

    def walk(data: bytes) -> None:
        offset = 0
        while offset + 8 <= len(data):
            size, kind = struct.unpack(">I4s", data[offset : offset + 8])
            header = 8
            if size == 1:
                size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
                header = 16
            elif size == 0:
                size = len(data) - offset
            if size < header:
                return
            body = data[offset + header : offset + size]
            if kind in containers:
                walk(body)
            elif kind in headers and body:
                if body[0] == 1:
                    times.extend(struct.unpack(">QQ", body[4:20]))
                else:
                    times.extend(struct.unpack(">II", body[4:12]))
            offset += size

    walk(path.read_bytes())
    return times


def export_problems(out_dir: Path | str) -> list[str]:
    """What is wrong with a built export: the log tree's problems and the export's own."""
    root = Path(out_dir)
    problems = log_tree_problems(root)
    problems += _symlinks(root)
    for child in root.iterdir():
        if child.name not in {"log", DICTIONARY_FILE}:
            problems.append(f"{child.name}: not part of an export")
    for spec in visible_schema().values():
        if spec.file_key is None and spec.source == "run" and not (root / spec.path).is_file():
            problems.append(f"{spec.path}: missing")
    for path in (root / "log").rglob("*.parquet"):
        extra = set(pq.read_schema(path).metadata or {})
        if extra:
            problems.append(f"{path.relative_to(root)}: metadata {sorted(extra)}")
    for path in (root / "log" / "video").glob("*.mp4"):
        if any(mp4_times(path)):
            problems.append(f"{path.relative_to(root)}: carries a creation or modification time")
    return problems


def export_runs(
    run_dirs: Sequence[Path | str],
    out_dir: Path | str,
    *,
    fleet: Fleet | None = None,
    audit_fleet: Fleet | None = None,
    task_classes: Sequence[str] = (),
    audit: bool = True,
    texts: Mapping[str, str] | None = None,
    overwrite: bool = False,
    before_publish: Callable[[Path], None] | None = None,
    health: Sequence[Mapping[str, Any]] | None = None,
    extra_tables: Callable[[Path], None] | None = None,
    audit_runs: Sequence[Path | str] = (),
) -> ExportReport:
    """Build the agent-visible tree of one or more finalized runs in `out_dir`.

    `fleet` gives the description (`describe_for_agents(fleet, True,
    task_classes)`, with the world spec's task classes) and the change and
    maintenance logs; without it they are taken from the runs, which must
    agree. `audit_fleet`, when given, supplies hidden property names to the check
    without contributing records to the export. `audit` runs the leak audit on the
    result, with `texts` searched as well, and refuses on any hit. `overwrite` replaces
    an earlier export in `out_dir`.
    `before_publish` checks the staging tree before it can stand at `out_dir`;
    a caller supplying its own audit sets `audit=False` to avoid a second audit.
    `health`, when given, is the health table's rows (the loop's history over several
    runs); without it the table comes from the runs, which must agree.
    `extra_tables(stage)` writes the visible tables only an export holds
    (`TableSpec.source`, the probes of the loop) into the staging tree before any check;
    `audit_runs` are the finalized runs they were built from, whose truth the leak audit
    searches for as well.
    """
    runs = [Path(r).resolve() for r in run_dirs]
    out = Path(out_dir).resolve()
    if not runs:
        raise ExportRefused(["no runs given"])
    problems = [f"schema: {p}" for p in schema_problems()]
    for run in runs:
        status = run_status(run)
        if not status.finalized:
            problems.append(
                f"{run}: not finalized ({len(status.partial)} unfinished jobs, "
                f"{len(set(status.done) ^ set(status.merged))} jobs not in the tables, "
                f"{len(status.missing)} tables missing, {status.staging_files} staged files)"
            )
        problems += [f"{run}: {p}" for p in log_tree_problems(run)]
        problems += _symlinks(run / "log")
    if out.exists() and any(out.iterdir()):
        is_export = (out / DICTIONARY_FILE).is_file() and (out / "log").is_dir()
        if not (overwrite and is_export):
            problems.append(f"{out} exists and is not empty (overwrite only replaces an export)")
    if problems:
        raise ExportRefused(problems)

    stage = out.parent / f".{out.name}.export-{uuid.uuid4().hex[:8]}"
    trash = out.parent / f".{out.name}.old-{uuid.uuid4().hex[:8]}"
    stage.mkdir(parents=True)
    try:
        report = _build(runs, stage, fleet, task_classes, health)
        if extra_tables is not None:
            extra_tables(stage)
        problems = export_problems(stage)
        if problems:
            raise ExportRefused(problems)
        leaks = None
        if audit:
            leaks = audit_export(
                stage,
                [*runs, *(Path(r).resolve() for r in audit_runs)],
                fleet=audit_fleet if audit_fleet is not None else fleet,
                texts=texts,
            )
            if not leaks.clean:
                raise ExportRefused([f"leak audit: {h.where}: {h.term!r}" for h in leaks.hits])
        if before_publish is not None:
            before_publish(stage)
        # The earlier export is moved aside, the new one put in its place, and only then is
        # the earlier one deleted; if anything fails in between, it is put back.
        if out.exists():
            os.replace(out, trash)
        os.replace(stage, out)
        shutil.rmtree(trash, ignore_errors=True)
    finally:
        if not out.exists() and trash.exists():
            os.replace(trash, out)
        if stage.exists():
            shutil.rmtree(stage)
    return ExportReport(out, report["episodes"], report["step_rows"], report["sizes"], leaks)


def _build(
    runs: list[Path],
    stage: Path,
    fleet: Fleet | None,
    task_classes: Sequence[str],
    health: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    sizes: dict[str, int] = {}

    def add(kind: str, n: int) -> None:
        sizes[kind] = sizes.get(kind, 0) + n

    # Episodes, and which run each comes from.
    owner: dict[str, Path] = {}
    episode_tables = []
    problems = []
    for run in runs:
        data = _visible(run / SCHEMA["episodes"].path, "episodes")
        for episode_id in data.column("episode_id").to_pylist():
            if not is_neutral(IdKind.episode, episode_id):
                problems.append(f"{run}: episode id {episode_id!r} is not neutral")
            elif episode_id in owner:
                problems.append(f"episode {episode_id} is in {owner[episode_id]} and {run}")
            else:
                owner[episode_id] = run
        episode_tables.append(data)
    if problems:
        raise ExportRefused(problems)
    episodes = sort_table("episodes", pa.concat_tables(episode_tables))
    add("episodes", _write(episodes, stage / SCHEMA["episodes"].path))

    # Steps and videos, episode by episode, from the run that holds it.
    step_rows = 0
    for row in episodes.to_pylist():
        episode_id, run = row["episode_id"], owner[row["episode_id"]]
        source = run / SCHEMA["steps"].path.format(episode_id=episode_id)
        if not source.is_file():
            raise ExportRefused([f"{run}: episode {episode_id} has no steps file"])
        steps = _visible(source, "steps")
        if set(steps.column("episode_id").to_pylist()) != {episode_id}:
            raise ExportRefused([f"{source}: rows of another episode"])
        has_frames = steps.column("frame_index").null_count < steps.num_rows
        step_rows += steps.num_rows
        add("steps", _write(steps, stage / SCHEMA["steps"].path.format(episode_id=episode_id)))
        for camera in Camera:
            relative = row[f"video_{camera.value}"]
            if relative is None:
                if has_frames:
                    raise ExportRefused([f"episode {episode_id}: frames logged, no video"])
                continue
            if not has_frames:
                raise ExportRefused([f"episode {episode_id}: a video, no frame_index logged"])
            expected = VIDEO_PATH.format(episode_id=episode_id, camera=camera.value)
            if relative != expected.removeprefix("log/"):
                raise ExportRefused([f"episode {episode_id}: video path {relative!r}"])
            video = run / expected
            if not video.is_file() or video.is_symlink():
                raise ExportRefused([f"{run}: video {expected} missing"])
            target = stage / expected
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video, target)
            add(f"video_{camera.value}", target.stat().st_size)

    # Events, sorted, of exported episodes only.
    events = pa.concat_tables([_visible(run / SCHEMA["events"].path, "events") for run in runs])
    stray = set(events.column("episode_id").to_pylist()) - set(owner)
    if stray:
        raise ExportRefused([f"events of episodes not in the export: {sorted(stray)[:5]}"])
    add("events", _write(sort_table("events", events), stage / SCHEMA["events"].path))

    # Fleet-level tables and the description: from the fleet, or from runs that agree.
    fleet_tables: dict[str, pa.Table | None] = {}
    for table in FLEET_TABLES:
        if table == "health" and health is not None:
            fleet_tables[table] = sort_table("health", to_arrow("health", list(health)))
            continue
        if fleet is not None and table in _FROM_FLEET:
            continue  # the given fleet supplies it below; the runs' copies need not agree
        found = [(run, _visible(run / SCHEMA[table].path, table)) for run in runs]
        fleet_tables[table] = _one_source(found, table)
    if fleet is not None:
        fleet_tables["change_log"] = to_arrow("change_log", change_log_rows(fleet))
        fleet_tables["maintenance"] = to_arrow("maintenance", maintenance_rows(fleet))
        fleet_tables["catalogue"] = to_arrow("catalogue", catalogue_rows(fleet, task_classes))
        description: Any = describe_for_agents(fleet, True, task_classes)
    else:
        found_descriptions = []
        for run in runs:
            path = run / DESCRIPTION_FILE
            if path.is_file():
                found_descriptions.append((run, json.loads(path.read_text())))
        if not found_descriptions:
            raise ExportRefused(["no description: pass the fleet or write it with the run"])
        description = found_descriptions[0][1]
        for run, other in found_descriptions[1:]:
            if other != description:
                raise ExportRefused([f"runs disagree on the description ({run}); pass the fleet"])
    for table in FLEET_TABLES:
        data = fleet_tables[table]
        if data is None:
            data = to_arrow(table, [])
        add(table, _write(data, stage / SCHEMA[table].path))
    # The action costs are the contracts' own (`contracts.actions.COSTS`), the same everywhere.
    add("costs", _write(to_arrow("costs", costs_rows()), stage / SCHEMA["costs"].path))
    description_path = stage / DESCRIPTION_FILE
    description_path.write_text(json.dumps(description, indent=1) + "\n")
    add("description", description_path.stat().st_size)
    dictionary = stage / DICTIONARY_FILE
    dictionary.write_text(data_dictionary())
    add("data_dictionary", dictionary.stat().st_size)
    return {"episodes": episodes.num_rows, "step_rows": step_rows, "sizes": sizes}
