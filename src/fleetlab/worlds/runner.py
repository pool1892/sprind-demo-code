"""The fleet runner at scale: a whole world as one resumable heavy job.

    tools/heavy-run uv run --locked python -m fleetlab.worlds run D1     # make world WORLD=D1

Steps, each resumable and skipped when its result is already there (`STEPS`):

1. `files`: the world files under `files/` (`fleetlab.worlds.world.write_world`); on a
   later call the world is generated again and must give the stored hash, or the run
   stops (`WorldChanged`), unless no episode has run yet, in which case the files are
   rewritten (a world composed again after new curves, before its runs).
2. `observed`, `later`: the engine runs of the observed window and of the experiment and
   scoring windows (`fleetlab.engine.fleet.run`, which resumes a killed run by itself:
   the log's writer drops unfinished jobs, the engine runs only jobs without a chain).
3. `observed-records`, `later-records`: the fleet's records into each finished run, which
   the engine does not write: the description and the change and maintenance logs
   (the observed run gets them cut to the observed window), and the daily health checks
   of the run's days from the generator's model.
4. `observed-events`, `later-events`: the events beyond the engine's grasp and release
   (`fleetlab.worlds.events`: drop, stall, near miss, collision), written into each run.
5. `export`: the agent-visible export of the observed run, with the leak audit
   (`fleetlab.log.export_runs`); refused on any hit.
6. `summary`: what the world measured against what the generator expected, and the
   gate-3 marks (`summary`), into `runner.json`, which then says `"ready": true`.

`runner.json` holds every step's start, end and seconds, so a killed and resumed world
shows both attempts.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.identity import SoftwareVersion
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.rollout import Job, RunLayout
from fleetlab.contracts.world import verify_world
from fleetlab.worlds.generate import World
from fleetlab.worlds.layout import RUN_WINDOWS, WorldPaths

STEPS = (
    "files",
    "observed",
    "observed-records",
    "observed-events",
    "later",
    "later-records",
    "later-events",
    "export",
    "summary",
)
MIOPEN_DIR = Path.home() / ".cache" / "robots-probes" / "engine" / "miopen"
PRECISION = "fp16_res32"
BATCH_SIZE = 4


class WorldChanged(RuntimeError):
    """The world regenerates to another hash than the one its runs were made from."""


@dataclass
class EngineSettings:
    """How the engine runs a world: workers, and RunConfig fields to override (tests run
    the toy cell on the CPU this way)."""

    workers: int = 10
    overrides: dict[str, Any] = field(default_factory=dict)
    contacts: bool = True  # the contact replay for collision events (LIBERO runs only)

    def config(self, out: Path, per_day: int) -> Any:
        from fleetlab.engine.fleet import RunConfig

        fields: dict[str, Any] = {
            "out": str(out),
            "batch_size": BATCH_SIZE,
            "workers": self.workers,
            "precision": PRECISION,
            "episodes_per_robot_day": per_day,
        }
        fields.update(self.overrides)
        return RunConfig(**fields)


def engine_software() -> SoftwareVersion:
    """The software the world's jobs are made for: the pinned database, batch 4, fp16_res32."""
    os.environ.setdefault("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR))
    from fleetlab.engine import miopen
    from fleetlab.engine import software as soft

    miopen.activate()
    return soft.current(miopen.require(BATCH_SIZE, PRECISION))


# ---- bookkeeping -----------------------------------------------------------------------------


def note(paths: WorldPaths, step: str, **fields: Any) -> None:
    """Merge fields into the step's record in runner.json (attempts kept as a list)."""
    paths.root.mkdir(parents=True, exist_ok=True)
    record = paths.read_record()
    steps = record.setdefault("steps", {})
    steps.setdefault(step, {}).update(fields)
    tmp = paths.record.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str) + "\n")
    os.replace(tmp, paths.record)


def set_fields(paths: WorldPaths, **fields: Any) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    record = paths.read_record()
    record.update(fields)
    tmp = paths.record.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str) + "\n")
    os.replace(tmp, paths.record)


class Timed:
    """Records a step's start, end and seconds in runner.json; a new attempt is appended."""

    def __init__(self, paths: WorldPaths, step: str) -> None:
        self.paths, self.step = paths, step

    def __enter__(self) -> Timed:
        self.start = time.time()
        record = self.paths.read_record().get("steps", {}).get(self.step, {})
        attempts = list(record.get("attempts", []))
        attempts.append({"started": round(self.start, 1)})
        note(self.paths, self.step, attempts=attempts, done=False)
        if self.step == "export":
            note(self.paths, self.step, leaks_clean=False)
        return self

    def __exit__(self, kind: object, *rest: object) -> None:
        end = time.time()
        record = self.paths.read_record()["steps"][self.step]
        attempts = list(record["attempts"])
        attempts[-1].update(
            {"ended": round(end, 1), "seconds": round(end - self.start, 1), "ok": kind is None}
        )
        note(self.paths, self.step, attempts=attempts, done=kind is None)


# ---- the steps -------------------------------------------------------------------------------


def run_jobs(world: World, name: str) -> list[Job]:
    return [j for w in RUN_WINDOWS[name] for j in world.jobs_in(w)]  # type: ignore[arg-type]


def run_complete(run_dir: Path, jobs: Sequence[Job]) -> bool:
    """Every job done, the run finalized and its chain sample held."""
    from fleetlab.integration.oracle import run_record
    from fleetlab.log import run_status

    if not (run_dir / "run.json").is_file():
        return False
    layout = RunLayout(run_dir)
    done = all(layout.is_done(j) for j in jobs)
    return done and run_status(run_dir).finalized and run_record(run_dir).chain_ok is True


def any_episode_ran(paths: WorldPaths) -> bool:
    for name in RUN_WINDOWS:
        chain = paths.run(name) / "truth" / "chain"
        if chain.is_dir() and any(chain.glob("*.npz")):
            return True
    return False


def any_run_started(paths: WorldPaths) -> bool:
    return any(paths.run(name).exists() for name in RUN_WINDOWS)


def step_files(
    world: World,
    paths: WorldPaths,
    write: Callable[[Any, Path, Sequence[str]], str],
) -> str:
    """Write the world files, or check that the world still regenerates to their hash.

    The four hashed files are never rewritten in place: when the world regenerates to the
    stored hash, only the derived truth-side files whose content changed are replaced
    (each through a temporary name); when it does not, the files are replaced as a whole
    directory only while no run directory exists, and refused otherwise.
    """
    from fleetlab.contracts.world import HASH_FILE, WORLD_FILES

    base_runs = [str(paths.observed)]
    hash_file = paths.files / HASH_FILE
    scratch = paths.root / ".world-check"
    shutil.rmtree(scratch, ignore_errors=True)
    value = write(world, scratch, base_runs)
    try:
        if hash_file.is_file():
            stored = hash_file.read_text().strip()
            if value == stored and verify_world(paths.files):
                hashed = {*WORLD_FILES, HASH_FILE}
                for source in sorted(scratch.iterdir()):
                    if source.name in hashed or not source.is_file():
                        continue
                    target = paths.files / source.name
                    if not target.is_file() or target.read_bytes() != source.read_bytes():
                        os.replace(source, target)
                return value
            if any_run_started(paths):
                raise WorldChanged(
                    f"{paths.files}: the world regenerates to {value[:16]}, not to the stored "
                    f"{stored[:16]}, and its runs exist; use a new root"
                )
        old = paths.root / ".files-old"
        shutil.rmtree(old, ignore_errors=True)
        if paths.files.exists():
            os.replace(paths.files, old)
        os.replace(scratch, paths.files)
        shutil.rmtree(old, ignore_errors=True)
        return value
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def step_run(world: World, paths: WorldPaths, name: str, engine: EngineSettings) -> dict[str, Any]:
    from fleetlab.engine.fleet import run

    jobs = run_jobs(world, name)
    run_dir = paths.run(name)
    if run_complete(run_dir, jobs):
        return {"skipped": True}
    started = time.time()
    result = run(engine.config(run_dir, world.spec.episodes_per_robot_day), jobs)
    extra = result.record.extra
    return {
        "jobs": len(jobs),
        "seconds": round(time.time() - started, 1),
        "chain_ok": result.record.chain_ok,
        "jobs_run_now": extra.get("jobs_run_now"),
        "throughput": extra.get("throughput"),
        "total_s": extra.get("total_s"),
        "incidents": _count(extra.get("incidents")),
    }


def step_records(world: World, paths: WorldPaths, name: str) -> dict[str, Any]:
    """The description, the change and maintenance logs and the health checks of the run's
    days into the finished run (the miniature fleet's `write_records`, by windows)."""
    from fleetlab.integration.chain import records_current, write_records

    run_dir = paths.run(name)
    windows = list(RUN_WINDOWS[name])
    if records_current(run_dir, world, windows):
        return {"skipped": True}
    write_records(run_dir, world, windows)
    days = [d for w in windows for d in _days(world, w)]
    return {"health_rows": sum(1 for r in world.health if r["day"] in days)}


def _days(world: World, window: str) -> range:
    r = getattr(world.spec.windows, window)
    return range(r.first, r.last + 1)


def _count(value: object) -> int:
    return len(value) if isinstance(value, list | dict) else 0


def step_events(paths: WorldPaths, name: str, engine: EngineSettings) -> dict[str, Any]:
    """The derived events into a finished run: from telemetry, and with `engine.contacts`
    from the contact replay of every episode (`fleetlab.worlds.events`), whose episodes
    must match their chains at every step."""
    import importlib

    events: Any = importlib.import_module("fleetlab.worlds.events")
    run_dir = paths.run(name)
    out: dict[str, Any] = {}
    contacts = None
    if engine.contacts:
        started = time.time()
        found = events.contact_pass(run_dir, workers=engine.workers)
        contacts = found.events
        out["contact_replay"] = {
            "episodes": len(found.seconds),
            "matched": len(found.events),
            "mismatched": dict(sorted(found.mismatched.items())),
            "seconds": round(time.time() - started, 1),
            "episode_seconds_mean": round(
                sum(found.seconds.values()) / max(1, len(found.seconds)), 2
            ),
        }
        if found.mismatched:
            note(paths, f"{name}-events", contact_replay=out["contact_replay"])
            raise RuntimeError(
                f"{run_dir}: {len(found.mismatched)} episodes did not replay to their chains "
                f"in the contact pass: {sorted(found.mismatched)[:5]}"
            )
    out["counts"] = events.add_events(run_dir, contacts)
    return out


def step_export(world: World, paths: WorldPaths) -> dict[str, Any]:
    from fleetlab.log import ExportRefused, export_runs
    from fleetlab.log.leaks import matching_export_check, write_export_check

    note(paths, "export", leaks_clean=False, audit_seconds=0.0)
    chain = _chain_ok(paths.observed)
    if chain is not True:
        raise RuntimeError(
            f"{paths.observed}: chain_ok is {chain!r}; a run whose chain sample broke is "
            "not exported (contracts sections 4 and 9)"
        )
    if (paths.export / "data_dictionary.md").is_file():
        # Neither a saved flag nor an older checker can bless this tree.
        if matching_export_check(
            paths.export,
            [paths.observed],
            fleet=world.fleet,
            receipt_dir=paths.root / "export-checks",
        ):
            return {"skipped": True, "leaks_clean": True}
        checked = write_export_check(
            paths.export,
            [paths.observed],
            fleet=world.fleet,
            receipt_dir=paths.root / "export-checks",
        )
        note(paths, "export", audit_seconds=checked["seconds"])
        if not checked["clean"]:
            raise ExportRefused(
                [
                    *checked["problems"],
                    *(f"leak audit: {h['where']}: {h['term']!r}" for h in checked["hits"]),
                ]
            )
        return {"skipped": True, "leaks_clean": True, "receipt": checked["receipt"]}
    staged_check: dict[str, Any] | None = None

    def check_before_publish(stage: Path) -> None:
        nonlocal staged_check
        staged_check = write_export_check(
            stage,
            [paths.observed],
            fleet=world.fleet,
            receipt_dir=paths.root / "export-checks",
            publish_as=paths.export,
        )
        note(paths, "export", audit_seconds=staged_check["seconds"])
        if not staged_check["clean"]:
            raise ExportRefused(
                [
                    *staged_check["problems"],
                    *(f"leak audit: {h['where']}: {h['term']!r}" for h in staged_check["hits"]),
                ]
            )

    report = export_runs(
        [paths.observed],
        paths.export,
        fleet=world.observed_fleet,
        audit_fleet=world.fleet,
        task_classes=world.task_classes,
        audit=False,
        before_publish=check_before_publish,
    )
    assert staged_check is not None  # The staging hook must finish before publication.
    return {
        "episodes": report.episodes,
        "step_rows": report.step_rows,
        "leaks_clean": True,
        "bytes": sum(report.sizes.values()),
        "receipt": staged_check["receipt"],
    }


# ---- the whole -------------------------------------------------------------------------------


class RunnerBusy(RuntimeError):
    """Another process runs this world (its runner lock is held)."""


@contextmanager
def runner_lock(paths: WorldPaths) -> Iterator[None]:
    """One runner per world root at a time (a `generate` and a `run`, or two runs)."""
    import fcntl

    paths.root.mkdir(parents=True, exist_ok=True)
    with (paths.root / "runner.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunnerBusy(f"{paths.root}: another runner holds the lock") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def is_ready(
    record: dict[str, Any], paths: WorldPaths | None = None, *, fleet: Fleet | None = None
) -> bool:
    """Every step done, both chains held and the exact export checked by current code."""
    from fleetlab.log.leaks import matching_export_check

    steps = record.get("steps", {})
    complete = all(steps.get(s, {}).get("done") for s in STEPS) and all(
        steps.get(name, {}).get("chain_ok") is True for name in RUN_WINDOWS
    )
    if not complete or paths is None:
        return False
    return matching_export_check(
        paths.export, [paths.observed], fleet=fleet, receipt_dir=paths.root / "export-checks"
    )


def readiness_problem(record: dict[str, Any], paths: WorldPaths) -> str | None:
    from fleetlab.log.leaks import export_check_problem

    steps = record.get("steps", {})
    if not all(steps.get(s, {}).get("done") for s in STEPS) or not all(
        steps.get(name, {}).get("chain_ok") is True for name in RUN_WINDOWS
    ):
        return "world steps or chain samples are unfinished; finish the world runner"
    return export_check_problem(paths.export, receipt_dir=paths.root / "export-checks")


def run_world(
    world: World,
    paths: WorldPaths,
    engine: EngineSettings,
    *,
    write: Callable[[Any, Path, Sequence[str]], str] | None = None,
    through: str | None = None,
    only: str | None = None,
    summarize: Callable[[World, WorldPaths], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every step of the world in order, resuming; returns runner.json's content.

    A step already done is skipped when what it reads did not change: the events of a run
    are derived again only when that run ran jobs in this call (or never had them).
    """
    from fleetlab.worlds import summary as summary_module
    from fleetlab.worlds.world import write_world

    writer = write or write_world
    summarizer = summarize or summary_module.summary
    chosen = list(STEPS)
    if through is not None:
        chosen = chosen[: chosen.index(through) + 1]
    if only is not None:
        chosen = [only]
    with runner_lock(paths):
        set_fields(paths, world_id=world.spec.world_id, ready=False)
        ran: set[str] = set()
        for step in chosen:
            before = paths.read_record().get("steps", {}).get(step, {})
            if step.endswith(("-records", "-events")):
                name = step.split("-")[0]
                if before.get("done") and name not in ran and only is None:
                    continue
            with Timed(paths, step):
                if step == "files":
                    value = step_files(world, paths, writer)
                    note(paths, step, hash=value)
                elif step in RUN_WINDOWS:
                    outcome = step_run(world, paths, step, engine)
                    if not outcome.get("skipped"):
                        ran.add(step)
                    note(paths, step, **outcome)
                    if outcome.get("skipped"):
                        note(paths, step, chain_ok=_chain_ok(paths.run(step)))
                elif step.endswith("-records"):
                    note(paths, step, **step_records(world, paths, step.split("-")[0]))
                elif step.endswith("-events"):
                    note(paths, step, **step_events(paths, step.split("-")[0], engine))
                elif step == "export":
                    note(paths, step, **step_export(world, paths))
                elif step == "summary":
                    set_fields(paths, summary=summarizer(world, paths))
        ready = is_ready(paths.read_record(), paths, fleet=world.fleet)
        if not ready:
            note(paths, "export", leaks_clean=False)
        set_fields(paths, ready=ready)
    return paths.read_record()


def _chain_ok(run_dir: Path) -> bool | None:
    from fleetlab.integration.oracle import run_record

    if not (run_dir / "run.json").is_file():
        return None
    return run_record(run_dir).chain_ok
