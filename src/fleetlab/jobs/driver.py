"""The job driver: one investigation of one job request, its release run and its record
(docs/design/jobs.md).

    python -m fleetlab.jobs.driver --request REQUEST.json --out DIR \
        [--library JOB-ONE-LIBRARY] [--backend codex --model gpt-6-astra --effort xhigh \
         --service-tier fast] [--max-sessions N] [--runner command|toy] [--scripted toy] \
        [--assignment HOST-RECORD.json --cohort NAME] [--prior-run EARLIER-DIR ...]

An investigation is a sequence of agent sessions, each one call of the agent runtime
(`fleetlab.agents.sessions.run_session`), as the loop's round driver runs them:

1. Before every session the host publishes the request (`RequestState`, host-owned) into
   the analysis' results, where the tool server's job tools read it
   (`fleetlab.toolserver.jobs`), and the session's export: the installation's record with
   the job tables of every batch run so far.
2. The session uses the tools: it reads the request, proposes the acceptance condition
   (the scripted shift lead answers at once), queues job programs on development starts,
   reads earlier results, writes checks, saves a job and finally gives its account.
3. After the session the host copies the acceptance exchange into the request's message
   log (`fleetlab.jobs.operator.sync_session`, the lead's rule decided again). Queued
   programs run as one batch (`JobRunner.run`: lane J's runner as a heavy job of its own,
   or the CPU toy), their outcomes are checked against their requests and recorded, and
   the next session starts on the new export.
4. The investigation is `done` when the agent's account is accepted; it is `incomplete`
   when an authentication or rate limit, a lost call or the session budget ends it first.
   An incomplete investigation stays incomplete: there is no resumption (the output
   directory must be new), and a fresh investigation is a separate run with its own
   budget.

After a done investigation the host freezes the library (the saved jobs, the checks, the
mounted library's entries and the agent's learned skills; lane J's `library` writes the
files) and runs the release: the saved job the account names, on the reserved release
starts, beside the naive chain on the same starts (`JobRunner.release`, lane J's operator
CLI), whose receipt counts the deliveries against the approved condition. The record
(`DIR/job_run.json`) holds the request, every session, every batch, the costs (sessions,
tool calls, model time, tokens, starts simulated), the account and the receipt.

`--library` mounts job one's library for job two: its jobs, checks and recipes into the
results (`job_library`, `run_check`), its learned skills beside the base skills.

Before anything starts, the host checks the request's allocation (`jobs.prior.preflight`):
a request with an allocation profile names its pool in the host's assignment record
(`--assignment`, `--cohort`); every public start means one identity across what the
investigation carries; and the request's evaluation starts are fresh, examined by no
earlier attempt the carried libraries record, for any request. Every start the driver
derives, for a development request or a release, is the request's pool's under its
profile (`jobs.starts.start_for`); a release run from other identities is refused. The
session cap is the request's budget unless `--max-sessions` lowers it.

`--prior-run` (repeatable) names earlier investigations of the job line, a numbered
continuation's predecessors (`jobs.prior`): their public records are read and checked
before the preflight, which then sees their episodes' starts and their exposure as well,
and written under DIR/prior once it passes: the remapped job tables and videos, which every
export holds before this investigation's batches; the host's registry; their exposure
lines, carried into every library this investigation freezes; and the public history
(results/history.json), which the tools read (`toolserver.jobs.PublicHistory`).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fleetlab.contracts.jobs import (
    GOAL_SUITE_TASKS,
    RELEASE_INIT_STATES,
    CheckSpec,
    JobProgram,
    JobRequest,
    ReleasePair,
    ReleaseReceipt,
    RequestState,
    StartIdentity,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.tools import JOB_TOOLS
from fleetlab.jobs.operator import (
    ShiftLead,
    account_of,
    describe,
    load_request,
    publish,
    sync_session,
)
from fleetlab.toolserver.files import LEARNED_FILE, append_line, read_lines
from fleetlab.toolserver.jobs import (
    CHECKS_FILE,
    JOB_MESSAGES,
    JOB_REQUESTS_FILE,
    JOB_RESULTS_FILE,
    JOB_TABLES,
    LEAD_FRACTION,
    LIBRARY_DIR,
    SAVED_JOBS_FILE,
    JobOutcome,
    job_outcome_problems,
    job_tool_texts,
)

__all__ = [
    "DEFAULT_AGENT",
    "INSTALLATION_ID",
    "JOB_SESSION_TOOLS",
    "SYSTEM",
    "BatchReply",
    "CommandRunner",
    "Installation",
    "JobBatch",
    "JobRunner",
    "ToyRunner",
    "installation",
    "investigate",
    "job_export",
    "write_job_tables",
]

REPO = Path(__file__).resolve().parents[3]
RESULTS = Path.home() / ".cache" / "robots-probes" / "demo" / "2026-10-20" / "tools"
INSTALLATION_ID = "w90"  # the installation's id as the export and the analysis name it
BATCH_FORMAT = "fleetlab.job-batch/1"
RECORD_FILE = "job_run.json"
STOPPING_ERRORS = ("auth", "rate_limit")
NO_ACCEPTANCE = "the host holds no approved acceptance condition; no program is tested"

# The tools of a job session: the analysis tools that read the job tables, the learned
# skills, the notes, and the job tools (contracts.tools.JOB_TOOLS).
JOB_SESSION_TOOLS: tuple[str, ...] = (
    "query",
    "get_frames",
    "list_skills",
    "read_skill",
    "memory_read",
    "memory_write",
    "write_learned_skill",
    *JOB_TOOLS,
)

SYSTEM = """You are the engineering agent of a robot installation: one robot arm at a kitchen
table, run by a fixed policy that performs single tasks when it is given an instruction.
The policy does not change. A shift lead asks the installation for a job; you make the
installation able to do it, using the tools of this session, and report to the shift lead.
Every number you report comes from what the tools returned. Complete the work
autonomously; report uncertainty rather than asking."""


@dataclass(frozen=True)
class AgentDefaults:
    backend: str = "codex"
    model: str = "gpt-6-astra"
    effort: str = "xhigh"
    service_tier: str = "fast"


DEFAULT_AGENT = AgentDefaults()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Starts.


# Starts.


def start_of(base_task: int, init_state: int, *, profile: str | None) -> StartIdentity:
    """The host's start for a base task and a stored initial state under the request's
    allocation profile (`fleetlab.jobs.starts.start_for`): the one call site of the
    derivation, for development requests and the toy's release alike."""
    from fleetlab.jobs.starts import start_for

    return start_for(base_task, init_state, profile=profile)


# The installation's record and the session exports.


@dataclass(frozen=True)
class Installation:
    """The installation as a fleet record of one robot and no episodes: the run the job
    exports are built from, and the truth-side inputs of the launch audit."""

    run: Path
    fleet: Fleet


def installation(base: Path) -> Installation:
    """Write the installation's finalized run under `base` (once)."""
    from fleetlab.contracts.identity import SoftwareVersion
    from fleetlab.contracts.model import ComponentState, ConfigVersion, Robot, Site
    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.contracts.vocabulary import Component
    from fleetlab.log import RunWriter

    robot = Robot(
        id="r01",
        site_id="s1",
        components={c: ComponentState(component=c, installed_day=0) for c in Component},
        traits={},
    )
    fleet = Fleet(
        id=INSTALLATION_ID,
        sites=[Site(id="s1", robots=[robot])],
        config=ConfigVersion(version=1, values={}),
        change_log=[],
        maintenance_log=[],
    )
    run = base / "run"
    if not (run / "run.json").is_file():
        software = SoftwareVersion(
            lock_hash="installation",
            libero_patch_set="none",
            mujoco_version="3.3.7",
            miopen_db_hash="none",
        )
        record = RunRecord(software=software, batch_size=1, replay_rules=[])
        with RunWriter(run, record, episodes_per_robot_day=1) as writer:
            writer.write_model(fleet, ())
            writer.finalize()
    return Installation(run, fleet)


def write_job_tables(stage: Path, runs: Sequence[Path]) -> None:
    """The job tables of every batch run, into an export's staging tree at the paths of
    `contracts.log.SCHEMA`, with the videos the episodes name (`log/video/`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fleetlab.contracts.log import SCHEMA, arrow_schema

    for name in JOB_TABLES:
        spec = SCHEMA.get(name)
        if spec is None:
            raise RuntimeError(f"the job table {name} is not registered in contracts.log")
        pattern = spec.path
        key = spec.file_key
        if key is not None:
            pattern = pattern.replace("{" + key + "}", "*")
        schema = arrow_schema(name, visible_only=True)
        parts = []
        for run in runs:
            for path in sorted(run.glob(pattern)):
                table = pq.read_table(path).select(schema.names).cast(schema)
                if key is not None:
                    target = stage / path.relative_to(run)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table.replace_schema_metadata(None), target)
                else:
                    parts.append(table)
        if parts:
            target = stage / spec.path
            target.parent.mkdir(parents=True, exist_ok=True)
            merged = pa.concat_tables(parts).replace_schema_metadata(None)
            pq.write_table(merged, target)
    for run in runs:
        for video in sorted((run / "log" / "video").glob("*.mp4")):
            target = stage / "log" / "video" / video.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video, target)


def job_export(install: Installation, runs: Sequence[Path], out: Path) -> Path:
    """A session's export: the installation's record, the job tables of `runs`, audited."""
    from fleetlab.log import export_runs

    export_runs(
        [install.run],
        out,
        fleet=install.fleet,
        task_classes=(),
        extra_tables=lambda stage: tables_writer()(stage, runs),
    )
    return out


# Runners.


@dataclass(frozen=True)
class JobBatch:
    """The jobs one session queued, as the runner receives them (`fleetlab.jobs.runner`):
    the request's state (goals and order rules, which a JobRequest does not carry) and each
    job's request. Episode ids follow the runner's numbering, job number times 100 plus
    the start's position from 1 (`j0003` gives `e000301` to `e000320`), unique across an
    investigation because job ids are."""

    session: int
    state: RequestState
    jobs: tuple[tuple[str, JobRequest], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format": BATCH_FORMAT,
            "session": self.session,
            "state": self.state.model_dump(mode="json"),
            "jobs": [
                {"job_id": job_id, "request": request.model_dump(mode="json")}
                for job_id, request in self.jobs
            ],
        }


def episode_id_of(job_id: str, position: int) -> str:
    """The runner's episode id of a job's `position`-th start, from 1."""
    return f"e{int(job_id[1:]) * 100 + position:06d}"


@dataclass
class BatchReply:
    """What a batch left: an outcome per job, the run directory with its tables, a record."""

    outcomes: list[JobOutcome]
    run_dir: Path | None
    record: dict[str, Any] = field(default_factory=dict)


class JobRunner(Protocol):
    """Runs a session's jobs and the release of a saved job (lane J's runtime, or the toy)."""

    def run(self, batch: JobBatch, out: Path) -> BatchReply: ...

    def release(
        self, job: str, library: Path, n: int, k: int | None, out: Path
    ) -> ReleaseReceipt: ...


def _outcome(
    job_id: str,
    status: str,
    episodes: Sequence[str] = (),
    digest: str = "",
    problems: Sequence[str] = (),
) -> JobOutcome:
    return JobOutcome(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        episode_ids=list(episodes) if status == "done" else [],
        message=JOB_MESSAGES[status],
        program_digest=digest,
        problems=[str(p)[:300] for p in problems][:10] if status == "refused" else [],
    )


@dataclass
class CommandRunner:
    """Lane J's runner and operator CLI, each call a heavy job of its own under
    `tools/heavy-run`, so the policy is held only while a batch runs."""

    runner: Sequence[str] = ("uv", "run", "--locked", "python", "-m", "fleetlab.jobs.runner")
    operator: Sequence[str] = ("uv", "run", "--locked", "python", "-m", "fleetlab.jobs")
    heavy: Sequence[str] = (str(REPO / "tools" / "heavy-run"),)
    queue_log: Path | None = None
    # "xvla" (heavy, the GPU) or "hold" (lane J's policy that holds still: the simulator on
    # the CPU without a model, a light job, for a check of the machinery).
    policy: str = "xvla"

    @classmethod
    def holding(cls, queue_log: Path | None = None) -> CommandRunner:
        """The runner with the holding policy, as a light job."""
        light = (str(REPO / "tools" / "heavy-run"), "--light", "-m", "6G")
        return cls(heavy=light, queue_log=queue_log, policy="hold")

    def _call(self, argv: Sequence[str], log: Path, purpose: str) -> tuple[int, float]:
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        _queue_log(self.queue_log, "queued", purpose)
        env = dict(os.environ)
        env.setdefault(
            "FLEETLAB_MIOPEN_DIR",
            str(Path.home() / ".cache" / "robots-probes" / "engine" / "miopen"),
        )
        policy = ["--policy", self.policy]
        with log.open("w") as stream:
            done = subprocess.run(
                [*self.heavy, *argv, *policy],
                cwd=REPO,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
            )
        seconds = time.perf_counter() - started
        _queue_log(self.queue_log, f"ended({done.returncode})", purpose)
        return done.returncode, seconds

    def run(self, batch: JobBatch, out: Path) -> BatchReply:
        out.mkdir(parents=True, exist_ok=False)
        batch_file = out / "batch.json"
        batch_file.write_text(json.dumps(batch.to_json(), indent=1))
        run_dir = out / "run"
        code, seconds = self._call(
            [*self.runner, "--batch", str(batch_file), "--out", str(run_dir)],
            out / "runner.log",
            f"jobs:session-{batch.session}",
        )
        record: dict[str, Any] = {
            "exit": code,
            "seconds": seconds,  # the call under tools/heavy-run, its wait for the lock included
            "run_dir": str(run_dir),
            **runner_timings(out / "runner.log"),
        }
        result_file = run_dir / "batch_result.json"
        if code != 0 or not result_file.is_file():
            # The run broke off: every job stays charged, and nothing it left is shown.
            record["failed"] = True
            return BatchReply([_outcome(j, "failed") for j, _ in batch.jobs], None, record)
        reported = {r["job_id"]: r for r in json.loads(result_file.read_text())["jobs"]}
        outcomes = []
        for job_id, _ in batch.jobs:
            entry = reported.get(job_id, {"status": "failed"})
            outcomes.append(
                _outcome(
                    job_id,
                    entry["status"],
                    entry.get("episode_ids", []),
                    str(entry.get("program_digest", "")),
                    entry.get("problems", []),
                )
            )
        return BatchReply(outcomes, run_dir, record)

    def release(self, job: str, library: Path, n: int, k: int | None, out: Path) -> ReleaseReceipt:
        out.mkdir(parents=True, exist_ok=False)
        argv = [*self.operator, "run", job, "--library", str(library), "--starts", "release",
                "--n", str(n), "--video", "--out", str(out / "run")]  # fmt: skip
        code, _ = self._call(argv, out / "release.log", f"jobs:release-{job}")
        receipt = out / "run" / "receipt.json"
        if code != 0 or not receipt.is_file():
            raise RuntimeError(f"the release run of {job} failed (exit {code}); see {out}")
        return ReleaseReceipt.model_validate_json(receipt.read_text())


def runner_timings(log: Path) -> dict[str, Any]:
    """The runner's own timings from its log (lane J's runner prints them): the policy's
    load, the runner's total, and each finished episode's wall time; null where absent."""
    import re

    text = log.read_text(errors="replace") if log.is_file() else ""
    load = re.search(r"policy \S+ ready in ([\d.]+) s", text)
    total = re.search(r"^runner: \{.*\} in ([\d.]+) s$", text, re.MULTILINE)
    walls = [float(w) for w in re.findall(r" wall=([\d.]+)s ", text)]
    return {
        "waited_for_lock": "another heavy job holds the lock" in text,
        "policy_load_seconds": float(load.group(1)) if load else None,
        "runner_seconds": float(total.group(1)) if total else None,
        "episode_seconds": walls,
    }


def _queue_log(path: Path | None, event: str, purpose: str) -> None:
    if path is None:
        return
    with path.open("a") as stream:
        stream.write(f"{_now()} {event} lane=tools commit={_commit()[:12]} purpose={purpose}\n")


# The toy: a CPU stand-in of the runner for checks and scripted runs.


class ToyRunner:
    """A CPU stand-in for lane J's runner: no simulator, no policy, a fixed toy physics.

    A policy step whose `until` names a goal reaches it. The toy's difficulty is the
    plan's: a goal reached by a policy step is undone on six starts in ten when the next
    step is another policy step straight away (the hand still holds the item); a release
    (a motion step opening the gripper) or a wait of at least ten steps in between keeps
    it. A goal that first holds before a goal an order rule puts first breaks that rule.
    The tables follow the registered schemas of `contracts.log`, columns filled by name
    (the ones the tool server reads) and by type (the rest); a run directory also holds the
    runner's record (`run.json`: the request, each job's program, starts and first
    episode), as lane J's does. With `videos` every episode that asked for video gets a
    scene video of two frames, the reset and the end. The release is laid out as the
    operator's: its attempt written to the library's exposure log before anything plays,
    then the program's (j0001) and the naive chain's (j0002) episodes in OUT/run with their
    tables and the receipt."""

    def __init__(self, state: RequestState, videos: bool = False) -> None:
        self.state = state
        self.videos = videos
        self.episodes = 0
        self.receipts: list[ReleaseReceipt] = []

    # The toy physics.

    def play(self, program: JobProgram, init_state: int) -> dict[str, Any]:
        goals = len(self.state.requested_goals)
        holds = [False] * goals
        first: list[int | None] = [None] * goals
        events = []
        step = 0
        last_goal: int | None = None
        released = True
        for index, s in enumerate(program.steps):
            start = step
            outcome = "reached"
            if s.kind.value == "policy":
                assert s.until is not None and s.max_steps is not None
                if last_goal is not None and not released and init_state % 10 < 6:
                    holds[last_goal] = False  # the item came along: the earlier goal undone
                taken = min(50, s.max_steps) if s.until.goal is not None else s.until.steps or 1
                step += taken
                if s.until.goal is not None:
                    holds[s.until.goal] = True
                    if first[s.until.goal] is None:
                        first[s.until.goal] = step
                    last_goal = s.until.goal
                released = False
            elif s.kind.value == "motion":
                assert s.segment is not None
                step += s.segment.duration_steps
                if s.segment.primitive.value == "open_gripper":
                    released = True
            elif s.kind.value == "home":
                assert s.duration_steps is not None
                step += s.duration_steps
                released = True  # the gripper opens on the way home
            else:
                assert s.steps is not None
                step += s.steps
                if s.steps >= 10:
                    released = True
            events.append((index, s.kind.value, start, step, outcome))
        kept = all(
            first[r.first] is not None
            and (first[r.then] is None or (first[r.first] or 0) <= (first[r.then] or 0))
            for r in self.state.order_rules
        )
        delivered = all(holds) and init_state % 10 != 9
        return {
            "holds": holds,
            "first": first,
            "events": events,
            "steps": step,
            "order_kept": kept,
            "delivered": delivered,
        }

    def naive(self) -> JobProgram:
        """The naive chain: each requested goal's skill, switched on the goal holding."""
        from fleetlab.contracts.jobs import JobStep

        steps = [
            JobStep.model_validate(
                {
                    "kind": "policy",
                    "instruction": f"achieve {g.text}",
                    "until": {"goal": i, "steps": None},
                    "max_steps": 300,
                    "segment": None,
                    "steps": None,
                    "on_timeout": None,
                }
            )
            for i, g in enumerate(self.state.requested_goals)
        ]
        return JobProgram(name="naive-chain", purpose="the skills chained", steps=steps, notes="")

    # The records.

    def _episode(
        self, job_id: str, program: JobProgram, position: int, start: StartIdentity, video: bool
    ) -> dict[str, Any]:
        self.episodes += 1
        played = self.play(program, start.init_state)
        return {
            "job_id": job_id,
            "episode_id": episode_id_of(job_id, position),
            "start": start,
            "program_digest": program.digest,
            "video": video and self.videos,
            **played,
        }

    @staticmethod
    def _run_record(
        run_dir: Path,
        state: RequestState,
        jobs: Sequence[tuple[str, JobProgram, Sequence[StartIdentity]]],
    ) -> None:
        """The runner's record of a run directory, as lane J's runner writes it."""
        record = {
            "format": "fleetlab.job-run/1",
            "toy": True,
            "state": state.model_dump(mode="json"),
            "jobs": [
                {
                    "job_id": job_id,
                    "program_digest": program.digest,
                    "program": program.model_dump(mode="json"),
                    "first_episode": int(job_id[1:]) * 100 + 1,
                    "starts": [s.model_dump(mode="json") for s in starts],
                }
                for job_id, program, starts in jobs
            ],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    def _write(self, run_dir: Path, episodes: Sequence[Mapping[str, Any]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from fleetlab.contracts.log import SCHEMA, arrow_schema

        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in JOB_TABLES}
        for e in episodes:
            start: StartIdentity = e["start"]
            video = None
            if e.get("video"):
                import numpy as np

                from fleetlab.contracts.env import Camera
                from fleetlab.jobs.tables import video_relative
                from fleetlab.video import write_mp4

                video = video_relative(e["episode_id"], Camera.scene)
                frames = [np.full((16, 16, 3), value, np.uint8) for value in (40, 200)]
                write_mp4(run_dir / "log" / video, frames, 5, repeatable=True)
            rows["job_episodes"].append(
                {
                    "job_id": e["job_id"],
                    "episode_id": e["episode_id"],
                    "program_digest": e["program_digest"],
                    "base_task": start.base_task,
                    "init_state": start.init_state,
                    "delivered": e["delivered"],
                    "order_kept": e["order_kept"],
                    "control_steps": e["steps"],
                    "goals_held_at_end": e["holds"],
                    "delivery_detail": "toy",
                    "order_detail": "toy",
                    "gripper_opening_at_end": 0.08,
                    "video_scene": video,
                    "video_wrist": None,
                }
            )
            for index, kind, begin, end, outcome in e["events"]:
                rows["job_step_events"].append(
                    {
                        "episode_id": e["episode_id"],
                        "program_step": index,
                        "kind": kind,
                        "start_step": begin,
                        "end_step": end,
                        "outcome": outcome,
                    }
                )
            for goal, held in enumerate(e["holds"]):
                first = e["first"][goal]
                # At the reset, where the goal first held, and at the end.
                marks = {0: False, e["steps"]: held}
                if first is not None and first < e["steps"]:
                    marks[first] = True
                for step, holds in sorted(marks.items()):
                    rows["goal_steps"].append(
                        {"episode_id": e["episode_id"], "step": step, "goal": goal, "holds": holds}
                    )
            for frame, step in enumerate((0, e["steps"])):
                rows["robot_steps"].append(
                    {
                        "episode_id": e["episode_id"],
                        "step": step,
                        "program_step": 0,
                        "instruction": "toy",
                        "gripper_opening": 0.08,
                        "frame_index": frame if video else None,
                        "phase": "end" if step else "program",
                    }
                )
                for requested in self.state.requested_goals:
                    for obj in requested.objects:
                        rows["object_steps"].append(
                            {"episode_id": e["episode_id"], "step": step, "object": obj}
                        )
        for name, table_rows in rows.items():
            spec = SCHEMA[name]
            schema = arrow_schema(name, visible_only=True)
            filled = [{f.name: r.get(f.name, _default(f.type)) for f in schema} for r in table_rows]
            if spec.file_key is not None:
                by_key: dict[str, list[dict[str, Any]]] = {}
                for r in filled:
                    by_key.setdefault(str(r[spec.file_key]), []).append(r)
                groups = {
                    spec.path.replace("{" + spec.file_key + "}", k): v for k, v in by_key.items()
                }
            else:
                groups = {spec.path: filled}
            for relative, group in groups.items():
                target = run_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(group, schema=schema), target)

    def run(self, batch: JobBatch, out: Path) -> BatchReply:
        out.mkdir(parents=True, exist_ok=False)
        (out / "batch.json").write_text(json.dumps(batch.to_json(), indent=1))
        run_dir = out / "run"
        self._run_record(run_dir, batch.state, [(j, r.program, r.starts) for j, r in batch.jobs])
        episodes = []
        outcomes = []
        for job_id, request in batch.jobs:
            mine = [
                self._episode(job_id, request.program, i, s, request.record_video)
                for i, s in enumerate(request.starts, 1)
            ]
            episodes += mine
            outcomes.append(
                _outcome(job_id, "done", [e["episode_id"] for e in mine], request.program.digest)
            )
        self._write(run_dir, episodes)
        record = {"toy": True, "starts": sum(len(r.starts) for _, r in batch.jobs)}
        return BatchReply(outcomes, run_dir, record)

    def release(
        self, job: str, library_root: Path, n: int, k: int | None, out: Path
    ) -> ReleaseReceipt:
        saved = load_saved_job(library_root, job)
        out.mkdir(parents=True, exist_ok=False)
        run_dir = out / "run"  # the operator's layout: the run, its tables and the receipt
        starts = [
            start_of(saved.base_task, i, profile=saved.allocation_profile)
            for i in list(RELEASE_INIT_STATES)[:n]
        ]
        shared = library()
        exposed: set[StartIdentity] = set()
        attempt = None
        if hasattr(shared, "record_release_exposure"):  # lane J's library, as its release CLI
            exposed = shared.exposed_identities(library_root)
            attempt = shared.record_release_exposure(library_root, saved, starts, run_dir)
        fresh = not exposed & set(starts)
        # Only now does anything play: the attempt is in the exposure log before its first
        # episode, as the operator's release writes it.
        naive = self.naive()
        operator = RequestState(
            request_text=f"operator run of the saved job {saved.name}",
            requested_goals=saved.requested_goals,
            base_task=saved.base_task,
            order_rules=saved.order_rules,
            budget=self.state.budget,
            acceptance=saved.acceptance,
            message_log=[],
            allocation_profile=saved.allocation_profile,
        )
        self._run_record(
            run_dir, operator, [("j0001", saved.program, starts), ("j0002", naive, starts)]
        )
        ours = [self._episode("j0001", saved.program, i, s, True) for i, s in enumerate(starts, 1)]
        theirs = [self._episode("j0002", naive, i, s, True) for i, s in enumerate(starts, 1)]
        self._write(run_dir, ours + theirs)
        pairs = [
            ReleasePair(
                start=mine["start"],
                episode_id=mine["episode_id"],
                delivered=mine["delivered"],
                order_kept=mine["order_kept"],
                naive_episode_id=other["episode_id"],
                naive_delivered=other["delivered"],
            )
            for mine, other in zip(ours, theirs, strict=True)
        ]
        delivered = sum(p.delivered for p in pairs)
        violations = sum(not p.order_kept for p in pairs)
        passed = None if k is None else delivered >= k
        receipt = ReleaseReceipt(
            job_name=job,
            program_digest=saved.program.digest,
            naive_digest=naive.digest,
            release_starts=starts,
            N=n,
            K=k,
            delivered=delivered,
            naive_delivered=sum(p.naive_delivered for p in pairs),
            rescued=sum(p.delivered and not p.naive_delivered for p in pairs),
            broken=sum(p.naive_delivered and not p.delivered for p in pairs),
            order_violations=violations,
            passed=passed,
            released=bool(passed) and violations == 0 and fresh,
            pairs=pairs,
            run_dir=str(run_dir),
            starts_fresh=fresh,
            lead_fraction=saved.lead_fraction,
        )
        (run_dir / "receipt.json").write_text(receipt.model_dump_json(indent=1) + "\n")
        if attempt is not None:
            shared.finish_release_exposure(library_root, attempt, receipt)
        self.receipts.append(receipt)
        return receipt


def _default(kind: Any) -> Any:
    """A value of an Arrow type for a toy column the toy does not name."""
    import pyarrow as pa

    if pa.types.is_boolean(kind):
        return False
    if pa.types.is_integer(kind):
        return 0
    if pa.types.is_floating(kind):
        return 0.0
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return ""
    if pa.types.is_fixed_size_list(kind):
        return [_default(kind.value_type)] * kind.list_size
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return []
    return None


# The task text (lane K's `fleetlab.jobs.task_text`, or the placeholder).


def placeholder_task_text(
    state: RequestState,
    session: int,
    library_mounted: bool,
    history: Mapping[str, int] | None = None,
) -> str:
    """A stand-in task until lane K's text lands (`--placeholder-task`): the request, the
    tools in order, the requirement to save the job and a check and to give the account."""
    goals = "; ".join(f"goal {i}: {g.text}" for i, g in enumerate(state.requested_goals))
    lines = [
        f"This is session {session} of the investigation of the shift lead's request.",
        f"The request: {state.request_text}",
        f"What must hold at the end: {goals}. The base task is "
        f"{GOAL_SUITE_TASKS[state.base_task]}.",
        "Read the request with operator_state. Propose the acceptance condition with "
        "submit_acceptance_proposal before testing any program. Compose job programs with "
        "submit_job_program on development starts; they run after the session, and the next "
        "session reads them with job_results and the job tables. Write a check with "
        "write_check, save the job with save_job, and finish with submit_job_account. "
        'A session that queued programs ends with status "waiting"; the one that submitted '
        'the account ends with status "done".',
    ]
    if library_mounted:
        lines.append(
            "The installation's job list (job_library) holds earlier jobs and checks; "
            "run_check runs a saved check."
        )
    if history:
        lines.append(
            f"{history['investigations']} earlier investigations' jobs are imported: "
            "job_results lists them with status imported, job_library has their history."
        )
    return "\n\n".join(lines)


def task_text_for(placeholder: bool) -> Callable[..., str]:
    """Lane K's task text, or the placeholder when asked for: (state, session,
    library_mounted, history=None), `history` the imported history's size or None."""
    if placeholder:
        return placeholder_task_text
    module = importlib.import_module("fleetlab.jobs.task_text")  # lane K
    text: Callable[..., str] = module.task_text
    return text


# The library (jobs/, checks/, recipes/, index.json). Lane J's `fleetlab.jobs.library` owns
# the format and is used when it is importable; `_StandInLibrary` writes the same files
# with the contracts' models until it lands. `library()` is the one place the driver chooses.


class _StandInLibrary:
    """Lane J's library functions as the driver calls them (`save_job`, `load_job`,
    `save_check`), with the index rebuilt on every save."""

    @staticmethod
    def _write(path: Path, text: str, replace: bool) -> Path:
        if path.is_file() and not replace and path.read_text() != text:
            raise FileExistsError(f"{path}: another entry under this name; replace=True")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text)
        temporary.replace(path)
        return path

    @classmethod
    def save_job(cls, lib: Path, saved: Any, replace: bool = False) -> Path:
        path = cls._write(
            lib / "jobs" / f"{saved.name}.json", saved.model_dump_json(indent=1), replace
        )
        write_index(lib)
        return path

    @staticmethod
    def load_job(lib: Path, name: str) -> Any:
        from fleetlab.contracts.jobs import SavedJob

        return SavedJob.model_validate_json((lib / "jobs" / f"{name}.json").read_text())

    @classmethod
    def save_check(cls, lib: Path, check: CheckSpec, replace: bool = False) -> Path:
        path = cls._write(
            lib / "checks" / f"{check.name}.json", check.model_dump_json(indent=1), replace
        )
        write_index(lib)
        return path


def library() -> Any:
    """Lane J's library module, or the stand-in."""
    try:
        return importlib.import_module("fleetlab.jobs.library")
    except ImportError:
        return _StandInLibrary


def _canonical_digest(path: Path) -> str:
    from fleetlab.contracts.hashing import canonical_json

    return hashlib.sha256(canonical_json(json.loads(path.read_text())).encode()).hexdigest()


def write_index(root: Path) -> Path:
    """The stand-in's `index.json` (`LibraryIndex`): every job, check and recipe with its
    purpose and the SHA-256 of its file's canonical JSON."""
    from fleetlab.contracts.jobs import LibraryEntry, LibraryIndex

    entries = []
    for kind, part in (("job", "jobs"), ("check", "checks"), ("recipe", "recipes")):
        for path in sorted((root / part).glob("*.json")):
            data = json.loads(path.read_text())
            purpose = data.get("purpose") or data.get("program", {}).get("purpose", "")
            entry = {
                "kind": kind,
                "name": path.stem,
                "purpose": purpose,
                "digest": _canonical_digest(path),
            }
            entries.append(LibraryEntry.model_validate(entry))
    path = root / "index.json"
    path.write_text(LibraryIndex(entries=entries).model_dump_json(indent=1))
    return path


def save_job(
    root: Path,
    name: str,
    program: JobProgram,
    state: RequestState,
    runs: Sequence[str],
    fraction: tuple[int, int] = LEAD_FRACTION,
    check: str | None = None,
    check_digest: str | None = None,
) -> Path:
    """A saved job (`SavedJob`) with what it must achieve, the runs it was saved from and
    the lead's bar (`fraction`) its acceptance was approved under."""
    from fleetlab.contracts.jobs import SavedJob

    saved = SavedJob(
        name=name,
        program=program,
        requested_goals=state.requested_goals,
        base_task=state.base_task,
        order_rules=state.order_rules,
        acceptance=state.acceptance,
        saved_from=list(runs),
        receipt=None,
        lead_fraction=list(fraction),
        check=check,
        check_digest=check_digest if check is not None else None,
        allocation_profile=state.allocation_profile,
    )
    path: Path = library().save_job(root, saved)
    return path


def load_saved_job(root: Path, name: str) -> Any:
    return library().load_job(root, name)


def attach_receipt(root: Path, name: str, receipt: ReleaseReceipt) -> None:
    """The release receipt into the saved job's entry."""
    saved = load_saved_job(root, name)
    library().save_job(root, saved.model_copy(update={"receipt": receipt}), replace=True)


def save_check(root: Path, check: CheckSpec) -> Path:
    path: Path = library().save_check(root, check)
    return path


EXPOSURE_FILE = "release_exposure.jsonl"  # lane J's `library.EXPOSURE_FILE`


def carry_exposure(source: Path, target: Path) -> None:
    """The library's release exposure log (every release attempt, written by lane J's
    release run before it runs) into another library: the lines the target lacks, appended
    byte for byte, so that a later request judges freshness on everything a release has
    examined (`library.carry_exposure`)."""
    from fleetlab.jobs.library import carry_exposure as carry

    carry(source, target)


def tables_writer() -> Callable[[Path, Sequence[Path]], None]:
    """Lane J's `fleetlab.jobs.tables.write_job_tables`, or the stand-in of this module."""
    try:
        module = importlib.import_module("fleetlab.jobs.tables")
    except ImportError:
        return write_job_tables
    shared: Callable[[Path, Sequence[Path]], None] = module.write_job_tables
    return shared


# The scripted toy agent (`--scripted toy`; fleetlab.loop.scripted's stand-in for the model):
# four sessions that use every job tool on the toy runner's records.


def _data(result: Any) -> Any:
    text = "\n".join(p["text"] for p in result.content if p.get("type") == "text")
    if result.is_error:
        raise RuntimeError(f"tool error: {text}")
    return json.loads(text)


def _step(kind: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": kind,
        "instruction": None,
        "until": None,
        "max_steps": None,
        "segment": None,
        "steps": None,
        "on_timeout": None,
    }
    base.update(fields)
    return base


def _policy(instruction: str, goal: int) -> dict[str, Any]:
    return _step(
        "policy", instruction=instruction, until={"goal": goal, "steps": None}, max_steps=300
    )


def toy_programs(state: RequestState) -> dict[str, dict[str, Any]]:
    """The scripted agent's programs: the naive chain, the chain with a release between
    the skills, and the reverse order (which breaks a rule putting goal 0 first)."""
    goals = list(range(len(state.requested_goals)))
    release = [
        _step("motion", segment={"primitive": "open_gripper", "axis": None, "amplitude_m": None,
                                 "duration_steps": 15, "period_steps": None}),
        _step("motion", segment={"primitive": "ramp", "axis": "z", "amplitude_m": 0.05,
                                 "duration_steps": 20, "period_steps": None}),
        _step("wait", steps=15),
    ]  # fmt: skip
    naive = [_policy(f"achieve goal {g}", g) for g in goals]
    released: list[dict[str, Any]] = []
    for g in goals:
        if released:
            released += release
        released.append(_policy(f"achieve goal {g}", g))

    def program(name: str, purpose: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        return {"name": name, "purpose": purpose, "steps": steps, "notes": "scripted"}

    return {
        "naive": program("chain-naive", "the skills chained on their goals", naive),
        "release": program("chain-release", "release and retreat between skills", released),
        "reverse": program("chain-reverse", "the skills in the other order", naive[::-1]),
    }


def toy_steps(state: RequestState, starts: int = 10, k: int = 16) -> list[Callable[[Any], str]]:
    """The four sessions of the scripted toy agent: the proposal and the naive chain; the
    check and two variants; the saves (one refused by the order rule) before the release;
    the account with the release's counts."""
    programs = toy_programs(state)
    check: dict[str, Any] = {
        "name": "goal-undone",
        "purpose": "episodes of a job in which a requested goal held and did not hold at the end",
        "parameters": [{"name": "goal", "type": "int"}],
        "sql": "SELECT e.job_id, count(DISTINCT g.episode_id) AS undone FROM goal_steps g "
        "JOIN job_episodes e USING (episode_id) WHERE g.goal = $goal AND g.holds AND NOT "
        "EXISTS (SELECT 1 FROM goal_steps h WHERE h.episode_id = g.episode_id AND h.goal = "
        "g.goal AND h.holds AND h.step = (SELECT max(step) FROM goal_steps k WHERE "
        "k.episode_id = g.episode_id)) GROUP BY e.job_id ORDER BY e.job_id",
        "expects": "zero for a job whose later steps keep the earlier goals",
    }

    def first(box: Any) -> str:
        _data(box.call("operator_state", {}))
        low = _data(box.call("submit_acceptance_proposal", {"N": 10, "K": 8, "reason": "few"}))
        assert not low["approved"], low
        ok = _data(box.call("submit_acceptance_proposal", {"N": 20, "K": k, "reason": "bar"}))
        assert ok["approved"], ok
        receipt = _data(
            box.call(
                "submit_job_program",
                {
                    "program": programs["naive"],
                    "starts": list(range(starts)),
                    "record_video": False,
                },
            )
        )
        assert receipt["accepted"], receipt
        _data(box.call("memory_write", {"notes": f"naive chain queued as {receipt['job_id']}"}))
        return "waiting"

    def installation_has(box: Any, part: str, name: str) -> bool:
        listed = _data(box.call("job_library", {}))["installation"][part]
        return any(entry["name"] == name for entry in listed)

    def second(box: Any) -> str:
        results = _data(box.call("job_results", {}))
        assert results[0]["status"] == "done", results
        _data(
            box.call(
                "query",
                {
                    "sql": "SELECT job_id, count(*) FILTER (WHERE delivered) AS d, "
                    "count(*) AS n FROM job_episodes GROUP BY job_id"
                },
            )
        )
        if not installation_has(box, "checks", check["name"]):  # job two reuses job one's
            written = _data(
                box.call(
                    "write_check", {"check": check, "check_with": [{"name": "goal", "value": "0"}]}
                )
            )
            assert written["accepted"], written
        _data(
            box.call(
                "run_check",
                {"name": "goal-undone", "job_id": None, "params": [{"name": "goal", "value": "0"}]},
            )
        )
        for name in ("release", "reverse"):
            chosen = list(range(starts)) if name == "release" else list(range(max(1, starts // 2)))
            receipt = _data(
                box.call(
                    "submit_job_program",
                    {"program": programs[name], "starts": chosen, "record_video": False},
                )
            )
            assert receipt["accepted"], receipt
        return "waiting"

    def third(box: Any) -> str:
        results = {r["name"]: r for r in _data(box.call("job_results", {}))}
        _data(
            box.call(
                "run_check",
                {
                    "name": "goal-undone",
                    "job_id": results["chain-release"]["job_id"],
                    "params": [{"name": "goal", "value": "0"}],
                },
            )
        )
        refused = _data(
            box.call(
                "save_job",
                {
                    "name": "loaded-carry-reverse",
                    "job_id": results["chain-reverse"]["job_id"],
                    "purpose": "the other order",
                    "check": "goal-undone",
                },
            )
        )
        name = "loaded-carry-2" if installation_has(box, "jobs", "loaded-carry") else "loaded-carry"
        saved = _data(
            box.call(
                "save_job",
                {
                    "name": name,
                    "job_id": results["chain-release"]["job_id"],
                    "purpose": "the job with a release between skills",
                    "check": "goal-undone",
                },
            )
        )
        assert saved["accepted"], saved
        learned = _data(box.call("write_learned_skill", toy_learned_skill()))
        assert learned["accepted"], learned
        _data(box.call("memory_write", {"notes": f"saved {name}; refused: {refused['problems']}"}))
        return "waiting"

    def fourth(box: Any) -> str:
        state = _data(box.call("operator_state", {}))
        release = state["release"]
        assert release is not None, state
        results = {r["name"]: r for r in _data(box.call("job_results", {}))}
        account = (
            "The job chains the skills with a release and a short retreat between them. On "
            "ten development starts the naive chain delivered "
            f"{results['chain-naive']['delivered']} and the chain with the release "
            f"{results['chain-release']['delivered']}; the other order broke the process rule. "
            f"On the {release['N']} reserved release starts the job delivered "
            f"{release['delivered']} (the condition asks for {release['K']}), the naive chain "
            f"{release['naive_delivered']}. The check goal-undone finds runs that undo the "
            "first goal."
        )
        done = _data(box.call("submit_job_account", {"job": release["job"], "account": account}))
        assert done["accepted"], done
        return "done"

    return [first, second, third, fourth]


def toy_learned_skill() -> dict[str, Any]:
    """The toy agent's learned skill: a later step undoing an earlier goal."""
    return {
        "name": "job-goal-undone",
        "description": "Find job runs in which a later step undid a goal an earlier step reached.",
        "applies_when": "a chained job delivers less often than its single skills succeed",
        "parameters": [{"name": "goal", "meaning": "the requested goal reached first"}],
        "evidence": [
            {
                "claim": "runs in which goal {goal} held at some step and not at the end",
                "sql": "SELECT count(DISTINCT g.episode_id) AS undone FROM goal_steps g "
                "JOIN job_episodes e USING (episode_id) WHERE g.goal = $goal AND g.holds "
                "AND NOT e.goals_held_at_end[$goal + 1]",
                "result": "a count of runs per job; zero once the transition releases the item",
            }
        ],
        "alternatives": [
            {
                "explanation": "the second skill fails from the new state on its own",
                "how_to_tell": "goal {goal} still holds at the end of the failed runs",
            }
        ],
        "next_experiment": "run the job with a release between the skills on the same starts",
        "check_with": [{"name": "goal", "value": "0"}],
    }


def toy_agent(
    run_root: Path, state: RequestState | None = None, starts: int = 10, k: int = 16
) -> Any:
    """The scripted toy agent for `state` (read from the session's results when None)."""
    from fleetlab.loop.scripted import ScriptedAgent

    if state is None:
        raise ValueError("the toy agent needs the request")
    return ScriptedAgent(toy_steps(state, starts, k), run_root=run_root)


# The investigation.


def _pending(outbox: Path, results: Path) -> list[dict[str, Any]]:
    done = {r["job_id"] for r in read_lines(results / JOB_RESULTS_FILE)}
    return [r for r in read_lines(outbox / JOB_REQUESTS_FILE) if r["job_id"] not in done]


def _batch(
    records: Sequence[Mapping[str, Any]], state: RequestState, session: int, video: bool = False
) -> JobBatch:
    """The pending requests as the runner's batch; `video` records every episode's video
    whatever the request asked (a real run's default, so that every clip is an original)."""
    jobs = []
    for record in records:
        submission = record["request"]
        program = JobProgram.model_validate(submission["program"])
        if program.digest != record["program_digest"]:
            raise ValueError(f"{record['job_id']}: the program changed since it was queued")
        starts = [
            start_of(state.base_task, int(i), profile=state.allocation_profile)
            for i in submission["starts"]
        ]
        request = JobRequest(
            program=program, starts=starts, record_video=video or bool(submission["record_video"])
        )
        jobs.append((str(record["job_id"]), request))
    return JobBatch(session, state, tuple(jobs))


def _record_outcomes(analysis: Any, batch: JobBatch, reply: BatchReply) -> None:
    """The runner's outcomes into the results, after the checks the tool server repeats."""
    requests = {r["job_id"]: r for r in read_lines(analysis.outbox / JOB_REQUESTS_FILE)}
    wanted = [j for j, _ in batch.jobs]
    if sorted(o.job_id for o in reply.outcomes) != sorted(wanted):
        raise ValueError("the runner's outcomes differ from the batch's jobs")
    for outcome in reply.outcomes:
        if problems := job_outcome_problems(outcome, requests[outcome.job_id]):
            raise ValueError("; ".join(problems))
    for outcome in reply.outcomes:
        append_line(analysis.results / JOB_RESULTS_FILE, outcome.model_dump(mode="json"))


def _session_summary(session: Any) -> dict[str, Any]:
    return {
        "number": session.number,
        "state": session.state,
        "ok": session.ok,
        "error": session.error,
        "seconds": session.seconds,
        "tool_calls": session.tool_calls,
        "submitted": session.submitted,
        "starts": len(session.attempts),
        "lost_twice": session.lost_twice,
        "run_dir": session.run_dir,
        "report": session.report,
    }


def public_library(library: Path) -> dict[str, list[dict[str, Any]]]:
    """Job one's library as the agent reads it (`toolserver.jobs.LIBRARY_FILES`): each
    saved job's name, purpose, program, digest, goals and release counts; each check as
    written; each recipe's name and purpose. Its keys and file names stay clear of the
    truth side's names, which the launch audit refuses in mounted state."""
    from fleetlab.contracts.jobs import SavedJob

    jobs = []
    for path in sorted((library / "jobs").glob("*.json")):
        saved = SavedJob.model_validate_json(path.read_text())
        receipt = saved.receipt
        jobs.append(
            {
                "name": saved.name,
                "purpose": saved.program.purpose,
                "program_digest": saved.program.digest,
                "program": saved.program.model_dump(mode="json"),
                "check": saved.check,
                "goals": [g.text for g in saved.requested_goals],
                "release": None
                if receipt is None
                else {
                    "N": receipt.N,
                    "K": receipt.K,
                    "delivered": receipt.delivered,
                    "naive_delivered": receipt.naive_delivered,
                },
            }
        )
    checks = [
        CheckSpec.model_validate_json(path.read_text()).model_dump(mode="json")
        for path in sorted((library / "checks").glob("*.json"))
    ]
    recipes = []
    for path in sorted((library / "recipes").glob("*.json")):
        data = json.loads(path.read_text())
        recipes.append({"name": data.get("name", path.stem), "purpose": data.get("purpose", "")})
    return {"jobs": jobs, "checks": checks, "recipes": recipes}


def _mount_library(library: Path, results: Path) -> dict[str, Any]:
    """Job one's library into the analysis' results (read-only for the agent) in its public
    form; its learned skills are mounted apart (/learned)."""
    from fleetlab.toolserver.jobs import LIBRARY_FILES

    target = results / LIBRARY_DIR
    target.mkdir()
    public = public_library(library)
    for part, entries in public.items():
        (target / LIBRARY_FILES[part]).write_text(json.dumps(entries, indent=1))
    return {
        "library": str(library),
        "entries": {
            part: [e["name"] for e in entries] for part, entries in public.items() if entries
        },
    }


def freeze_library(
    analysis: Any,
    state: RequestState,
    out: Path,
    mounted: Path | None,
    label: str,
    fraction: tuple[int, int] = LEAD_FRACTION,
    carry: Sequence[Path] = (),
) -> dict[str, Any]:
    """The installation's library after the investigation: the mounted library's entries,
    the jobs and checks the agent saved, and its learned skills, each with provenance; the
    exposure lines of the mounted library and of `carry` (the imported ones)."""
    from fleetlab.contracts.learned import LearnedSkillDraft, render_learned_skill

    out.mkdir(parents=True, exist_ok=False)
    if mounted is not None:
        for part in ("jobs", "checks", "recipes", "skills"):
            if (mounted / part).is_dir():
                shutil.copytree(mounted / part, out / part)
        carry_exposure(mounted, out)
    for lines in carry:
        carry_exposure(lines, out)
    requests = {r["job_id"]: r for r in read_lines(analysis.outbox / JOB_REQUESTS_FILE)}
    done = {
        r["job_id"]
        for r in read_lines(analysis.results / JOB_RESULTS_FILE)
        if r["status"] == "done"
    }
    saved: dict[str, dict[str, Any]] = {}
    for record in read_lines(analysis.outbox / SAVED_JOBS_FILE):
        saved[record["name"]] = record
    jobs = []
    for name, record in saved.items():
        program = JobProgram.model_validate(record["program"])
        runs = sorted(
            j for j, r in requests.items() if r["program_digest"] == program.digest and j in done
        )
        save_job(
            out,
            name,
            program,
            state,
            runs,
            fraction,
            record.get("check"),
            record.get("check_digest"),
        )
        jobs.append(name)
    checks: dict[str, CheckSpec] = {}
    for record in read_lines(analysis.outbox / CHECKS_FILE):
        checks[record["check"]["name"]] = CheckSpec.model_validate(record["check"])
    for check in checks.values():
        save_check(out, check)
    skills = {}
    for record in read_lines(analysis.outbox / LEARNED_FILE):
        skills[record["draft"]["name"]] = record
    for name, record in skills.items():
        draft = LearnedSkillDraft.model_validate(record["draft"])
        target = out / "skills" / name
        target.mkdir(parents=True, exist_ok=True)
        provenance = f"{label}, session {record['session']}"
        (target / "SKILL.md").write_text(render_learned_skill(draft, provenance))
    if library() is _StandInLibrary:
        write_index(out)
    return {"root": str(out), "jobs": jobs, "checks": sorted(checks), "skills": sorted(skills)}


@dataclass
class Investigation:
    """How one investigation runs."""

    agent: Any  # agents.sessions.AgentSpec
    runner: JobRunner
    max_sessions: int | None = None  # the request's budget; a number lowers it
    label: str = "job investigation"
    library: Path | None = None  # job one's library, for job two
    scripted: Any = None  # a ScriptedAgent (fleetlab.loop.scripted): no model, no gate
    placeholder_task: bool = False
    release: bool = True
    env: Any = None  # the agents' ToolEnv; built from the job skills when None
    skills: Path | None = None  # a skill library root; the jobs role's skills if None
    skill_names: tuple[str, ...] | None = None
    # The policy's skills as operator_state lists them; None reads LIBERO's goal suite.
    policy_skills: tuple[dict[str, Any], ...] | None = None
    # The scripted lead's bar: the share of N it asks to be delivered (numerator, denominator).
    lead_fraction: tuple[int, int] = LEAD_FRACTION
    # The model time per investigation, over every call of every session.
    model_minutes: float = 90.0
    # Record every development episode's video, whatever the request asked (the release
    # always records both chains' videos).
    video_all: bool = False
    # The host's pool assignment record and the cohort this investigation runs
    # (`jobs.prior.preflight`); needed when the request names an allocation profile.
    assignment: Path | None = None
    cohort: str | None = None
    # Earlier investigations of the job line to import (`jobs.prior`, `--prior-run`).
    prior_runs: tuple[Path, ...] = ()


def goal_suite_skills() -> tuple[dict[str, Any], ...]:
    """The policy's skills (operator_state's `policy_skills`): each goal-suite task's number,
    its instruction and the goal its BDDL file states, from lane J's `runner.GoalSuite`."""
    suite = importlib.import_module("fleetlab.jobs.runner").GoalSuite()
    return tuple(
        {"task": i, "instruction": language, "goal": "; ".join(g.text for g in goals)}
        for i, (language, goals) in enumerate(zip(suite.language, suite.goals, strict=True))
    )


def toy_skills() -> tuple[dict[str, Any], ...]:
    """The toy's skills: the goal-suite tasks by name, without LIBERO."""
    return tuple(
        {"task": i, "instruction": name.replace("_", " "), "goal": None}
        for i, name in enumerate(GOAL_SUITE_TASKS)
    )


def _release(
    analysis: Any,
    state: RequestState,
    how: Investigation,
    out: Path,
    job: str,
    session: int,
    carry: Sequence[Path] = (),
) -> dict[str, Any]:
    """The release of the job saved last in `session`: a library of the jobs and checks saved
    so far, the saved program and the naive chain on the reserved release starts (the
    runner's `release`), the receipt into that library, and its counts into the results,
    where operator_state shows them in the next session."""
    from fleetlab.jobs.starts import release_starts_for
    from fleetlab.toolserver.jobs import RELEASE_FILE, public_release

    assert state.acceptance is not None
    root = out / "release"
    frozen = freeze_library(
        analysis, state, root / "library", how.library, how.label, how.lead_fraction, carry
    )
    clock = time.perf_counter()
    expected = release_starts_for(
        state.base_task, state.acceptance.N, profile=state.allocation_profile
    )
    receipt = how.runner.release(
        job, Path(frozen["root"]), state.acceptance.N, state.acceptance.K, root / "run"
    )
    if receipt.release_starts != expected:
        raise ValueError(
            f"the release of {job} ran from other starts than the request's allocation gives"
        )
    attach_receipt(Path(frozen["root"]), job, receipt)
    public = public_release(receipt, session)
    (analysis.results / RELEASE_FILE).write_text(json.dumps(public, indent=1))
    return {
        "job": job,
        "after_session": session,
        "run_dir": str(root / "run"),  # the program's and the naive chain's episodes, videos
        "library": str(root / "library"),
        "receipt": receipt.model_dump(mode="json"),
        "elapsed_seconds": time.perf_counter() - clock,  # the wait for the heavy lock included
        "condition": f"delivered on at least {state.acceptance.K} of {state.acceptance.N}",
    }


def job_skill_names(root: Path) -> tuple[str, ...]:
    """The job skills of a library: lane K's `jobs-*` directories."""
    return tuple(sorted(p.name for p in root.glob("jobs-*") if (p / "SKILL.md").is_file()))


def investigate(request_file: Path, out: Path, how: Investigation) -> dict[str, Any]:
    """One investigation of the request in `request_file`; its record is `out/job_run.json`.

    Raises FileExistsError for an existing `out`: an investigation is never resumed."""
    from unittest import mock

    from fleetlab.agents.audit import AuditContext
    from fleetlab.agents.sessions import Analysis, run_session
    from fleetlab.jobs.prior import exposure_of, preflight, read_prior
    from fleetlab.toolserver.jobs import HISTORY_FILE

    clock_started = time.perf_counter()
    out = out.expanduser().resolve()
    state = load_request(request_file)
    prior = read_prior(how.prior_runs) if how.prior_runs else None
    exposure = exposure_of([how.library] if how.library is not None else [])
    if prior is not None:
        exposure.update(prior.exposure)
    checked = preflight(
        state,
        request_file,
        exposure=exposure,
        seen=prior.identities if prior is not None else (),
        assignment=how.assignment,
        cohort=how.cohort,
    )  # raises before anything is written
    out.mkdir(parents=True, exist_ok=False)
    record_file = out / RECORD_FILE
    cap = state.budget.sessions
    if how.max_sessions is not None:
        cap = min(how.max_sessions, cap)
    record: dict[str, Any] = {
        "kind": "job-investigation",
        "label": how.label,
        "source_commit": _commit(),
        "started_at": _now(),
        "request_file": str(request_file),
        "request": describe(state),
        "agent": {k: getattr(how.agent, k) for k in ("backend", "model", "effort", "service_tier")},
        "scripted": how.scripted is not None,
        "max_sessions": cap,
        "model_minutes": how.model_minutes,
        "allocation_profile": state.allocation_profile,
        "preflight": checked,
        "prior": None,
        "library_mounted": None,
        "sessions": [],
        "batches": [],
        "outcome": "running",
    }

    def save() -> None:
        record_file.write_text(json.dumps(record, indent=1, default=str) + "\n")

    save()
    try:
        imported = prior.write(out / "prior") if prior is not None else None
    except ValueError as refused:  # a source changed while it was imported
        record.update(outcome="failed", ended_because=f"the import was refused: {refused}")
        save()
        raise
    # Every exposure line the investigation carries, its mounted library's and its
    # predecessors', kept in its own directory before anything runs: an investigation
    # that imports this one sees them even if this one is broken off before it freezes a
    # library (`jobs.prior.CARRIED_EXPOSURE`).
    carried: list[Path] = []
    if imported is not None or how.library is not None:
        from fleetlab.jobs.prior import CARRIED_EXPOSURE

        held = out / CARRIED_EXPOSURE
        held.mkdir(parents=True, exist_ok=True)
        if how.library is not None:
            carry_exposure(how.library, held)
        carried = [held]
        exposure_file = held / EXPOSURE_FILE
        record["exposure_carried"] = {
            "file": str(exposure_file),
            "lines": len(exposure_file.read_text().splitlines()) if exposure_file.is_file() else 0,
        }
        save()
    if imported is not None:
        record["prior"] = {
            "sources": [str(Path(p).expanduser().resolve()) for p in how.prior_runs],
            "registry": str(imported["registry"]),
            "registry_sha256": imported["registry_sha256"],
            "history": str(out / "prior" / HISTORY_FILE),
            "jobs": imported["jobs"],
            "episodes": imported["episodes"],
        }
        save()
    install = installation(out / "installation")
    learned = None
    if how.library is not None:
        skills_dir = how.library / "skills"
        learned = skills_dir if skills_dir.is_dir() and any(skills_dir.iterdir()) else None
    analysis = Analysis.create(INSTALLATION_ID, how.label, base=out / "analyses", learned=learned)
    record["analysis"] = str(analysis.root)
    if how.library is not None:
        record["library_mounted"] = {
            **_mount_library(how.library, analysis.results),
            "learned": analysis.meta.get("learned"),
        }
    history: dict[str, int] | None = None
    if imported is not None:
        # The public history, the host's own copy byte for byte (its digest is in the registry).
        shutil.copyfile(out / "prior" / HISTORY_FILE, analysis.results / HISTORY_FILE)
        history = {
            "investigations": len(imported["public"].investigations),
            "jobs": imported["jobs"],
            "episodes": imported["episodes"],
        }
    skills = how.policy_skills if how.policy_skills is not None else goal_suite_skills()
    record["policy_skills"] = list(skills)
    record["lead_fraction"] = f"{how.lead_fraction[0]}/{how.lead_fraction[1]}"
    lead = ShiftLead(fraction=how.lead_fraction)
    publish(analysis.results, state, skills, how.lead_fraction)
    task_text = task_text_for(how.placeholder_task)
    env = how.env
    if env is None:
        from fleetlab.agents.toolenv import build_tool_env

        if how.skills is None and how.skill_names is None:
            env = build_tool_env(role="jobs")  # lane K's skills (fleetlab.skill_library)
        else:
            root = how.skills or REPO / "skills"
            env = build_tool_env(skills=root, skill_names=how.skill_names or job_skill_names(root))
    record["tool_env"] = env.digest()
    # The imported history first, then every batch's run as it comes back.
    runs: list[Path] = [imported["tree"]] if imported is not None and imported["tree"] else []
    export = job_export(install, runs, out / "exports" / "session-01")
    audit = AuditContext((install.run,), install.fleet)
    patch = (
        mock.patch("fleetlab.agents.sessions.run_agent", how.scripted)
        if how.scripted is not None
        else contextlib.nullcontext()
    )
    episodes = 0
    ended_because = f"no account in {cap} sessions"
    outcome = "incomplete"
    try:
        with patch:
            for number in range(1, cap + 1):
                used = _model_seconds(analysis)
                left = how.model_minutes * 60 - used
                if left <= 0:
                    ended_because = (
                        f"the investigation's {how.model_minutes:g} model minutes are used "
                        f"({used:.0f} s over {number - 1} sessions); it is incomplete"
                    )
                    break
                agent = dataclasses.replace(how.agent, timeout=min(float(how.agent.timeout), left))
                task = task_text(state, number, how.library is not None, history)
                if how.scripted is None and number == 1:
                    _gate(env, how.agent.backend, export, install, task, state, out)
                session = run_session(
                    analysis,
                    export,
                    agent,
                    SYSTEM,
                    task,
                    env,
                    tools=JOB_SESSION_TOOLS,
                    run_root=out / "agent-runs",
                    audit=audit,
                )
                state, problems = sync_session(state, analysis.outbox, session.number, lead)
                publish(analysis.results, state, skills, how.lead_fraction)
                summary = _session_summary(session)
                queued = [
                    r["job_id"]
                    for r in read_lines(analysis.outbox / JOB_REQUESTS_FILE)
                    if int(r["session"]) == session.number
                ]
                saved = [
                    r
                    for r in read_lines(analysis.outbox / SAVED_JOBS_FILE)
                    if int(r["session"]) == session.number
                ]
                # The runtime's state knows experiments and probes only; a session that
                # queued jobs or saved one for release is waiting for them.
                summary.update(
                    runtime_state=session.state,
                    state="waiting" if queued or saved else session.state,
                    queued=queued,
                    task=task,
                    export=str(export),
                    sync_problems=problems,
                )
                record["sessions"].append(summary)
                record["costs"] = _costs(analysis, how.agent.model, record["batches"])
                save()
                if session.error in STOPPING_ERRORS or session.lost_twice:
                    ended_because = (
                        f"session {session.number}: {session.error or 'lost twice'}; the "
                        "investigation is incomplete and is not resumed"
                    )
                    break
                if not session.ok and (queued or saved):
                    # A call lost after the agent's work was accepted: the work and its
                    # charges stand (the runtime keeps them), the session did not finish.
                    ended_because = (
                        f"session {session.number}: {session.error} after accepted work "
                        f"({', '.join(queued) or 'a save'}); the investigation is incomplete "
                        "and is not resumed"
                    )
                    break
                pending = _pending(analysis.outbox, analysis.results)
                if pending:
                    batch = _batch(pending, state, session.number, how.video_all)
                    where = out / "runs" / f"session-{session.number:02d}"
                    # Entered before the runner starts, so that an interruption while the
                    # batch waits for the heavy queue or runs leaves it in the record.
                    entry: dict[str, Any] = {
                        "session": session.number,
                        "status": "running",
                        "jobs": [
                            {
                                "job_id": j,
                                "program": r.program.name,
                                "starts": len(r.starts),
                                "program_digest": r.program.digest,
                                "record_video": r.record_video,
                            }
                            for j, r in batch.jobs
                        ],
                        "starts_planned": sum(len(r.starts) for _, r in batch.jobs),
                        "batch_dir": str(where),
                    }
                    record["batches"].append(entry)
                    save()
                    clock = time.perf_counter()
                    if state.acceptance is None:
                        # The host holds no approved condition: nothing is tested, free.
                        reply = BatchReply(
                            [
                                _outcome(j, "refused", problems=[NO_ACCEPTANCE])
                                for j, _ in batch.jobs
                            ],
                            None,
                            {"refused": NO_ACCEPTANCE},
                        )
                    else:
                        reply = how.runner.run(batch, where)
                    entry["elapsed_seconds"] = time.perf_counter() - clock
                    _record_outcomes(analysis, batch, reply)
                    if reply.run_dir is not None:
                        runs.append(reply.run_dir)
                    simulated = sum(len(o.episode_ids) for o in reply.outcomes)
                    episodes += simulated
                    charged = sum(
                        len(r.starts)
                        for (_, r), o in zip(batch.jobs, reply.outcomes, strict=True)
                        if o.status != "refused"
                    )
                    entry.update(
                        status="failed" if reply.record.get("failed") else "done",
                        outcomes=[o.model_dump(mode="json") for o in reply.outcomes],
                        starts_charged=charged,
                        # Episodes that ran to their end and are shown; a broken-off run
                        # shows none, so its starts are charged and not simulated.
                        starts_simulated=simulated,
                        starts_unfinished=charged - simulated,
                        **reply.record,
                    )
                    record["costs"] = _costs(analysis, how.agent.model, record["batches"])
                    export = job_export(
                        install, runs, out / "exports" / f"session-{session.number + 1:02d}"
                    )
                    save()
                    continue
                if saved and how.release and "release" not in record and state.acceptance:
                    # Entered before the release runs, so that an interruption leaves it in
                    # the record (the library's exposure log has it from the runner too).
                    record["release_attempt"] = {
                        "job": str(saved[-1]["name"]),
                        "after_session": session.number,
                        "N": state.acceptance.N,
                        "base_task": state.base_task,
                        "allocation_profile": state.allocation_profile,
                    }
                    save()
                    record["release"] = _release(
                        analysis, state, how, out, str(saved[-1]["name"]), session.number, carried
                    )
                    save()
                    continue
                if account_follows(account_of(analysis.outbox), session.number, record):
                    outcome, ended_because = "done", f"session {session.number}: the account"
                    break
    except BaseException as error:
        # An interruption (a signal, Ctrl-C, the shell's limit) leaves the investigation
        # incomplete, as a lost call does; any other error is a failure. Either way the
        # record keeps the costs so far, and nothing resumes it.
        stopped = isinstance(error, (KeyboardInterrupt, SystemExit, Interrupted))
        for entry in record["batches"]:
            if entry.get("status") == "running":
                entry["status"] = "interrupted"
        record.update(
            outcome="incomplete" if stopped else "failed",
            ended_because=f"interrupted ({type(error).__name__}: {error}); not resumed"
            if stopped
            else f"failed ({type(error).__name__}: {error})",
            error=f"{type(error).__name__}: {error}",
            ended_at=_now(),
            wall_seconds=time.perf_counter() - clock_started,
        )
        with contextlib.suppress(Exception):
            record["costs"] = _costs(analysis, how.agent.model, record["batches"])
        save()
        raise
    record.update(outcome=outcome, ended_because=ended_because)
    record["acceptance"] = state.acceptance.model_dump() if state.acceptance else None
    record["message_log"] = [m.model_dump(mode="json") for m in state.message_log]
    record["account"] = account_of(analysis.outbox)
    record["costs"] = _costs(analysis, how.agent.model, record["batches"])
    save()
    if outcome == "done":
        record["library"] = freeze_library(
            analysis, state, out / "library", how.library, how.label, how.lead_fraction, carried
        )
    if "release" in record:
        if outcome == "done":
            receipt = ReleaseReceipt.model_validate(record["release"]["receipt"])
            attach_receipt(out / "library", record["release"]["job"], receipt)
            carry_exposure(out / "release" / "library", out / "library")
        record["costs"]["release_starts"] = 2 * int(record["release"]["receipt"]["N"])
    record["released"] = bool(record.get("release", {}).get("receipt", {}).get("released"))
    record["run_dirs"] = run_dirs(record)
    record["ended_at"] = _now()
    record["wall_seconds"] = time.perf_counter() - clock_started
    save()
    return record


def _gate(
    env: Any,
    backend: str,
    export: Path,
    install: Installation,
    task: str,
    state: RequestState,
    out: Path,
) -> None:
    """Before the first model call: the CLI gate where stale, the source audit of this
    export, the prompts and the job tools' texts with the request's budget, the login."""
    from fleetlab.agents.batch import Batch

    texts = {"system": SYSTEM, "task": task, **job_tool_texts(state.budget)}
    backends: list[Any] = [backend]
    batch = Batch.after_gate(
        env, backends, export, [install.run], fleet=install.fleet, texts=texts, rerun=False
    )
    (out / "gate.json").write_text(json.dumps(batch.records, indent=1, default=str) + "\n")


def account_follows(
    account: Mapping[str, Any] | None, session: int, record: Mapping[str, Any]
) -> bool:
    """Whether `account` ends the investigation after `session`: given in that session and,
    when a release ran, after it and naming the released job; otherwise naming none."""
    if account is None or int(account["session"]) != session:
        return False
    job = account["answer"].get("job")
    release = record.get("release")
    if release is None:
        return job is None
    return bool(job == release["job"] and session > int(release["after_session"]))


class Interrupted(BaseException):
    """SIGTERM or SIGHUP to the driver (`main`): the investigation stops incomplete."""


def _on_signal(signum: int, frame: Any) -> None:
    raise Interrupted(f"signal {signum}")


def run_dirs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Where the runs of an investigation are, for the film: each development batch's run
    directory and the release run's."""
    release = record.get("release") or {}
    return {
        "development": [
            {"session": b["session"], "run_dir": b.get("run_dir")} for b in record["batches"]
        ],
        "release": release.get("run_dir"),
    }


def _sum_known(values: Any) -> float | None:
    known = [float(v) for v in values if v is not None]
    return sum(known) if known else None


def _model_seconds(analysis: Any) -> float:
    """The model time of every call of every session so far."""
    return sum(float(a.get("seconds") or 0) for s in analysis.session_records() for a in s.attempts)


def _costs(analysis: Any, model: str, batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every cost of the investigation: model starts, wall and model time, tokens and tool
    calls summed over every start (`loop.costs.session_costs`), and the starts simulated."""
    from fleetlab.loop.costs import session_costs

    sessions = analysis.session_records()
    return {
        **session_costs(sessions, model),
        "batches": len(batches),
        "starts_planned": sum(int(b["starts_planned"]) for b in batches),
        "starts_charged": sum(int(b["starts_charged"]) for b in batches),
        "starts_simulated": sum(int(b.get("starts_simulated") or 0) for b in batches),
        "starts_unfinished": sum(int(b.get("starts_unfinished") or 0) for b in batches),
        # Queue-inclusive: from the batch's hand-over to the runner's reply, the wait for the
        # heavy lock and the policy's load included.
        "batch_elapsed_seconds": sum(float(b.get("elapsed_seconds") or 0) for b in batches),
        # The runner's own reports (lane J's runner log); null when a batch has none.
        "runner_seconds": _sum_known(b.get("runner_seconds") for b in batches),
        "episode_seconds": _sum_known(
            sum(b["episode_seconds"]) if b.get("episode_seconds") is not None else None
            for b in batches
        ),
    }


# The command line.


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.jobs.driver")
    parser.add_argument("--request", required=True, type=Path, help="the RequestState JSON")
    parser.add_argument("--out", required=True, type=Path, help="a new directory")
    parser.add_argument("--library", type=Path, help="job one's library, for job two")
    parser.add_argument("--backend", default=DEFAULT_AGENT.backend, choices=("codex", "claude"))
    parser.add_argument("--model", default=DEFAULT_AGENT.model)
    parser.add_argument("--effort", default=DEFAULT_AGENT.effort)
    parser.add_argument("--service-tier", default=DEFAULT_AGENT.service_tier)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per session")
    parser.add_argument(
        "--max-sessions", type=int, help="fewer sessions than the request's budget allows"
    )
    parser.add_argument("--label", default="job investigation")
    parser.add_argument("--runner", choices=("command", "toy"), default="command")
    parser.add_argument(
        "--policy",
        choices=("xvla", "hold"),
        default="xvla",
        help="the command runner's policy: X-VLA (heavy) or lane J's holding policy (light)",
    )
    parser.add_argument("--scripted", choices=("toy",), help="a scripted agent, no model")
    parser.add_argument("--toy-starts", type=int, default=10, help="the scripted agent's starts")
    parser.add_argument(
        "--placeholder-task",
        action="store_true",
        help="the stand-in task text, until lane K's lands",
    )
    parser.add_argument("--no-release", action="store_true")
    parser.add_argument("--skills", type=Path, help="a skill library root")
    parser.add_argument("--skill", action="append", default=None, help="a skill to mount")
    parser.add_argument("--queue-log", type=Path, help="the lanes' shared heavy-queue log")
    parser.add_argument(
        "--lead-fraction",
        type=lead_fraction,
        default=LEAD_FRACTION,
        help="the share of N the scripted lead asks to be delivered, as A/B (default 4/5)",
    )
    parser.add_argument(
        "--model-minutes",
        type=float,
        default=90.0,
        help="the investigation's model time over every call (default 90)",
    )
    parser.add_argument(
        "--video-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record every development episode's video, whatever the request asked",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        help="the host's pool assignment record (fleetlab.coordinator-cohorts/1)",
    )
    parser.add_argument("--cohort", help="the cohort of the assignment record this run is")
    parser.add_argument(
        "--prior-run",
        type=Path,
        action="append",
        default=[],
        help="an earlier investigation of the job line to import (repeatable, in order)",
    )
    return parser.parse_args(argv)


def lead_fraction(text: str) -> tuple[int, int]:
    """`--lead-fraction`: A/B with 0 < A <= B, for example 4/5 or 2/3."""
    try:
        a, b = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: give A/B, for example 2/3") from None
    if not 0 < a <= b:
        raise argparse.ArgumentTypeError(f"{text!r}: 0 < A <= B")
    return a, b


def main(argv: Sequence[str] | None = None) -> int:
    from fleetlab.agents.sessions import AgentSpec

    args = parse(argv)
    agent = AgentSpec(
        args.backend,
        args.model,
        effort=args.effort,
        service_tier=args.service_tier,
        timeout=args.timeout,
    )
    state = load_request(args.request)
    runner: JobRunner
    if args.runner == "toy":
        runner = ToyRunner(state)
    elif args.policy == "hold":
        runner = CommandRunner.holding(queue_log=args.queue_log)
    else:
        runner = CommandRunner(queue_log=args.queue_log)
    scripted = None
    if args.scripted == "toy":
        scripted = toy_agent(args.out.resolve() / "scripted-agent", state, args.toy_starts)
    how = Investigation(
        agent=agent,
        runner=runner,
        max_sessions=args.max_sessions,
        label=args.label,
        library=args.library.resolve() if args.library else None,
        scripted=scripted,
        placeholder_task=args.placeholder_task,
        release=not args.no_release,
        skills=args.skills,
        skill_names=tuple(args.skill) if args.skill else None,
        policy_skills=toy_skills() if args.runner == "toy" else None,
        lead_fraction=args.lead_fraction,
        model_minutes=args.model_minutes,
        video_all=args.video_all,
        assignment=args.assignment.resolve() if args.assignment else None,
        cohort=args.cohort,
        prior_runs=tuple(p.resolve() for p in args.prior_run),
    )
    import signal

    from fleetlab.jobs.prior import ImportRefused, PreflightError

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _on_signal)
    try:
        record = investigate(args.request.resolve(), args.out, how)
    except (PreflightError, ImportRefused) as refused:
        print(json.dumps({"refused": refused.problems}, indent=1), file=sys.stderr)
        return 2
    receipt = (record.get("release") or {}).get("receipt") or {}
    summary = {
        "outcome": record["outcome"],
        "ended_because": record.get("ended_because"),
        "sessions": len(record["sessions"]),
        "starts_charged": record.get("costs", {}).get("starts_charged"),
        "release": {k: receipt.get(k) for k in ("job_name", "N", "K", "delivered",
                                                "naive_delivered", "passed", "released")}
        if receipt
        else None,
        "record": str(args.out.resolve() / RECORD_FILE),
    }  # fmt: skip
    print(json.dumps(summary, indent=1, default=str))
    return 0 if record["outcome"] == "done" else 3


if __name__ == "__main__":
    sys.exit(main())
