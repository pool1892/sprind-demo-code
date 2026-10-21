"""The miniature fleet end to end: world, run, oracle, export,
analysis, fresh replays, score, fixtures. Each step is resumable and reads what the
steps before it wrote under one root directory.

Layout under the root (`ChainPaths`):

```
world/                  the world files (contracts section 9) and the truth-side files
runs/w10-observed/      the observed window as run (days 0 and 1); its export is what the agent sees
runs/w10-later/         the experiment and scoring windows as run, plus the oracle's replays
runs/w10-fresh/         replays the scorer needs beyond the oracle (added changes), if any
export/                 export_runs of the observed run: log/ and data_dictionary.md
analyses/<id>/          the analysis: memory, outbox (diagnosis, fix set), sessions, queues
agent-runs/             the CLI calls of the analysis (argv, stream, box/toolbox/tool_calls.jsonl)
oracle/                 fleetlab.oracle's report: oracle.json, summary.md, episodes.parquet
score/                  report.json and report.md
steps/<step>.json       per step: when it started and ended, and for a heavy step its queue time
```

Heavy steps (the runs) are started as their own process under `tools/heavy-run`, so that
the heavy lock is held only while the policy is loaded; the others run in this process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import Job, RunLayout
from fleetlab.contracts.tools import TOOLS
from fleetlab.contracts.world import verify_world
from fleetlab.worlds.generate import World, generate, write_truth_files, write_world
from fleetlab.worlds.mini import VERSION

REPO = Path(__file__).resolve().parents[3]
HEAVY_RUN = REPO / "tools" / "heavy-run"
MIOPEN_DIR = Path.home() / ".cache" / "robots-probes" / "engine" / "miopen"
PRECISION = "fp16_res32"
BATCH_SIZE = 4

STEPS = (
    "world",
    "run",
    "oracle",
    "export",
    "analysis-logs",
    "analysis-experiments",
    "fresh",
    "score",
    "fixtures",
)
LOGS_ONLY_TOOLS = tuple(
    t.name for t in TOOLS if t.name not in ("request_experiment", "experiment_results")
)


@dataclass(frozen=True)
class ChainPaths:
    root: Path

    @property
    def world(self) -> Path:
        return self.root / "world"

    def run(self, name: str) -> Path:
        return self.root / "runs" / f"w10-{name}"

    @property
    def observed(self) -> Path:
        return self.run("observed")

    @property
    def later(self) -> Path:
        return self.run("later")

    @property
    def fresh(self) -> Path:
        return self.run("fresh")

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def analyses(self) -> Path:
        return self.root / "analyses"

    @property
    def agent_runs(self) -> Path:
        return self.root / "agent-runs"

    @property
    def oracle(self) -> Path:
        return self.root / "oracle"

    @property
    def score(self) -> Path:
        return self.root / "score"

    def step_file(self, step: str) -> Path:
        return self.root / "steps" / f"{step}.json"

    def run_dirs(self) -> list[Path]:
        return [p for p in (self.observed, self.later, self.fresh) if (p / "run.json").is_file()]


# ---- bookkeeping ------------------------------------------------------------------------------


def note_step(paths: ChainPaths, step: str, **fields: Any) -> None:
    """Merge fields into the step's record (`steps/<step>.json`)."""
    path = paths.step_file(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n")


def software() -> Any:
    """The software the jobs are made for: the pinned database for batch 4 in fp16_res32."""
    os.environ.setdefault("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR))
    from fleetlab.engine import miopen
    from fleetlab.engine import software as soft

    return soft.current(miopen.require(BATCH_SIZE, PRECISION))


# ---- world ------------------------------------------------------------------------------------


def step_world(paths: ChainPaths) -> World:
    """Generate the world; write it the first time, else check that it regenerates to its hash."""
    from fleetlab.worlds.mini import mini_spec

    world = generate(mini_spec(), software())
    base_runs = [str(paths.observed)]
    if not (paths.world / "hash").is_file():
        value = write_world(world, paths.world, base_runs)
        note_step(paths, "world", hash=value, written=time.time())
        return world
    stored = (paths.world / "hash").read_text().strip()
    scratch = paths.root / ".world-check"
    shutil.rmtree(scratch, ignore_errors=True)
    value = write_world(world, scratch, base_runs)
    shutil.rmtree(scratch, ignore_errors=True)
    if value != stored or not verify_world(paths.world):
        raise RuntimeError(
            f"{paths.world}: the world regenerates to {value}, not to the stored {stored} "
            "(the software or the generator changed); use a new chain directory"
        )
    write_truth_files(world, paths.world, base_runs)  # derived; refreshed with the code
    return world


# ---- the runs (heavy) ----------------------------------------------------------------------


def run_plans(world: World) -> dict[str, list[Job]]:
    """The jobs of each run directory: observed; experiment and scoring with the oracle."""
    from fleetlab.integration.oracle import oracle_jobs

    later = world.jobs_in("experiment") + world.jobs_in("scoring") + oracle_jobs(world).jobs
    return {"observed": world.jobs_in("observed"), "later": later}


RUN_WINDOWS: dict[str, list[str]] = {"observed": ["observed"], "later": ["experiment", "scoring"]}


def run_complete(run_dir: Path, jobs: Sequence[Job]) -> bool:
    """Every job done and merged, and the chain sample held."""
    from fleetlab.integration.oracle import run_record
    from fleetlab.log import run_status

    if not (run_dir / "run.json").is_file():
        return False
    layout = RunLayout(run_dir)
    done = all(layout.is_done(j) for j in jobs)
    status = run_status(run_dir)
    # Every job merged into the run's tables; run-level tables a newer log writes (the
    # fleet's records) are the records step's, so a missing one does not rerun the jobs.
    merged = (
        not status.partial and status.staging_files == 0 and set(status.done) == set(status.merged)
    )
    return done and merged and run_record(run_dir).chain_ok is True


def _records_for(world: World, windows: Sequence[str]) -> tuple[Any, list[dict[str, Any]]]:
    fleet = world.observed_fleet if list(windows) == ["observed"] else world.fleet
    rows = [r for w in windows for r in world.health_in(w)]  # type: ignore[arg-type]
    rows.sort(key=lambda r: (r["day"], r["robot_id"], r["component"], r["metric"]))
    return fleet, rows


def records_current(run_dir: Path, world: World, windows: Sequence[str]) -> bool:
    """Whether the run's description and health readings are the world's as generated now."""
    import pyarrow.parquet as pq

    from fleetlab.contracts.model import describe_for_agents
    from fleetlab.log import run_status

    description = run_dir / "log" / "description.json"
    health = run_dir / "log" / "health.parquet"
    if not (description.is_file() and health.is_file()) or run_status(run_dir).missing:
        return False
    fleet, rows = _records_for(world, windows)
    want = json.dumps(describe_for_agents(fleet, True, world.task_classes), indent=1) + "\n"
    have = pq.read_table(health).to_pylist()
    key = ("day", "robot_id", "component", "metric", "value")
    same_rows = [tuple(r[k] for k in key) for r in have] == [tuple(r[k] for k in key) for r in rows]
    return description.read_text() == want and same_rows


def write_records(run_dir: Path, world: World, windows: Sequence[str]) -> None:
    """The fleet's records into a finished run: description, change and maintenance logs,
    and the health readings of the run's days (the engine writes none of them)."""
    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.log import RunWriter

    record = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    fleet, rows = _records_for(world, windows)
    with RunWriter(
        run_dir, record, episodes_per_robot_day=world.spec.episodes_per_robot_day
    ) as writer:
        writer.write_model(fleet, world.task_classes)
        writer.write_table(
            "health",
            sorted(rows, key=lambda r: (r["day"], r["robot_id"], r["component"], r["metric"])),
        )
        writer.finalize()


def run_jobs(run_dir: Path, jobs: Sequence[Job], world: World, workers: int) -> dict[str, Any]:
    """One engine run on a resumable directory; returns the run's record fields of interest."""
    from fleetlab.engine.fleet import RunConfig, run

    config = RunConfig(
        out=str(run_dir),
        batch_size=BATCH_SIZE,
        workers=workers,
        precision=PRECISION,
        episodes_per_robot_day=world.spec.episodes_per_robot_day,
    )
    started = time.time()
    result = run(config, jobs)
    return {
        "jobs": len(jobs),
        "seconds": round(time.time() - started, 1),
        "chain_ok": result.record.chain_ok,
        "throughput": result.record.extra.get("throughput"),
    }


def step_run(paths: ChainPaths, world: World, workers: int = 10) -> None:
    """The heavy part, in this process: both runs, then the fleet's records into each."""
    note_step(paths, "run", started=time.time())
    for name, jobs in run_plans(world).items():
        run_dir = paths.run(name)
        if run_complete(run_dir, jobs):
            continue
        info = run_jobs(run_dir, jobs, world, workers)
        write_records(run_dir, world, RUN_WINDOWS[name])
        note_step(paths, "run", **{name: info})
    note_step(paths, "run", ended=time.time())


def heavy(
    paths: ChainPaths, step: str, extra: Sequence[str] = (), record_as: str | None = None
) -> None:
    """Start one step as its own process under tools/heavy-run and wait for it.

    The queue time is the time from the call to the step's own start. The step's
    record starts afresh on every call; `record_as` keeps a copy under another name
    (the experiment queue runs once per waiting session).
    """
    invoked = time.time()
    paths.step_file(step).parent.mkdir(parents=True, exist_ok=True)
    paths.step_file(step).write_text(json.dumps({"invoked": invoked}) + "\n")
    command = [
        str(HEAVY_RUN),
        "uv",
        "run",
        "--locked",
        "python",
        "-m",
        "fleetlab.integration",
        "--dir",
        str(paths.root),
        "--only",
        step,
        *extra,
    ]
    env = {
        **os.environ,
        "FLEETLAB_MIOPEN_DIR": os.environ.get("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR)),
    }
    status = subprocess.run(command, cwd=REPO, env=env, check=False).returncode
    record = json.loads(paths.step_file(step).read_text())
    started = record.get("started")
    note_step(
        paths,
        step,
        exit_status=status,
        queue_s=round(started - invoked, 1) if started else None,
        returned=time.time(),
    )
    if record_as is not None:
        paths.step_file(record_as).write_text(paths.step_file(step).read_text())
    if status != 0:
        raise RuntimeError(f"heavy step {step} ended with exit status {status}")


# ---- oracle -----------------------------------------------------------------------------------


def step_oracle(paths: ChainPaths, world: World) -> dict[str, Any]:
    """The oracle's report over the later run (and fresh replays), through `fleetlab.oracle`:
    `oracle/oracle.json`, `summary.md`, `episodes.parquet`, and `truth/oracle.parquet` in the
    later run. The chain's root is a world root: `world/`, `runs/`, `oracle/`."""
    from fleetlab.oracle.steps import Oracle

    started = time.time()
    oracle = Oracle.open(paths.root, proof=False)
    report = oracle.step_report(parquet_into=paths.later)
    note_step(paths, "oracle", started=started, ended=time.time())
    return report.model_dump(mode="json")


# ---- export -----------------------------------------------------------------------------------


def step_export(paths: ChainPaths, world: World) -> Path:
    """The agent-visible export of the observed run, with the leak audit; once."""
    from fleetlab.log import ExportRefused, export_runs
    from fleetlab.log.leaks import write_export_check

    note_step(paths, "export", leaks_clean=False)
    if (paths.export / "data_dictionary.md").is_file():
        checked = write_export_check(
            paths.export,
            [paths.observed],
            fleet=world.fleet,
            receipt_dir=paths.root / "export-checks",
        )
        if not checked["clean"]:
            raise ExportRefused(
                [
                    *checked["problems"],
                    *(f"leak audit: {h['where']}: {h['term']!r}" for h in checked["hits"]),
                ]
            )
        note_step(paths, "export", leaks_clean=True, receipt=checked["receipt"])
        return paths.export
    started = time.time()
    report = export_runs(
        [paths.observed],
        paths.export,
        fleet=world.observed_fleet,
        audit_fleet=world.fleet,
        task_classes=world.task_classes,
    )
    note_step(
        paths,
        "export",
        started=started,
        ended=time.time(),
        episodes=report.episodes,
        step_rows=report.step_rows,
        leaks_clean=None if report.leaks is None else report.leaks.clean,
    )
    return paths.export


# ---- analysis ---------------------------------------------------------------------------------

SYSTEM = (
    "You are the analysis agent for a fleet of robot arms that pick up objects and put them "
    "down at a target. You work only through the tools you are given; you have no shell. "
    "Compute every number with the query tool (the skills give the SQL); do not compute in "
    "your head, and report only numbers a tool returned to you. The skills, listed by "
    "list_skills and read with read_skill, are your procedures; follow them."
)

TASK = """Analyse this fleet and say why its attempts fail and what to do about it.

The fleet's records are in the log and in the system description, which describe returns
one part at a time (dictionary, fleet, vocabulary, costs): sites, robots, components,
configuration, change log, maintenance log, health readings and the costs of actions in
operator hours. The catalogue of actions, with their costs, is the table catalogue; query
it like the other tables. The log holds the observed
window, days 0 and 1. After it come an experiment window (day 2) and a scoring window
(day 3), in which the same robots do the same kind of work, 13 episodes per robot and day
as in the observed days. {experiments}

1. Follow the skills in their order: triage, episode-cards, failure-modes, scope,
   sensitivity-from-logs, {experiment_skill}rank-actions, write-diagnosis. Write your
   findings to memory as the skills say.
2. Submit your diagnosis with submit_diagnosis: the causes you find, each with its
   target (a scope and the component, configuration key or condition), its direction,
   your confidence, the evidence and the failure mode; the mode summaries; warnings and
   retrain indicators if you find any.
3. Submit your ranked fix set with submit_fix_set: at least ten actions of the catalogue,
   in order of expected gain in points of fleet success over the scoring window, each with
   a low and a high value and a rationale. The list will be cut at three and six actions
   and at four and eight operator hours.

Finish when both are submitted."""

LOGS_ONLY = "No experiments can be run in this analysis: work from the log alone."
WITH_EXPERIMENTS = (
    "You may request A/B experiments in the experiment window (request_experiment); they run "
    "after this session ends and you see their results in the next session."
)


CONDITIONS = ("logs", "experiments")


@dataclass
class HeavyQueueEngine:
    """The runtime's `Engine` for the miniature fleet: the engine's experiment queue
    (`fleetlab.engine.experiments.run_queue`) in its own process under
    `tools/heavy-run`, so that the analysis holds the heavy lock only while arms run."""

    paths: ChainPaths
    workers: int = 10

    def run_queue(self, queue_file: Path, queue: Any) -> Any:
        from fleetlab.agents.sessions import EngineReply

        step = f"queue-{queue.analysis_id}-{queue.session:02d}"
        heavy(
            self.paths,
            "queue-heavy",
            ["--queue", str(queue_file), "--workers", str(self.workers)],
            record_as=step,
        )
        return EngineReply.model_validate_json(_reply_file(queue_file).read_text())


def _reply_file(queue_file: Path) -> Path:
    return queue_file.with_name(queue_file.stem + ".reply.json")


def run_queue_here(paths: ChainPaths, queue_file: Path, workers: int) -> None:
    """The heavy part of `HeavyQueueEngine.run_queue`, in its own process."""
    from fleetlab.engine.experiments import WorldTruth, run_queue
    from fleetlab.engine.fleet import RunConfig

    world = WorldTruth.load(paths.world / "world_truth.json")
    config = RunConfig(out="", batch_size=BATCH_SIZE, workers=workers, precision=PRECISION)
    reply = run_queue(queue_file, world, paths.world, config=config)
    _reply_file(queue_file).write_text(reply.model_dump_json(indent=1) + "\n")


MAX_SESSIONS = 4


def analysis_dir(paths: ChainPaths, condition: str = "logs") -> Path | None:
    """The analysis directory of a condition, once its analysis has been created."""
    record = paths.step_file(f"analysis-{condition}")
    if record.is_file():
        root = json.loads(record.read_text()).get("analysis_dir")
        if root:
            return Path(root)
    # An analysis created by the runtime before its first session ended carries the label.
    if paths.analyses.is_dir():
        for meta in sorted(paths.analyses.glob("*/analysis.json")):
            if json.loads(meta.read_text()).get("label", "").endswith(f" {condition}"):
                return meta.parent
    return None


def analysis_state(root: Path) -> str:
    """ "finished" (an answer, or the last session ended without queuing, or no sessions
    left), "waiting" (the last session queued experiments), or "new" (no session yet)."""
    if (root / "outbox" / "fix_set.json").is_file():
        return "finished"
    sessions = sorted((root / "sessions").glob("session-*.json"))
    if not sessions:
        return "new"
    last = json.loads(sessions[-1].read_text()).get("state")
    if last == "waiting" and len(sessions) < MAX_SESSIONS:
        return "waiting"
    return "finished"


def analyses(paths: ChainPaths) -> dict[str, Path]:
    """Every finished analysis by condition; one without an answer is scored as missing."""
    out: dict[str, Path] = {}
    for condition in CONDITIONS:
        root = analysis_dir(paths, condition)
        if root is not None and analysis_state(root) == "finished":
            out[condition] = root
    return out


def _session_rows(roots: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((roots / "sessions").glob("session-*.json")):
        s = json.loads(path.read_text())
        rows.append(
            {
                key: s.get(key)
                for key in (
                    "number",
                    "state",
                    "seconds",
                    "tool_calls",
                    "error",
                    "usage",
                    "run_dir",
                    "queued",
                )
            }
        )
    return rows


def _resume(root: Path, engine: HeavyQueueEngine, run: Any, check_arms: bool = True) -> Any:
    """Answer the last session's queue if it has no answer yet, then carry on with the
    sessions left (`run`: the analysis' `run_analysis` with everything but the analysis
    and the export bound)."""
    from fleetlab.agents.sessions import Analysis, EngineError, ExperimentQueue, arm_problems

    analysis = Analysis(root)
    number = analysis.sessions_done()
    queue_file = root / "queues" / f"session-{number:02d}.json"
    queue = ExperimentQueue.model_validate_json(queue_file.read_text())
    reply = engine.run_queue(queue_file, queue)
    if check_arms:
        problems = arm_problems(reply)
        if problems:
            raise EngineError("; ".join(problems))
    done = analysis.results_by_id()
    analysis.record_results([r for r in reply.results if r.experiment_id not in done])
    return run(analysis, reply.export)


def step_analysis(
    paths: ChainPaths, world: World, condition: str, model: str = "opus", workers: int = 10
) -> Path:
    """One analysis through the runtime: the eight analysis skills, one strong model, sessions.

    `condition` "logs": the log alone, without the two experiment tools; "experiments":
    the agent may request A/B tests, which the engine's queue runs between sessions.
    Runs once per condition: a finished analysis is kept (scored as missing if it gave
    no answer), one that stopped while waiting for its experiments is resumed.
    """
    from fleetlab.agents.audit import AuditContext
    from fleetlab.agents.batch import AnalysisJob, Batch
    from fleetlab.agents.sessions import AgentSpec, run_analysis
    from fleetlab.agents.toolenv import build_tool_env

    existing = analysis_dir(paths, condition)
    state = "new" if existing is None else analysis_state(existing)
    if existing is not None and state == "finished":
        return existing
    engine = HeavyQueueEngine(paths, workers) if condition == "experiments" else None
    task = TASK.format(
        experiments=LOGS_ONLY if engine is None else WITH_EXPERIMENTS,
        experiment_skill="" if engine is None else "sensitivity-by-experiment, ",
    )
    tools = LOGS_ONLY_TOOLS if engine is None else None
    step = f"analysis-{condition}"
    note_step(paths, step, started=time.time(), condition=condition, model=model)
    env = build_tool_env(role="analysis")
    paths.analyses.mkdir(parents=True, exist_ok=True)
    paths.agent_runs.mkdir(parents=True, exist_ok=True)
    batch = Batch.after_gate(
        env,
        ["claude"],
        paths.export,
        [paths.observed],
        fleet=world.observed_fleet,
        texts={"system": SYSTEM, "task": task},
        rerun=False,
        concurrency=1,
        run_root=paths.agent_runs,
        analyses_root=paths.analyses,
        audit=AuditContext(
            tuple(paths.run_dirs()),
            world.observed_fleet,
            paths.world / "experiments",
            (world.spec.windows.experiment.first, world.spec.windows.experiment.last),
        ),
    )
    agent = AgentSpec("claude", model)
    if existing is not None and state == "waiting" and engine is not None:
        note_step(paths, step, analysis_dir=str(existing), resumed=time.time())

        def run(analysis: Any, export: Path) -> Any:
            return run_analysis(
                analysis,
                export,
                agent,
                SYSTEM,
                task,
                env,
                engine,
                max_sessions=MAX_SESSIONS - analysis.sessions_done(),
                tools=tools,
                run_root=paths.agent_runs,
                audit=batch.audit,
            )

        _resume(existing, engine, run)
        root = existing
    else:
        job = AnalysisJob(
            world.spec.world_id,
            paths.export,
            agent,
            SYSTEM,
            task,
            label=f"{VERSION} {model} {condition}",
            max_sessions=MAX_SESSIONS,
            tools=tools,
        )
        (record,) = batch.analyses([job], engine)
        if record is None:
            raise RuntimeError(f"the analysis did not start: {batch.stopped}")
        root = paths.analyses / record.analysis_id
    note_step(
        paths,
        step,
        ended=time.time(),
        analysis_dir=str(root),
        analysis_id=root.name,
        sessions=_session_rows(root),
        system=SYSTEM,
        task=task,
        batch_stopped=batch.stopped,
    )
    return root


REFUSED_RESULT = "exceeds maximum allowed tokens"


def refused_calls(run_dir: Path) -> list[tuple[str, str]]:
    """The tool calls whose result the CLI refused to show the model, as (tool, arguments).

    Claude Code refuses an MCP result above its token limit and hands the model a
    note instead; the tool server logged the call as answered. Read
    from the CLI's stream (`stdout.jsonl`); codex's stream has no such case known.
    """
    stream = run_dir / "stdout.jsonl"
    if not stream.is_file():
        return []
    uses: dict[str, tuple[str, str]] = {}
    refused: list[tuple[str, str]] = []
    for line in stream.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = str(block.get("name", "")).rsplit("__", 1)[-1]
                uses[str(block.get("id"))] = (name, json.dumps(block.get("input"), sort_keys=True))
            elif block.get("type") == "tool_result":
                text = json.dumps(block.get("content"))
                use = uses.get(str(block.get("tool_use_id")))
                if use is not None and REFUSED_RESULT in text:
                    refused.append(use)
    return refused


def tool_log(analysis_root: Path) -> list[dict[str, Any]]:
    """Every tool call of every session of the analysis, in order; a call whose result the
    CLI refused to show the model carries `delivered: false`."""
    from fleetlab.agents.run import call_log

    calls: list[dict[str, Any]] = []
    for session in sorted((analysis_root / "sessions").glob("session-*.json")):
        run_dir = json.loads(session.read_text()).get("run_dir")
        if not run_dir:
            continue
        log = call_log(Path(run_dir))
        if not log.is_file():
            continue
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        for tool, arguments in refused_calls(Path(run_dir)):
            for record in records:
                same = json.dumps(record.get("arguments"), sort_keys=True) == arguments
                if record.get("tool") == tool and same and "delivered" not in record:
                    record["delivered"] = False
                    break
        calls += records
    return calls


# ---- main -------------------------------------------------------------------------------------


def chain(
    paths: ChainPaths,
    steps: Sequence[str],
    *,
    workers: int = 10,
    model: str = "opus",
    fixtures_version: str | None = None,
) -> None:
    """Run the chain's steps in order; heavy ones under tools/heavy-run."""
    paths.root.mkdir(parents=True, exist_ok=True)
    world = step_world(paths)
    for step in steps:
        if step == "world":
            continue
        if step == "run":
            if not all(run_complete(paths.run(n), j) for n, j in run_plans(world).items()):
                heavy(paths, "run-heavy", ["--workers", str(workers)])
            for name, windows in RUN_WINDOWS.items():  # light: the records follow the world
                if not records_current(paths.run(name), world, windows):
                    write_records(paths.run(name), world, windows)
                    note_step(paths, "run", **{f"records_{name}": time.time()})
        elif step == "oracle":
            step_oracle(paths, world)
        elif step == "export":
            step_export(paths, world)
        elif step.startswith("analysis-"):
            step_analysis(paths, world, step.removeprefix("analysis-"), model, workers)
        elif step in ("fresh", "score", "fixtures"):
            from fleetlab.integration import scoring_step

            getattr(scoring_step, f"step_{step}")(
                paths, world, workers=workers, version=fixtures_version
            )
        else:
            raise ValueError(f"unknown step {step!r}; steps: {STEPS}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m fleetlab.integration")
    parser.add_argument("--dir", type=Path, required=True, help="the chain's root directory")
    parser.add_argument("--only", default=None, help="one step (run-heavy is the run's process)")
    parser.add_argument("--through", default=STEPS[-1], help="run the steps up to this one")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default="opus")
    parser.add_argument("--queue", type=Path, default=None, help="queue-heavy: the queue file")
    parser.add_argument(
        "--fixtures-version", default=None, help=f"the fixtures' version (default {VERSION})"
    )
    args = parser.parse_args(argv)
    paths = ChainPaths(args.dir.expanduser().resolve())
    if args.only == "run-heavy":
        world = step_world(paths)
        note_step(paths, "run-heavy", started=time.time())
        step_run(paths, world, args.workers)
        note_step(paths, "run-heavy", ended=time.time())
        return 0
    if args.only == "queue-heavy":
        if args.queue is None:
            parser.error("queue-heavy needs --queue FILE")
        note_step(paths, "queue-heavy", started=time.time())
        run_queue_here(paths, args.queue, args.workers)
        note_step(paths, "queue-heavy", ended=time.time())
        return 0
    if args.only == "fresh-heavy":
        from fleetlab.integration import scoring_step

        world = step_world(paths)
        note_step(paths, "fresh-heavy", started=time.time())
        scoring_step.run_fresh(paths, world, args.workers)
        note_step(paths, "fresh-heavy", ended=time.time())
        return 0
    if args.only is not None:
        steps: Sequence[str] = [args.only]
    else:
        steps = STEPS[: STEPS.index(args.through) + 1]
    chain(
        paths,
        steps,
        workers=args.workers,
        model=args.model,
        fixtures_version=args.fixtures_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
