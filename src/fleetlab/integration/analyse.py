"""Run and score a blind analysis of an existing development-world export.

The evaluator reads the world; the sandbox receives only the export, the
analysis skills, and the public window description. Experiment runs and scores
are written below a new output directory, never into the stored world.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fleetlab.agents.audit import AuditContext, completed_runs, tree_manifest
from fleetlab.agents.batch import Batch
from fleetlab.agents.checks import GateError, source_audit
from fleetlab.agents.sessions import (
    AgentSpec,
    Analysis,
    EngineReply,
    ExperimentQueue,
    continuation_prompt,
    reply_problems,
    run_session,
)
from fleetlab.agents.toolenv import ToolEnv, build_tool_env
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.world import WorldSpec
from fleetlab.integration.accounting import (
    PRICE_SOURCE,
    PRICES,
    collect_accounting,
    library_revision,
    repeat_group,
    reserve_repeat,
    verify_repeat,
)
from fleetlab.integration.chain import (
    BATCH_SIZE,
    HEAVY_RUN,
    LOGS_ONLY,
    LOGS_ONLY_TOOLS,
    MIOPEN_DIR,
    PRECISION,
    REPO,
    SYSTEM,
    WITH_EXPERIMENTS,
)
from fleetlab.loop.learned import with_learning


def task_prompt(spec: WorldSpec, condition: str, *, learning: bool = True) -> str:
    """Only public scheduling information crosses from the spec into the prompt.

    `learning` appends the paragraph on writing a learned skill
    (`fleetlab.loop.learned.LEARNING_INSTRUCTIONS`; docs/design/loop.md, section 9).
    """
    windows = spec.windows
    schedule = " ".join(
        f"The {name} window covers days "
        f"{getattr(windows, name).first} through {getattr(windows, name).last}."
        for name in ("observed", "experiment", "scoring")
    )
    experiments = LOGS_ONLY if condition == "logs" else WITH_EXPERIMENTS
    experiment_skill = "" if condition == "logs" else "sensitivity-by-experiment, "
    prompt = f"""Analyse this fleet and say why its attempts fail and what to do about it.

{schedule} Day numbers in the tables use this numbering.
The same robots do the same kind of work in the scoring window. Derive exposure
shares and observed rates from the log. {experiments}

The tools are your complete interface. Read the system description in parts with
describe (dictionary, fleet, vocabulary, costs), and query the catalogue table
for allowed actions and costs. Read the relevant skills with read_skill.

Follow triage, episode-cards, failure-modes, scope, sensitivity-from-logs,
{experiment_skill}rank-actions, write-diagnosis. Return to an earlier step when
new evidence contradicts a hypothesis. Save evidence and unresolved alternatives
with memory_write; memory is the only state that crosses sessions.

Submit both outputs: submit_diagnosis with causes, exact target and scope,
direction, confidence, evidence, mode summaries, warnings and retraining
indicators; submit_fix_set with at least ten catalogue actions. For each action,
report its expected singleton gain in percentage points of whole-fleet success
in the scoring window, with uncertainty and rationale. Order the actions for
the additional benefit they bring to the selected set, accounting for overlap,
redundant repairs and complementary causes as rank-actions describes. Keep the
singleton estimates honest even when that order differs from sorting them.
The list is cut at three and six actions and at four and eight operator hours.
Compute numbers with query.

When experiments are available, the total budget is eight A/B requests, at most
60 attempts per arm, with a power calculation for each request. Batch independent
requests when useful. Write notes and end with status waiting after requesting
experiments; read memory and results in the next session. Finish with status
done only after both outputs are submitted. Complete the authorized analysis
autonomously; report uncertainty rather than asking for clarification.
"""
    return with_learning(prompt) if learning else prompt


def inputs(world_root: Path) -> tuple[Path, WorldSpec, Fleet, list[Path]]:
    """Evaluator-only input paths and full fleet for the source/leak audit."""
    from fleetlab.oracle.world import world_dir_of, world_ready

    if problem := world_ready(world_root):
        raise ValueError(problem)
    files = world_dir_of(world_root)
    spec = WorldSpec.model_validate_json((files / "spec.json").read_text())
    fleet = Fleet.model_validate_json((files / "model.json").read_text())
    raw = json.loads((files / "world_truth.json").read_text())
    runs = [Path(p) for p in raw["base_runs"]]
    return files, spec, fleet, runs


def check_observed_export(export: Path, spec: WorldSpec, runs: list[Path], fleet: Fleet) -> int:
    """Bind the initial export to this world's observed episode records."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fleetlab.contracts.log import allowed_columns
    from fleetlab.log import LogReader

    description = json.loads((export / "log" / "description.json").read_text())
    if description["fleet"]["id"] != fleet.id:
        raise ValueError("export and world fleet ids differ")
    with LogReader(export) as reader:
        rows = reader.query(
            "SELECT count(*) AS n, min(day) AS first, max(day) AS last, "
            "count(experiment_id) AS experiments FROM episodes"
        ).to_pylist()
    row = rows[0]
    observed = spec.windows.observed
    if (
        not row["n"]
        or row["first"] < observed.first
        or row["last"] > observed.last
        or row["experiments"]
    ):
        raise ValueError(f"initial export is not exclusively the observed window: {row}")
    columns = list(allowed_columns("episodes"))
    expected = pa.concat_tables(
        [pq.read_table(run / "log" / "episodes.parquet", columns=columns) for run in runs]
    ).sort_by([("episode_id", "ascending")])
    actual = pq.read_table(export / "log" / "episodes.parquet", columns=columns).sort_by(
        [("episode_id", "ascending")]
    )
    if not actual.equals(expected):
        raise ValueError("export episode records differ from this world's base runs")
    return int(row["n"])


def check_base_runs(
    runs: list[Path],
    export: Path,
    output: Path,
    fleet: Fleet,
    public_fleet: Fleet,
    task_classes: Sequence[str],
) -> None:
    """Prove that the engine can republish the exact cleared observation inputs."""
    from fleetlab.log import export_runs

    rebuilt = output / "checked-base-export"
    # Use the same observed fleet as the engine's export_for. The full fleet is
    # audit vocabulary only: it also contains changes from later windows.
    export_runs(runs, rebuilt, fleet=public_fleet, audit_fleet=fleet, task_classes=task_classes)
    if tree_manifest(rebuilt) != tree_manifest(export):
        raise ValueError("experiment base runs do not reproduce the exact cleared export")


def copy_waiting_analysis(source: Path, output: Path, expected: dict[str, Any]) -> Analysis:
    """Copy an ended checkpoint, including completed/partly run queues, without rewriting it."""
    source = source.expanduser().resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise ValueError("continuation output must be outside its source result directory")
    previous = json.loads((source / "run.json").read_text())
    keys = ("world_root", "world_hash", "export", "condition", "repeat", "model", "effort")
    for key in (*keys, "library_revision"):
        if key in expected and previous.get(key) != expected[key]:
            raise ValueError(f"continuation differs from its source in {key}")
    if previous.get("state") != "waiting" or not previous.get("ended"):
        raise ValueError("continuation requires an ended waiting analysis")
    original = Path(previous["analysis"]).resolve()
    if original.parent != source:
        raise ValueError("source analysis must be inside its recorded result directory")
    tree_manifest(original)
    analysis = Analysis(original)
    number = analysis.sessions_done()
    if not number:
        raise ValueError("continuation requires a completed session")
    session = json.loads((original / "sessions" / f"session-{number:02d}.json").read_text())
    queue_path = original / "queues" / f"session-{number:02d}.json"
    queue = ExperimentQueue.model_validate_json(queue_path.read_text())
    requested = [q for q in analysis.requests() if q.session == number]
    if (
        session.get("state") != "waiting"
        or session.get("number") != number
        or session.get("queued") != [q.experiment_id for q in requested]
        or queue.experiments != requested
        or queue.session != number
        or queue.analysis_id != analysis.analysis_id
        or queue.world_id != analysis.world_id
        or (session.get("export") is not None and queue.export != session["export"])
    ):
        raise ValueError("pending requests differ from the completed session or saved queue")
    audit = json.loads((Path(session["run_dir"]) / "audit.json").read_text())
    if not audit.get("passed") or audit.get("manifest", {}).get("export") != tree_manifest(
        Path(queue.export)
    ):
        raise ValueError("cleared export changed since the completed source session")
    # A reply is a recoverable checkpoint even if the parent died before recording it.
    reply_file = queue_path.with_suffix(".reply.json")
    timing_file = queue_path.with_suffix(".timing.json")
    timing = json.loads(timing_file.read_text()) if timing_file.is_file() else {}
    if timing and not timing.get("ended"):
        raise ValueError("source queue has not ended; its engine state cannot be copied")
    if reply_file.is_file():
        reply = EngineReply.model_validate_json(reply_file.read_text())
        if problems := reply_problems(reply, queue):
            raise ValueError("saved queue reply: " + "; ".join(problems))
        done = analysis.results_by_id()
        if any(done.get(r.experiment_id, r) != r for r in reply.results):
            raise ValueError("stored results disagree with the saved queue reply")
    elif analysis.results_by_id() and not analysis.pending():
        raise ValueError("completed queue has no recoverable reply")
    engine = source / "engine"
    if engine.exists():
        if not reply_file.is_file() and not timing.get("ended"):
            raise ValueError("source engine has no ended queue checkpoint")
        tree_manifest(engine)
        shutil.copytree(engine, output / "engine")
    target = output / original.name
    shutil.copytree(original, target)
    return Analysis(target)


@dataclass
class DevelopmentEngine:
    """Queue experiments outside the sandbox; audit the new export before returning it."""

    files: Path
    output: Path
    workers: int
    env: ToolEnv
    fleet: Fleet
    runs: list[Path]
    texts: dict[str, str]
    base_runs: tuple[Path, ...] = ()
    experiment_days: tuple[int, int] | None = None

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        reply_file = queue_file.with_suffix(".reply.json")
        timing_file = queue_file.with_suffix(".timing.json")
        command = [
            str(HEAVY_RUN),
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "fleetlab.integration.analyse",
            "queue",
            "--truth",
            str(self.files / "world_truth.json"),
            "--engine-root",
            str(self.output / "engine"),
            "--queue",
            str(queue_file),
            "--reply",
            str(reply_file),
            "--workers",
            str(self.workers),
        ]
        for run in self.base_runs:
            command.extend(["--base-run", str(run)])
        if not reply_file.is_file():
            queued_at = time.time()
            timing_file.write_text(
                json.dumps(
                    {
                        "dispatch_id": uuid.uuid4().hex,
                        "queued_at": queued_at,
                        "state": "queued",
                        "queue_seconds": 0.0,
                        "simulation_seconds": None,
                    }
                )
            )
            command.extend(["--queued-at", str(queued_at)])
            try:
                subprocess.run(command, cwd=REPO, check=True)
            except KeyboardInterrupt:
                timing = json.loads(timing_file.read_text())
                timing["state"] = "interrupted"
                if not timing.get("started"):
                    timing["queue_seconds"] = None
                timing_file.write_text(json.dumps(timing, indent=2) + "\n")
                raise
            finally:
                timing = json.loads(timing_file.read_text())
                if timing["state"] == "queued":
                    timing.update(
                        queue_seconds=time.time() - queued_at,
                        simulation_seconds=0.0,
                        state="dispatch_failed" if not reply_file.is_file() else "done",
                        ended=time.time(),
                    )
                elif not timing.get("ended"):
                    # Only the worker can attest that a started queue ended.
                    # An interrupted caller may leave its systemd scope alive.
                    timing["state"] = "interrupted"
                timing_file.write_text(json.dumps(timing, indent=2) + "\n")
        reply = EngineReply.model_validate_json(reply_file.read_text())
        experiment_runs = completed_runs(self.output / "engine" / "experiments")
        audit_started = time.perf_counter()
        try:
            audit = source_audit(
                reply.export,
                [*self.runs, *experiment_runs],
                self.env,
                fleet=self.fleet,
                texts={
                    **self.texts,
                    "engine results": json.dumps(
                        [r.model_dump(mode="json") for r in reply.results]
                    ),
                },
            )
            (queue_file.with_suffix(".audit.json")).write_text(
                json.dumps(audit.as_json(), indent=2) + "\n"
            )
            if not audit.passed:
                raise GateError("experiment export: " + "; ".join(audit.failures()))
            if problems := reply_problems(reply, queue, self.experiment_days):
                raise ValueError("; ".join(problems))
        finally:
            if timing_file.is_file():
                timing = json.loads(timing_file.read_text())
                timing["audit_seconds"] = (
                    timing.get("audit_seconds", 0) + time.perf_counter() - audit_started
                )
                timing_file.write_text(json.dumps(timing, indent=2) + "\n")
        return reply


def queue_main(args: argparse.Namespace) -> None:
    """Only invoked by DevelopmentEngine, under tools/heavy-run."""
    from fleetlab.engine.experiments import WorldTruth, run_queue
    from fleetlab.engine.fleet import RunConfig

    started = time.time()
    timing_file = args.queue.with_suffix(".timing.json")
    timing = json.loads(timing_file.read_text())
    timing.update(started=started, queue_seconds=started - args.queued_at, state="running")
    timing_file.write_text(json.dumps(timing, indent=2) + "\n")
    phases: dict[str, float] = {}
    try:
        world = WorldTruth.load(args.truth)
        if args.base_run:
            world = world.model_copy(
                update={"base_runs": [str(p.resolve()) for p in args.base_run]}
            )
        config = RunConfig(out="", batch_size=BATCH_SIZE, workers=args.workers, precision=PRECISION)
        reply = run_queue(args.queue, world, args.engine_root, config=config, timings=phases)
        args.reply.write_text(reply.model_dump_json(indent=2) + "\n")
        timing["state"] = "done"
    finally:
        if timing["state"] == "running":
            timing["state"] = "failed"
        timing.update(phases)
        timing.update(ended=time.time(), worker_seconds=time.time() - started)
        timing_file.write_text(json.dumps(timing, indent=2) + "\n")


def analyse(args: argparse.Namespace) -> int:
    world_root, export, output = (
        path.expanduser().resolve() for path in (args.world_root, args.export, args.out)
    )
    if output == world_root or world_root in output.parents:
        raise ValueError("analysis output must be outside the stored world")
    previous: dict[str, Any] | None = None
    if args.resume_from:
        source = args.resume_from.expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("continuation output must be outside its source result directory")
        previous = json.loads((source / "run.json").read_text())
    if args.model not in PRICES:
        raise ValueError("no reviewed API-price snapshot for the configured model")
    files, spec, fleet, runs = inputs(world_root)
    episodes = check_observed_export(export, spec, runs, fleet)
    output.mkdir(parents=True, exist_ok=False)
    base_runs = [p.expanduser().resolve() for p in args.base_run] or runs
    if args.condition == "experiments":
        world_truth = json.loads((files / "world_truth.json").read_text())
        check_base_runs(
            base_runs,
            export,
            output,
            fleet,
            Fleet.model_validate(world_truth["fleet"]),
            world_truth["task_classes"],
        )
    task = task_prompt(spec, args.condition)
    texts = {"system": SYSTEM, "task": task}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    env = build_tool_env(role="analysis", ablate_fault_classes=args.ablate_fault_classes)
    assert env.skills is not None
    revision = library_revision(env.skills)
    world_hash = (files / "hash").read_text().strip()
    group = repeat_group(world_hash, args.condition, revision, args.model, args.effort)
    reservation = None if previous is None else previous["repeat_reservation"]
    if previous is not None:
        if reservation is None:
            raise ValueError("continuation has no registered model call")
        verify_repeat(reservation, group)
        if reservation["repeat"] != previous["repeat"]:
            raise ValueError("continuation repeat differs from its source")
    metadata: dict[str, Any] = {
        "source_commit": commit,
        "world_root": str(world_root),
        "world_hash": world_hash,
        "export": str(export),
        "condition": args.condition,
        "repeat": None if reservation is None else reservation["repeat"],
        "repeat_reservation": reservation,
        "library_revision": revision,
        "model": args.model,
        "effort": args.effort,
        "service_tier": "fast",
        "prices": PRICES[args.model],
        "price_source": PRICE_SOURCE,
        "price_checked": "2026-10-10",
        "ablate_fault_classes": args.ablate_fault_classes,
        "tool_env": env.digest(),
        "started": time.time(),
        "system": SYSTEM,
        "task": task,
        "base_runs": [str(p) for p in base_runs],
        "resume_from": str(args.resume_from.expanduser().resolve()) if args.resume_from else None,
        "lost_twice": False,  # scorer's no-answer signal for design entry 11
    }
    record_file = output / "run.json"
    record_file.write_text(json.dumps(metadata, indent=2) + "\n")

    def first_model_call(call_dir: Path) -> None:
        if metadata["repeat_reservation"] is None:
            receipt = reserve_repeat(group, output, call_dir)
            metadata.update(repeat=receipt["repeat"], repeat_reservation=receipt)
            record_file.write_text(json.dumps(metadata, indent=2) + "\n")

    analysis: Analysis | None = None
    state = "preflight"
    experiment_days = (spec.windows.experiment.first, spec.windows.experiment.last)
    try:
        batch = Batch.after_gate(
            env, ["codex"], export, runs, fleet=fleet, texts=texts, rerun=False, concurrency=1
        )
        (output / "gate.json").write_text(json.dumps(batch.records, indent=2) + "\n")
        analysis = (
            copy_waiting_analysis(args.resume_from, output, metadata)
            if args.resume_from
            else Analysis.create(spec.world_id, args.condition, base=output)
        )
        metadata["analysis"] = str(analysis.root)
        if previous is not None:
            export = Path(previous["final_export"])
        engine = (
            DevelopmentEngine(
                files,
                output,
                args.workers,
                env,
                fleet,
                runs,
                texts,
                tuple(base_runs),
                experiment_days,
            )
            if args.condition == "experiments"
            else None
        )
        agent = AgentSpec("codex", args.model, effort=args.effort, service_tier="fast")
        allowed = LOGS_ONLY_TOOLS if engine is None else None
        state = "waiting" if analysis.sessions_done() or analysis.pending() else "not started"
        # Refused requests don't use the eight-request budget. Extra readouts
        # permit correction, with a finite session ceiling for a failing agent.
        for number in range(analysis.sessions_done() + 1, args.max_sessions + 1):
            if engine is not None and (analysis.pending() or (previous is not None and number > 1)):
                queue_file = analysis.root / "queues" / f"session-{number - 1:02d}.json"
                if queue_file.is_file():
                    queue = ExperimentQueue.model_validate_json(queue_file.read_text())
                    known = {q.experiment_id: q for q in analysis.requests()}
                    if any(known.get(q.experiment_id) != q for q in queue.experiments):
                        raise ValueError("saved queue differs from recorded requests")
                else:
                    written = analysis.write_queue(number - 1, export)
                    assert written is not None
                    queue_file, queue = written
                try:
                    reply = engine.run_queue(queue_file, queue)
                except Exception:
                    # No result is delivered for a failed machine/worker. Keep
                    # the waiting queue and its immutable plan for --resume-from.
                    metadata["stop_reason"] = "experiment_execution_failed"
                    raise
                done = analysis.results_by_id()
                if any(done.get(r.experiment_id, r) != r for r in reply.results):
                    raise ValueError("stored results differ from queue reply")
                analysis.record_results([r for r in reply.results if r.experiment_id not in done])
                export = reply.export
                previous = None
            print(f"session {number}: {args.condition}", flush=True)
            session = run_session(
                analysis,
                export,
                agent,
                SYSTEM,
                task if number == 1 else continuation_prompt(number, task),
                env,
                tools=allowed,
                run_root=output / "agent-runs",
                on_start=first_model_call,
                audit=AuditContext(
                    tuple(runs), fleet, output / "engine" / "experiments", experiment_days
                ),
            )
            state = session.state
            metadata["lost_twice"] = session.lost_twice
            print(json.dumps(asdict(session)), flush=True)
            if session.error in ("auth", "rate_limit") or state != "waiting" or engine is None:
                break
            if not analysis.pending():
                break
            # Checkpoint the exact queue before a possible interruption. Cached
            # plans bind these bytes; a continuation must not remint its queue.
            analysis.write_queue(number, export)
    finally:
        if analysis is not None:
            sessions = analysis.session_records()
            if sessions and not sessions[-1].in_progress:
                # The session may have saved its final start before an
                # interruption prevented run_session from returning it.
                state = sessions[-1].state
                metadata["lost_twice"] = sessions[-1].lost_twice
        metadata.update(ended=time.time(), state=state, final_export=str(export))
        results = {} if analysis is None else analysis.results_by_id()
        requests = [] if analysis is None else analysis.requests()
        delivered = [r for r in results.values() if r.status == "done"]
        metadata["experiments"] = {
            "requested": len(requests),
            "completed": len(delivered),
            "refused": sum(r.status == "refused" for r in results.values()),
            "pending": len(requests) - len(results),
            "delivered_episodes": sum(len(a.episode_ids) for r in delivered for a in r.arms),
        }
        metadata["accounting"] = collect_accounting(output, metadata, episodes)
        record_file.write_text(json.dumps(metadata, indent=2) + "\n")
        if analysis is not None and metadata["repeat"] is not None:
            verify_repeat(metadata["repeat_reservation"], group)
            score_analysis(world_root, analysis, output, args.condition, metadata["repeat"])
    return 0 if state == "done" and analysis and analysis.diagnosis() and analysis.fix_set() else 1


def score_analysis(
    world_root: Path, analysis: Analysis, output: Path, condition: str, repeat: int
) -> None:
    """Evaluator-only finalization; callers and tests can isolate model orchestration."""
    from fleetlab.scoring.__main__ import score_world
    from fleetlab.scoring.markdown import report_markdown

    report = score_world(
        world_root,
        analysis.root,
        condition="with experiments" if condition == "experiments" else "logs alone",
        repeat=repeat,
    )
    score_dir = output / "score"
    score_dir.mkdir()
    (score_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (score_dir / "report.md").write_text(report_markdown(report))
    print(score_dir / "report.md", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--world-root", type=Path, required=True)
    run.add_argument("--export", type=Path, required=True, help="export cleared by its leak audit")
    run.add_argument("--out", type=Path, required=True, help="new result directory")
    run.add_argument("--condition", choices=("logs", "experiments"), required=True)
    run.add_argument("--model", default="gpt-6-astra")
    run.add_argument("--ablate-fault-classes", action="store_true")
    run.add_argument("--effort", choices=("medium", "high", "xhigh"), default="xhigh")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-sessions", type=int, default=17)
    run.add_argument(
        "--base-run",
        type=Path,
        action="append",
        default=[],
        help="rebuilt observed run, checked against the exact cleared export",
    )
    run.add_argument(
        "--resume-from",
        type=Path,
        help="ended waiting checkpoint, including saved experiment results",
    )
    queue = commands.add_parser("queue", help=argparse.SUPPRESS)
    for name in ("truth", "engine-root", "queue", "reply"):
        queue.add_argument(f"--{name}", type=Path, required=True)
    queue.add_argument("--workers", type=int, default=4)
    queue.add_argument("--queued-at", type=float, required=True)
    queue.add_argument("--base-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    os.environ.setdefault("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR))
    if args.command == "queue":
        queue_main(args)
        return 0
    return analyse(args)


if __name__ == "__main__":
    sys.exit(main())
