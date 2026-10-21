"""The round driver of the improvement loop (docs/design/loop.md, section 8).

One engagement of an agent with a fleet is a sequence of rounds. A round is the
fleet's operation under the current ledger, the export of everything the fleet has
seen so far, an investigation with experiments from the current state, and the
agent's `RoundDecision`, which makes the next version of the ledger. After the last
round the states are evaluated on the reserved attempts (lane state's `evaluate`).

This module's first part is the investigation: one analysis with experiments,
through the agent runtime's sessions (`fleetlab.agents.sessions.run_analysis`) and
the engine's experiment queue, each queue a heavy job of its own under
`tools/heavy-run` so that the analysis never holds the heavy lock while the model
thinks. It is deliberately thin. The stage-4 pipeline (`make analyse`) wraps the
same two pieces in the retired campaign's machinery (a repeat register, the old
scorer, a fixed budget of eight requests of sixty); the driver keeps what protects
the agent and the record (the CLI gate, the launch audit of every mounted byte, the
leak audit of every new export, the public refusal codes) and drops the rest.

The budget is counted in robot attempts per round (`AttemptBudget`): the episodes
over every arm of every request. A refused request costs nothing, because the tool
server does not charge it; every attempt that was simulated counts, failed ones and
the engine's chain-sample replays included in the cost record.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import (
    AgentSpec,
    Analysis,
    AnalysisRecord,
    EngineReply,
    ExperimentQueue,
    SessionRecord,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.world import WorldSpec
from fleetlab.loop.scripted import ScriptedAgent, request_step, submit_analysis_step

__all__ = [
    "ATTEMPTS_PER_HOUR",
    "DEFAULT_ROUND_ATTEMPTS",
    "AttemptBudget",
    "HeavyQueue",
    "ObservedWorld",
    "costs_of",
    "investigate",
    "investigation_task",
]

REPO = Path(__file__).resolve().parents[3]
# The exit status of an aborted engagement or investigation (a required decision or
# answer missing after an auth error, a rate limit, a lost call or exhausted sessions).
EXIT_ABORTED = 3
HEAVY_RUN = REPO / "tools" / "heavy-run"
LANE = "driver"

# What this machine simulates: about 1,500 attempts an hour through the heavy queue
# with four workers, faulted fleets included (failed attempts run to the step limit
# and cost about 1.7 times a success; memory notes `failed-episodes-cost-more`,
# `engine-memory-guard-workers`; the miniature fleet's queue ran 320 arms in 468 s
# with ten workers, 2,884 an hour).
ATTEMPTS_PER_HOUR = 1500

# The default experiment budget of a round: 480 attempts, about twenty minutes of the
# heavy queue at ATTEMPTS_PER_HOUR. It buys, for example, four two-arm experiments of
# 60 per arm, or six of 40, or two three-arm sweeps of 80: enough to settle the large
# effects a fleet's faults leave (a site at 27% against 94%: twenty per arm suffice)
# and one or two finer comparisons, but not to test every suspicion, so choosing what
# to test stays the agent's work. Two rounds spend 960, the old analysis-wide budget
# of eight requests of sixty, which the plan proposed for both rounds together. With
# the sessions (minutes each) a round then takes about an hour of wall time, and a
# demonstration of four runs of two rounds fits a night beside the operation and
# evaluation runs.
DEFAULT_ROUND_ATTEMPTS = 480


@dataclass(frozen=True)
class AttemptBudget:
    """A round's experiment budget in robot attempts: episodes over every arm of every request.

    An arm is limited only by what is left of the round's attempts, unless a run sets
    `per_arm` (the miniature run's 20); `requests` caps the count of requests, None for no
    count of its own. The tool server enforces them from the round's first session on
    (`budget_first_session`), so that earlier rounds' requests do not count; refused
    requests are not charged. (The stage-4 pipeline's 60 per arm is the tool server's
    historical default for paths that set no budget, never this one's.)
    """

    attempts: int = DEFAULT_ROUND_ATTEMPTS
    per_arm: int | None = None
    requests: int | None = None

    def __post_init__(self) -> None:
        if self.attempts < 2 or (self.per_arm is not None and self.per_arm < 1):
            raise ValueError(f"an impossible budget: {self}")
        if self.requests is not None and self.requests < 1:
            raise ValueError(f"an impossible budget: {self}")

    def limits(self, first_session: int = 1) -> dict[str, int]:
        """The tool server's limits besides `max_experiment_episodes` (`run_analysis(limits=)`)."""
        return {
            # Every request costs at least two attempts, so attempts // 2 never binds alone.
            "max_experiment_requests": self.requests or max(1, self.attempts // 2),
            # Without a run setting an arm may take all the attempts the round has left.
            "max_episodes_per_arm": self.per_arm or self.attempts,
            "budget_first_session": first_session,
        }

    def text(self) -> str:
        """The budget in the words of the task prompt."""
        count = f"at most {self.requests} requests, " if self.requests is not None else ""
        arm = (
            f" and at most {self.per_arm} attempts per arm"
            if self.per_arm is not None
            else "; an arm may take any part of it"
        )
        return (
            f"{count}{self.attempts} robot attempts in all (an experiment costs its episodes "
            f"per arm times its arms){arm}"
        )

    def hours(self) -> float:
        """The budget's simulation time at this machine's rate."""
        return self.attempts / ATTEMPTS_PER_HOUR


@dataclass(frozen=True)
class ObservedWorld:
    """A development world as part one's investigation reads it: its observed window and
    files. Truth side: never mounted, never shown. (The rounds use lane state's
    `fleetlab.loop.world.LoopWorld`.)"""

    root: Path
    files: Path  # the world files: spec.json, model.json, world_truth.json
    spec: WorldSpec
    fleet: Fleet  # the full model: vocabulary for the leak audits only
    public_fleet: Fleet  # the fleet as its operators record it (world_truth.fleet)
    task_classes: tuple[str, ...]
    base_runs: tuple[Path, ...]  # the as-run observed window

    @classmethod
    def load(cls, root: Path) -> ObservedWorld:
        from fleetlab.oracle.world import world_dir_of, world_ready

        root = root.expanduser().resolve()
        if problem := world_ready(root):
            raise ValueError(problem)
        files = world_dir_of(root)
        truth = json.loads((files / "world_truth.json").read_text())
        return cls(
            root=root,
            files=files,
            spec=WorldSpec.model_validate_json((files / "spec.json").read_text()),
            fleet=Fleet.model_validate_json((files / "model.json").read_text()),
            public_fleet=Fleet.model_validate(truth["fleet"]),
            task_classes=tuple(truth["task_classes"]),
            base_runs=tuple(Path(p) for p in truth["base_runs"]),
        )

    @property
    def world_id(self) -> str:
        return str(self.spec.world_id)

    @property
    def experiment_days(self) -> tuple[int, int]:
        window = self.spec.windows.experiment
        return (window.first, window.last)

    def export_observed(self, out: Path) -> Path:
        """The observed window's export, built with this checkout's writer and leak-audited."""
        from fleetlab.log import export_runs

        export_runs(
            list(self.base_runs),
            out,
            fleet=self.public_fleet,
            audit_fleet=self.fleet,
            task_classes=self.task_classes,
        )
        return out


def investigation_task(spec: WorldSpec, budget: AttemptBudget) -> str:
    """The task of a one-round investigation with experiments, the analysis answer format.

    Only public scheduling information crosses from the spec into the prompt; it is
    the stage-4 experiments prompt (`integration.analyse.task_prompt`) with the budget
    in force instead of the old eight requests of sixty.
    """
    from fleetlab.integration.analyse import task_prompt

    old = (
        "When experiments are available, the total budget is eight A/B requests, at most\n"
        "60 attempts per arm, with a power calculation for each request."
    )
    prompt = task_prompt(spec, "experiments")
    if old not in prompt:
        raise ValueError("the stage-4 task prompt changed; update investigation_task")
    return prompt.replace(
        old,
        f"The experiment budget of this analysis is {budget.text()}. Every request needs a\n"
        "power calculation; a request above the budget is refused and costs nothing.",
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _queue_log(path: Path | None, event: str, commit: str, purpose: str, attempts: str) -> None:
    """One line of the lanes' shared heavy-queue log (time, lane, commit, purpose, attempts)."""
    if path is None:
        return
    stamp = subprocess.check_output(["date", "-Is"], text=True).strip()
    line = f"{stamp} {event} lane={LANE} commit={commit} purpose={purpose} attempts={attempts}"
    with path.open("a") as stream:
        stream.write(line.replace("\n", " ") + "\n")


@dataclass
class HeavyQueue:
    """The runtime's `Engine`: the inner engine's queue as a heavy job, with the queue log
    and a cost record per queue (attempts per experiment and arm, jobs, timings)."""

    inner: Any  # an Engine whose run_queue starts tools/heavy-run (DevelopmentEngine)
    purpose: str
    queue_log: Path | None = None
    commit: str = field(default_factory=_commit)
    records: list[dict[str, Any]] = field(default_factory=list)

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        planned = sum(q.request.episodes_per_arm * len(q.request.arms) for q in queue.experiments)
        record: dict[str, Any] = {
            "queue": str(queue_file),
            "session": queue.session,
            "experiments": [q.experiment_id for q in queue.experiments],
            "planned_attempts": planned,
            "launched": _now(),
        }
        self.records.append(record)
        what = (
            f"{self.purpose.replace(' ', '_')}:queue-session-{queue.session}"
            f"({len(queue.experiments)}-requests)"
        )
        _queue_log(self.queue_log, "queued", self.commit, what, f"{planned}(planned-arm-attempts)")
        started = time.perf_counter()
        try:
            reply: EngineReply = self.inner.run_queue(queue_file, queue)
        except BaseException as error:
            record.update(state="failed", error=type(error).__name__, ended=_now())
            record["wall_seconds"] = time.perf_counter() - started
            record.update(_engine_jobs(queue_file, queue))
            _queue_log(
                self.queue_log,
                f"ended-failed({type(error).__name__})",
                self.commit,
                what,
                _simulated_text(record.get("simulated") or {}),
            )
            raise
        record.update(state="done", ended=_now(), wall_seconds=time.perf_counter() - started)
        record["results"] = [
            {
                "experiment_id": r.experiment_id,
                "status": r.status,
                "message": r.message,
                "attempts_per_arm": [len(a.episode_ids) for a in r.arms],
            }
            for r in reply.results
        ]
        record["delivered_attempts"] = sum(
            len(a.episode_ids) for r in reply.results if r.status == "done" for a in r.arms
        )
        record.update(_engine_jobs(queue_file, queue))
        _queue_log(
            self.queue_log,
            "ended",
            self.commit,
            what,
            _simulated_text(record.get("simulated") or {}),
        )
        return reply


def _simulated_text(simulated: Mapping[str, Any]) -> str:
    """The queue log's attempts field: arm attempts, replays, confirmed failed
    simulations and unconfirmed assignments (`loop.costs.simulated_jobs`); a queue's
    probe run (`probes`, lane probe) apart, on the same semantics: completed probe
    episodes, replays and confirmed failed simulations (together the JSON's
    `simulated_confirmed`), then, where present, partial jobs (records left by a run that
    broke off) and unconfirmed assignments (unknown, kept charged as reserved)."""
    text = (
        f"{simulated.get('run_jobs', '?')}(attempts)+{simulated.get('replays', '?')}"
        f"(chain-sample-replays)+{simulated.get('failed_confirmed', '?')}(failed-simulations)"
        f"+{simulated.get('unconfirmed', '?')}(unconfirmed-assignments)"
    )
    probes = simulated.get("probes")
    keys = ("run_jobs", "replays", "failed_confirmed", "partial_jobs", "unconfirmed")
    if probes and any(probes.get(k) for k in keys):
        text += (
            f";probes:{probes.get('run_jobs', 0)}(probe-attempts)+{probes.get('replays', 0)}"
            f"(chain-sample-replays)+{probes.get('failed_confirmed', 0)}(failed-simulations)"
        )
        if probes.get("partial_jobs"):
            text += f"+{probes['partial_jobs']}(partial)"
        if probes.get("unconfirmed"):
            text += f"+{probes['unconfirmed']}(unconfirmed-assignments)"
    return text


def _engine_jobs(queue_file: Path, queue: ExperimentQueue) -> dict[str, Any]:
    """What the engine simulated for a queue, from its run directory and its timing file
    (DevelopmentEngine's layout: <output>/engine/experiments/<analysis>/session-NN)."""
    from fleetlab.loop.costs import simulated_jobs

    out: dict[str, Any] = {}
    timing = queue_file.with_suffix(".timing.json")
    if timing.is_file():
        out["timing"] = json.loads(timing.read_text())
    # queue_file is <analysis>/queues/session-NN.json; the analysis lies in <output>.
    output = queue_file.parents[2]
    run_dir = output / "engine" / "experiments" / queue.analysis_id / f"session-{queue.session:02d}"
    out["simulated"] = simulated_jobs(run_dir)
    return out


def costs_of(
    sessions: Sequence[SessionRecord],
    analysis: Analysis,
    queues: Sequence[Mapping[str, Any]],
    model: str,
    first_session: int = 1,
) -> dict[str, Any]:
    """Every cost of an investigation: model starts, wall time, tokens and tool calls summed
    over every start (`loop.costs.session_costs`), the attempts simulated per experiment
    and arm (a refused request costs nothing), and the heavy queue's jobs and times."""
    from fleetlab.loop.costs import experiment_costs, probe_costs, queue_costs, session_costs

    experiments = experiment_costs(analysis, first_session)
    return {
        **session_costs(sessions, model),
        **experiments,
        **probe_costs(analysis, first_session),
        "requests_charged": experiments["requests"] - experiments["requests_refused"],
        **queue_costs(queues),
        "queues": list(queues),
    }


def investigate(
    world: ObservedWorld,
    out: Path,
    *,
    agent: AgentSpec,
    budget: AttemptBudget,
    max_sessions: int,
    workers: int = 4,
    queue_log: Path | None = None,
    label: str = "investigation",
    scripted: ScriptedAgent | None = None,
) -> dict[str, Any]:
    """One analysis with experiments of the observed window, its record written to
    `out/run.json`: the real path request, simulation, public result, the next session
    reading it, and the decision (diagnosis and ranked fix set).

    With `scripted` the sessions are a script's (`fleetlab.loop.scripted`), no model is
    called and the CLI gate is not needed; the queue and the engine run as for a model.
    """
    from unittest import mock

    from fleetlab.agents.audit import AuditContext
    from fleetlab.agents.batch import Batch
    from fleetlab.agents.sessions import run_analysis
    from fleetlab.agents.toolenv import build_tool_env
    from fleetlab.integration.analyse import DevelopmentEngine
    from fleetlab.integration.chain import SYSTEM

    out = out.expanduser().resolve()
    if out == world.root or world.root in out.parents:
        raise ValueError("the investigation's output lies outside the stored world")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record: dict[str, Any] = {
        "kind": "investigation",
        "label": label,
        "source_commit": _commit(),
        "world_root": str(world.root),
        "world_hash": (world.files / "hash").read_text().strip(),
        "agent": asdict(agent),
        "budget": {**asdict(budget), "text": budget.text(), "hours": budget.hours()},
        "max_sessions": max_sessions,
        "workers": workers,
        "started": started,
        "started_at": _now(),
    }
    run_file = out / "run.json"
    run_file.write_text(json.dumps(record, indent=1) + "\n")
    export = world.export_observed(out / "export")
    task = investigation_task(world.spec, budget)
    texts = {"system": SYSTEM, "task": task}
    env = build_tool_env(role="analysis")
    runs = list(world.base_runs)
    if scripted is None:
        # The CLI gate where its record is missing or stale, the source audit of this
        # export, prompts and mounted code, and the login check: before any model call.
        batch = Batch.after_gate(
            env, [agent.backend], export, runs, fleet=world.fleet, texts=texts, rerun=False
        )
        (out / "gate.json").write_text(json.dumps(batch.records, indent=1, default=str) + "\n")
    else:
        record["scripted"] = True
    analysis = Analysis.create(world.world_id, label, base=out)
    engine = HeavyQueue(
        DevelopmentEngine(
            world.files,
            out,
            workers,
            env,
            world.fleet,
            runs,
            texts,
            tuple(world.base_runs),
            world.experiment_days,
        ),
        purpose=f"{label} on {world.world_id}",
        queue_log=queue_log,
    )
    record.update(analysis=str(analysis.root), task=task, system=SYSTEM, tool_env=env.digest())
    run_file.write_text(json.dumps(record, indent=1) + "\n")
    analysis_record = AnalysisRecord(analysis.analysis_id, analysis.world_id)
    patch = (
        mock.patch("fleetlab.agents.sessions.run_agent", scripted)
        if scripted is not None
        else contextlib.nullcontext()
    )
    try:
        with patch:
            analysis_record = run_analysis(
                analysis,
                export,
                agent,
                SYSTEM,
                task,
                env,
                engine,
                max_sessions=max_sessions,
                max_experiment_episodes=budget.attempts,
                limits=budget.limits(),
                run_root=out / "agent-runs",
                audit=AuditContext(
                    tuple(runs), world.fleet, out / "engine" / "experiments", world.experiment_days
                ),
            )
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        sessions = analysis_record.sessions or analysis.session_records()
        record.update(
            ended=time.time(),
            ended_at=_now(),
            states=[s.state for s in sessions],
            sessions=[_session_summary(s) for s in sessions],
            costs=costs_of(sessions, analysis, engine.records, agent.model),
            submitted={
                "diagnosis": analysis.diagnosis() is not None,
                "fix_set": analysis.fix_set() is not None,
            },
        )
        record["wall_seconds"] = record["ended"] - started
        run_file.write_text(json.dumps(record, indent=1, default=str) + "\n")
    return record


def _session_summary(session: SessionRecord) -> dict[str, Any]:
    return {
        "number": session.number,
        "state": session.state,
        "ok": session.ok,
        "error": session.error,
        "seconds": session.seconds,
        "tool_calls": session.tool_calls,
        "queued": session.queued,
        "submitted": session.submitted,
        "starts": len(session.attempts),
        "lost_twice": session.lost_twice,
        "run_dir": session.run_dir,
        "report": session.report,
    }


def run_rounds(
    world_root: Path | Any,
    out: Path,
    *,
    agent: AgentSpec,
    budget: AttemptBudget,
    max_sessions: int,
    workers: int = 4,
    queue_log: Path | None = None,
    label: str = "engagement",
    scripted: ScriptedAgent | None = None,
    heavy: bool = True,
    setup: Any = None,
    layout: Any = None,
    env: Any = None,
    panel_per_robot_day: int | None = None,
    learning: bool = True,
    learned: Path | None = None,
    fleet_label: str = "an earlier fleet",
    analysis: Analysis | None = None,
    probes: bool = False,
    panel_video_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """An engagement of rounds on a stored world (docs/design/loop.md, section 8), with
    lane state's layout for the world, its ledger, operation, export, experiments and
    evaluation. Every simulation step is a heavy job of its own unless `heavy` is False
    (the toy cell of the CPU tests). The record is `out/engagement/engagement.json`;
    lane state's files (layout, ledgers, rounds, evaluation) lie in `out/loop/`. `env` is
    the agents' tool environment (default: the analysis skills of this checkout, with
    probes also the base skill on step responses, `loop.probes.PROBE_SKILLS`).

    `analysis` is a fresh analysis to run in, made by the caller: the reuse comparison makes
    each arm's, with its own id and, in one arm, the learned library mounted
    (`fleetlab.loop.reuse`); then `learned` stays None."""
    from unittest import mock

    from fleetlab.agents.batch import Batch
    from fleetlab.agents.toolenv import build_tool_env
    from fleetlab.engine.fleet import RunConfig
    from fleetlab.integration.chain import SYSTEM
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.operate import EngineSetup
    from fleetlab.loop.rounds import Engagement, run_engagement
    from fleetlab.loop.world import LoopWorld
    from fleetlab.loop.world_loop import HeavySteps, WorldLoop

    out = out.expanduser().resolve()
    # A loaded LoopWorld (the toy world, with its own fault registry) or a world's root.
    world = world_root if isinstance(world_root, LoopWorld) else LoopWorld.load(world_root)
    if out == world.root or world.root in out.parents:
        raise ValueError("the engagement's output lies outside the stored world")
    out.mkdir(parents=True, exist_ok=False)
    setup = setup or EngineSetup(
        config=RunConfig(out="", batch_size=4, workers=workers, precision="fp16_res32"),
        panel_video_indices=tuple(panel_video_indices),
    )
    loop_root = out / "loop"
    steps = (
        HeavySteps(world.root, loop_root, loop_root / "setup.json", queue_log) if heavy else None
    )
    state = WorldLoop.create(
        world,
        layout or layout_for_world(world.spec),
        loop_root,
        setup,
        steps,
        panel_per_robot_day=panel_per_robot_day,
    )
    if env is None:
        from fleetlab.loop.probes import probe_tool_env_args

        # The analysis skills; with probes, the base skill on step responses beside them.
        env = build_tool_env(**probe_tool_env_args(probes))
    gates: list[Any] = []

    def gate(export: Path, texts: Mapping[str, str], runs: Sequence[Path]) -> None:
        # The CLI gate where its record is missing or stale, the source audit of the
        # round's export and prompts, and the login check, before the round's first call.
        batch = Batch.after_gate(
            env, [agent.backend], export, list(runs), fleet=world.model, texts=texts, rerun=False
        )
        gates.append(batch.records)

    how = Engagement(
        agent=agent,
        budget=budget,
        max_sessions=max_sessions,
        system=SYSTEM,
        env=env,
        fleet=world.model,
        label=label,
        gate=None if scripted is not None else gate,
        learning=learning,
        learned=learned,
        fleet_label=fleet_label,
        analysis=analysis,
        probes=probes,
    )
    patch = (
        mock.patch("fleetlab.agents.sessions.run_agent", scripted)
        if scripted is not None
        else contextlib.nullcontext()
    )
    try:
        with patch:
            record = run_engagement(state, world.world_id, out / "engagement", how)
    finally:
        extra = {
            "source_commit": _commit(),
            "world_root": str(world.root),
            "loop": str(loop_root),
            "scripted": scripted is not None,
            "budget_hours_at_machine_rate": budget.hours(),
            "heavy_steps": steps.records if steps is not None else [],
            "simulated": state.simulated(),
            "gates": gates,
        }
        (out / "driver.json").write_text(json.dumps(extra, indent=1, default=str) + "\n")
    return {**record, **extra}


def step_main(args: argparse.Namespace) -> int:
    """One simulation step of a loop, in its own process (`HeavySteps`, under heavy-run)."""
    from fleetlab.agents.sessions import ExperimentQueue
    from fleetlab.loop.layout import LoopLayout
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.state import Ledger
    from fleetlab.loop.world import LoopWorld
    from fleetlab.loop.world_loop import setup_from_json

    world = LoopWorld.load(args.world_root)
    paths = LoopPaths(args.loop)
    layout = LoopLayout.model_validate_json(paths.layout.read_text())
    setup = setup_from_json(json.loads(args.setup.read_text()))
    if args.step == "operate":
        from fleetlab.loop.operate import operate

        operate(world, layout, args.round, Ledger.load(args.ledger), paths, setup)
    elif args.step == "queue":
        from fleetlab.loop.experiments import RoundEngine

        queue = ExperimentQueue.model_validate_json(args.queue.read_text())
        engine = RoundEngine(
            world,
            layout,
            args.round,
            Ledger.load(args.ledger),
            paths,
            setup,
            earlier=[args.analysis] * (args.round - 1),
            analysis_id=args.analysis,
        )
        reply = engine.run_queue(args.queue, queue)
        temporary = args.reply.with_name(f".{args.reply.name}.tmp")
        temporary.write_text(reply.model_dump_json(indent=1) + "\n")
        temporary.replace(args.reply)
    elif args.step == "evaluate":
        from fleetlab.loop.evaluate import evaluate

        states = {}
        for item in args.state:
            name, _, path = item.partition("=")
            states[name] = Ledger.load(path)
        evaluate(world, layout, states, paths, setup, per_robot_day=args.per_robot_day)
    return 0


def _fleet_label(text: str) -> str:
    """`--fleet-label`: how a learned skill's provenance names the fleet. The learned freeze
    after the rounds refuses a label that holds an identifier, the fleet's own id among them
    (`fleetlab.loop.learned.freeze`); checked here, such a label is refused before anything
    runs instead of after the rounds."""
    from fleetlab.contracts.skills import neutral_identifiers

    if problems := neutral_identifiers(text):
        raise argparse.ArgumentTypeError(
            f"{text!r} holds identifiers {problems}: a learned skill's provenance names the "
            "fleet in words, such as 'the first fleet', never by an id"
        )
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.driver", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rounds = commands.add_parser("rounds", help="an engagement of rounds on a stored world")
    rounds.add_argument("--world-root", type=Path, required=True)
    rounds.add_argument("--out", type=Path, required=True, help="a new directory")
    rounds.add_argument("--backend", default="codex", choices=("codex", "claude"))
    rounds.add_argument("--model", default="gpt-6-astra")
    rounds.add_argument("--effort", default="xhigh")
    rounds.add_argument("--service-tier", default="fast")
    rounds.add_argument("--max-sessions", type=int, default=4, help="per round")
    rounds.add_argument("--attempts", type=int, default=DEFAULT_ROUND_ATTEMPTS, help="per round")
    rounds.add_argument("--per-arm", type=int, help="a limit per arm; default: none")
    rounds.add_argument("--requests", type=int, help="per round; default: no count of its own")
    rounds.add_argument("--workers", type=int, default=4)
    rounds.add_argument("--queue-log", type=Path, help="append start and end lines of heavy jobs")
    rounds.add_argument("--label", default="engagement")
    rounds.add_argument(
        "--scripted", type=Path, help="no model: a script of sessions (JSON, loop.scripted)"
    )
    rounds.add_argument(
        "--panel-per-robot-day", type=int, help="evaluation attempts per robot and day"
    )
    rounds.add_argument(
        "--panel-video-index",
        type=int,
        action="append",
        default=[],
        help="panel attempts with this index within their robot and day record video",
    )
    rounds.add_argument(
        "--no-learning", action="store_true", help="no learned skills written or frozen"
    )
    rounds.add_argument("--learned", type=Path, help="a frozen learned library to mount")
    rounds.add_argument(
        "--fleet-label",
        type=_fleet_label,
        default="an earlier fleet",
        help="the fleet in a skill's provenance, in words: no identifier, not the fleet's id",
    )
    rounds.add_argument(
        "--probes",
        action="store_true",
        help="offer the agent probes it composes (request_probe, probe_results; loop.md 10)",
    )
    step = commands.add_parser("step", help=argparse.SUPPRESS)
    step.add_argument("step", choices=("operate", "queue", "evaluate"))
    step.add_argument("--world-root", type=Path, required=True)
    step.add_argument("--loop", type=Path, required=True)
    step.add_argument("--setup", type=Path, required=True)
    step.add_argument("--round", type=int)
    step.add_argument("--ledger", type=Path)
    step.add_argument("--analysis")
    step.add_argument("--queue", type=Path)
    step.add_argument("--reply", type=Path)
    step.add_argument("--state", action="append", default=[], help="name=ledger file")
    step.add_argument("--per-robot-day", type=int, help="evaluation panel attempts")
    one = commands.add_parser("investigate", help="one analysis with experiments (part one)")
    one.add_argument("--world-root", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True, help="a new directory")
    one.add_argument("--backend", default="codex", choices=("codex", "claude"))
    one.add_argument("--model", default="gpt-6-astra")
    one.add_argument("--effort", default="xhigh")
    one.add_argument("--service-tier", default="fast")
    one.add_argument("--max-sessions", type=int, default=4)
    one.add_argument("--attempts", type=int, default=DEFAULT_ROUND_ATTEMPTS)
    one.add_argument("--per-arm", type=int, help="a limit per arm; default: none")
    one.add_argument("--requests", type=int)
    one.add_argument("--workers", type=int, default=4)
    one.add_argument("--queue-log", type=Path, help="append start and end lines of heavy jobs")
    one.add_argument("--label", default="investigation")
    one.add_argument(
        "--scripted-request",
        type=json.loads,
        help="no model: a script requests this experiment (JSON), then submits",
    )
    one.add_argument(
        "--scripted-cause", type=json.loads, help="the target the script's diagnosis names (JSON)"
    )
    args = parser.parse_args(argv)
    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache" / "robots-probes" / "engine" / "miopen")
    )
    if args.command == "step":
        return step_main(args)
    if args.command == "rounds":
        from fleetlab.loop.scripted import scripted_from_json

        out = args.out.expanduser().resolve()
        record = run_rounds(
            args.world_root,
            out,
            agent=AgentSpec(
                args.backend, args.model, effort=args.effort, service_tier=args.service_tier
            ),
            budget=AttemptBudget(args.attempts, args.per_arm, args.requests),
            max_sessions=args.max_sessions,
            workers=args.workers,
            queue_log=args.queue_log,
            label=args.label,
            scripted=(
                scripted_from_json(json.loads(args.scripted.read_text()), out / "scripted-runs")
                if args.scripted is not None
                else None
            ),
            panel_per_robot_day=args.panel_per_robot_day,
            panel_video_indices=tuple(args.panel_video_index),
            learning=not args.no_learning,
            learned=args.learned,
            fleet_label=args.fleet_label,
            probes=args.probes,
        )
        print(
            json.dumps(
                {k: record.get(k) for k in ("outcome", "ended_because", "states", "evaluation")},
                default=str,
            )
        )
        # A deliberate stop or every round decided is a success; an engagement a round left
        # without a decision (auth, rate limit, a lost call, sessions exhausted) is not.
        return EXIT_ABORTED if record.get("outcome") == "aborted" else 0
    if args.command == "investigate":
        scripted = None
        if args.scripted_request is not None:
            scripted = ScriptedAgent(
                [request_step(args.scripted_request), submit_analysis_step(args.scripted_cause)],
                run_root=args.out.expanduser().resolve() / "agent-runs",
            )
        record = investigate(
            ObservedWorld.load(args.world_root),
            args.out,
            agent=AgentSpec(
                args.backend, args.model, effort=args.effort, service_tier=args.service_tier
            ),
            budget=AttemptBudget(args.attempts, args.per_arm, args.requests),
            max_sessions=args.max_sessions,
            workers=args.workers,
            queue_log=args.queue_log,
            label=args.label,
            scripted=scripted,
        )
        print(json.dumps({k: record[k] for k in ("states", "submitted", "costs")}, default=str))
        if record["submitted"]["diagnosis"] and record["submitted"]["fix_set"]:
            return 0
        errors = {s.get("error") for s in record.get("sessions", [])}
        return EXIT_ABORTED if errors & {"auth", "rate_limit"} else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
