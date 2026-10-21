"""The driver's `LoopState` over lane state's modules, for one world (truth side).

`WorldLoop` maps what the round driver asks (`fleetlab.loop.rounds.LoopState`) onto
`fleetlab.loop.layout`, `state`, `operate`, `experiments`, `export` and `evaluate`, with
lane state's conventions for the number of changes and the one-look plan
(`decision_changes`, `changes_in_force`, `one_look_ledger`). It runs the simulation
steps in process (`heavy=None`, the toy cell of the CPU tests) or each one as a heavy job
of its own (`HeavySteps`: `tools/heavy-run` around `python -m fleetlab.loop.driver step
...`, with the lanes' queue log), so that the driver never holds the heavy lock while the
model thinks.

The ledger versions are stored by lane state (`ledgers/ledger-r<N>.json`); the one-look
plan's ledger beside them in `one-look/`.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import EngineReply, ExperimentQueue
from fleetlab.contracts.actions import Action
from fleetlab.contracts.decisions import MAX_ADOPTED_PER_ROUND, RoundDecision
from fleetlab.loop.layout import LoopLayout
from fleetlab.loop.operate import EngineSetup
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.state import (
    MAX_CHANGES_PER_ROUND,
    Adoption,
    Ledger,
    Replacement,
    Undo,
    changes_in_force,
    decision_changes,
    one_look_ledger,
)
from fleetlab.loop.world import LoopWorld
from fleetlab.toolserver.decisions import PublicLedger, PublicLedgerEntry

__all__ = ["HeavySteps", "WorldLoop", "public_ledger_of", "setup_from_json", "setup_to_json"]

if MAX_ADOPTED_PER_ROUND != MAX_CHANGES_PER_ROUND:  # one convention (loop.state), one number
    raise ImportError("contracts.decisions and loop.state disagree on the changes per round")


def public_ledger_of(ledger: Ledger, round_number: int) -> PublicLedger:
    """The ledger as the fleet's operators know it: entries, days and status, no rationale
    and nothing on whether a change helped."""
    return PublicLedger(
        round=round_number,
        changes=[
            PublicLedgerEntry(
                entry_id=e.entry_id,
                round=e.round,
                adopted_day=e.adopted_day,
                action=e.action,
                status=e.status,
                replaced_by=e.replaced_by,
            )
            for e in ledger.entries
        ],
    )


def setup_to_json(setup: EngineSetup) -> dict[str, Any]:
    """The engine setup for a heavy step's own process: the run configuration and every
    other field of lane state's `EngineSetup`."""
    out = {f.name: getattr(setup, f.name) for f in fields(setup) if f.name != "config"}
    return {"config": asdict(setup.config), **out}


def setup_from_json(data: Mapping[str, Any]) -> EngineSetup:
    from fleetlab.engine.fleet import RunConfig

    rest = {k: v for k, v in data.items() if k != "config"}
    return EngineSetup(config=RunConfig(**data["config"]), **rest)


@dataclass
class HeavySteps:
    """Each simulation step of a loop as a heavy job (`python -m fleetlab.loop.driver step`)."""

    world_root: Path
    loop_root: Path
    setup_file: Path
    queue_log: Path | None = None
    lane: str = "driver"
    records: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        step: str,
        purpose: str,
        attempts: str,
        args: Sequence[str],
        simulated: Callable[[], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run one step as a heavy job; `simulated` tells the queue log's end line what the
        step simulated (`loop.costs.simulated_jobs`), failed simulations included."""
        from fleetlab.loop.driver import HEAVY_RUN, REPO, _commit, _queue_log, _simulated_text

        commit = _commit()
        record: dict[str, Any] = {"step": step, "purpose": purpose, "args": list(args)}
        self.records.append(record)
        command = [
            str(HEAVY_RUN),
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "fleetlab.loop.driver",
            "step",
            step,
            "--world-root",
            str(self.world_root),
            "--loop",
            str(self.loop_root),
            "--setup",
            str(self.setup_file),
            *args,
        ]
        _queue_log(self.queue_log, "queued", commit, purpose, attempts)
        started = time.perf_counter()
        try:
            subprocess.run(command, cwd=REPO, check=True)
        except BaseException as error:
            record.update(state="failed", error=type(error).__name__)
            if simulated is not None:
                record["simulated"] = dict(simulated())
            failed = f"ended-failed({type(error).__name__})"
            done = _simulated_text(record.get("simulated") or {})
            _queue_log(self.queue_log, failed, commit, purpose, done)
            raise
        finally:
            record["wall_seconds"] = time.perf_counter() - started
        record["state"] = "done"
        if simulated is not None:
            record["simulated"] = dict(simulated())
        ended = _simulated_text(record["simulated"]) if simulated is not None else attempts
        _queue_log(self.queue_log, "ended", commit, purpose, ended)
        return record


@dataclass
class HeavyRoundEngine:
    """lane state's `RoundEngine` for one round, its queue run as a heavy job; the reply is
    read back and its records kept for the cost record (`HeavyQueue`-shaped)."""

    loop: WorldLoop
    round_number: int
    ledger_file: Path
    analysis_id: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        assert self.loop.heavy is not None
        reply_file = queue_file.with_suffix(".reply.json")
        planned = sum(q.request.episodes_per_arm * len(q.request.arms) for q in queue.experiments)
        record: dict[str, Any] = {
            "queue": str(queue_file),
            "session": queue.session,
            "round": self.round_number,
            "experiments": [q.experiment_id for q in queue.experiments],
            "planned_attempts": planned,
            # Probes the agent composed (lane probe), apart from the arms' task attempts.
            "probes": [q.probe_id for q in queue.probes],
            "planned_probe_attempts": sum(q.request.attempts for q in queue.probes),
        }
        self.records.append(record)
        started = time.perf_counter()
        try:
            if not reply_file.is_file():
                self.loop.heavy.run(
                    "queue",
                    f"round-{self.round_number}-queue-session-{queue.session}",
                    f"{planned}(planned-arm-attempts)",
                    [
                        "--round",
                        str(self.round_number),
                        "--ledger",
                        str(self.ledger_file),
                        "--analysis",
                        self.analysis_id,
                        "--queue",
                        str(queue_file),
                        "--reply",
                        str(reply_file),
                    ],
                    simulated=lambda: self.loop.queue_jobs(
                        self.round_number, self.analysis_id, queue.session
                    )["simulated_all"],
                )
            reply = EngineReply.model_validate_json(reply_file.read_text())
        except BaseException as error:
            record.update(state="failed", error=type(error).__name__)
            raise
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            record.update(self.loop.queue_jobs(self.round_number, self.analysis_id, queue.session))
        record["state"] = "done"
        record["delivered_attempts"] = sum(
            len(a.episode_ids) for r in reply.results if r.status == "done" for a in r.arms
        )
        record["probe_outcomes"] = [
            {
                "probe_id": o.probe_id,
                "status": o.status,
                "attempts_executed": o.attempts_executed,
                "attempts_reserved": o.attempts_reserved,
            }
            for o in reply.probes
        ]
        record["charged_probe_attempts"] = sum(o.charged for o in reply.probes)
        return reply


@dataclass
class WorldLoop:
    """`fleetlab.loop.rounds.LoopState` for one world and one engagement."""

    world: LoopWorld
    layout: LoopLayout
    paths: LoopPaths
    setup: EngineSetup
    heavy: HeavySteps | None = None
    exports_made: int = 0
    # Attempts per robot and evaluation day (None: the world's own day length).
    panel_per_robot_day: int | None = None

    @classmethod
    def create(
        cls,
        world: LoopWorld,
        layout: LoopLayout,
        root: Path,
        setup: EngineSetup,
        heavy: HeavySteps | None = None,
        panel_per_robot_day: int | None = None,
    ) -> WorldLoop:
        paths = LoopPaths(root)
        root.mkdir(parents=True, exist_ok=True)
        paths.layout.write_text(layout.model_dump_json(indent=1) + "\n")
        (root / "setup.json").write_text(json.dumps(setup_to_json(setup), indent=1) + "\n")
        return cls(world, layout, paths, setup, heavy, panel_per_robot_day=panel_per_robot_day)

    # the chronology

    def rounds(self) -> int:
        return len(self.layout.rounds)

    def operation_days(self, round_number: int) -> tuple[int, int]:
        window = self.layout.round(round_number).ops
        return (window.first, window.last)

    def experiment_days(self, round_number: int) -> tuple[int, int]:
        window = self.layout.round(round_number).experiment
        return (window.first, window.last)

    # the ledger and its conventions (lane state's)

    def empty_ledger(self) -> Ledger:
        ledger = Ledger.empty(self.world.world_id)
        ledger.save(self.paths.ledgers)
        return ledger

    def adopt(self, ledger: Ledger, round_number: int, decision: RoundDecision) -> Ledger:
        changes = decision_changes(decision.adopt, decision.replace)
        if changes > MAX_CHANGES_PER_ROUND:
            raise ValueError(f"a decision of {changes} changes; at most {MAX_CHANGES_PER_ROUND}")
        new = ledger.adopt(
            round_number,
            self.layout.adoption_day(round_number),
            adopt=[Adoption(action=c.action, rationale=c.rationale) for c in decision.adopt],
            replace=[
                Replacement(entry_id=r.entry_id, action=r.action, rationale=r.rationale)
                for r in decision.replace
            ],
            undo=[Undo(entry_id=u.entry_id, rationale=u.rationale) for u in decision.undo],
        )
        new.save(self.paths.ledgers)
        return new

    def public_ledger(self, ledger: Ledger, round_number: int) -> PublicLedger:
        return public_ledger_of(ledger, round_number)

    def changes(self, ledger: Ledger) -> int:
        """Lane state's count of a state's changes: its entries in force at the end."""
        return changes_in_force(ledger)

    def one_look(self, actions: Sequence[Action], final: Ledger) -> Ledger:
        """Lane state's cut of round 1's list (`rounds.one_look_actions`: adoptions, then
        alternatives) at the final state's number of changes, on round 1's adoption day."""
        plan = one_look_ledger(
            self.world.world_id,
            list(actions),
            changes_in_force(final),
            self.layout.adoption_day(1),
        )
        plan.save(self.paths.root / "one-look")
        return plan

    # simulation and records

    def ledger_file(self, ledger: Ledger) -> Path:
        return ledger.save(self.paths.ledgers)

    def operate(self, round_number: int, ledger: Ledger) -> list[Path]:
        from fleetlab.loop.operate import operate

        if self.heavy is not None and round_number > 1:
            # The heavy job runs `operate` and writes the round's records; the run lies in
            # rounds/r<K>/ops (LoopPaths). Nothing simulates in this process.
            from fleetlab.loop.costs import simulated_jobs

            self.heavy.run(
                "operate",
                f"round-{round_number}-operation",
                "regular-work",
                ["--round", str(round_number), "--ledger", str(self.ledger_file(ledger))],
                simulated=lambda: simulated_jobs(self.paths.ops(round_number)),
            )
            return [self.paths.ops(round_number)]
        return [operate(self.world, self.layout, round_number, ledger, self.paths, self.setup)]

    def export(self, round_number: int, ledger: Ledger, analysis_id: str) -> Path:
        from fleetlab.loop.export import export_round

        # One analysis for the whole engagement: each earlier round's arms lie under
        # rounds/r<k>/experiments/<analysis>; this round's arms arrive with the queue.
        self.exports_made += 1
        out = self.paths.exports(round_number) / f"start-{self.exports_made:02d}"
        return export_round(
            self.world,
            self.layout,
            round_number,
            ledger,
            self.paths,
            out,
            analyses=[analysis_id] * (round_number - 1),
        )

    def engine(self, round_number: int, ledger: Ledger, analysis_id: str) -> Any:
        from fleetlab.loop.experiments import RoundEngine

        if self.heavy is not None:
            return HeavyRoundEngine(self, round_number, self.ledger_file(ledger), analysis_id)
        return RoundEngine(
            self.world,
            self.layout,
            round_number,
            ledger,
            self.paths,
            self.setup,
            earlier=[analysis_id] * (round_number - 1),
            analysis_id=analysis_id,
        )

    def queue_jobs(self, round_number: int, analysis_id: str, session: int) -> dict[str, Any]:
        """What the engine simulated for one queue (`loop.costs.simulated_jobs`): run jobs,
        chain-sample replays, confirmed failed simulations, unconfirmed assignments; for
        the arms (`simulated`) and, apart, for the queue's probe run (`simulated_probes`);
        `simulated_all` is the arms' with the probes' nested, for the queue log."""
        from fleetlab.loop.costs import simulated_jobs
        from fleetlab.loop.probes import probe_root

        run_dir = self.paths.experiments(round_number, analysis_id) / f"session-{session:02d}"
        probe_dir = probe_root(self.paths, round_number, analysis_id) / f"session-{session:02d}"
        arms, probes = simulated_jobs(run_dir), simulated_jobs(probe_dir)
        return {
            "simulated": arms,
            "simulated_probes": probes,
            "simulated_all": {**arms, "probes": probes},
        }

    def evaluate(self, states: Mapping[str, Ledger]) -> Path:
        from fleetlab.loop.evaluate import evaluate

        if self.heavy is not None:
            directory = self.paths.root / "states"
            args: list[str] = []
            for name, ledger in states.items():
                path = directory / name / ledger.file_name()
                ledger.save(path.parent)
                args += ["--state", f"{name}={path}"]
            if self.panel_per_robot_day is not None:
                args += ["--per-robot-day", str(self.panel_per_robot_day)]
            self.heavy.run(
                "evaluate",
                "evaluation-of-the-states",
                f"{len(states)}(states)",
                args,
                simulated=lambda: self.simulated()["evaluation"],
            )
        else:
            evaluate(
                self.world,
                self.layout,
                states,
                self.paths,
                self.setup,
                per_robot_day=self.panel_per_robot_day,
            )
        return self.paths.evaluation

    def simulated(self) -> dict[str, Any]:
        """Every job this loop simulated, by kind (the demonstration's compute beside the
        agent's experiments): operation runs, arm runs, evaluation runs, each with its
        finished run jobs and replays, confirmed failed simulations and unconfirmed
        assignments (`loop.costs.simulated_jobs`)."""
        from fleetlab.loop.costs import simulated_jobs

        keys = ("run_jobs", "replays", "partial_jobs", "failed_confirmed", "unconfirmed")

        def count(run_dirs: Sequence[Path]) -> dict[str, int]:
            each = [simulated_jobs(d) for d in run_dirs if d.is_dir()]
            totals = {k: sum(int(e[k]) for e in each) for k in keys}
            totals["simulated_confirmed"] = sum(int(e["simulated_confirmed"]) for e in each)
            return totals

        numbers = range(1, self.rounds() + 1)
        arms = [
            d
            for n in numbers
            for d in sorted((self.paths.round(n) / "experiments").glob("*/session-*"))
            if d.is_dir()
        ]
        # Probes the agent composed (lane probe): their runs, apart from the arms'.
        probes = [
            d
            for n in numbers
            for d in sorted((self.paths.round(n) / "experiments").glob("*/probes/session-*"))
            if d.is_dir()
        ]
        evaluation = self.paths.evaluation
        states = sorted(d for d in evaluation.glob("*") if d.is_dir())
        return {
            # round 1's operation is the world's stored observed run: nothing simulated
            "operation": count([self.paths.ops(n) for n in numbers if n > 1]),
            "experiments": count(arms),
            "probes": count(probes),
            "evaluation": count(states),
        }

    # the audit's vocabulary

    def audit_runs(self) -> tuple[Path, ...]:
        from fleetlab.agents.audit import completed_runs

        # The world's own stored runs (their canaries and truth vocabulary) and every
        # finished run of the loop so far (operation and arms), as lane state's export does.
        rounds = self.paths.root / "rounds"
        found = completed_runs(rounds) if rounds.is_dir() else ()
        return tuple(dict.fromkeys((*self.world.runs, *found)))

    def experiment_root(self, analysis_id: str) -> Path:
        return self.paths.root / "rounds"
