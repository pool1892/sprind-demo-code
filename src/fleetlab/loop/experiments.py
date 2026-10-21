"""Experiments from the current state (docs/design/loop.md, section 4).

    engine = RoundEngine(world, layout, round_number=2, ledger=ledger_r1, paths=paths,
                         setup=setup)
    reply = engine.run_queue(queue_file, queue)      # the runtime's Engine protocol

The runtime hands the engine an `ExperimentQueue` after a session ended "waiting"
(`fleetlab.agents.sessions`). For a round of the loop:

1. **Preflight.** Every request is checked against the fleet as the round's export shows
   it (the public model after the ledger): `toolserver.action_validation.request_problems`,
   which also refuses an action outside the current catalogue (`not_in_catalogue`). A
   refused request costs nothing and is answered with its fixed public code.
2. **Plan.** The engine's own planner (`fleetlab.engine.experiments.plan_queue`) mints the
   arms on the round's experiment days, with one change: each arm attempt's faults are
   the ledger's effective state for that robot and day with the arm's actions folded on
   top as a delta (`LoopWorld.state_on(robot, day, ledger, arm)`). The control arm (no
   actions) is the current state itself, so a retained change is in force in both arms.
   Allocations are kept per analysis and round
   (`rounds/r<K>/experiments/<analysis>/allocations.json`); a round's episode ids start
   after every id an earlier round of the loop allocated, so one export can hold them all.
3. **Run.** Both arms are always simulated, through `fleet.run`, in
   `rounds/r<K>/experiments/<analysis>/session-NN/`, resumable from the saved plan.
   The queue's probes (`contracts.probes`) run after the arms, from the same current
   state and allocations, in a probe run of their own (`fleetlab.loop.probes`).
4. **Export and reply.** The reply's export is the round's export with every arm and
   probe so far (`fleetlab.loop.export.export_round`), leak-checked before it is
   returned; the reply has one outcome per probe.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.loop.export import export_round
from fleetlab.loop.layout import LoopLayout, days_of
from fleetlab.loop.operate import EngineSetup, _complete, add_events
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.probes import run_probes
from fleetlab.loop.state import Ledger
from fleetlab.loop.world import LoopWorld

__all__ = ["RoundEngine", "arm_faults_for", "plan_round_queue"]


def arm_faults_for(world: LoopWorld, ledger: Ledger) -> Any:
    """The planner's hook: (probe identity, arm actions) -> the attempt's faults, the
    ledger's state for the robot and day with the arm folded on top."""

    def arm_faults(probe: EpisodeIdentity, actions: Sequence[Action]) -> list[FaultSpec]:
        state = world.state_on(probe.robot_id, probe.day, ledger, actions)
        return list(state.faults)

    return arm_faults


def plan_round_queue(
    queue: Any,
    world: LoopWorld,
    layout: LoopLayout,
    round_number: int,
    ledger: Ledger,
    allocations: dict[str, Any],
    *,
    software: Any,
    batch_size: int,
    public: Any,
    record_video: bool = True,
) -> Any:
    """`fleetlab.engine.experiments.plan_queue` on the round's experiment days with the
    ledger's state; `allocations` is updated in place. Returns the engine's `Planned`."""
    from fleetlab.engine.experiments import plan_queue

    return plan_queue(
        queue,
        world.truth,
        allocations,
        software=software,
        batch_size=batch_size,
        truth=world.truth_table(),
        record_video=record_video,
        public_model=public,
        days=days_of(layout.round(round_number).experiment),
        arm_faults=arm_faults_for(world, ledger),
    )


@dataclass
class RoundEngine:
    """The runtime's `Engine` for one round of a loop on one world.

    `earlier` names the analysis of each earlier round, in round order (an empty string
    for a round without one), so that the round's export holds their arms too. One
    analysis per round: a second analysis id in the same engine is refused.
    """

    world: LoopWorld
    layout: LoopLayout
    round_number: int
    ledger: Ledger  # the ledger in force during the round (version round - 1)
    paths: LoopPaths
    setup: EngineSetup
    earlier: Sequence[str] = ()
    analysis_id: str | None = None

    def first_episode(self) -> int:
        """The first episode number this round may allocate: after every number the
        world and every earlier round of the loop allocated."""
        numbers = [self.world.truth.first_free_episode]
        for number in range(1, self.round_number):
            root = self.paths.round(number) / "experiments"
            for path in root.glob("*/allocations.json"):
                numbers.append(int(json.loads(path.read_text()).get("next_episode", 0)))
        return max(numbers)

    def export_dir(self, analysis_id: str, session: int) -> Path:
        return self.paths.exports(self.round_number) / f"{analysis_id}-session-{session:02d}"

    def run_queue(self, queue_file: Path, queue: Any) -> Any:
        """Answer the queue with an `EngineReply` (results and the round's new export)."""
        from fleetlab.agents.sessions import EngineReply, ExperimentQueue
        from fleetlab.contracts.rollout import Job
        from fleetlab.contracts.tools import ExperimentResult
        from fleetlab.engine import fleet
        from fleetlab.engine.experiments import _locked
        from fleetlab.log import LogReader
        from fleetlab.toolserver.action_validation import public_fleet
        from fleetlab.toolserver.public_results import request_results_problems

        if not isinstance(queue, ExperimentQueue):
            queue = ExperimentQueue.model_validate_json(Path(queue_file).read_text())
        if queue.world_id != self.world.world_id:
            raise ValueError(f"the queue is for world {queue.world_id}, not {self.world.world_id}")
        if self.analysis_id is None:
            self.analysis_id = queue.analysis_id
        elif queue.analysis_id != self.analysis_id:
            raise ValueError(f"round {self.round_number} runs analysis {self.analysis_id} only")
        root = self.paths.experiments(self.round_number, queue.analysis_id)
        name = f"session-{queue.session:02d}"
        run_dir, plan_file = root / name, root / f"{name}.plan.json"
        software = fleet.prepare(self.setup.config)
        digest = hash_json(queue.model_dump(mode="json"))
        if plan_file.is_file():
            saved = json.loads(plan_file.read_text())
            if saved.get("queue_sha256") != digest:
                raise ValueError("saved experiment plan does not match this queue")
            jobs = [Job.model_validate(j) for j in saved["jobs"]]
            results = [ExperimentResult.model_validate(r) for r in saved["results"]]
        else:
            with LogReader(queue.export) as reader:
                public = public_fleet(reader.description())
            with _locked(root / "allocations.json") as allocations:
                allocations.setdefault("next_episode", self.first_episode())
                planned = plan_round_queue(
                    queue,
                    self.world,
                    self.layout,
                    self.round_number,
                    self.ledger,
                    allocations,
                    software=software,
                    batch_size=self.setup.config.batch_size,
                    public=public,
                    record_video=self.setup.video,
                )
                plan_file.parent.mkdir(parents=True, exist_ok=True)
                plan_file.write_text(
                    json.dumps(
                        {
                            "queue": str(queue_file),
                            "queue_sha256": digest,
                            "round": self.round_number,
                            "ledger_version": self.ledger.version,
                            "jobs": [j.model_dump(mode="json") for j in planned.jobs],
                            "results": [r.model_dump(mode="json") for r in planned.results],
                        }
                    )
                )
            jobs, results = planned.jobs, planned.results
        requests = {q.experiment_id: q.request for q in queue.experiments}
        if problems := request_results_problems(results, requests):
            raise ValueError("unsafe experiment plan results: " + "; ".join(problems))
        if jobs:
            if not _complete(run_dir, jobs):
                fleet.run(
                    self.setup.for_run(run_dir, self.world.truth.episodes_per_robot_day), jobs
                )
            events = self.world.event_setup(self.setup)
            if events.events:
                add_events(run_dir, events)
        probes = run_probes(
            self.world,
            self.layout,
            self.round_number,
            self.ledger,
            self.paths,
            self.setup,
            queue.analysis_id,
            queue.session,
            queue.probes,
        )
        export = export_round(
            self.world,
            self.layout,
            self.round_number,
            self.ledger,
            self.paths,
            self.export_dir(queue.analysis_id, queue.session),
            analyses=self.analyses(queue.analysis_id),
        )
        return EngineReply(results=results, export=export, probes=probes)

    def analyses(self, current: str) -> list[str]:
        """The analysis of every round up to this one, by round: `earlier` padded with
        empty names for rounds without one, then this round's."""
        earlier = list(self.earlier)
        if len(earlier) > self.round_number - 1:
            raise ValueError(
                f"round {self.round_number} has {self.round_number - 1} earlier rounds, "
                f"not {len(earlier)}"
            )
        return [*earlier, *[""] * (self.round_number - 1 - len(earlier)), current]


def main(argv: Sequence[str] | None = None) -> int:
    """Answer one queue of a round, for a caller that runs it under `tools/heavy-run`:

        tools/heavy-run uv run --locked python -m fleetlab.loop.experiments \
            --world ROOT --loop DIR --round 2 --ledger DIR/ledgers/ledger-r1.json \
            --queue Q.json --reply R.json [--earlier ANALYSIS_R1] [--workers 6]
    """
    import argparse

    from fleetlab.engine.fleet import RunConfig
    from fleetlab.loop.layout import LoopLayout as Layout

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.experiments")
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--reply", type=Path, required=True)
    parser.add_argument("--earlier", action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-contacts", action="store_true")
    args = parser.parse_args(argv)
    paths = LoopPaths(args.loop)
    world = LoopWorld.load(args.world)
    layout = Layout.model_validate_json(paths.layout.read_text())
    config = RunConfig(out="", batch_size=4, workers=args.workers, precision="fp16_res32")
    engine = RoundEngine(
        world,
        layout,
        args.round,
        Ledger.load(args.ledger),
        paths,
        EngineSetup(config=config, contacts=not args.no_contacts),
        earlier=args.earlier,
    )
    reply = engine.run_queue(args.queue, None)
    args.reply.write_text(reply.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
