"""The engine's side of the runtime's experiment queue (contracts, section 12).

An analysis that requested experiments ends its session "waiting", and the
runtime hands the engine an `ExperimentQueue` (`fleetlab.agents.sessions`).
`run_queue` answers it:

1. **Plan.** For every queued experiment and every arm, the arm's episodes
   are minted in the experiment window's seed stream: a robot of the request
   and a day of `windows.experiment` per slot (robots in turn, then days), a
   fresh episode index per robot and day after the as-run ones (n, n+1, ...)
   and a fresh neutral episode id, both kept in the analysis' own
   `experiments/<analysis>/allocations.json` so that no later session of it
   reuses one, seeds from `seeds_for`, the task and the
   start state drawn from (world seed, experiment id, slot), shared by the arms
   when `paired_starts`. The identity carries `experiment_id` and, as its
   `faults`, the arm's post-action fault list: the faults in force for that
   robot on that day with the arm's actions resolved against them
   (`contracts.actions.resolve_all`), removed faults dropped and added changes
   appended. A run job carries no `remove` or `add`. Both arms are always
   simulated, also when an action targets something healthy. A request the
   world cannot run (an unknown robot, an action that cannot be resolved) is
   answered "refused" with the reason, and its episodes are not minted.
2. **Run.** The minted jobs go through `fleet.run` in their own run
   directory, `experiments/<analysis>/session-NN/`, which is resumable: the
   plan is written before the run, and a queue answered again reads it back
   instead of minting anew.
3. **Export.** The analysis' next export is the log's `export_runs` over the
   world's base runs and every experiment run of this analysis so far, in
   `exports/<analysis>/session-NN/`.
4. **Reply.** One `ExperimentResult` per experiment with each arm's episode
   ids, and the new export (`EngineReply`); the runtime checks that every arm
   episode is in it with its experiment id (`arm_problems`).

The world comes as a `WorldTruth` (truth side, never mounted), which the
world generator writes.

Choices where the contracts are silent:

- Arm episodes take the episode indices after the day's as-run ones and are
  written with the world's own day length: `started_at_for` puts them from
  16:00 every three minutes, 160 per robot and day, after the
  day's regular work. A request whose arm episodes would start after 23:59:59
  on some robot and day is refused with the public code `exceeds_robot_day_capacity`,
  checked before minting from the request and the analysis'
  allocations alone, so that the reason can be shown: the capacity is layout.
- Allocations are per analysis: analyses are alternative histories of one
  world, and an analysis' export holds only its own experiments, so each
  restarts the indices after the as-run episodes and the episode ids at the
  world's first free number. Episode ids are unique within an analysis, which
  is what its export needs; two analyses may use the same ids.
- Tasks come from the robot's own list when the world gives one
  (`robot_tasks`), else from the world's.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from fleetlab.contracts._base import ConfigValue, ContractModel
from fleetlab.contracts.actions import Action, resolve_all
from fleetlab.contracts.faults import FaultSpec, active_faults
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.identity import (
    ARM_SLOTS,
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    Task,
    seeds_for,
    started_at_for,
)
from fleetlab.contracts.ids import FaultId, IdKind, RobotId, WorldId, id_number, neutral_id
from fleetlab.contracts.model import Fleet, effective_config
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.tools import ArmResult, ExperimentResult
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey
from fleetlab.contracts.world import Windows
from fleetlab.engine.factories import resolve
from fleetlab.engine.manifests import MAX_STEPS
from fleetlab.toolserver.public_results import (
    COMPLETED,
    OVER_CAPACITY,
    REFUSED,
    request_results_problems,
)

__all__ = [
    "QueueEngine",
    "WorldTruth",
    "allocations_file",
    "plan_queue",
    "run_queue",
]

_DRAW_SCHEME = 1

# (the arm's identity before its actions, the arm's actions) -> the attempt's faults.
ArmFaults = Callable[[EpisodeIdentity, Sequence[Action]], Sequence[FaultSpec]]


class WorldTruth(ContractModel):
    """What the engine needs to know about a world to run its experiments (truth side)."""

    world_id: WorldId
    world_seed: int = Field(ge=0)
    fleet: Fleet  # the system model as the experiment window starts
    windows: Windows
    fault_schedule: list[FaultSpec]
    fault_levels: dict[FaultId, dict[int, float]] = Field(default_factory=dict)  # drift, per day
    episodes_per_robot_day: int = Field(ge=1)  # of the as-run world
    tasks: list[Task]
    robot_tasks: dict[RobotId, list[Task]] = Field(default_factory=dict)
    task_classes: list[str] = Field(default_factory=list)
    start_states: int = Field(default=50, ge=1)
    policy: PolicySpec
    image_size: int = Field(default=360, ge=1)
    max_steps: dict[str, int] = Field(default_factory=dict)  # per suite; else LIBERO's
    base_runs: list[str]  # the finalized runs the analyses' first export was built from
    first_free_episode: int = Field(ge=1)

    @classmethod
    def load(cls, path: Path) -> WorldTruth:
        return cls.model_validate_json(path.read_text())

    def faults_on(self, robot_id: str, day: int) -> list[FaultSpec]:
        """The faults in force for a robot on a day, each at its level of that day."""
        out = []
        for fault in active_faults(self.fault_schedule, self.fleet, robot_id, day):
            level = self.fault_levels.get(fault.fault_id, {}).get(day)
            out.append(fault if level is None else fault.model_copy(update={"level": level}))
        return [f for f in out if f.is_fault]

    def current(self, robot_id: str) -> dict[ConfigKey | ConditionKey, ConfigValue]:
        """The robot's configuration values and its site's conditions (`resolve`'s `current`)."""
        robot = self.fleet.robot(robot_id)
        values: dict[ConfigKey | ConditionKey, ConfigValue] = {}
        for key, value in effective_config(self.fleet, robot_id).items():
            values[key] = value
        for condition, setting in self.fleet.site(robot.site_id).conditions.items():
            values[condition] = setting
        return values

    def tasks_of(self, robot_id: str) -> list[Task]:
        return list(self.robot_tasks.get(robot_id) or self.tasks)

    def steps_for(self, suite: str) -> int:
        return self.max_steps.get(suite) or MAX_STEPS.get(suite, 280)


@dataclass(frozen=True)
class Planned:
    """What a queue becomes: the jobs to run and the answer to give."""

    jobs: list[Job]
    results: list[ExperimentResult]


def _draw(world_seed: int, experiment_id: str, stream: str, slot: int) -> np.random.Generator:
    entropy = [_DRAW_SCHEME, world_seed, id_number(IdKind.experiment, experiment_id), slot]
    entropy.append(int.from_bytes(stream.encode(), "big") % 2**32)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _place(robots: Sequence[str], days: Sequence[int], slot: int) -> tuple[str, int]:
    """The robot and day of a slot: robots in turn, then days."""
    return robots[slot % len(robots)], days[(slot // len(robots)) % len(days)]


def _over_capacity(
    request: Any, days: Sequence[int], next_index: Mapping[str, int], per_day: int
) -> bool:
    """Whether the request's arms would pass a robot-day's `ARM_SLOTS`, counting what
    the analysis' earlier experiments and probes (`loop.probes`, the same `next_index`)
    already hold there; public layout, no truth."""
    robots = list(request.robots)
    if not robots or not days:
        return False
    wanted: dict[str, int] = {}
    for slot in range(request.episodes_per_arm):
        robot_id, day = _place(robots, days, slot)
        key = f"{robot_id}/{day}"
        wanted[key] = wanted.get(key, 0) + len(request.arms)
    return any(
        int(next_index.get(key, per_day)) - per_day + count > ARM_SLOTS
        for key, count in wanted.items()
    )


@contextmanager
def _locked(path: Path) -> Iterator[dict[str, Any]]:
    """An analysis' allocations, read and written under a lock (one engine at a time)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
        yield state
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state, indent=1, sort_keys=True))
        temporary.replace(path)


def plan_queue(
    queue: Any,
    world: WorldTruth,
    allocations: dict[str, Any],
    *,
    software: SoftwareVersion,
    batch_size: int,
    truth: Mapping[str, Any],
    record_video: bool = True,
    public_model: Fleet | None = None,
    days: Sequence[int] | None = None,
    arm_faults: ArmFaults | None = None,
) -> Planned:
    """Mint every arm episode of the queue; `allocations` is updated in place.

    `days` are the days the arms run on (default: the world's experiment window);
    `arm_faults(probe, actions)` gives an arm attempt's faults (default: the world's
    faults in force with the arm's actions resolved against them, `resolve_all`). The
    loop's rounds pass their own experiment days and the ledger's state with the arm as a
    delta (`fleetlab.loop.experiments`).
    """
    if queue.world_id != world.world_id:
        raise ValueError(f"the queue is for world {queue.world_id}, not {world.world_id}")
    next_episode = int(allocations.get("next_episode", world.first_free_episode))
    next_index: dict[str, int] = dict(allocations.get("next_index", {}))
    if days is None:
        days = list(range(world.windows.experiment.first, world.windows.experiment.last + 1))
    days = list(days)
    jobs: list[Job] = []
    results: list[ExperimentResult] = []
    from fleetlab.toolserver.action_validation import request_problems, request_refusal_code

    for queued in queue.experiments:
        experiment_id, request = queued.experiment_id, queued.request
        if problems := request_problems(request, public_model or world.fleet):
            logging.getLogger(__name__).warning("experiment %s: %s", experiment_id, problems)
            results.append(
                ExperimentResult(
                    experiment_id=experiment_id,
                    status="refused",
                    arms=[],
                    message=request_refusal_code(problems),
                )
            )
            continue
        if _over_capacity(request, days, next_index, world.episodes_per_robot_day):
            logging.getLogger(__name__).warning(
                "experiment %s: more than %d arm episodes on a robot-day", experiment_id, ARM_SLOTS
            )
            results.append(
                ExperimentResult(
                    experiment_id=experiment_id, status="refused", arms=[], message=OVER_CAPACITY
                )
            )
            continue
        try:
            minted = _mint(
                world,
                experiment_id,
                request,
                days,
                next_episode,
                next_index,
                software=software,
                batch_size=batch_size,
                truth=truth,
                arm_faults=arm_faults,
            )
        except (KeyError, ValueError):
            # The exception can name a hidden fault kind, trait or schedule. Its
            # traceback stays in the host job log, never in an agent-visible reply.
            logging.getLogger(__name__).exception("experiment %s refused", experiment_id)
            results.append(
                ExperimentResult(
                    experiment_id=experiment_id, status="refused", arms=[], message=REFUSED
                )
            )
            continue
        identities, next_episode, next_index = minted
        arms = [
            ArmResult(arm_index=a, episode_ids=[i.episode_id for i in ids])
            for a, ids in enumerate(identities)
        ]
        n = request.episodes_per_arm
        results.append(
            ExperimentResult(
                experiment_id=experiment_id,
                status="done",
                arms=arms,
                message=COMPLETED,
            )
        )
        for slot in range(n):  # the arms interleaved, slot by slot
            for ids in identities:
                jobs.append(Job(identity=ids[slot], record_video=record_video))
    allocations["next_episode"] = next_episode
    allocations["next_index"] = next_index
    return Planned(jobs=jobs, results=results)


def _mint(
    world: WorldTruth,
    experiment_id: str,
    request: Any,
    days: Sequence[int],
    next_episode: int,
    next_index: dict[str, int],
    *,
    software: SoftwareVersion,
    batch_size: int,
    truth: Mapping[str, Any],
    arm_faults: ArmFaults | None = None,
) -> tuple[list[list[EpisodeIdentity]], int, dict[str, int]]:
    """The arms' identities, or an error that refuses the request (nothing is allocated then)."""
    index = dict(next_index)
    episode = next_episode
    robots = list(request.robots)
    for robot_id in robots:
        world.fleet.robot(robot_id)  # KeyError for a robot the world does not have
    arms: list[list[EpisodeIdentity]] = [[] for _ in request.arms]
    for slot in range(request.episodes_per_arm):
        robot_id, day = _place(robots, days, slot)
        site_id = world.fleet.robot(robot_id).site_id
        in_force = world.faults_on(robot_id, day)
        tasks = world.tasks_of(robot_id)
        for arm_index, arm in enumerate(request.arms):
            stream = "paired" if request.paired_starts else f"arm{arm_index}"
            rng = _draw(world.world_seed, experiment_id, stream, slot)
            task = tasks[int(rng.integers(len(tasks)))]
            start_state = int(rng.integers(world.start_states))
            key = f"{robot_id}/{day}"
            episode_index = index.get(key, world.episodes_per_robot_day)
            index[key] = episode_index + 1
            scene_seed, noise_seed = seeds_for(world.world_seed, robot_id, day, episode_index)
            probe = EpisodeIdentity(  # the identity before the arm's actions, to resolve them
                world_id=world.world_id,
                episode_id=neutral_id(IdKind.episode, episode),
                robot_id=robot_id,
                site_id=site_id,
                day=day,
                episode_index=episode_index,
                suite=task.suite,
                task_index=task.task_index,
                start_state=start_state,
                scene_seed=scene_seed,
                noise_seed=noise_seed,
                policy=world.policy,
                batch_size=batch_size,
                row=None,
                image_size=world.image_size,
                max_steps=world.steps_for(task.suite),
                faults=in_force,
                software=software,
                experiment_id=experiment_id,
            )
            # Refuses an index whose start time would pass 23:59:59; the
            # public check `_over_capacity` refuses such a request first.
            started_at_for(probe, world.episodes_per_robot_day)
            if arm_faults is None:
                change = resolve_all(arm.actions, probe, in_force, truth, world.current(robot_id))
                faults = [f for f in in_force if f.fault_id not in change.removed]
                faults += list(change.added)
            else:
                faults = list(arm_faults(probe, arm.actions))
            arms[arm_index].append(probe.model_copy(update={"faults": faults}))
            arms[arm_index][-1] = EpisodeIdentity.model_validate(
                arms[arm_index][-1].model_dump()
            )  # the validators again, on the post-action fault list
            episode += 1
    return arms, episode, index


def allocations_file(world_dir: Path, analysis_id: str) -> Path:
    """The analysis' own allocations: episode ids and per robot and day the next index."""
    return world_dir / "experiments" / analysis_id / "allocations.json"


def queue_dirs(world_dir: Path, analysis_id: str, session: int) -> tuple[Path, Path]:
    """The run directory and the export directory of one queue."""
    name = f"session-{session:02d}"
    return (
        world_dir / "experiments" / analysis_id / name,
        world_dir / "exports" / analysis_id / name,
    )


def run_queue(
    queue_path: Path,
    world: WorldTruth,
    world_dir: Path,
    *,
    config: Any,
    export: bool = True,
    record_video: bool = True,
    timings: dict[str, float] | None = None,
) -> Any:
    """Run every experiment of the queue at `queue_path` and answer with an `EngineReply`.

    `config` is a `fleet.RunConfig` for the runs (workers, batch size,
    precision, check, fault registry); its `out` and day length are set here.
    Answering a queue again resumes it: the plan is read back, finished jobs
    are skipped.
    """
    from fleetlab.agents.sessions import EngineReply, ExperimentQueue
    from fleetlab.engine import fleet
    from fleetlab.log import LogReader
    from fleetlab.toolserver.action_validation import (
        public_fleet,
        request_problems,
        request_refusal_code,
    )

    queue = ExperimentQueue.model_validate_json(Path(queue_path).read_text())
    run_dir, export_dir = queue_dirs(world_dir, queue.analysis_id, queue.session)
    plan_file = run_dir.parent / f"{run_dir.name}.plan.json"
    config = replace(
        config,
        out=str(run_dir),
        episodes_per_robot_day=world.episodes_per_robot_day,
        allow_manifest_day_length=False,
    )
    phases = timings if timings is not None else {}
    started = time.perf_counter()
    software = fleet.prepare(config)
    phases["prepare_seconds"] = time.perf_counter() - started
    if plan_file.is_file():
        saved = json.loads(plan_file.read_text())
        if saved.get("queue_sha256") != hash_json(queue.model_dump(mode="json")):
            raise ValueError("saved experiment plan does not match this queue")
        jobs = [Job.model_validate(j) for j in saved["jobs"]]
        results = [ExperimentResult.model_validate(r) for r in saved["results"]]
        if problems := request_results_problems(
            results, {q.experiment_id: q.request for q in queue.experiments}
        ):
            raise ValueError("unsafe experiment plan results: " + "; ".join(problems))
        with LogReader(queue.export) as reader:
            public = public_fleet(reader.description())
        # A saved plan from an earlier runner cannot bypass current public preflight.
        invalid = {
            q.experiment_id: request_refusal_code(problems)
            for q in queue.experiments
            if (problems := request_problems(q.request, public))
        }
        jobs = [job for job in jobs if job.identity.experiment_id not in invalid]
        results = [
            ExperimentResult(
                experiment_id=result.experiment_id,
                status="refused",
                arms=[],
                message=invalid[result.experiment_id],
            )
            if result.experiment_id in invalid
            else result
            for result in results
        ]
    else:
        with LogReader(queue.export) as reader:
            public = public_fleet(reader.description())
        truth = _truth_table(config.fault_registry)
        with _locked(allocations_file(world_dir, queue.analysis_id)) as allocations:
            planned = plan_queue(
                queue,
                world,
                allocations,
                software=software,
                batch_size=config.batch_size,
                truth=truth,
                record_video=record_video,
                public_model=public,
            )
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            plan_file.write_text(
                json.dumps(
                    {
                        "queue": str(queue_path),
                        "queue_sha256": hash_json(queue.model_dump(mode="json")),
                        "jobs": [j.model_dump(mode="json") for j in planned.jobs],
                        "results": [r.model_dump(mode="json") for r in planned.results],
                    }
                )
            )
        jobs, results = planned.jobs, planned.results
    if problems := request_results_problems(
        results, {q.experiment_id: q.request for q in queue.experiments}
    ):
        raise ValueError("unsafe experiment plan results: " + "; ".join(problems))
    started = time.perf_counter()
    if jobs:
        try:
            # Machine/worker failures are not request refusals. Propagate them
            # without publishing a reply; the saved plan and run checkpoint
            # remain available to a continuation of this analysis.
            fleet.run(config, jobs)
        finally:
            phases["simulation_seconds"] = time.perf_counter() - started
    else:
        phases["simulation_seconds"] = 0.0
    new_export = Path(queue.export)
    started = time.perf_counter()
    if export:
        new_export = export_for(world, world_dir, queue.analysis_id, export_dir)
    phases["export_seconds"] = time.perf_counter() - started
    return EngineReply(results=results, export=new_export)


def _truth_table(registry_path: str) -> dict[str, Any]:
    import importlib

    module = importlib.import_module(registry_path.partition(":")[0])
    load_all = getattr(module, "load_all", None)
    if callable(load_all):
        load_all()
    table: dict[str, Any] = resolve(registry_path).truth_table()
    return table


def export_for(world: WorldTruth, world_dir: Path, analysis_id: str, export_dir: Path) -> Path:
    """The analysis' export: the world's base runs and every experiment run of the analysis."""
    from fleetlab.log import export_runs, run_status

    runs = [Path(r) for r in world.base_runs]
    runs += sorted(
        d
        for d in (world_dir / "experiments" / analysis_id).glob("session-*")
        if d.is_dir() and run_status(d).finalized
    )
    export_runs(
        runs, export_dir, fleet=world.fleet, task_classes=world.task_classes, overwrite=True
    )
    return export_dir


@dataclass
class QueueEngine:
    """The runtime's `Engine` (`fleetlab.agents.sessions.Engine`) for one world."""

    world: WorldTruth
    world_dir: Path
    config: Any  # a fleet.RunConfig

    def run_queue(self, queue_file: Path, queue: Any) -> Any:
        return run_queue(queue_file, self.world, self.world_dir, config=self.config)


def scripted_analysis(
    analysis_base: Path,
    world: WorldTruth,
    export: Path,
    engine: QueueEngine,
    request: Mapping[str, Any],
    tool_env: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """An analysis whose agent is a script: session 1 requests `request` and waits, session 2
    reads the results and ends. The runtime's `run_analysis(check_arms=True)` drives it, so
    the engine's reply goes through the runtime's checks; no model is called.

    Returns the analysis record and what the script saw through the tools.
    """
    from unittest import mock

    from fleetlab.agents.run import AgentResult, ToolAccess
    from fleetlab.agents.sessions import AgentSpec, Analysis, run_analysis
    from fleetlab.toolserver.server import Paths, ToolBox

    seen: list[dict[str, Any]] = []

    def script(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            ),
            session=tools.session,
        )
        try:
            if tools.session == 1:
                receipt = box.call("request_experiment", dict(request)).content[0]["text"]
                seen.append({"session": 1, "receipt": json.loads(receipt)})
                report = {"status": "waiting", "summary": "requested one experiment"}
            else:
                results = json.loads(box.call("experiment_results", {}).content[0]["text"])
                rows = box.call(
                    "query",
                    {
                        "sql": "select experiment_id, count(*) as n, "
                        "sum((outcome = 'success')::int) as ok from episodes "
                        "where experiment_id is not null group by experiment_id"
                    },
                ).content[0]["text"]
                seen.append({"session": 2, "results": results, "rows": json.loads(rows)})
                report = {"status": "done", "summary": "read the results"}
        finally:
            box.close()
        return AgentResult(ok=True, result=report, usage={}, seconds=0.0)

    analysis = Analysis.create(world.world_id, base=analysis_base)
    with mock.patch("fleetlab.agents.sessions.run_agent", script):
        record = run_analysis(
            analysis,
            export,
            AgentSpec("claude", "haiku"),
            "a scripted session",
            "request one experiment, then read it",
            tool_env,
            engine,
            max_sessions=2,
            check_arms=True,
        )
    return record, seen
