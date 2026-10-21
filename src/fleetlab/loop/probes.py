"""Probes in the loop: the probes an analysis composed, planned, run and exported
(docs/design/loop.md, section 10; `contracts.probes`).

    outcomes = run_probes(world, layout, 2, ledger_r1, paths, setup, "a1", 3, queued)
    write_probe_tables(stage, probe_runs(paths, 2, ["a0", "a1"]))   # inside the export

A probe episode is an ordinary episode of the engine with two differences: its policy is
the probe player (`fleetlab.engine.probe_player`), an in-worker policy that plays the
program carried by the identity (`EpisodeIdentity.probe`), and it has no task to do (the
scene is the robot's first task's, untouched by the motion). Everything else is an
experiment arm's:

- it runs on the round's experiment days, on the robot as it currently is: its faults
  are the ledger's effective state for the robot and day (`LoopWorld.state_on`), with
  every adopted change and every lasting condition in force, and nothing added or taken
  away by the probe;
- its episode id and its index of the day come from the analysis' allocations of the
  round (`allocations.json`), under the same lock as the arms', so ids never collide;
- it goes through the normal engine and log writer (`fleetlab.engine.fleet.run`), with
  its chain, in a run of its own (`rounds/r<K>/experiments/<analysis>/probes/session-NN`);
  the run serves no policy (`NoServerPolicy`) and runs on the CPU, so it is light work.

A probe is refused, and costs nothing, when a robot it names is not in the world or its
attempt cannot be run as a probe (a configuration that changes the policy's precision).
A probe run that breaks off reports each of its probes failed: its partial episodes are
never exported (the run is not finalized) and never named, and its whole request stays
charged, split into the episodes the run left a record of and a reserved rest whose
fate the runner cannot confirm.

The export reads the probe runs' visible steps and the plan beside them and writes the
probe tables (`write_probe_tables`): `probe_episodes`, `probe_segments` and
`probe_steps`. They hold the commanded target (the start position the robot reported
plus the program's offset), the gripper command, and what the robot measured. Nothing in
them comes from the truth side: no fault, no configured value, no executed command.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.contracts.identity import EpisodeIdentity, seeds_for
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.log import SCHEMA, export_table_markdown
from fleetlab.contracts.probes import (
    ProbeOutcome,
    ProbeProgram,
    QueuedProbe,
    program_trajectory,
)
from fleetlab.contracts.rollout import Job
from fleetlab.loop.layout import LoopLayout, days_of
from fleetlab.loop.operate import EngineSetup, _complete
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.state import Ledger
from fleetlab.loop.world import LoopWorld, hook_chain

__all__ = [
    "PLAN_FILE",
    "PROBE_DICTIONARY",
    "PROBE_TASK",
    "PROBE_TOOLS",
    "PlannedProbes",
    "plan_probes",
    "probe_check",
    "probe_config",
    "probe_root",
    "probe_runs",
    "probe_tool_texts",
    "recipe_check",
    "run_evidence",
    "run_probes",
    "step_lags",
    "with_probes",
    "write_probe_tables",
]

PLAN_FILE = "probe_plan.json"  # beside a probe run's log/ (host side, never exported)
FAILED_FILE = "probe_failed.json"  # beside it: the run broke off and was reported failed
PRECISION = "fp32"  # the player computes nothing; the run loads no model
PROBE_DICTIONARY = """## Probes

- A probe is a short scripted motion you composed with `request_probe`, played on one
  robot without the policy and without a task; one row of `probe_episodes` is one
  playing of it (`repeat` counts them per robot). Probe episodes are never in
  `episodes` or `steps`.
- Timing of `probe_steps`: row k is one instant, k control steps (`time_s` = k / 20 s)
  after the probe started. At that instant the state is measured first (`eef_pos`,
  `eef_quat`, `joint_pos`, `joint_vel`, `gripper_opening`) and then command k is issued
  (`target_pos`, `gripper_command`), as the probe submitted it, before anything on the
  way to the arm acts on it. A command acts during the next control step: with nothing
  in between, it shows first in the measurement of row k + 1. The last row, at the
  program's step count N, is the state after the last command and has none.
- `target_pos` is the position the robot reported in row 0 plus the program's offset;
  `segment` and `probe_segments.first_step` tie each command to its primitive.
"""


def probe_root(paths: LoopPaths, round_number: int, analysis_id: str) -> Path:
    """Where an analysis' probe runs of a round lie, beside its experiment arms."""
    return paths.experiments(round_number, analysis_id) / "probes"


@dataclass
class PlannedProbes:
    """What the queued probes become: jobs to run, an outcome per probe, and the plan the
    export reads (episodes with their probe, robot, day and repeat; each probe's program)."""

    jobs: list[Job] = field(default_factory=list)
    outcomes: list[ProbeOutcome] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    programs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def plan(self) -> dict[str, Any]:
        return {
            "format": "fleetlab.probe-plan/1",
            "jobs": [j.model_dump(mode="json") for j in self.jobs],
            "outcomes": [o.model_dump(mode="json") for o in self.outcomes],
            "episodes": self.episodes,
            "programs": self.programs,
        }

    @classmethod
    def from_plan(cls, data: dict[str, Any]) -> PlannedProbes:
        return cls(
            jobs=[Job.model_validate(j) for j in data["jobs"]],
            outcomes=[ProbeOutcome.model_validate(o) for o in data["outcomes"]],
            episodes=list(data["episodes"]),
            programs=dict(data["programs"]),
        )


def plan_probes(
    world: LoopWorld,
    layout: LoopLayout,
    round_number: int,
    ledger: Ledger,
    probes: Sequence[QueuedProbe],
    allocations: dict[str, Any],
    *,
    software: Any,
) -> PlannedProbes:
    """Mint every probe episode; `allocations` (the analysis' of the round) is updated in
    place. Repeats are spread over the round's experiment days in turn."""
    from fleetlab.engine.probe_player import probe_spec

    days = days_of(layout.round(round_number).experiment)
    truth = world.truth
    next_episode = int(allocations.get("next_episode", truth.first_free_episode))
    next_index: dict[str, int] = dict(allocations.get("next_index", {}))
    planned = PlannedProbes()
    for queued in probes:
        request, probe_id = queued.request, queued.probe_id
        program = request.program
        identities: list[tuple[EpisodeIdentity, int]] = []
        episode, index = next_episode, dict(next_index)
        runnable = True
        for robot_id in request.robots:
            try:
                robot = truth.fleet.robot(robot_id)
            except KeyError:
                runnable = False
                break
            task = truth.tasks_of(robot_id)[0]
            for repeat in range(request.repeats):
                day = days[repeat % len(days)]
                key = f"{robot_id}/{day}"
                episode_index = index.get(key, truth.episodes_per_robot_day)
                index[key] = episode_index + 1
                scene_seed, noise_seed = seeds_for(truth.world_seed, robot_id, day, episode_index)
                faults = list(world.state_on(robot_id, day, ledger).faults)
                identity = EpisodeIdentity(
                    world_id=truth.world_id,
                    episode_id=neutral_id(IdKind.episode, episode),
                    robot_id=robot_id,
                    site_id=robot.site_id,
                    day=day,
                    episode_index=episode_index,
                    suite=task.suite,
                    task_index=task.task_index,
                    start_state=0,
                    scene_seed=scene_seed,
                    noise_seed=noise_seed,
                    policy=probe_spec(truth.policy.control_mode, PRECISION),
                    batch_size=1,
                    row=None,
                    image_size=truth.image_size,
                    max_steps=program.steps,
                    faults=faults,
                    software=software,
                    probe=program,
                )
                effective = hook_chain(world.registry, identity)(faults)
                if effective.precision != PRECISION:
                    runnable = False  # the engine cannot change a loaded policy's precision
                    break
                identities.append((identity, repeat))
                episode += 1
            if not runnable:
                break
        if not runnable:
            planned.outcomes.append(
                ProbeOutcome(
                    probe_id=probe_id,
                    status="refused",
                    episode_ids=[],
                    message="probe_not_runnable",
                    attempts_executed=0,
                    attempts_reserved=0,
                )
            )
            continue
        next_episode, next_index = episode, index
        planned.programs[probe_id] = program.model_dump(mode="json")
        for identity, repeat in identities:
            planned.jobs.append(Job(identity=identity, record_video=False))
            planned.episodes.append(
                {
                    "probe_id": probe_id,
                    "episode_id": identity.episode_id,
                    "robot_id": identity.robot_id,
                    "site_id": identity.site_id,
                    "day": identity.day,
                    "repeat": repeat,
                }
            )
        planned.outcomes.append(
            ProbeOutcome(
                probe_id=probe_id,
                status="done",
                episode_ids=[i.episode_id for i, _ in identities],
                message="done",
                attempts_executed=len(identities),
                attempts_reserved=0,
            )
        )
    allocations["next_episode"] = next_episode
    allocations["next_index"] = next_index
    return planned


def probe_config(setup: EngineSetup, run_dir: Path, per_day: int) -> Any:
    """The run configuration of a probe run: the setup's simulator, check and faults, the
    probe player in the worker, no policy server, no GPU, no video."""
    from fleetlab.engine.probe_player import PLAYER_FACTORY, SERVER_FACTORY

    config = setup.for_run(run_dir, per_day)
    return replace(
        config,
        policy_factory=SERVER_FACTORY,
        teleoperator=PLAYER_FACTORY,
        device="cpu",
        miopen=None,
        precision=PRECISION,
        batch_size=1,
        video=False,
        workers=min(config.workers, 4),
    )


def run_probes(
    world: LoopWorld,
    layout: LoopLayout,
    round_number: int,
    ledger: Ledger,
    paths: LoopPaths,
    setup: EngineSetup,
    analysis_id: str,
    session: int,
    probes: Sequence[QueuedProbe],
) -> list[ProbeOutcome]:
    """Plan (once, under the analysis' allocation lock) and run the session's probes;
    resumable from the saved plan. Returns one outcome per probe."""
    from fleetlab.engine import fleet
    from fleetlab.engine.experiments import _locked

    if not probes:
        return []
    root = probe_root(paths, round_number, analysis_id)
    run_dir = root / f"session-{session:02d}"
    plan_file = run_dir / PLAN_FILE
    if plan_file.is_file():
        planned = PlannedProbes.from_plan(json.loads(plan_file.read_text()))
        if sorted(o.probe_id for o in planned.outcomes) != sorted(q.probe_id for q in probes):
            raise ValueError(f"{plan_file} plans other probes than this queue's")
    else:
        software = fleet.prepare(probe_config(setup, run_dir, 1))
        experiments = paths.experiments(round_number, analysis_id)
        with _locked(experiments / "allocations.json") as allocations:
            if "next_episode" not in allocations:
                allocations["next_episode"] = _first_episode(world, paths, round_number)
            planned = plan_probes(
                world, layout, round_number, ledger, probes, allocations, software=software
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            temporary = plan_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(planned.plan(), indent=1))
            temporary.replace(plan_file)
    if (run_dir / FAILED_FILE).is_file():
        return _failed(run_dir, planned)  # reported failed once; never completed later
    if planned.jobs and not _complete(run_dir, planned.jobs):
        try:
            fleet.run(
                probe_config(setup, run_dir, world.truth.episodes_per_robot_day), planned.jobs
            )
        except Exception:  # the host log keeps the traceback; the agent sees a fixed code
            logging.getLogger(__name__).exception("probe run %s broke off", run_dir)
        if not _complete(run_dir, planned.jobs):
            (run_dir / FAILED_FILE).write_text(json.dumps({"failed": True}) + "\n")
            return _failed(run_dir, planned)
    return planned.outcomes


def _failed(run_dir: Path, planned: PlannedProbes) -> list[ProbeOutcome]:
    """The outcomes of a probe run that broke off. Partial work is never shown: the run is
    not exported (`probe_runs` skips it) and no episode is named. Each probe that would
    have run is failed and keeps its whole request charged, in two parts: the episodes
    the run left a record of (finished or partly written: known executed work), and the
    rest, reserved, because an in-worker probe that a lost worker was playing leaves no
    record and the runner cannot tell it from one that never started (a review of cddcc89)."""
    from fleetlab.log import run_status

    status = run_status(run_dir)
    recorded = set(status.done) | set(status.partial)
    out = []
    for outcome in planned.outcomes:
        if outcome.status != "done":
            out.append(outcome)
            continue
        executed = sum(e in recorded for e in outcome.episode_ids)
        out.append(
            ProbeOutcome(
                probe_id=outcome.probe_id,
                status="failed",
                episode_ids=[],
                message="execution_failed",
                attempts_executed=executed,
                attempts_reserved=len(outcome.episode_ids) - executed,
            )
        )
    return out


def _first_episode(world: LoopWorld, paths: LoopPaths, round_number: int) -> int:
    """After every episode number the world and every earlier round allocated (as
    `RoundEngine.first_episode`)."""
    numbers = [world.truth.first_free_episode]
    for number in range(1, round_number):
        for path in (paths.round(number) / "experiments").glob("*/allocations.json"):
            numbers.append(int(json.loads(path.read_text()).get("next_episode", 0)))
    return max(numbers)


def probe_runs(paths: LoopPaths, round_number: int, analyses: Sequence[str] = ()) -> list[Path]:
    """The finished probe runs an export of round `round_number` holds: those of each
    round's analysis in `analyses` (one per round, in round order), sessions in order."""
    from fleetlab.log import run_status

    runs: list[Path] = []
    for number, analysis in enumerate(analyses, start=1):
        if number > round_number or not analysis:
            continue
        root = probe_root(paths, number, analysis)
        for d in sorted(root.glob("session-*")):
            if not (d / PLAN_FILE).is_file() or (d / FAILED_FILE).is_file():
                continue  # a run reported failed is never shown, also if it ran on later
            jobs = PlannedProbes.from_plan(json.loads((d / PLAN_FILE).read_text())).jobs
            finished = jobs and (d / "run.json").is_file() and run_status(d).finalized
            if finished and _complete(d, jobs):
                runs.append(d)
    return runs


def _write(table: str, rows: list[dict[str, Any]], path: Path) -> None:
    from fleetlab.log._rows import sort_table, to_arrow

    path.parent.mkdir(parents=True, exist_ok=True)
    data = sort_table(table, to_arrow(table, rows))
    pq.write_table(data, path, compression="zstd", store_schema=False)


def probe_step_rows(
    episode_id: str, program: ProbeProgram, steps: pa.Table
) -> list[dict[str, Any]]:
    """The `probe_steps` rows of one probe episode from its visible `steps` rows."""
    trajectory = program_trajectory(program.segments)
    rows = sorted(steps.to_pylist(), key=lambda r: r["step"])
    start = np.asarray(rows[0]["eef_pos"], np.float64)
    out = []
    for row in rows:
        k = row["step"]
        commanded = row["commanded_action"] is not None and k < len(trajectory.segment)
        target = (start + trajectory.offset[k]).astype(np.float32) if commanded else None
        qpos = row["gripper_qpos"]
        out.append(
            {
                "episode_id": episode_id,
                "step": k,
                "time_s": row["time_s"],
                "segment": int(trajectory.segment[k]) if commanded else None,
                "target_pos": None if target is None else [float(v) for v in target],
                "gripper_command": float(trajectory.gripper[k]) if commanded else None,
                "eef_pos": row["eef_pos"],
                "eef_quat": row["eef_quat"],
                "joint_pos": row["joint_pos"],
                "joint_vel": row["joint_vel"],
                "gripper_opening": float(qpos[0] - qpos[1]),
            }
        )
    return out


def write_probe_tables(stage: Path, runs: Sequence[Path]) -> None:
    """The probe tables of an export being built in `stage`, from the probe runs, and
    their part of the data dictionary. Probe episodes are refused if the export already
    holds them as regular episodes."""
    from fleetlab.log.dictionary import DICTIONARY_FILE

    if not runs:
        return
    regular = set(
        pq.read_table(stage / SCHEMA["episodes"].path, columns=["episode_id"])
        .column("episode_id")
        .to_pylist()
    )
    episodes: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for run in runs:
        plan = json.loads((Path(run) / PLAN_FILE).read_text())
        programs = {k: ProbeProgram.model_validate(v) for k, v in plan["programs"].items()}
        for probe_id, program in sorted(programs.items()):
            trajectory = program_trajectory(program.segments)
            for index, s in enumerate(program.segments):
                segments.append(
                    {
                        "probe_id": probe_id,
                        "segment": index,
                        "primitive": s.primitive.value,
                        "axis": None if s.axis is None else s.axis.value,
                        "amplitude_m": s.amplitude_m,
                        "duration_steps": s.duration_steps,
                        "period_steps": s.period_steps,
                        "first_step": trajectory.first_step[index],
                    }
                )
        for entry in plan["episodes"]:
            episode_id = entry["episode_id"]
            if episode_id in regular:
                raise ValueError(f"probe episode {episode_id} is also a regular episode")
            source = Path(run) / SCHEMA["steps"].path.format(episode_id=episode_id)
            steps = pq.read_table(source)
            rows = probe_step_rows(episode_id, programs[entry["probe_id"]], steps)
            target = stage / SCHEMA["probe_steps"].path.format(episode_id=episode_id)
            _write("probe_steps", rows, target)
            episodes.append({**entry, "steps": rows[-1]["step"]})
    _write("probe_episodes", episodes, stage / SCHEMA["probe_episodes"].path)
    _write("probe_segments", segments, stage / SCHEMA["probe_segments"].path)
    dictionary = stage / DICTIONARY_FILE
    text = dictionary.read_text().rstrip() + "\n\n" + export_table_markdown().rstrip()
    dictionary.write_text(text + "\n\n" + PROBE_DICTIONARY)


# A probe check without an agent: run one probe request on a world and read the lag.


def step_lags(rows: Sequence[dict[str, Any]], threshold_m: float = 0.001) -> list[dict[str, Any]]:
    """For each change of the commanded target in one probe episode's `probe_steps` rows (in
    step order): the row it was issued in, and how many rows later the measured position
    first moved by `threshold_m` toward it and first covered half of it, per axis."""
    out: list[dict[str, Any]] = []
    commanded = [r for r in rows if r["target_pos"] is not None]
    for i in range(1, len(commanded)):
        before, now = commanded[i - 1]["target_pos"], commanded[i]["target_pos"]
        for axis in range(3):
            change = float(now[axis]) - float(before[axis])
            if abs(change) < 1e-6:
                continue
            k = commanded[i]["step"]
            start = float(rows[k]["eef_pos"][axis])
            moved = half = None
            for r in rows[k + 1 :]:
                progress = (float(r["eef_pos"][axis]) - start) * (1 if change > 0 else -1)
                if moved is None and progress >= threshold_m:
                    moved = r["step"] - k
                if half is None and progress >= abs(change) / 2:
                    half = r["step"] - k
            out.append(
                {
                    "issued_at": k,
                    "axis": "xyz"[axis],
                    "change_m": round(change, 6),
                    "first_movement_rows": moved,
                    "half_way_rows": half,
                }
            )
    return out


def probe_check(
    world: LoopWorld,
    layout: LoopLayout,
    ledger: Ledger,
    paths: LoopPaths,
    setup: EngineSetup,
    analysis_id: str,
    request: Any,
    out: Path,
    round_number: int = 1,
    earlier: Sequence[str] = (),
) -> dict[str, Any]:
    """Run one probe request on the world's round as a session would have it run, build the
    round's export with the probe tables (leak-checked), and read every commanded step's
    response per probe episode from the export, as an agent's query would.

    `ledger` is the ledger in force during the round; `earlier` names the analysis of each
    earlier round, as `RoundEngine.analyses` pads it (an empty name for a round without
    one), so that a round's export holds the earlier rounds' probes too. The probe of
    round k is numbered k, its run the round's session 1."""
    from fleetlab.contracts.probes import ProbeRequest
    from fleetlab.loop.export import export_round

    if len(earlier) > round_number - 1:
        raise ValueError(f"round {round_number} has {round_number - 1} earlier rounds")
    analyses = [*earlier, *[""] * (round_number - 1 - len(earlier)), analysis_id]
    parsed = ProbeRequest.model_validate(request)
    probe_id = neutral_id(IdKind.probe, round_number)
    queued = [QueuedProbe(probe_id=probe_id, session=1, request=parsed)]
    outcomes = run_probes(world, layout, round_number, ledger, paths, setup, analysis_id, 1, queued)
    export = export_round(world, layout, round_number, ledger, paths, out, analyses=analyses)
    from fleetlab.log import LogReader

    episodes: list[dict[str, Any]] = []
    with LogReader(export) as reader:
        for e in reader.query("SELECT * FROM probe_episodes ORDER BY robot_id, repeat").to_pylist():
            rows = reader.query(
                "SELECT * FROM probe_steps WHERE episode_id = ? ORDER BY step", [e["episode_id"]]
            ).to_pylist()
            episodes.append({**e, "edges": step_lags(rows)})
        columns = {
            name: [c[0] for c in reader.con.execute(f"DESCRIBE {name}").fetchall()]
            for name in ("probe_episodes", "probe_segments", "probe_steps")
        }
    return {
        "request": parsed.model_dump(mode="json"),
        "outcomes": [o.model_dump(mode="json") for o in outcomes],
        "export": str(export),
        "columns": columns,
        "episodes": episodes,
    }


# A probe recipe on named robots: a learned skill's probe, run and read as its skill says.

RECIPE_ANALYSIS = "recipe-check"
RESULT_PAGE = "index.md"
RESULT_FILE = "recipe-check.json"
FIGURE_FILE = "probe.png"
HOST_DIR = "host"  # under the output: the probe runs and the provenance (truth side)
HOST_FILE = "provenance.json"
MAX_EVIDENCE_ROWS = 200


def _plain(value: Any) -> Any:
    """A query value as JSON takes it (numbers, text, lists; anything else as text)."""
    return json.loads(json.dumps(value, default=str))


def run_evidence(export: Path, skill_md: str, values: dict[str, Any]) -> list[dict[str, Any]]:
    """Each evidence SQL of a rendered learned skill on an export, as the tool server's
    `query` runs it (the visible tables in memory, file access off, one SELECT), with its
    `$parameters` bound from `values`. An SQL that needs a value not given is reported and
    not run; at most `MAX_EVIDENCE_ROWS` rows are kept."""
    import duckdb

    from fleetlab.contracts.learned import evidence_sql, sql_parameters
    from fleetlab.log import LogReader

    out: list[dict[str, Any]] = []
    reader = LogReader(export, materialize=True)
    try:
        reader.con.execute("SET threads = 2")
        reader.lock(materialized=True)
        for helper, sql in evidence_sql(skill_md):
            names = sql_parameters(sql)
            entry: dict[str, Any] = {
                "helper": helper,
                "sql": sql,
                "parameters": {n: values[n] for n in names if n in values},
            }
            missing = [n for n in names if n not in values]
            if missing:
                out.append({**entry, "status": "skipped", "missing": missing})
                continue
            statements = reader.con.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                out.append({**entry, "status": "refused", "error": "not one SELECT"})
                continue
            try:
                cursor = reader.con.execute(sql, {n: values[n] for n in names} if names else None)
                columns = [d[0] for d in cursor.description or []]
                rows = cursor.fetchmany(MAX_EVIDENCE_ROWS)
            except duckdb.Error as error:
                out.append({**entry, "status": "failed", "error": str(error)})
                continue
            entry |= {"status": "done", "columns": columns, "rows": _plain([list(r) for r in rows])}
            out.append(entry)
    finally:
        reader.close()
    return out


def _settled(rows: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]) -> list[float | None]:
    """For each commanded change: measured minus commanded on its axis, in millimetres, in
    the row the next change is issued in (or the last row): where the hand settled."""
    issued = sorted({e["issued_at"] for e in edges})
    last = rows[-1]["step"]
    out: list[float | None] = []
    for e in edges:
        later = [k for k in issued if k > e["issued_at"]]
        row = later[0] if later else last
        axis = "xyz".index(e["axis"])
        command = rows[row - 1]["target_pos"] if row >= 1 else None
        if command is None:
            out.append(None)
            continue
        out.append(round((float(rows[row]["eef_pos"][axis]) - float(command[axis])) * 1000, 2))
    return out


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def _page(result: dict[str, Any]) -> str:
    """The result page of a recipe check, for a person: what ran, the response per probe
    episode, the evidence queries' rows and the figure."""
    request = result["request"]
    bound = ", ".join(f"{k} = {v}" for k, v in result["robots"].items())
    lines = [
        f"# Probe recipe `{result['skill']}` on {', '.join(request['robots'])}",
        "",
        f"The probe `{request['name']}` that the learned skill `{result['skill']}` carries,",
        f"run on the robots named ({bound}) of fleet `{result['fleet_id']}` under its",
        f"original state, {request['repeats']} repetitions each, as a light probe run; then",
        "the skill's evidence SQL on the run's export, with `$probe` bound to the probe's id",
        f"(`{result['probe_id']}`).",
        "",
        f"- Purpose: {request['purpose']}",
        f"- Expectation: {request['expectation']}",
        f"- Outcome: {result['outcome']['status']}; library `{result['library_digest'][:16]}`",
        f"  ({result['provenance']}).",
        "",
        "## Response per probe episode",
        "",
        "Rows count from the row a commanded change was issued in (50 ms each); settled is",
        "measured minus commanded on that axis when the next change is issued (mm); the",
        "start is where the hand was in row 0, which the probe's targets are offsets from.",
        "",
    ]
    table_rows = []
    for e in result["episodes"]:
        start = ", ".join(f"{v:.4f}" for v in e["start"])
        for edge, settled in zip(e["edges"], e["settled_mm"], strict=True):
            table_rows.append(
                [
                    e["robot_id"],
                    e["repeat"],
                    start,
                    edge["issued_at"],
                    edge["axis"],
                    round(edge["change_m"] * 1000, 1),
                    edge["first_movement_rows"],
                    edge["half_way_rows"],
                    settled,
                ]
            )
    header = ["robot", "repeat", "start (m)", "issued at", "axis", "change (mm)"]
    header += ["first move", "half-way", "settled (mm)"]
    lines += _table(header, table_rows)
    lines += ["", f"![commanded against measured]({FIGURE_FILE})", "", "## Evidence", ""]
    for item in result["evidence"]:
        lines += [f"### {item['helper']}", "", "```sql", item["sql"], "```", ""]
        if item["status"] == "done":
            lines += _table(item["columns"], item["rows"]) if item["rows"] else ["(no rows)"]
        else:
            reason = item.get("missing") or item.get("error")
            lines += [f"Not run ({item['status']}): {reason}."]
        lines += [""]
    return "\n".join(lines)


def recipe_check(
    world: LoopWorld,
    setup: EngineSetup,
    library: Path,
    skill: str,
    robots: Sequence[str],
    out: Path,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the probe recipe of a learned skill (`loop.learned.FrozenLibrary.recipe`) on the
    named robots of a stored world, one robot for each of its robot parameters in order:
    a light probe run of round 1 under the original state, the round's export with the probe
    tables (leak-checked), the skill's evidence SQL on it with `$probe` bound to the probe's
    id, the robots bound to their parameters and `params` to the rest, and the result page
    (`index.md`, `probe.png`, `recipe-check.json` under `out`).

    `out` is a new or empty directory. Before anything runs, the skill passes the learned
    skill audit against this world's runs and full fleet (`agents.audit.audit_learned_skill`,
    as at an agent's launch), and its recipe is the probe its SKILL.md renders. What a person
    reads (the page, the figure, `recipe-check.json`, the export) names the fleet by its
    public id; the world's id, the paths and the probe runs stay under `out/host/`."""
    from fleetlab.agents.audit import audit_learned_skill
    from fleetlab.contracts.learned import PROBE_PARAMETER, probe_block, typed_value
    from fleetlab.contracts.skills import SKILL_FILE
    from fleetlab.log import LogReader
    from fleetlab.loop.exhibit.figures import probe_figure
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.learned import FrozenLibrary

    if out.exists() and any(out.iterdir()):
        # A recipe's probe always has the same analysis, session and id: a used directory
        # would answer a new request with the measurements of an earlier one.
        raise ValueError(f"{out} is not empty; a recipe check writes into a new directory")
    frozen = FrozenLibrary.load(library)
    probe = frozen.recipe(skill)
    if probe is None:
        raise ValueError(f"the skill {skill!r} of {library} carries no probe")
    skill_dir = frozen.skills / skill
    skill_md = (skill_dir / SKILL_FILE).read_text(encoding="utf-8")
    if probe_block(skill_md) != probe.model_dump(mode="json"):
        raise ValueError(f"the recipe of {skill!r} is not the probe its SKILL.md renders")
    # The skill was audited against its source fleet when it was frozen; it runs here only
    # if it passes the same audit against this fleet, as at an agent's launch.
    problems = audit_learned_skill(skill_dir, tuple(world.runs), world.model)
    if problems:
        raise ValueError(f"the skill {skill!r} may not be used on this fleet: {problems}")
    request = probe.request(robots)
    out.mkdir(parents=True, exist_ok=True)
    host = out / HOST_DIR  # truth side: the probe runs and the provenance, never shown
    summary = probe_check(
        world,
        layout_for_world(world.spec),
        Ledger.empty(world.world_id),
        LoopPaths(host / "loop"),
        setup,
        RECIPE_ANALYSIS,
        request.model_dump(mode="json"),
        out / "export",
    )
    (outcome,) = summary["outcomes"]
    bound = dict(zip(probe.robot_parameters, request.robots, strict=True))
    values: dict[str, Any] = {k: typed_value(v) for k, v in (params or {}).items()}
    values |= {**bound, PROBE_PARAMETER: outcome["probe_id"]}
    evidence = run_evidence(Path(summary["export"]), skill_md, values)
    steps: dict[str, list[dict[str, Any]]] = {}
    with LogReader(summary["export"]) as reader:
        for e in summary["episodes"]:
            steps[e["episode_id"]] = reader.query(
                "SELECT * FROM probe_steps WHERE episode_id = ? ORDER BY step", [e["episode_id"]]
            ).to_pylist()
    episodes = [
        {
            "episode_id": e["episode_id"],
            "robot_id": e["robot_id"],
            "repeat": e["repeat"],
            "start": [float(v) for v in steps[e["episode_id"]][0]["eef_pos"]],
            "edges": e["edges"],
            "settled_mm": _settled(steps[e["episode_id"]], e["edges"]),
        }
        for e in summary["episodes"]
    ]
    entry = next(s for s in frozen.record["skills"] if s["name"] == skill)
    result = {
        "skill": skill,
        "library_digest": frozen.digest,
        "provenance": entry["provenance"],
        "fleet_id": world.truth.fleet.id,  # the public description's id, never the world's
        "robots": bound,
        "request": request.model_dump(mode="json"),
        "probe_id": outcome["probe_id"],
        "outcome": outcome,
        "episodes": episodes,
        "evidence": evidence,
    }
    provenance = {
        "world_id": world.world_id,
        "world_root": str(world.root),
        "library": str(frozen.root),
        "export": summary["export"],
        "probe_runs": str(host / "loop"),
    }
    (host / HOST_FILE).write_text(json.dumps(provenance, indent=1) + "\n")
    panels = [
        (
            f"{parameter} = {robot}",
            [steps[e["episode_id"]] for e in summary["episodes"] if e["robot_id"] == robot],
        )
        for parameter, robot in bound.items()
    ]
    axis = next((s.axis.value for s in request.segments if s.axis is not None), "x")
    origin = "toy" if world.truth.policy.name.startswith("toy") else ""
    title = f"probe {request.name}: commanded against measured"
    probe_figure(out / FIGURE_FILE, panels, title, origin, axis=axis)
    (out / RESULT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    (out / RESULT_PAGE).write_text(_page(result) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """One probe request on a stored world, without an agent, for a real check:

        tools/heavy-run uv run --locked python -m fleetlab.loop.probes \
            --world ROOT --loop DIR --request REQUEST.json --out EXPORT [--workers 4]

    Round 1's experiment days under the original state (an empty ledger). Prints and
    writes `<loop>/probe-check.json`: the outcome, the export's probe columns and every
    commanded step's lag per probe episode, read from the export's tables.

    Or a learned skill's probe recipe on named robots (`recipe_check`), a light job:

        tools/heavy-run --light uv run --locked python -m fleetlab.loop.probes \
            --library LIB --skill NAME --robot r05 --robot r01 --world ROOT --out DIR \
            [--param name=value ...] [--workers 4]

    one `--robot` for each of the probe's robot parameters, in order; `--param` gives a
    value for any other parameter its evidence SQL takes; `DIR` is a new or empty directory.
    Writes `DIR/index.md` (the result page), `DIR/probe.png` and `DIR/recipe-check.json`,
    keeps the probe runs and the provenance under `DIR/host/`, and prints the page's path.
    """
    import argparse

    from fleetlab.engine.fleet import RunConfig
    from fleetlab.loop.layout import layout_for_world

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.probes")
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--loop", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", default="probe-check")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--library", type=Path, help="a frozen learned library (recipe mode)")
    parser.add_argument("--skill", help="the learned skill whose probe recipe runs")
    parser.add_argument("--robot", action="append", default=[], help="in parameter order")
    parser.add_argument("--param", action="append", default=[], help="name=value")
    args = parser.parse_args(argv)
    if args.library is not None:
        if not args.skill or not args.robot:
            parser.error("a recipe check needs --skill and one --robot per robot parameter")
        params = dict(item.split("=", 1) for item in args.param)
    elif args.loop is None or args.request is None:
        parser.error("a probe check needs --loop and --request (or --library for a recipe)")
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)

    world = LoopWorld.load(args.world)
    config = RunConfig(out="", workers=args.workers)
    setup = EngineSetup(config=config, contacts=False, events=False, video=False)
    if args.library is not None:
        recipe_check(world, setup, args.library, args.skill, args.robot, args.out, params)
        print(args.out / RESULT_PAGE)
        return 0
    layout = layout_for_world(world.spec)
    paths = LoopPaths(args.loop)
    summary = probe_check(
        world,
        layout,
        Ledger.empty(world.world_id),
        paths,
        setup,
        args.analysis,
        json.loads(args.request.read_text()),
        args.out,
    )
    text = json.dumps(summary, indent=1, default=str)
    (args.loop / "probe-check.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# The opt-in probe path of an engagement (`fleetlab.loop.rounds.Engagement.probes`).

PROBE_TOOLS: tuple[str, ...] = ("request_probe", "probe_results")
# The base skill that goes with the probe tools (skills/instrument-step-response): mounted
# beside the analysis skills when an engagement offers probes, never otherwise.
PROBE_SKILLS: tuple[str, ...] = ("instrument-step-response",)


def probe_tool_env_args(probes: bool) -> dict[str, Any]:
    """`agents.toolenv.build_tool_env`'s arguments for an engagement: the analysis role's
    skills, and with probes the base skill on step responses beside them."""
    if not probes:
        return {"role": "analysis"}
    from fleetlab.skill_library import ANALYSIS_SKILLS, library_dir

    return {"skills": library_dir(), "skill_names": (*ANALYSIS_SKILLS, *PROBE_SKILLS)}


PROBE_TASK = """Probes. Besides experiments you can compose probes of your own: a short
scripted motion of a robot's hand and gripper, played without the policy and without a
task (request_probe). You choose the robots, the repetitions and the program, built from
hold, step, ramp, sine, open_gripper and close_gripper on the x, y or z axis of the end
effector, within the bounds the tool states; you name the probe and say what question it
answers and what you expect to see. Probes run after the session ends, like experiments,
on the robots as they are now, and each probe episode costs one attempt of this round's
experiment budget. The next session finds them with probe_results and their records in
the tables probe_episodes, probe_segments and probe_steps: every control step's commanded
target and gripper command beside the measured hand position, joints and gripper
opening, timed as the data dictionary states. Query them like any table."""

PROBE_LEARNING = """A probe you want later investigations to be able to run again goes into a
learned skill's probe field: your request_probe request with robot_parameters, declared
parameters in order, in place of its robots. Its evidence then holds the query that reads the
probe's probe_steps with $probe, the probe's id, so that the probe and its analysis can be run
again on other robots."""


def with_probes(task: str, learning: bool = True) -> str:
    """The round task with the probe paragraph (and, with learning, how to keep a probe)."""
    text = task.rstrip() + "\n\n" + PROBE_TASK
    if learning:
        text += "\n" + PROBE_LEARNING
    return text


def probe_tool_texts(limits: dict[str, int] | None = None) -> dict[str, str]:
    """The probe tools' descriptions as the agent reads them, for the launch audit's texts."""
    from fleetlab.contracts.tools import TOOLS
    from fleetlab.toolserver.server import Limits, tool_description

    chosen = Limits(**dict(limits or {}))  # type: ignore[arg-type]
    return {
        f"tool {spec.name}": tool_description(spec, chosen)
        for spec in TOOLS
        if spec.name in PROBE_TOOLS
    }
