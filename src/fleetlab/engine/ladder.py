"""The ladder of worlds, the engine's side: the smoke world and the nightly worlds.

**The smoke world** (`w30`): one robot, 8 episodes of `libero_spatial`, the
scene camera turned 3 degrees on the second day. It runs on every merge
(`make smoke-world`, about two minutes as a heavy job): every job's chain is
checked against the reference stored in the repository
(`smoke_reference.json`), so a change that moves a bit anywhere in an
episode shows. The run directory is named after the commit, so a run killed
half way resumes, and a new commit runs afresh. Exactness holds per machine:
the reference is this machine's.

**The nightly worlds** (`make nightly`, one heavy job): one site, four robots,
three days (observed, experiment, scoring) at 40 episodes per robot and day,
480 as run, three faults on three robots (a scene camera turned, commands
executed late, worn gripper pads), and one healthy robot. Per world:

- the as-run episodes and, in the same run, the oracle's replays of the
  scoring day: every scoring episode under the power set of its faults;
- a live pass over the scoring day's episodes with the timeout-rule
  supervisor, which hands an attempt still running at step 160 to the
  scripted teleoperator;
- the scores: success per window and robot, each fault's effect and the best
  gain at k fixes from the oracle, what the live pass rescued.

`w20` keeps its seed, so its every chain and outcome must equal the previous
night's (the regression test of the engine and the replay rules); `w21` draws
a fresh seed each night (which robot carries which fault, at which calibrated
level, from which day) and is scored the same way: the gap between the two is
the over-fitting signal. After both, a canary replays the first ten of `w20`'s
episodes, hours after they ran. The comparison with the previous night goes to
`comparison.md` (for the dashboard's progress page, `--nightly FILE`) and
`comparison.json`; every run keeps its memory series (`run.json`), which is
the soak test's record.

The nightly worlds come from the world generator (`fleetlab.worlds`), so
each has its world files (`<world>/files/`: spec, model, description,
manifest, fault table, health) and agents can be scored on it later. The smoke
world is built here from contracts types and stays apart from the generator: it
tests the engine, and a new generator version must not move its reference.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts.faults import FaultSpec, Schedule, Target, active_faults
from fleetlab.contracts.identity import (
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    Task,
    seeds_for,
)
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.model import ComponentState, ConfigVersion, Fleet, Robot, Site
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SupervisorSpec
from fleetlab.contracts.vocabulary import Component, ConfigKey
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec

REFERENCE = Path(__file__).with_name("smoke_reference.json")
SUITE = "libero_spatial"
START_STATES = 50
LIVE_AFTER = 160  # the timeout rule: an attempt still running at this step is handed over
CONTINUATION_STEPS = 700  # the teleoperator's budget (up to 600 plus settling)
TELEOPERATOR = "fleetlab.policies.teleoperator:Teleoperator"
CANARY = 10


WINDOWS = ("observed", "experiment", "scoring")


@dataclass
class WorldPlan:
    """What a ladder run needs of a world: its episodes, faults, windows and day length."""

    world_id: str
    seed: int
    faults: tuple[FaultSpec, ...]
    windows: dict[str, tuple[int, ...]]  # observed, experiment, scoring: their days
    per_day: int
    identities: list[EpisodeIdentity]
    generated: Any = None  # the generator's World, written next to the runs

    @property
    def scoring_days(self) -> tuple[int, ...]:
        return self.windows["scoring"]

    def window_of(self, day: int) -> str:
        return next(name for name, days in self.windows.items() if day in days)


@dataclass(frozen=True)
class LadderWorld:
    """A world built by hand: robots at one site, days, episodes per day, faults (the smoke
    world, and toy worlds in tests)."""

    world_id: str
    seed: int
    robots: tuple[str, ...]
    days: int
    per_day: int
    faults: tuple[FaultSpec, ...]
    windows: tuple[int, int, int] = (0, 1, 2)  # observed, experiment, scoring day
    note: str = ""

    def fleet(self) -> Fleet:
        def robot(robot_id: str) -> Robot:
            return Robot(
                id=robot_id,
                site_id="s1",
                components={c: ComponentState(component=c, installed_day=0) for c in Component},
            )

        return Fleet(
            id=self.world_id,
            sites=[Site(id="s1", robots=[robot(r) for r in self.robots])],
            config=ConfigVersion(version=1, values={ConfigKey.control_action_delay_steps: 0}),
        )

    def identities(
        self, software: SoftwareVersion, policy: PolicySpec, batch_size: int = 4
    ) -> list[EpisodeIdentity]:
        fleet = self.fleet()
        out = []
        number = 1
        for day in range(self.days):
            for robot_id in self.robots:
                in_force = active_faults(list(self.faults), fleet, robot_id, day)
                for index in range(self.per_day):
                    scene, noise = seeds_for(self.seed, robot_id, day, index)
                    draw = _rng(self.seed, "start", robot_id, day, index)
                    out.append(
                        EpisodeIdentity(
                            world_id=self.world_id,
                            episode_id=neutral_id(IdKind.episode, number),
                            robot_id=robot_id,
                            site_id="s1",
                            day=day,
                            episode_index=index,
                            suite=SUITE,
                            task_index=index % 10,
                            start_state=int(draw.integers(START_STATES)),
                            scene_seed=scene,
                            noise_seed=noise,
                            policy=policy,
                            batch_size=batch_size,
                            row=None,
                            image_size=360,
                            max_steps=280,
                            faults=in_force,
                            software=software,
                        )
                    )
                    number += 1
        return out

    def plan(self, software: SoftwareVersion, policy: PolicySpec, batch_size: int = 4) -> WorldPlan:
        return WorldPlan(
            self.world_id,
            self.seed,
            self.faults,
            {name: (day,) for name, day in zip(WINDOWS, self.windows, strict=True)},
            self.per_day,
            self.identities(software, policy, batch_size),
        )


def _rng(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256(json.dumps([seed, *parts], default=str).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


def _camera(fault: int, robot: str, degrees: float, onset: int = 0) -> FaultSpec:
    return FaultSpec(
        fault_id=neutral_id(IdKind.fault, fault),
        kind="camera_pose_offset",
        params={},
        level=degrees,
        target=Target(scope=robot, component=Component.scene_camera),
        schedule=Schedule(onset_day=onset),
    )


def _latency(fault: int, robot: str, steps: float, onset: int = 0) -> FaultSpec:
    return FaultSpec(
        fault_id=neutral_id(IdKind.fault, fault),
        kind="action_latency",
        params={},
        level=steps,
        target=Target(scope=robot, config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=onset),
        nominal=0,
    )


def _wear(fault: int, robot: str, level: float, onset: int = 0) -> FaultSpec:
    return FaultSpec(
        fault_id=neutral_id(IdKind.fault, fault),
        kind="gripper_wear",
        params={},
        level=level,
        target=Target(scope=robot, component=Component.gripper),
        schedule=Schedule(onset_day=onset),
    )


def smoke_world() -> LadderWorld:
    """`w30`: one robot, two days of four episodes, the camera turned 3 degrees on day 1."""
    return LadderWorld(
        world_id="w30",
        seed=30,
        robots=("r01",),
        days=2,
        per_day=4,
        faults=(_camera(1, "r01", 3.0, onset=1),),
        note="the smoke world",
    )


NIGHTLY_TASKS = tuple(Task(SUITE, i) for i in range(10))  # the calibrated suite


def nightly_spec(randomized_seed: int | None = None, per_day: int = 40) -> WorldSpec:
    """`w20` (seed 20) or `w21` (drawn from `randomized_seed`): 4 robots, 3 days, 40 a day.

    w20: the scene camera of r02 turned 6 degrees, commands 17 steps late on
    r03, worn gripper pads on r04 from day 1, r01 healthy. w21: the same three
    types on three robots drawn, the camera and the delay at a calibrated level
    drawn (`fleetlab.worlds.calibration`), each from day 0 or 1. Only wear 1.0
    is calibrated, so its level stays.
    """
    from fleetlab.worlds.generate import policy_spec_nominal

    robots = ("r01", "r02", "r03", "r04")
    if randomized_seed is None:
        world_id, seed = "w20", 20
        faults = (_camera(1, "r02", 6.0), _latency(2, "r03", 17.0), _wear(3, "r04", 1.0, onset=1))
    else:
        world_id, seed = "w21", randomized_seed
        draw = _rng(randomized_seed, "faults")
        order = [robots[i] for i in draw.permutation(len(robots))]
        faults = (
            _camera(1, order[0], float(draw.choice([3.0, 6.0, 9.0])), int(draw.integers(2))),
            _latency(2, order[1], float(draw.choice([14.0, 17.0, 20.0])), int(draw.integers(2))),
            _wear(3, order[2], 1.0, int(draw.integers(2))),
        )
    return WorldSpec(
        world_id=world_id,
        seed=seed,
        size="small",
        sites=1,
        robots_per_site=len(robots),
        days=3,
        windows=Windows(
            observed=DayRange(first=0, last=0),
            experiment=DayRange(first=1, last=1),
            scoring=DayRange(first=2, last=2),
        ),
        fault_schedule=list(faults),
        patterns=Patterns(),
        policy=policy_spec_nominal(),
        tasks=list(NIGHTLY_TASKS),
        episodes_per_robot_day=per_day,
    )


def generated_plan(
    spec: WorldSpec, software: SoftwareVersion, texts: dict[Task, str] | None = None
) -> WorldPlan:
    """A world of the generator as the ladder runs it."""
    from fleetlab.worlds.generate import generate

    world = generate(spec, software, texts)
    windows = {
        name: tuple(range(r.first, r.last + 1))
        for name, r in (
            ("observed", spec.windows.observed),
            ("experiment", spec.windows.experiment),
            ("scoring", spec.windows.scoring),
        )
    }
    return WorldPlan(
        spec.world_id,
        spec.seed,
        tuple(spec.fault_schedule),
        windows,
        spec.episodes_per_robot_day,
        world.identities(),
        world,
    )


# ---- running a world ---------------------------------------------------------------------------


def oracle_jobs(identities: Sequence[EpisodeIdentity], days: Sequence[int]) -> list[Job]:
    """Every scoring episode under every proper subset of its faults (the power set).

    The ladder's worlds have no decoys or confounders, so this equals
    `fleetlab.integration.oracle.oracle_jobs` on them (a test holds it).
    """
    jobs = []
    for ident in identities:
        if ident.day not in days or not ident.faults:
            continue
        ids = [f.fault_id for f in ident.faults]
        for mask in range(1, 2 ** len(ids)):
            remove = [ids[i] for i in range(len(ids)) if mask >> i & 1]
            jobs.append(Job(identity=ident, mode=JobMode.replay, remove=remove))
    return jobs


def live_jobs(
    identities: Sequence[EpisodeIdentity], days: Sequence[int], after: int = LIVE_AFTER
) -> list[Job]:
    """The days' episodes again under the timeout rule, which hands over at step `after`."""
    supervisor = SupervisorSpec(
        name="timeout", params={"after": after, "decision": "hand_over", "decision_ms": 500}
    )
    return [Job(identity=i, supervisor=supervisor) for i in identities if i.day in days]


def chain_digest(path: Path) -> str:
    """One hash over a chain file's arrays (state, images, executed actions, fixture hash)."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            h.update(name.encode())
            h.update(np.ascontiguousarray(data[name]).tobytes())
    return h.hexdigest()[:16]


def digests(run_dir: Path) -> dict[str, str]:
    """The digest of every chain of a run (as-run jobs, replays, chain sample), by file name."""
    folder = RunLayout(run_dir).table("chain", "key").parent
    return {p.stem: chain_digest(p) for p in sorted(folder.glob("*.npz"))}


def run_dirs(root: Path, name: str) -> dict[str, Path]:
    """A world's two run directories: `name` (as run and the oracle) and `name-live`.

    The dashboard names a run's page after its directory and refuses one name
    from two directories, so a night's runs carry the world and the date.
    """
    return {"world": root / name, "live": root / f"{name}-live"}


@dataclass
class WorldRun:
    plan: WorldPlan
    dirs: dict[str, Path]
    seconds: dict[str, float] = field(default_factory=dict)
    world_hash: str | None = None  # a generated world's hash (contracts.world.seal_world)


def run_software(config: Any) -> SoftwareVersion:
    """The software the runs of `config` are made with (activates the pinned database)."""
    from fleetlab.engine import miopen, software

    miopen.activate(config.miopen)
    return software.current(miopen.require(config.batch_size, config.precision, config.miopen))


def run_world(
    plan: WorldPlan,
    root: Path,
    config: Any,
    *,
    name: str = "world",
    oracle: bool = True,
    live: bool = True,
) -> WorldRun:
    """The as-run episodes with the oracle replays, then the live pass; resumable.

    The runs go to `run_dirs(root, name)`. A generated world's files are
    written to `root/files` first (the world hash in `files/hash`).
    """
    from fleetlab.engine import fleet

    out = WorldRun(plan, run_dirs(root, name))
    if plan.generated is not None:
        from fleetlab.worlds.generate import write_world

        base_runs = [str(out.dirs["world"])]
        out.world_hash = write_world(plan.generated, root / "files", base_runs)
    identities = plan.identities
    jobs = [Job(identity=i, record_video=True) for i in identities]
    if oracle:
        jobs += oracle_jobs(identities, plan.scoring_days)
    base = replace(config, episodes_per_robot_day=plan.per_day)
    t = time.perf_counter()
    fleet.run(replace(base, out=str(out.dirs["world"]), workers=fitting_workers(base)), jobs)
    out.seconds["world"] = time.perf_counter() - t
    if live:
        t = time.perf_counter()
        live_config = replace(
            base,
            out=str(out.dirs["live"]),
            teleoperator=config.teleoperator or TELEOPERATOR,
            continuation_steps=config.continuation_steps or CONTINUATION_STEPS,
            workers=fitting_workers(base),
        )
        fleet.run(live_config, live_jobs(identities, plan.scoring_days))
        out.seconds["live"] = time.perf_counter() - t
    return out


def fitting_workers(config: Any) -> int:
    """As many workers as the memory available holds, at most `config.workers`, at least 4
    (or `config.workers`, if fewer).

    The layout does not change a bit of an episode (rows are independent of
    their neighbours, the batch size is fixed), so a night that finds less
    memory free runs slower, not differently.
    """
    from fleetlab.engine.fleet import mem_available_gib

    room = mem_available_gib() - config.mem_floor_gib - config.server_gib - 1.0
    return int(min(config.workers, max(4, room // config.worker_gib)))


def scores(plan: WorldPlan, dirs: dict[str, Path]) -> dict[str, Any]:
    """What a night compares: outcomes by window and robot, the oracle, the live pass."""
    from fleetlab.contracts.oracle import OracleTable, achievable_gain, fault_effect
    from fleetlab.log.runfiles import oracle_rows, read_identities, read_table

    world_dir = dirs["world"]
    identities = read_identities(world_dir)
    episodes = {r["episode_id"]: r for r in read_table(world_dir, "episodes").to_pylist()}
    verdicts = {r["episode_id"]: r for r in read_table(world_dir, "verdicts").to_pylist()}
    by_window: dict[str, list[int]] = {n: [0, 0] for n in WINDOWS}
    by_robot: dict[str, list[int]] = {}
    for ident in identities:
        ok = episodes[ident.episode_id]["outcome"] == "success"
        window = plan.window_of(ident.day)
        by_window[window][0] += ok
        by_window[window][1] += 1
        cell = by_robot.setdefault(ident.robot_id, [0, 0])
        cell[0] += ok
        cell[1] += 1
    scoring = [i for i in identities if i.day in plan.scoring_days]
    scoring_ids = {i.episode_id for i in scoring}
    table = OracleTable(scoring, [r for r in oracle_rows(world_dir) if r.episode_id in scoring_ids])
    effects = {}
    for fault in plan.faults:
        if any(fault.fault_id in i.fault_ids for i in scoring):
            gain = fault_effect(fault.fault_id, table)
            effects[fault.fault_id] = {
                "kind": fault.kind, "target": fault.target.scope, "level": fault.level,
                "rescued": gain.rescued, "broken": gain.broken, "net": gain.net,
            }  # fmt: skip
    achievable = {}
    for k in (1, 2, 3):
        gain = achievable_gain(k, table)
        achievable[str(k)] = gain.net
    result: dict[str, Any] = {
        "world_id": plan.world_id,
        "seed": plan.seed,
        "faults": [
            {
                "fault_id": f.fault_id,
                "kind": f.kind,
                "robot": f.target.scope,
                "level": f.level,
                "onset": f.schedule.onset_day,
            }
            for f in plan.faults
        ],
        "success": {k: {"ok": v[0], "n": v[1]} for k, v in by_window.items()},
        "by_robot": {k: {"ok": v[0], "n": v[1]} for k, v in sorted(by_robot.items())},
        "true_success": sum(v["true_success"] for v in verdicts.values()),
        "fault_effects": effects,
        "achievable_net": achievable,
        "runs": {},
    }
    for name in ("world", "live"):
        run_dir = dirs[name]
        if not (run_dir / "run.json").is_file():
            continue
        record = json.loads((run_dir / "run.json").read_text())
        extra = record["extra"]
        result["runs"][name] = {
            "chain_ok": record["chain_ok"],
            "chain_sample": len(record["chain_sample"]),
            "jobs": extra.get("jobs"),
            "jobs_done": extra.get("jobs_done"),
            "throughput": extra.get("throughput"),
            "total_s": extra.get("total_s"),
            "incidents": len(extra.get("incidents") or []),
            "worker_processes": len(extra.get("worker_lives") or []),
            "peak_pss_gib": round((extra.get("peaks") or {}).get("pss_total_mib", 0) / 1024, 2),
            "peak_gtt_gib": round((extra.get("peaks") or {}).get("gtt_gib", 0), 2),
        }
    live_dir = dirs["live"]
    if (live_dir / "run.json").is_file():
        outcomes = read_table(live_dir, "job_outcomes").to_pylist()
        cont = [o for o in outcomes if o["mode"] == "continue"]
        live_eps = {r["episode_id"]: r for r in read_table(live_dir, "episodes").to_pylist()}
        result["live"] = {
            "episodes": len(live_eps),
            "handed_over": sum(1 for r in live_eps.values() if r["outcome"] == "handed_over"),
            "succeeded_alone": sum(1 for r in live_eps.values() if r["outcome"] == "success"),
            "teleoperator_placed": sum(1 for o in cont if o["success"]),
        }
    return result


# ---- the smoke world ---------------------------------------------------------------------------


def _commit() -> str:
    from fleetlab.engine.fleet import code_version

    code = code_version()
    return str(code["commit"])[:12] + ("-dirty" if code["dirty"] else "")


def smoke(out_root: Path, config: Any, *, update: bool = False) -> int:
    """Run the smoke world and check its chains against the stored reference; 0 when equal."""
    from fleetlab.policies.xvla import nominal_spec

    world = smoke_world()
    commit = _commit()
    t = time.perf_counter()
    plan = world.plan(run_software(config), nominal_spec(config.precision), config.batch_size)
    run = run_world(plan, out_root, config, name=f"w30-{commit}", oracle=False, live=False)
    seconds = time.perf_counter() - t
    run_dir = run.dirs["world"]
    record = json.loads((run_dir / "run.json").read_text())
    found = digests(run_dir)
    episodes = {r["episode_id"]: r["outcome"] for r in _read(run_dir, "episodes")}
    current = {
        "world_id": world.world_id,
        "software": record["software"],
        "chains": found,
        "outcomes": episodes,
    }
    summary = {
        "run_dir": str(run_dir),
        "seconds": round(seconds, 1),
        "chain_ok": record["chain_ok"],
        "successes": sum(o == "success" for o in episodes.values()),
    }
    if update or not REFERENCE.is_file():
        REFERENCE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        summary["reference"] = f"written to {REFERENCE}"
        print(json.dumps(summary, indent=1))
        return 0 if record["chain_ok"] else 1
    reference = json.loads(REFERENCE.read_text())
    differ = sorted(k for k in reference["chains"] if found.get(k) != reference["chains"][k])
    missing = sorted(set(reference["chains"]) - set(found))
    summary.update(
        identical=len(reference["chains"]) - len(differ),
        compared=len(reference["chains"]),
        differ=differ,
        missing=missing,
        software_changed=sorted(
            k for k, v in reference["software"].items() if record["software"].get(k) != v
        ),
    )
    print(json.dumps(summary, indent=1))
    ok = not differ and not missing and record["chain_ok"]
    if not ok:
        print("smoke world: the chains differ from the stored reference", file=sys.stderr)
    return 0 if ok else 1


def _read(run_dir: Path, table: str) -> list[dict[str, Any]]:
    from fleetlab.log.runfiles import read_table

    rows: list[dict[str, Any]] = read_table(run_dir, table).to_pylist()
    return rows


# ---- the nightly worlds ------------------------------------------------------------------------


def night_scores(night: Path) -> dict[str, Any]:
    path = night / "scores.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def previous_night(root: Path, night: Path) -> Path | None:
    earlier = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.name < night.name and (d / "scores.json").is_file()
    )
    return earlier[-1] if earlier else None


def nightly(
    root: Path,
    config: Any,
    *,
    date: str | None = None,
    canary: bool = True,
    per_day: int | None = None,
) -> Path:
    """Both nightly worlds, the canary, the scores and the comparison with the previous night.

    `per_day` shrinks both worlds for a dry run of the whole pipeline; such a
    night belongs in a root of its own, apart from the real nights.
    """
    from fleetlab.engine import fleet

    date = date or datetime.date.today().isoformat()
    night = root / date
    night.mkdir(parents=True, exist_ok=True)
    seed = int(date.replace("-", ""))
    started = time.time()
    soft = run_software(config)
    days = {} if per_day is None else {"per_day": per_day}
    worlds = {
        "deterministic": generated_plan(nightly_spec(**days), soft),
        "randomized": generated_plan(nightly_spec(seed, **days), soft),
    }
    timing: dict[str, Any] = {}
    hashes: dict[str, str | None] = {}
    dirs: dict[str, dict[str, Path]] = {}
    for name, world in worlds.items():
        run = run_world(world, night / name, config, name=f"{world.world_id}-{date}")
        timing[name] = run.seconds
        hashes[name] = run.world_hash
        dirs[name] = run.dirs
    canary_dir = night / f"canary-{date}"
    if canary:
        # The first episodes of the fixed world again, hours after they ran.
        first = [
            Job(identity=j.identity, mode=JobMode.replay)
            for j in _world_jobs(dirs["deterministic"]["world"])[:CANARY]
        ]
        t = time.perf_counter()
        canary_config = replace(
            config,
            out=str(canary_dir),
            episodes_per_robot_day=worlds["deterministic"].per_day,
            chain_sample=False,
            workers=min(fitting_workers(config), CANARY),
        )
        fleet.run(canary_config, first)
        timing["canary"] = time.perf_counter() - t
    out: dict[str, Any] = {"date": date, "started": started, "finished": time.time(),
                           "timing": timing}  # fmt: skip
    for name, world in worlds.items():
        out[name] = scores(world, dirs[name])
        out[name]["world_hash"] = hashes[name]
        out[name]["world_files"] = world_file_hashes(night / name / "files")
        out[name]["generator_source"] = generator_source_hash()
        out[name]["digests"] = {
            run: digests(dirs[name][run]) for run in ("world", "live")
        }  # fmt: skip
    if canary:
        out["canary"] = canary_check(dirs["deterministic"]["world"], canary_dir)
    out["run_dirs"] = [str(d) for v in dirs.values() for d in v.values()]
    (night / "scores.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    previous = previous_night(root, night)
    comparison = compare_nights(out, night_scores(previous) if previous else None, previous)
    (night / "comparison.json").write_text(json.dumps(comparison["json"], indent=1))
    (night / "comparison.md").write_text(comparison["markdown"])
    return night


def world_file_hashes(files: Path) -> dict[str, str]:
    """Each of the four hashed world files' own SHA-256 (16 digits): which one changed."""
    from fleetlab.contracts.world import WORLD_FILES

    return {
        name: hashlib.sha256((files / name).read_bytes()).hexdigest()[:16]
        for name in WORLD_FILES
        if (files / name).is_file()
    }


def generator_source_hash() -> str:
    """A hash over the generator's sources (`fleetlab.worlds`): did the generator change?"""
    import fleetlab.worlds

    h = hashlib.sha256()
    for path in sorted(Path(fleetlab.worlds.__file__).parent.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()[:16]


def _world_jobs(run_dir: Path) -> list[Job]:
    from fleetlab.log.runfiles import read_identities

    return [Job(identity=i) for i in read_identities(run_dir)]


def canary_check(world_dir: Path, canary_dir: Path) -> dict[str, Any]:
    from fleetlab.engine.chain import Chain, compare

    world = RunLayout(world_dir)
    canary = RunLayout(canary_dir)
    same = 0
    detail = []
    for job in _world_jobs(world.root)[:CANARY]:
        replay = Job(identity=job.identity, mode=JobMode.replay)
        result = compare(Chain.load(canary.chain(replay)), Chain.load(world.chain(job)))
        same += result.identical
        detail.append({"episode_id": job.identity.episode_id, "identical": result.identical})
    return {"compared": len(detail), "identical": same, "detail": detail}


def compare_nights(
    tonight: dict[str, Any], before: dict[str, Any] | None, before_dir: Path | None
) -> dict[str, Any]:
    """The comparison of a night with the one before: the identity check, the scores, the gap."""
    det, ran = tonight["deterministic"], tonight["randomized"]
    identity: dict[str, Any] = {"previous": str(before_dir) if before_dir else None}
    if before:
        for run in ("world", "live"):
            a, b = det["digests"].get(run, {}), before["deterministic"]["digests"].get(run, {})
            keys = sorted(set(a) | set(b))
            differ = [k for k in keys if a.get(k) != b.get(k)]
            identity[run] = {"compared": len(keys), "identical": len(keys) - len(differ),
                             "differ": differ[:20]}  # fmt: skip
        identity["world_hash_same"] = det.get("world_hash") == before["deterministic"].get(
            "world_hash"
        )
        now_files, then_files = det.get("world_files"), before["deterministic"].get("world_files")
        identity["world_files_changed"] = (
            sorted(
                f for f in set(now_files) | set(then_files) if now_files.get(f) != then_files.get(f)
            )
            if now_files and then_files
            else None
        )
        now_gen, then_gen = (
            det.get("generator_source"),
            before["deterministic"].get("generator_source"),
        )
        identity["generator_changed"] = None if not now_gen or not then_gen else now_gen != then_gen
        identity["outcomes_identical"] = (
            det["success"] == before["deterministic"]["success"]
            and det["by_robot"] == before["deterministic"]["by_robot"]
            and det["fault_effects"] == before["deterministic"]["fault_effects"]
        )
    lines = [f"# Nightly worlds, {tonight['date']}", ""]
    hours = (tonight["finished"] - tonight["started"]) / 3600
    lines += [f"Both worlds and the canary in {hours:.1f} hours.", ""]
    if before:
        world_id = identity["world"]
        live_id = identity["live"]
        episodes_same = (
            not world_id["differ"] and not live_id["differ"] and identity["outcomes_identical"]
        )
        if episodes_same and identity["world_hash_same"]:
            verdict = "identical"
        elif episodes_same:
            verdict = "chains and outcomes identical, world files **changed**"
        else:
            verdict = "**different**"
        lines += [
            f"**Identity check of w20 against {Path(str(before_dir)).name}: {verdict}.** "
            f"Chains of the world run {world_id['identical']} of {world_id['compared']} identical, "
            f"of the live pass {live_id['identical']} of {live_id['compared']}; outcomes and "
            f"fault effects {'the same' if identity['outcomes_identical'] else 'changed'}; "
            f"world files {'the same' if identity['world_hash_same'] else 'changed'}.",
            "",
        ]
        if not identity["world_hash_same"]:
            changed = identity["world_files_changed"]
            which = (
                ", ".join(changed)
                if changed is not None
                else "which of them is not known (the earlier night kept only the world hash)"
            )
            gen = {
                True: "The generator's sources changed between the two nights",
                False: "The generator's sources are the same",
                None: "Whether the generator changed is not recorded for the earlier night",
            }[identity["generator_changed"]]
            lines += [
                f"World files changed: {which}. {gen}. This night is the reference for the "
                "world hash from now on; a second change without a change to the generator is "
                "a regression.",
                "",
            ]
    else:
        lines += ["No earlier night to compare w20 with: this night is the first reference.", ""]
    if "canary" in tonight:
        c = tonight["canary"]
        lines += [
            f"Canary: the first {c['compared']} episodes of w20 replayed after both worlds, "
            f"{c['identical']} of {c['compared']} identical.",
            "",
        ]
    lines += ["| | w20 (fixed seed) | w21 (seed " + str(ran["seed"]) + ") |", "|---|---|---|"]
    for window in ("observed", "experiment", "scoring"):
        a, b = det["success"][window], ran["success"][window]
        lines.append(f"| success, {window} day | {a['ok']}/{a['n']} | {b['ok']}/{b['n']} |")
    for k in ("1", "2", "3"):
        lines.append(
            f"| best net gain with {k} fix{'es' if k != '1' else ''} (scoring day) | "
            f"{det['achievable_net'][k]} | {ran['achievable_net'][k]} |"
        )
    if "live" in det and "live" in ran:
        a, b = det["live"], ran["live"]
        lines.append(
            f"| live pass: handed over / placed by the teleoperator | "
            f"{a['handed_over']} / {a['teleoperator_placed']} | "
            f"{b['handed_over']} / {b['teleoperator_placed']} |"
        )
    for run in ("world", "live"):
        a, b = det["runs"].get(run, {}), ran["runs"].get(run, {})
        if a and b:
            lines.append(
                f"| {run} run: jobs done with the chain sample, sample held, minutes | "
                f"{a['jobs_done']}, "
                f"{a['chain_ok']}, {a['total_s'] / 60:.0f} | {b['jobs_done']}, {b['chain_ok']}, "
                f"{b['total_s'] / 60:.0f} |"
            )
    lines += ["", "Faults of w21 tonight: " + "; ".join(
        f"{f['kind']} {f['level']:g} on {f['robot']} from day {f['onset']}" for f in ran["faults"]
    ) + ".", ""]  # fmt: skip
    if before:
        b = before["deterministic"]
        lines += [
            "w20's scores the night before: "
            + ", ".join(f"{w} {b['success'][w]['ok']}/{b['success'][w]['n']}" for w in b["success"])
            + ".",
            "",
        ]
    return {"json": {"date": tonight["date"], "identity": identity}, "markdown": "\n".join(lines)}


# ---- the soak record ---------------------------------------------------------------------------


def soak_summary(run_dir: Path) -> dict[str, Any]:
    """Memory over a run and its workers' lives, from run.json: what grows, what is replaced.

    The slope is a least-squares line through the server's PSS over the run,
    in GiB per hour; the workers' PSS at the end of a life is compared between
    the first and the last quarter of their lives.
    """
    record = json.loads((run_dir / "run.json").read_text())
    extra = record["extra"]
    # Older runs recorded losses and completed job details, without a recovery flag.
    # Infer recovery only with both a completion record and the stored chain; leave
    # their files untouched. A run's sample holding does not prove a continuation.
    done = {j["key"] for j in extra.get("jobs_detail") or []}
    incidents = [
        i
        if "recovered" in i
        else {
            **i,
            "recovered": all(
                k in done and RunLayout(run_dir).table("chain", k).is_file()
                for k in i.get("jobs") or []
            ),
            "recovery_inferred": True,
        }
        for i in extra.get("incidents") or []
    ]
    series = np.asarray(extra.get("memory_series") or [], dtype=float).reshape(-1, 7)
    lives = extra.get("worker_lives") or []
    out: dict[str, Any] = {
        "run": str(run_dir),
        "hours": round(float(extra.get("total_s") or 0.0) / 3600, 2),
        "jobs_done": extra.get("jobs_done"),
        "chain_ok": record["chain_ok"],
        "incidents": len(incidents),
        "incidents_recovered": sum(i.get("recovered") is True for i in incidents),
        "incidents_detail": incidents,
        "worker_lives": len(lives),
        "samples": int(series.shape[0]),
    }
    if series.shape[0] >= 2:
        t_h = series[:, 0] / 3600
        server = series[:, 3]
        slope = float(np.polyfit(t_h, server, 1)[0]) if np.ptp(t_h) > 0 else 0.0
        out.update(
            server_pss_gib={"first": server[0], "last": server[-1], "max": server.max()},
            server_slope_gib_per_h=round(slope, 3),
            workers_pss_max_gib=float(series[:, 4].max()),
            largest_worker_max_gib=float(series[:, 6].max()),
            mem_available_min_gib=float(series[:, 2].min()),
            gtt_max_gib=float(series[:, 1].max()),
        )
    if len(lives) >= 4:
        quarter = max(1, len(lives) // 4)
        ends = [float(b.get("pss_mib") or 0.0) / 1024 for b in lives]
        out["worker_pss_at_end_gib"] = {
            "first_quarter_median": round(float(np.median(ends[:quarter])), 3),
            "last_quarter_median": round(float(np.median(ends[-quarter:])), 3),
        }
    return out


# ---- the soak test ------------------------------------------------------------------------------

SOAK_STATE = "soak.json"
SOAK_MAX_INCIDENTS_PER_HOUR = 1.0


def soak_plan(seed: int, per_day: int | None = None) -> Any:
    """The plan maker of the soak: a randomized nightly world of `seed`, from the generator."""

    def make(soft: SoftwareVersion) -> WorldPlan:
        spec = nightly_spec(seed) if per_day is None else nightly_spec(seed, per_day=per_day)
        return generated_plan(spec, soft)

    return make


def soak_run(
    root: Path,
    config: Any,
    *,
    hours: float = 6.0,
    base_seed: int,
    per_day: int | None = None,
    max_worlds: int = 100,
    plans: Any = None,
    soft: SoftwareVersion | None = None,
) -> dict[str, Any]:
    """Randomized nightly worlds one after another until `hours` of running, then the record.

    World k has the seed `base_seed * 100 + k`, its 480 episodes as run, the
    oracle's replays and the live pass (`run_world`), and after it the canary:
    world 0's first ten episodes replayed and compared with their chains, so
    the last canary compares episodes hours apart. A world starts while the
    worlds done so far ran less than `hours`. `soak.json` keeps the worlds done
    and every session; a rerun after a kill resumes the world it stopped in
    (the runs are resumable) and goes on, and a rerun after the end only
    writes the record again (`soak-summary.json`, `soak-summary.md`).
    `plans(k)` replaces the generator's worlds (tests).
    """
    from fleetlab.engine import fleet

    root.mkdir(parents=True, exist_ok=True)
    state_path = root / SOAK_STATE
    if state_path.is_file():
        state = json.loads(state_path.read_text())
    else:
        state = {"base_seed": base_seed, "hours": hours, "per_day": per_day, "worlds": [],
                 "sessions": []}  # fmt: skip
    session: dict[str, Any] = {"started": time.time(), "code": _commit(), "worlds": []}
    state["sessions"].append(session)
    _save(state_path, state)
    soft = soft or run_software(config)
    worlds: list[dict[str, Any]] = state["worlds"]
    k = 0
    while k < max_worlds:
        if k < len(worlds) and worlds[k].get("finished"):
            k += 1
            continue
        if sum(w.get("seconds", 0.0) for w in worlds) >= state["hours"] * 3600:
            break
        seed = state["base_seed"] * 100 + k
        if k == len(worlds):
            worlds.append({"index": k, "seed": seed})
        entry = worlds[k]
        entry["started"] = time.time()
        session["worlds"].append(k)
        _save(state_path, state)
        make = plans(k) if plans is not None else soak_plan(seed, state["per_day"])
        plan = make(soft)
        wroot = root / f"{k:02d}"
        t = time.perf_counter()
        run = run_world(plan, wroot, config, name=f"soak-{state['base_seed']}-{k:02d}")
        first = run_dirs(root / "00", f"soak-{state['base_seed']}-00")["world"]
        canary_dir = wroot / f"soak-{state['base_seed']}-{k:02d}-canary"
        c = time.perf_counter()
        replays = [Job(identity=j.identity, mode=JobMode.replay) for j in _world_jobs(first)]
        fleet.run(
            replace(
                config,
                out=str(canary_dir),
                episodes_per_robot_day=plan.per_day,
                chain_sample=False,
                workers=min(fitting_workers(config), CANARY),
            ),
            replays[:CANARY],
        )
        timing = {**run.seconds, "canary": time.perf_counter() - c}
        entry.update(
            world_id=plan.world_id,
            faults=[
                {
                    "kind": f.kind,
                    "robot": f.target.scope,
                    "level": f.level,
                    "onset": f.schedule.onset_day,
                }
                for f in plan.faults
            ],
            dirs={**{n: str(d) for n, d in run.dirs.items()}, "canary": str(canary_dir)},
            timing=timing,
            seconds=time.perf_counter() - t,
            canary=canary_check(first, canary_dir),
            finished=time.time(),
        )
        _save(state_path, state)
        k += 1
    record = soak_record(state)
    (root / "soak-summary.json").write_text(json.dumps(record, indent=1, default=float))
    (root / "soak-summary.md").write_text(soak_markdown(record))
    return record


def _save(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=1))
    temporary.replace(path)


def _slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or max(x) - min(x) <= 0:
        return None
    return round(float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0]), 4)


def soak_record(state: dict[str, Any]) -> dict[str, Any]:
    """The soak's record over all its runs: memory, workers, GPU, incidents, throughput, chains.

    Hours are counted from the start of the first world. Per run: its start,
    the server's PSS (first, last, max sample), the workers' PSS at the end of
    their lives (median), GPU memory and memory available at their extremes,
    incidents (a worker that died, failed or fell silent past the watchdog),
    throughput, the chain sample; per world the canary.
    """
    worlds = [w for w in state["worlds"] if w.get("finished")]
    if not worlds:
        return {"worlds": 0, "passed": False, "failed_conditions": ["No world finished."]}
    t0 = worlds[0]["started"]
    runs: list[dict[str, Any]] = []
    for w in worlds:
        offset = w["started"] - t0
        for part in ("world", "live", "canary"):
            run_dir = Path(w["dirs"][part])
            summary = soak_summary(run_dir)
            extra = json.loads((run_dir / "run.json").read_text())["extra"]
            lives = [float(b.get("pss_mib") or 0.0) / 1024 for b in extra.get("worker_lives") or []]
            runs.append(
                {
                    "world": w["index"],
                    "part": part,
                    "hour": round(offset / 3600, 3),
                    "jobs_done": summary["jobs_done"],
                    "minutes": round(w["timing"][part] / 60, 2),
                    "per_hour": (extra.get("throughput") or {}).get("per_hour_overall"),
                    "chain_ok": summary["chain_ok"],
                    "incidents": summary["incidents"],
                    "incidents_recovered": summary["incidents_recovered"],
                    "incidents_detail": summary["incidents_detail"],
                    "worker_lives": summary["worker_lives"],
                    "worker_pss_end_median_gib": round(float(np.median(lives)), 3)
                    if lives
                    else None,
                    "server_pss_gib": summary.get("server_pss_gib"),
                    "server_pss_peak_gib": round(
                        float((extra.get("peaks") or {}).get("pss_server_mib", 0)) / 1024, 3
                    ),
                    "gtt_max_gib": summary.get("gtt_max_gib"),
                    "mem_available_min_gib": summary.get("mem_available_min_gib"),
                }
            )
            offset += w["timing"][part]
    world_runs = [r for r in runs if r["part"] == "world"]
    last = worlds[-1]
    wall_h = (last["finished"] - t0) / 3600
    hours = [r["hour"] for r in runs]
    gtt = [r["gtt_max_gib"] for r in runs if r["gtt_max_gib"] is not None]
    avail = [r["mem_available_min_gib"] for r in runs if r["mem_available_min_gib"] is not None]
    canaries = [w["canary"] for w in worlds]
    running_h = sum(w["seconds"] for w in worlds) / 3600
    incidents = sum(r["incidents"] for r in runs)
    recovered = sum(r["incidents_recovered"] for r in runs)
    record = {
        "worlds": len(worlds),
        # sessions that ran a world: one, if the soak ran unattended from start to end
        "sessions": sum(1 for s in state["sessions"] if s.get("worlds")),
        "hours_wall": round(wall_h, 2),
        "hours_running": round(running_h, 2),
        "hours_running_exact": running_h,
        "hours_required": state["hours"],
        "jobs_done": sum(r["jobs_done"] or 0 for r in runs),
        "incidents": incidents,
        "incidents_recovered": recovered,
        "incidents_unrecovered": incidents - recovered,
        "incidents_per_hour": incidents / running_h if running_h > 0 else None,
        "incidents_per_hour_limit": SOAK_MAX_INCIDENTS_PER_HOUR,
        "incidents_detail": [
            {**i, "world": r["world"], "part": r["part"]}
            for r in runs
            for i in r["incidents_detail"]
        ],
        "chain_samples_present": sum(1 for r in runs if r["chain_ok"] is not None),
        "chain_samples_held": sum(1 for r in runs if r["chain_ok"] is True),
        "chain_samples_broken": [
            f"{r['world']}/{r['part']}" for r in runs if r["chain_ok"] is False
        ],
        "canaries": {
            "worlds": len(canaries),
            "compared": sum(c["compared"] for c in canaries),
            "identical": sum(c["identical"] for c in canaries),
        },
        "server_pss_peak_gib": {
            "first": runs[0]["server_pss_peak_gib"],
            "last": runs[-1]["server_pss_peak_gib"],
            "max": max(r["server_pss_peak_gib"] for r in runs),
            "slope_per_hour": _slope(hours, [r["server_pss_peak_gib"] for r in runs]),
        },
        "worker_pss_end_median_gib": {
            "first": world_runs[0]["worker_pss_end_median_gib"],
            "last": world_runs[-1]["worker_pss_end_median_gib"],
            "slope_per_hour": _slope(
                [r["hour"] for r in world_runs],
                [r["worker_pss_end_median_gib"] or 0.0 for r in world_runs],
            ),
        },
        "gtt_max_gib": max(gtt) if gtt else None,
        "mem_available_min_gib": min(avail) if avail else None,
        "throughput_world_runs": {
            "first": world_runs[0]["per_hour"],
            "last": world_runs[-1]["per_hour"],
            "slope_per_hour": _slope(
                [r["hour"] for r in world_runs], [r["per_hour"] or 0.0 for r in world_runs]
            ),
        },
        "worker_lives": sum(r["worker_lives"] for r in runs),
        "runs": runs,
        "worlds_detail": [
            {
                "index": w["index"],
                "seed": w["seed"],
                "faults": w["faults"],
                "canary": f"{w['canary']['identical']}/{w['canary']['compared']}",
            }
            for w in worlds
        ],
    }
    conditions = [
        (
            running_h >= state["hours"],
            f"Hours requirement not reached ({state['hours']:g} required).",
        ),
        (record["sessions"] == 1, f"Expected one session, recorded {record['sessions']}."),
        (
            record["incidents_unrecovered"] == 0,
            f"Unrecovered incidents: {record['incidents_unrecovered']}.",
        ),
        (
            record["incidents_per_hour"] is not None
            and record["incidents_per_hour"] <= SOAK_MAX_INCIDENTS_PER_HOUR,
            f"Incident rate exceeds {SOAK_MAX_INCIDENTS_PER_HOUR:g} per running hour."
            if record["incidents_per_hour"] is not None
            else "Incident rate is unavailable.",
        ),
        (
            not record["chain_samples_broken"],
            f"Broken chain samples: {', '.join(record['chain_samples_broken'])}.",
        ),
        (
            record["canaries"]["identical"] == record["canaries"]["compared"],
            f"Canaries identical: {record['canaries']['identical']} "
            f"of {record['canaries']['compared']}.",
        ),
    ]
    record["failed_conditions"] = [why for passed, why in conditions if not passed]
    record["passed"] = not record["failed_conditions"]
    return record


def soak_markdown(record: dict[str, Any]) -> str:
    if not record.get("worlds"):
        return "# Soak test\n\nNo world finished.\n"
    s, w, t = (record["server_pss_peak_gib"], record["worker_pss_end_median_gib"],
               record["throughput_world_runs"])  # fmt: skip
    rate = record["incidents_per_hour"]
    rate_text = f"{rate:.2f}" if rate is not None else "not available"
    limit = record["incidents_per_hour_limit"]
    if rate is not None and rate > limit and float(rate_text) <= limit:
        rate_text = f">{limit:.2f}"
    hours_text = str(record["hours_running"])
    if record["hours_running_exact"] < record["hours_required"] <= record["hours_running"]:
        hours_text = f"<{record['hours_required']:g}"
    lines = [
        "# Soak test",
        "",
        f"{record['worlds']} worlds, {hours_text} hours running, "
        f"{record['hours_wall']} hours wall clock, {record['sessions']} session(s), "
        f"{record['jobs_done']} jobs, {record['worker_lives']} worker lives; "
        f"incidents {record['incidents']} ({record['incidents_recovered']} recovered, "
        f"{record['incidents_unrecovered']} unrecovered), "
        f"{rate_text} per running hour "
        f"(limit {record['incidents_per_hour_limit']:g}); "
        f"chain samples held: {record['chain_samples_held']} of "
        f"{record['chain_samples_present']} runs with a sample; broken: "
        f"{', '.join(record['chain_samples_broken']) or 'none'}; canary "
        f"{record['canaries']['identical']} of {record['canaries']['compared']} identical. "
        f"Passed: {'yes' if record['passed'] else 'no'}.",
        "",
        f"Server PSS peak per run: {s['first']} GiB in the first run, {s['last']} in the last, "
        f"at most {s['max']}, slope {s['slope_per_hour']} GiB per hour. Workers' PSS at the "
        f"end of a life (median per world run): {w['first']} then {w['last']} GiB, slope "
        f"{w['slope_per_hour']} GiB per hour. GPU memory at most {record['gtt_max_gib']} GiB; "
        f"memory available at least {record['mem_available_min_gib']} GiB. Throughput of the "
        f"world runs: {t['first'] or 0:.0f} then {t['last'] or 0:.0f} episodes an hour, slope "
        f"{t['slope_per_hour']} per hour.",
    ]
    if record["failed_conditions"]:
        lines.extend(["", "Failed conditions: " + " ".join(record["failed_conditions"])])
    for incident in record["incidents_detail"]:
        cause = (incident.get("why") or "unknown").splitlines()[-1]
        recovery = "recovered" if incident.get("recovered") is True else "unrecovered"
        lines.extend(
            [
                "",
                f"Incident in world {incident['world']}/{incident['part']}, "
                f"worker {incident.get('worker')}: {recovery}; cause: {cause}",
            ]
        )
        if "exitcode" in incident:
            exit_text = (
                str(incident["exitcode"]) if incident["exitcode"] is not None else "not recorded"
            )
            lines.append(
                f"Worker exit status: {exit_text}; "
                f"parent requested kill: {'yes' if incident.get('parent_killed') else 'no'}."
            )
        if incident.get("worker_error"):
            lines.append(f"Worker report: {incident['worker_error'].splitlines()[-1]}")
        for key in incident.get("jobs") or []:
            attempt = (incident.get("attempts") or {}).get(key)
            check = (incident.get("replay_checks") or {}).get(key)
            attempt_text = (
                f"lost assignment {attempt}"
                if attempt is not None
                else "assignment loss not recorded"
            )
            status = (incident.get("assignment_status") or {}).get(key)
            status_text = {
                "not_delivered": "job was not delivered",
                "execution_unconfirmed": "execution start and outcome unconfirmed",
                "execution_started": "execution started, completion unconfirmed",
                "completed_report_late": "completion report arrived after the loss",
            }.get(str(status), "execution outcome not recorded")
            if check is None:
                replay_text = "replay check not recorded"
            elif check.get("checked"):
                replay_text = f"replay checked, verdict {check.get('verdict') or 'not recorded'}"
            else:
                replay_text = "not checked by replay"
            lines.append(f"Job {key}: {attempt_text}; {status_text}; {replay_text}.")
    lines.extend(
        [
            "",
            "| world | part | hour | jobs | minutes | per hour | server PSS peak | worker PSS at "
            "end of life | GPU max | available min | incidents | chain sample |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in record["runs"]:
        per_hour = f"{r['per_hour']:.0f}" if r["per_hour"] else ""
        sample_text = {True: "held", False: "broken", None: "no sample"}[r["chain_ok"]]
        lines.append(
            f"| {r['world']} | {r['part']} | {r['hour']:.2f} | {r['jobs_done']} | "
            f"{r['minutes']:.1f} | {per_hour} | {r['server_pss_peak_gib']} | "
            f"{r['worker_pss_end_median_gib']} | {r['gtt_max_gib']} | "
            f"{r['mem_available_min_gib']} | {r['incidents']} | {sample_text} |"
        )
    lines += ["", "| world | seed | faults | canary |", "|---|---|---|---|"]
    for d in record["worlds_detail"]:
        faults = "; ".join(f"{f['kind']} {f['level']:g} on {f['robot']} from day {f['onset']}"
                           for f in d["faults"])  # fmt: skip
        lines.append(f"| {d['index']} | {d['seed']} | {faults} | {d['canary']} |")
    return "\n".join(lines) + "\n"


# ---- the command line --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.engine.ladder")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="the smoke world, checked against the stored reference")
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--update-reference", action="store_true")
    n = sub.add_parser("nightly", help="both nightly worlds and the comparison with last night")
    n.add_argument(
        "--root", type=Path, default=Path.home() / ".cache" / "robots-probes" / "nightly"
    )
    n.add_argument("--date", default=None)
    n.add_argument("--no-canary", action="store_true")
    n.add_argument("--per-day", type=int, default=None, help="a dry run: fewer episodes a day")
    k = sub.add_parser("soak", help="memory and workers' lives of the runs of a night")
    k.add_argument("runs", type=Path, nargs="+")
    r = sub.add_parser("soak-run", help="randomized worlds until HOURS of running")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--hours", type=float, default=6.0)
    r.add_argument("--base-seed", type=int, default=None, help="default: today's date")
    r.add_argument("--per-day", type=int, default=None, help="a dry run: fewer episodes a day")
    r.add_argument("--max-worlds", type=int, default=100)
    for p in (s, n, r):
        p.add_argument("--workers", type=int, default=10)
        p.add_argument("--miopen-dir", type=Path, default=None)
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["soak"]:
        runs = parser.parse_args(argv).runs
        print(json.dumps([soak_summary(r) for r in runs], indent=1, default=float))
        return 0
    args = parser.parse_args(argv)
    import os

    if args.miopen_dir is not None:
        os.environ["FLEETLAB_MIOPEN_DIR"] = str(args.miopen_dir)
    from fleetlab.runtime import configure_process, data_dir

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.engine.fleet import RunConfig

    config = RunConfig(out="", workers=args.workers, batch_size=4, precision="fp16_res32")
    if args.command == "smoke":
        out = args.out or data_dir() / "smoke-world"
        return smoke(
            out, replace(config, workers=min(args.workers, 8)), update=args.update_reference
        )
    if args.command == "soak-run":
        base = args.base_seed or int(datetime.date.today().isoformat().replace("-", ""))
        record = soak_run(
            args.root,
            config,
            hours=args.hours,
            base_seed=base,
            per_day=args.per_day,
            max_worlds=args.max_worlds,
        )
        print((args.root / "soak-summary.md").read_text())
        return 0 if record.get("passed") else 1
    night = nightly(
        args.root, config, date=args.date, canary=not args.no_canary, per_day=args.per_day
    )
    print((night / "comparison.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
