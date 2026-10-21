"""Dose-response calibration of the fault types (docs/design/faults.md).

A fault type's curve is its success rate at each of its calibration points
(`calibration` on the type: a level with its params and target), measured on
the same episodes as a healthy run: the same tasks, start states, scene seeds
and noise seeds, so that every faulted episode has a healthy twin and the
comparison is paired. The episodes run through the stage-2 rollout engine in
the fleet's precision; this module builds the manifests and reads the runs.

    # the manifests: one healthy, one per fault type (cheap; reads the engine's software)
    uv run --locked python -m fleetlab.dose_response plan healthy CAL/healthy.jsonl
    uv run --locked python -m fleetlab.dose_response plan camera_pose_offset CAL/camera.jsonl

    # each run is a heavy job through the engine
    tools/heavy-run uv run --locked python -m fleetlab.engine run MANIFEST RUN_DIR

    # or the whole calibration of docs/design/faults.md (`CALIBRATION_PLAN`), one heavy job,
    # resumable, into ~/.cache/robots-probes/faults/runs/<name>
    tools/heavy-run uv run --locked python -m fleetlab.dose_response run [NAME ...]

    # the tables (cheap)
    uv run --locked python -m fleetlab.dose_response analyse --healthy RUN_DIR KIND=RUN_DIR ...

Episodes: `episodes` per point, spread evenly over the ten tasks of the suite,
start states 0 to `episodes / 10 - 1`. Identities: world `w90`, robot `r01`
at site `s1`, day 1; the episode index is the episode's number within the
point, so the seeds (`seeds_for(WORLD_SEED, "r01", 1, index)`) are the same
for every point and for the healthy run. Episode ids encode the point:
`e` + block (three digits) + index (three digits), block 0 being healthy and
the blocks of a type fixed by the order of `CALIBRATED`. The fault is `f01`
at the narrowest scope its type allows (the robot, or its site).

Statistics per point (`compare`): the success rate with Wilson's 95%
interval; against the healthy twins, the episodes broken and rescued, the
exact McNemar test, and the paired difference with Newcombe's 95% score
interval (method 10), as spikes/xvla-speed section 8 computed them.

Stage 3 adds, without changing an episode id of stage 2:

- the new kinds, appended to `CALIBRATED`, and points appended to old kinds;
- episode ranges: a point extended from 300 to 1,000 pairs runs indices 300
  to 999 only (`first`), into its own run directory;
- companions: a point may run with points of other types in the same
  episodes (`DosePoint.companions`, faults `f02`, ...); a data fault measured
  on top of a fault that makes failures is compared with the same episodes'
  true outcome, which its data hook cannot change;
- joints (`JOINTS`): two calibration points of two types together, in blocks
  from `JOINT_BLOCK`, to measure an interaction against the singles on the
  same episodes (`interaction`);
- suites: `libero_object` episodes in blocks shifted by 500
  (`SUITE_BLOCK_OFFSET`), so that ids stay unique within world `w90`;
- the stage-3 plan, its runner and its analysis, which writes the curves
  into `fleetlab/faults/curves.json`, in `fleetlab.dose_response_stage3`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetlab.contracts.faults import FaultSpec, Schedule
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.ids import IdKind, ScopeKind, id_number, neutral_id
from fleetlab.contracts.rollout import Job, RunLayout

WORLD_SEED = 90
WORLD_ID = "w90"
ROBOT_ID, SITE_ID, DAY = "r01", "s1", 1
FAULT_ID = "f01"
SUITE = "libero_spatial"
MAX_STEPS = 280
IMAGE_SIZE = 360
BLOCK_WIDTH = 1000  # episodes per point at most

# The calibrated kinds, in a fixed order that numbers their blocks; the two generic
# change kinds act through these and have no curve of their own. Append only: a kind
# inserted in the middle would renumber the blocks of the kinds after it, and runs
# made before would be read as the wrong points. A point is likewise appended to a
# type's `calibration`, never inserted.
CALIBRATED: tuple[str, ...] = (
    "camera_pose_offset",
    "camera_degradation",
    "action_latency",
    "policy_knob",
    "control_mode_change",
    "software_release",
    "gripper_wear",
    "object_batch_variant",
    "furniture_shift",
    "illumination_change",
    "backdrop_change",
    "success_detector_error",
    # stage 3
    "lens_contamination",
    "camera_latency",
    "gripper_force_loss",
    "gripper_sticky_release",
    "arm_torque_loss",
    "controller_overload",
    "calibration_update",
    "image_pipeline_release",
    "table_height_change",
    "light_color_shift",
    "success_false_positive",
)
POINTS_PER_KIND = 10
# Blocks of joint points (two faults together), and the shift of another suite's blocks.
JOINT_BLOCK = 400
SUITE_BLOCK_OFFSET = {"libero_spatial": 0, "libero_object": 500}


@dataclass(frozen=True)
class Joint:
    """Two calibration points, (kind, label) each, run together in the same episodes."""

    name: str
    members: tuple[tuple[str, str], tuple[str, str]]


# Append only, like `CALIBRATED`: a joint's block is its place here.
JOINTS: tuple[Joint, ...] = (
    Joint(
        "wear+heavier", (("gripper_wear", "0.5 wear"), ("object_batch_variant", "4 mass doublings"))
    ),
    Joint(
        "weaker+heavier",
        (("gripper_force_loss", "0.25 force lost"), ("object_batch_variant", "4 mass doublings")),
    ),
    # The pair the stage-3 curves suggested: a release carrying both lost 90 points where
    # each alone lost 4 to 6 (faults.md, section 8.5).
    Joint(
        "delay+short-chunks",
        (("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk")),
    ),
)


def joint(name: str) -> Joint:
    for j in JOINTS:
        if j.name == name:
            return j
    raise KeyError(f"no joint named {name!r}")


def calibration_points(kind: str) -> tuple[Any, ...]:
    """The calibration points of a registered fault type."""
    import fleetlab.faults as faults

    points: tuple[Any, ...] = getattr(faults.get(kind), "calibration", ())
    if not points or len(points) >= POINTS_PER_KIND:
        raise ValueError(f"{kind}: {len(points)} calibration points, expected 1 to 9")
    return points


def block_of(kind: str, point_index: int, suite: str = SUITE) -> int:
    """The block number of a point; block 0 is the healthy run (both shifted for `suite`)."""
    block = 1 + CALIBRATED.index(kind) * POINTS_PER_KIND + point_index
    assert block < JOINT_BLOCK
    return block + SUITE_BLOCK_OFFSET[suite]


def healthy_block(suite: str = SUITE) -> int:
    return SUITE_BLOCK_OFFSET[suite]


def joint_block(name: str, suite: str = SUITE) -> int:
    return JOINT_BLOCK + JOINTS.index(joint(name)) + SUITE_BLOCK_OFFSET[suite]


def point_index(kind: str, label: str) -> int:
    """The index of a calibration point by its label."""
    for i, p in enumerate(calibration_points(kind)):
        if p.label == label:
            return i
    raise KeyError(f"{kind} has no calibration point {label!r}")


def episode_id(block: int, index: int) -> str:
    if not 0 <= index < BLOCK_WIDTH:
        raise ValueError(f"episode index {index} outside a block of {BLOCK_WIDTH}")
    return neutral_id(IdKind.episode, block * BLOCK_WIDTH + index)


def split_episode_id(value: str) -> tuple[int, int]:
    """(block, index) of a calibration episode id."""
    number = id_number(IdKind.episode, value)
    return number // BLOCK_WIDTH, number % BLOCK_WIDTH


def episodes(count: int, first: int = 0) -> list[tuple[int, int, int]]:
    """(index, task, start state) of the episodes `first` to `count - 1` of a point.

    Spread evenly over the ten tasks: index i is task i % 10 at start state
    i // 10, so a point extended from 300 to 1,000 pairs adds indices 300 to
    999 and leaves the first 300 as they were.
    """
    if count % 10 or first % 10:
        raise ValueError("the episodes per point are a multiple of ten (ten tasks)")
    return [(i, i % 10, i // 10) for i in range(first, count)]


def fault_spec(kind: str, point: Any, fault_id: str = FAULT_ID) -> FaultSpec:
    """The fault of a calibration point, at the narrowest scope its type allows."""
    import fleetlab.faults as faults
    from fleetlab.faults._common import target_of

    truth = faults.get(kind).truth
    if ScopeKind.robot in truth.scopes:
        scope = ROBOT_ID
    elif ScopeKind.site in truth.scopes:
        scope = SITE_ID
    else:
        scope = "fleet"
    spec = FaultSpec(
        fault_id=fault_id,
        kind=kind,
        params=dict(point.params),
        level=float(point.level),
        target=target_of(scope, point.target, truth.target_kind),
        schedule=Schedule(onset_day=0),
        nominal=point.nominal,
    )
    faults.REGISTRY.check(spec)
    return spec


def make_identity(
    block: int,
    index: int,
    task: int,
    start: int,
    policy: PolicySpec,
    software: SoftwareVersion,
    batch_size: int,
    faults: Sequence[FaultSpec] = (),
    suite: str = SUITE,
) -> EpisodeIdentity:
    scene, noise = seeds_for(WORLD_SEED, ROBOT_ID, DAY, index)
    return EpisodeIdentity(
        world_id=WORLD_ID,
        episode_id=episode_id(block, index),
        robot_id=ROBOT_ID,
        site_id=SITE_ID,
        day=DAY,
        episode_index=index,
        suite=suite,
        task_index=task,
        start_state=start,
        scene_seed=scene,
        noise_seed=noise,
        policy=policy,
        batch_size=batch_size,
        row=None,
        image_size=IMAGE_SIZE,
        max_steps=MAX_STEPS,
        faults=list(faults),
        software=software,
    )


def point_faults(kind: str, point: Any) -> list[FaultSpec]:
    """The faults of a calibration point: its own (`f01`) and its companions' (`f02`, ...)."""
    out = [fault_spec(kind, point)]
    for n, (other, label) in enumerate(getattr(point, "companions", ()), start=2):
        out.append(
            fault_spec(other, calibration_points(other)[point_index(other, label)], f"f{n:02d}")
        )
    return out


def joint_faults(name: str) -> list[FaultSpec]:
    """The two faults of a joint point, `f01` and `f02`."""
    return [
        fault_spec(kind, calibration_points(kind)[point_index(kind, label)], f"f{n:02d}")
        for n, (kind, label) in enumerate(joint(name).members, start=1)
    ]


def plan_jobs(
    kind: str | None,
    count: int,
    policy: PolicySpec,
    software: SoftwareVersion,
    batch_size: int = 4,
    points: Sequence[int] | None = None,
    *,
    first: int = 0,
    suite: str = SUITE,
    joint_name: str | None = None,
) -> list[Job]:
    """The jobs of one manifest: the healthy episodes (`kind` None), a type's points, or a joint.

    Episode indices `first` to `count - 1` of each point (`episodes`).
    """
    chosen = episodes(count, first)

    def jobs_of(block: int, faults: Sequence[FaultSpec]) -> list[Job]:
        return [
            Job(
                identity=make_identity(
                    block, i, task, start, policy, software, batch_size, faults, suite
                )
            )
            for i, task, start in chosen
        ]

    if joint_name is not None:
        return jobs_of(joint_block(joint_name, suite), joint_faults(joint_name))
    if kind is None:
        return jobs_of(healthy_block(suite), [])
    all_points = calibration_points(kind)
    indices = range(len(all_points)) if points is None else points
    jobs: list[Job] = []
    for p in indices:
        jobs.extend(jobs_of(block_of(kind, p, suite), point_faults(kind, all_points[p])))
    return jobs


# ---------------------------------------------------------------------------- reading runs


@dataclass(frozen=True)
class Outcome:
    """One episode as the calibration reads it."""

    episode_id: str
    true_success: bool  # the fleet's check, before the data hook
    logged_success: bool  # the visible outcome, after the data hook
    libero_success: bool  # LIBERO's own check fired at some step
    steps: int
    run_id: str = ""  # the run directory the episode was read from


@dataclass(frozen=True)
class RunOutcomes:
    run_id: str
    chain_ok: bool | None
    outcomes: dict[str, Outcome]


def read_run(run_dir: Path) -> RunOutcomes:
    """The outcomes of a run directory, with its id (the directory's name) and `chain_ok`."""
    import pyarrow.parquet as pq

    layout = RunLayout(run_dir)
    verdicts = pq.read_table(layout.table("verdicts")).to_pylist()
    logged = {r["episode_id"]: r for r in pq.read_table(layout.table("episodes")).to_pylist()}
    out = {}
    for row in verdicts:
        eid = row["episode_id"]
        visible = logged.get(eid)
        if visible is None:
            continue  # a replay of the chain sample, or a job without a visible record
        out[eid] = Outcome(
            episode_id=eid,
            true_success=bool(row["true_success"]),
            logged_success=visible["outcome"] == "success",
            libero_success=row.get("libero_first_success_step") is not None,
            steps=int(visible["steps"]),
            run_id=run_dir.name,
        )
    chain_ok = None
    if layout.run_record.is_file():
        chain_ok = json.loads(layout.run_record.read_text()).get("chain_ok")
    return RunOutcomes(run_id=run_dir.name, chain_ok=chain_ok, outcomes=out)


def merge_runs(runs: Sequence[RunOutcomes]) -> RunOutcomes:
    """Several runs as one: the episodes of all, each run's id, chain_ok only if every one held."""
    if len(runs) == 1:
        return runs[0]
    outcomes: dict[str, Outcome] = {}
    for run in runs:
        clash = set(outcomes) & set(run.outcomes)
        if clash:
            raise ValueError(
                f"runs {run.run_id} and an earlier one share episodes: {sorted(clash)[:3]}"
            )
        outcomes.update(run.outcomes)
    flags = [run.chain_ok for run in runs]
    chain_ok = None if None in flags else all(flag is True for flag in flags)
    return RunOutcomes(
        run_id=" + ".join(run.run_id for run in runs), chain_ok=chain_ok, outcomes=outcomes
    )


# ---------------------------------------------------------------------------- statistics


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson's score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """The paired difference p1 - p2 and Newcombe's 95% interval (1998, method 10).

    a: both succeed; b: only the first; c: only the second; d: neither.
    """
    n = a + b + c + d
    if n == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    product = (a + b) * (c + d) * (a + c) * (b + d)
    # The correlation of the paired outcomes, with Newcombe's continuity correction.
    cross = a * d - b * c
    corrected = cross - n / 2 if cross > n / 2 else (0.0 if cross >= 0 else cross)
    phi = corrected / math.sqrt(product) if product > 0 else 0.0
    theta = p1 - p2
    delta = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    epsilon = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (theta, max(-1.0, theta - delta), min(1.0, theta + epsilon))


@dataclass(frozen=True)
class PointResult:
    label: str
    level: float
    run_id: str
    chain_ok: bool | None
    n: int
    successes: int
    interval: tuple[float, float]
    libero_successes: int
    logged_successes: int
    healthy_successes: int  # of the healthy twins, by the same outcome
    broken: int  # healthy twin succeeded, faulted failed
    rescued: int  # healthy twin failed, faulted succeeded
    p_value: float
    difference: tuple[float, float, float]  # faulted minus healthy, with its interval
    mean_steps: float

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")


def compare(
    healthy: dict[int, Outcome],
    faulted: dict[int, Outcome],
    label: str,
    level: float,
    run: RunOutcomes,
    outcome: Callable[[Outcome], bool] = lambda o: o.true_success,
    twin_outcome: Callable[[Outcome], bool] | None = None,
) -> PointResult:
    """One point against its twins, paired by episode index.

    `twin_outcome` reads the twins (default: `outcome`): a data fault on top of
    a companion is compared with its own episodes' true outcome.
    """
    pairs = sorted(set(healthy) & set(faulted))
    if not pairs:
        raise ValueError(f"{label}: no episode has a healthy twin")
    read_twin = twin_outcome or outcome
    a = b = c = d = 0
    for i in pairs:
        h, f = read_twin(healthy[i]), outcome(faulted[i])
        a += h and f
        b += h and not f
        c += f and not h
        d += not h and not f
    successes = a + c
    # Faulted first: "only the first" is a rescue (c), "only the second" a break (b).
    theta, low, high = newcombe_paired(a, c, b, d)
    return PointResult(
        label=label,
        level=level,
        # The runs this point's episodes came from (several when runs were merged).
        run_id=" + ".join(sorted({faulted[i].run_id or run.run_id for i in pairs})),
        chain_ok=run.chain_ok,
        n=len(pairs),
        successes=successes,
        interval=wilson(successes, len(pairs)),
        libero_successes=sum(faulted[i].libero_success for i in pairs),
        logged_successes=sum(faulted[i].logged_success for i in pairs),
        healthy_successes=a + b,
        broken=b,
        rescued=c,
        p_value=mcnemar_exact(b, c),
        difference=(theta, low, high),
        mean_steps=sum(faulted[i].steps for i in pairs) / len(pairs),
    )


def by_index(run: RunOutcomes, block: int) -> dict[int, Outcome]:
    out = {}
    for eid, outcome in run.outcomes.items():
        b, i = split_episode_id(eid)
        if b == block:
            out[i] = outcome
    return out


def logged_outcome(outcome: Outcome) -> bool:
    return outcome.logged_success


def true_outcome(outcome: Outcome) -> bool:
    return outcome.true_success


def analyse(
    kind: str,
    healthy_run: RunOutcomes,
    run: RunOutcomes,
    logged: bool | None = None,
    suite: str = SUITE,
) -> list[PointResult]:
    """Every calibration point of a type that the run holds, against the healthy run.

    A data fault changes what the fleet logs, not what the robot does, so its
    curve is on the logged outcome (`logged` None picks that for the data
    class); every other type's is on the fleet's check before the data hook.
    A data fault measured on top of companions is compared with the same
    episodes' true outcome, which its hook cannot change: the difference is
    what the detector adds to or takes from the record.
    """
    if healthy_run.chain_ok is not True or run.chain_ok is not True:
        raise ValueError("calibration requires passing replay validation for both runs")
    import fleetlab.faults as faults
    from fleetlab.contracts.vocabulary import FaultClass

    data = faults.get(kind).fault_class is FaultClass.data
    if logged is None:
        logged = data
    outcome = logged_outcome if logged else true_outcome
    healthy = by_index(healthy_run, healthy_block(suite))
    results = []
    for p, point in enumerate(calibration_points(kind)):
        faulted = by_index(run, block_of(kind, p, suite))
        if not faulted:
            continue
        companions: tuple[Any, ...] = getattr(point, "companions", ())
        if data and companions:
            results.append(
                compare(faulted, faulted, point.label, point.level, run, outcome, true_outcome)
            )
        else:
            results.append(compare(healthy, faulted, point.label, point.level, run, outcome))
    return results


# The smallest loss that counts: a cause is detectable from two points of fleet
# success, so a decoy must exclude a loss that large.
DECOY_MARGIN = 0.02


def decoy_level(result: PointResult, alpha: float = 0.05) -> bool:
    """Whether one level is a confirmed decoy: no change beyond noise, and the
    interval of the difference excludes a loss of `DECOY_MARGIN`."""
    return result.p_value >= alpha and result.difference[1] >= -DECOY_MARGIN


def verdict(results: Sequence[PointResult], alpha: float = 0.05) -> str:
    """What the statistics say about a type, before the judgement on realistic levels.

    "kept": some point loses successes beyond noise (exact paired test).
    "decoy": at every point the interval of the difference excludes a loss of
    `DECOY_MARGIN` or more. "decoy at ...": no point loses beyond noise, and
    the named levels are confirmed decoys (`decoy_level`); the others are
    flat but not confirmed. "flat, not yet a decoy": no point differs, but a
    loss that counts is still inside every interval (100 pairs show a loss
    of about 6 to 8 points reliably; excluding 2 takes several hundred to a
    few thousand pairs, depending on how many episodes flip either way).
    "Realistic level" is a judgement on the levels themselves, stated with the
    table (docs/design/faults.md).
    """
    if any(r.p_value < alpha and r.difference[0] < 0 for r in results):
        return "kept"
    worst = max(-r.difference[1] for r in results)
    if worst <= DECOY_MARGIN:
        return "decoy (a loss of 2 points excluded at every level)"
    confirmed = [r.label for r in results if decoy_level(r, alpha)]
    if confirmed:
        return f"decoy at {', '.join(confirmed)}; the other levels flat, not confirmed"
    if all(r.p_value >= alpha for r in results):
        return f"flat, not yet a decoy: a loss of up to {100 * worst:.1f} points is not excluded"
    return "unclear: a significant change only upward"


def point_role(
    result: PointResult, data: bool = False, inert: bool = False, alpha: float = 0.05
) -> str:
    """A point's role in the curves: "effect" (outcomes change beyond noise: fewer
    successes, or for a data fault either way), "decoy" (`decoy_level` on at
    least 1,000 pairs), "inert" (a type without a mechanism whose episodes all
    kept their outcome), or "small_cause" (none of these)."""
    if result.p_value < alpha and (data or result.difference[0] < 0):
        return "effect"
    if inert and result.broken == 0 and result.rescued == 0:
        return "inert"
    if decoy_level(result, alpha) and result.n >= DECOY_PAIRS:
        return "decoy"
    return "small_cause"


DECOY_PAIRS = 1000


def type_verdict(roles: Sequence[str]) -> str:
    """The type's verdict from its points' roles: "kept" (some point is an effect),
    "decoy" (a confirmed decoy level, no effect), "inert" (no mechanism, every
    point unchanged), "dropped" (none of these)."""
    if "effect" in roles:
        return "kept"
    if "decoy" in roles:
        return "decoy"
    if roles and all(r == "inert" for r in roles):
        return "inert"
    return "dropped"


def healthy_summary(run: RunOutcomes, suite: str = SUITE) -> dict[str, Any]:
    healthy = by_index(run, healthy_block(suite))
    n = len(healthy)
    s = sum(o.true_success for o in healthy.values())
    return {
        "run_id": run.run_id,
        "chain_ok": run.chain_ok,
        "n": n,
        "successes": s,
        "interval": wilson(s, n),
        "libero_successes": sum(o.libero_success for o in healthy.values()),
    }


def markdown_table(kind: str, results: Sequence[PointResult], healthy: dict[str, Any]) -> str:
    """The table of one type for docs/design/faults.md."""

    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    rate = healthy["successes"] / max(1, healthy["n"])
    lines = [
        f"| {kind} | Episodes | Success | 95% interval | Healthy twins | LIBERO check | "
        "Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        f"| healthy | {healthy['n']} | {healthy['successes']} ({pct(rate)}%) | "
        f"{pct(healthy['interval'][0])} to {pct(healthy['interval'][1])} | | "
        f"{healthy['libero_successes']} | | | | | `{healthy['run_id']}` |",
    ]
    for r in results:
        d, lo, hi = r.difference
        lines.append(
            f"| {r.label} | {r.n} | {r.successes} ({pct(r.rate)}%) | "
            f"{pct(r.interval[0])} to {pct(r.interval[1])} | {r.healthy_successes} | "
            f"{r.libero_successes} | {r.logged_successes} | "
            f"{r.broken} / {r.rescued} | {r.p_value:.2g} | "
            f"{100 * d:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}) | `{r.run_id}` |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- command line


# The calibration as run for docs/design/faults.md: (run name, kind or None for the
# healthy run, episodes per point, point indices or None for all), in the order they
# run: the yardsticks first, then the physical and site faults, then the rest. The
# candidate decoys get 300 pairs, enough to exclude a loss of two points when few
# episodes flip; the healthy run has 300, of which the first 100 pair with the others.
CALIBRATION_PLAN: tuple[tuple[str, str | None, int, tuple[int, ...] | None], ...] = (
    ("healthy", None, 300, None),
    ("camera_pose_offset", "camera_pose_offset", 100, None),
    ("action_latency", "action_latency", 100, None),
    ("gripper_wear", "gripper_wear", 100, None),
    ("object_batch_variant", "object_batch_variant", 100, None),
    ("furniture_shift", "furniture_shift", 100, None),
    ("illumination_change", "illumination_change", 300, None),
    ("backdrop_change-walls", "backdrop_change", 300, (0, 1)),
    ("backdrop_change-table", "backdrop_change", 100, (2,)),
    ("control_mode_change", "control_mode_change", 100, None),
    ("policy_knob", "policy_knob", 100, None),
    ("camera_degradation", "camera_degradation", 100, None),
    ("success_detector_error", "success_detector_error", 100, None),
    ("software_release", "software_release", 100, None),
)
CALIBRATION_ROOT = Path.home() / ".cache" / "robots-probes" / "faults"


def run_plan(names: Sequence[str], root: Path = CALIBRATION_ROOT, workers: int = 10) -> int:
    """Run the calibration plan (or the named runs of it) through the engine, one after another.

    Heavy: loads X-VLA. Stops after the healthy run if it succeeds in fewer than
    85% of its episodes or its replay sample broke.
    """
    import os

    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    miopen = _engine("fleetlab.engine.miopen")
    fleet = _engine("fleetlab.engine.fleet")
    miopen.activate()
    software = _engine("fleetlab.engine.software").current(miopen.require(4, "fp16_res32"))
    policy = _nominal_policy("fp16_res32")
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    for name, kind, count, points in CALIBRATION_PLAN:
        if names and name not in names:
            continue
        jobs = plan_jobs(kind, count, policy, software, 4, points)
        manifest = root / "manifests" / f"{name}.jsonl"
        manifest.write_text("".join(job.model_dump_json() + "\n" for job in jobs))
        config = fleet.RunConfig(
            out=str(root / "runs" / name),
            workers=workers,
            batch_size=4,
            precision="fp16_res32",
            episodes_per_robot_day=BLOCK_WIDTH,  # the engine asks for it since stage 2's runs
        )
        result = fleet.run(config, jobs)
        extra = result.record.extra
        line = {
            "run": name,
            "jobs": len(jobs),
            "chain_ok": result.record.chain_ok,
            "successes": extra.get("successes"),
            "throughput": extra.get("throughput"),
        }
        print(json.dumps(line), flush=True)
        with (root / "progress.jsonl").open("a") as log:
            log.write(json.dumps(line) + "\n")
        if kind is None:
            rate = (extra.get("successes") or 0) / len(jobs)
            if rate < 0.85 or result.record.chain_ok is False:
                print(f"the healthy run looks wrong ({rate:.0%}); stopping")
                return 1
    return 0


def _engine(module: str) -> Any:
    """A module of the rollout engine, which lives on its own branch until it is merged."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise SystemExit(f"the rollout engine is not in this checkout ({error})") from None


def _engine_software(batch_size: int, precision: str, miopen: str) -> SoftwareVersion:
    """The software version the engine records, read the way its own manifests read it."""
    db = _engine("fleetlab.engine.miopen").require(batch_size, precision, miopen)
    current: SoftwareVersion = _engine("fleetlab.engine.software").current(db)
    return current


def _nominal_policy(precision: str) -> PolicySpec:
    spec: PolicySpec = _engine("fleetlab.policies.xvla").nominal_spec(precision)
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.dose_response")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write a calibration manifest")
    plan.add_argument("kind", help="a calibrated fault kind, or 'healthy'")
    plan.add_argument("out", type=Path)
    plan.add_argument("--episodes", type=int, default=100, help="per point (multiple of 10)")
    plan.add_argument("--points", type=str, default=None, help="comma-separated point indices")
    plan.add_argument("--batch-size", type=int, default=4)
    plan.add_argument("--precision", default="fp16_res32")
    plan.add_argument("--miopen", default="pinned")
    run_cmd = sub.add_parser("run", help="run the calibration plan through the engine (heavy)")
    run_cmd.add_argument("names", nargs="*", help="run names of CALIBRATION_PLAN; all by default")
    run_cmd.add_argument(
        "--workers", type=int, default=10, help="fewer when the engine finds too little memory"
    )
    ana = sub.add_parser("analyse", help="print the tables of finished runs")
    ana.add_argument("--healthy", type=Path, required=True, action="append")
    ana.add_argument("runs", nargs="+", help="KIND=RUN_DIR, a kind may come more than once")
    ana.add_argument("--json", type=Path, default=None, help="also write the numbers here")
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_plan(args.names, workers=args.workers)

    if args.command == "plan":
        policy = _nominal_policy(args.precision)
        software = _engine_software(args.batch_size, args.precision, args.miopen)
        kind = None if args.kind == "healthy" else args.kind
        points = [int(p) for p in args.points.split(",")] if args.points else None
        jobs = plan_jobs(kind, args.episodes, policy, software, args.batch_size, points)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(job.model_dump_json() + "\n" for job in jobs))
        print(f"{len(jobs)} jobs -> {args.out}")
        return 0

    healthy_run = merge_runs([read_run(path) for path in args.healthy])
    healthy = healthy_summary(healthy_run)
    dump: dict[str, Any] = {"healthy": healthy, "types": {}}
    by_kind: dict[str, list[RunOutcomes]] = {}
    for item in args.runs:
        kind, _, path = item.partition("=")
        by_kind.setdefault(kind, []).append(read_run(Path(path)))
    for kind, runs in by_kind.items():
        results = analyse(kind, healthy_run, merge_runs(runs))
        print(f"\n### {kind}: {verdict(results)}\n")
        print(markdown_table(kind, results, healthy))
        dump["types"][kind] = {
            "verdict": verdict(results),
            "points": [
                r.__dict__ | {"rate": r.rate, "decoy_level": decoy_level(r)} for r in results
            ],
        }
    if args.json:
        args.json.write_text(json.dumps(dump, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
