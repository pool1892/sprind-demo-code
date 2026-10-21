"""The stage-3 calibration of the fault library (docs/design/faults.md, section 8).

One resumable heavy job runs `PLAN` through the rollout engine, run after run,
into `~/.cache/robots-probes/faults/runs/s3-*`, beside the stage-2 runs; the
analysis reads every run there and writes the curves into
`fleetlab/faults/curves.json`.

    tools/heavy-run uv run --locked python -m fleetlab.dose_response_stage3 run [NAME ...]
    uv run --locked python -m fleetlab.dose_response_stage3 status
    uv run --locked python -m fleetlab.dose_response_stage3 analyse [--write] [--out FILE]

What it measures, in the order it runs (the worlds' needs first):

1. a healthy run of 300 `libero_spatial` episodes, the twins of everything
   else, and whether its chains equal the stage-2 healthy run's bit for bit
   (another lock file since then); only if they do not, 700 more healthy
   episodes for the 1,000-pair decoy (otherwise stage 2's are the twins);
2. every new kind at every calibration point on 100 episodes, and the points
   appended to stage-2 kinds (the scene camera tilting down, a release
   carrying two faults);
3. the interaction pair of the worlds (`dose_response.JOINTS`): both singles
   and the joint on the same 300 episodes, and an exploratory pair on 100;
4. the decoy candidate (`light_color_shift`) at 300 pairs per level, then
   1,000 at the level a world would use, unless the 300 already show a loss;
5. two filmed episodes of every new kind at its strongest point, with their
   healthy twins, for the gate's videos.

`libero_object` (a healthy run, `gripper_wear` at its four levels,
`object_batch_variant` at its mass levels, the interaction pair) is in the
plan as optional runs, made only when named: the world generator's
supplement (`fleetlab.worlds.supplement`, world `w91`) measures
the same points on that suite and was queued first, so running them here by
default would spend the queue twice.

The episodes of a point are its indices `first` to `count - 1`
(`dose_response.episodes`), so an extension never repeats an episode and
every faulted episode has a healthy twin with the same task, start state and
seeds. A run directory that holds all its jobs is skipped; one that holds
some is resumed by the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab import dose_response as dr

ROOT = dr.CALIBRATION_ROOT
OBJECT = "libero_object"
WORKERS = 8  # ten did not fit the engine's memory floor beside other jobs (stage 2)
STAGE2_HEALTHY = ("healthy", "healthy-1000")
# The decoy candidate's level a world would use (lamps of about 3000 K, the full way).
DECOY_LABEL = ("light_color_shift", "1 warmer")
# The new kinds' points filmed for the gate: (kind, label) of the strongest realistic level.
FILMED: tuple[tuple[str, str], ...] = (
    ("lens_contamination", "1 scene film"),
    ("camera_latency", "12 steps scene lag"),
    ("gripper_force_loss", "0.5 force lost"),
    ("gripper_sticky_release", "20 steps late open"),
    ("arm_torque_loss", "0.2 torque lost"),
    ("controller_overload", "0.1 cycles missed"),
    ("calibration_update", "scene calibration rewritten"),
    ("image_pipeline_release", "10% cropped"),
    ("table_height_change", "2 cm lower"),
    ("light_color_shift", "1 warmer"),
    ("success_false_positive", "1 on a 6 deg camera"),
    ("camera_pose_offset", "4 deg scene tilt down"),
    ("software_release", "release with a 14-step delay and 10 actions per chunk"),
)


# Effects found at p 0.02 to 0.05 on 100 pairs (first pass), extended to 300.
EDGE_EFFECTS: tuple[tuple[str, str], ...] = (
    ("lens_contamination", "1 scene film"),
    ("camera_pose_offset", "4 deg scene tilt down"),
    ("image_pipeline_release", "10% cropped"),
    ("table_height_change", "2 cm higher"),
    ("controller_overload", "0.1 cycles missed"),
)


@dataclass(frozen=True)
class PlanRun:
    """One run of the plan: the healthy episodes (`kind` None), a type's points, a joint
    (`joint_name`), or the films (`films`); episode indices `first` to `count - 1`.

    `when` makes it conditional: "healthy_differs" (the stage-3 healthy run is
    not the stage-2 one bit for bit) or "decoy_holds" (the 300 pairs of the
    decoy candidate's level show no loss beyond noise). An `optional` run is
    made only when it is named.
    """

    name: str
    kind: str | None = None
    count: int = 100
    points: tuple[int, ...] | None = None
    first: int = 0
    suite: str = dr.SUITE
    joint_name: str | None = None
    films: bool = False
    when: str = ""
    optional: bool = False


def labels(kind: str, *names: str) -> tuple[int, ...]:
    return tuple(dr.point_index(kind, name) for name in names)


def plan() -> tuple[PlanRun, ...]:
    """The stage-3 calibration, in the order it runs."""
    new = dr.CALIBRATED[dr.CALIBRATED.index("lens_contamination") :]
    single = [k for k in new if k != "light_color_shift"]
    tilt = labels(
        "camera_pose_offset",
        "2 deg scene tilt down",
        "4 deg scene tilt down",
        "6 deg scene tilt down",
    )
    two = labels("software_release", "release with a 14-step delay and 10 actions per chunk")
    mass = labels(
        "object_batch_variant", "2 mass doublings", "4 mass doublings", "6 mass doublings"
    )
    return (
        PlanRun("s3-healthy", None, 300),
        PlanRun("s3-healthy-1000", None, 1000, first=300, when="healthy_differs"),
        *(PlanRun(f"s3-{kind}", kind) for kind in single),
        PlanRun("s3-camera_pose_offset-tilt", "camera_pose_offset", points=tilt),
        PlanRun("s3-software_release-two", "software_release", points=two),
        PlanRun("s3-gripper_wear-300", "gripper_wear", 300, labels("gripper_wear", "0.5 wear")),
        PlanRun(
            "s3-object_batch_variant-300",
            "object_batch_variant",
            300,
            labels("object_batch_variant", "4 mass doublings"),
        ),
        PlanRun("s3-joint-wear-heavier", count=300, joint_name="wear+heavier"),
        PlanRun("s3-joint-weaker-heavier", count=100, joint_name="weaker+heavier"),
        # Added after the first pass: the pair the curves suggested (a release carrying a
        # 14-step delay and 10 actions per chunk lost 90 points). Its singles' episodes 0
        # to 99 are stage 2's runs, whose healthy twins are stage 3's bit for bit.
        PlanRun(
            "s3-action_latency-300",
            "action_latency",
            300,
            labels("action_latency", "14 steps"),
            first=100,
        ),
        PlanRun(
            "s3-policy_knob-300",
            "policy_knob",
            300,
            labels("policy_knob", "10 actions per chunk"),
            first=100,
        ),
        PlanRun("s3-joint-delay-chunks", count=300, joint_name="delay+short-chunks"),
        # Added after the review of the results: the five effects at p 0.02 to 0.05 on 100
        # pairs, extended to 300 so that a world does not rest a detectable cause on one.
        *(
            PlanRun(f"s3-{kind}-300", kind, 300, labels(kind, label), first=100)
            for kind, label in EDGE_EFFECTS
        ),
        PlanRun("s3-light_color_shift", "light_color_shift", 300),
        PlanRun(
            "s3-light_color_shift-1000",
            "light_color_shift",
            1000,
            labels(*DECOY_LABEL),
            first=300,
            when="decoy_holds",
        ),
        PlanRun("s3-films", count=2, films=True),
        PlanRun("s3-object-healthy", None, 300, suite=OBJECT, optional=True),
        PlanRun("s3-object-gripper_wear", "gripper_wear", suite=OBJECT, optional=True),
        PlanRun(
            "s3-object-object_batch_variant",
            "object_batch_variant",
            points=mass,
            suite=OBJECT,
            optional=True,
        ),
        PlanRun(
            "s3-object-joint-wear-heavier", joint_name="wear+heavier", suite=OBJECT, optional=True
        ),
    )


def jobs_for(run: PlanRun, policy: Any, software: Any) -> list[Any]:
    """The jobs of one run of the plan."""
    if run.films:  # the first `count` episodes of each filmed point and of the healthy run
        jobs = dr.plan_jobs(None, 10, policy, software)
        for kind, label in FILMED:
            index = (dr.point_index(kind, label),)
            jobs += dr.plan_jobs(kind, 10, policy, software, 4, index)
        return [
            job.model_copy(update={"record_video": True})
            for job in jobs
            if job.identity.episode_index < run.count
        ]
    return dr.plan_jobs(
        run.kind,
        run.count,
        policy,
        software,
        4,
        run.points,
        first=run.first,
        suite=run.suite,
        joint_name=run.joint_name,
    )


# ---------------------------------------------------------------------------- running


def run_dir(name: str, root: Path = ROOT) -> Path:
    return root / "runs" / name


def complete(path: Path, jobs: int) -> bool:
    """Whether a run directory holds all its jobs, each with an outcome, and a run record."""
    from fleetlab.contracts.rollout import RunLayout

    if not RunLayout(path).run_record.is_file():
        return False
    try:
        return len(dr.read_run(path).outcomes) >= jobs
    except Exception:
        return False


def validate_run(
    path: Path, item: PlanRun, jobs: Sequence[Any], min_healthy: float
) -> dr.RunOutcomes:
    """Validate fresh and resumed runs identically; retain a rejection across restarts."""
    receipt = path / "calibration-validation.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    if previous.get("accepted") is False:
        raise ValueError(f"{path.name}: previously rejected calibration: {previous['reason']}")
    held = dr.read_run(path)
    expected = {job.identity.episode_id for job in jobs}
    reason = ""
    if set(held.outcomes) != expected:
        reason = "outcomes do not match the planned episodes"
    elif held.chain_ok is not True:
        reason = f"replay validation did not pass (chain_ok={held.chain_ok})"
    elif item.kind is None and item.joint_name is None and not item.films:
        rate = sum(o.true_success for o in held.outcomes.values()) / max(1, len(expected))
        if rate < min_healthy:
            reason = f"healthy success {rate:.4f} below floor {min_healthy:.4f}"
    receipt.write_text(
        json.dumps(
            {
                "accepted": not reason,
                "reason": reason,
                "min_healthy": min_healthy,
                "chain_ok": held.chain_ok,
            },
            indent=1,
        )
        + "\n"
    )
    if reason:
        raise ValueError(f"{path.name}: {reason}")
    return held


def same_chains(
    a: Path, b: Path, episode_ids: Sequence[str], other_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Compare chains in two runs (state, images, executed actions, fixture hash): the
    same episode ids in both, or `episode_ids` in `a` against `other_ids` in `b`."""
    from fleetlab.contracts.rollout import RunLayout
    from fleetlab.engine.chain import Chain, compare

    identical, differ, missing = 0, [], []
    for eid, oid in zip(episode_ids, other_ids or episode_ids, strict=True):
        pa, pb = RunLayout(a).table("chain", eid), RunLayout(b).table("chain", oid)
        if not (pa.is_file() and pb.is_file()):
            missing.append(eid)
            continue
        if compare(Chain.load(pa), Chain.load(pb)).identical:
            identical += 1
        else:
            differ.append(eid)
    return {
        "runs": [a.name, b.name],
        "compared": identical + len(differ),
        "identical": identical,
        "differ": differ,
        "missing": missing,
    }


def healthy_identity(root: Path = ROOT) -> dict[str, Any]:
    """Whether the stage-3 healthy run is stage 2's, bit for bit, on its 300 episodes."""
    ids = [dr.episode_id(0, i) for i in range(300)]
    return same_chains(run_dir("s3-healthy", root), run_dir(STAGE2_HEALTHY[0], root), ids)


def identity_held(result: dict[str, Any]) -> bool:
    return bool(result["compared"] == 300 and not result["differ"] and not result["missing"])


def decoy_holds(root: Path = ROOT) -> bool:
    """Whether the decoy candidate is still one after 300 pairs: no level of the type loses
    successes beyond noise (a type with an effect is kept, and its flat levels are small
    causes, so 1,000 pairs at one of them would confirm nothing a world uses)."""
    kind, _ = DECOY_LABEL
    healthy = dr.read_run(run_dir("s3-healthy", root))
    results = dr.analyse(kind, healthy, dr.read_run(run_dir("s3-light_color_shift", root)))
    return not any(r.p_value < 0.05 and r.difference[0] < 0 for r in results)


@dataclass
class EngineSetup:
    """What the runs are made with: the policy's spec, the software, the engine's settings
    beyond the run directory, the workers and the day length, and the healthy run's floor."""

    policy: Any
    software: Any
    config: dict[str, Any]
    min_healthy: float = 0.85


def xvla_setup() -> EngineSetup:
    """X-VLA in fp16_res32, four batch rows, the pinned MIOpen database (heavy)."""
    import os

    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    miopen = dr._engine("fleetlab.engine.miopen")
    miopen.activate()
    software = dr._engine("fleetlab.engine.software").current(miopen.require(4, "fp16_res32"))
    return EngineSetup(
        policy=dr._nominal_policy("fp16_res32"),
        software=software,
        config={"batch_size": 4, "precision": "fp16_res32"},
    )


def run(
    names: Sequence[str] = (),
    root: Path = ROOT,
    workers: int = WORKERS,
    *,
    setup: EngineSetup | None = None,
    runs: Sequence[PlanRun] | None = None,
) -> int:
    """Run the plan (or the named runs of it) through the engine; heavy with X-VLA.

    `setup` and `runs` replace X-VLA and the plan, for a dry run with the
    engine's scripted policy on the CPU.
    """
    fleet = dr._engine("fleetlab.engine.fleet")
    setup = setup or xvla_setup()
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    for item in runs if runs is not None else plan():
        if (names or item.optional) and item.name not in names:
            continue
        jobs = jobs_for(item, setup.policy, setup.software)
        out = run_dir(item.name, root)
        if complete(out, len(jobs)):
            try:
                validate_run(out, item, jobs, setup.min_healthy)
            except ValueError as error:
                _progress(root, {"run": item.name, "state": f"rejected: {error}"})
                return 1
            _progress(root, {"run": item.name, "state": "complete before"})
            continue
        if item.when == "healthy_differs":
            identity = healthy_identity(root)
            (root / "s3-healthy-identity.json").write_text(json.dumps(identity, indent=1))
            if identity_held(identity):
                _progress(root, {"run": item.name, "state": "skipped: stage 2's healthy holds"})
                continue
        if item.when == "decoy_holds" and not decoy_holds(root):
            _progress(root, {"run": item.name, "state": "skipped: a loss at 300 pairs"})
            continue
        manifest = root / "manifests" / f"{item.name}.jsonl"
        manifest.write_text("".join(job.model_dump_json() + "\n" for job in jobs))
        result = _run_with_memory(fleet, root, item.name, out, jobs, workers, setup)
        extra = result.record.extra
        # The run directory's own count: after a resume, the engine's counts cover only
        # the jobs of the last invocation (the first start of stage 3 stopped on that).
        try:
            held = validate_run(out, item, jobs, setup.min_healthy).outcomes
        except ValueError as error:
            _progress(root, {"run": item.name, "state": f"rejected: {error}"})
            return 1
        successes = sum(o.true_success for o in held.values())
        line = {
            "run": item.name,
            "state": "ran",
            "jobs": len(jobs),
            "episodes": len(held),
            "chain_ok": result.record.chain_ok,
            "successes": successes,
            "throughput": extra.get("throughput"),
        }
        _progress(root, line)
    return 0


# The machine is shared: another project's job can take 20 GiB for a few minutes. A run
# refused or stopped for memory waits and resumes (the engine resumes a run directory),
# with as many workers as fit; the worker count does not change an episode (engine README,
# exactness: 200 of 200 identical with 10 and 6 workers).
MIN_WORKERS = 4
MEMORY_WAIT_S = 120.0
MEMORY_RETRIES = 15


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def fit_workers(fleet: Any, wanted: int, config: dict[str, Any]) -> int:
    """The most workers, up to `wanted`, that the engine's memory guard lets start now."""
    defaults = fleet.RunConfig.__dataclass_fields__
    floor = config.get("mem_floor_gib", getattr(defaults.get("mem_floor_gib"), "default", 8.0))
    server = config.get("server_gib", getattr(defaults.get("server_gib"), "default", 4.0))
    worker = config.get("worker_gib", getattr(defaults.get("worker_gib"), "default", 2.1))
    room = fleet.mem_available_gib() - floor - server
    return max(0, min(wanted, int(room // worker)))


def _memory_trouble(error: BaseException) -> bool:
    text = str(error)
    return "GiB available" in text or "memory available" in text or "GPU memory" in text


def _run_with_memory(
    fleet: Any,
    root: Path,
    name: str,
    out: Path,
    jobs: list[Any],
    workers: int,
    setup: EngineSetup,
) -> Any:
    """Run one run of the plan, waiting out a shortage of memory and resuming after one."""
    for attempt in range(MEMORY_RETRIES + 1):
        cpu = setup.config.get("device") == "cpu"
        n = workers if cpu else fit_workers(fleet, workers, setup.config)
        if n >= min(MIN_WORKERS, workers):
            config = fleet.RunConfig(
                out=str(out),
                workers=n,
                episodes_per_robot_day=dr.BLOCK_WIDTH,
                **setup.config,
            )
            try:
                return fleet.run(config, jobs)
            except Exception as error:
                if not _memory_trouble(error) or attempt == MEMORY_RETRIES:
                    raise
                why = str(error)
        else:
            why = f"only {n} workers fit"
        if attempt == MEMORY_RETRIES:
            break
        _progress(root, {"run": name, "state": f"waiting for memory: {why}", "workers": n})
        _sleep(MEMORY_WAIT_S)
    raise RuntimeError(f"{name}: not enough memory after {MEMORY_RETRIES} waits")


def _progress(root: Path, line: dict[str, Any]) -> None:
    import time

    line = {**line, "stage": 3, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(line), flush=True)
    with (root / "progress.jsonl").open("a") as log:
        log.write(json.dumps(line) + "\n")


def status(root: Path = ROOT) -> list[tuple[str, int, int]]:
    """(run, jobs planned, episodes with an outcome) for every run of the plan."""
    from fleetlab.contracts.identity import PolicySpec, SoftwareVersion
    from fleetlab.contracts.policy import ControlMode

    policy = PolicySpec(
        name="xvla-libero",
        revision="0",
        precision="fp16_res32",
        control_mode=ControlMode.absolute,
        actions_executed_per_chunk=30,
        sampling_steps=10,
        image_views=4,
    )
    software = SoftwareVersion(
        lock_hash="0" * 16, libero_patch_set="0", mujoco_version="0", miopen_db_hash="0"
    )
    out = []
    for item in plan():
        planned = len(jobs_for(item, policy, software))
        path = run_dir(item.name, root)
        try:
            done = len(dr.read_run(path).outcomes) if path.is_dir() else 0
        except Exception:
            done = 0
        out.append((item.name, planned, done))
    return out


# ---------------------------------------------------------------------------- analysis


def interaction(
    healthy: dict[int, dr.Outcome],
    a: dict[int, dr.Outcome],
    b: dict[int, dr.Outcome],
    joint: dict[int, dr.Outcome],
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """The effects of two faults alone and together on the same episodes, and their interaction.

    Per episode (the fleet's check, 1 or 0): the effect of A is A minus healthy,
    of B likewise, of the joint the joint minus healthy; the interaction is the
    joint effect minus the sum of the single effects, `AB - A - B + healthy`,
    averaged over the episodes all four runs hold. Negative: the two together
    cost more than their sum. The intervals are percentile bootstraps over
    episodes (`draws` resamples, fixed `seed`), paired as the episodes are.
    """
    idx = sorted(set(healthy) & set(a) & set(b) & set(joint))
    if not idx:
        raise ValueError("no episode is in all four runs")
    y = {
        name: np.array([float(run[i].true_success) for i in idx])
        for name, run in (("h", healthy), ("a", a), ("b", b), ("ab", joint))
    }
    per_episode = {
        "effect_a": y["a"] - y["h"],
        "effect_b": y["b"] - y["h"],
        "effect_joint": y["ab"] - y["h"],
        "interaction": y["ab"] - y["a"] - y["b"] + y["h"],
    }
    rng = np.random.default_rng(seed)
    resample = rng.integers(0, len(idx), size=(draws, len(idx)))
    out: dict[str, Any] = {
        "n": len(idx),
        "healthy": int(y["h"].sum()),
        "a": int(y["a"].sum()),
        "b": int(y["b"].sum()),
        "joint": int(y["ab"].sum()),
    }
    for name, values in per_episode.items():
        means = values[resample].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        out[name] = [round(float(values.mean()), 4), round(float(low), 4), round(float(high), 4)]
    worse = out["interaction"][2] < 0
    better = out["interaction"][1] > 0
    out["role"] = "interaction" if worse else ("sub-additive" if better else "additive")
    return out


def source_dirs(root: Path, run_id: str) -> list[str]:
    """Findable directories, including every part of a merged calibration run."""
    return [str(run_dir(name, root).expanduser().resolve()) for name in run_id.split(" + ")]


def point_dict(
    result: dr.PointResult,
    role: str,
    *,
    root: Path = ROOT,
    healthy: dr.RunOutcomes | None = None,
) -> dict[str, Any]:
    """A point as curves.json stores it (the stage-2 fields and rounding)."""
    if result.chain_ok is not True:
        raise ValueError(f"{result.run_id}: replay validation did not pass")
    return {
        "chain_ok": result.chain_ok,
        "label": result.label,
        "level": result.level,
        "n": result.n,
        "successes": result.successes,
        "libero_successes": result.libero_successes,
        "logged_successes": result.logged_successes,
        "healthy_successes": result.healthy_successes,
        "broken": result.broken,
        "rescued": result.rescued,
        "p_value": float(f"{result.p_value:.3g}"),
        "difference": [round(x, 4) for x in result.difference],
        "run_id": result.run_id,
        "run_dirs": source_dirs(root, result.run_id),
        "healthy_run_dirs": source_dirs(root, healthy.run_id) if healthy else [],
        "role": role,
    }


def _runs(root: Path, *names: str) -> dr.RunOutcomes | None:
    found = [dr.read_run(run_dir(n, root)) for n in names if _readable(run_dir(n, root))]
    for run in found:
        receipt = run_dir(run.run_id, root) / "calibration-validation.json"
        if run.chain_ok is not True or (
            receipt.is_file() and not json.loads(receipt.read_text())["accepted"]
        ):
            raise ValueError(f"{run.run_id}: rejected calibration run")
    return dr.merge_runs(found) if found else None


def _readable(path: Path) -> bool:
    try:
        dr.read_run(path)
    except Exception:
        return False
    return True


def _is_inert(kind: str) -> bool:
    import fleetlab.faults as faults
    from fleetlab.contracts.faults import FaultTypeBase

    hooks = ("cell_hook", "observation_hook", "action_hook", "policy_input_hook", "data_hook")
    cls = type(faults.get(kind))
    return all(getattr(cls, h) is getattr(FaultTypeBase, h) for h in hooks)


def _is_data(kind: str) -> bool:
    import fleetlab.faults as faults
    from fleetlab.contracts.vocabulary import FaultClass

    return faults.get(kind).fault_class is FaultClass.data


def roles_of(kind: str, results: Sequence[dr.PointResult]) -> list[str]:
    data, inert = _is_data(kind), _is_inert(kind)
    return [dr.point_role(r, data=data, inert=inert) for r in results]


def healthy_spatial(root: Path) -> tuple[dr.RunOutcomes | None, str]:
    """The stage-3 healthy twins: its own 300, and 300 to 999 from its own extension or,
    when the chains held, from stage 2's."""
    first = _runs(root, "s3-healthy")
    if first is None:
        return None, "no stage-3 healthy run yet"
    summary = dr.healthy_summary(first)
    if summary["successes"] / max(1, summary["n"]) < EngineSetup.min_healthy:
        raise ValueError("stage-3 healthy calibration is below its default floor")
    if _readable(run_dir("s3-healthy-1000", root)):
        rest = _runs(root, "s3-healthy-1000")
        note = "stage 3's own 1,000"
    else:
        identity = root / "s3-healthy-identity.json"
        held = identity.is_file() and identity_held(json.loads(identity.read_text()))
        rest = _runs(root, STAGE2_HEALTHY[1]) if held else None
        note = (
            "stage 3's first 300, stage 2's 300 to 999 (identical chains)"
            if held
            else "stage 3's own only"
        )
    return (dr.merge_runs([first, rest]) if rest else first), note


def analyse(root: Path = ROOT) -> dict[str, Any]:
    """Everything the stage-3 runs measured, as curves.json and the tables use it."""
    out: dict[str, Any] = {"types": {}, "appended": {}, "joints": [], "object": {}, "checks": {}}
    healthy, note = healthy_spatial(root)
    if healthy is None:
        out["note"] = note
        return out
    out["healthy"] = dr.healthy_summary(healthy) | {
        "note": note,
        "run_dirs": source_dirs(root, healthy.run_id),
    }
    from fleetlab.contracts.rollout import RunLayout

    record = RunLayout(run_dir("s3-healthy", root)).run_record
    if record.is_file():
        out["software"] = json.loads(record.read_text()).get("software")
    identity = root / "s3-healthy-identity.json"
    if identity.is_file():
        out["checks"]["healthy_identity"] = json.loads(identity.read_text())

    def curve(kind: str, names: Sequence[str]) -> list[dict[str, Any]]:
        runs = _runs(root, *names)
        if runs is None:
            return []
        results = dr.analyse(kind, healthy, runs)
        out = []
        for r, role in zip(results, roles_of(kind, results), strict=True):
            point = point_dict(r, role, root=root, healthy=healthy)
            companions = dr.calibration_points(kind)[dr.point_index(kind, r.label)].companions
            if companions:  # a data type on top of other faults: its twins are not healthy
                point["companions"] = [list(c) for c in companions]
                point["compared_with"] = "the same episodes' true outcome"
                point["healthy_run_dirs"] = point["run_dirs"]
            out.append(point)
        return out

    new = dr.CALIBRATED[dr.CALIBRATED.index("lens_contamination") :]
    for kind in new:
        names = [f"s3-{kind}", f"s3-{kind}-300", f"s3-{kind}-1000"]
        points = curve(kind, names)
        if points:
            out["types"][kind] = {
                "verdict": dr.type_verdict([p["role"] for p in points]),
                "points": points,
            }
    for kind, names in (
        ("camera_pose_offset", ["s3-camera_pose_offset-tilt", "s3-camera_pose_offset-300"]),
        ("software_release", ["s3-software_release-two"]),
    ):
        points = curve(kind, names)
        if points:
            out["appended"][kind] = points

    pairs: tuple[tuple[str, str, tuple[Sequence[str], Sequence[str]]], ...] = (
        (
            "wear+heavier",
            "s3-joint-wear-heavier",
            (["s3-gripper_wear-300"], ["s3-object_batch_variant-300"]),
        ),
        (
            "weaker+heavier",
            "s3-joint-weaker-heavier",
            (["s3-gripper_force_loss"], ["s3-object_batch_variant-300"]),
        ),
        (
            "delay+short-chunks",
            "s3-joint-delay-chunks",
            (["action_latency", "s3-action_latency-300"], ["policy_knob", "s3-policy_knob-300"]),
        ),
    )
    for name, run_name, singles in pairs:
        entry = _joint(root, name, run_name, singles, healthy, dr.SUITE)
        if entry:
            out["joints"].append(entry)

    obj_healthy = _runs(root, "s3-object-healthy")
    if obj_healthy is not None:
        out["object"]["healthy"] = dr.healthy_summary(obj_healthy, OBJECT) | {
            "run_dirs": source_dirs(root, obj_healthy.run_id)
        }
        for kind in ("gripper_wear", "object_batch_variant"):
            runs = _runs(root, f"s3-object-{kind}")
            if runs is None:
                continue
            results = dr.analyse(kind, obj_healthy, runs, suite=OBJECT)
            roles = roles_of(kind, results)
            out["object"][kind] = [
                point_dict(r, role, root=root, healthy=obj_healthy)
                for r, role in zip(results, roles, strict=True)
            ]
        entry = _joint(
            root,
            "wear+heavier",
            "s3-object-joint-wear-heavier",
            (["s3-object-gripper_wear"], ["s3-object-object_batch_variant"]),
            obj_healthy,
            OBJECT,
        )
        if entry:
            out["joints"].append(entry)

    # The release carrying two faults against the two faults as a joint, chain by chain.
    if run_dir("s3-joint-delay-chunks", root).is_dir():
        release = dr.block_of("software_release", 1)
        pair = dr.joint_block("delay+short-chunks")
        out["checks"]["release with two faults against the joint"] = same_chains(
            run_dir("s3-software_release-two", root),
            run_dir("s3-joint-delay-chunks", root),
            [dr.episode_id(release, i) for i in range(100)],
            [dr.episode_id(pair, i) for i in range(100)],
        )
    # The inert type's episodes against their healthy twins, chain by chain.
    inert_block = dr.block_of("calibration_update", 0)
    if run_dir("s3-calibration_update", root).is_dir():
        out["checks"]["calibration_update against healthy"] = same_chains(
            run_dir("s3-calibration_update", root),
            run_dir("s3-healthy", root),
            [dr.episode_id(inert_block, i) for i in range(100)],
            [dr.episode_id(0, i) for i in range(100)],
        )
    # Episodes of stage 2 run again in stage 3: bit for bit, or not.
    for kind, label, stage2, stage3 in (
        ("gripper_wear", "0.5 wear", "gripper_wear", "s3-gripper_wear-300"),
        (
            "object_batch_variant",
            "4 mass doublings",
            "object_batch_variant",
            "s3-object_batch_variant-300",
        ),
    ):
        block = dr.block_of(kind, dr.point_index(kind, label))
        if run_dir(stage2, root).is_dir() and run_dir(stage3, root).is_dir():
            ids = [dr.episode_id(block, i) for i in range(100)]
            out["checks"][f"{kind} {label}"] = same_chains(
                run_dir(stage3, root), run_dir(stage2, root), ids
            )
    return out


def analyse_object(root: Path, runs: Sequence[PlanRun]) -> dict[str, Any]:
    """Analyse explicitly selected object runs, including extensions and paired joints."""
    selected = [item for item in runs if item.suite == OBJECT and not item.films]
    healthy_names = [
        item.name for item in selected if item.kind is None and item.joint_name is None
    ]
    healthy = _runs(root, *healthy_names)
    out: dict[str, Any] = {"types": {}, "appended": {}, "joints": [], "object": {}}
    if healthy is None:
        return out
    summary = dr.healthy_summary(healthy, OBJECT)
    if summary["successes"] / max(1, summary["n"]) < EngineSetup.min_healthy:
        raise ValueError("object healthy calibration is below its default floor")
    out["object"]["healthy"] = summary | {"run_dirs": source_dirs(root, healthy.run_id)}
    kinds = dict.fromkeys(item.kind for item in selected if item.kind is not None)
    for kind in kinds:
        assert kind is not None
        found = _runs(root, *(item.name for item in selected if item.kind == kind))
        if found is None:
            continue
        results = dr.analyse(kind, healthy, found, suite=OBJECT)
        points = []
        for result, role in zip(results, roles_of(kind, results), strict=True):
            point = point_dict(result, role, root=root, healthy=healthy)
            companions = dr.calibration_points(kind)[dr.point_index(kind, result.label)].companions
            if companions:
                point["companions"] = [list(c) for c in companions]
                point["compared_with"] = "the same episodes' true outcome"
                point["healthy_run_dirs"] = point["run_dirs"]
            points.append(point)
        out["object"][kind] = points
    for item in selected:
        if item.joint_name is None:
            continue
        members = dr.joint(item.joint_name).members
        singles = tuple([r.name for r in selected if r.kind == kind] for kind, _ in members)
        entry = _joint(root, item.joint_name, item.name, (singles[0], singles[1]), healthy, OBJECT)
        if entry:
            out["joints"].append(entry)
    return out


def _joint(
    root: Path,
    name: str,
    run_name: str,
    singles: tuple[Sequence[str], Sequence[str]],
    healthy: dr.RunOutcomes,
    suite: str,
) -> dict[str, Any] | None:
    """A joint point against both singles (each merged from its runs) and the healthy twins."""
    joint_run = _runs(root, run_name)
    single_runs = [_runs(root, *names) for names in singles]
    if joint_run is None or any(r is None for r in single_runs):
        return None
    spec = dr.joint(name)
    members = []
    outcome_sets = []
    for (kind, label), runs in zip(spec.members, single_runs, strict=True):
        assert runs is not None
        block = dr.block_of(kind, dr.point_index(kind, label), suite)
        outcome_sets.append(dr.by_index(runs, block))
        members.append({"kind": kind, "label": label})
    h = dr.by_index(healthy, dr.healthy_block(suite))
    ab = dr.by_index(joint_run, dr.joint_block(name, suite))
    stats = interaction(h, outcome_sets[0], outcome_sets[1], ab)
    paired = dr.compare(h, ab, name, 1.0, joint_run)
    return {
        "name": name,
        "chain_ok": joint_run.chain_ok,
        "members": members,
        "suite": suite,
        "n": stats["n"],
        "successes": stats["joint"],
        "healthy_successes": stats["healthy"],
        "single_successes": [stats["a"], stats["b"]],
        "singles": [
            round(stats["a"] / stats["healthy"], 4) if stats["healthy"] else 1.0,
            round(stats["b"] / stats["healthy"], 4) if stats["healthy"] else 1.0,
        ],
        "difference": [round(x, 4) for x in paired.difference],
        "p_value": float(f"{paired.p_value:.3g}"),
        "effects": {k: stats[k] for k in ("effect_a", "effect_b", "effect_joint")},
        "interaction": stats["interaction"],
        # The joint point's role against healthy, as the world generator's supplement
        # writes it, and the interaction's own: "interaction", "additive", "sub-additive".
        "role": dr.point_role(paired),
        "interaction_role": stats["role"],
        "run_id": paired.run_id,
        "run_dirs": source_dirs(root, paired.run_id),
        "healthy_run_dirs": source_dirs(root, healthy.run_id),
        "single_run_dirs": [source_dirs(root, r.run_id) for r in single_runs if r is not None],
    }


def write_curves(result: dict[str, Any], path: Path | None = None) -> Path:
    """Merge the analysis into curves.json: new kinds, appended points, the supplement."""
    from importlib import resources

    target = path or Path(str(resources.files("fleetlab.faults").joinpath("curves.json")))
    entries = [p for entry in result["types"].values() for p in entry["points"]]
    entries += [p for points in result["appended"].values() for p in points]
    entries += list(result["joints"])
    entries += [
        p for kind, points in result.get("object", {}).items() if kind != "healthy" for p in points
    ]
    entries += [result[key] for key in ("healthy",) if key in result]
    if "healthy" in result.get("object", {}):
        entries.append(result["object"]["healthy"])
    if any(p.get("chain_ok") is not True for p in entries):
        raise ValueError("cannot publish a calibration without passing replay validation")
    if any(not p.get("run_dirs") for p in entries):
        raise ValueError("cannot publish a calibration without source run directories")
    curves = json.loads(target.read_text())
    for kind, entry in result["types"].items():
        before = curves["types"].get(kind, {})
        curves["types"][kind] = {
            **before,
            "verdicts": {**before.get("verdicts", {}), dr.SUITE: entry["verdict"]},
            "note": entry.get("note") or before.get("note", ""),  # a written note stays
            "points": entry["points"],
        }
    for kind, points in result["appended"].items():  # stage 3's points of stage-2 types
        stored = curves["types"][kind]["points"]
        fresh = {p["label"]: p for p in points}
        kept = [fresh.pop(p["label"], p) for p in stored]  # a re-measured point is replaced
        curves["types"][kind]["points"] = kept + list(fresh.values())
        curves["types"][kind]["verdicts"][dr.SUITE] = dr.type_verdict(
            [p["role"] for p in curves["types"][kind]["points"]]
        )
    supplement: dict[str, Any] = curves.setdefault(
        "supplement", {"healthy": {}, "points": [], "joints": []}
    )
    if "healthy" in result.get("object", {}):
        h = result["object"]["healthy"]
        supplement["healthy"][OBJECT] = {
            "run_id": h["run_id"],
            "n": h["n"],
            "successes": h["successes"],
            "libero_successes": h["libero_successes"],
            "run_dirs": h["run_dirs"],
        }
    for kind, points in result.get("object", {}).items():
        if kind == "healthy":
            continue
        for point in points:
            key = (kind, OBJECT, point["label"])
            supplement["points"] = [
                p for p in supplement["points"] if (p["kind"], p["suite"], p["label"]) != key
            ] + [{"kind": kind, "suite": OBJECT, **point}]
        roles = [
            p["role"] for p in supplement["points"] if p["kind"] == kind and p["suite"] == OBJECT
        ]
        curves["types"][kind].setdefault("verdicts", {})[OBJECT] = dr.type_verdict(roles)
    for joint_entry in result["joints"]:
        supplement["joints"] = [
            j
            for j in supplement["joints"]
            if (j["name"], j["suite"]) != (joint_entry["name"], joint_entry["suite"])
        ] + [joint_entry]
    if "healthy" in result:
        curves["stage3"] = {
            "software": result.get("software"),
            "healthy": result.get("healthy"),
            "checks": result.get("checks"),
        }
    target.write_text(json.dumps(curves, indent=1) + "\n")
    return target


def tables(result: dict[str, Any]) -> str:
    """The Markdown tables of the analysis, for docs/design/faults.md."""

    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    def row(p: dict[str, Any]) -> str:
        d, lo, hi = p["difference"]
        rate = p["successes"] / p["n"] if p["n"] else 0.0
        return (
            f"| {p['label']} | {p['n']} | {p['successes']} ({pct(rate)}%) | "
            f"{p['healthy_successes']} | {p['libero_successes']} | {p['logged_successes']} | "
            f"{p['broken']} / {p['rescued']} | {p['p_value']:.2g} | "
            f"{100 * d:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}) | {p['role']} | `{p['run_id']}` |"
        )

    head = (
        "| {name} | Episodes | {outcome} | {twins} | LIBERO's check | Logged | Broken / rescued | "
        "McNemar p | Difference, points (95% interval) | Role | Run |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    parts = []

    def heading(kind: str) -> str:
        return head.format(
            name=kind,
            outcome="Logged successes" if _is_data(kind) else "Placed",
            twins="Physically placed (same attempts)" if _is_data(kind) else "Twins",
        )

    for kind, entry in result.get("types", {}).items():
        parts.append(f"#### {kind}: {entry['verdict']}\n\n" + heading(kind))
        parts[-1] += "\n" + "\n".join(row(p) for p in entry["points"])
    for kind, points in result.get("appended", {}).items():
        parts.append(f"#### {kind}: points added in stage 3\n\n" + heading(kind))
        parts[-1] += "\n" + "\n".join(row(p) for p in points)
    for kind in ("gripper_wear", "object_batch_variant"):
        points = result.get("object", {}).get(kind)
        if points:
            parts.append(f"#### {kind} on libero_object\n\n" + heading(kind))
            parts[-1] += "\n" + "\n".join(row(p) for p in points)
    if result.get("joints"):
        lines = [
            "| Pair | Suite | Episodes | Healthy | A alone | B alone | Together | "
            "Effect A | Effect B | Joint effect | Interaction (95% interval) | Role |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for j in result["joints"]:

            def eff(v: list[float]) -> str:
                return f"{100 * v[0]:+.1f} ({100 * v[1]:+.1f} to {100 * v[2]:+.1f})"

            a, b = j["single_successes"]
            lines.append(
                f"| {j['name']} | {j['suite']} | {j['n']} | {j['healthy_successes']} | {a} | {b} | "
                f"{j['successes']} | {eff(j['effects']['effect_a'])} | "
                f"{eff(j['effects']['effect_b'])} | {eff(j['effects']['effect_joint'])} | "
                f"{eff(j['interaction'])} | {j['interaction_role']} |"
            )
        parts.append("#### Interactions\n\n" + "\n".join(lines))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------- films


def films(run: Path, out: Path, copy_videos: bool = False) -> Path:
    """A page of the filmed run: per filmed point, its episodes beside their healthy twins,
    as strips of five scene frames (JPEG) with the outcomes and where the videos are.

    Kept in the repository as docs/design/faults-films/ (faults.md, section 9): the
    strips are small; the videos (8 MB) stay in the run directory, named on the page,
    unless `copy_videos` puts them beside the page and links them.
    """
    import fleetlab.faults as faults

    (out / "media").mkdir(parents=True, exist_ok=True)
    outcomes = dr.read_run(run).outcomes
    video = run / "log" / "video"
    shown = str(video).replace(str(Path.home()), "~")
    where = (
        "the links are the scene and wrist videos"
        if copy_videos
        else f"the videos (scene and wrist, MP4) are in `{shown}/` under each episode's id"
    )
    lines = [
        "# The fault library on film (stage 3)",
        "",
        "Episodes of every new fault type at a strong level, each beside its healthy twin "
        "(same task, start state and seeds), X-VLA fp16_res32 on `libero_spatial`, from run "
        f"`{run.name}` of the stage-3 calibration. Each strip shows five "
        "frames of the scene camera from the first step to the last, upright, as the fleet "
        f"logs them (after the observation hooks); {where}. Made with `python -m "
        "fleetlab.dose_response_stage3 films`.",
        "",
    ]
    by_class: dict[str, list[tuple[str, str]]] = {}
    for kind, label in FILMED:
        by_class.setdefault(faults.get(kind).fault_class.value, []).append((kind, label))
    count = 1 + max(dr.split_episode_id(e)[1] for e in outcomes)
    for cls in ("sensor", "physical", "configuration", "software", "site", "data"):
        if cls not in by_class:
            continue
        lines += [f"## {cls}", ""]
        for kind, label in by_class[cls]:
            block = dr.block_of(kind, dr.point_index(kind, label))
            lines += [f"### `{kind}`, {label}", ""]
            for i in range(count):
                for name, b in (("healthy", 0), ("faulted", block)):
                    eid = dr.episode_id(b, i)
                    lines += _film(outcomes, video, out, name, eid, copy_videos)
            lines.append("")
    (out / "index.md").write_text("\n".join(lines) + "\n")
    return out / "index.md"


def _film(
    outcomes: dict[str, dr.Outcome],
    video: Path,
    out: Path,
    name: str,
    eid: str,
    copy_videos: bool,
) -> list[str]:
    import shutil

    from PIL import Image

    scene, wrist = video / f"{eid}.scene.mp4", video / f"{eid}.wrist.mp4"
    o = outcomes.get(eid)
    if o is None or not scene.is_file():
        return [f"- {name} `{eid}`: no video"]
    text = (
        f"{'placed' if o.true_success else 'failed'}, logged "
        f"{'success' if o.logged_success else 'failure'}, {o.steps} steps"
    )
    still = out / "media" / f"{eid}.jpg"
    if not still.is_file():
        Image.fromarray(_strip(scene)).save(still, quality=80)
    head = f"- {name} `{eid}` ({text})"
    if copy_videos:
        for src in (scene, wrist):
            if src.is_file() and not (out / "media" / src.name).is_file():
                shutil.copy(src, out / "media" / src.name)
        head += f": [scene](media/{scene.name}), [wrist](media/{wrist.name})"
    return [head, f"\n  ![{name} {eid}](media/{still.name})\n"]


def _strip(path: Path, count: int = 5) -> Any:
    import imageio.v2 as imageio

    reader: Any = imageio.get_reader(path)
    frames = [np.asarray(f) for f in reader]
    reader.close()
    picks = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return np.concatenate([frames[i] for i in picks], axis=1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.dose_response_stage3")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="run the stage-3 plan through the engine (heavy)")
    run_cmd.add_argument("names", nargs="*", help="run names of the plan; all by default")
    run_cmd.add_argument("--workers", type=int, default=WORKERS)
    sub.add_parser("status", help="how far the plan's runs are")
    ana = sub.add_parser("analyse", help="the tables, and the curves into curves.json")
    ana.add_argument("--write", action="store_true", help="merge the curves into curves.json")
    ana.add_argument("--out", type=Path, default=None, help="also write the analysis as JSON")
    film = sub.add_parser("films", help="a page of the filmed run (frame strips)")
    film.add_argument("out", type=Path)
    film.add_argument("--run", type=Path, default=run_dir("s3-films"))
    film.add_argument("--copy-videos", action="store_true", help="put the videos beside it")
    args = parser.parse_args(argv)
    if args.command == "films":
        print(films(args.run, args.out, args.copy_videos))
        return 0
    if args.command == "run":
        return run(args.names, workers=args.workers)
    if args.command == "status":
        for name, planned, done in status():
            optional = " (optional)" if name.startswith("s3-object-") else ""
            print(f"{name:40s} {done:5d} / {planned}{optional}")
        return 0
    result = analyse()
    print(tables(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, default=list))
    if args.write:
        print(f"curves -> {write_curves(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
