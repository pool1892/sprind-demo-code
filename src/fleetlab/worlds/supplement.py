"""The curves the worlds need beyond the faults' calibration: `libero_object`
and the interaction pairs.

The faults' dose-response (docs/design/faults.md section 6) was measured on
`libero_spatial` only, and section 7 there says a world with `libero_object` tasks
needs those curves before its levels are set from them. The worlds run both suites,
and their interaction (worn gripper pads with a heavier
object batch) needs a joint point that no single curve gives. This module measures
both the way `fleetlab.dose_response` measures a curve: 100 episodes per point (ten
tasks by start states 0 to 9), each paired with a healthy twin of the same suite (same
task, start state, scene seed and noise seed), X-VLA in fp16_res32, four batch rows,
the fleet's check. Identities: world `w91`, robot `r01` at site `s1`, day 1, the
episode index the episode's number within its point; episode ids `e` + block (three
digits) + index (three digits), block 0 healthy `libero_object`, block 1 healthy
`libero_spatial`, the points after them in `POINTS` order.

    tools/heavy-run uv run --locked python -m fleetlab.worlds supplement run [--root DIR]
    uv run --locked python -m fleetlab.worlds supplement analyse [--root DIR]

`analyse` writes `curves_supplement.json` beside this module, which
`fleetlab.worlds.curves` reads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from fleetlab.contracts.faults import FaultSpec, Schedule
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.ids import IdKind, ScopeKind, neutral_id
from fleetlab.contracts.rollout import Job
from fleetlab.worlds.curves import OBJECT, SPATIAL, SUPPLEMENT_FILE, calibrated_label

WORLD_ID = "w91"
WORLD_SEED = 91
ROBOT_ID, SITE_ID, DAY = "r01", "s1", 1
BLOCK_WIDTH = 1000
EPISODES = 100
DEFAULT_ROOT = Path.home() / ".cache" / "robots-probes" / "worlds" / "supplement"
RUN_NAME = "w91-supplement"

HEALTHY_BLOCKS = {OBJECT: 0, SPATIAL: 1}


@dataclass(frozen=True)
class Point:
    """One supplement point: a suite and one fault, or two for an interaction pair."""

    suite: str
    members: tuple[tuple[str, str], ...]  # (kind, calibration label)

    @property
    def name(self) -> str:
        return " + ".join(label for _, label in self.members) + f" on {self.suite}"


_PAIRS = (
    (("gripper_wear", "0.5 wear"), ("object_batch_variant", "2 mass doublings")),
    (("gripper_wear", "0.5 wear"), ("object_batch_variant", "4 mass doublings")),
    (("gripper_wear", "0.75 wear"), ("object_batch_variant", "2 mass doublings")),
)
_OBJECT_SINGLES = (
    ("camera_pose_offset", "3 deg scene pan"),
    ("camera_pose_offset", "6 deg scene pan"),
    ("camera_pose_offset", "9 deg scene pan"),
    ("camera_degradation", "4 px blur"),
    ("action_latency", "17 steps"),
    ("action_latency", "20 steps"),
    ("policy_knob", "5 sampling steps"),
    ("gripper_wear", "0.5 wear"),
    ("gripper_wear", "0.75 wear"),
    ("gripper_wear", "1 wear"),
    ("object_batch_variant", "2 mass doublings"),
    ("object_batch_variant", "4 mass doublings"),
    ("object_batch_variant", "6 mass doublings"),
    ("furniture_shift", "4 cm sideways"),
    ("control_mode_change", "relative"),
    ("success_detector_error", "0.2 missed"),
    ("success_detector_error", "0.4 missed"),
    # appended after the first run: the decoys and the small causes on libero_object
    ("illumination_change", "2 stops dimmer"),
    ("backdrop_change", "1 walls and floor"),
    ("illumination_change", "1 stops dimmer"),
    ("gripper_wear", "0.25 wear"),
    # milder levels, flat on libero_spatial, to see whether they act on libero_object
    ("action_latency", "8 steps"),
    ("action_latency", "14 steps"),
    ("camera_degradation", "2 px blur"),
    ("furniture_shift", "2 cm sideways"),
)
# The joint points first (the worlds' interactions depend on them), then libero_object.
POINTS: tuple[Point, ...] = (
    *(Point(SPATIAL, pair) for pair in _PAIRS),
    *(Point(OBJECT, pair) for pair in _PAIRS),
    *(Point(OBJECT, (single,)) for single in _OBJECT_SINGLES),
    # Append, never renumber the recorded blocks. Future measurements include
    # singles from this world seed rather than borrowing the w90 calibration.
    *(Point(SPATIAL, (single,)) for single in dict.fromkeys(m for pair in _PAIRS for m in pair)),
)


def block_of(index: int) -> int:
    return 2 + index


def episode_id(block: int, index: int) -> str:
    return neutral_id(IdKind.episode, block * BLOCK_WIDTH + index)


def fault(kind: str, label: str, number: int) -> FaultSpec:
    """The fault of a calibration point at the narrowest scope its type allows."""
    import fleetlab.faults as faults
    from fleetlab.faults._common import target_of
    from fleetlab.worlds.compose import dose_point

    p = dose_point(kind, label)
    truth = faults.get(kind).truth
    if ScopeKind.robot in truth.scopes:
        scope = ROBOT_ID
    elif ScopeKind.site in truth.scopes:
        scope = SITE_ID
    else:
        scope = "fleet"
    spec = FaultSpec(
        fault_id=neutral_id(IdKind.fault, number),
        kind=kind,
        params=dict(p.params),
        level=float(p.level),
        target=target_of(scope, p.target, truth.target_kind),
        schedule=Schedule(onset_day=0),
        nominal=p.nominal,
    )
    faults.REGISTRY.check(spec)
    return spec


def identity(
    block: int,
    index: int,
    suite: str,
    policy: PolicySpec,
    software: SoftwareVersion,
    faults: Sequence[FaultSpec] = (),
) -> EpisodeIdentity:
    task, start = index % 10, index // 10
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
        batch_size=4,
        row=None,
        image_size=360,
        max_steps=280,
        faults=list(faults),
        software=software,
    )


def plan(policy: PolicySpec, software: SoftwareVersion) -> list[Job]:
    """Every job of the supplement: the two healthy blocks, then the points in order."""
    jobs = [
        Job(identity=identity(block, i, suite, policy, software))
        for suite, block in sorted(HEALTHY_BLOCKS.items(), key=lambda kv: kv[1])
        for i in range(EPISODES)
    ]
    for n, point in enumerate(POINTS):
        faults = [fault(k, label, m + 1) for m, (k, label) in enumerate(point.members)]
        jobs += [
            Job(identity=identity(block_of(n), i, point.suite, policy, software, faults))
            for i in range(EPISODES)
        ]
    return jobs


def run(root: Path = DEFAULT_ROOT, workers: int = 10) -> int:
    """The supplement's runs through the engine (heavy); resumable."""
    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    from fleetlab.engine import fleet, miopen
    from fleetlab.engine import software as soft
    from fleetlab.worlds.generate import policy_spec_nominal

    miopen.activate()
    software = soft.current(miopen.require(4, "fp16_res32"))
    jobs = plan(policy_spec_nominal(), software)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    config = fleet.RunConfig(
        out=str(root / RUN_NAME),
        workers=workers,
        batch_size=4,
        precision="fp16_res32",
        episodes_per_robot_day=EPISODES,
    )
    result = fleet.run(config, jobs)
    line = {
        "jobs": len(jobs),
        "chain_ok": result.record.chain_ok,
        "throughput": result.record.extra.get("throughput"),
    }
    print(json.dumps(line), flush=True)
    return 0 if result.record.chain_ok else 1


def _role(r: Any) -> str:
    from fleetlab.dose_response import decoy_level

    if r.p_value < 0.05 and r.difference[0] < 0:
        return "effect"
    if decoy_level(r):
        return "decoy"
    return "small_cause"


def analyse(root: Path = DEFAULT_ROOT, out: Path | None = None) -> dict[str, Any]:
    """The supplement's points against their healthy twins, written for `curves`."""
    from fleetlab.dose_response import by_index, compare, logged_outcome, read_run, true_outcome

    run = read_run(root / RUN_NAME)
    healthy = {suite: by_index(run, block) for suite, block in HEALTHY_BLOCKS.items()}
    identities = {
        job.identity.episode_id: job.identity
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if (job := Job.model_validate_json(line))
    }
    points: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    singles_by_key: dict[tuple[str, str, str], Any] = {}
    results: list[tuple[Point, Any]] = []
    for n, point in enumerate(POINTS):
        faulted = by_index(run, block_of(n))
        if not faulted or not healthy[point.suite]:
            continue
        from fleetlab.worlds.compose import dose_point, fault_class

        kinds = [k for k, _ in point.members]
        data = all(fault_class(k).value == "data" for k in kinds)
        label = " + ".join(lb for _, lb in point.members)
        level = float(dose_point(*point.members[0]).level)
        r = compare(
            healthy[point.suite],
            faulted,
            label,
            level,
            run,
            logged_outcome if data else true_outcome,
        )
        results.append((point, r))
        if len(point.members) == 1:
            singles_by_key[(point.suite, *point.members[0])] = faulted
    for point, r in results:
        record = {
            "suite": point.suite,
            "n": r.n,
            "successes": r.successes,
            "libero_successes": r.libero_successes,
            "logged_successes": r.logged_successes,
            "healthy_successes": r.healthy_successes,
            "broken": r.broken,
            "rescued": r.rescued,
            "p_value": round(r.p_value, 6),
            "difference": [round(x, 4) for x in r.difference],
            "run_id": r.run_id,
            "chain_ok": r.chain_ok,
            "role": _role(r),
        }
        if len(point.members) == 1:
            kind, label = point.members[0]
            points.append({"kind": kind, "label": label, "level": r.level, **record})
        else:
            singles = [singles_by_key.get((point.suite, *m)) for m in point.members]
            stats = None
            if singles[0] is not None and singles[1] is not None:
                index = POINTS.index(point)
                joint = by_index(run, block_of(index))
                h = healthy[point.suite]
                sets = [h, singles[0], singles[1], joint]
                common = set.intersection(*(set(x) for x in sets))
                if common and all(set(x) == common for x in sets):
                    for i in common:
                        twins = [identities[x[i].episode_id] for x in sets]
                        signatures = [
                            {
                                k: v
                                for k, v in t.model_dump().items()
                                if k not in {"episode_id", "faults"}
                            }
                            for t in twins
                        ]
                        if any(t != signatures[0] for t in signatures[1:]):
                            raise ValueError("joint and singles are not the same attempts")
                    from fleetlab.dose_response_stage3 import interaction

                    stats = interaction(h, singles[0], singles[1], joint)
            paired = (
                {}
                if stats is None
                else {
                    "single_successes": [stats["a"], stats["b"]],
                    "singles": [stats["a"] / stats["healthy"], stats["b"] / stats["healthy"]],
                    "interaction": stats["interaction"],
                    "interaction_role": stats["role"],
                    "effects": {k: stats[k] for k in ("effect_a", "effect_b", "effect_joint")},
                }
            )
            joints.append(
                {
                    "members": [{"kind": k, "label": lb} for k, lb in point.members],
                    "singles": None,
                    **record,
                    **paired,
                }
            )
    healthy_rates = {
        suite: {"n": len(h), "successes": sum(o.true_success for o in h.values())}
        for suite, h in healthy.items()
        if h
    }
    document = {
        "about": "Supplement to fleetlab/faults/curves.json, measured by fleetlab.worlds."
        "supplement (robots-75s.2): libero_object points and the interaction pairs' joint "
        "points, X-VLA fp16_res32, four batch rows, the fleet's check (placed), 100 episodes "
        "per point paired with healthy twins of the same suite (tasks 0-9 by start states "
        "0-9, world w91). role: effect (fewer successes beyond noise, exact McNemar p < 0.05), "
        "decoy (no change beyond noise and a two-point loss excluded), small_cause (neither). "
        "singles and interaction: only when healthy, A, B and A+B were measured on "
        "identical attempts in this run; absent when those singles were not run.",
        "run": str(root / RUN_NAME),
        "healthy": healthy_rates,
        "points": points,
        "joints": joints,
    }
    target = out or Path(__file__).with_name(SUPPLEMENT_FILE)
    target.write_text(json.dumps(document, indent=1) + "\n")
    return document


# Round 3: a fixed screen followed by one independent confirmation. Keep this
# separate from the historical supplement blocks; their identities never change.
MILD_PAIRS = ((8, 10), (8, 15), (10, 15), (10, 20))
MILD_SEED = 941010
MILD_WORLD = "w94"
MILD_ROOT = Path.home() / ".cache/robots-probes/worlds-v3/20261010-mild-pair"


def mild_cells(pair: tuple[int, int] | None = None) -> tuple[tuple[int, int], ...]:
    """(delay, chunk); zero means the corresponding fault is absent."""
    if pair is not None:
        return ((0, 0), (pair[0], 0), (0, pair[1]), pair)
    return ((0, 0), (8, 0), (10, 0), (0, 10), (0, 15), (0, 20), *MILD_PAIRS)


def mild_faults(delay: int, chunk: int) -> list[FaultSpec]:
    """Existing mechanisms at the predeclared doses; no runtime mechanism changes."""
    out = []
    if delay:
        out.append(fault("action_latency", "8 steps", 1).model_copy(update={"level": delay}))
    if chunk:
        out.append(
            fault("policy_knob", "10 actions per chunk", 2).model_copy(
                update={"params": {"value": chunk}}
            )
        )
    return out


def mild_plan(
    policy: PolicySpec,
    software: SoftwareVersion,
    *,
    pair: tuple[int, int] | None = None,
    confirmation_episodes: int = 200,
) -> list[Job]:
    """Same attempts across arms; confirmation is disjoint from every screen arm."""
    if confirmation_episodes not in (100, 200):
        raise ValueError("confirmation uses 100 or 200 independent attempts per arm")
    first, end = (0, 100) if pair is None else (100, 100 + confirmation_episodes)
    jobs = []
    cells = mild_cells(pair)
    for suite_number, suite in enumerate((SPATIAL, OBJECT)):
        for cell_number, cell in enumerate(cells):
            block = 1 + suite_number * len(cells) + cell_number
            for index in range(first, end):
                scene, noise = seeds_for(MILD_SEED, ROBOT_ID, DAY, index)
                ident = identity(block, index, suite, policy, software, mild_faults(*cell))
                ident = ident.model_copy(
                    update={"world_id": MILD_WORLD, "scene_seed": scene, "noise_seed": noise}
                )
                jobs.append(Job(identity=ident))
    return jobs


def mild_statistics(
    root: Path, pair: tuple[int, int] | None = None, *, confirmation_episodes: int = 200
) -> dict[str, Any]:
    """Only complete, replay-validated stages with exact planned identities are analysed."""
    from fleetlab import dose_response as dr
    from fleetlab.dose_response_stage3 import interaction, point_dict
    from fleetlab.log.runfiles import read_identities
    from fleetlab.worlds.curves import load

    jobs = [
        Job.model_validate_json(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    wanted = {job.identity.episode_id: job.identity for job in jobs}
    run = dr.read_run(root / "run")
    if run.chain_ok is not True or set(run.outcomes) != set(wanted):
        raise ValueError("mild calibration requires every planned outcome and a held chain sample")
    recorded = {i.episode_id: i for i in read_identities(root / "run") if i.episode_id in wanted}
    if recorded != wanted:
        raise ValueError("recorded calibration identities differ from the sealed manifest")
    cells = mild_cells(pair)
    arms: dict[tuple[str, tuple[int, int]], dict[int, dr.Outcome]] = {}
    signatures: dict[tuple[str, int], dict[str, Any]] = {}
    for ident in wanted.values():
        d = next((int(f.level) for f in ident.faults if f.kind == "action_latency"), 0)
        c = next((int(str(f.params["value"])) for f in ident.faults if f.kind == "policy_knob"), 0)
        key = (ident.suite, ident.episode_index)
        signature = ident.model_dump(exclude={"episode_id", "faults"})
        previous = signatures.setdefault(key, signature)
        if previous != signature:
            raise ValueError("the four arms are not the same attempts")
        arms.setdefault((ident.suite, (d, c)), {})[ident.episode_index] = run.outcomes[
            ident.episode_id
        ]
    if confirmation_episodes not in (100, 200):
        raise ValueError("confirmation uses 100 or 200 independent attempts per arm")
    expected_indices = set(
        range(0, 100) if pair is None else range(100, 100 + confirmation_episodes)
    )
    if any(set(arm) != expected_indices for arm in arms.values()) or len(arms) != 2 * len(cells):
        raise ValueError("calibration arms have incomplete or unequal attempt indices")
    points: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    healthy = {}
    for suite in (SPATIAL, OBJECT):
        h = arms[(suite, (0, 0))]
        healthy[suite] = {"n": len(h), "successes": sum(x.true_success for x in h.values())}
        if healthy[suite]["successes"] / len(h) < 0.90:
            raise ValueError("healthy calibration control below 90%; stop before using the points")
        for cell in cells[1:]:
            delay, chunk = cell
            label = f"{delay} steps" if delay else f"{chunk} actions per chunk"
            kind = "action_latency" if delay else "policy_knob"
            outcome = arms[(suite, cell)]
            result = dr.compare(h, outcome, label, float(delay or 1), run)
            record = {**point_dict(result, dr.point_role(result)), "suite": suite, "chain_ok": True}
            if delay and chunk:
                stats = interaction(
                    h, arms[(suite, (delay, 0))], arms[(suite, (0, chunk))], outcome
                )
                members = [
                    {"kind": "action_latency", "label": f"{delay} steps"},
                    {"kind": "policy_knob", "label": f"{chunk} actions per chunk"},
                ]
                joints.append(
                    {
                        **record,
                        "dose": list(cell),
                        "members": members,
                        "single_successes": [stats["a"], stats["b"]],
                        "singles": [stats["a"] / stats["healthy"], stats["b"] / stats["healthy"]],
                        "interaction": stats["interaction"],
                        "interaction_role": stats["role"],
                        "effects": {
                            key: stats[key] for key in ("effect_a", "effect_b", "effect_joint")
                        },
                        "corrected_joint_success": load().healthy[suite]
                        * stats["joint"]
                        / stats["healthy"],
                    }
                )
            else:
                points.append({"kind": kind, **record})
    return {
        "stage": "screen" if pair is None else "confirmation",
        "n_per_arm": 100 if pair is None else confirmation_episodes,
        "pair": None if pair is None else list(pair),
        "run": str(root / "run"),
        "healthy": healthy,
        "points": points,
        "joints": joints,
        "chain_ok": True,
        "selection_data_pooled": False,
    }


def mild_eligible(joint: dict[str, Any], floor: float) -> bool:
    """Predeclared small-single, floor and paired-superadditivity criterion."""
    return (
        joint["corrected_joint_success"] >= floor
        and joint["interaction"][2] < 0
        and all(joint["effects"][name][0] >= -0.10 for name in ("effect_a", "effect_b"))
    )


def mild_select(document: dict[str, Any]) -> tuple[int, int] | None:
    """Fixed screen rule, not a second selection after inspecting confirmation."""
    ranked = []
    for order, pair in enumerate(MILD_PAIRS):
        eligible = [
            j for j in document["joints"] if tuple(j["dose"]) == pair and mild_eligible(j, 0.45)
        ]
        if eligible:
            distance = sum(abs(j["corrected_joint_success"] - 0.60) for j in eligible) / len(
                eligible
            )
            ranked.append((-len(eligible), distance, order, pair))
    return min(ranked)[-1] if ranked else None


def mild_stage(
    root: Path, workers: int, pair: tuple[int, int] | None, *, confirmation_episodes: int = 200
) -> int:
    """One measured stage, invoked in a child process so model memory is released."""
    from fleetlab.engine import fleet, miopen
    from fleetlab.engine import software as soft
    from fleetlab.worlds.generate import policy_spec_nominal

    if not 1 <= workers <= 6:
        raise ValueError("at most six workers")
    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    miopen.activate()
    jobs = mild_plan(
        policy_spec_nominal(),
        soft.current(miopen.require(4, "fp16_res32")),
        pair=pair,
        confirmation_episodes=confirmation_episodes,
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest = "".join(j.model_dump_json() + "\n" for j in jobs)
    path = root / "manifest.jsonl"
    if path.exists() and path.read_text() != manifest:
        raise ValueError("refusing a changed calibration manifest")
    if not path.exists():
        path.write_text(manifest)
    result = fleet.run(
        fleet.RunConfig(
            out=str(root / "run"),
            workers=workers,
            batch_size=4,
            precision="fp16_res32",
            episodes_per_robot_day=300,
        ),
        jobs,
    )
    if result.record.chain_ok is not True:
        return 1
    document = mild_statistics(root, pair, confirmation_episodes=confirmation_episodes)
    (root / "analysis.json").write_text(json.dumps(document, indent=2) + "\n")
    return 0


def mild_driver(root: Path, workers: int, seconds: float) -> int:
    """Single queued job, bounded after GPU admission; never hides failed/partial stages."""
    import signal
    import subprocess
    import sys
    import time
    from datetime import UTC, datetime
    from types import FrameType

    if not 1 <= workers <= 6 or not 0 < seconds <= 10700:
        raise ValueError("at most six workers and 10,700 driver seconds")
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    receipt: dict[str, Any] = {
        "started": datetime.now(UTC).isoformat(),
        "budget_seconds": seconds,
        "workers": workers,
        "stages": [],
        "state": "running",
    }
    attempt = len(list(root.glob("execution-*.json"))) + 1
    ledger = root / f"execution-{attempt:02d}.json"
    active: subprocess.Popen[Any] | None = None

    def stop(signum: int, frame: FrameType | None) -> None:
        # The outer hard timeout must kill the stage's separate process group too.
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait()
        receipt["state"] = f"interrupted by signal {signum}"
        raise SystemExit(124)

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)

    def save() -> None:
        receipt["elapsed_seconds"] = round(time.monotonic() - start, 3)
        ledger.write_text(json.dumps(receipt, indent=2) + "\n")

    def stage(name: str, pair: tuple[int, int] | None = None) -> int:
        nonlocal active
        remaining = seconds - (time.monotonic() - start)
        if remaining <= 30:
            return 124
        command = [
            sys.executable,
            "-m",
            "fleetlab.worlds.supplement",
            "stage",
            "--root",
            str(root / name),
            "--workers",
            str(workers),
        ]
        if pair is not None:
            command += ["--candidate", str(MILD_PAIRS.index(pair))]
        began = time.monotonic()
        with (root / f"{name}-attempt-{attempt:02d}.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            active = proc
            try:
                while proc.poll() is None:
                    save()
                    if time.monotonic() - start >= seconds - 15:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        receipt["state"] = "budget exhausted"
                        return 124
                    time.sleep(2)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                active = None
        code = int(proc.returncode)
        receipt["stages"].append({"stage": name, "exit": code, "seconds": time.monotonic() - began})
        save()
        return code

    try:
        save()
        code = stage("screen")
        if code:
            receipt["state"] = "screen failed or interrupted"
            return code
        screen = json.loads((root / "screen/analysis.json").read_text())
        selected = mild_select(screen)
        selection = {"pair": selected, "rule": "e98f814 README; independent confirmation only"}
        (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        if selected is None:
            receipt["state"] = "no screen candidate satisfies the criteria"
            return 2
        forecast = receipt["stages"][0]["seconds"] * 0.8 * 1.15 + 60
        if seconds - (time.monotonic() - start) < forecast:
            receipt["state"] = "insufficient remaining budget for confirmation"
            receipt["confirmation_forecast_seconds"] = forecast
            return 124
        code = stage("confirmation", selected)
        if code:
            receipt["state"] = "confirmation failed or interrupted"
            return code
        confirmed = json.loads((root / "confirmation/analysis.json").read_text())
        eligible = [j["suite"] for j in confirmed["joints"] if mild_eligible(j, 0.40)]
        receipt["eligible_suites"] = eligible
        receipt["state"] = "confirmed" if eligible else "no confirmed feasible interaction"
        return 0 if eligible else 2
    finally:
        save()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def blur_plan(confirmation: Path) -> list[Job]:
    """The predeclared 3-pixel fleet fallback, paired to independent healthy controls.

    Run this manifest with the engine CLI from the original frozen checkout. No
    healthy outcome is resimulated or substituted, and no software field is changed.
    """
    document = json.loads((confirmation / "analysis.json").read_text())
    if (
        document.get("stage") != "confirmation"
        or document.get("chain_ok") is not True
        or not any(mild_eligible(joint, 0.40) for joint in document["joints"])
    ):
        raise ValueError("fleet fallback requires an independently confirmed interaction")
    healthy = [
        Job.model_validate_json(line)
        for line in (confirmation / "manifest.jsonl").read_text().splitlines()
        if not json.loads(line)["identity"]["faults"]
    ]
    n = document.get("n_per_arm", 200)
    if n not in (100, 200):
        raise ValueError("fleet fallback needs the declared independent confirmation sample")
    keys = {(j.identity.suite, j.identity.episode_index) for j in healthy}
    if len(healthy) != 2 * n or keys != {
        (suite, index) for suite in (SPATIAL, OBJECT) for index in range(100, 100 + n)
    }:
        raise ValueError("fleet fallback needs all independent healthy identities")
    blur = fault("camera_degradation", "4 px blur", 1).model_copy(update={"level": 3.0})
    return [
        job.model_copy(
            update={
                "identity": job.identity.model_copy(
                    update={
                        "episode_id": neutral_id(
                            IdKind.episode,
                            800000
                            + (1000 if job.identity.suite == OBJECT else 0)
                            + job.identity.episode_index,
                        ),
                        "faults": [blur],
                    }
                )
            }
        )
        for job in healthy
    ]


def blur_statistics(root: Path, confirmation: Path) -> dict[str, Any]:
    """Analyse only the exact declared manifest with complete replay-validated outcomes."""
    from fleetlab import dose_response as dr
    from fleetlab.dose_response_stage3 import point_dict
    from fleetlab.log.runfiles import read_identities

    planned = blur_plan(confirmation)
    stored = [
        Job.model_validate_json(line) for line in (root / "manifest.jsonl").read_text().splitlines()
    ]
    if stored != planned:
        raise ValueError("fleet fallback manifest differs from the predeclared healthy twins")
    wanted = {j.identity.episode_id: j.identity for j in planned}
    run = dr.read_run(root / "run")
    healthy_run = dr.read_run(confirmation / "run")
    recorded = {i.episode_id: i for i in read_identities(root / "run") if i.episode_id in wanted}
    if run.chain_ok is not True or healthy_run.chain_ok is not True or recorded != wanted:
        raise ValueError("fleet fallback requires held chains and exact recorded identities")
    if set(run.outcomes) != set(wanted):
        raise ValueError("fleet fallback requires every planned outcome")
    healthy_jobs = [
        Job.model_validate_json(line)
        for line in (confirmation / "manifest.jsonl").read_text().splitlines()
    ]
    healthy_ids = {j.identity.episode_id: j.identity for j in healthy_jobs if not j.identity.faults}
    recorded_healthy = {
        i.episode_id: i
        for i in read_identities(confirmation / "run")
        if i.episode_id in healthy_ids
    }
    if recorded_healthy != healthy_ids or not set(healthy_ids) <= set(healthy_run.outcomes):
        raise ValueError("fleet fallback's healthy control is incomplete or changed")
    points = []
    for suite in (SPATIAL, OBJECT):
        baseline = {
            i.episode_index: healthy_run.outcomes[eid]
            for eid, i in healthy_ids.items()
            if i.suite == suite
        }
        measured = {
            i.episode_index: run.outcomes[eid] for eid, i in wanted.items() if i.suite == suite
        }
        if sum(out.true_success for out in baseline.values()) < 0.90 * len(baseline):
            raise ValueError("healthy fleet calibration control below 90%")
        result = dr.compare(baseline, measured, "3 px blur", 3.0, run)
        points.append(
            {
                "kind": "camera_degradation",
                "suite": suite,
                "chain_ok": True,
                **point_dict(result, dr.point_role(result)),
                "run_id": str(root / "run"),
            }
        )
    return {
        "stage": "fleet-blur",
        "dose": 3,
        "n_per_suite": len(planned) // 2,
        "healthy_run": str(confirmation / "run"),
        "run": str(root / "run"),
        "points": points,
        "chain_ok": True,
        "effective_on_both_suites": all(p["role"] == "effect" for p in points),
    }


def validate_resource_confirmation(root: Path, pair: tuple[int, int]) -> None:
    """Never turn an incomplete 200-attempt confirmation into a smaller accepted one."""
    receipts = sorted(root.glob("execution-*.json"))
    if not receipts:
        raise ValueError("the primary driver has no recorded budget refusal")
    receipt = json.loads(receipts[-1].read_text())
    selection = json.loads((root / "selection.json").read_text())
    if (
        receipt["state"] != "insufficient remaining budget for confirmation"
        or (root / "confirmation").exists()
        or any(stage["stage"] == "confirmation" for stage in receipt["stages"])
        or selection["pair"] != list(pair)
    ):
        raise ValueError("reduced confirmation requires the same selection and an unstarted stage")


def mild_main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Predeclared mild interaction calibration")
    parser.add_argument("action", choices=("run", "stage"))
    parser.add_argument("--root", type=Path, default=MILD_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=10700)
    parser.add_argument("--candidate", type=int, choices=range(len(MILD_PAIRS)))
    parser.add_argument("--confirmation-episodes", type=int, choices=(100, 200), default=200)
    parser.add_argument("--budget-refusal", type=Path)
    args = parser.parse_args()
    if args.confirmation_episodes == 100:
        if args.action != "stage" or args.candidate is None or args.budget_refusal is None:
            parser.error(
                "reduced confirmation requires a standalone selected stage and budget refusal"
            )
        validate_resource_confirmation(args.budget_refusal, MILD_PAIRS[args.candidate])
    if args.action == "stage":
        pair = None if args.candidate is None else MILD_PAIRS[args.candidate]
        return mild_stage(
            args.root, args.workers, pair, confirmation_episodes=args.confirmation_episodes
        )
    return mild_driver(args.root, args.workers, args.seconds)


if __name__ == "__main__":
    raise SystemExit(mild_main())


# Task-level materialization of existing measurements (CPU only).
@cache
def _task_rows(path: str) -> list[tuple[EpisodeIdentity, Any]]:
    from fleetlab import dose_response as dr
    from fleetlab.log.runfiles import read_identities

    run = dr.read_run(Path(path))
    if run.chain_ok is not True:
        raise ValueError(("unconfirmed chain", path))
    identities = read_identities(path)
    ids = [i.episode_id for i in identities]
    if len(set(ids)) != len(ids) or set(ids) != set(run.outcomes):
        raise ValueError(("incomplete or changed calibration outcomes", path))
    return [(i, run.outcomes[i.episode_id]) for i in identities]


def _task_signature(i: EpisodeIdentity) -> tuple[Any, ...]:
    return (
        i.suite,
        i.task_index,
        i.start_state,
        i.scene_seed,
        i.noise_seed,
        i.policy.model_dump_json(),
        i.batch_size,
        i.image_size,
        i.max_steps,
        i.software.model_dump_json(exclude={"lock_hash"}),
    )


def _task_collect(
    paths: Sequence[str], suite: str, members: Sequence[tuple[str, str]]
) -> dict[tuple[Any, ...], tuple[int, int]]:
    out: dict[tuple[Any, ...], tuple[int, int]] = {}
    for path in paths:
        for i, o in _task_rows(str(path)):
            if i.suite != suite:
                continue
            labels = tuple(sorted((f.kind, calibrated_label(f)) for f in i.faults))
            if labels != tuple(sorted(members)):
                continue
            key = _task_signature(i)
            from fleetlab.worlds.estimate import is_data

            analysed = o.logged_success if any(is_data(f) for f in i.faults) else o.true_success
            value = (int(analysed), int(o.logged_success))
            if key in out and out[key] != value:
                raise ValueError(("conflicting duplicate", path, key))
            out[key] = value
    return out


def task_calibration_record() -> dict[str, Any]:
    from importlib import resources

    library = json.loads(resources.files("fleetlab.faults").joinpath("curves.json").read_text())
    sup = json.loads(
        resources.files("fleetlab.worlds").joinpath("curves_supplement.json").read_text()
    )
    entries: dict[tuple[tuple[tuple[str, str], ...], str], dict[str, Any]] = {}
    for kind, e in library["types"].items():
        for p in e["points"]:
            entries[((kind, p["label"]),), "libero_spatial"] = p
    for doc in (sup, library["supplement"]):
        for p in doc.get("points", []):
            entries[((p["kind"], p["label"]),), p["suite"]] = {**p, "fallback_run": doc.get("run")}
        for p in doc.get("joints", []):
            members = tuple(sorted((m["kind"], m["label"]) for m in p["members"]))
            entries[members, p["suite"]] = {**p, "fallback_run": doc.get("run")}
    out: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for (members, suite), p in entries.items():
        if any(k == "success_false_positive" for k, _ in members):
            excluded.append(
                {
                    "members": members,
                    "suite": suite,
                    "reason": "Conditional on a camera fault, not a stand-alone task ratio.",
                }
            )
            continue
        paths = p.get("run_dirs") or [p["fallback_run"]]
        healthy_paths = p.get("healthy_run_dirs") or [p["fallback_run"]]
        a = _task_collect(paths, suite, members)
        h = _task_collect(healthy_paths, suite, ())
        if not a or not set(a) <= set(h):
            raise ValueError(("unpaired", members, suite, len(a), len(set(a) - set(h))))
        singles = []
        if p.get("single_successes") is not None:
            by_member = {
                tuple((m["kind"], m["label"])): paths
                for m, paths in zip(
                    p["members"], p.get("single_run_dirs", [paths, paths]), strict=True
                )
            }
            singles = [_task_collect(by_member[m], suite, [m]) for m in members]
            if any(not set(a) <= set(v) for v in singles):
                raise ValueError(("unpaired singles", members, suite))
        cells: list[dict[str, Any]] = []
        for task in range(10):
            ks = [k for k in a if k[1] == task]
            cells.append(
                {
                    "task_index": task,
                    "n": len(ks),
                    "successes": sum(a[k][0] for k in ks),
                    "logged_successes": sum(a[k][1] for k in ks),
                    "healthy_successes": sum(h[k][0] for k in ks),
                    "paired_counts": [
                        sum((a[k][0], h[k][0]) == pair for k in ks)
                        for pair in ((1, 1), (1, 0), (0, 1), (0, 0))
                    ],
                }
            )
            if singles:
                import numpy as np

                values = np.array(
                    [a[k][0] - singles[0][k][0] - singles[1][k][0] + h[k][0] for k in ks]
                )
                samples = (
                    np.random.default_rng(0).choice(values, size=(10000, len(ks))).mean(axis=1)
                )
                cells[-1]["single_successes"] = [sum(v[k][0] for k in ks) for v in singles]
                cells[-1]["interaction"] = [
                    float(values.mean()),
                    *map(float, np.quantile(samples, [0.025, 0.975])),
                ]
        for field in ("n", "successes", "healthy_successes", "logged_successes"):
            if field in p and sum(c[field] for c in cells) != p[field]:
                raise ValueError(
                    (
                        "counts disagree",
                        members,
                        suite,
                        field,
                        sum(c[field] for c in cells),
                        p[field],
                    )
                )
        out.append(
            {
                "members": [list(m) for m in members],
                "suite": suite,
                "run_dirs": paths,
                "healthy_run_dirs": healthy_paths,
                "tasks": cells,
            }
        )
    healthy = {}
    for suite, p in {
        "libero_spatial": library["healthy"],
        **library["supplement"]["healthy"],
    }.items():
        a = _task_collect(p["run_dirs"], suite, ())
        cells = [
            {
                "task_index": t,
                "n": sum(k[1] == t for k in a),
                "successes": sum(v[0] for k, v in a.items() if k[1] == t),
            }
            for t in range(10)
        ]
        assert sum(c["n"] for c in cells) == p["n"]
        assert sum(c["successes"] for c in cells) == p["successes"]
        healthy[suite] = {"run_dirs": p["run_dirs"], "tasks": cells}
    return {
        "method": "Exact recorded cohorts, grouped by task; healthy twins matched by simulator "
        "and policy identity; duplicate attempts counted once. No screening data.",
        "points": out,
        "healthy": healthy,
        "excluded": excluded,
    }
