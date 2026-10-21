"""Step-response probes on one robot under each lookalike condition (lane P, step 1).

    tools/heavy-run --light -m 10G uv run --locked python spikes/instrument/probe_shapes.py \
        OUT [--workers 3] [--only NAME ...] [--program step|train] [--scene SUITE:TASK]

The host-selected scripted probe check of docs/design/loop.md section 10 ("Measured"),
repeated on robot r01 of the miniature fleet w10-v2 under one planted condition at a
time. It reuses the probe route as it is: the probe player as the in-worker policy
(`fleetlab.engine.probe_player`), the probe run's configuration (`loop.probes.probe_config`:
no policy server, CPU, no video), the engine (`fleetlab.engine.fleet.run`, with its chain
sample), the export's conversion of a probe run's visible steps (`loop.probes.probe_step_rows`)
and the lag read-out (`loop.probes.step_lags`). What it does not reuse is the loop's
planning (`plan_probes`), which takes a robot's faults from the stored world's schedule:
w10-v2 carries none of these conditions, so each episode's identity is minted here the
way `plan_probes` mints it (round 1's experiment day, an index of the day after the
world's own episodes, `seeds_for`), with exactly one planted fault (none when healthy).
Every condition runs in a run of its own with the same seeds, so repetition r of every
condition is the same episode apart from the fault.

The program is the section-10 check's: hold 10, step +3 cm on x for 40, step back to 0
for 40, hold 10 control steps (100 commanded steps), three repetitions.

This is host-side development evidence: the conditions and their levels are ours (the
"kept" levels of docs/design/faults.md sections 6 and 8), not an agent's choice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORLD = Path.home() / ".cache" / "robots-probes" / "mini" / "w10-v2"
ROBOT = "r01"
REPEATS = 3
FIRST_EPISODE = 500_000  # far above the world's own and any loop allocation (417 on)

SEGMENTS: list[dict[str, Any]] = [
    {
        "primitive": "hold",
        "axis": None,
        "amplitude_m": None,
        "duration_steps": 10,
        "period_steps": None,
    },
    {
        "primitive": "step",
        "axis": "x",
        "amplitude_m": 0.03,
        "duration_steps": 40,
        "period_steps": None,
    },
    {
        "primitive": "step",
        "axis": "x",
        "amplitude_m": 0.0,
        "duration_steps": 40,
        "period_steps": None,
    },
    {
        "primitive": "hold",
        "axis": None,
        "amplitude_m": None,
        "duration_steps": 10,
        "period_steps": None,
    },
]


def _step_x(amplitude: float, steps: int = 40) -> dict[str, Any]:
    return {
        "primitive": "step",
        "axis": "x",
        "amplitude_m": amplitude,
        "duration_steps": steps,
        "period_steps": None,
    }


# A train of four edges (rows 10, 50, 90, 130; 180 steps): whether the dead time at each
# edge stays the same or grows over the probe.
TRAIN: list[dict[str, Any]] = [
    SEGMENTS[0],
    _step_x(0.03),
    _step_x(0.0),
    _step_x(0.03),
    _step_x(0.0),
    SEGMENTS[0],
]
PROGRAMS: dict[str, list[dict[str, Any]]] = {"step": SEGMENTS, "train": TRAIN}


@dataclass(frozen=True)
class Condition:
    name: str
    label: str  # for the figure and the table
    kind: str | None  # None: healthy
    level: float = 0.0
    component: str | None = None  # a component fault's component
    config_key: str | None = None  # a configuration fault's key
    nominal: Any = None
    params: dict[str, Any] | None = None


# The kept levels (docs/design/faults.md, sections 6 and 8.5); the adaptation spike's delay.
CONDITIONS: tuple[Condition, ...] = (
    Condition("healthy", "healthy", None),
    Condition(
        "remote_link_latency-17",
        "remote link, 17 steps",
        "remote_link_latency",
        17.0,
        component="controller",
    ),
    Condition(
        "controller_overload-0.1",
        "controller overload 0.1",
        "controller_overload",
        0.1,
        component="controller",
    ),
    Condition(
        "controller_overload-0.2",
        "controller overload 0.2",
        "controller_overload",
        0.2,
        component="controller",
    ),
    Condition(
        "controller_overload-0.3",
        "controller overload 0.3",
        "controller_overload",
        0.3,
        component="controller",
    ),
    Condition(
        "arm_torque_loss-0.3", "arm torque loss 0.3", "arm_torque_loss", 0.3, component="arm"
    ),
    Condition(
        "arm_torque_loss-0.4", "arm torque loss 0.4", "arm_torque_loss", 0.4, component="arm"
    ),
    Condition(
        "arm_torque_loss-0.5", "arm torque loss 0.5", "arm_torque_loss", 0.5, component="arm"
    ),
    Condition(
        "camera_latency-24",
        "camera latency, 24 steps",
        "camera_latency",
        24.0,
        component="scene_camera",
    ),
    Condition(
        "camera_pose_offset-6",
        "scene camera panned 6 deg",
        "camera_pose_offset",
        6.0,
        component="scene_camera",
    ),
    Condition(
        "image_pipeline_release-crop20",
        "image release, 20% crop",
        "image_pipeline_release",
        20.0,
        config_key="software.version",
        nominal="2.4.1",
        params={"mode": "crop", "version": "2.6.0"},
    ),
    Condition(
        "calibration_update",
        "calibration rewritten",
        "calibration_update",
        1.0,
        config_key="camera.scene.calibration",
        nominal=[0.66, 0.0, 1.61, 0.0, 33.0, 90.0],
        params={"value": [0.662, -0.003, 1.608, 0.0, 33.4, 90.2]},
    ),
)


def fault_of(condition: Condition) -> list[Any]:
    from fleetlab.contracts.faults import FaultSpec, Schedule, Target
    from fleetlab.contracts.vocabulary import Component, ConfigKey

    if condition.kind is None:
        return []
    if condition.component is not None:
        target = Target(scope=ROBOT, component=Component(condition.component))
    else:
        assert condition.config_key is not None
        target = Target(scope=ROBOT, config_key=ConfigKey(condition.config_key))
    return [
        FaultSpec(
            fault_id="f21",
            kind=condition.kind,
            params=dict(condition.params or {}),
            level=condition.level,
            target=target,
            schedule=Schedule(onset_day=0),
            nominal=condition.nominal,
        )
    ]


def jobs_for(
    world: Any,
    layout: Any,
    index: int,
    condition: Condition,
    software: Any,
    segments: list[dict[str, Any]],
    scene: tuple[str, int] | None = None,
) -> list[Any]:
    """One probe episode per repetition, minted as `loop.probes.plan_probes` mints them;
    `scene` (suite, task index) replaces the robot's first task, as a robot whose first
    task is another would have it."""
    from fleetlab.contracts.identity import EpisodeIdentity, seeds_for
    from fleetlab.contracts.ids import IdKind, neutral_id
    from fleetlab.contracts.probes import ProbeProgram
    from fleetlab.contracts.rollout import Job
    from fleetlab.engine.probe_player import probe_spec
    from fleetlab.loop.layout import days_of
    from fleetlab.loop.probes import PRECISION
    from fleetlab.loop.world import hook_chain

    truth = world.truth
    program = ProbeProgram.model_validate({"segments": segments})
    days = days_of(layout.round(1).experiment)
    robot = truth.fleet.robot(ROBOT)
    task = truth.tasks_of(ROBOT)[0]
    if scene is not None:
        task = next(t for t in truth.tasks if (t.suite, t.task_index) == scene)
    faults = fault_of(condition)
    out = []
    for repeat in range(REPEATS):
        day = days[repeat % len(days)]
        episode_index = truth.episodes_per_robot_day + repeat
        scene_seed, noise_seed = seeds_for(truth.world_seed, ROBOT, day, episode_index)
        identity = EpisodeIdentity(
            world_id=truth.world_id,
            episode_id=neutral_id(IdKind.episode, FIRST_EPISODE + 100 * index + repeat),
            robot_id=ROBOT,
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
            raise ValueError(f"{condition.name}: the probe cannot run under this condition")
        out.append(Job(identity=identity, record_video=False))
    return out


def edge_metrics(
    rows: list[dict[str, Any]], issued: int, change: float, end: int
) -> dict[str, Any]:
    """The x response to one commanded step issued in row `issued`, up to row `end`.

    Progress is the measured x minus the x measured in the issuing row, signed toward the
    step. Rows count from the issuing row; None when a threshold is never reached before
    `end`. Rise: rows from 10% to 90% of the step.
    """
    sign = 1.0 if change > 0 else -1.0
    size = abs(change)
    x0 = float(rows[issued]["eef_pos"][0])
    z0 = float(rows[0]["eef_pos"][2])
    progress = [
        (r["step"] - issued, (float(r["eef_pos"][0]) - x0) * sign)
        for r in rows[issued + 1 : end + 1]
    ]

    def first(share: float) -> int | None:
        return next((k for k, p in progress if p >= share * size), None)

    t10, t90 = first(0.1), first(0.9)
    final = progress[-1][1] / size
    peak = max(p for _, p in progress) / size
    return {
        "issued_at": issued,
        "change_m": change,
        "first_movement_rows": next((k for k, p in progress if p >= 0.001), None),
        "ten_percent_rows": t10,
        "half_way_rows": first(0.5),
        "ninety_percent_rows": t90,
        "rise_10_90_rows": None if t10 is None or t90 is None else t90 - t10,
        "reached_at_end": round(final, 4),
        "overshoot": round(max(0.0, peak - 1.0), 4),
        "end_row": end,
        "z_at_end_mm": round((float(rows[end]["eef_pos"][2]) - z0) * 1000, 2),
    }


def best_alignment_lag(rows: list[dict[str, Any]], max_lag: int = 30) -> dict[str, Any]:
    """`spikes/adaptation/lag.sql`'s estimate on one probe episode: the lag L whose
    commanded target L + 1 rows earlier lies closest (mean 3-D distance) to the measured
    position, scored on the same rows for every L (from row max_lag + 1 on)."""
    import numpy as np

    target = [r["target_pos"] for r in rows]
    measured = np.asarray([r["eef_pos"] for r in rows], np.float64)
    scores = []
    for lag in range(max_lag + 1):
        d = [
            float(np.linalg.norm(np.asarray(target[k - 1 - lag], np.float64) - measured[k]))
            for k in range(max_lag + 1, len(rows))
            if target[k - 1 - lag] is not None
        ]
        scores.append(sum(d) / len(d))
    best = int(np.argmin(scores))
    return {"lag_rows": best, "distance_mm": round(scores[best] * 1000, 3)}


def edges_of(program: Any) -> list[tuple[int, float, int]]:
    """(issuing row, change of x, last row read) of every step of the program: each
    step's response is read until the next step is issued, the last until the end."""
    from fleetlab.contracts.probes import ProbePrimitive, program_trajectory

    trajectory = program_trajectory(program.segments)
    steps = [i for i, s in enumerate(program.segments) if s.primitive is ProbePrimitive.step]
    out = []
    for n, i in enumerate(steps):
        issued = trajectory.first_step[i]
        change = float(program.segments[i].amplitude_m) - trajectory.base[i][0]
        end = trajectory.first_step[steps[n + 1]] if n + 1 < len(steps) else program.steps
        out.append((issued, change, end))
    return out


def analyse(
    run_dir: Path, condition: Condition, jobs: list[Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from fleetlab.contracts.probes import ProbeProgram
    from fleetlab.loop.probes import probe_step_rows, step_lags

    program = ProbeProgram.model_validate({"segments": segments})
    record = json.loads((run_dir / "run.json").read_text())
    episodes = []
    for repeat, job in enumerate(jobs):
        eid = job.identity.episode_id
        steps = pq.read_table(run_dir / "log" / "steps" / f"{eid}.parquet")
        rows = probe_step_rows(eid, program, steps)
        episodes.append(
            {
                "episode_id": eid,
                "repeat": repeat,
                "rows": len(rows),
                "step_lags": step_lags(rows),
                "edges": [
                    edge_metrics(rows, issued, round(change, 6), end)
                    for issued, change, end in edges_of(program)
                ],
                "alignment": best_alignment_lag(rows),
                "series": {
                    "step": [r["step"] for r in rows],
                    "target_x": [
                        None if r["target_pos"] is None else r["target_pos"][0] for r in rows
                    ],
                    "eef": [r["eef_pos"] for r in rows],
                },
            }
        )
    return {
        "condition": condition.name,
        "label": condition.label,
        "kind": condition.kind,
        "level": condition.level,
        "faults": [f.model_dump(mode="json") for f in fault_of(condition)],
        "run": str(run_dir),
        "chain_ok": record.get("chain_ok"),
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--program", choices=sorted(PROGRAMS), default="step")
    parser.add_argument("--scene", default=None, help="SUITE:TASK instead of the first task")
    args = parser.parse_args()

    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.engine import fleet
    from fleetlab.engine.fleet import RunConfig
    from fleetlab.log import run_status
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.operate import EngineSetup
    from fleetlab.loop.probes import probe_config
    from fleetlab.loop.world import LoopWorld

    world = LoopWorld.load(WORLD)
    layout = layout_for_world(world.spec)
    setup = EngineSetup(
        config=RunConfig(out="", workers=args.workers), contacts=False, events=False, video=False
    )
    per_day = world.truth.episodes_per_robot_day
    segments = PROGRAMS[args.program]
    scene = None
    if args.scene:
        suite, task_index = args.scene.split(":")
        scene = (suite, int(task_index))
    args.out.mkdir(parents=True, exist_ok=True)
    summary_file = args.out / "probe-shapes.json"
    kept: dict[str, Any] = {}
    if summary_file.is_file():  # a rerun of some conditions keeps the others' results
        kept = {r["condition"]: r for r in json.loads(summary_file.read_text())}
    order = [c.name for c in CONDITIONS]
    for index, condition in enumerate(CONDITIONS):
        if args.only and condition.name not in args.only:
            continue
        run_dir = args.out / "runs" / condition.name
        software = fleet.prepare(probe_config(setup, run_dir, per_day))
        jobs = jobs_for(world, layout, index, condition, software, segments, scene)
        done = (run_dir / "run.json").is_file() and run_status(run_dir).finalized
        if not done:
            fleet.run(probe_config(setup, run_dir, per_day), jobs)
        result = analyse(run_dir, condition, jobs, segments)
        kept[condition.name] = result
        summary = [
            (
                e["repeat"],
                [
                    (
                        g["first_movement_rows"],
                        g["half_way_rows"],
                        g["rise_10_90_rows"],
                        g["reached_at_end"],
                    )
                    for g in e["edges"]
                ],
                e["alignment"]["lag_rows"],
            )
            for e in result["episodes"]
        ]
        print(f"{condition.name}: chain_ok={result['chain_ok']} {summary}", flush=True)
        results = [kept[name] for name in order if name in kept]
        summary_file.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
