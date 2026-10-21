"""A small toy fleet for the skill tests, written through the episode log's writer.

`tests/log_toy.py` has ten episodes, too few for rates, samples and change
points. This fleet has six robots at two sites over five days, eight episodes
per robot and day (240 episodes), with three planted causes and the patterns
the skills must handle:

- a component cause on one robot of the first site, from day 0: about half
  its attempts fail, mostly by dropping the object; its grip readings in
  `health` are low;
- a configuration change at the second site on day 2 that hurts, logged in the
  change log on the same day and scope as a harmless software update (a
  confounded pair the logs cannot separate);
- a drift on another robot of the first site: its success falls day by day and
  its scene camera's calibration residual rises;
- a harmless calibration update on one robot of the second site (no effect);
- a task mix that differs by site, so that a site's raw rate is not its
  quality: the first site does more of the hard tasks;
- one experiment after the observed days, on the gripper robot: a control
  arm and an arm with the gripper replaced, its episodes marked with the
  experiment's id (`episodes.experiment_id`), as a later session sees them.

Telemetry is drawn from fixed seeds and only roughly shaped like a pick and
place (approach, close, lift, carry, release), enough for the skills' SQL to
find phases, drops and missed grasps. Nothing here touches the simulator.
The planted causes are module constants so that tests can check what the
skills' helpers find; they are written to the truth side only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from log_toy import POLICY, SOFTWARE, record

from fleetlab.contracts.env import REASON_PLACED, REASON_STEP_LIMIT, Camera
from fleetlab.contracts.faults import Drift, FaultSpec, Schedule, Target
from fleetlab.contracts.hashing import hash_bytes
from fleetlab.contracts.identity import EpisodeIdentity, neutral_id, seeds_for
from fleetlab.contracts.live import StepTelemetry
from fleetlab.contracts.log import EventKind
from fleetlab.contracts.model import (
    Change,
    ChangeAuthor,
    ComponentState,
    ConfigVersion,
    Fleet,
    Maintenance,
    Robot,
    Site,
)
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.log import EpisodeWriter, RunWriter, export_runs

WORLD_SEED = 11
DAYS = 5
EPISODES_PER_DAY = 8
MAX_STEPS = 64
FRAME_SIZE = 16
SITES = {"s1": ("r01", "r02", "r03"), "s2": ("r04", "r05", "r06")}
TASKS = (
    "put the bowl on the plate",
    "put the mug in the basket",
    "pick up the cheese and place it in the tray",
    "stack the black bowl on the white bowl",
)
TASK_CLASSES = ("put", "pick", "stack")
TASK_BASE = (0.97, 0.93, 0.85, 0.75)  # nominal success per task
TASK_WEIGHTS = {"s1": (0.1, 0.2, 0.3, 0.4), "s2": (0.4, 0.3, 0.2, 0.1)}

# The planted causes (truth side).
GRIPPER_ROBOT = "r02"
GRIPPER_FACTOR = 0.5
DELAY_SITE = "s2"
DELAY_DAY = 2
DELAY_FACTOR = 0.4
DRIFT_ROBOT = "r03"
DRIFT_PER_DAY = 0.12
BENIGN_ROBOT = "r05"
TRAIT = "pad_batch_code"
KINDS = ("toy_pad_slip", "toy_command_buffer", "toy_mount_creep")
EXPERIMENT_ID = "x0001"
ARM_EPISODES = 12  # per arm


def fleet() -> Fleet:
    def robot(robot_id: str, site_id: str) -> Robot:
        config = ConfigVersion(version=0)
        if robot_id == BENIGN_ROBOT:
            config = ConfigVersion(
                version=1, values={ConfigKey.camera_scene_calibration: "cal-b"}, since_day=1
            )
        return Robot(
            id=robot_id,
            site_id=site_id,
            components={c: ComponentState(component=c, installed_day=0) for c in Component},
            config=config,
            traits={TRAIT: 3 if robot_id == GRIPPER_ROBOT else 1},
        )

    sites = [
        Site(
            id=site_id,
            robots=[robot(r, site_id) for r in robots],
            conditions={ConditionKey.lighting: "standard", ConditionKey.object_batch: "batch-a"},
            config=(
                ConfigVersion(
                    version=1,
                    values={
                        ConfigKey.control_action_delay_steps: 4,
                        ConfigKey.software_version: "1.4.1",
                    },
                    since_day=DELAY_DAY,
                )
                if site_id == DELAY_SITE
                else ConfigVersion(version=0)
            ),
        )
        for site_id, robots in SITES.items()
    ]
    return Fleet(
        id="w02",
        sites=sites,
        config=ConfigVersion(
            version=1,
            values={
                ConfigKey.control_action_delay_steps: 0,
                ConfigKey.software_version: "1.4.0",
                ConfigKey.policy_sampling_steps: 10,
                ConfigKey.camera_scene_calibration: "cal-a",
            },
        ),
        change_log=[
            Change(
                day=1,
                scope=BENIGN_ROBOT,
                key=ConfigKey.camera_scene_calibration,
                old="cal-a",
                new="cal-b",
                author=ChangeAuthor.auto,
            ),
            Change(
                day=DELAY_DAY,
                scope=DELAY_SITE,
                key=ConfigKey.control_action_delay_steps,
                old=0,
                new=4,
                author=ChangeAuthor.ops,
            ),
            Change(
                day=DELAY_DAY,
                scope=DELAY_SITE,
                key=ConfigKey.software_version,
                old="1.4.0",
                new="1.4.1",
                author=ChangeAuthor.vendor,
            ),
        ],
        maintenance_log=[
            Maintenance(
                day=1,
                robot_id=GRIPPER_ROBOT,
                component=Component.gripper,
                verb=Verb.clean,
                note="pads cleaned",
            ),
            Maintenance(
                day=3, robot_id="r04", component=Component.arm, verb=Verb.service, note="routine"
            ),
        ],
    )


def site_of(robot_id: str) -> str:
    return next(site for site, robots in SITES.items() if robot_id in robots)


def planted_faults() -> list[FaultSpec]:
    return [
        FaultSpec(
            fault_id="f01",
            kind=KINDS[0],
            params={"friction_scale": 0.3},
            level=0.7,
            target=Target(scope=GRIPPER_ROBOT, component=Component.gripper),
            schedule=Schedule(onset_day=0),
        ),
        FaultSpec(
            fault_id="f02",
            kind=KINDS[1],
            params={"value": 4},
            level=1.0,
            target=Target(scope=DELAY_SITE, config_key=ConfigKey.control_action_delay_steps),
            schedule=Schedule(onset_day=DELAY_DAY),
            nominal=0,
        ),
        FaultSpec(
            fault_id="f03",
            kind=KINDS[2],
            params={"degrees_per_day": 1.5},
            level=0.2,
            target=Target(scope=DRIFT_ROBOT, component=Component.scene_camera),
            schedule=Schedule(onset_day=0, drift=Drift(shape="linear", rate=0.2)),
        ),
    ]


def success_probability(robot_id: str, day: int, task: int) -> float:
    p = TASK_BASE[task]
    if robot_id == GRIPPER_ROBOT:
        p *= GRIPPER_FACTOR
    if site_of(robot_id) == DELAY_SITE and day >= DELAY_DAY:
        p *= DELAY_FACTOR
    if robot_id == DRIFT_ROBOT:
        p *= max(0.0, 1.0 - DRIFT_PER_DAY * day)
    return p


@dataclass(frozen=True)
class Planned:
    identity: Job
    task: int
    success: bool
    failure: str | None  # "drop", "miss", "overshoot", "wander" or None


def plan() -> list[Planned]:
    """Every episode: identity, task, outcome and the way it fails, all from fixed seeds."""
    out: list[Planned] = []
    number = 0
    for day in range(DAYS):
        for site_id, robots in SITES.items():
            for robot_id in robots:
                for index in range(EPISODES_PER_DAY):
                    number += 1
                    rng = np.random.default_rng([WORLD_SEED, number])
                    task = int(rng.choice(len(TASKS), p=TASK_WEIGHTS[site_id]))
                    success = bool(rng.random() < success_probability(robot_id, day, task))
                    failure = None
                    if not success:
                        failure = _failure_kind(robot_id, day, rng)
                    scene, noise = seeds_for(WORLD_SEED, robot_id, day, index)
                    active = [
                        f
                        for f in planted_faults()
                        if f.target.covers(robot_id, site_id) and f.schedule.active_on(day)
                    ]
                    identity = EpisodeIdentity(
                        world_id="w02",
                        episode_id=neutral_id("episode", number),
                        robot_id=robot_id,
                        site_id=site_id,
                        day=day,
                        episode_index=index,
                        suite="libero_spatial",
                        task_index=task,
                        start_state=index,
                        scene_seed=scene,
                        noise_seed=noise,
                        policy=POLICY,
                        batch_size=4,
                        row=None,
                        image_size=FRAME_SIZE,
                        max_steps=MAX_STEPS,
                        faults=active,
                        software=SOFTWARE,
                    )
                    job = Job(identity=identity, record_video=index == 0)
                    out.append(Planned(job, task, success, failure))
    return out


def arm_plan() -> list[Planned]:
    """The experiment's episodes, after the observed days: arm 0 as it is, arm 1 repaired."""
    out: list[Planned] = []
    number = len(plan())
    for arm in (0, 1):
        for i in range(ARM_EPISODES):
            number += 1
            day, index = divmod(arm * ARM_EPISODES + i, EPISODES_PER_DAY)
            day += DAYS
            rng = np.random.default_rng([WORLD_SEED, number])
            task = int(rng.choice(len(TASKS), p=TASK_WEIGHTS["s1"]))
            p = TASK_BASE[task] * (GRIPPER_FACTOR if arm == 0 else 1.0)
            success = bool(rng.random() < p)
            failure = None if success else ("drop" if arm == 0 else "miss")
            scene, noise = seeds_for(WORLD_SEED, GRIPPER_ROBOT, day, index)
            active = [f for f in planted_faults() if f.target.covers(GRIPPER_ROBOT, "s1")]
            identity = EpisodeIdentity(
                world_id="w02",
                episode_id=neutral_id("episode", number),
                robot_id=GRIPPER_ROBOT,
                site_id="s1",
                day=day,
                episode_index=index,
                suite="libero_spatial",
                task_index=task,
                start_state=i,  # paired starts: the same start in both arms
                scene_seed=scene,
                noise_seed=noise,
                policy=POLICY,
                batch_size=4,
                row=None,
                image_size=FRAME_SIZE,
                max_steps=MAX_STEPS,
                faults=active if arm == 0 else [],
                software=SOFTWARE,
                experiment_id=EXPERIMENT_ID,
            )
            out.append(Planned(Job(identity=identity), task, success, failure))
    return out


def arm_episode_ids() -> dict[int, list[str]]:
    """The experiment's episodes by arm, as `experiment_results` reports them."""
    ids = [p.identity.identity.episode_id for p in arm_plan()]
    return {0: ids[:ARM_EPISODES], 1: ids[ARM_EPISODES:]}


def _failure_kind(robot_id: str, day: int, rng: np.random.Generator) -> str:
    if robot_id == GRIPPER_ROBOT and rng.random() < 0.8:
        return "drop"
    if robot_id == DRIFT_ROBOT and rng.random() < 0.8:
        return "miss"
    if site_of(robot_id) == DELAY_SITE and day >= DELAY_DAY and rng.random() < 0.8:
        return "overshoot"
    return str(rng.choice(["miss", "wander"]))


# Finger opening: open, closed on an object, closed on nothing. The shares of the open
# width (0.5 and 0.01) are where real grips settle (skills/README.md).
OPEN, HELD, SHUT = 0.04, 0.02, 0.0004
GRASP_STEP = 4  # the gripper is commanded closed and closes at once (a real one takes longer)
DROP_STEP = 40  # after the skills' settle window of 25 steps and its 10-step reading
TROUBLE_STEP = 30  # where an overshoot or a stall begins


def telemetry(planned: Planned) -> tuple[list[StepTelemetry], list[tuple[int, str, str]]]:
    """Steps and events of one episode, shaped by its outcome and failure.

    Events are the fleet's: grasp and release report the gripper's command, as
    the engine's event detection does; drop, collision, stall and near_miss
    report what was seen.
    """
    number = int(planned.identity.identity.episode_id[1:])
    rng = np.random.default_rng([WORLD_SEED, number, 1])
    n = 48 + int(rng.integers(0, 3)) if planned.success else MAX_STEPS
    events: list[tuple[int, str, str]] = []
    rows: list[StepTelemetry] = []
    failure = planned.failure
    for k in range(n + 1):
        z = 0.30 - 0.05 * min(k, GRASP_STEP)
        opening = OPEN
        if k >= GRASP_STEP:
            if failure == "miss":
                opening = SHUT
            else:
                opening = HELD
                z = 0.10 + 0.04 * min(k - GRASP_STEP, 4)
        if failure == "drop" and k >= DROP_STEP:
            opening = SHUT
        if planned.success and k >= n - 1:
            opening = OPEN
        speed = 0.3 + 0.05 * rng.normal()
        if failure == "overshoot" and k >= TROUBLE_STEP:
            speed = 0.9 * (-1) ** k
        if failure == "wander" and k >= TROUBLE_STEP:
            speed = 0.01
        command = None
        if k < n:
            command = np.zeros(7, dtype=np.float32)
            command[:3] = rng.normal(0.0, 0.1, 3) + speed
            command[6] = 1.0 if GRASP_STEP <= k < (n - 1 if planned.success else n) else -1.0
        rows.append(
            StepTelemetry(
                step=k,
                time_s=k * 0.05,
                joint_pos=rng.normal(0.0, 0.5, 7),
                joint_vel=np.full(7, speed) + rng.normal(0.0, 0.02, 7),
                gripper_qpos=np.array([opening, -opening]),
                eef_pos=np.array([0.1 * k, 0.0, z]),
                eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                commanded_action=(np.full(7, np.nan, np.float32) if command is None else command),
                chunk_index=k // 2,
                chunk_step=k % 2,
            )
        )
    events.append((GRASP_STEP, EventKind.grasp.value, "gripper commanded closed"))
    if failure == "drop":
        events.append((DROP_STEP, EventKind.drop.value, "the object left the gripper"))
    if failure == "overshoot":
        events.append((TROUBLE_STEP + 2, EventKind.collision.value, "contact with the table edge"))
    if failure == "wander":
        events.append((TROUBLE_STEP + 4, EventKind.stall.value, "no progress for a while"))
    if planned.success:
        events.append((n - 1, EventKind.release.value, "gripper commanded open"))
        if rng.random() < 0.15:
            events.append((20, EventKind.near_miss.value, "the object slipped and was caught"))
    return rows, events


def flat_frame(number: int, step: int, camera: Camera) -> npt.NDArray[np.uint8]:
    value = (number * 13 + step * 7 + (90 if camera is Camera.wrist else 0)) % 200 + 20
    return np.full((FRAME_SIZE, FRAME_SIZE, 3), value, dtype=np.uint8)


def health_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng([WORLD_SEED, 999])
    rows: list[dict[str, object]] = []
    for day in range(DAYS):
        for robots in SITES.values():
            for robot_id in robots:
                force = 40.0 + rng.normal(0.0, 1.5) - (12.0 if robot_id == GRIPPER_ROBOT else 0.0)
                residual = 0.5 + abs(rng.normal(0.0, 0.15))
                if robot_id == DRIFT_ROBOT:
                    residual += 0.45 * day
                readings = [
                    ("gripper", "grip_force", force),
                    ("gripper", "cycle_count", 1000.0 + 50.0 * day + rng.normal(0.0, 5.0)),
                    ("scene_camera", "calibration_residual", residual),
                    ("controller", "cpu_load", 0.4 + rng.normal(0.0, 0.05)),
                ]
                for component, metric, value in readings:
                    rows.append(
                        {
                            "day": day,
                            "robot_id": robot_id,
                            "component": component,
                            "metric": metric,
                            "value": round(float(value), 3),
                        }
                    )
    return rows


def write_fleet_run(run_dir: Path) -> Fleet:
    """The whole run, finalized."""
    model = fleet()
    with RunWriter(run_dir, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        run.write_model(model, TASK_CLASSES)
        run.write_table("health", health_rows())
        for planned in plan() + arm_plan():
            job = planned.identity
            number = int(job.identity.episode_id[1:])
            writer = EpisodeWriter(
                run_dir,
                job,
                task_text=TASKS[planned.task],
                episodes_per_robot_day=EPISODES_PER_DAY,
            )
            rows, events = telemetry(planned)
            for row in rows:
                frames = None
                if job.record_video and (row.step % 2 == 0 or row.step == rows[-1].step):
                    frames = {c: flat_frame(number, row.step, c) for c in Camera}
                writer.add_step(row, frames)
            for step, kind, detail in events:
                writer.add_event(step, kind, detail)
            last = rows[-1].step
            writer.set_verdict(
                planned.success,
                REASON_PLACED if planned.success else REASON_STEP_LIMIT,
                last - 1 if planned.success else None,
                last if planned.success else None,
                placed_step=last if planned.success else None,
            )
            writer.set_fixture_hash(hash_bytes(f"fixture/{job.key}".encode()))
            for row in rows:
                tag = f"{job.key}/{row.step}".encode()
                hashes = {c: hash_bytes(tag + c.value.encode()) for c in Camera}
                executed = None if row.step == 0 else np.zeros(7, dtype=np.float32)
                writer.add_chain(row.step, hash_bytes(tag), hashes, executed)
            writer.finish(outcome=planned.success)
        run.finalize()
    return model


def write_fleet_export(base: Path) -> tuple[Path, Path]:
    """The run and its export under `base`: (run directory, export directory)."""
    run_dir, export = base / "run", base / "export"
    model = write_fleet_run(run_dir)
    export_runs([run_dir], export, fleet=model, task_classes=TASK_CLASSES)
    return run_dir, export
