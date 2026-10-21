"""Toy data for the episode log tests, made from the contracts' types.

A fleet of two robots at two sites, ten episodes of a few steps each over three
days (two per robot and day, of a world's four), one robot with a camera fault
from day 0 and a configuration fault from day 1, video for every other
episode, two replays from the reset and one continuation from a snapshot. Nothing here touches the
simulator: the telemetry, frames and hashes are drawn from fixed seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import REASON_PLACED, REASON_STEP_LIMIT, Camera
from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.hashing import hash_bytes
from fleetlab.contracts.identity import (
    REPLAY_RULES,
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    neutral_id,
    seeds_for,
)
from fleetlab.contracts.live import Decision, DecisionRecord, StepTelemetry
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
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode, RunRecord, SnapshotRef
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.log import EpisodeWriter, RunWriter

WORLD_SEED = 7
SOFTWARE = SoftwareVersion(
    lock_hash="lock0", libero_patch_set="p1", mujoco_version="3.3.7", miopen_db_hash="db0"
)
POLICY = PolicySpec(
    name="toy_policy",
    revision="rev1",
    precision="float32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=2,
    sampling_steps=10,
    image_views=2,
)
TASKS = ("put the bowl on the plate", "open the drawer")
TASK_CLASSES = ("put", "open")
TRAIT = "mount_tolerance_mm"
CAMERA_FAULT = "toy_camera_wobble"
CONFIG_FAULT = "toy_command_lag"
FRAME_SIZE = 16
EPISODES_PER_DAY = 4  # the world's episodes per robot and day


def record() -> RunRecord:
    return RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[r.name for r in REPLAY_RULES])


def toy_fleet() -> Fleet:
    def robot(robot_id: str, site_id: str, tolerance: float) -> Robot:
        return Robot(
            id=robot_id,
            site_id=site_id,
            components={c: ComponentState(component=c, installed_day=0) for c in Component},
            traits={TRAIT: tolerance},
        )

    return Fleet(
        id="w01",
        sites=[
            Site(id="s1", robots=[robot("r01", "s1", 0.1)]),
            Site(
                id="s2",
                robots=[
                    robot("r02", "s2", 0.9).model_copy(
                        update={
                            "config": ConfigVersion(
                                version=1,
                                values={ConfigKey.control_action_delay_steps: 3},
                                since_day=1,
                            )
                        }
                    )
                ],
            ),
        ],
        config=ConfigVersion(version=1, values={ConfigKey.control_action_delay_steps: 0}),
        change_log=[
            Change(
                day=1,
                scope="r02",
                key=ConfigKey.control_action_delay_steps,
                old=0,
                new=3,
                author=ChangeAuthor.ops,
            )
        ],
        maintenance_log=[
            Maintenance(
                day=1, robot_id="r01", component=Component.gripper, verb=Verb.clean, note="pads"
            )
        ],
    )


def faults() -> list[FaultSpec]:
    return [
        FaultSpec(
            fault_id="f01",
            kind=CAMERA_FAULT,
            params={"degrees": 6.0},
            level=0.5,
            target=Target(scope="r02", component=Component.scene_camera),
            schedule=Schedule(onset_day=0),
        ),
        FaultSpec(
            fault_id="f02",
            kind=CONFIG_FAULT,
            params={"value": 3},
            level=1.0,
            target=Target(scope="r02", config_key=ConfigKey.control_action_delay_steps),
            schedule=Schedule(onset_day=1),
            nominal=0,
        ),
    ]


def identities() -> list[EpisodeIdentity]:
    out = []
    for number in range(1, 11):
        robot_id = "r01" if number % 2 else "r02"
        site_id = "s1" if robot_id == "r01" else "s2"
        day = (number - 1) // 4
        index = (number - 1) // 2 % 2
        scene, noise = seeds_for(WORLD_SEED, robot_id, day, index)
        active = [f for f in faults() if robot_id == "r02" and f.schedule.active_on(day)]
        out.append(
            EpisodeIdentity(
                world_id="w01",
                episode_id=neutral_id("episode", number),
                robot_id=robot_id,
                site_id=site_id,
                day=day,
                episode_index=index,
                suite="libero_spatial",
                task_index=number % 2,
                start_state=number,
                scene_seed=scene,
                noise_seed=noise,
                policy=POLICY,
                batch_size=4,
                row=None,
                image_size=FRAME_SIZE,
                max_steps=50,
                faults=active,
                software=SOFTWARE,
            )
        )
    return out


def run_jobs() -> list[Job]:
    return [Job(identity=i, record_video=i.episode_id[-1] in "13579") for i in identities()]


def replay_jobs() -> list[Job]:
    by_id = {i.episode_id: i for i in identities()}
    return [
        Job(identity=by_id["e000002"], mode=JobMode.replay, remove=["f01"]),
        Job(identity=by_id["e000006"], mode=JobMode.replay, remove=["f01", "f02"]),
    ]


def continue_job() -> Job:
    """A hand-over continuation of e000004 from a snapshot at step 2."""
    identity = {i.episode_id: i for i in identities()}["e000004"]
    snapshot = SnapshotRef(episode_id="e000004", step=2, path="truth/snapshots/e000004.2.bin")
    return Job(identity=identity, mode=JobMode.continue_, from_snapshot=snapshot)


def other_jobs() -> list[Job]:
    """Every job that is not a run: the replays and the continuation."""
    return [*replay_jobs(), continue_job()]


def all_jobs() -> list[Job]:
    return run_jobs() + other_jobs()


def frame_value(number: int, step: int, camera: Camera) -> int:
    """The grey level of a toy frame: flat images survive H.264 nearly unchanged."""
    return (number * 20 + step * 7 + (100 if camera is Camera.wrist else 0)) % 200 + 20


def flat_frame(number: int, step: int, camera: Camera) -> npt.NDArray[np.uint8]:
    value = frame_value(number, step, camera)
    return np.full((FRAME_SIZE, FRAME_SIZE, 3), value, dtype=np.uint8)


@dataclass(frozen=True)
class Episode:
    """What a toy episode consists of, so that tests can compare what was read back."""

    steps: list[StepTelemetry]
    frames: list[dict[Camera, npt.NDArray[np.uint8]] | None]
    chain: list[tuple[int, str, dict[Camera, str], npt.NDArray[np.float32] | None]]
    success: bool


def episode_data(job: Job) -> Episode:
    identity = job.identity
    number = int(identity.episode_id[1:])
    rng = np.random.default_rng(number * 100 + (0 if job.mode is JobMode.run else 1))
    n = 3 + number % 3  # actions; one more state than actions
    steps: list[StepTelemetry] = []
    frames: list[dict[Camera, npt.NDArray[np.uint8]] | None] = []
    chain: list[tuple[int, str, dict[Camera, str], npt.NDArray[np.float32] | None]] = []
    for k in range(n + 1):
        command = None if k == n else rng.normal(size=7).astype(np.float32)
        steps.append(
            StepTelemetry(
                step=k,
                time_s=k * 0.05,
                joint_pos=rng.normal(size=7),
                joint_vel=rng.normal(size=7),
                gripper_qpos=rng.uniform(0, 0.04, size=2),
                eef_pos=rng.normal(size=3),
                eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                commanded_action=np.full(7, np.nan, np.float32) if command is None else command,
                chunk_index=k // 2,
                chunk_step=k % 2,
            )
        )
        if job.record_video and (k % 2 == 0 or k == n):
            frames.append({c: flat_frame(number, k, c) for c in Camera})
        else:
            frames.append(None)
        tag = f"{job.key}/{k}".encode()
        hashes = {c: hash_bytes(tag + c.value.encode()) for c in Camera}
        first = 0 if job.from_snapshot is None else job.from_snapshot.step
        executed = None if k == first else rng.normal(size=7).astype(np.float32)
        if k >= first:
            chain.append((k, hash_bytes(tag), hashes, executed))
    return Episode(steps, frames, chain, success=number % 3 != 0)


def fixture_hash_of(job: Job) -> str:
    """The toy's fixture hash: the same for every job of an episode, as a real one is."""
    return hash_bytes(f"fixture/{job.identity.episode_id}".encode())


def write_job(run_dir: Path, job: Job, *, finish: bool = True) -> EpisodeWriter:
    """Write one toy job through an episode writer, as a worker would."""
    data = episode_data(job)
    identity = job.identity
    writer = EpisodeWriter(
        run_dir,
        job,
        task_text=TASKS[identity.task_index],
        episodes_per_robot_day=EPISODES_PER_DAY,
    )
    if job.mode is JobMode.run:
        for telemetry, frame in zip(data.steps, data.frames, strict=True):
            writer.add_step(telemetry, frame)
        if int(identity.episode_id[1:]) % 4 == 1:
            writer.add_event(1, EventKind.grasp, "closed on the bowl")
            writer.add_event(2, EventKind.release, "opened above the plate")
        if identity.episode_id == "e000004":
            writer.add_decision(
                DecisionRecord(
                    step=2,
                    decision=Decision.continue_,
                    hold_steps=1,
                    decision_ms=812.5,
                    model_ms=640.0,
                )
            )
            writer.set_effective_policy(POLICY.model_copy(update={"sampling_steps": 5}))
        last = data.steps[-1].step
        writer.set_verdict(
            data.success,
            REASON_PLACED if data.success else REASON_STEP_LIMIT,
            2,
            last if data.success else None,
            placed_step=1 if data.success else None,
        )
    writer.set_fixture_hash(fixture_hash_of(job))
    for step, state_hash, hashes, executed in data.chain:
        writer.add_chain(step, state_hash, hashes, executed)
    if finish:
        if job.mode is JobMode.run:
            writer.finish(outcome=data.success)
        else:
            success = not data.success
            writer.finish(success=success, placed_step=3 if success else None, libero_step=None)
    return writer


def write_toy_run(run_dir: Path, jobs: list[Job] | None = None, *, finalize: bool = True) -> Fleet:
    """A complete toy run: model, every job, finalized."""
    fleet = toy_fleet()
    with RunWriter(run_dir, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        run.write_model(fleet, TASK_CLASSES)
        run.write_table(
            "health",
            [
                {
                    "day": d,
                    "robot_id": r,
                    "component": "gripper",
                    "metric": "grip_force_n",
                    "value": 40.0 + d,
                }
                for d in range(5)
                for r in ("r01", "r02")
            ],
        )
        for job in run.pending(jobs if jobs is not None else all_jobs()):
            write_job(run_dir, job)
        if finalize:
            run.finalize()
    return fleet
