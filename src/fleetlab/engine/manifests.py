"""Manifests: lists of jobs, as JSON lines, and the ones the engine's checks run.

A manifest line is a `contracts.rollout.Job` in JSON. The world generator
(stage 3) writes the worlds' manifests; this module builds the engine's own:

- `reference`: one suite, 10 tasks by 20 start states, no fault: the sample
  the reference evaluation measured (`spikes/baseline`, 192/200 and 197/200).
- `handover`: healthy episodes the timeout supervisor hands to the in-worker
  teleoperator (the takeover path in the fleet layout).
- `mix200`: the faulted mix of spikes/xvla-speed: robot r01 healthy (40),
  r02 with the scene camera turned 6 degrees (80), r03 with its actions
  executed 17 steps late (80), libero_spatial, start states 0 to 7. The two
  faults are the engine's stand-ins (`engine.standin_faults`) until the
  faults agent's types land.

Identities follow the contracts: neutral ids, seeds from `seeds_for(world
seed, robot, day, episode index)`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component, ConfigKey

MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def write(path: Path, jobs: Iterable[Job]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [job.model_dump_json() + "\n" for job in jobs]
    path.write_text("".join(lines))
    return len(lines)


def read(path: Path) -> list[Job]:
    return [Job.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def identity(
    *,
    world_seed: int,
    episode_number: int,
    robot: int,
    site: int,
    day: int,
    episode_index: int,
    suite: str,
    task_index: int,
    start_state: int,
    policy: PolicySpec,
    batch_size: int,
    software: SoftwareVersion,
    image_size: int = 360,
    max_steps: int | None = None,
    faults: Sequence[FaultSpec] = (),
    world: int = 1,
) -> EpisodeIdentity:
    robot_id = neutral_id(IdKind.robot, robot)
    scene, noise = seeds_for(world_seed, robot_id, day, episode_index)
    return EpisodeIdentity(
        world_id=neutral_id(IdKind.world, world),
        episode_id=neutral_id(IdKind.episode, episode_number),
        robot_id=robot_id,
        site_id=neutral_id(IdKind.site, site),
        day=day,
        episode_index=episode_index,
        suite=suite,
        task_index=task_index,
        start_state=start_state,
        scene_seed=scene,
        noise_seed=noise,
        policy=policy,
        batch_size=batch_size,
        row=None,
        image_size=image_size,
        max_steps=max_steps if max_steps is not None else MAX_STEPS.get(suite, 280),
        faults=list(faults),
        software=software,
    )


def camera_fault(
    fault_number: int, robot_id: str, degrees: float, *, standin: bool = True
) -> FaultSpec:
    """The scene camera panned: the stand-in, or fleetlab.faults' camera_pose_offset."""
    return FaultSpec(
        fault_id=neutral_id(IdKind.fault, fault_number),
        kind="standin_camera_pan" if standin else "camera_pose_offset",
        params={"degrees": degrees} if standin else {},
        level=degrees,
        target=Target(scope=robot_id, component=Component.scene_camera),
        schedule=Schedule(onset_day=0),
    )


def delay_fault(fault_number: int, robot_id: str, steps: int, *, standin: bool = True) -> FaultSpec:
    """Actions executed late: the stand-in, or fleetlab.faults' action_latency."""
    return FaultSpec(
        fault_id=neutral_id(IdKind.fault, fault_number),
        kind="standin_action_delay" if standin else "action_latency",
        params={"steps": steps} if standin else {},
        level=float(steps),
        target=Target(scope=robot_id, config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=0),
        nominal=0,
    )


def reference(
    suite: str,
    policy: PolicySpec,
    batch_size: int,
    software: SoftwareVersion,
    *,
    starts: int = 20,
    world_seed: int = 7,
    first_episode: int = 1,
) -> list[Job]:
    """10 tasks by `starts` start states of one suite, robot r01, no fault."""
    jobs = []
    n = first_episode
    for task in range(10):
        for start in range(starts):
            ident = identity(
                world_seed=world_seed,
                episode_number=n,
                robot=1,
                site=1,
                day=0,
                episode_index=n,
                suite=suite,
                task_index=task,
                start_state=start,
                policy=policy,
                batch_size=batch_size,
                software=software,
            )
            jobs.append(Job(identity=ident))
            n += 1
    return jobs


def mix200(
    policy: PolicySpec,
    batch_size: int,
    software: SoftwareVersion,
    *,
    record_video: bool = False,
    world_seed: int = 11,
    starts: int = 8,
    tasks: int = 10,
    standin: bool = True,
) -> list[Job]:
    """The faulted mix: r01 healthy for start states 0 to 3, r02 camera 6 degrees, r03 delay 17."""
    cam = camera_fault(1, "r02", 6.0, standin=standin)
    delay = delay_fault(2, "r03", 17, standin=standin)
    jobs = []
    n = 1
    index = {1: 0, 2: 0, 3: 0}  # each robot's episodes of the day, in order
    for task in range(tasks):
        for start in range(starts):
            for robot, faults in ((1, []), (2, [cam]), (3, [delay])):
                if robot == 1 and start >= starts // 2:
                    continue
                ident = identity(
                    world_seed=world_seed,
                    episode_number=n,
                    robot=robot,
                    site=1,
                    day=0,
                    episode_index=index[robot],
                    suite="libero_spatial",
                    task_index=task,
                    start_state=start,
                    policy=policy,
                    batch_size=batch_size,
                    software=software,
                    faults=faults,
                )
                jobs.append(Job(identity=ident, record_video=record_video))
                n += 1
                index[robot] += 1
    return jobs


def handover(
    policy: PolicySpec,
    batch_size: int,
    software: SoftwareVersion,
    *,
    episodes: int = 8,
    after: int = 40,
    world_seed: int = 13,
) -> list[Job]:
    """Healthy libero_spatial episodes that the timeout supervisor hands over at step `after`.

    Each spawns a "continue" job for the in-worker teleoperator from its snapshot.
    """
    from fleetlab.contracts.rollout import SupervisorSpec

    supervisor = SupervisorSpec(
        name="timeout", params={"after": after, "decision": "hand_over", "decision_ms": 500}
    )
    jobs = []
    for n in range(episodes):
        ident = identity(
            world_seed=world_seed,
            episode_number=n + 1,
            robot=4,
            site=1,
            day=0,
            episode_index=n,
            suite="libero_spatial",
            task_index=n % 10,
            start_state=n // 10,
            policy=policy,
            batch_size=batch_size,
            software=software,
        )
        jobs.append(Job(identity=ident, supervisor=supervisor))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.engine.manifests")
    parser.add_argument("kind", choices=["reference", "mix200", "handover"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--precision", default="fp16_res32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--starts", type=int, default=None)
    parser.add_argument("--miopen", default="pinned")
    parser.add_argument(
        "--faults",
        choices=["standin", "fleetlab"],
        default="standin",
        help="mix200's two faults: the engine's stand-ins or fleetlab.faults' types",
    )
    args = parser.parse_args()
    from fleetlab.engine import miopen, software
    from fleetlab.policies.xvla import nominal_spec

    db = miopen.require(args.batch_size, args.precision, args.miopen)
    soft = software.current(db)
    spec = nominal_spec(args.precision)
    if args.kind == "reference":
        jobs = reference(args.suite, spec, args.batch_size, soft, starts=args.starts or 20)
    elif args.kind == "handover":
        jobs = handover(spec, args.batch_size, soft, episodes=args.starts or 8)
    else:
        jobs = mix200(
            spec,
            args.batch_size,
            soft,
            record_video=args.video,
            starts=args.starts or 8,
            standin=args.faults == "standin",
        )
    print(f"{args.out}: {write(args.out, jobs)} jobs")


if __name__ == "__main__":
    main()
