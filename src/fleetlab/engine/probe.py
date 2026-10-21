"""The engine's probe world, and one experiment queue through the runtime on it.

    tools/heavy-run uv run --locked python -m fleetlab.engine.probe DIR [--per-arm 10]

A three-robot world on `libero_spatial` with X-VLA (fp16_res32): r01 healthy,
r02 with its scene camera turned 6 degrees, r03 with its actions 17 steps late
(`fleetlab.faults`' types), all at site s1; day 0 is the observed window, day
1 the experiment window. The script runs the observed window (8 episodes per
robot), exports it, and then an analysis whose agent is a script
(`experiments.scripted_analysis`): session 1 requests one experiment, the
control arm against `recalibrate scene_camera` on r02, `--per-arm` episodes
each with paired starts; the engine answers the queue; the runtime's
`run_analysis(check_arms=True)` checks the reply; session 2 reads the results.
Prints what the second session saw and writes it to DIR/probe_result.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.engine.probe")
    parser.add_argument("dir", type=Path)
    parser.add_argument("--per-arm", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--per-robot", type=int, default=8)
    args = parser.parse_args()
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.contracts.actions import Action
    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.identity import Task
    from fleetlab.contracts.model import ComponentState, ConfigVersion, Fleet, Robot, Site
    from fleetlab.contracts.rollout import Job
    from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
    from fleetlab.contracts.world import DayRange, Windows
    from fleetlab.engine import manifests, miopen, software
    from fleetlab.engine.experiments import QueueEngine, WorldTruth, scripted_analysis
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.log import export_runs
    from fleetlab.policies.xvla import nominal_spec

    base = args.dir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("ROBOTS_AGENT_ROOT", str(base / "agent-root"))  # never the real one
    miopen.activate()
    soft = software.current(miopen.require(4, "fp16_res32"))
    policy = nominal_spec("fp16_res32")

    def robot(robot_id: str) -> Robot:
        return Robot(
            id=robot_id,
            site_id="s1",
            components={c: ComponentState(component=c, installed_day=0) for c in Component},
        )

    fleet_model = Fleet(
        id="w01",
        sites=[Site(id="s1", robots=[robot("r01"), robot("r02"), robot("r03")])],
        config=ConfigVersion(version=1, values={ConfigKey.control_action_delay_steps: 0}),
    )
    faults = [
        manifests.camera_fault(1, "r02", 6.0, standin=False),
        manifests.delay_fault(2, "r03", 17, standin=False),
    ]
    tasks = [Task("libero_spatial", t) for t in range(10)]
    draft = WorldTruth(
        world_id="w01",
        world_seed=17,
        fleet=fleet_model,
        windows=Windows(
            observed=DayRange(first=0, last=0),
            experiment=DayRange(first=1, last=1),
            scoring=DayRange(first=2, last=2),
        ),
        fault_schedule=faults,
        episodes_per_robot_day=args.per_robot,
        tasks=tasks,
        task_classes=["pick and place"],
        policy=policy,
        base_runs=[],
        first_free_episode=1,
    )
    # The observed window, as run.
    jobs, n = [], 1
    for number, robot_id in enumerate(("r01", "r02", "r03"), start=1):
        for index in range(args.per_robot):
            ident = manifests.identity(
                world_seed=17, episode_number=n, robot=number, site=1, day=0,
                episode_index=index, suite="libero_spatial", task_index=index % 10,
                start_state=index, policy=policy, batch_size=4, software=soft,
                faults=draft.faults_on(robot_id, 0),
            )  # fmt: skip
            jobs.append(Job(identity=ident, record_video=True))
            n += 1
    config = RunConfig(
        out=str(base / "runs" / "observed"),
        workers=args.workers,
        batch_size=4,
        precision="fp16_res32",
        episodes_per_robot_day=args.per_robot,
    )
    run(config, jobs)
    export = base / "export"
    if not (export / "log").is_dir():
        export_runs([config.out], export, fleet=fleet_model, task_classes=draft.task_classes)
    world = draft.model_copy(update={"base_runs": [config.out], "first_free_episode": n})
    (base / "world_truth.json").write_text(world.model_dump_json(indent=1))

    recalibrate = Action(
        verb=Verb.recalibrate, target=Target(scope="r02", component=Component.scene_camera)
    )
    request = {
        "arms": [{"actions": []}, {"actions": [recalibrate.model_dump(mode="json")]}],
        "robots": ["r02"],
        "episodes_per_arm": args.per_arm,
        "paired_starts": True,
        "power_note": "a probe of the engine's queue, not a power calculation",
    }
    engine = QueueEngine(world, base, config)
    record, seen = scripted_analysis(
        base / "analyses", world, export, engine, request, project_tool_env(base / "toolenv")
    )
    result = {
        "states": [s.state for s in record.sessions],
        "queue_file": record.sessions[0].queue_file,
        "seen": seen,
    }
    (base / "probe_result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
