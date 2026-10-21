"""A toy stored world for the loop's CPU tests (and the driver's): the engine's toy cell,
policy and faults, written to disk the way a world and its observed run are stored.

    world, setup = write_toy_world(root)     # root/world/..., root/runs/w01-observed
    layout = layout_for_world(world.spec)    # rounds on days 0-1 | 2, 3 | 4; evaluation 5

Two sites with two robots each (`r01`, `r02` at `s1`; `r03`, `r04` at `s2`), four days
(observed 0-1, experiment 2, scoring 3), four attempts per robot and day, made by the
miniature fleet's version-0 slot rule (so the loop mints days past the horizon the same
way). Faults: `f01` r02's toy scene camera shifted by 3 pixels (fixed by `recalibrate`),
`f02` r03's commands 3 steps late (`control.action_delay_steps` 0 to 3 at `r03`, its own
change on day 0), `f03` the sampling steps at `s1` changed from 2 to 5 on day 1, which the
toy policy does not read (a decoy). The fault registry is the toy oracle's
(`fleetlab.oracle.toy`), which simulates `config_change` for the delay key. No health
readings: the toy fleet has none. Nothing here imports torch, mujoco or LIBERO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import EpisodeIdentity, Task, seeds_for
from fleetlab.contracts.ids import IdKind, neutral_id
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
from fleetlab.contracts.rollout import Job, RunRecord
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec, seal_world
from fleetlab.engine import toy
from fleetlab.loop.operate import EngineSetup
from fleetlab.loop.world import LoopWorld

__all__ = [
    "REGISTRY",
    "WORLD_ID",
    "paper_spec",
    "toy_setup",
    "write_paper_world",
    "write_toy_world",
]

WORLD_ID = "w01"
SEED = 3
PER_DAY = 4
DAYS = 4
REGISTRY = "fleetlab.oracle.toy:TOY_ORACLE_FAULTS"
TASKS = [Task(toy.TOY_SUITE, 0), Task(toy.TOY_SUITE, 1)]
TASK_CLASSES = ["carry the cube", "carry the cube further"]
IMAGE = 32
MAX_STEPS = 60
DELAY = ConfigKey.control_action_delay_steps
SAMPLING = ConfigKey.policy_sampling_steps


def faults() -> list[FaultSpec]:
    return [
        FaultSpec(
            fault_id="f01",
            kind="toy_camera_shift",
            params={"pixels": 3},
            level=1.0,
            target=Target(scope="r02", component=Component.scene_camera),
            schedule=Schedule(onset_day=0),
        ),
        FaultSpec(
            fault_id="f02",
            kind="toy_delay",
            params={"steps": 3},
            level=1.0,
            target=Target(scope="r03", config_key=DELAY),
            schedule=Schedule(onset_day=0),
            nominal=0,
        ),
        FaultSpec(
            fault_id="f03",
            kind="toy_knob",
            params={"steps": 5},
            level=1.0,
            target=Target(scope="s1", config_key=SAMPLING),
            schedule=Schedule(onset_day=1),
            nominal=2,
        ),
    ]


def spec() -> WorldSpec:
    return WorldSpec(
        world_id=WORLD_ID,
        seed=SEED,
        size="small",
        sites=2,
        robots_per_site=2,
        days=DAYS,
        windows=Windows(
            observed=DayRange(first=0, last=1),
            experiment=DayRange(first=2, last=2),
            scoring=DayRange(first=3, last=3),
        ),
        fault_schedule=faults(),
        patterns=Patterns(decoys=["f03"]),
        policy=toy.TOY_SPEC,
        tasks=TASKS,
        episodes_per_robot_day=PER_DAY,
    )


def _robots() -> list[tuple[str, str]]:
    return [("r01", "s1"), ("r02", "s1"), ("r03", "s2"), ("r04", "s2")]


CHANGES = [
    Change(day=0, scope="r03", key=DELAY, old=0, new=3, author=ChangeAuthor.vendor),
    Change(day=1, scope="s1", key=SAMPLING, old=2, new=5, author=ChangeAuthor.ops),
]
UPKEEP = [
    Maintenance(day=1, robot_id="r04", component=Component.gripper, verb=Verb.clean, note="pads"),
    Maintenance(day=3, robot_id="r01", component=Component.arm, verb=Verb.service, note="due"),
]


def fleet(until_day: int) -> Fleet:
    """The fleet's records as they stand at the end of `until_day`."""
    changes = [c for c in CHANGES if c.day <= until_day]
    own: dict[str, dict[ConfigKey, Any]] = {}
    for c in changes:
        own.setdefault(c.scope, {})[c.key] = c.new

    def config(scope: str) -> ConfigVersion:
        values = own.get(scope, {})
        days = [c.day for c in changes if c.scope == scope]
        return ConfigVersion(version=len(days), values=values, since_day=max(days, default=0))

    def robot(robot_id: str, site_id: str) -> Robot:
        serviced = [
            m.day
            for m in UPKEEP
            if m.robot_id == robot_id and m.day <= until_day and m.verb is Verb.service
        ]
        return Robot(
            id=robot_id,
            site_id=site_id,
            components={
                c: ComponentState(
                    component=c,
                    installed_day=-20,
                    last_service_day=max(serviced) if serviced and c is Component.arm else -5,
                )
                for c in Component
            },
            config=config(robot_id),
        )

    conditions: dict[ConditionKey, Any] = {
        ConditionKey.lighting: 0.0,
        ConditionKey.placement: [0.0, 0.0],
        ConditionKey.object_batch: [0.0, 0.0],
        ConditionKey.clutter: 0.0,
    }
    sites = [
        Site(
            id=site_id,
            robots=[robot(r, s) for r, s in _robots() if s == site_id],
            conditions=dict(conditions),
            config=config(site_id),
        )
        for site_id in ("s1", "s2")
    ]
    fleet_values: dict[ConfigKey, Any] = {
        SAMPLING: 2,
        ConfigKey.policy_actions_executed_per_chunk: 5,
        ConfigKey.policy_image_views: 2,
        ConfigKey.policy_precision: "fp32",
        ConfigKey.control_mode: "absolute",
        DELAY: 0,
        ConfigKey.camera_scene_calibration: [0.0, 0.0, 0.0],
        ConfigKey.camera_wrist_calibration: [0.0, 0.0, 0.0],
        ConfigKey.software_version: "1.0",
    }
    return Fleet(
        id=WORLD_ID,
        sites=sites,
        config=ConfigVersion(version=1, values=fleet_values),
        change_log=changes,
        maintenance_log=[m for m in UPKEEP if m.day <= until_day],
    )


def jobs(software: Any) -> list[Job]:
    """The world's run jobs, every day, by the version-0 slot rule."""
    from fleetlab.contracts.faults import active_faults
    from fleetlab.worlds import generate

    world_spec = spec()
    slots = [
        (day, site_index, robot_id, index)
        for day in range(DAYS)
        for site_index, site_id in enumerate(("s1", "s2"))
        for robot_id, s in _robots()
        if s == site_id
        for index in range(PER_DAY)
    ]
    numbers = generate._episode_numbers(world_spec, len(slots))
    out = []
    for (day, site_index, robot_id, index), number in zip(slots, numbers, strict=True):
        task = generate.draw_task(world_spec, site_index, robot_id, day, index)
        start = int(
            generate.rng(SEED, "start", robot_id, day, index).integers(generate.START_STATES)
        )
        scene, noise = seeds_for(SEED, robot_id, day, index)
        identity = EpisodeIdentity(
            world_id=WORLD_ID,
            episode_id=neutral_id(IdKind.episode, number),
            robot_id=robot_id,
            site_id=neutral_id(IdKind.site, site_index + 1),
            day=day,
            episode_index=index,
            suite=task.suite,
            task_index=task.task_index,
            start_state=start,
            scene_seed=scene,
            noise_seed=noise,
            policy=toy.TOY_SPEC,
            batch_size=4,
            row=None,
            image_size=IMAGE,
            max_steps=MAX_STEPS,
            faults=active_faults(faults(), fleet(day), robot_id, day),
            software=software,
        )
        out.append(Job(identity=identity, record_video=False))
    return out


def toy_setup(workers: int = 2) -> EngineSetup:
    """The engine settings of the loop's toy runs (CPU, the toy cell, the toy registry)."""
    from fleetlab.engine.fleet import RunConfig
    from fleetlab.oracle.toy import engine_overrides

    config = RunConfig(
        out="",
        batch_size=4,
        rerender_light=False,
        **engine_overrides(workers),
    )
    return EngineSetup(config=config, contacts=False, events=False, video=False)


def write_toy_world(root: Path | str, workers: int = 2) -> tuple[LoopWorld, EngineSetup]:
    """Write the toy world under `root` and run its observed window; resumable."""
    from fleetlab.engine import software as soft
    from fleetlab.engine.fleet import run
    from fleetlab.log import RunWriter

    root = Path(root)
    files = root / "world"
    files.mkdir(parents=True, exist_ok=True)
    software = soft.current("none")
    manifest = jobs(software)
    world_spec = spec()
    observed_run = root / "runs" / f"{WORLD_ID}-observed"
    (files / "spec.json").write_text(world_spec.model_dump_json(indent=1) + "\n")
    (files / "model.json").write_text(fleet(DAYS - 1).model_dump_json(indent=1) + "\n")
    (files / "description.json").write_text("{}\n")
    (files / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in manifest))
    seal_world(files)
    truth = {
        "world_id": WORLD_ID,
        "world_seed": SEED,
        "fleet": fleet(world_spec.windows.observed.last).model_dump(mode="json"),
        "windows": world_spec.windows.model_dump(mode="json"),
        "fault_schedule": [f.model_dump(mode="json") for f in faults()],
        "fault_levels": {},
        "episodes_per_robot_day": PER_DAY,
        "tasks": [list(t) for t in TASKS],
        "task_classes": TASK_CLASSES,
        "start_states": 50,
        "policy": toy.TOY_SPEC.model_dump(mode="json"),
        "image_size": IMAGE,
        "max_steps": {toy.TOY_SUITE: MAX_STEPS},
        "base_runs": [str(observed_run)],
        "first_free_episode": len(manifest) + 1,
    }
    (files / "world_truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    (files / "task_texts.json").write_text(
        json.dumps(
            [
                {"suite": t.suite, "task_index": t.task_index, "text": text}
                for t, text in zip(TASKS, TASK_CLASSES, strict=True)
            ]
        )
        + "\n"
    )
    (files / "generator.json").write_text(json.dumps({"generator": "fleetlab.worlds v0"}) + "\n")
    setup = toy_setup(workers)
    observed = [j for j in manifest if j.identity.day in world_spec.windows.observed]
    if not (observed_run / "log" / "description.json").is_file():
        run(setup.for_run(observed_run, PER_DAY), observed)
        record = RunRecord.model_validate_json((observed_run / "run.json").read_text())
        with RunWriter(observed_run, record, episodes_per_robot_day=PER_DAY) as writer:
            writer.write_model(fleet(world_spec.windows.observed.last), TASK_CLASSES)
            writer.write_table("health", [])
            writer.finalize()
    return LoopWorld.load(root, registry=REGISTRY), setup


def paper_spec(camera_robot: str = "r01") -> WorldSpec:
    """The miniature fleet's spec over ten days (observed 0-5, experiment 6-7, scoring
    8-9), its camera fault on `camera_robot`, two attempts per robot and day."""
    from fleetlab.worlds.mini import mini_spec

    base = mini_spec().model_dump(mode="json")
    for fault in base["fault_schedule"]:
        if fault["kind"] == "camera_pose_offset":
            fault["target"]["scope"] = camera_robot
    base.update(
        world_id="w11",
        days=10,
        episodes_per_robot_day=2,
        windows={
            "observed": {"first": 0, "last": 5},
            "experiment": {"first": 6, "last": 7},
            "scoring": {"first": 8, "last": 9},
        },
    )
    return WorldSpec.model_validate(base)


def write_paper_world(root: Path | str, camera_robot: str = "r01") -> LoopWorld:
    """A world with the real fault kinds and health readings, written by the miniature
    fleet's version-0 generator, without any run: for the parts of the loop that read a
    world but simulate nothing (records, health, identities, the materialized state)."""
    from fleetlab.engine import software as soft
    from fleetlab.worlds import generate

    root = Path(root)
    spec = paper_spec(camera_robot)
    texts = {task: f"task {n}" for n, task in enumerate(spec.tasks)}
    world = generate.generate(spec, soft.current("none"), texts)
    generate.write_world(world, root / "world")
    return LoopWorld.load(root)
