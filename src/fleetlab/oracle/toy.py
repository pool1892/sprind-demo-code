"""A toy world for the oracle's tests: the engine's toy cell (`fleetlab.engine.toy`), toy
faults with a decoy, and a toy `config_change` so that an action that adds a change can
be replayed fresh. Nothing here imports torch, mujoco or LIBERO.

One site `s1` with three robots, three days (observed 0, experiment 1, scoring 2), four
episodes per robot and day: 36 episodes, 12 of them scored. Faults: `f01` the toy scene
camera of r02 shifted by 4 pixels (a cell hook), `f02` commands of r03 executed 3 steps
late (`control.action_delay_steps` 0 to 3, an action hook), and the decoy `f03`, the
sampling steps at s1 changed from 2 to 5 (`policy.sampling_steps`, which the toy policy
does not read, so it is flat by construction). The world files are written as the world
generator writes them (`world/`: spec, model, description, manifest, hash, fault table).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import CONFIG_CHANGE_KIND
from fleetlab.contracts.faults import (
    Direction,
    FaultSpec,
    FaultState,
    FaultTypeBase,
    Schedule,
    Target,
)
from fleetlab.contracts.identity import EpisodeIdentity, Task
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.model import ComponentState, ConfigVersion, Fleet, Robot, Site
from fleetlab.contracts.policy import Action7
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, FaultClass
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec, seal_world
from fleetlab.engine import manifests, toy
from fleetlab.faults import FaultRegistry
from fleetlab.worlds.truth import FaultTable, FaultTruth, Role, WindowName

WORLD_SEED = 3
EPISODES_PER_ROBOT_DAY = 4
DAYS = 3
ROBOTS = 3
REGISTRY = "fleetlab.oracle.toy:TOY_ORACLE_FAULTS"
MAX_STEPS = 60
IMAGE = 32


class _Value(BaseModel):
    value: int


class ToyConfigChange(FaultTypeBase):
    """The toy's `config_change`: `control.action_delay_steps` set to `params.value` steps
    (an action hook, like `toy_delay`); any other key does nothing."""

    kind = CONFIG_CHANGE_KIND
    fault_class = FaultClass.configuration
    Params = _Value
    dose_levels = (0.0, 1.0)
    truth = toy.ToyDelay.truth

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return toy.DelayState(spec=spec, episode_id=identity.episode_id)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        if state.spec.target.config_key is not ConfigKey.control_action_delay_steps:
            return action
        assert isinstance(state, toy.DelayState)
        steps = int(_Value.model_validate(state.spec.params).value)
        state.line.append(np.asarray(action, np.float32).copy())
        if len(state.line) > steps:
            return state.line.popleft()
        return state.line[0]


TOY_ORACLE_FAULTS = FaultRegistry()
for _type in (
    toy.ToyCameraShift(),
    toy.ToyDelay(),
    toy.ToyImageNoise(),
    toy.ToyKnob(),
    toy.ToyDetector(),
    ToyConfigChange(),
):
    TOY_ORACLE_FAULTS.add(_type)


def faults() -> list[FaultSpec]:
    camera = FaultSpec(
        fault_id="f01",
        kind="toy_camera_shift",
        params={"pixels": 4},
        level=1.0,
        target=Target(scope="r02", component=Component.scene_camera),
        schedule=Schedule(onset_day=0),
    )
    delay = FaultSpec(
        fault_id="f02",
        kind="toy_delay",
        params={"steps": 3},
        level=1.0,
        target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=0),
        nominal=0,
    )
    decoy = FaultSpec(
        fault_id="f03",
        kind="toy_knob",
        params={"steps": 5},
        level=1.0,
        target=Target(scope="s1", config_key=ConfigKey.policy_sampling_steps),
        schedule=Schedule(onset_day=1),
        nominal=2,
    )
    return [camera, delay, decoy]


def spec() -> WorldSpec:
    return WorldSpec(
        world_id="w99",
        seed=WORLD_SEED,
        size="small",
        sites=1,
        robots_per_site=ROBOTS,
        days=DAYS,
        windows=Windows(
            observed=DayRange(first=0, last=0),
            experiment=DayRange(first=1, last=1),
            scoring=DayRange(first=2, last=2),
        ),
        fault_schedule=faults(),
        patterns=Patterns(decoys=["f03"]),
        policy=toy.TOY_SPEC,
        tasks=[Task(toy.TOY_SUITE, 0), Task(toy.TOY_SUITE, 1)],
        episodes_per_robot_day=EPISODES_PER_ROBOT_DAY,
    )


def jobs(software: Any) -> list[Job]:
    """The world's run jobs, every window, with the faults in force on each day."""
    out: list[Job] = []
    number = 0
    schedule = faults()
    for day in range(DAYS):
        for r in range(1, ROBOTS + 1):
            robot = neutral_id(IdKind.robot, r)
            active = [
                f for f in schedule if f.target.covers(robot, "s1") and f.schedule.active_on(day)
            ]
            for index in range(EPISODES_PER_ROBOT_DAY):
                number += 1
                identity = manifests.identity(
                    world_seed=WORLD_SEED,
                    episode_number=number,
                    robot=r,
                    site=1,
                    day=day,
                    episode_index=index,
                    suite=toy.TOY_SUITE,
                    task_index=(number + day) % 2,
                    start_state=(3 * number + r) % 7,
                    policy=toy.TOY_SPEC,
                    batch_size=4,
                    software=software,
                    image_size=IMAGE,
                    max_steps=MAX_STEPS,
                    faults=active,
                    world=99,
                )
                out.append(Job(identity=identity))
    return out


def fleet() -> Fleet:
    """The fleet at the end of the world: r03's delay and s1's sampling steps as configured."""

    def robot(r: int) -> Robot:
        robot_id = neutral_id(IdKind.robot, r)
        values: dict[ConfigKey, ConfigValue] = (
            {ConfigKey.control_action_delay_steps: 3} if robot_id == "r03" else {}
        )
        return Robot(
            id=robot_id,
            site_id="s1",
            components={c: ComponentState(component=c, installed_day=-10) for c in Component},
            config=ConfigVersion(version=1 if values else 0, values=values),
        )

    values: dict[ConfigKey, Any] = {
        ConfigKey.policy_sampling_steps: 2,
        ConfigKey.policy_actions_executed_per_chunk: 5,
        ConfigKey.policy_image_views: 2,
        ConfigKey.policy_precision: "fp32",
        ConfigKey.control_mode: "absolute",
        ConfigKey.control_action_delay_steps: 0,
        ConfigKey.camera_scene_calibration: [0.0, 0.0, 0.0],
        ConfigKey.camera_wrist_calibration: [0.0, 0.0, 0.0],
        ConfigKey.software_version: "1.0",
    }
    site = Site(
        id="s1",
        robots=[robot(r) for r in range(1, ROBOTS + 1)],
        conditions={
            ConditionKey.lighting: 0.0,
            ConditionKey.placement: [0.0, 0.0],
            ConditionKey.object_batch: [0.0, 0.0],
            ConditionKey.clutter: 0.0,
        },
        config=ConfigVersion(version=1, values={ConfigKey.policy_sampling_steps: 5}),
    )
    return Fleet(id="w99", sites=[site], config=ConfigVersion(version=1, values=values))


def fault_table(
    manifest: Sequence[Job], *, detectable: Sequence[str] = ("f01", "f02")
) -> FaultTable:
    """The hidden fault table; `detectable` names the causes the toy generator would list."""
    windows = spec().windows
    names: tuple[WindowName, ...] = ("observed", "experiment", "scoring")
    days = {n: getattr(windows, n) for n in names}
    per_window: dict[WindowName, int] = {
        n: sum(1 for j in manifest if j.identity.day in days[n]) for n in names
    }
    rows = []
    for fault in faults():
        role: Role = "decoy" if fault.fault_id == "f03" else "effective"
        episodes: dict[WindowName, int] = {
            n: sum(
                1
                for j in manifest
                if j.identity.day in days[n] and fault.fault_id in j.identity.fault_ids
            )
            for n in names
        }
        expected: dict[WindowName, float] = {n: 0.0 for n in names}
        qualified: list[WindowName] = ["scoring"] if fault.fault_id in detectable else []
        mapping = TOY_ORACLE_FAULTS.truth_table()[fault.kind]
        component = fault.target.component is not None
        rows.append(
            FaultTruth(
                fault_id=fault.fault_id,
                kind=fault.kind,
                fault_class=FaultClass.sensor if component else FaultClass.configuration,
                target=fault.target,
                role=role,
                onset_day=fault.schedule.onset_day,
                level=fault.level,
                directions=sorted(
                    mapping.directions or {Direction.degraded}, key=lambda d: d.value
                ),
                fixing_verbs=sorted(mapping.fixing_verbs, key=lambda v: v.value),
                episodes=episodes,
                expected_loss_points=expected,
                identifiable=True,
                qualified_by=qualified,
                detectable=fault.fault_id in detectable,
                small=role == "effective" and fault.fault_id not in detectable,
                calibration="toy",
            )
        )
    return FaultTable(
        world_id="w99",
        windows=windows,
        episodes_per_window=per_window,
        rule="toy",
        faults=rows,
    )


def write_world(root: Path, software: Any) -> list[Job]:
    """Write the toy world's files under `root/world`; return its run jobs."""
    folder = Path(root) / "world"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = jobs(software)
    (folder / "spec.json").write_text(spec().model_dump_json(indent=1) + "\n")
    (folder / "model.json").write_text(fleet().model_dump_json(indent=1) + "\n")
    (folder / "description.json").write_text("{}\n")
    (folder / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in manifest))
    seal_world(folder)
    (folder / "fault_table.json").write_text(fault_table(manifest).model_dump_json(indent=1) + "\n")
    return manifest


def engine_overrides(workers: int = 2) -> dict[str, Any]:
    """The engine settings of a toy run (CPU, the toy cell, policy and registry)."""
    return {
        "device": "cpu",
        "precision": "fp32",
        "miopen": None,
        "workers": workers,
        "env_factory": "fleetlab.engine.toy:ToyEnv",
        "policy_factory": "fleetlab.engine.toy:ToyPolicy",
        "check_factory": "fleetlab.engine.toy:ToyCheck",
        "fault_registry": REGISTRY,
        "life": 8,
        "server_threads": 1,
    }


def identities(manifest: Sequence[Job]) -> list[EpisodeIdentity]:
    return [j.identity for j in manifest]


__all__ = [
    "REGISTRY",
    "TOY_ORACLE_FAULTS",
    "ToyConfigChange",
    "engine_overrides",
    "fault_table",
    "faults",
    "fleet",
    "identities",
    "jobs",
    "spec",
    "write_world",
]
