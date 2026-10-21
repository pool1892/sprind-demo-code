"""The miniature fleet: its spec, version 1.

Two sites of four robots, four simulated days, 13 episodes per robot and day (416 as
run), the twenty tasks of `libero_spatial` and `libero_object`, and three faults from
the merged fault types at calibrated levels (`worlds.calibration`):

- `f01`: the scene camera of one robot of site s1 turned 6 degrees (`camera_pose_offset`,
  pan), from day 0. A sensor fault, robot-scoped, no `Change`. The robot is the one of s1
  with the largest drawn mounting tolerance (a hidden trait), so the seed decides it.
- `f02`: commands executed 17 steps late at site s2 (`action_latency` on
  `control.action_delay_steps`, 0 to 17), from day 1. A configuration fault, site-scoped,
  with its `Change` on day 1.
- `f03`, the decoy: `policy.sampling_steps` 10 to 5 at site s1 (`policy_knob`), from day 1,
  with its `Change` on day 1 (calibrated flat: 100 of 100 against 98 of 100 healthy twins).

Windows: observed days 0 and 1, experiment day 2, scoring day 3.
"""

from __future__ import annotations

from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import Task
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.vocabulary import Component, ConfigKey
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec
from fleetlab.worlds.generate import TRAIT_NAMES, policy_spec_nominal, robot_ids, traits_for

WORLD_ID = "w10"
SEED = 10
# The fixtures' version (`~/.cache/robots-fixtures/<VERSION>/`). mini-v1: the first
# export, whose description held the whole catalogue; mini-v2: the same world and runs,
# exported with the visible tables `costs` and `catalogue` and a description read in
# parts (contracts batch 9). mini-v1 stays for the tests that pin it.
VERSION = "mini-v2"
TASKS = [Task("libero_spatial", i) for i in range(10)] + [
    Task("libero_object", i) for i in range(10)
]


def _camera_robot(seed: int, robots: list[str]) -> str:
    """The robot whose camera gets knocked: the largest mounting tolerance."""
    return max(robots, key=lambda r: (traits_for(seed, r)[TRAIT_NAMES[0]], r))


def mini_spec(seed: int = SEED, world_id: str = WORLD_ID) -> WorldSpec:
    """The miniature fleet's spec. The camera robot is drawn from the seed's traits."""
    sites, per_site, days = 2, 4, 4
    shell = WorldSpec(
        world_id=world_id,
        seed=seed,
        size="small",
        sites=sites,
        robots_per_site=per_site,
        days=days,
        windows=Windows(
            observed=DayRange(first=0, last=1),
            experiment=DayRange(first=2, last=2),
            scoring=DayRange(first=3, last=3),
        ),
        fault_schedule=[],
        patterns=Patterns(),
        policy=policy_spec_nominal(),
        tasks=TASKS,
        episodes_per_robot_day=13,
    )
    s1, s2 = neutral_id(IdKind.site, 1), neutral_id(IdKind.site, 2)
    camera = FaultSpec(
        fault_id=neutral_id(IdKind.fault, 1),
        kind="camera_pose_offset",
        params={},
        level=6.0,
        target=Target(
            scope=_camera_robot(seed, robot_ids(shell, 0)), component=Component.scene_camera
        ),
        schedule=Schedule(onset_day=0),
    )
    latency = FaultSpec(
        fault_id=neutral_id(IdKind.fault, 2),
        kind="action_latency",
        params={},
        level=17.0,
        target=Target(scope=s2, config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=1),
        nominal=0,
    )
    decoy = FaultSpec(
        fault_id=neutral_id(IdKind.fault, 3),
        kind="policy_knob",
        params={"value": 5},
        level=1.0,
        target=Target(scope=s1, config_key=ConfigKey.policy_sampling_steps),
        schedule=Schedule(onset_day=1),
        nominal=10,
    )
    return WorldSpec.model_validate(
        {
            **shell.model_dump(mode="json"),
            "fault_schedule": [f.model_dump(mode="json") for f in (camera, latency, decoy)],
            "patterns": Patterns(decoys=[decoy.fault_id]).model_dump(mode="json"),
        }
    )
