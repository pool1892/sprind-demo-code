"""The effective-policy precedence of the one state model (`fleetlab.loop.state.materialize`)
checked against the engine's own hook chain (`fleetlab.engine.episode.effective_spec` with
the registry's fault types), over a table of combinations: a release that carries the
policy setting and one that does not, a retained set, an arm's override, a revert, an
undo. A review of cc1ff68: a setting hooked before a release that carries the same setting
had no effect. No simulator: the hooks are the engine's, run without an episode.
"""

from __future__ import annotations

from typing import Any

import pytest

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.model import (
    Change,
    ChangeAuthor,
    ComponentState,
    ConfigVersion,
    Fleet,
    Robot,
    Site,
    effective_config,
)
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.engine import manifests
from fleetlab.engine import software as soft
from fleetlab.engine.episode import effective_spec
from fleetlab.faults import REGISTRY
from fleetlab.loop.state import (
    Ledger,
    RobotDay,
    RobotState,
    Step,
    believed_fleet,
    materialize,
)
from fleetlab.loop.world import hook_chain
from fleetlab.worlds.generate import NOMINAL_CONFIG, policy_spec_nominal

TRUTH = REGISTRY.truth_table()
CHUNK = ConfigKey.policy_actions_executed_per_chunk
SOFTWARE = ConfigKey.software_version
SITES = {"r01": "s1", "r02": "s1", "r03": "s2"}


def release(carries: Any) -> FaultSpec:
    return FaultSpec(
        fault_id="f01",
        kind="software_release",
        params={"version": "4.3.0", "carries": carries},
        level=1.0,
        target=Target(scope="fleet", config_key=SOFTWARE),
        schedule=Schedule(onset_day=3),
        nominal="4.2.1",
    )


KNOB = release(
    [
        {
            "kind": "policy_knob",
            "level": 1.0,
            "params": {"value": 10},
            "target": CHUNK.value,
            "nominal": 30,
        }
    ]
)
BLUR = release(
    [
        {
            "kind": "camera_degradation",
            "level": 3.0,
            "params": {"mode": "blur"},
            "target": "scene_camera",
        }
    ]
)


def fleet() -> Fleet:
    values = {**NOMINAL_CONFIG, SOFTWARE: "4.3.0"}

    def robot(robot_id: str) -> Robot:
        return Robot(
            id=robot_id,
            site_id=SITES[robot_id],
            components={c: ComponentState(component=c, installed_day=-9) for c in Component},
        )

    return Fleet(
        id="w01",
        sites=[
            Site(id="s1", robots=[robot("r01"), robot("r02")]),
            Site(id="s2", robots=[robot("r03")]),
        ],
        config=ConfigVersion(version=2, values=values, since_day=3),
        change_log=[
            Change(
                day=3,
                scope="fleet",
                key=SOFTWARE,
                old="4.2.1",
                new="4.3.0",
                author=ChangeAuthor.auto,
            )
        ],
    )


FLEET = fleet()
PROBE = manifests.identity(
    world_seed=1,
    episode_number=1,
    robot=1,
    site=1,
    day=9,
    episode_index=0,
    suite="libero_spatial",
    task_index=0,
    start_state=0,
    policy=policy_spec_nominal(),
    batch_size=4,
    software=soft.current("none"),
)


def set_(scope: str, value: int) -> Action:
    return Action(verb=Verb.set, target=Target(scope=scope, config_key=CHUNK), value=value)


def revert(scope: str, key: ConfigKey = CHUNK) -> Action:
    return Action(verb=Verb.revert, target=Target(scope=scope, config_key=key))


def state(world: FaultSpec, ledger: Ledger, arm: list[Action], robot: str = "r01") -> RobotState:
    day = 9
    steps = [*ledger.active_steps(day), *(Step("arm", day, a) for a in arm)]
    site = FLEET.site(SITES[robot])
    values: dict[Any, Any] = dict(effective_config(FLEET, robot))
    values.update(site.conditions)
    return materialize(
        [world],
        RobotDay(robot, site.id, day),
        steps,
        TRUTH,
        values,
        {"fleet": dict(FLEET.config.values), site.id: {}, robot: {}},
        hook_chain("fleetlab.faults:REGISTRY", PROBE),
    )


def engine_chunk(faults: tuple[FaultSpec, ...]) -> int:
    """What an attempt runs with: the engine's own effective_spec over the faults."""
    pairs = [(REGISTRY.get(f.kind), REGISTRY.get(f.kind).new_episode(f, PROBE)) for f in faults]
    return effective_spec(PROBE, pairs).actions_executed_per_chunk


EMPTY = Ledger.empty("w01")
L_SET20 = EMPTY.adopt(1, 8, adopt=[set_("r01", 20)])
L_SET30 = EMPTY.adopt(1, 8, adopt=[set_("r01", 30)])
L_UNDONE = EMPTY.adopt(1, 6, adopt=[set_("r01", 20)]).adopt(2, 8, undo=["L1"])
L_ROLLBACK = EMPTY.adopt(1, 8, adopt=[revert("fleet", SOFTWARE)])
L_ROLLBACK_SET = EMPTY.adopt(1, 8, adopt=[revert("fleet", SOFTWARE), set_("s1", 20)])
L_FLEET20 = EMPTY.adopt(1, 8, adopt=[set_("fleet", 20)])

# (case, release, ledger, arm, what the attempt runs with, what the records say)
TABLE = [
    ("release carries the setting, nothing adopted", KNOB, EMPTY, [], 10, 30),
    ("a retained set over the release", KNOB, L_SET20, [], 20, 20),
    ("a retained set to the believed value", KNOB, L_SET30, [], 30, 30),
    ("an arm overrides the retained set", KNOB, L_SET20, [set_("r01", 25)], 25, 25),
    ("an arm reverts the retained set", KNOB, L_SET20, [revert("r01")], 10, 30),
    ("the retained set undone", KNOB, L_UNDONE, [], 10, 30),
    ("the release rolled back", KNOB, L_ROLLBACK, [], 30, 30),
    ("rolled back, then a site setting", KNOB, L_ROLLBACK_SET, [], 20, 20),
    ("a fleet-wide set over the release", KNOB, L_FLEET20, [], 20, 20),
    ("an arm's site setting over the release", KNOB, EMPTY, [set_("s1", 15)], 15, 15),
    ("an arm rolls the release back", KNOB, L_SET20, [revert("fleet", SOFTWARE)], 20, 20),
    ("release without the setting", BLUR, EMPTY, [], 30, 30),
    ("a retained set, release without it", BLUR, L_SET20, [], 20, 20),
    ("an arm reverts it, release without it", BLUR, L_SET20, [revert("r01")], 30, 30),
]


@pytest.mark.parametrize(
    ("world", "ledger", "arm", "runs", "says"),
    [row[1:] for row in TABLE],
    ids=[row[0] for row in TABLE],
)
def test_precedence_against_the_engines_hooks(
    world: FaultSpec, ledger: Ledger, arm: list[Action], runs: int, says: int
) -> None:
    result = state(world, ledger, arm)
    assert engine_chunk(result.faults) == runs
    assert result.current[CHUNK] == says
    if not arm:  # the records of the ledger say what the robot's configuration is
        assert effective_config(believed_fleet(lambda d: FLEET, ledger, 9), "r01")[CHUNK] == says
    # Where the fleet's own configuration decides, the records and the attempt agree; a
    # release's carried setting shows only in what the attempt runs (it is hidden).
    hidden = world is KNOB and runs == 10
    assert (runs == says) != hidden
    # The world's behaviour comes first, then the ledger's changes, then the arm's.
    sources = [f.kind for f in result.faults]
    assert sources == sorted(sources, key=lambda k: k == "config_change")
