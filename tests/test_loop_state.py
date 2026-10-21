"""The ledger and the one interpretation of a state (docs/design/loop.md, section 2), on
the real fault registry's truth mappings. No simulator: these are the fold's rules.

Regression tests from a review of c3985fd: a retained `set` on a healthy key
survives into operation and into an experiment's control arm; a treatment arm's override
or revert of a retained action yields its intended state.
"""

from __future__ import annotations

from pathlib import Path
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
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.faults import REGISTRY
from fleetlab.loop.state import (
    MAX_CHANGES_PER_ROUND,
    Adoption,
    Ledger,
    Replacement,
    RobotDay,
    RobotState,
    Step,
    Undo,
    apply_arm,
    believed_fleet,
    changes_in_force,
    decision_changes,
    effective_steps,
    materialize,
    one_look_ledger,
)

TRUTH = REGISTRY.truth_table()
DELAY = ConfigKey.control_action_delay_steps
CHUNK = ConfigKey.policy_actions_executed_per_chunk
NOMINAL: dict[ConfigKey, Any] = {DELAY: 0, CHUNK: 30, ConfigKey.policy_sampling_steps: 10}
SITES = {"r01": "s1", "r02": "s1", "r03": "s2", "r04": "s2"}

# The world: r02 commands 17 steps late (its own change), all of s2 too (the site's
# change), r04 executes 15 actions per chunk, r01's scene camera is turned.
LATE_R02 = FaultSpec(
    fault_id="f01",
    kind="action_latency",
    level=17.0,
    target=Target(scope="r02", config_key=DELAY),
    schedule=Schedule(onset_day=0),
    nominal=0,
)
LATE_S2 = LATE_R02.model_copy(
    update={"fault_id": "f02", "target": Target(scope="s2", config_key=DELAY)}
)
SHORT_R04 = FaultSpec(
    fault_id="f03",
    kind="policy_knob",
    params={"value": 15},
    level=1.0,
    target=Target(scope="r04", config_key=CHUNK),
    schedule=Schedule(onset_day=1),
    nominal=30,
)
CAMERA_R01 = FaultSpec(
    fault_id="f04",
    kind="camera_pose_offset",
    level=6.0,
    target=Target(scope="r01", component=Component.scene_camera),
    schedule=Schedule(onset_day=0),
)
SCHEDULE = [LATE_R02, LATE_S2, SHORT_R04, CAMERA_R01]


def world_fleet() -> Fleet:
    def robot(robot_id: str, values: dict[ConfigKey, Any]) -> Robot:
        return Robot(
            id=robot_id,
            site_id=SITES[robot_id],
            components={c: ComponentState(component=c, installed_day=-9) for c in Component},
            config=ConfigVersion(version=1 if values else 0, values=values),
        )

    conditions: dict[ConditionKey, Any] = {
        ConditionKey.object_batch: [0.0, 0.0],
        ConditionKey.lighting: [0.0, 0.0],
    }
    return Fleet(
        id="w01",
        sites=[
            Site(
                id="s1",
                robots=[robot("r01", {}), robot("r02", {DELAY: 17})],
                conditions=conditions,
            ),
            Site(
                id="s2",
                robots=[robot("r03", {}), robot("r04", {CHUNK: 15})],
                conditions=conditions,
                config=ConfigVersion(version=1, values={DELAY: 17}),
            ),
        ],
        config=ConfigVersion(version=1, values=dict(NOMINAL)),
        change_log=[
            Change(day=0, scope="r02", key=DELAY, old=0, new=17, author=ChangeAuthor.vendor),
            Change(day=0, scope="s2", key=DELAY, old=0, new=17, author=ChangeAuthor.auto),
            Change(day=1, scope="r04", key=CHUNK, old=30, new=15, author=ChangeAuthor.ops),
        ],
    )


FLEET = world_fleet()


def world_faults(robot_id: str, day: int) -> list[FaultSpec]:
    """Every fault in force on the robot, shadowed configuration faults included."""
    site = SITES[robot_id]
    return [f for f in SCHEDULE if f.schedule.active_on(day) and f.target.covers(robot_id, site)]


def layers(robot_id: str) -> dict[str, dict[ConfigKey, Any]]:
    site = FLEET.site(SITES[robot_id])
    return {
        "fleet": dict(FLEET.config.values),
        site.id: dict(site.config.values),
        robot_id: dict(FLEET.robot(robot_id).config.values),
    }


def current(robot_id: str) -> dict[Any, Any]:
    site = FLEET.site(SITES[robot_id])
    return {**effective_config(FLEET, robot_id), **site.conditions}


def state_on(
    robot_id: str, day: int, ledger: Ledger, arm: list[Action] | None = None
) -> RobotState:
    state = materialize(
        world_faults(robot_id, day),
        RobotDay(robot_id, SITES[robot_id], day),
        ledger.active_steps(day),
        TRUTH,
        current(robot_id),
        layers(robot_id),
    )
    return apply_arm(state, arm or [], TRUTH)


def kinds(state: RobotState) -> list[tuple[str, Any]]:
    """(kind, what it sets) of every fault in force, sorted."""
    out = []
    for f in state.faults:
        value = f.params.get("value", f.level) if f.target.kind is not None else f.level
        out.append((f.kind, value))
    return sorted(out, key=str)


def set_(scope: str, key: ConfigKey, value: Any) -> Action:
    return Action(verb=Verb.set, target=Target(scope=scope, config_key=key), value=value)


def revert(scope: str, key: ConfigKey) -> Action:
    return Action(verb=Verb.revert, target=Target(scope=scope, config_key=key))


def recalibrate(robot_id: str) -> Action:
    return Action(
        verb=Verb.recalibrate, target=Target(scope=robot_id, component=Component.scene_camera)
    )


def physical(state: RobotState, key: ConfigKey) -> Any:
    """The value the robot runs with: the narrowest change on the key, else the nominal."""
    from fleetlab.contracts.ids import scope_rank
    from fleetlab.faults.records import change_entry

    on_key = [f for f in state.faults if f.target.config_key is key]
    if not on_key:
        return NOMINAL[key]
    narrowest = max(on_key, key=lambda f: scope_rank(f.target.scope))
    entry = change_entry(narrowest, 0, ChangeAuthor.ops)
    assert entry is not None
    return entry.new


# ---- the ledger ------------------------------------------------------------------------------


def test_adopt_replace_undo_and_active_by_day(tmp_path: Path) -> None:
    l0 = Ledger.empty("w01")
    l1 = l0.adopt(
        1,
        8,
        adopt=[
            Adoption(action=set_("r03", CHUNK, 10), rationale="shorter replanning"),
            Adoption(action=recalibrate("r01"), rationale="residual high"),
        ],
    )
    assert l0.entries == [] and l0.version == 0  # the old version is unchanged
    assert [e.entry_id for e in l1.entries] == ["L1", "L2"] and l1.version == 1
    assert l1.active(7) == [] and len(l1.active(8)) == 2
    l2 = l1.adopt(
        2,
        12,
        replace=[Replacement(entry_id="L1", action=set_("r03", CHUNK, 20), rationale="less")],
        undo=[],
        adopt=[set_("s2", DELAY, 0)],
    )
    old, new = l2.entry("L1"), l2.entry("L3")
    assert (old.status, old.retired_day, old.replaced_by) == ("replaced", 12, "L3")
    assert new.replaces == "L1" and new.adopted_day == 12
    assert [e.entry_id for e in l2.active_entries(11)] == ["L1", "L2"]  # history stays exact
    assert [e.entry_id for e in l2.active_entries(12)] == ["L2", "L3", "L4"]
    l3 = l2.adopt(3, 14, undo=[Undo(entry_id="L4", rationale="no gain")])
    assert l3.entry("L4").status == "undone" and l3.entry("L4").retired_day == 14
    assert [e.entry_id for e in l3.active_entries(14)] == ["L2", "L3"]
    assert changes_in_force(l3) == 2
    # Every version stored, never rewritten with other content.
    for ledger in (l0, l1, l2, l3):
        ledger.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"ledger-r{n}.json" for n in range(4)]
    assert Ledger.load(tmp_path / "ledger-r2.json") == l2
    with pytest.raises(ValueError, match="another ledger"):
        l3.model_copy(update={"version": 2}).save(tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ledger: ledger.adopt(3, 20), "follows version"),
        (lambda ledger: ledger.adopt(2, 8), "not after"),
        (lambda ledger: ledger.adopt(2, 12, undo=["L2"]), "physical"),
        (lambda ledger: ledger.adopt(2, 12, undo=["L9"]), "no ledger entry"),
        (lambda ledger: ledger.adopt(2, 12, adopt=[set_("r03", CHUNK, 10)]), "already in force"),
        (
            lambda ledger: ledger.adopt(
                2,
                12,
                undo=["L1"],
                replace=[Replacement(entry_id="L1", action=set_("r03", CHUNK, 5))],
            ),
            "twice",
        ),
        (
            lambda ledger: ledger.adopt(
                2, 12, adopt=[Action(verb=Verb.set, target=Target(scope="r03", config_key=CHUNK))]
            ),
            "complete",
        ),
    ],
)
def test_adopt_refuses(change: Any, message: str) -> None:
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", CHUNK, 10), recalibrate("r01")])
    with pytest.raises(ValueError, match=message):
        change(l1)


def test_conventions_shared_by_driver_and_evaluator() -> None:
    assert MAX_CHANGES_PER_ROUND == 3
    assert decision_changes(adopt=[1, 2], replace=[3]) == 3
    assert decision_changes(adopt=[], replace=[]) == 0
    ranked = [
        Action(verb=Verb.route_to_teleop, target=Target(scope="fleet"), value="carry"),
        recalibrate("r01"),
        Action(verb=Verb.set, target=Target(scope="r03", config_key=CHUNK)),  # no value
        set_("s2", DELAY, 0),
        recalibrate("r01"),
        set_("r04", CHUNK, 30),
    ]
    plan = one_look_ledger("w01", ranked, 2, 8)
    assert plan.version == 1 and [e.action for e in plan.entries] == [
        recalibrate("r01"),
        set_("s2", DELAY, 0),
    ]
    assert all(e.adopted_day == 8 for e in plan.entries)
    assert one_look_ledger("w01", ranked, 0, 8).entries == []


# ---- the fold --------------------------------------------------------------------------------


def test_regression_retained_set_on_a_healthy_key_survives() -> None:
    """Review point (a): a retained set on a healthy key is in force in operation and in the
    control arm of a later experiment; it is not a no-op against `current`."""
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", CHUNK, 10)])
    for day in (8, 9, 12):
        state = state_on("r03", day, l1)
        added = [f for f in state.faults if f.kind == "config_change"]
        assert [(f.params["value"], f.nominal, f.schedule.onset_day) for f in added] == [
            (10, 30, 8)
        ]
        assert state.current[CHUNK] == 10
    control = state_on("r03", 12, l1, arm=[])
    assert kinds(control) == kinds(state_on("r03", 12, l1))
    assert state_on("r03", 7, l1).faults == tuple(world_faults("r03", 7))  # before adoption


def test_regression_arm_overrides_or_reverts_a_retained_action() -> None:
    """Review point (b): an arm's set overrides a retained set; an arm's revert undoes it."""
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", CHUNK, 10), set_("r04", CHUNK, 20)])
    override = state_on("r03", 12, l1, arm=[set_("r03", CHUNK, 25)])
    assert physical(override, CHUNK) == 25
    assert [f.params["value"] for f in override.faults if f.kind == "config_change"] == [25]
    back = state_on("r03", 12, l1, arm=[revert("r03", CHUNK)])
    assert back.faults == tuple(world_faults("r03", 12)) and physical(back, CHUNK) == 30
    # r04's retained set had overridden the world's change at r04: reverting it brings the
    # world's 15 back, as the change log (15, then 20) says a revert would.
    back4 = state_on("r04", 12, l1, arm=[revert("r04", CHUNK)])
    assert physical(back4, CHUNK) == 15 and SHORT_R04.fault_id in back4.fault_ids()
    # Each revert undoes the latest change left: the retained set, then the world's own.
    twice = state_on("r04", 12, l1, arm=[revert("r04", CHUNK), revert("r04", CHUNK)])
    assert physical(twice, CHUNK) == 30 and SHORT_R04.fault_id not in twice.fault_ids()


def test_revert_of_the_worlds_change_and_values_after_it() -> None:
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[revert("r02", DELAY)])
    state = state_on("r02", 9, l1)
    assert LATE_R02.fault_id not in state.fault_ids() and state.current[DELAY] == 0
    # A later set compares with the value in force after the ledger (0), not the world's
    # (17): setting 17 again is a change with its hook, not a no-op.
    again = state_on("r02", 12, l1, arm=[set_("r02", DELAY, 17)])
    assert physical(again, DELAY) == 17
    assert [(f.params["value"], f.nominal) for f in again.faults] == [(17, 0)]
    # The arm reverting the retained revert puts the world's change back.
    undone = state_on("r02", 12, l1, arm=[revert("r02", DELAY)])
    assert undone.fault_ids() == [LATE_R02.fault_id]


def test_exact_scope_rule_and_robot_overrides() -> None:
    # Reverting at the site removes the site's change for its robots.
    site = Ledger.empty("w01").adopt(1, 8, adopt=[revert("s2", DELAY)])
    assert physical(state_on("r03", 9, site), DELAY) == 0
    # Reverting at a robot where the change was made at its site removes nothing.
    robot = Ledger.empty("w01").adopt(1, 8, adopt=[revert("r03", DELAY)])
    assert state_on("r03", 9, robot).fault_ids() == [LATE_S2.fault_id]
    # A robot's own setting overrides its site's change; set to the value in force it
    # is recorded and runs no hook, and a later site-wide change does not reach it.
    pin = Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", DELAY, 0)])
    pinned = state_on("r03", 9, pin)
    assert [(f.kind, f.params["value"], f.nominal) for f in pinned.faults] == [
        ("config_change", 0, 0)
    ]
    assert physical(pinned, DELAY) == 0
    wider = state_on("r03", 12, pin, arm=[set_("s2", DELAY, 5)])
    assert physical(wider, DELAY) == 0
    # Reverting the robot's own setting gives the robot its site's value again.
    lifted = state_on("r03", 12, pin, arm=[revert("r03", DELAY)])
    assert lifted.fault_ids() == [LATE_S2.fault_id]
    # A narrower change decides: a site-wide set does not reach r04's own chunk setting.
    assert physical(state_on("r04", 12, Ledger.empty("w01"), [set_("s2", CHUNK, 10)]), CHUNK) == 15


def test_component_verbs_and_onsets() -> None:
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[recalibrate("r01")])
    assert CAMERA_R01.fault_id not in state_on("r01", 9, l1).fault_ids()
    assert CAMERA_R01.fault_id in state_on("r01", 7, l1).fault_ids()
    assert state_on("r02", 9, l1).fault_ids() == [LATE_R02.fault_id]
    # A step never meets a fault that starts after it on the same scope and key: the loop
    # refuses to order those (no stored world has one).
    early = Ledger.empty("w01").adopt(1, 0, adopt=[revert("r04", CHUNK)])
    with pytest.raises(ValueError, match="precedes a fault"):
        state_on("r04", 3, early)


def test_entry_changes_have_their_own_ids() -> None:
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", CHUNK, 10)])
    l2 = l1.adopt(2, 12, undo=["L1"], adopt=[set_("r01", CHUNK, 10)])
    l3 = l2.adopt(3, 14, adopt=[set_("r03", CHUNK, 10)])  # the same action, a new entry
    first = [f.fault_id for f in state_on("r03", 9, l3).faults if f.kind == "config_change"]
    later = [f.fault_id for f in state_on("r03", 15, l3).faults if f.kind == "config_change"]
    assert first != later and len(set(first + later)) == 2


def test_effective_steps_cancel_in_pairs() -> None:
    a = Step("L1", 8, set_("r03", CHUNK, 10))
    b = Step("arm", 12, revert("r03", CHUNK))
    c = Step("arm", 12, revert("r03", CHUNK))
    d = Step("arm", 12, set_("s2", CHUNK, 12))
    assert effective_steps([a, b]) == []
    assert effective_steps([a, d, b]) == [d]
    assert effective_steps([a, b, c]) == [c]  # the third undoes the world's change


def test_regression_shadowed_wider_set_is_revealed() -> None:
    """A review of 2e151ff: a retained site-wide set that r04's own setting shadows comes
    back when r04's own setting is reverted, in the ledger and in an arm."""
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[set_("s2", CHUNK, 10)])
    assert physical(state_on("r04", 9, l1), CHUNK) == 15  # r04's own value decides
    assert physical(state_on("r03", 9, l1), CHUNK) == 10
    l2 = l1.adopt(2, 12, adopt=[revert("r04", CHUNK)])
    for state in (state_on("r04", 13, l2), state_on("r04", 12, l1, arm=[revert("r04", CHUNK)])):
        assert physical(state, CHUNK) == 10 and state.current[CHUNK] == 10
    assert effective_config(believed_fleet(lambda d: FLEET, l2, 13), "r04")[CHUNK] == 10


def test_fleet_set_under_site_and_robot_overrides() -> None:
    l1 = Ledger.empty("w01").adopt(1, 8, adopt=[set_("fleet", CHUNK, 12), set_("s2", CHUNK, 20)])
    expected = {"r01": 12, "r02": 12, "r03": 20, "r04": 15}
    assert {r: physical(state_on(r, 9, l1), CHUNK) for r in SITES} == expected
    assert physical(state_on("r04", 12, l1, arm=[revert("r04", CHUNK)]), CHUNK) == 20
    l2 = l1.adopt(2, 12, undo=["L2"], adopt=[revert("r04", CHUNK)])
    assert {r: physical(state_on(r, 13, l2), CHUNK) for r in SITES} == dict.fromkeys(SITES, 12)
    arm = [revert("fleet", CHUNK)]  # cancels the retained fleet-wide set
    assert physical(state_on("r03", 13, l2, arm=arm), CHUNK) == 30


# ---- the records agree with the physics ------------------------------------------------------

LEDGERS = [
    Ledger.empty("w01"),
    Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", CHUNK, 10), revert("r02", DELAY)]),
    Ledger.empty("w01").adopt(1, 8, adopt=[revert("s2", DELAY), set_("r04", CHUNK, 30)]),
    Ledger.empty("w01").adopt(1, 8, adopt=[set_("r03", DELAY, 0), set_("s1", CHUNK, 12)]),
    Ledger.empty("w01")
    .adopt(1, 8, adopt=[set_("r04", CHUNK, 20), revert("r02", DELAY)])
    .adopt(2, 12, replace=[Replacement(entry_id="L1", action=revert("r04", CHUNK))], undo=["L2"]),
    Ledger.empty("w01")
    .adopt(1, 8, adopt=[set_("s2", CHUNK, 10)])
    .adopt(2, 12, adopt=[revert("r04", CHUNK)]),
    Ledger.empty("w01")
    .adopt(1, 8, adopt=[set_("fleet", CHUNK, 12), set_("s2", CHUNK, 20), set_("r02", DELAY, 4)])
    .adopt(2, 12, undo=["L2"], adopt=[revert("r04", CHUNK), revert("s2", DELAY)]),
]


@pytest.mark.parametrize("ledger", LEDGERS, ids=range(len(LEDGERS)))
def test_records_agree_with_the_simulated_values(ledger: Ledger) -> None:
    """For every robot and key the believed configuration after the ledger is the value
    the robot runs with, and the records hold the ledger's own entries only."""
    day = 13
    believed = believed_fleet(lambda d: FLEET, ledger, day)
    for robot_id in SITES:
        state = state_on(robot_id, day, ledger)
        values = effective_config(believed, robot_id)
        for key in (DELAY, CHUNK):
            assert values[key] == physical(state, key), (robot_id, key)
            assert state.current[key] == values[key]
    world = {c.model_dump_json() for c in FLEET.change_log}
    mine = [c for c in believed.change_log if c.model_dump_json() not in world]
    assert all(c.author is ChangeAuthor.ops and c.day in (8, 12) for c in mine)
    text = believed.model_dump_json()
    for word in ("action_latency", "policy_knob", "config_change", "f01", "f02", "f9"):
        assert word not in text


def test_records_of_set_revert_undo_and_maintenance() -> None:
    l1 = Ledger.empty("w01").adopt(
        1, 8, adopt=[set_("r03", CHUNK, 10), revert("r02", DELAY), recalibrate("r01")]
    )
    l2 = l1.adopt(2, 12, undo=["L1"])
    believed = believed_fleet(lambda d: FLEET, l2, 13)
    mine = [(c.day, c.scope, c.key.value, c.old, c.new) for c in believed.change_log][3:]
    assert mine == [
        (8, "r02", DELAY.value, 17, 0),
        (8, "r03", CHUNK.value, 30, 10),
        (12, "r03", CHUNK.value, 10, 30),
    ]
    assert [(m.day, m.robot_id, m.verb.value) for m in believed.maintenance_log] == [
        (8, "r01", "recalibrate")
    ]
    # Before the undo's day, the version of the ledger with the set shows it in force.
    on9 = believed_fleet(lambda d: FLEET, l2, 9)
    assert effective_config(on9, "r03")[CHUNK] == 10
    assert effective_config(believed, "r03")[CHUNK] == 30
    # Conditions: the site's believed condition after a change.
    cond = Action(
        verb=Verb.change_condition,
        target=Target(scope="s1", condition_key=ConditionKey.object_batch),
        value=[1.0, 0.0],
    )
    after = believed_fleet(lambda d: FLEET, Ledger.empty("w01").adopt(1, 8, adopt=[cond]), 9)
    assert after.site("s1").conditions[ConditionKey.object_batch] == [1.0, 0.0]
    assert after.site("s2").conditions[ConditionKey.object_batch] == [0.0, 0.0]
