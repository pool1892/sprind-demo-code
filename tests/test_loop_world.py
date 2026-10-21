"""The loop's view of a stored world, on a world the miniature fleet's own generator writes
(real fault kinds, health readings, no run): the materialized state against the as-run
identities, the records and health after a ledger, the reserved panel. No simulator."""

from __future__ import annotations

from pathlib import Path

import pytest

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.model import effective_config
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.engine import software as soft
from fleetlab.loop.layout import layout_for_world
from fleetlab.loop.state import Ledger
from fleetlab.loop.toy import write_paper_world
from fleetlab.loop.world import LoopWorld

DELAY = ConfigKey.control_action_delay_steps


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> LoopWorld:
    return write_paper_world(tmp_path_factory.mktemp("paper"))


def recalibrate(robot_id: str) -> Action:
    return Action(
        verb=Verb.recalibrate, target=Target(scope=robot_id, component=Component.scene_camera)
    )


def test_original_state_is_the_as_run_world(world: LoopWorld) -> None:
    empty = Ledger.empty(world.world_id)
    for identity in world.identities:
        mine = world.state_on(identity.robot_id, identity.day, empty).faults
        assert sorted(f.fault_id for f in mine) == sorted(f.fault_id for f in identity.faults)


def test_regression_description_health_matches_health_table(world: LoopWorld) -> None:
    """A review of 2e151ff: after recalibrate(r01) on day 8, the description's latest camera
    residual of r01 on day 9 is the health table's, and both show the repair."""
    l1 = Ledger.empty(world.world_id).adopt(1, 8, adopt=[recalibrate("r01")])
    public = world.public_fleet(9, l1)
    shown = public.robot("r01").components[Component.scene_camera].health
    rows = {
        r["metric"]: r["value"]
        for r in world.health_rows([9], l1)
        if r["robot_id"] == "r01" and r["component"] == "scene_camera"
    }
    assert shown == rows
    stored = {
        r["metric"]: r["value"]
        for r in world.health
        if r["day"] == 9 and r["robot_id"] == "r01" and r["component"] == "scene_camera"
    }
    assert rows["calibration_residual"] < stored["calibration_residual"] - 40  # the repair
    # Before the adoption day nothing moves.
    before = world.health_rows([7], l1)
    key = ("day", "robot_id", "component", "metric")
    stored7 = sorted((r for r in world.health if r["day"] == 7), key=lambda r: [r[k] for k in key])
    assert before == stored7
    # No hidden marker: the records name no fault, kind or trait.
    text = public.model_dump_json()
    for word in ("camera_pose_offset", "f01", "f9", "mounting_tolerance"):
        assert word not in text


def test_records_and_values_after_a_ledger(world: LoopWorld) -> None:
    revert = Action(verb=Verb.revert, target=Target(scope="s2", config_key=DELAY))
    l1 = Ledger.empty(world.world_id).adopt(1, 8, adopt=[revert, recalibrate("r01")])
    public = world.public_fleet(9, l1)
    assert effective_config(public, "r05")[DELAY] == 0
    mine = [c for c in public.change_log if c.author.value == "ops" and c.day == 8]
    assert [(c.scope, c.key, c.old, c.new) for c in mine] == [("s2", DELAY, 17, 0)]
    assert [(m.day, m.robot_id, m.verb) for m in public.maintenance_log if m.day == 8] == [
        (8, "r01", Verb.recalibrate)
    ]
    for robot in public.robots():
        state = world.state_on(robot.id, 9, l1)
        assert state.current[DELAY] == effective_config(public, robot.id)[DELAY]
        assert not any(f.target.config_key is DELAY for f in state.faults)
    assert all(f.kind != "camera_pose_offset" for f in world.state_on("r01", 9, l1).faults)


def test_panel_follows_the_worlds_own_rule(world: LoopWorld) -> None:
    layout = layout_for_world(world.spec)
    software = soft.current("none")
    days = list(range(layout.evaluation.first, layout.evaluation.last + 1))
    assert days == [12, 13] and all(d >= world.days for d in days)
    panel = world.panel_identities(days, software)
    assert len(panel) == 2 * len(world.truth.fleet.robots()) * world.truth.episodes_per_robot_day
    exported = {i.episode_id for i in world.identities}
    assert not exported & {i.episode_id for i in panel}
    assert len({i.episode_id for i in panel}) == len(panel)
    # The same rule gives the world's own as-run tasks and start states on stored days.
    for identity in world.identities[:40]:
        task, start = world.slot_tasks(identity.robot_id, identity.day, identity.episode_index)
        assert (task.suite, task.task_index, start) == (
            identity.suite,
            identity.task_index,
            identity.start_state,
        )
    assert world.panel_identities(days, software) == panel  # minted deterministically
    small = world.panel_identities(days, software, per_robot_day=1)
    assert {i.episode_index for i in small} == {0}
    # Faults hold past the horizon (the camera fault persists).
    assert any(f.kind == "camera_pose_offset" for f in world.faults_on("r01", 13))


def test_regular_work_is_the_as_run_identities(world: LoopWorld) -> None:
    software = soft.current("none")
    l1 = Ledger.empty(world.world_id).adopt(1, 8, adopt=[recalibrate("r01")])
    work = world.regular_identities([8, 9], software, l1)
    as_run = {i.episode_id: i for i in world.identities if i.day in (8, 9)}
    assert [i.episode_id for i in work] == list(as_run)
    for identity in work:
        original = as_run[identity.episode_id]
        assert (identity.scene_seed, identity.noise_seed, identity.task_index) == (
            original.scene_seed,
            original.noise_seed,
            original.task_index,
        )
        assert identity.software == software
        if identity.robot_id == "r01":
            assert not any(f.kind == "camera_pose_offset" for f in identity.faults)
    with pytest.raises(ValueError, match="no regular work"):
        world.regular_identities([10], software, l1)


def test_loader_refuses_a_change_after_a_configuration_fault(tmp_path: Path) -> None:
    world = write_paper_world(tmp_path / "w")
    model = world.model
    extra = model.change_log[0].model_copy(update={"day": 9})  # a later change, same place
    broken = model.model_copy(update={"change_log": [*model.change_log, extra]})
    (world.files / "model.json").write_text(broken.model_dump_json())
    key = model.change_log[0]
    faulted = any(
        f.target.scope == key.scope and f.target.config_key is key.key
        for f in world.spec.fault_schedule
    )
    if faulted:
        with pytest.raises(ValueError, match="follows a configuration fault"):
            LoopWorld.load(tmp_path / "w")


def test_loop_runs_record_the_event_kinds_the_worlds_runs_carry(world: LoopWorld) -> None:
    """The miniature fleet's runs predate the derived events; the development worlds' runs
    carry them with the contact replay. A loop run extends the history in kind."""
    import json
    from dataclasses import replace

    from fleetlab.loop.toy import toy_setup

    setup = replace(toy_setup(), events=True, contacts=True)
    assert world.events_record is None  # the paper world, like the miniature fleet
    assert world.event_setup(setup).events is False
    runner = world.root / "runner.json"
    try:
        runner.write_text(json.dumps({"steps": {"observed-events": {"done": True}}}))
        assert world.event_setup(setup).events and not world.event_setup(setup).contacts
        runner.write_text(
            json.dumps({"steps": {"observed-events": {"done": True, "contact_replay": {}}}})
        )
        assert world.event_setup(setup).events and world.event_setup(setup).contacts
    finally:
        runner.unlink()
