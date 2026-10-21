"""The version-0 world generator on the miniature fleet's spec: pure,
sealed by its hash, the one rule of the records, the fault table, the oracle's replays."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.world import verify_world
from fleetlab.integration.oracle import oracle_jobs
from fleetlab.worlds.generate import TRAIT_NAMES, World, generate, write_world
from fleetlab.worlds.mini import TASKS, mini_spec

SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)
TEXTS = {Task(t.suite, t.task_index): f"task {t.suite} {t.task_index}" for t in TASKS}


@pytest.fixture(scope="module")
def world() -> World:
    return generate(mini_spec(), SOFTWARE, TEXTS)


def test_generation_is_pure_and_sealed(world: World, tmp_path: Path) -> None:
    again = generate(mini_spec(), SOFTWARE, TEXTS)
    a = write_world(world, tmp_path / "a")
    b = write_world(again, tmp_path / "b")
    assert a == b
    assert verify_world(tmp_path / "a")
    assert (tmp_path / "a" / "hash").read_text().strip() == a
    other = generate(mini_spec(seed=11), SOFTWARE, TEXTS)
    assert write_world(other, tmp_path / "c") != a


def test_shape_of_the_world(world: World) -> None:
    spec = world.spec
    assert [s.id for s in world.fleet.sites] == ["s1", "s2"]
    assert [r.id for r in world.fleet.robots()] == [f"r0{i}" for i in range(1, 9)]
    assert len(world.jobs) == 8 * 4 * 13
    ids = [j.identity.episode_id for j in world.jobs]
    assert len(set(ids)) == len(ids)
    per_robot_day = Counter((j.identity.robot_id, j.identity.day) for j in world.jobs)
    assert set(per_robot_day.values()) == {13}
    for job in world.jobs:
        job.identity.check_seeds(spec.seed)
        assert job.record_video
    # Episode ids say nothing about the order of the episodes.
    first_day = [j.identity.episode_id for j in world.jobs if j.identity.day == 0]
    assert sorted(first_day) != first_day[: len(first_day)]


def test_faults_by_robot_and_day(world: World) -> None:
    camera = world.spec.fault_schedule[0]
    seen = {(j.identity.robot_id, j.identity.day): j.identity.fault_ids for j in world.jobs}
    for (robot, day), faults in seen.items():
        expected = set()
        if robot == camera.target.scope:
            expected.add("f01")
        if day >= 1:
            expected.add("f03" if robot <= "r04" else "f02")
        assert faults == expected, (robot, day)


def test_records_follow_the_one_rule(world: World) -> None:
    changes = [(c.day, c.scope, c.key.value, c.old, c.new) for c in world.fleet.change_log]
    assert (1, "s2", "control.action_delay_steps", 0, 17) in changes
    assert (1, "s1", "policy.sampling_steps", 10, 5) in changes
    keys = Counter(c.key.value for c in world.fleet.change_log)
    assert keys["software.version"] == 1 and keys["camera.scene.calibration"] == 1
    camera_robot = world.spec.fault_schedule[0].target.scope
    assert all(c.scope != camera_robot for c in world.fleet.change_log)
    assert len(world.fleet.maintenance_log) == 3
    assert world.fleet.site("s2").config.values["control.action_delay_steps"] == 17  # type: ignore[index]
    # The description holds no trait and no fault.
    text = json.dumps(world.observed_fleet.model_dump(mode="json"))
    for name in TRAIT_NAMES:
        assert name not in text
    assert all(r.traits for r in world.fleet.robots())
    assert isinstance(world.observed_fleet, Fleet)


def test_health_moves_with_the_camera_fault_as_a_clue(world: World) -> None:
    camera_robot = world.spec.fault_schedule[0].target.scope
    residual = {
        (row["robot_id"], row["day"]): row["value"]
        for row in world.health
        if row["component"] == "scene_camera"
    }
    faulted = [v for (r, _), v in residual.items() if r == camera_robot]
    healthy = [v for (r, _), v in residual.items() if r != camera_robot]
    assert sum(faulted) / len(faulted) > sum(healthy) / len(healthy) + 20.0
    assert all(v > 0.0 for v in residual.values())  # no exact zeros
    # The day-to-day noise is comparable to a 3 degree offset:
    # the healthy readings spread well beyond the 25 px of their baselines.
    assert max(healthy) - min(healthy) > 40.0


def test_directions_follow_the_class_rule(world: World) -> None:
    by_id = world.fault_table.by_id()
    assert [d.value for d in by_id["f01"].directions] == ["degraded"]  # no drift schedule
    assert [d.value for d in by_id["f02"].directions] == ["changed", "misconfigured"]


def test_fault_table(world: World) -> None:
    table = world.fault_table
    by_id = table.by_id()
    assert [f.fault_id for f in table.detectable()] == ["f01", "f02"]
    assert by_id["f03"].role == "decoy" and not by_id["f03"].detectable
    assert by_id["f01"].episodes == {"observed": 26, "experiment": 13, "scoring": 13}
    assert by_id["f02"].episodes["observed"] == 52
    assert by_id["f02"].expected_loss_points["scoring"] == pytest.approx(22.5)
    assert table.episodes_per_window == {"observed": 208, "experiment": 104, "scoring": 104}


def test_oracle_replays(world: World) -> None:
    plan = oracle_jobs(world)
    assert (plan.contract, plan.confirmation, plan.over_cap) == (65, 65, [])
    removed = Counter((j.identity.robot_id <= "r04", tuple(j.remove)) for j in plan.jobs)
    assert removed[(False, ("f02",))] == 52
    assert removed[(True, ("f03",))] == 52  # every s1 robot, the decoy removed
    assert removed[(True, ("f01",))] == 13 and removed[(True, ("f01", "f03"))] == 13
    assert all(j.identity.day == 3 for j in plan.jobs)
