"""The loop fleets (fleetlab.worlds.loop_fleets): each fleet's design holds as built.

CPU only: the fleets are built with stand-in software and task texts, written into a
temporary root, and read back by lane state's `LoopWorld`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest

from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.contracts.model import describe_for_agents
from fleetlab.worlds import loop_fleets as lf

SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)
TEXTS = {t: f"task {t.suite} {t.task_index}" for t in lf.TASKS}
NAMES = sorted(lf.RECIPES)


@cache
def built(name: str) -> lf.LoopFleetWorld:
    return lf.build(lf.RECIPES[name], SOFTWARE, dict(TEXTS))


def lead() -> lf.LoopFleetWorld:
    return built("LEAD")


def _by_handle(world: lf.LoopFleetWorld) -> dict[str, lf.Planted]:
    return {p.handle: p for p in world.loop_recipe.planted}


@pytest.mark.parametrize("name", NAMES)
def test_twelve_robots_at_three_sites_in_the_loop_layout(name: str) -> None:
    world = built(name)
    assert len(list(world.fleet.robots())) == 12
    assert [s.id for s in world.fleet.sites] == ["s1", "s2", "s3"]
    layout = lf.loop_layout(world)
    assert (layout.round(1).ops.first, layout.round(1).ops.last) == (0, 5)
    assert (layout.round(2).ops.first, layout.round(2).ops.last) == (8, 11)
    assert layout.evaluation.first > layout.round(2).experiment.last
    assert len(world.jobs_in("observed")) == 6 * 12 * 12


@pytest.mark.parametrize("name", NAMES)
def test_nothing_starts_after_round_one_operation(name: str) -> None:
    world = built(name)
    last = lf.loop_layout(world).round(1).ops.last
    assert all(f.schedule.onset_day <= last for f in world.spec.fault_schedule)
    assert all(f.schedule.duration_days is None for f in world.spec.fault_schedule)


@pytest.mark.parametrize("name", NAMES)
def test_a_start_in_band(name: str) -> None:
    forecast = built(name).design["success_percent"]["window"]
    assert 60.0 <= forecast["observed"] <= 75.0
    assert built(name).design["sources"].get("unmeasured", 0) == 0


@pytest.mark.parametrize("name", NAMES)
def test_no_identity_or_fault_name_in_the_visible_description(name: str) -> None:
    import fleetlab.faults as faults

    world = built(name)
    text = json.dumps(describe_for_agents(world.observed_fleet, True, world.task_classes))
    for fault in world.spec.fault_schedule:
        assert fault.fault_id not in text
    for kind in faults.REGISTRY.truth_table():
        assert kind not in text
    assert world.spec.world_id not in text


def test_the_lasting_condition_covers_one_site_from_day_zero_and_no_verb_fixes_it() -> None:
    import fleetlab.faults as faults

    world = lead()
    lasting = [world.fault_of(h) for h in world.loop_recipe.lasting]
    assert {f.target.scope for f in lasting} == {"r01", "r02", "r03", "r04"}
    assert all(f.kind == "remote_link_latency" and f.schedule.onset_day == 0 for f in lasting)
    assert not faults.REGISTRY.truth_table()["remote_link_latency"].fixing_verbs
    notes = {s.id: s.notes for s in world.observed_fleet.sites}
    assert notes["s1"] and not notes["s2"] and not notes["s3"]


def test_the_records_show_the_coincidence_and_the_decoy_but_no_physical_cause() -> None:
    world = lead()
    log = world.observed_fleet.change_log
    s2_day1 = {c.key.value for c in log if c.scope == "s2" and c.day == 1}
    assert s2_day1 == {"software.version", "camera.scene.calibration"}
    assert [c.new for c in log if c.scope == "s3" and c.key.value == "software.version"] == [
        "4.3.1"
    ]
    assert any(c.scope == "r06" and c.key.value == "control.action_delay_steps" for c in log)
    # The unlogged causes write nothing. On s3's day 2 the only record is the harmless
    # release (the decoy shares the day with the tables moving); r03's pads and r10's camera
    # leave no change at their robots, and no maintenance entry fixes either.
    planted = _by_handle(world)
    s3_day2 = [c for c in log if c.scope == "s3" and c.day == planted["f08"].onset]
    assert [(c.key.value, c.new) for c in s3_day2] == [("software.version", "4.3.1")]
    for handle in ("f09", "f11"):
        scope = planted[handle].scope
        assert not [c for c in log if c.scope == scope and c.day == planted[handle].onset]
    fixes = {("r03", "gripper"), ("r10", "scene_camera")}
    assert not [
        m
        for m in world.fleet.maintenance_log
        if (m.robot_id, m.component.value) in fixes
        and m.verb.value in ("replace", "service", "recalibrate")
    ]


def test_patterns_and_roles() -> None:
    world = lead()
    roles = {f.fault_id: f.role for f in world.fault_table.faults}
    effective, inert = world.spec.patterns.confounders[0]
    assert roles[effective] == "effective" and roles[inert] == "inert"
    (decoy,) = world.spec.patterns.decoys
    assert roles[decoy] == "decoy"
    assert sum(1 for r in roles.values() if r == "effective") == 9


def test_more_worth_doing_than_three_changes() -> None:
    world = lead()
    losses = world.design["loss_points_if_removed_alone"]
    planted = _by_handle(world)
    # Fixable causes (lasting faults aside) with a forecast loss in the scoring window.
    fixable = [
        h for h, loss in losses.items() if planted[h].part != "lasting" and loss["scoring"] >= 0.5
    ]
    # Five repairs and the adaptation at the lasting site: six changes' worth, more than
    # the three a round may make.
    assert len(fixable) == 5
    assert len(fixable) + 1 > 3


def test_the_later_fleet_differs_where_a_remembered_answer_would_matter() -> None:
    lead_world, later = lead(), built("LATER")

    def lasting(world: lf.LoopFleetWorld) -> tuple[set[str], set[float], set[str]]:
        faults = [world.fault_of(h) for h in world.loop_recipe.lasting]
        sites = {world.fleet.robot(f.target.scope).site_id for f in faults}
        suites = {dict(world.loop_recipe.site_suites)[s] for s in sites}
        return sites, {f.level for f in faults}, suites

    assert lasting(lead_world) == ({"s1"}, {17.0}, {"libero_spatial"})
    assert lasting(later) == ({"s3"}, {14.0}, {"libero_object"})
    # Two sites carry the remote-link note in the later fleet; only one has a loss to adapt.
    notes = {s.id: bool(s.notes) for s in later.observed_fleet.sites}
    assert notes == {"s1": True, "s2": False, "s3": True}
    losses = later.design["loss_points_if_removed_alone"]
    assert all(abs(losses[h]["observed"]) < 0.5 for h in ("f01", "f02", "f03", "f04"))
    # The other faults are other kinds than the lead fleet's own causes.
    lead_kinds = {p.kind for p in lead_world.loop_recipe.planted if p.part != "lasting"} - {
        "software_release"
    }
    later_kinds = {p.kind for p in later.loop_recipe.planted if "lasting" not in p.part}
    assert not (later_kinds - {"software_release"}) & lead_kinds


@pytest.mark.parametrize("name", NAMES)
def test_written_world_loads_in_the_loop(name: str, tmp_path: Path) -> None:
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.state import Ledger
    from fleetlab.loop.world import LoopWorld

    world = built(name)
    wid = world.spec.world_id
    root = tmp_path / wid
    value = lf.write_world(world, root / "files", [str(root / "runs" / f"{wid}-observed")])
    assert value == lf.world_hash(world)
    lf.write_beside(world, root)
    # LoopWorld wants the stored runs only for round 1's operation; the files alone load.
    loaded = LoopWorld.load(root)
    assert loaded.generator_version == "v1"
    assert layout_for_world(loaded.spec) == lf.loop_layout(world)
    empty = Ledger.empty(loaded.world_id)
    # Health under the empty ledger is the stored health, and the panel past the horizon
    # keeps every lasting fault in force at its site only.
    rows = loaded.health_rows([8, 9, 10, 11], empty)
    assert rows == [r for r in world.health if r["day"] in (8, 9, 10, 11)]
    panel = loaded.panel_identities([14], SOFTWARE)
    assert len(panel) == 12 * 12
    sites = {
        world.fleet.robot(world.fault_of(h).target.scope).site_id for h in world.loop_recipe.lasting
    }
    assert all(
        any(f.kind == "remote_link_latency" and f.level > 8 for f in i.faults)
        == (i.site_id in sites)
        for i in panel
    )
    assert json.loads((root / "loop_layout.json").read_text())["world_days"] == 12
    if name == "LEAD":
        state = loaded.state_on("r03", 9, empty)
        assert sorted(f.kind for f in state.faults) == ["gripper_wear", "remote_link_latency"]


@pytest.mark.parametrize("name", NAMES)
def test_recipes_compose(name: str) -> None:
    composed = lf.compose(lf.RECIPES[name])
    assert composed.spec.fault_schedule
    assert set(composed.parts) == {p.handle for p in lf.RECIPES[name].planted}


def test_tasks_are_the_two_suites() -> None:
    assert {t.suite for t in lf.TASKS} == {"libero_spatial", "libero_object"}
    assert Task("libero_spatial", 0) in lf.TASKS


def test_measured_tallies_a_run_by_site_robot_and_day(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    world = lead()
    jobs = world.jobs_in("observed")[:6]
    rows = [
        {
            "episode_id": j.identity.episode_id,
            "robot_id": j.identity.robot_id,
            "site_id": j.identity.site_id,
            "day": j.identity.day,
            "outcome": "success" if n % 2 == 0 else "failure",
        }
        for n, j in enumerate(jobs)
    ]
    (tmp_path / "log").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "log" / "episodes.parquet")
    out = lf.measured(tmp_path)
    assert out["fleet"]["all"] == {"successes": 3, "attempts": 6, "percent": 50.0}
    assert sum(c["attempts"] for c in out["robot_day"].values()) == 6
    by_suite = lf.suites_measured(tmp_path, world)
    assert sum(c["attempts"] for c in by_suite.values()) == 6
