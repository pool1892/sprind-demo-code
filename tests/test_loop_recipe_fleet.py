"""Story B's lookalike fleets (lane P; `fleetlab.worlds.loop_fleets`,
`STORY_RECIPES`): the remote link at one site, a controller that falls behind with the same
symptom in the records at another, an image release logged on the day of a camera
calibration record at the third; and the later fleet for the second use, the link at another
site and a robot whose camera images arrive late. CPU only: built with stand-in software and
task texts, written into a temporary root and read back by lane state's `LoopWorld`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import SoftwareVersion
from fleetlab.contracts.model import describe_for_agents
from fleetlab.contracts.vocabulary import ConfigKey
from fleetlab.worlds import loop_fleets as lf

SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)
TEXTS = {t: f"task {t.suite} {t.task_index}" for t in lf.TASKS}
NAMES = sorted(lf.STORY_RECIPES)
COMPENSATION = ConfigKey.control_latency_compensation_steps


@cache
def built(name: str) -> lf.LoopFleetWorld:
    return lf.build(lf.STORY_RECIPES[name], SOFTWARE, dict(TEXTS))


def faults_by_handle(world: lf.LoopFleetWorld) -> dict[str, FaultSpec]:
    return {p.handle: world.fault_of(p.handle) for p in world.loop_recipe.planted}


@pytest.mark.parametrize("name", NAMES)
def test_twelve_robots_spatial_tasks_and_a_measured_forecast(name: str) -> None:
    world = built(name)
    assert len(list(world.fleet.robots())) == 12
    assert [s.id for s in world.fleet.sites] == ["s1", "s2", "s3"]
    assert len(world.jobs_in("observed")) == 6 * 12 * 12
    assert {j.identity.suite for j in world.jobs} == {"libero_spatial"}
    last = lf.loop_layout(world).round(1).ops.last
    assert all(f.schedule.onset_day <= last for f in world.spec.fault_schedule)
    assert all(f.schedule.duration_days is None for f in world.spec.fault_schedule)
    # Every part rests on measured cells: no unmeasured or uncalibrated forecast.
    assert set(world.design["sources"]) <= {"joint", "healthy", "product"}
    assert 55.0 <= world.design["success_percent"]["window"]["observed"] <= 85.0
    # Every loop fleet has the compensation setting: recorded, described and offered.
    assert COMPENSATION in world.spec.later_settings
    assert world.fleet.config.values[COMPENSATION] == 0


def test_story_b_plants_the_link_the_lookalike_and_the_same_day_confounder() -> None:
    import fleetlab.faults as faults

    world = built("STORYB")
    by = faults_by_handle(world)
    lasting = [world.fault_of(h) for h in world.loop_recipe.lasting]
    assert {f.target.scope for f in lasting} == {"r01", "r02", "r03", "r04"}
    assert all(f.kind == "remote_link_latency" and f.level == 17.0 for f in lasting)
    assert all(f.schedule.onset_day == 0 for f in lasting)
    assert not faults.REGISTRY.truth_table()["remote_link_latency"].fixing_verbs
    overload = by["f05"]
    assert (overload.kind, overload.level) == ("controller_overload", 0.2)
    assert overload.target.scope == "r06" and overload.target.component is not None
    assert overload.target.component.value == "controller"
    release, record = by["f06"], by["f07"]
    assert release.kind == "image_pipeline_release" and release.level == 20.0
    assert release.params == {"version": "4.3.0", "mode": "crop"}
    assert release.nominal == "4.2.1" and release.target.scope == "s3"
    assert record.kind == "calibration_update" and record.target.scope == "s3"
    notes = {s.id: s.notes for s in world.observed_fleet.sites}
    assert notes["s1"] and not notes["s2"] and not notes["s3"]
    log = world.observed_fleet.change_log
    s3_day2 = [c for c in log if c.scope == "s3" and c.day == 2]
    assert {c.key.value for c in s3_day2} == {"software.version", "camera.scene.calibration"}
    versions = [(c.old, c.new) for c in s3_day2 if c.key.value == "software.version"]
    assert versions == [("4.2.1", "4.3.0")]
    # The overloaded controller writes nothing: no change, no maintenance, at r06.
    assert not [c for c in log if c.scope == "r06"]
    assert not [m for m in world.fleet.maintenance_log if m.robot_id == "r06"]
    roles = {f.fault_id: f.role for f in world.fault_table.faults}
    assert roles[world.handles["f07"]] == "inert" and roles[world.handles["f05"]] == "effective"


def test_the_later_fleet_moves_the_link_and_adds_the_late_camera() -> None:
    world = built("STORYB-LATER")
    lasting = {world.fault_of(h).target.scope for h in world.loop_recipe.lasting}
    assert lasting == {"r09", "r10", "r11", "r12"}
    notes = {s.id: s.notes for s in world.observed_fleet.sites}
    assert notes["s3"] and not notes["s1"] and not notes["s2"]
    camera = world.fault_of("f05")
    assert (camera.kind, camera.level, camera.target.scope) == ("camera_latency", 24.0, "r06")
    assert camera.target.component is not None and camera.target.component.value == "scene_camera"
    assert camera.schedule.onset_day == 2
    assert not [c for c in world.observed_fleet.change_log if c.scope == "r06"]


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


@pytest.mark.parametrize("name", NAMES)
def test_the_written_world_loads_and_a_probe_runs_on_the_robot_as_it_is(
    name: str, tmp_path: Path
) -> None:
    from fleetlab.contracts.probes import ProbeRequest, QueuedProbe
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.probes import plan_probes
    from fleetlab.loop.state import Ledger
    from fleetlab.loop.world import LoopWorld

    world = built(name)
    wid = world.spec.world_id
    root = tmp_path / wid
    value = lf.write_world(world, root / "files", [str(root / "runs" / f"{wid}-observed")])
    assert value == lf.world_hash(world)
    lf.write_beside(world, root)
    loaded = LoopWorld.load(root)
    layout = layout_for_world(loaded.spec)
    empty = Ledger.empty(loaded.world_id)
    kind = {"STORYB": "controller_overload", "STORYB-LATER": "camera_latency"}[name]
    day = layout.round(1).experiment.first
    assert [f.kind for f in loaded.state_on("r06", day, empty).faults] == [kind]
    assert not loaded.state_on("r05", day, empty).faults
    request = ProbeRequest.model_validate(
        {
            "name": "step-lag-x",
            "purpose": "how the hand follows a commanded step",
            "expectation": "the hand moves in the next row",
            "robots": ["r06", "r05"],
            "repeats": 1,
            "segments": [
                {
                    "primitive": "step",
                    "axis": "x",
                    "amplitude_m": 0.03,
                    "duration_steps": 20,
                    "period_steps": None,
                },
            ],
        }
    )
    planned = plan_probes(
        loaded,
        layout,
        1,
        empty,
        [QueuedProbe(probe_id="p0001", session=1, request=request)],
        {},
        software=SOFTWARE,
    )
    assert [o.status for o in planned.outcomes] == ["done"]
    faults = {j.identity.robot_id: [f.kind for f in j.identity.faults] for j in planned.jobs}
    assert faults == {"r06": [kind], "r05": []}


def test_the_demonstration_draws_move_the_parts_and_keep_the_link_apart() -> None:
    lead = lf.storyb_lead(4242, "w82", "w10")
    assert lead.world_id == "w82" and lead.public_fleet_id == "w10" and lead.seed == 4242
    template = lf.STORY_RECIPES["STORYB"]
    assert [(p.kind, p.label, p.onset) for p in lead.planted] == [
        (p.kind, p.label, p.onset) for p in template.planted
    ]
    lead_site = lf.condition_site(lead)
    for seed in (1, 2, 3, 4, 5, 6):
        later = lf.storyb_later(seed, "w83", "w11", lead_site)
        assert lf.condition_site(later) != lead_site
        assert lf.compose(later).spec.fault_schedule
    drawn = lf.build(lead, SOFTWARE, dict(TEXTS))
    assert len(drawn.spec.fault_schedule) == len(template.planted)
