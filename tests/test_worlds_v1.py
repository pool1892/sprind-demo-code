"""The version-1 world generator: reproducible from the seed, the one rule of
the records, the estimate against the curves, the detectability list, the patterns, the
gate's sizing rule, the health model, and the description without truth."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest

from fleetlab.contracts.faults import Drift, FaultSpec, Schedule
from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.contracts.vocabulary import Component, ConfigKey, FaultClass
from fleetlab.contracts.world import verify_world
from fleetlab.worlds import base, curves, estimate
from fleetlab.worlds.compose import (
    DECOYS,
    INERT,
    MAX_EFFECTIVE,
    compose,
    fault_class,
    make_fault,
)
from fleetlab.worlds.recipe import DEVELOPMENT, INTEGRATION, TASKS, WorldRecipe
from fleetlab.worlds.world import WorldV1, drift_levels, generate, world_truth, write_world

SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)
TEXTS = {Task(t.suite, t.task_index): f"task {t.suite} {t.task_index}" for t in TASKS}


def example_curves() -> curves.Curves:
    """Synthetic flat object decoys for mechanics tests, NOT evidence for a real world.

    Current measured object points are effects. Real development composition must
    refuse them; test_worlds_review exercises that boundary against the real record.
    """
    c = curves.load()
    points = dict(c.points)
    # Freeze these legacy mechanics fixtures to their original measurements;
    # production round-3 recipes have separate real-data seals and tests.
    for key in list(points):
        if (key[0], key[2]) in {
            ("policy_knob", "15 actions per chunk"),
            ("camera_degradation", "3 px blur"),
        }:
            del points[key]
    library = json.loads(resources.files("fleetlab.faults").joinpath("curves.json").read_text())
    legacy = json.loads(
        resources.files("fleetlab.worlds").joinpath("curves_supplement.json").read_text()
    )
    for suite, rows in (
        (curves.SPATIAL, library["types"]["action_latency"]["points"]),
        (curves.OBJECT, [p for p in legacy["points"] if p["kind"] == "action_latency"]),
    ):
        raw = next(p for p in rows if p["label"] == "8 steps")
        points["action_latency", suite, "8 steps"] = curves._point("action_latency", raw, suite)
    for key, point in list(points.items()):
        if point.suite == curves.OBJECT and point.kind in {
            "illumination_change",
            "backdrop_change",
            "light_color_shift",
        }:
            points[key] = replace(
                point,
                n=1000,
                successes=985,
                logged_successes=985,
                healthy_successes=985,
                role="decoy",
                difference=(0, -0.01, 0.01),
            )
    return replace(
        c,
        points=points,
        joints=tuple(
            j for j in c.joints if ("policy_knob", "15 actions per chunk") not in j.members
        ),
    )


def recipes() -> dict[str, WorldRecipe]:
    """CPU mechanics fixtures without the production robot-day floor.

    These are not accepted development worlds or evidence for the frozen design.
    The production calibration and recipe seals have separate regressions.
    """
    d2 = INTEGRATION["D2"].model_copy(update={"calibration_sources": None, "sealed_hash": None})
    d3 = d2.model_copy(
        update={"name": "D3", "world_id": "w45", "seed": INTEGRATION["D3"].seed, "decoys": 1}
    )
    return {"D1": DEVELOPMENT["D1"], "D2": d2, "D3": d3}


@pytest.fixture(scope="module")
def worlds() -> dict[str, WorldV1]:
    return {
        name: generate(recipe, SOFTWARE, TEXTS, example_curves())
        for name, recipe in recipes().items()
    }


# ---- reproducible -----------------------------------------------------------------------------


def test_generation_is_reproducible_from_the_seed(
    worlds: dict[str, WorldV1], tmp_path: Path
) -> None:
    recipe = recipes()["D2"]
    a = write_world(worlds["D2"], tmp_path / "a")
    b = write_world(generate(recipe, SOFTWARE, TEXTS, example_curves()), tmp_path / "b")
    assert a == b and verify_world(tmp_path / "a")
    names = ["spec.json", "model.json", "description.json", "manifest.jsonl", "fault_table.json"]
    names += ["detectability.json", "health.jsonl", "world_truth.json", "generator.json"]
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    # Rebuild the fixed synthetic schedule with a different seed. New compositions
    # use task-v2; the archived suite-v1 recipe cannot be cloned to a new seed.
    from fleetlab.worlds.world import build

    changed = recipe.model_copy(update={"seed": recipe.seed ^ 4242, "estimator": "task-v2"})
    original = worlds["D2"].composition
    other = build(
        replace(
            original, recipe=changed, spec=original.spec.model_copy(update={"seed": changed.seed})
        ),
        SOFTWARE,
        TEXTS,
        example_curves(),
    )
    assert write_world(other, tmp_path / "c") != a


def test_the_three_worlds_differ_and_carry_their_ids(worlds: dict[str, WorldV1]) -> None:
    ids = {w.spec.world_id for w in worlds.values()}
    assert ids == {"w41", "w44", "w45"}
    for world in worlds.values():
        assert all(j.identity.world_id == world.spec.world_id for j in world.jobs)


def test_pair_kind_changes_hidden_truth_only(worlds: dict[str, WorldV1]) -> None:
    from fleetlab.worlds.world import build

    world = worlds["D2"]
    before = world.spec.patterns
    switched = before.model_copy(
        update={
            "interactions": before.additive_pairs,
            "additive_pairs": before.interactions,
        }
    )
    comp = replace(world.composition, spec=world.spec.model_copy(update={"patterns": switched}))
    after = build(comp, SOFTWARE, TEXTS, example_curves())
    assert after.spec.patterns != world.spec.patterns
    assert after.observed_fleet == world.observed_fleet
    assert after.jobs == world.jobs
    assert after.health == world.health


def test_production_recipes_require_the_floor_and_late_linear_drift() -> None:
    for name in ("D2", "D3"):
        recipe = DEVELOPMENT[name]
        assert recipe.robot_success_floor == 0.40 and recipe.warnable_drift
    assert sum(DEVELOPMENT[name].fleet_fault for name in ("D2", "D3")) == 1
    assert DEVELOPMENT["D3"].config_share == 2 / 3


# ---- the episodes -----------------------------------------------------------------------------


def test_episodes_per_robot_and_day_seeds_and_neutral_ids(worlds: dict[str, WorldV1]) -> None:
    world = worlds["D2"]
    spec = world.spec
    per = Counter((j.identity.robot_id, j.identity.day) for j in world.jobs)
    assert set(per.values()) == {spec.episodes_per_robot_day}
    assert len(per) == spec.sites * spec.robots_per_site * spec.days
    ids = [j.identity.episode_id for j in world.jobs]
    assert len(set(ids)) == len(ids)
    assert ids != sorted(ids)  # a seeded permutation: an id says nothing about the episode
    for job in world.jobs:
        job.identity.check_seeds(spec.seed)
        assert job.record_video
    observed = world.jobs_in("observed")
    assert len(observed) == 12 * 6 * spec.episodes_per_robot_day


def test_each_site_has_a_task_emphasis(worlds: dict[str, WorldV1]) -> None:
    world = worlds["D2"]
    marks = {e.site_id: e for e in world.emphasis}
    for site, mark in marks.items():
        jobs = [j for j in world.jobs if j.identity.site_id == site]
        share = sum(j.identity.suite == mark.favoured for j in jobs) / len(jobs)
        assert abs(share - mark.share) < 0.06, (site, share, mark.share)
    assert len({m.favoured for m in marks.values()}) == 2  # both suites are favoured somewhere


def test_faults_in_force_and_the_drift_s_daily_level(worlds: dict[str, WorldV1]) -> None:
    world = worlds["D2"]
    spec = world.spec
    neutral = spec.patterns.excluded_from_replay()
    drift_id = spec.patterns.drifts[0]
    drift = next(f for f in spec.fault_schedule if f.fault_id == drift_id)
    levels = drift_levels(world)[drift_id]
    assert levels[0] == drift.level and max(levels.values()) <= 9.0
    assert all(levels[d + 1] > levels[d] for d in range(spec.days - 1))
    for job in world.jobs:
        ident = job.identity
        expected = base.in_force(spec.fault_schedule, ident.robot_id, ident.site_id, ident.day)
        assert [f.fault_id for f in ident.faults] == [f.fault_id for f in expected]
        for f in ident.faults:
            if f.fault_id == drift_id:
                assert f.level == levels[ident.day]
        assert sum(1 for f in ident.faults if f.fault_id not in neutral) <= MAX_EFFECTIVE
    truth = world_truth(world, ["runs/x"])
    assert truth["fault_levels"][drift_id] == {str(d): v for d, v in levels.items()}


# ---- the records ------------------------------------------------------------------------------


def test_the_one_rule_for_change_entries(worlds: dict[str, WorldV1]) -> None:
    import fleetlab.faults as faults

    for world in worlds.values():
        log = world.fleet.change_log
        schedule = world.spec.fault_schedule
        for fault in schedule:
            writes = faults.get(fault.kind).truth.writes_change
            mine = [
                c
                for c in log
                if c.scope == fault.target.scope
                and c.day == fault.schedule.onset_day
                and fault.target.config_key is not None
                and c.key is fault.target.config_key
            ]
            if writes:
                assert len(mine) == 1, fault.fault_id
                assert mine[0].old == fault.nominal
                assert mine[0].new == faults.get(fault.kind).new_value(fault)  # type: ignore[attr-defined]
            else:
                assert fault.target.config_key is None
        # Nothing in a change marks it as a fault, and benign entries exist on top.
        text = json.dumps([c.model_dump(mode="json") for c in log])
        for fault in schedule:
            assert fault.kind not in text and fault.fault_id not in text
        assert "fault" not in text
        written = {(c.day, c.scope, c.key) for c in log}
        by_faults = {
            (f.schedule.onset_day, f.target.scope, f.target.config_key)
            for f in schedule
            if f.target.config_key is not None
        }
        benign = written - by_faults
        assert len(benign) >= 3
        # A benign change never touches a key that acts: only versions at fleet scope and
        # the cameras' believed poses.
        for _day, scope, key in benign:
            assert (key is ConfigKey.software_version and scope == "fleet") or key in (
                ConfigKey.camera_scene_calibration,
                ConfigKey.camera_wrist_calibration,
            )
        # A camera with a pose fault gets no calibration record.
        posed = {
            (f.target.scope, f.target.component.value)
            for f in schedule
            if f.kind == "camera_pose_offset" and f.target.component is not None
        }
        for c in log:
            if c.key is ConfigKey.camera_scene_calibration:
                assert (c.scope, "scene_camera") not in posed
            if c.key is ConfigKey.camera_wrist_calibration:
                assert (c.scope, "wrist_camera") not in posed


def test_maintenance_never_fixes_a_fault(worlds: dict[str, WorldV1]) -> None:
    import fleetlab.faults as faults

    truth = faults.REGISTRY.truth_table()
    for world in worlds.values():
        assert world.fleet.maintenance_log  # benign entries exist
        for entry in world.fleet.maintenance_log:
            for fault in world.spec.fault_schedule:
                if (
                    fault.target.scope == entry.robot_id
                    and fault.target.component is entry.component
                ):
                    assert entry.verb not in truth[fault.kind].fixing_verbs


def test_the_description_holds_no_truth(worlds: dict[str, WorldV1], tmp_path: Path) -> None:
    world = worlds["D3"]
    write_world(world, tmp_path)
    text = (tmp_path / "description.json").read_text()
    for name in base.TRAIT_NAMES:
        assert name not in text
    for fault in world.spec.fault_schedule:
        assert f'"{fault.fault_id}"' not in text
        assert fault.kind not in text
    days = {c["day"] for c in json.loads(text)["fleet"]["change_log"]}
    assert max(days) <= world.spec.windows.observed.last
    model = json.loads((tmp_path / "model.json").read_text())
    assert all(r["traits"] for s in model["sites"] for r in s["robots"])


# ---- the health model -------------------------------------------------------------------------


def test_health_readings_carry_faults_noise_and_benign_drift(worlds: dict[str, WorldV1]) -> None:
    world = worlds["D2"]
    rows = world.health
    assert len(rows) == 12 * 6 * world.spec.days
    drift = next(f for f in world.spec.fault_schedule if f.fault_id in world.spec.patterns.drifts)

    def series(robot: str, component: str, metric: str) -> list[float]:
        return [
            r["value"]
            for r in rows
            if (r["robot_id"], r["component"], r["metric"]) == (robot, component, metric)
        ]

    camera = series(drift.target.scope, "scene_camera", "calibration_residual")
    # The turning camera shows through the noise (20 px a day): its last three days stand
    # above its first three and above the other robots' scene cameras on the last day.
    assert sum(camera[-3:]) / 3 > sum(camera[:3]) / 3 + 10
    last = [
        r["value"]
        for r in rows
        if r["day"] == world.spec.days - 1
        and (r["component"], r["metric"]) == ("scene_camera", "calibration_residual")
    ]
    assert camera[-1] > sorted(last)[len(last) // 2]
    worn = [f for f in world.spec.fault_schedule if f.kind == "gripper_wear"]
    healthy_force = [
        r["value"]
        for r in rows
        if r["metric"] == "grip_force"
        and r["day"] == world.spec.days - 1
        and r["robot_id"] not in {f.target.scope for f in worn}
    ]
    for f in worn:
        force = series(f.target.scope, "gripper", "grip_force")
        assert force[-1] < sum(healthy_force) / len(healthy_force)
    # Healthy robots vary from day to day too: a reading is a clue, not a label.
    other = series(
        "r12" if drift.target.scope != "r12" else "r11", "scene_camera", "calibration_residual"
    )
    assert len(set(other)) == len(other)


def test_configuration_and_site_faults_never_appear_in_health() -> None:
    from fleetlab.worlds import health

    robots = base.all_robots(1, 2)
    delay = make_fault("f01", "action_latency", "17 steps", "r01", 0)
    light = make_fault("f02", "illumination_change", "2 stops dimmer", "s1", 0)
    plain = health.rows(5, robots, 3, 14, [], [], [])
    assert health.rows(5, robots, 3, 14, [delay, light], [], []) == plain


# ---- the estimate against the curves ----------------------------------------------------------


def test_the_estimate_follows_the_curves() -> None:
    c = curves.load()
    camera = make_fault("f01", "camera_pose_offset", "6 deg scene pan", "r01", 0)
    p = curves.load().points[("camera_pose_offset", curves.SPATIAL, "6 deg scene pan")]
    ctx = estimate.Context(curves=c, neutral=frozenset(), small=frozenset())
    healthy = c.healthy[curves.SPATIAL]
    assert estimate.probability([], curves.SPATIAL, ctx) == pytest.approx(healthy)
    assert estimate.probability([camera], curves.SPATIAL, ctx) == pytest.approx(
        healthy * p.successes / p.healthy_successes
    )
    # Two faults multiply; removing one gives back what the other leaves.
    delay = make_fault("f02", "action_latency", "17 steps", "r01", 0)
    q = c.points[("action_latency", curves.SPATIAL, "17 steps")]
    both = estimate.probability([camera, delay], curves.SPATIAL, ctx)
    assert both == pytest.approx(healthy * p.ratio * q.ratio)
    eps = [estimate.Episode(curves.SPATIAL, (camera, delay)), estimate.Episode(curves.SPATIAL, ())]
    gain = estimate.removal_effect(["f01"], eps, ctx, logged=False)
    assert gain == pytest.approx(100 * (healthy * q.ratio - both) / 2)
    # A data fault changes the logged outcome only.
    detector = make_fault("f03", "success_detector_error", "0.4 missed", "r01", 0)
    d = c.points[("success_detector_error", curves.SPATIAL, "0.4 missed")]
    assert estimate.probability([detector], curves.SPATIAL, ctx) == pytest.approx(healthy)
    assert estimate.probability([detector], curves.SPATIAL, ctx, logged=True) == pytest.approx(
        healthy * d.logged_ratio
    )
    # Decoys and inert entries count as nothing; a small cause never as a gain.
    decoy = make_fault("f04", *DECOYS[0], "s1", 0)
    inert = make_fault("f05", *INERT, "r01", 0)
    ctx2 = estimate.Context(curves=c, neutral=frozenset({"f04", "f05"}), small=frozenset())
    assert estimate.probability([decoy, inert], curves.SPATIAL, ctx2) == pytest.approx(healthy)
    small = estimate.Context(curves=c, neutral=frozenset(), small=frozenset({"f05"}))
    assert estimate.probability([inert], curves.SPATIAL, small) == pytest.approx(healthy)


def test_a_drift_level_is_interpolated_between_calibrated_points() -> None:
    c = curves.load()
    drifting = make_fault(
        "f01",
        "camera_pose_offset",
        "3 deg scene pan",
        "r01",
        0,
        drift=Drift(shape="linear", rate=1.0),
        level=1.5,
    )
    at = {d: drifting.model_copy(update={"level": base.day_level(drifting, d)}) for d in range(8)}
    assert [at[d].level for d in (0, 3, 6)] == [1.5, 4.5, 7.5]
    p3 = c.points[("camera_pose_offset", curves.SPATIAL, "3 deg scene pan")].ratio
    p6 = c.points[("camera_pose_offset", curves.SPATIAL, "6 deg scene pan")].ratio
    assert c.ratio(at[0], curves.SPATIAL).true == pytest.approx(1 - 0.5 * (1 - p3))
    assert c.ratio(at[3], curves.SPATIAL).true == pytest.approx((p3 + p6) / 2)
    with pytest.raises(KeyError, match="above the highest"):
        c.ratio(drifting.model_copy(update={"level": 9.5}), curves.SPATIAL)


def test_a_release_is_valued_as_what_it_carries() -> None:
    c = curves.load()
    release = make_fault(
        "f01",
        "software_release",
        "action_latency|17 steps",
        "s1",
        2,
        nominal_version="4.2.1",
        version="4.3.0",
    )
    delay = make_fault("f02", "action_latency", "17 steps", "s1", 2)
    assert c.ratio(release, curves.SPATIAL).true == c.ratio(delay, curves.SPATIAL).true


# ---- patterns, detectability and the sizing rule ----------------------------------------------


def test_the_patterns_of_each_world(worlds: dict[str, WorldV1]) -> None:
    d1, d2, d3 = worlds["D1"].spec, worlds["D2"].spec, worlds["D3"].spec
    assert (len(d1.patterns.confounders), len(d1.patterns.decoys)) == (0, 1)
    assert not d1.patterns.interactions and not d1.patterns.drifts
    assert len(d2.patterns.decoys) == 2 and len(d3.patterns.decoys) == 1
    for spec in (d2, d3):
        by_id = {f.fault_id: f for f in spec.fault_schedule}
        ((effective, inert),) = spec.patterns.confounders
        assert by_id[effective].schedule.onset_day == by_id[inert].schedule.onset_day
        assert by_id[inert].kind in {"policy_knob", "calibration_update"}
        assert by_id[inert].target.scope == by_id[effective].target.scope
        assert spec.patterns.interactions
        for a, b in spec.patterns.interactions:
            assert {by_id[a].kind, by_id[b].kind} == {"action_latency", "policy_knob"}
            assert by_id[a].target.scope == by_id[b].target.scope
        assert spec.patterns.additive_pairs
        for a, b in spec.patterns.additive_pairs:
            assert {by_id[a].kind, by_id[b].kind} == {"gripper_wear", "object_batch_variant"}
        (drift,) = spec.patterns.drifts
        assert by_id[drift].schedule.drift is not None
    for spec in (d1, d2, d3):
        by_id = {f.fault_id: f for f in spec.fault_schedule}
        for fid in spec.patterns.decoys:  # decoys only at the confirmed levels
            from fleetlab.worlds.curves import calibrated_label

            assert (by_id[fid].kind, calibrated_label(by_id[fid])) in (
                *DECOYS,
                ("light_color_shift", "1 warmer"),
            )


def test_detectability_and_the_sizing_rule(worlds: dict[str, WorldV1]) -> None:
    for name, world in worlds.items():
        recipe = world.recipe
        table = world.detect
        counted = table.counted(recipe.provisional)
        assert len(table.observed_counted(recipe.provisional)) >= recipe.min_detectable, name
        for cause in counted:
            assert max(cause.effect["observed"], cause.effect["scoring"]) >= 2.0
            assert cause.carriers["observed"] >= 20
            assert cause.confirmed and set(cause.qualified_by) <= {"observed", "scoring"}
            if not recipe.provisional:
                assert not cause.transferred and not cause.assumed
        lo, hi = recipe.aim
        assert lo <= table.window_success["observed"]["logged"] <= hi, name
        neutral = {n["fault_id"] for n in table.neutral}
        assert neutral == set(world.spec.patterns.excluded_from_replay())
        listed = {fid for u in table.units for fid in u.fault_ids} | neutral
        assert listed == {f.fault_id for f in world.spec.fault_schedule}
        # The scorer's table agrees with the list on single faults.
        rows = world.fault_table.by_id()
        for cause in table.units:
            if len(cause.fault_ids) == 1:
                row = rows[cause.fault_ids[0]]
                assert row.detectable == cause.detectable
                assert row.expected_loss_points["observed"] == pytest.approx(
                    cause.effect["observed"], abs=1e-3
                )
    d2 = worlds["D2"].detect
    members = worlds["D2"].spec.patterns.interactions[0]
    assert all(any(u.fault_ids == (fid,) and u.detectable for u in d2.units) for fid in members)
    assert d2.pairs and d2.pairs_superadditive
    interaction = next(p for p in d2.pairs if p["kind"] == "interaction")
    assert interaction["paired"] and interaction["interaction_points"][2] < 0
    additive = next(p for p in d2.pairs if p["kind"] == "additive_pair")
    assert additive["additive"] and not additive["superadditive"]


def test_a_recipe_that_cannot_be_met_is_refused() -> None:
    impossible = recipes()["D1"].model_copy(update={"min_detectable": 9, "attempts": 3})
    with pytest.raises(ValueError, match="no composition"):
        compose(impossible, example_curves())


def test_the_composer_keeps_one_thing_per_fault(worlds: dict[str, WorldV1]) -> None:
    for world in worlds.values():
        seen: set[tuple[str, str]] = set()
        for f in world.spec.fault_schedule:
            if f.kind == "software_release":
                continue
            key = (f.target.scope, str(f.target.thing))
            assert key not in seen, key
            seen.add(key)
        robots_with = {
            f.target.scope for f in world.spec.fault_schedule if f.target.scope.startswith("r")
        }
        assert 12 - len(robots_with) >= 2


def test_fault_classes_cover_the_catalogue(worlds: dict[str, WorldV1]) -> None:
    classes = {fault_class(f.kind) for w in worlds.values() for f in w.spec.fault_schedule}
    assert {
        FaultClass.sensor,
        FaultClass.physical,
        FaultClass.configuration,
        FaultClass.site,
    } <= classes


def test_make_fault_refuses_an_unknown_point() -> None:
    with pytest.raises(KeyError):
        make_fault("f01", "camera_pose_offset", "7 deg scene pan", "r01", 0)
    spec = FaultSpec(
        fault_id="f01",
        kind="camera_pose_offset",
        params={},
        level=4.0,
        target=make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0).target,
        schedule=Schedule(onset_day=0),
    )
    assert curves.calibrated_label(spec) is None


def test_no_accidental_co_onsets(worlds: dict[str, WorldV1]) -> None:
    """Two faults starting on one day at scopes sharing a robot would be an undesigned
    confounder; only the designed pairs may."""
    from fleetlab.worlds.compose import Composer

    for world in worlds.values():
        composer = Composer(world.recipe)
        assert composer.co_onsets(world.spec.fault_schedule, world.spec.patterns) == []


def test_transferred_and_assumed_points_count_only_provisionally() -> None:
    from fleetlab.worlds.detect import Cause

    base_fields = dict(
        unit="f01",
        fault_ids=("f01",),
        kinds=("gripper_wear",),
        roles=("plain",),
        scopes=("r01",),
        classes=("physical",),
        data=False,
        effect={"observed": 3.0, "experiment": 3.0, "scoring": 3.0},
        carriers={"observed": 84, "experiment": 28, "scoring": 28},
        observed_suites={},
        qualified_by=("observed", "scoring"),
        identifiable=True,
        detectable=True,
        confirmed=True,
        transferred=False,
        assumed=False,
        sources=(),
    )
    assert Cause(**base_fields).counts  # type: ignore[arg-type]
    for flag in ("transferred", "assumed"):
        cause = Cause(**{**base_fields, flag: True})  # type: ignore[arg-type]
        assert not cause.counts and cause.counts_provisionally
    weak = Cause(**{**base_fields, "confirmed": False})  # type: ignore[arg-type]
    assert not weak.counts and not weak.counts_provisionally


def test_benign_records_avoid_fault_onsets_and_faulted_cameras(worlds: dict[str, WorldV1]) -> None:
    from fleetlab.contracts.ids import scope_covers

    sites = dict(base.all_robots(3, 4))
    for world in worlds.values():
        schedule = world.spec.fault_schedule
        written = {(f.schedule.onset_day, f.target.scope, f.target.config_key) for f in schedule}
        for change in world.fleet.change_log:
            if (change.day, change.scope, change.key) in written or change.day == 0:
                continue  # a fault's own change, or the day-0 patch
            covered = [r for r in sites if scope_covers(change.scope, r, sites[r])]
            for f in schedule:
                if f.schedule.onset_day == change.day:
                    assert not any(f.target.covers(r, sites[r]) for r in covered), (change, f)
        for entry in world.fleet.maintenance_log:
            for f in schedule:
                if f.schedule.onset_day == entry.day and entry.day > 0:
                    assert not f.target.covers(entry.robot_id, sites[entry.robot_id])
        cameras = {
            (f.target.scope, f.target.component)
            for f in schedule
            if f.kind in ("camera_pose_offset", "camera_degradation")
        }
        for change in world.fleet.change_log:
            if change.key is ConfigKey.camera_scene_calibration:
                assert (change.scope, Component.scene_camera) not in cameras
            if change.key is ConfigKey.camera_wrist_calibration:
                assert (change.scope, Component.wrist_camera) not in cameras


def test_releases_are_numbered_by_onset_and_replace_the_running_version(
    worlds: dict[str, WorldV1],
) -> None:
    for world in worlds.values():
        timeline = base.software_timeline(
            world.spec.seed, world.spec.days, world.spec.fault_schedule
        )
        releases = sorted(
            (f for f in world.spec.fault_schedule if f.kind == "software_release"),
            key=lambda f: (f.schedule.onset_day, f.fault_id),
        )
        versions = [str(f.params["version"]) for f in releases]
        assert versions == sorted(versions, key=lambda v: [int(x) for x in v.split(".")])
        for f in releases:
            assert f.nominal == base.fleet_version_on(timeline, f.schedule.onset_day)


def test_each_world_keeps_reference_robots(worlds: dict[str, WorldV1]) -> None:
    for world in worlds.values():
        neutral = world.spec.patterns.excluded_from_replay() | world.composition.small
        obs = world.spec.windows.observed
        touched = {
            j.identity.robot_id
            for j in world.jobs
            if j.identity.day in obs and any(f.fault_id not in neutral for f in j.identity.faults)
        }
        assert 12 - len(touched) >= world.recipe.min_clean_robots
