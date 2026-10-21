"""Historical suite-v1 placement stays reproducible with the new task restrictions.

Current task-v2 qualification is exercised by test_worlds_task_guards.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_worlds_review import recipe
from test_worlds_v1 import SOFTWARE, TEXTS

from fleetlab.worlds import base, curves, estimate
from fleetlab.worlds.compose import Composer, make_fault
from fleetlab.worlds.recipe import TASKS
from fleetlab.worlds.summary import composition_report
from fleetlab.worlds.world import context, episode_slots, generate


def test_site_restriction_preserves_other_sites_and_has_no_pattern_label() -> None:
    args = (2**62 + 937, 3, 4, 10, 14, TASKS)
    baseline = base.slots(*args)
    restricted = base.slots(*args, (("s1", curves.SPATIAL),))
    assert [s for s in baseline if s.site_id != "s1"] == [
        s for s in restricted if s.site_id != "s1"
    ]
    assert {s.suite for s in restricted if s.site_id == "s1"} == {curves.SPATIAL}
    assert {s.suite for s in restricted if s.site_id != "s1"} == {curves.SPATIAL, curves.OBJECT}
    assert len(restricted) == 1680
    marks = base.emphasis(args[0], 3, TASKS, (("s1", curves.SPATIAL),))
    assert marks[0].favoured == curves.SPATIAL and marks[0].share == 1


@pytest.mark.parametrize(
    "restriction",
    [
        (("s9", curves.SPATIAL),),
        (("s1", "missing"),),
        (("s1", curves.SPATIAL), ("s1", curves.OBJECT)),
    ],
)
def test_invalid_site_restrictions_fail(restriction: tuple[tuple[str, str], ...]) -> None:
    with pytest.raises(ValueError, match="site suite restrictions"):
        recipe(tasks=TASKS, site_suites=restriction)


def test_legacy_spatial_decoys_can_share_the_restricted_site_and_follow_the_manifest() -> None:
    r = recipe(sites=3, tasks=TASKS, site_suites=(("s1", curves.SPATIAL),), decoys=2)
    world = generate(r, SOFTWARE, TEXTS)
    decoys = [f for f in world.spec.fault_schedule if f.fault_id in world.spec.patterns.decoys]
    assert len(decoys) == 2 and {f.target.scope for f in decoys} == {"s1"}
    assert len({f.target.condition_key for f in decoys}) == 2
    assert tuple(Composer(r).slots) == episode_slots(world.jobs)
    carriers = [j.identity for j in world.jobs if j.identity.faults]
    assert carriers and {i.suite for i in carriers} == {curves.SPATIAL}
    ctx = context(world.composition, world.calibration)
    expected = estimate.robot_day_success(world.spec.fault_schedule, episode_slots(world.jobs), ctx)
    assert composition_report(world)["lowest_robot_day"] == min(
        expected, key=lambda x: min(x["true"], x["logged"])
    )
    assert "site_suites" not in world.observed_fleet.model_dump_json()
    from fleetlab.engine.experiments import WorldTruth
    from fleetlab.worlds.world import world_truth

    truth = WorldTruth.model_validate(world_truth(world))
    assert {t.suite for t in truth.tasks_of("r01")} == {curves.SPATIAL}
    assert {t.suite for t in truth.tasks_of("r05")} == {curves.SPATIAL, curves.OBJECT}


def test_new_descriptors_are_not_mistaken_for_measured_points() -> None:
    c = curves.load()
    added = make_fault("f01", "policy_knob", "15 actions per chunk", "r01", 0)
    absent = replace(
        c,
        points={
            key: point
            for key, point in c.points.items()
            if (key[0], key[2]) != ("policy_knob", "15 actions per chunk")
        },
    )
    with pytest.raises(KeyError):
        absent.ratio(added, curves.SPATIAL)
    original = make_fault("f02", "action_latency", "14 steps", "r01", 0)
    assert (
        c.ratio(original, curves.SPATIAL).true
        == c.point("action_latency", "14 steps", curves.SPATIAL)[0].ratio
    )


def test_pair_placement_checks_floor_and_small_singles_on_actual_tasks() -> None:
    c = curves.load()
    pair = (("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk"))
    joints = tuple(
        replace(
            j,
            successes=j.healthy_successes * 0.62,
            single_successes=(round(j.healthy_successes * 0.96), round(j.healthy_successes * 0.96)),
        )
        if set(j.members) == set(pair) and j.suite == curves.SPATIAL
        else j
        for j in c.joints
    )
    synthetic = replace(c, joints=joints)
    r = recipe(
        sites=3,
        tasks=TASKS,
        site_suites=(("s1", curves.SPATIAL),),
        robot_success_floor=0.4,
        interaction_single_loss_points=10,
    )
    comp = Composer(r, synthetic)
    permitted = comp.pair_robot_choices(pair)
    assert set(base.robot_ids(4, 0)) <= set(permitted)
    assert all(
        min(row["true"], row["logged"]) >= 0.4
        for robot in permitted
        for row in estimate.robot_day_success(
            [make_fault("f01", *pair[0], robot, 0), make_fault("f02", *pair[1], robot, 1)],
            [s for s in comp.slots if s.robot_id == robot],
            estimate.Context(synthetic, frozenset(), frozenset(), (("f01", "f02"),)),
        )
    )
    assert Composer(r, c).pair_robot_choices(pair) == []


def test_dose_search_and_settling_cannot_cross_the_robot_floor() -> None:
    from fleetlab.worlds.compose import Draft, Occupancy, Plain

    comp = Composer(recipe(robot_success_floor=0.4, aim=(0.1, 0.2)))
    plain = Plain(
        "camera_pose_offset",
        "r01",
        ("3 deg scene pan", "6 deg scene pan", "9 deg scene pan"),
        0,
        0,
        0,
        "f01",
    )
    draft = Draft(plains=[plain])
    settled = comp.settle(draft, 0)
    assert plain.level == 1
    assert comp.floor_holds(settled, draft)
    assert not comp.permitted_move(draft, plain, "up")
    assert plain.level == 1  # probing must restore the draft
    assert Composer(recipe()).permitted_move(draft, plain, "up")

    comp.draft(0)  # initialise per-draft occupancy counters
    comp.plain_menu = [("control_mode_change", "robot")]
    assert (
        comp._draw_plain(
            base.rng(comp.recipe.seed, "floor-test"),
            Occupancy(comp.robots),
            set(),
            "any",
            "f02",
            lambda: "4.3.0",
            draft=Draft(),
        )
        is None
    )


def test_search_floor_matches_full_robot_day_accounting() -> None:
    from fleetlab.worlds.compose import Draft

    r = recipe(sites=3, tasks=TASKS, site_suites=(("s1", curves.SPATIAL),))
    faults = [
        make_fault(
            "f01",
            "software_release",
            "camera_degradation|4 px blur",
            "s2",
            2,
            nominal_version="2.4.1",
            version="4.3.0",
        ),
        make_fault("f02", "action_latency", "14 steps", "r05", 1),
        make_fault("f03", "success_detector_error", "0.4 missed", "r09", 0),
    ]
    draft = Draft(fixed=faults)
    for floor in (0.4, 0.6, 0.9):
        comp = Composer(r.model_copy(update={"robot_success_floor": floor}))
        for scope in (None, "s1", "r05", "r09"):
            selected = [
                s for s in comp.slots if scope is None or s.site_id == scope or s.robot_id == scope
            ]
            expected = all(
                min(row["true"], row["logged"]) >= floor
                for row in estimate.robot_day_success(faults, selected, comp.context(draft))
            )
            assert comp.floor_holds(faults, draft, scope) == expected


def test_legacy_generated_warnable_drift_starts_positive_and_survives_experiment_selection() -> (
    None
):
    from fleetlab.engine.experiments import WorldTruth
    from fleetlab.worlds.compose import Composition
    from fleetlab.worlds.world import build, world_truth

    comp = Composer(recipe(drifts=1, warnable_drift=True))
    draft = comp.draft(0)
    schedule = comp.schedule(draft)
    assert len(schedule) == 1
    drift = schedule[0]
    assert drift.level > 0
    assert estimate.drift_crossing_day(drift, comp.slots, comp.context(draft)) in (8, 9)
    composition = Composition(
        comp.recipe, comp.spec(schedule, draft.patterns), 0, draft.roles, frozenset(), False, {}
    )
    world = build(composition, SOFTWARE, TEXTS, comp.curves)
    truth = WorldTruth.model_validate(world_truth(world, ["runs/example"]))
    for day in range(10):
        present = truth.faults_on(drift.target.scope, day)
        assert len(present) == 1
        assert present[0].level == base.day_level(drift, day)


def test_interaction_does_not_average_an_ineligible_suite_into_acceptance() -> None:
    c = curves.load()
    pair = (("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk"))
    joints = tuple(
        replace(
            j,
            successes=j.healthy_successes * 0.62,
            single_successes=(
                round(j.healthy_successes * (0.80 if j.suite == curves.OBJECT else 0.96)),
                round(j.healthy_successes * 0.96),
            ),
            interaction=(-0.2, -0.3, -0.1),
        )
        if set(j.members) == set(pair)
        else j
        for j in c.joints
    )
    comp = Composer(
        recipe(
            sites=3,
            tasks=TASKS,
            site_suites=(("s1", curves.SPATIAL),),
            robot_success_floor=0.4,
            interaction_single_loss_points=10,
        ),
        replace(c, joints=joints),
    )
    assert set(comp.pair_robot_choices(pair)) == set(base.robot_ids(4, 0))


def test_mild_configuration_pair_can_be_planted_at_site_scope() -> None:
    c = curves.load()
    pair = (("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk"))
    joints = tuple(
        replace(
            j,
            successes=j.healthy_successes * 0.62,
            single_successes=(round(j.healthy_successes * 0.96), round(j.healthy_successes * 0.96)),
        )
        if set(j.members) == set(pair)
        else j
        for j in c.joints
    )
    comp = Composer(
        recipe(
            interactions=1,
            interaction_scope="site",
            robot_success_floor=0.4,
            interaction_single_loss_points=10,
        ),
        replace(c, joints=joints),
    )
    draft = comp.draft(0)
    assert len(draft.patterns.interactions) == 1
    ids = set(draft.patterns.interactions[0])
    members = [f for f in draft.fixed if f.fault_id in ids]
    assert len(members) == 2
    assert len({f.target.scope for f in members}) == 1
    assert members[0].target.scope in ("s1", "s2")
    assert comp.floor_holds(members, draft)
    assert (
        len({s.robot_id for s in comp.slots if members[0].target.covers(s.robot_id, s.site_id)})
        == 4
    )


def test_fleet_fallback_is_not_effect_evidence_until_measured() -> None:
    from fleetlab.worlds.compose import effect_levels

    c = curves.load()
    if ("camera_degradation", curves.SPATIAL, "3 px blur") not in c.points:
        assert "3 px blur" not in effect_levels(c)["camera_degradation"]
        release = make_fault(
            "f01",
            "software_release",
            "camera_degradation|3 px blur",
            "fleet",
            1,
            nominal_version="2.4.1",
            version="4.3.0",
        )
        assert not c.ratio(release, curves.SPATIAL).effect
        assert c.ratio(release, curves.SPATIAL).interpolated
