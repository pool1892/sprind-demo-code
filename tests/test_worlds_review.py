"""Independent rule boundaries from an independent worlds-v2 review (CPU only).

Synthetic probabilities isolate cause accounting; they are not calibration evidence.
No test creates or runs a production test world.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_worlds_v1 import SOFTWARE, TEXTS, example_curves, recipes

from fleetlab.contracts.faults import Drift
from fleetlab.contracts.identity import Task
from fleetlab.contracts.world import Patterns
from fleetlab.worlds import base, curves, detect, estimate
from fleetlab.worlds.compose import (
    ADDITIVE_PAIR,
    INTERACTION_PAIRS,
    Composer,
    Draft,
    Occupancy,
    calibration_value,
    compose,
    make_fault,
    things_of,
)
from fleetlab.worlds.recipe import (
    DEVELOPMENT,
    FOURTH_GENERATION_JOBS,
    INTEGRATION,
    SMALL_WINDOWS,
    WorldRecipe,
    external_recipe,
    stated_fields,
)
from fleetlab.worlds.world import changes, generate, world_hash, write_world


def recipe(**updates: Any) -> WorldRecipe:
    # These existing rule-isolation fixtures exercise the historical suite-v1
    # estimator. Future-world guarantees use task_recipe in test_worlds_task_guards.
    values = dict(
        name="CPU fixture",
        world_id="w44",
        seed=5540438229502422191,
        estimator="suite-v1",
        sites=2,
        robots_per_site=4,
        tasks=(Task(curves.SPATIAL, 0),),
        band=(0.0001, 0.9999),
        aim=(0.001, 0.999),
        scoring_floor=0.0,
        plain=0,
        small=0,
        min_clean_robots=0,
        min_detectable=0,
        margin=0,
        confounders=0,
        decoys=0,
        interactions=0,
        drifts=0,
    )
    return WorldRecipe.model_validate({**values, **updates})


def check(
    faults: list[Any],
    patterns: Patterns | None = None,
    *,
    use: curves.Curves | None = None,
    **updates: Any,
) -> list[str]:
    composer = Composer(recipe(**updates), use)
    return composer.check(Draft(fixed=faults, patterns=patterns or Patterns()), faults)[0]


def paired_table(
    a_ratio: int, b_ratio: int, joint_ratio: int, *, pattern: str = "interactions", n: int = 20
) -> detect.DetectTable:
    c = curves.load()
    a_key, b_key = INTERACTION_PAIRS[0]
    points = dict(c.points)
    for (kind, label), success in ((a_key, a_ratio), (b_key, b_ratio)):
        key = kind, curves.SPATIAL, label
        points[key] = replace(
            points[key],
            n=1000,
            healthy_successes=1000,
            successes=success,
            logged_successes=success,
            role="effect",
        )
    joint, _ = c.joint(a_key, b_key, curves.SPATIAL)
    c = replace(
        c,
        points=points,
        joints=(
            replace(
                joint,
                n=1000,
                healthy_successes=1000,
                successes=joint_ratio,
                single_successes=(a_ratio, b_ratio),
            ),
        ),
    )
    faults = [make_fault("f01", *a_key, "r01", 0), make_fault("f02", *b_key, "r01", 0)]
    eps = [estimate.Episode(curves.SPATIAL, tuple(faults))] * n
    by_window = dict.fromkeys(detect.WINDOWS, eps)
    patterns = Patterns.model_validate({pattern: [("f01", "f02")]})
    ctx = estimate.Context(c, frozenset(), frozenset(), (("f01", "f02"),))
    return detect.detectability(
        faults, patterns, set(), [], SMALL_WINDOWS, ctx, by_window=by_window
    )


@pytest.mark.parametrize(
    ("a", "b", "joint", "expected", "small"),
    [
        (900, 900, 500, {"f01", "f02"}, set()),
        (710, 950, 700, {"f01"}, {"f02"}),
        (990, 990, 974, {"f01+f02"}, {"f01", "f02"}),
        (995, 995, 990, set(), {"f01", "f02"}),
    ],
)
def test_causes_remove_each_member_in_the_composed_world(
    a: int, b: int, joint: int, expected: set[str], small: set[str]
) -> None:
    table = paired_table(a, b, joint)
    assert {u.unit for u in table.counted()} == expected
    assert {u.unit for u in table.units if not u.detectable} == small
    singles = {u.unit: u for u in table.units if len(u.fault_ids) == 1}
    healthy = curves.load().healthy[curves.SPATIAL]
    assert singles["f01"].effect["observed"] == pytest.approx(healthy * (b - joint) / 10)
    assert singles["f02"].effect["observed"] == pytest.approx(healthy * (a - joint) / 10)


def test_additive_pairs_never_become_joint_causes() -> None:
    table = paired_table(990, 990, 974, pattern="additive_pairs")
    assert not table.counted()
    assert {u.unit for u in table.units} == {"f01", "f02"}


def test_twenty_observed_carriers_is_a_real_boundary() -> None:
    assert len(paired_table(900, 900, 500, n=20).counted()) == 2
    table = paired_table(900, 900, 500, n=19)
    assert not table.counted() and all(not u.detectable for u in table.units)


def test_overlapping_pairs_are_not_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    faults = [make_fault(f"f0{i}", "action_latency", "14 steps", f"r0{i}", 0) for i in (1, 2, 3)]
    monkeypatch.setattr(
        estimate,
        "removal_effect",
        lambda removed, *a, **kw: 3.0 if len(list(removed)) == 2 else 0.5,
    )
    monkeypatch.setattr(detect, "_support", lambda *a, **kw: (True, False, False, ()))
    eps = [estimate.Episode(curves.SPATIAL, tuple(faults))] * 20
    ctx = estimate.Context(curves.load(), frozenset(), frozenset())
    # No joint carriers in scoring: this test isolates accounting, not pair qualification.
    table = detect.detectability(
        faults,
        Patterns(interactions=[("f01", "f02"), ("f02", "f03")]),
        set(),
        [],
        SMALL_WINDOWS,
        ctx,
        by_window={"observed": eps, "experiment": [], "scoring": []},
    )
    assert {u.unit for u in table.counted()} == {"f01+f02", "f02+f03"}
    assert all(len(u.fault_ids) <= 2 for u in table.units)


def test_observed_sizing_is_distinct_from_either_window_detectability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = paired_table(900, 900, 500).units[0]
    scoring = replace(
        template,
        qualified_by=("scoring",),
        confirmed_by=("scoring",),
        effect={"observed": 1.0, "experiment": 2.0, "scoring": 3.0},
    )
    table = detect.DetectTable([replace(scoring, unit=f"f{i:02}") for i in range(1, 7)], [], [], {})
    assert len(table.counted()) == 6 and not table.observed_counted()
    monkeypatch.setattr(detect, "detectability", lambda *a, **kw: table)
    assert any("0 observed detectable causes, 6 wanted" in p for p in check([], min_detectable=6))
    observed = replace(template, qualified_by=("observed",), confirmed_by=("observed",))
    table.units[:] = [replace(observed, unit=f"f{i:02}") for i in range(1, 7)]
    assert not check([], min_detectable=6)
    assert any("6 observed detectable causes, 7 wanted" in p for p in check([], min_detectable=7))


@pytest.mark.parametrize("kind", ["interaction", "additive"])
def test_composer_requires_both_pairs_paired_support(kind: str) -> None:
    c = curves.load()
    keys = INTERACTION_PAIRS[0] if kind == "interaction" else ADDITIVE_PAIR
    a = make_fault("f01", *keys[0], "r01", 0)
    b = make_fault("f02", *keys[1], "r01" if kind == "interaction" else "s1", 0)
    patterns = Patterns.model_validate(
        {"interactions" if kind == "interaction" else "additive_pairs": [("f01", "f02")]}
    )
    assert not check([a, b], patterns, use=c)
    damaged = replace(
        c,
        joints=tuple(
            replace(j, single_successes=None, interaction=None)
            if set(j.members) == set(keys)
            else j
            for j in c.joints
        ),
    )
    problems = check([a, b], patterns, use=damaged)
    assert any("paired" in p for p in problems)


def test_confounder_and_legacy_drift_must_be_detectable() -> None:
    weak = make_fault("f01", "gripper_wear", "0.25 wear", "r01", 0)
    inert = make_fault("f02", "policy_knob", "5 sampling steps", "r01", 0)
    assert "the confounder's effective fault is not detectable" in check(
        [weak, inert], Patterns(confounders=[("f01", "f02")])
    )
    drift = make_fault(
        "f01",
        "camera_pose_offset",
        "3 deg scene pan",
        "r01",
        0,
        level=0.0,
        drift=Drift(shape="linear", rate=0.001),
    )
    assert "the drift is not detectable" in check([drift], Patterns(drifts=["f01"]))


def test_four_fault_replay_cap() -> None:
    faults = [
        make_fault(f"f{i:02}", "camera_pose_offset", "3 deg scene pan", "r01", 0)
        for i in range(1, 6)
    ]
    assert not any("effective faults" in p for p in check(faults[:4]))
    assert "an episode carries 5 effective faults" in check(faults)


def test_transferred_curves_cannot_count() -> None:
    c = curves.load()
    c = replace(c, points={k: v for k, v in c.points.items() if k[1] != curves.OBJECT})
    fault = make_fault("f01", "camera_pose_offset", "6 deg scene pan", "r01", 0)
    eps = [estimate.Episode(curves.OBJECT, (fault,))] * 20
    ctx = estimate.Context(c, frozenset(), frozenset())
    table = detect.detectability(
        [fault],
        Patterns(),
        set(),
        [],
        SMALL_WINDOWS,
        ctx,
        by_window=dict.fromkeys(detect.WINDOWS, eps),
    )
    assert table.units[0].transferred and table.units[0].detectable
    assert not table.counted() and len(table.counted(True)) == 1


def test_fleet_fault_requires_measured_effect_on_all_its_suites() -> None:
    inert = make_fault("f01", "policy_knob", "5 sampling steps", "fleet", 0)
    tasks = (Task(curves.SPATIAL, 0), Task(curves.OBJECT, 0))
    assert "fleet-wide fault lacks a measured effect on each suite" in check([inert], tasks=tasks)
    release = make_fault(
        "f01",
        "software_release",
        "camera_degradation|4 px blur",
        "fleet",
        0,
        nominal_version="4.2.0",
        version="4.3.0",
    )
    assert not check([release], tasks=tasks)


def test_robot_day_floor_checks_the_combination_and_none_disables_it() -> None:
    a = make_fault("f01", "camera_pose_offset", "6 deg scene pan", "r01", 0)
    b = make_fault("f02", "action_latency", "17 steps", "r01", 0)
    for fault in (a, b):
        assert not any("success floor" in p for p in check([fault], robot_success_floor=0.4))
    assert any(
        "robot r01 day 0 below success floor 0.4" in p
        for p in check([a, b], robot_success_floor=0.4)
    )
    assert not any("success floor" in p for p in check([a, b], robot_success_floor=None))


def test_known_interaction_cannot_meet_the_production_floor() -> None:
    current = curves.load()
    old_pair = {("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk")}
    old = replace(current, joints=tuple(j for j in current.joints if set(j.members) == old_pair))
    for name in ("D2", "D3"):
        with pytest.raises(ValueError, match=r"at most 0.0883.*0.4.*milder"):
            compose(
                DEVELOPMENT[name].model_copy(
                    update={"sealed_hash": None, "calibration_sources": None}
                ),
                old,
            )


def test_linear_drift_crossing_uses_its_own_scheduled_task_mix() -> None:
    c = curves.load()
    ratio = c.points[("camera_pose_offset", curves.SPATIAL, "3 deg scene pan")].ratio
    rate = 0.05 * 3 / (c.healthy[curves.SPATIAL] * (1 - ratio) * 8) * 1.001
    drift = make_fault(
        "f01",
        "camera_pose_offset",
        "3 deg scene pan",
        "r01",
        0,
        level=0.0,
        drift=Drift(shape="linear", rate=rate),
    )
    composer = Composer(recipe(warnable_drift=True))
    ctx = composer.context(Draft())
    assert estimate.drift_crossing_day(drift, composer.slots, ctx) == 8
    from fleetlab.worlds.compose import Composition
    from fleetlab.worlds.world import build

    pattern = Patterns(drifts=["f01"])
    spec = composer.spec([drift], pattern)
    comp = Composition(composer.recipe, spec, 0, {"f01": "drift"}, frozenset(), False, {})
    made = build(comp, SOFTWARE, TEXTS)
    row = made.fault_table.by_id()["f01"]
    assert row.drift_crossing_day == 8 and row.small
    assert not check([drift], Patterns(drifts=["f01"]), warnable_drift=True)
    slow = drift.model_copy(
        update={
            "schedule": drift.schedule.model_copy(
                update={"drift": Drift(shape="linear", rate=0.001)}
            )
        }
    )
    assert estimate.drift_crossing_day(slow, composer.slots, ctx) is None
    assert any(
        "crossing day" in p for p in check([slow], Patterns(drifts=["f01"]), warnable_drift=True)
    )
    step = drift.model_copy(
        update={
            "schedule": drift.schedule.model_copy(update={"drift": Drift(shape="step", rate=rate)})
        }
    )
    assert "drift must grow linearly over the whole world" in check(
        [step], Patterns(drifts=["f01"])
    )
    later = drift.model_copy(
        update={"schedule": drift.schedule.model_copy(update={"onset_day": 3})}
    )
    assert "drift must grow linearly over the whole world" in check(
        [later], Patterns(drifts=["f01"])
    )


def test_warnable_drift_requires_a_visible_precursor(monkeypatch: pytest.MonkeyPatch) -> None:
    drift = make_fault(
        "f01",
        "action_latency",
        "14 steps",
        "r01",
        0,
        level=0.0,
        drift=Drift(shape="linear", rate=1.0),
    )
    monkeypatch.setattr(estimate, "drift_crossing_day", lambda *a, **kw: 8)
    assert "drift lacks a visible health precursor" in check(
        [drift], Patterns(drifts=["f01"]), warnable_drift=True
    )


def test_calibration_record_occupies_its_camera() -> None:
    occ = Occupancy(base.all_robots(2, 4))
    occ.take("r01", things_of("camera_pose_offset", "3 deg scene pan"))
    assert not occ.free("r01", things_of("calibration_update", "scene calibration rewritten"))
    assert occ.free("r02", things_of("calibration_update", "scene calibration rewritten"))
    occ.take("fleet", things_of("software_release", "camera_degradation|4 px blur"))
    assert not occ.free("r02", things_of("calibration_update", "scene calibration rewritten"))


def test_planted_calibration_uses_benign_values_and_authors() -> None:
    from fleetlab.contracts.vocabulary import ConfigKey
    from fleetlab.worlds.world import _author

    composer = Composer(recipe())
    inert = make_fault("f02", "calibration_update", "scene calibration rewritten", "r02", 1)
    draft = Draft(fixed=[inert])
    schedule = composer.schedule(draft)
    planted = schedule[0]
    assert isinstance(planted.nominal, list)
    expected = calibration_value(
        composer.recipe.seed,
        "r02",
        1,
        ConfigKey.camera_scene_calibration.value,
        [float(v) for v in planted.nominal],
    )
    assert planted.params["value"] == expected != inert.params["value"]
    spec = composer.spec(schedule, Patterns())
    for row in changes(spec):
        if row.key not in (ConfigKey.camera_scene_calibration, ConfigKey.camera_wrist_calibration):
            continue
        assert isinstance(row.old, list)
        assert row.new == calibration_value(
            spec.seed, row.scope, row.day, row.key.value, [float(v) for v in row.old]
        )
        assert row.author == _author(spec.seed, "calibration", row.scope, row.day, row.key)


def test_site_confounder_calibration_has_the_same_scope_kind_as_benign_records() -> None:
    from fleetlab.contracts.ids import scope_covers

    composer = Composer(recipe(confounders=1, confounder_companion="calibration_record"))
    composer.plain_menu = [("object_batch_variant", "site")]
    draft = composer.draft(0)
    schedule = composer.schedule(draft)
    by_id = {f.fault_id: f for f in schedule}
    effective_id, inert_id = draft.patterns.confounders[0]
    effective, inert = by_id[effective_id], by_id[inert_id]
    assert effective.target.scope.startswith("s")
    assert inert.kind == "calibration_update" and inert.target.scope.startswith("r")
    assert scope_covers(
        effective.target.scope, inert.target.scope, composer.site_of[inert.target.scope]
    )
    assert inert.schedule.onset_day == effective.schedule.onset_day
    assert not set(things_of("object_batch_variant", composer.effect_levels[effective.kind][0])) & (
        set(things_of("calibration_update", "scene calibration rewritten"))
    )
    broken = inert.model_copy(
        update={"target": inert.target.model_copy(update={"scope": effective.target.scope})}
    )
    messages, _ = composer.check(draft, [f if f != inert else broken for f in schedule])
    assert "inert calibration record lacks a benign robot-scope counterpart" in messages
    with pytest.raises(ValueError, match="leaves no camera"):
        recipe(fleet_fault=True, confounder_companion="calibration_record")


def test_decoy_flatness_is_per_suite_and_requires_confirmation() -> None:
    fault = make_fault("f01", "illumination_change", "2 stops dimmer", "s1", 0)
    pattern = Patterns(decoys=["f01"])
    assert not check([fault], pattern)
    assert any("decoy lacks" in p for p in check([fault], pattern, tasks=(Task(curves.OBJECT, 0),)))
    c = curves.load()
    key = (fault.kind, curves.SPATIAL, "2 stops dimmer")
    damaged = replace(
        c,
        points={
            **c.points,
            key: replace(
                c.points[key], n=100, successes=98, logged_successes=98, healthy_successes=98
            ),
        },
    )
    assert any("decoy lacks" in p for p in check([fault], pattern, use=damaged))


def test_healthy_reference_keeps_the_larger_measurement() -> None:
    c = curves.load()
    assert c.healthy[curves.SPATIAL] == 968 / 1000
    assert c.healthy_counts[curves.SPATIAL] == (968, 1000)
    assert c.healthy[curves.OBJECT] == 985 / 1000


def test_patterns_schema_is_closed_in_both_modes() -> None:
    for mode in ("validation", "serialization"):
        schema = Patterns.model_json_schema(mode=mode)
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == {
            "confounders",
            "decoys",
            "interactions",
            "additive_pairs",
            "drifts",
        }
    assert "additive_pairs" not in Patterns().model_dump()
    assert Patterns(additive_pairs=[("f01", "f02")]).model_dump()["additive_pairs"] == [
        ("f01", "f02")
    ]
    with pytest.raises(ValueError):
        Patterns.model_validate({"arbitrary_truth": 1})


def test_historical_recipe_rejects_new_calibration() -> None:
    for name in ("D2", "D3"):
        assert INTEGRATION[name].sealed_hash is not None
        with pytest.raises(ValueError, match="sealed recipe calibration changed"):
            compose(INTEGRATION[name])


def test_prediction_intervals_are_reproducible_and_acknowledge_their_limits() -> None:
    fault = make_fault("f01", "camera_pose_offset", "6 deg scene pan", "r01", 0)
    eps = [estimate.Episode(curves.SPATIAL, (fault,))] * 100
    ctx = estimate.Context(curves.load(), frozenset(), frozenset())
    first = estimate.prediction(eps, ctx, draws=128)
    assert first == estimate.prediction(eps, ctx, draws=128)
    point = estimate.mean_success(eps, ctx)
    lo, hi = first["true"]["prediction_interval"]
    assert lo < point < hi and hi - lo > 0.1
    assert first["true"]["calibration_sd_points"] > 0
    assert "task heterogeneity" in first["limits"] and "not validated coverage" in first["limits"]


def test_private_recipe_draws_seed_once_and_resumes_only_from_truth(tmp_path: Path) -> None:
    # A private file states every field and names its calibration jobs (round 8).
    raw = stated_fields(recipe())
    raw.update(
        kind="test",
        estimator="measured-v3",
        world_id="w99",
        tasks=[{"suite": curves.SPATIAL, "task_index": 7}],
        calibration_jobs=list(FOURTH_GENERATION_JOBS),
        # ... and leaves nothing empty that the code's recipe states (round 9).
        calibration_sources=list(curves.measured_sources(FOURTH_GENERATION_JOBS)),
        robot_success_floor=0.3,
        interaction_single_loss_points=10,
        confounder_companion="calibration_record",
    )
    path = tmp_path / "private-template.json"
    text = json.dumps(raw)
    path.write_text(text)
    first = external_recipe(path)
    assert path.read_text() == text and "seed" not in json.loads(text)
    record = tmp_path / "truth" / "generator.json"
    record.parent.mkdir()
    record.write_text(json.dumps({"recipe": first.model_dump(mode="json")}))
    assert external_recipe(path, stored=record) == first
    assert external_recipe(path).seed != first.seed
    path.write_text(json.dumps({**raw, "seed": first.seed}))
    with pytest.raises(ValueError, match="omit seed"):
        external_recipe(path)
    path.write_text(json.dumps({**raw, "name": "changed"}))
    with pytest.raises(ValueError, match="differs"):
        external_recipe(path, stored=record)
    with pytest.raises(ValueError, match="outside the repository"):
        external_recipe(Path(__file__).parent / "recipe.json")


def test_seed_draw_uses_all_62_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.worlds import recipe as module

    for value in (0, 1 << 40, (1 << 62) - 1):

        def bits(n: int, draw: int = value) -> int:
            assert n == 62
            return draw

        monkeypatch.setattr("fleetlab.worlds.recipe.secrets.randbits", bits)
        assert module.new_world_seed() == (1 << 62) | value


@pytest.mark.parametrize(("a", "b", "joint"), [(900, 900, 500), (710, 950, 700), (990, 990, 974)])
def test_fault_table_and_detectability_flags_cannot_disagree(a: int, b: int, joint: int) -> None:
    from fleetlab.worlds.world import fault_table

    table = paired_table(a, b, joint)
    keys = INTERACTION_PAIRS[0]
    faults = [make_fault(f"f0{i + 1}", *key, "r01", 0) for i, key in enumerate(keys)]
    composer = Composer(recipe())
    patterns = Patterns(interactions=[("f01", "f02")])
    spec = composer.spec(faults, patterns)
    eps = [estimate.Episode(curves.SPATIAL, tuple(faults))] * 20
    truth = fault_table(
        spec,
        [],
        table,
        composer.context(Draft(patterns=patterns)),
        dict.fromkeys(detect.WINDOWS, eps),
    ).by_id()
    for unit in table.units:
        if len(unit.fault_ids) != 1:
            continue
        row = truth[unit.fault_ids[0]]
        assert row.expected_loss_points["observed"] == pytest.approx(
            unit.effect["observed"], abs=0.001
        )
        assert row.qualified_by == list(unit.qualified_by)
        assert row.detectable == unit.detectable
        assert row.small == (not unit.detectable)
        assert row.drift_crossing_day is None


def test_summary_gate_uses_observed_causes_frozen_band_and_catalogue_ratio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from test_worlds_runner import toy_world

    from fleetlab.contracts import actions
    from fleetlab.worlds import summary
    from fleetlab.worlds.layout import WorldPaths

    world = toy_world()
    identities = [j.identity for j in world.jobs_in("observed")]
    cause = replace(
        paired_table(900, 900, 500).units[0], qualified_by=("scoring",), confirmed_by=("scoring",)
    )
    table = detect.DetectTable([replace(cause, unit=f"f{i:02}") for i in range(6)], [], [], {})
    object.__setattr__(world, "detect", table)
    paths = WorldPaths(tmp_path, "w01")
    faults = len(world.spec.fault_schedule)
    for successes, n, in_band in ((3, 5, True), (3, 4, True), (7, 12, False), (10, 12, False)):
        rows = {
            str(i): {"identity": ident, "logged": i < successes, "true": i < successes, "steps": 1}
            for i, ident in enumerate(identities[:n])
        }
        monkeypatch.setattr(summary, "outcomes", lambda p, current=rows: current)
        for ratio, enough in ((9, False), (10, True)):
            monkeypatch.setattr(
                actions, "catalogue", lambda *a, count=ratio * faults, **kw: [None] * count
            )
            marks = summary.summary(world, paths)["gate3"]
            assert marks["observed_success_in_band"]["band"] == [0.60, 0.75]
            assert marks["observed_success_in_band"]["met"] is in_band
            assert marks["catalogue_ratio"]["met"] is enough
            assert marks["detectable_causes"]["counted"] == 6
            assert marks["detectable_causes"]["met"] is False
    table.units[:] = [
        replace(u, qualified_by=("observed",), confirmed_by=("observed",)) for u in table.units
    ]
    assert summary.summary(world, paths)["gate3"]["detectable_causes"]["met"] is True


def test_sealed_hash_blocks_any_changed_world_before_writing(tmp_path: Path) -> None:
    r = recipes()["D2"]
    world = generate(r, SOFTWARE, TEXTS, example_curves())
    digest = world_hash(world)
    pinned = r.model_copy(update={"sealed_hash": digest})
    assert world_hash(generate(pinned, SOFTWARE, TEXTS, example_curves())) == digest
    broken = r.model_copy(update={"sealed_hash": "0" * 64})
    with pytest.raises(ValueError, match="sealed hash"):
        generate(broken, SOFTWARE, TEXTS, example_curves())
    unverified = replace(world, recipe=broken)
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="sealed hash"):
        write_world(unverified, target)
    assert not target.exists()


# Creation through the command line takes only the code's own recipe since round 9; its
# test, with the ledger of draws, is in tests/test_worlds_round9.py.


@pytest.mark.parametrize(
    ("name", "sealed"),
    [
        ("D1", "d7bd6900f256781e44047c49225bbe701400188d7ca5db6fd73dbbb8a25f17d3"),
        ("D2", "d8b9f851e93ecc61ec8fd8c7382b09ea2ece617ec9ccb00630aab71894497ded"),
        ("D3", "932812bf436fdd6a83a6b2912f26c9d3cee203625a36e7695e1a6f327b54df5c"),
    ],
)
def test_recipe_regression_hashes_fail_on_recomposition(name: str, sealed: str) -> None:
    # These CPU fixtures have explicitly synthetic object decoys and fake software.
    # Their constants are seals, not values computed from the implementation under test.
    world = generate(recipes()[name], SOFTWARE, TEXTS, example_curves())
    assert world_hash(world) == sealed


def test_new_calibration_requires_a_new_recipe_even_if_the_points_are_unused() -> None:
    c = curves.load()
    pinned = recipe(calibration_sources=c.sources)
    Composer(pinned, c)
    with pytest.raises(ValueError, match="calibration changed"):
        Composer(pinned, replace(c, sources=(*c.sources, "another measured point")))


def test_public_ids_and_version_numbers_do_not_encode_even_part_of_seed() -> None:
    from fleetlab.contracts.vocabulary import ConfigKey

    for name in ("D2", "D3"):
        r = recipes()[name]
        world = generate(r, SOFTWARE, TEXTS, example_curves())
        assert world.fleet.id == world.observed_fleet.id == r.public_fleet_id == "w01"
        releases = sum(f.kind == "software_release" for f in world.spec.fault_schedule)
        allowed = {"4.2.0", "4.2.1", "4.2.2"} | {f"4.{3 + i}.0" for i in range(releases)}
        versions = {
            str(value)
            for change in world.fleet.change_log
            if change.key is ConfigKey.software_version
            for value in (change.old, change.new)
        }
        assert versions <= allowed


def test_summary_and_generator_record_include_the_sealed_prediction_intervals(
    tmp_path: Path,
) -> None:
    from fleetlab.worlds.layout import WorldPaths
    from fleetlab.worlds.summary import composition_report, summary
    from fleetlab.worlds.world import generator_record

    world = generate(recipes()["D2"], SOFTWARE, TEXTS, example_curves())
    record = generator_record(world)
    report = summary(world, WorldPaths(tmp_path, world.spec.world_id))
    assert report["predictions"] == record["predictions"] == world.predictions
    assert len(record["predictions_sha256"]) == 64
    handoff = composition_report(world)
    assert len(handoff["observed_causes"]) >= 6
    assert len(handoff["largest_three_single_fault_fixes"]) == 3
    assert handoff["lowest_robot_day"]["true"] < 0.4  # deliberately NOT a production fixture
    assert {r["visible_in"][0] for r in handoff["faults"]} == {"change_log", "health", "behaviour"}


def test_requested_fleet_fault_and_drift_cannot_disappear() -> None:
    assert "the requested fleet-wide fault is missing" in check([], fleet_fault=True)
    assert "the requested drift is missing" in check([], drifts=1)


def test_production_recipe_marks_cannot_be_silently_lowered() -> None:
    for name in ("D2", "D3"):
        r = DEVELOPMENT[name]
        assert r.band == (0.60, 0.75) and r.min_detectable >= 6
        assert 0.60 <= r.aim[0] < r.aim[1] <= 0.75
        assert r.confounders == r.interactions == r.additive_pairs == r.drifts == 1
        assert r.decoys >= 1 and r.robot_success_floor == 0.40


def test_real_mixed_suite_decoys_fail_before_random_search() -> None:
    mixed = recipe(decoys=1, tasks=(Task(curves.SPATIAL, 0), Task(curves.OBJECT, 0)))
    with pytest.raises(ValueError, match="no confirmed-flat placement"):
        compose(mixed)


def test_small_means_below_detection_not_merely_unconfirmed() -> None:
    cause = replace(paired_table(900, 900, 500).units[0], confirmed=False)
    assert not cause.counts and cause.detectable and not cause.small


def test_configuration_share_and_observed_success_band_are_guards() -> None:
    physical = make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0)
    assert any(
        "configuration and software share" in p for p in check([physical], config_share=2 / 3)
    )
    setting = make_fault("f01", "action_latency", "14 steps", "r01", 0)
    assert not check([setting], config_share=2 / 3)
    assert any(
        "expected observed success" in p for p in check([], band=(0.60, 0.75), aim=(0.645, 0.705))
    )


def test_no_test_recipe_is_in_repository_and_symlinks_do_not_bypass_boundary(
    tmp_path: Path,
) -> None:
    assert all(r.kind == "development" for r in [*DEVELOPMENT.values(), *INTEGRATION.values()])
    alias = tmp_path / "outside-link.json"
    alias.symlink_to(Path(__file__).resolve())
    with pytest.raises(ValueError, match="outside the repository"):
        external_recipe(alias)


def test_zero_success_calibration_does_not_claim_zero_uncertainty() -> None:
    faults = tuple(
        make_fault(f"f0{i + 1}", *key, "r01", 0) for i, key in enumerate(INTERACTION_PAIRS[0])
    )
    ctx = estimate.Context(curves.load(), frozenset(), frozenset(), (("f01", "f02"),))
    result = estimate.prediction([estimate.Episode(curves.OBJECT, faults)] * 1000, ctx, draws=256)
    assert result["true"]["expected"] == 0.0
    assert result["true"]["calibration_sd_points"] > 0
    assert result["true"]["calibration_interval"][1] > 0
    assert result["true"]["prediction_interval"][1] > 0


def test_private_recipe_rejects_another_git_worktree(tmp_path: Path) -> None:
    checkout = tmp_path / "another-checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /unused/shared-repository\n")
    path = checkout / "untracked-private-recipe.json"
    path.write_text('{"kind":"test"}')
    with pytest.raises(ValueError, match="outside the repository"):
        external_recipe(path)
