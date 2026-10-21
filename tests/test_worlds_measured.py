"""Round-6 guarantees. Synthetic cells isolate rules; archived cells are separate evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worlds_review import recipe
from test_worlds_task_guards import cells, task_recipe
from test_worlds_v1 import SOFTWARE

from fleetlab.contracts.identity import Task
from fleetlab.worlds import base, calibration, curves, estimate
from fleetlab.worlds.compose import Composer, Draft, make_fault
from fleetlab.worlds.recipe import SMALL_WINDOWS, WorldRecipe


def cell(rate: float, n: int = 10000) -> curves.MeasuredCell:
    return curves.MeasuredCell(n, round(n * rate), round(n * rate), ("synthetic CPU fixture",))


def test_refinement_manifest_covers_thin_pair_counterfactuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real CPU planner; no engine job is admitted or simulated.

    The counts are those that exposed the omission, set here so that the
    test does not depend on which top-ups have been imported since.
    """
    from collections import Counter

    from fleetlab.contracts.rollout import Job

    pair = (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"))
    quarter = (curves.SPATIAL, 9, (("gripper_wear", "0.25 wear"),))
    table = dict(curves.measured_cells())
    thin: dict[curves.CellKey, tuple[int, int]] = {
        (curves.SPATIAL, 7, (pair[0],)): (15, 12),
        (curves.OBJECT, 7, (pair[0],)): (10, 8),
        (curves.OBJECT, 7, (pair[1],)): (10, 8),
        quarter: (10, 8),
    }
    for thin_key, (n, successes) in thin.items():
        table[thin_key] = curves.MeasuredCell(n, successes, successes, table[thin_key].sources)
    monkeypatch.setattr(curves, "measured_cells", lambda: table)
    root = tmp_path / "plan"
    plan = calibration.refinement_plan(root)
    stages = plan["stages"]
    assert isinstance(stages, dict)
    counts: Counter[curves.CellKey] = Counter()
    for stage in stages:
        if not stage.startswith("thin-"):
            continue
        for line in (root / f"{stage}.jsonl").read_text().splitlines():
            identity = Job.model_validate_json(line).identity
            counts[
                identity.suite,
                identity.task_index,
                calibration.mechanism_members(identity.faults),
            ] += 1
    tested = 0
    for suite in (curves.SPATIAL, curves.OBJECT):
        for member in pair:
            for task in range(10):
                key: curves.CellKey = (suite, task, (member,))
                before = table[key]
                if not before.sufficient:
                    assert before.n + counts[key] == 50, key
                    tested += 1
    assert tested >= 3  # the omitted single cells, not an empty assertion
    # The linear drift cannot interpolate around an existing thin quarter-wear
    # knot. Its spatial task 9 cell was another omitted prerequisite.
    assert table[quarter].n + counts[quarter] == 50
    # The folded worlds already made this joint cell sufficiently large; a new
    # plan must not repeat the old top-up just because its original cohort was thin.
    joint: curves.CellKey = (curves.SPATIAL, 1, pair)
    assert table[joint].sufficient and counts[joint] == 0


def context(rows: dict[tuple[tuple[str, str], ...], float], n: int = 10000) -> estimate.Context:
    return estimate.Context(
        curves.load(),
        frozenset(),
        frozenset(),
        task_aware=True,
        measured={(curves.SPATIAL, 0, tuple(sorted(k))): cell(v, n) for k, v in rows.items()},
    )


def test_composer_episodes_preserve_the_scheduled_task() -> None:
    comp = Composer(
        task_recipe(tasks=(Task(curves.SPATIAL, 0), Task(curves.SPATIAL, 1))), cells(0.9, 0.2)
    )
    comp.slots = (base.Slot("r01", "s1", 0, 0, Task(curves.SPATIAL, 1), 0),)
    fault = make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0)
    assert comp.episodes([fault], range(1))[0].task_index == 1
    assert comp.observed_success([fault], comp.context(Draft())) == pytest.approx(0.2)


def test_lower_band_reserve_and_scoring_reserve_are_enforced() -> None:
    comp = Composer(
        task_recipe(band=(0.60, 0.75), aim=(0.61, 0.74)), cells(0.629, 0.629, healthy=0.629)
    )
    assert any("expected observed" in s for s in comp.check(Draft(), [])[0])
    comp = Composer(task_recipe(scoring_floor=0.5), cells(0.51, 0.51, healthy=0.51))
    assert any("scoring success" in s for s in comp.check(Draft(), [])[0])


def test_pair_screen_keeps_the_margin_on_robot_windows() -> None:
    c = curves.load()
    pair = (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"))
    original = c.for_task(curves.SPATIAL, 0)
    joint, _ = original.joint(*pair, curves.SPATIAL)
    for p, eligible in ((0.42, False), (0.44, True)):
        updated = replace(
            joint,
            n=1000,
            healthy_successes=1000,
            successes=p * 1000,
            single_successes=(1000, 1000),
            singles=(1.0, 1.0),
            interaction=(-0.5, -0.6, -0.4),
        )
        task = replace(original, healthy={curves.SPATIAL: 1.0}, joints=(updated,))
        comp = Composer(
            task_recipe(robot_success_floor=0.4, interaction_single_loss_points=10),
            replace(c, tasks={(curves.SPATIAL, 0): task}),
        )
        comp.slots = tuple(
            base.Slot("r01", "s1", d, 0, Task(curves.SPATIAL, 0), 0) for d in range(1, 10)
        )
        assert ("r01" in comp.pair_robot_choices(pair)) is eligible


def test_measured_carrier_preflight_is_not_vetoed_by_an_archived_suite_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.worlds.compose import compose

    c = curves.load()
    c = replace(c, joints=tuple(replace(j, successes=0) for j in c.joints))
    monkeypatch.setattr(Composer, "pair_scope_choices", lambda *a: ["r01"])

    def reached(*args: Any) -> Draft:
        raise RuntimeError("task-qualified draft reached")

    monkeypatch.setattr(Composer, "draft", reached)
    with pytest.raises(RuntimeError, match="task-qualified draft reached"):
        compose(task_recipe(estimator="measured-v3", interactions=1, robot_success_floor=0.4), c)


def test_pair_screen_includes_the_whole_window_and_requires_removal_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"))
    c = curves.load()
    original = c.for_task(curves.SPATIAL, 0)
    joint, _ = original.joint(*pair, curves.SPATIAL)
    tasks, measured = {}, {}
    for task, rate in enumerate((0.35, 0.95, 0.6)):
        j = replace(
            joint,
            n=1000,
            healthy_successes=1000,
            successes=1000 * rate,
            single_successes=(1000, 1000),
            singles=(1.0, 1.0),
            interaction=(-0.4, -0.5, -0.3),
        )
        tasks[curves.SPATIAL, task] = replace(original, joints=(j,))
        for members, p in (((), 1.0), ((pair[0],), 1.0), ((pair[1],), 1.0), (pair, rate)):
            measured[curves.SPATIAL, task, tuple(sorted(members))] = cell(p)
    monkeypatch.setattr(curves, "measured_cells", lambda: measured)
    comp = Composer(
        task_recipe(estimator="measured-v3", robot_success_floor=0.4), replace(c, tasks=tasks)
    )
    comp.slots = tuple(
        base.Slot(
            "r01", "s1", day, 0, Task(curves.SPATIAL, 1 if day == 0 else 0 if day < 6 else 2), 0
        )
        for day in range(10)
    )
    # Observed: (1 + 5*.35)/6 = .4583; later windows .6. Screening only
    # days carrying BOTH faults would incorrectly compare .35 with the floor.
    assert "r01" in comp.pair_robot_choices(pair)
    # Task 0 occurs only after both members act. Its solo delay cell is needed
    # by a removal, although the all-fault window calculation never reads it.
    measured[curves.SPATIAL, 0, (pair[0],)] = cell(0.8, n=15)
    checked = Composer(comp.recipe, replace(c, tasks=tasks))
    checked.slots = comp.slots
    assert "r01" not in checked.pair_robot_choices(pair)


def test_floor_is_each_robot_window_and_all_faults_not_each_day() -> None:
    fault = make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0)
    c = cells(0.1, 0.8)
    ctx = estimate.Context(c, frozenset(), frozenset(), task_aware=True)
    slots = tuple(
        base.Slot("r01", "s1", day, 0, Task(curves.SPATIAL, day % 2), 0) for day in range(10)
    )
    rows = estimate.robot_window_success([fault], slots, ctx, SMALL_WINDOWS)
    assert len(rows) == 3 and all(r["true"] == pytest.approx(0.45) for r in rows)
    assert min(r["true"] for r in estimate.robot_day_success([fault], slots, ctx)) == pytest.approx(
        0.1
    )
    comp = Composer(task_recipe(robot_success_floor=0.4), c)
    comp.slots = slots
    assert comp.floor_holds([fault], Draft())
    # A good observed window must not mask a bad scoring window.
    comp.slots = tuple(replace(s, task=Task(curves.SPATIAL, 0)) if s.day >= 8 else s for s in slots)
    assert not comp.floor_holds([fault], Draft())
    comp.recipe = comp.recipe.model_copy(update={"robot_success_floor": None})
    assert comp.floor_holds([fault], Draft())


def test_no_product_fallback_for_overlap_or_missing_counterfactual() -> None:
    a = make_fault("f01", "action_latency", "8 steps", "r01", 0)
    b = make_fault("f02", "policy_knob", "15 actions per chunk", "r01", 0)
    ka, kb = calibration.mechanism_members([a])[0], calibration.mechanism_members([b])[0]
    ctx = context({(): 0.98, (ka,): 0.95, (kb,): 0.96})
    with pytest.raises(KeyError, match="unmeasured task context"):
        estimate.probability([a, b], curves.SPATIAL, ctx, task_index=0)
    assert ctx.measured is not None
    ctx.measured[curves.SPATIAL, 0, tuple(sorted((ka, kb)))] = cell(0.70)
    assert estimate.probability([a, b], curves.SPATIAL, ctx, task_index=0) == 0.70
    del ctx.measured[curves.SPATIAL, 0, (kb,)]
    with pytest.raises(KeyError, match="unmeasured task context"):
        estimate.measured_causes([estimate.Episode(curves.SPATIAL, (a, b), 0)], ctx)


def test_release_expands_to_measured_context_and_preserves_decoy() -> None:
    release = make_fault(
        "f01",
        "software_release",
        "action_latency|8 steps + policy_knob|15 actions per chunk",
        "fleet",
        0,
        nominal_version="4.2.0",
        version="4.3.0",
    )
    decoy = make_fault("f02", "illumination_change", "2 stops dimmer", "s1", 0)
    inert = make_fault("f03", "calibration_update", "scene calibration rewritten", "r01", 0)
    members = calibration.mechanism_members([release, decoy, inert])
    assert len(members) == 3 and (decoy.kind, "2 stops dimmer") in members
    ctx = context({(): 0.98, members: 0.61})
    assert estimate.probability([release, decoy, inert], curves.SPATIAL, ctx, task_index=0) == 0.61


def test_dose_search_skips_unmeasured_contexts_even_without_a_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = make_fault("f01", "action_latency", "8 steps", "r01", 0)
    b = make_fault("f02", "policy_knob", "15 actions per chunk", "r01", 0)
    ka, kb = calibration.mechanism_members([a])[0], calibration.mechanism_members([b])[0]
    ctx = context({(): 0.98, (ka,): 0.95, (kb,): 0.96})
    comp = Composer(task_recipe(robot_success_floor=None))
    comp.slots = (base.Slot("r01", "s1", 0, 0, Task(curves.SPATIAL, 0), 0),)
    monkeypatch.setattr(comp, "context", lambda draft: ctx)
    assert not comp.floor_holds([a, b], Draft())
    assert comp.floor_holds([a], Draft())
    assert ctx.measured is not None
    ctx.measured[curves.SPATIAL, 0, tuple(sorted((ka, kb)))] = cell(0.60)
    assert comp.floor_holds([a, b], Draft())
    del ctx.measured[curves.SPATIAL, 0, (kb,)]
    assert not comp.floor_holds([a, b], Draft())


def test_data_fault_cannot_be_forecast_as_an_oracle_success_cause() -> None:
    fault = make_fault("f01", "success_detector_error", "0.4 missed", "r01", 0)
    key = calibration.mechanism_members([fault])
    # Noisy cohorts can have different physical rates; the data hook itself
    # cannot alter physical success, so it cannot satisfy this cause gate.
    ctx = context({(): 0.98, key: 0.50})
    result = estimate.measured_causes([estimate.Episode(curves.SPATIAL, (fault,), 0)], ctx)
    assert result["count"] == 0 and result["faults"] == {}


def test_midrange_cells_require_fifty_and_interpolation_requires_joint_corners() -> None:
    f = make_fault("f01", "gripper_wear", "1 wear", "r01", 0, level=0.625)
    ka, kb = ("gripper_wear", "0.5 wear"), ("gripper_wear", "0.75 wear")
    ctx = context({(): 0.98, (ka,): 0.9, (kb,): 0.6}, n=10)
    with pytest.raises(KeyError, match="thin task context"):
        estimate.probability([f], curves.SPATIAL, ctx, task_index=0)
    assert ctx.measured is not None
    ctx.measured[curves.SPATIAL, 0, (kb,)] = cell(0.6, 50)
    assert estimate.probability([f], curves.SPATIAL, ctx, task_index=0) == pytest.approx(0.75)
    b = make_fault("f02", "object_batch_variant", "4 mass doublings", "s1", 0)
    with pytest.raises(KeyError, match="unmeasured task context"):
        estimate.probability([f, b], curves.SPATIAL, ctx, task_index=0)
    assert not cell(0.16, 49).sufficient and not cell(0.84, 49).sufficient
    assert cell(0.14, 50).sufficient and not cell(1.0, 9).sufficient


def test_continuous_dose_uses_its_measured_context_grid_without_skipping_thin_knots() -> None:
    wear = make_fault("f01", "gripper_wear", "1 wear", "r01", 0, level=0.1)
    blur = make_fault("f02", "camera_degradation", "3 px blur", "r01", 0)
    kb = ("camera_degradation", "3 px blur")
    quarter, half = ("gripper_wear", "0.25 wear"), ("gripper_wear", "0.5 wear")
    ctx = context({(): 0.98, (kb,): 0.90, (quarter,): 0.94, (kb, half): 0.60})
    # The solo quarter-wear observation says nothing about its blur joint.
    assert estimate.probability([wear, blur], curves.SPATIAL, ctx, task_index=0) == pytest.approx(
        0.84
    )
    assert ctx.measured is not None
    ctx.measured[curves.SPATIAL, 0, tuple(sorted((kb, quarter)))] = cell(0.5, 10)
    with pytest.raises(KeyError, match="thin task context"):
        estimate.probability([wear, blur], curves.SPATIAL, ctx, task_index=0)


def test_forecast_retains_error_envelope_and_reuses_cell_draws() -> None:
    ctx = context({(): 0.65}, 1000)
    eps = [estimate.Episode(curves.SPATIAL, (), 0)] * 1008
    report = estimate.measured_prediction(eps, ctx, draws=4096)
    row = report["logged"]
    a, b = row["conditional_prediction_interval"]
    x, y = row["prediction_interval"]
    assert x == pytest.approx(a - 0.007, abs=0.00011)
    assert y == pytest.approx(b + 0.060, abs=0.00011)
    assert report["model_error_points"] == [-0.7, 6.0]
    assert 0.015 < row["calibration_sd_points"] / 100 < 0.016
    assert 0 < row["band_probability_for_all_recorded_errors"] < 1
    assert report == estimate.measured_prediction(eps, ctx, draws=4096)
    for window, error in (("experiment", [-3.7, 6.6]), ("scoring", [-4.8, 5.4])):
        later = estimate.measured_prediction(eps, ctx, draws=256, window=window)
        assert later["model_error_points"] == error
        assert later["window"] == window


def test_planned_search_target_leaves_room_for_the_interval_its_recipe_is_held_to() -> None:
    from fleetlab.worlds.recipe import planned_recipe

    episodes = [estimate.Episode(curves.SPATIAL, (), 0)] * 1008

    def interval(rate: float, retained_error: str) -> tuple[float, float]:
        ctx = replace(context({(): rate}, 1000000), retained_error=retained_error)
        lo, hi = estimate.measured_prediction(episodes, ctx)["logged"]["prediction_interval"]
        return lo, hi

    # The fourth generation retains no error: both ends of its target keep the
    # forecast's own spread inside the band.
    r = planned_recipe("D2", world_id="w99", seed=2**62 + 13579)
    assert r.retained_error == "none" and r.aim == (0.640, 0.715)
    assert r.band[0] <= interval(r.aim[0], "none")[0]
    assert interval(r.aim[1], "none")[1] <= r.band[1]
    # With the third generation's error the same target would not: its upper end leaves
    # the band, and the target that did fit was 64.0 to 65.7%.
    assert interval(r.aim[1], "generation-3")[1] > r.band[1]
    lo, hi = interval((0.640 + 0.657) / 2, "generation-3")
    assert r.band[0] <= lo <= hi <= r.band[1]
    # Precise synthetic cells prove the search direction only, not feasibility
    # or coverage of any real world's measurements.


@pytest.mark.parametrize(
    ("probability", "interval", "cause_failure", "band_failure"),
    (
        (0.9499, [0.61, 0.74], True, False),
        (0.95, [0.61, 0.74], False, False),
        (1.0, [0.5999, 0.74], False, True),
        (1.0, [0.61, 0.7501], False, True),
    ),
)
def test_measured_admission_requires_both_resampling_and_the_entire_interval(
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
    interval: list[float],
    cause_failure: bool,
    band_failure: bool,
) -> None:
    comp = Composer(
        task_recipe(estimator="measured-v3", min_detectable=6, band=(0.60, 0.75), aim=(0.61, 0.74))
    )
    ctx = context({(): 0.65})
    monkeypatch.setattr(comp, "context", lambda draft: ctx)
    monkeypatch.setattr(
        estimate,
        "measured_prediction",
        lambda *a, **k: {"logged": {"prediction_interval": interval}},
    )
    monkeypatch.setattr(
        estimate,
        "measured_causes",
        lambda *a, **k: {"probability_at_least_minimum": probability, "faults": {}},
    )
    reasons, _ = comp.check(Draft(), [])
    assert any("scoring cause probability" in r for r in reasons) is cause_failure
    assert any("observed prediction interval" in r for r in reasons) is band_failure


def test_cause_resampling_uses_single_removals_and_only_interaction_fallback() -> None:
    a = make_fault("f01", "action_latency", "8 steps", "r01", 0)
    b = make_fault("f02", "policy_knob", "15 actions per chunk", "r01", 0)
    ka, kb = calibration.mechanism_members([a])[0], calibration.mechanism_members([b])[0]
    eps = [estimate.Episode(curves.SPATIAL, (a, b), 0)] * 336
    both = context({(): 0.98, (ka,): 0.95, (kb,): 0.96, (ka, kb): 0.7})
    result = estimate.measured_causes(eps, both, [("f01", "f02")], minimum=2)
    assert result["count"] == 2 and result["probability_at_least_minimum"] >= 0.95
    # Neither removal alone reaches two, but removal of both reaches three.
    joint = context({(): 0.73, (ka,): 0.715, (kb,): 0.715, (ka, kb): 0.70}, n=1000000)
    assert estimate.measured_causes(eps, joint, minimum=1)["count"] == 0
    fallback = estimate.measured_causes(eps, joint, [("f01", "f02")], minimum=1)
    assert fallback["count"] == 1 and fallback["probability_at_least_minimum"] >= 0.95
    # The uncertainty is shared, not 336 independent draws of a calibration rate.
    uncertain = context({(): 0.73, (ka,): 0.72, (kb,): 0.72, (ka, kb): 0.70}, n=50)
    r = estimate.measured_causes(eps, uncertain, minimum=2)
    assert r["probability_at_least_minimum"] < 0.95


def test_decoy_assay_is_task_weighted_and_missing_or_effective_context_fails() -> None:
    good = [
        {"task_index": 0, "n": 1000, "paired_counts": [980, 0, 0, 20]},
        {"task_index": 1, "n": 1000, "paired_counts": [980, 0, 0, 20]},
    ]
    assert calibration.decoy_mix_interval(good, {0: 80, 1: 20})["flat"]
    assert not calibration.decoy_mix_interval(good, {2: 20})["flat"]
    bad = [good[0], {"task_index": 1, "n": 1000, "paired_counts": [200, 0, 780, 20]}]
    assert not calibration.decoy_mix_interval(bad, {0: 1, 1: 99})["flat"]


def test_a_decoy_is_flat_only_when_its_interval_excludes_the_margin() -> None:
    """Zero inside the interval is not enough: a two-point loss on the carriers must be out.

    Dropping the margin from the rule passed every other test of the decoy rule.
    """
    assert calibration.DECOY_MARGIN == 0.02
    wide = [{"task_index": 0, "n": 100, "paired_counts": [90, 5, 5, 0]}]
    narrow = [{"task_index": 0, "n": 10000, "paired_counts": [9000, 500, 500, 0]}]
    few = calibration.decoy_mix_interval(wide, {0: 10})
    difference = few["difference"]
    assert isinstance(difference, list)
    assert difference[1] < -0.02 < 0 < difference[2] and not few["flat"]
    assert calibration.decoy_mix_interval(narrow, {0: 10})["flat"]


def test_write_world_itself_rejects_own_checkout_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.worlds import world

    root = Path(world.__file__).resolve().parents[3] / "data/private-path-rule-fixture"
    obj = SimpleNamespace(recipe=SimpleNamespace(kind="test", sealed_hash=None))
    monkeypatch.setattr(world, "serialized_world", lambda w: {})
    monkeypatch.setattr(world, "world_hash", lambda w: "0" * 64)

    def forbidden(*a: Any, **k: Any) -> None:
        pytest.fail("private writer reached mkdir inside its own checkout")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(ValueError, match="outside the repository"):
        world.write_world(obj, root)  # type: ignore[arg-type]


def test_private_summary_cannot_print_its_causes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fleetlab.worlds import __main__ as cli
    from fleetlab.worlds import recipe as recipes
    from fleetlab.worlds import runner, summary, world

    r = recipe(kind="test", estimator="measured-v3", world_id="w99", seed=2**62 + 789)
    root = tmp_path / "private"
    (root / "files").mkdir(parents=True)
    (root / "files/generator.json").write_text("{}")
    monkeypatch.setattr(recipes, "external_recipe", lambda *a, **k: r)
    monkeypatch.setattr(runner, "engine_software", lambda: SOFTWARE)
    monkeypatch.setattr(world, "generate", lambda *a, **k: SimpleNamespace(recipe=r))
    monkeypatch.setattr(
        summary, "summary", lambda *a: {"ready": True, "causes": ["private fault and seed"]}
    )
    assert cli.main(["summary", "--recipe", str(tmp_path / "template"), "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out) == {"private": True, "ready": True}


def test_private_worlds_cannot_select_old_forecasts() -> None:
    with pytest.raises(ValueError, match="measured context forecasts"):
        WorldRecipe(name="private", world_id="w99", kind="test", estimator="task-v2")
    assert WorldRecipe(name="private", world_id="w99", kind="test").estimator == "measured-v3"


def test_archived_world_cells_preserve_all_mechanisms_and_pin_sources() -> None:
    raw = json.loads(Path(curves.__file__).with_name("curves_supplement.json").read_text())
    record = raw["task_refinement"]
    assert record["cells"] and all(row["sources"] for row in record["cells"])
    assert any(len(row["members"]) >= 3 for row in record["cells"])
    assert any(
        any(m[0] == "illumination_change" for m in row["members"]) for row in record["cells"]
    )
    assert all(row["n"] == sum(v["n"] for v in row["sources"].values()) for row in record["cells"])
    assert curves.measured_sources()[:-1] == curves.load().sources
    assert curves.measured_sources()[-1].startswith("task_refinement sha256 ")


def test_data_fault_true_rate_is_not_its_logged_rate() -> None:
    table = curves.measured_cells()
    keys = [
        k
        for k in table
        if k[0] == curves.SPATIAL and k[2] == (("success_detector_error", "0.4 missed"),)
    ]
    assert keys
    assert any(table[k].successes > table[k].logged_successes for k in keys)


def test_fleet_fault_needs_positive_task_mix_evidence_not_a_suite_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fault = make_fault(
        "f01",
        "software_release",
        "camera_degradation|3 px blur",
        "fleet",
        0,
        nominal_version="4.2.0",
        version="4.3.0",
    )
    key = calibration.mechanism_members([fault])
    r = recipe(estimator="measured-v3", world_id="w99", seed=2**62 + 234, fleet_fault=True)
    for healthy, admitted in ((0.65, False), (0.80, True)):
        ctx = context({(): healthy, key: 0.65})
        comp = Composer(r)
        monkeypatch.setattr(comp, "context", lambda draft, current=ctx: current)
        reasons, _ = comp.check(Draft(fixed=[fault]), [fault])
        assert any("positive measured task-mix removal" in s for s in reasons) is not admitted


def test_inert_wrist_companion_has_the_same_benign_distribution() -> None:
    from fleetlab.contracts.vocabulary import ConfigKey
    from fleetlab.worlds.compose import calibration_value, things_of
    from fleetlab.worlds.recipe import planned_recipe

    f = make_fault("f01", "calibration_update", "wrist calibration rewritten", "r01", 0)
    assert f.target.config_key is ConfigKey.camera_wrist_calibration
    assert things_of(f.kind, "wrist calibration rewritten") == ["wrist_camera"]
    assert calibration.mechanism_members([f]) == ()
    r = planned_recipe("D2", world_id="w99", seed=2**62 + 123)
    assert r.confounder_companion == "calibration_record"
    comp = Composer(r)
    d = Draft(fixed=[f])
    value = comp.schedule(d)[0].params["value"]
    assert value == calibration_value(r.seed, "r01", 0, f.target.config_key.value, f.nominal)  # type: ignore[arg-type]


def test_world_materializer_retains_decoys_and_rejects_conflicting_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fleetlab.dose_response import Outcome
    from fleetlab.worlds import supplement
    from fleetlab.worlds.generate import policy_spec_nominal

    (tmp_path / "files").mkdir()
    (tmp_path / "files/spec.json").write_text('{"world_id":"w99"}')
    a = make_fault("f01", "action_latency", "8 steps", "r01", 0)
    d = make_fault("f02", "illumination_change", "2 stops dimmer", "s1", 0)
    i = supplement.identity(0, 0, curves.SPATIAL, policy_spec_nominal(), SOFTWARE, [a, d])
    o = Outcome(i.episode_id, True, True, True, 1)
    monkeypatch.setattr(supplement, "_task_rows", lambda path: [(i, o)])
    rows = calibration.world_task_cells([tmp_path])["cells"]
    assert isinstance(rows, list) and len(rows) == 1
    assert rows[0]["n"] == 1 and len(rows[0]["members"]) == 2
    monkeypatch.setattr(
        supplement,
        "_task_rows",
        lambda path: [(i, o if "observed" in path else replace(o, true_success=False))],
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        calibration.world_task_cells([tmp_path])


def test_real_world_refinement_recomputes_from_its_source_runs() -> None:
    from fleetlab.worlds import supplement

    raw = json.loads(Path(curves.__file__).with_name("curves_supplement.json").read_text())
    record = raw["task_refinement"]
    roots = [Path.home() / ".cache/robots-probes/worlds" / f"{w}-opaque-v1" for w in ("w46", "w47")]
    paths = [p for row in raw["task_calibration"]["points"] for p in row["run_dirs"]]
    jobs = record.get("calibration_roots") or record.get("calibration_root")
    calibration_roots = [jobs] if isinstance(jobs, str) else list(jobs or [])
    if not all(
        Path(p).exists()
        for p in (*roots, *paths, *(Path(r) / "plan.json" for r in calibration_roots))
    ):
        pytest.skip("archived measurement runs are not installed on this host")
    rebuilt = calibration.refinement_record(roots, jobs)
    assert json.loads(json.dumps(rebuilt)) == record
    assert sum(row["n"] for row in record["cells"]) >= 3360
    # Same cached raw outcomes also underlie the paired and single measurement checks.
    assert supplement._task_rows.cache_info().currsize > 0
    # The follow-up job's identities were unused for their contexts: none counted twice.
    assert not any(
        "task-cells-r6c" in key and key.endswith(":duplicate") for key in record["counts"]
    )
    # Every same-attempt comparison names its decoy among its members and pairs every task.
    for assay in record["decoy_assays"]:
        assert assay["decoy"] in [list(m) for m in assay["members"]]
        assert len(assay["tasks"]) == 10 and all(
            sum(task["paired_counts"]) == task["n"] for task in assay["tasks"]
        )


def test_preregistered_continuation_refines_midrange_cells_and_keeps_decoys_paired() -> None:
    blur = (curves.SPATIAL, 0, (("camera_degradation", "3 px blur"),))
    light = (curves.SPATIAL, 0, (*blur[2], ("illumination_change", "2 stops dimmer")))
    wall = (curves.SPATIAL, 0, (*blur[2], ("backdrop_change", "1 walls and floor")))
    parent = {blur: (20, 20, 20), light: (20, 8, 8), wall: (20, 20, 20)}
    assert calibration.continuation_counts(parent, {}) == {light: 30}
    assert calibration.continuation_counts(parent, {}, paired_decoys=True) == dict.fromkeys(
        parent, 30
    )
    assert calibration.continuation_counts(parent, {light: cell(0.5, 40)}) == {}
    # Logged errors need precision too, even when the physical cell is extreme.
    assert calibration.continuation_counts({blur: (10, 10, 5)}, {}) == {blur: 40}


def test_every_stage_of_paired_decoy_arms_continues_its_arms_together() -> None:
    assert calibration.paired_stage("decoys")
    assert all(calibration.paired_stage(name) for name, *_ in calibration.FOLLOWUP_ARMS)
    assert not calibration.paired_stage("contexts-1")
    assert not calibration.paired_stage("topups-1")
    assert not calibration.paired_stage("decoysmisnamed")
    # A stage is run from the plan by name: every refinement's parent is a stage before it.
    order = list(calibration.FOLLOWUP_ORDER)
    for name in order:
        if name.endswith("-refine"):
            assert order.index(name.removesuffix("-refine")) == order.index(name) - 1


def test_followup_plan_is_the_sealed_one_and_pairs_unused_identities() -> None:
    """The manifests on disk are the plan this commit names, and they keep its rules.

    Read from the sealed root, not regenerated: the planner depends on the counts at the
    time, and after its own import it would exclude the identities it used.
    """
    import hashlib

    from fleetlab.contracts.rollout import Job
    from fleetlab.worlds import supplement

    sealed = json.loads(Path(__file__).with_name("worlds_r6c_plan.json").read_text())
    assert tuple(sealed["order"]) == calibration.FOLLOWUP_ORDER
    assert sealed["budget_seconds"] == 5400
    assert all(stage["jobs"] <= 900 for stage in sealed["stages"].values())
    assert sum(c["added"] for c in sealed["selected_cells"]) == sum(
        sealed["stages"][name]["jobs"] for name in ("topups-1", "topups-2")
    )
    assert all(c["before"] + c["added"] == 50 for c in sealed["selected_cells"])
    root = Path(sealed["root"]).expanduser()
    if not (root / "plan.json").is_file():
        pytest.skip("the sealed calibration root is not on this host")
    plan = json.loads((root / "plan.json").read_text())
    assert plan["order"] == sealed["order"] and plan["stages"] == sealed["stages"]
    manifests: dict[str, list[Any]] = {}
    for name, stage in sealed["stages"].items():
        text = (root / f"{name}.jsonl").read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == stage["sha256"], name
        manifests[name] = [Job.model_validate_json(line).identity for line in text.splitlines()]
    seen: set[str] = set()
    for name, suite, control, decoys in calibration.FOLLOWUP_ARMS:
        by_arm: dict[tuple[tuple[str, str], ...], list[Any]] = {}
        for stage in (name, f"{name}-refine"):
            for identity in manifests[stage]:
                assert identity.suite == suite
                # The control faults keep their identities in every arm.
                assert [(f.kind, f.fault_id) for f in identity.faults[: len(control)]] == [
                    (kind, f"f{n + 1:02d}") for n, (kind, _) in enumerate(control)
                ]
                by_arm.setdefault(calibration.mechanism_members(identity.faults), []).append(
                    identity
                )
        assert set(by_arm) == {
            tuple(sorted(control)),
            *(tuple(sorted((*control, decoy))) for decoy in decoys),
        }
        attempts = [sorted(supplement._task_signature(i) for i in arm) for arm in by_arm.values()]
        assert all(arm == attempts[0] for arm in attempts)  # the same attempts in every arm
        assert len(set(attempts[0])) == len(attempts[0]) == 500  # fifty per task, none twice
    for identities in manifests.values():
        for identity in identities:
            assert identity.episode_id not in seen
            seen.add(identity.episode_id)
    assert len(seen) == sum(stage["jobs"] for stage in sealed["stages"].values())


def test_third_plan_is_the_sealed_one_and_keeps_its_rules() -> None:
    """The third job's manifests on disk are the plan this commit names.

    Read from the sealed root, as for the follow-up job: after its own import the
    planner would exclude the identities it used.
    """
    import hashlib

    from fleetlab.contracts.rollout import Job
    from fleetlab.worlds import supplement

    sealed = json.loads(Path(__file__).with_name("worlds_r6d_plan.json").read_text())
    assert tuple(sealed["order"]) == calibration.THIRD_ORDER
    assert sealed["budget_seconds"] == calibration.THIRD_BUDGET_SECONDS == 10800
    assert all(stage["jobs"] <= 900 for stage in sealed["stages"].values())
    order = list(calibration.THIRD_ORDER)
    for name, stage in sealed["stages"].items():
        # Every stage is followed at once by its sealed continuation.
        if name.endswith("-refine"):
            assert stage["parent"] == name.removesuffix("-refine")
            assert order.index(stage["parent"]) == order.index(name) - 1
        else:
            assert "parent" not in stage and order[order.index(name) + 1] == f"{name}-refine"
    arms = [stage for stage in calibration.THIRD_STAGES if stage[2] == "arms"]
    assert [(a["stages"][0], a["suite"]) for a in sealed["assays"]] == [
        (name, suite) for name, suite, *_ in arms
    ]
    assert all(calibration.paired_stage(name) for name, *_ in arms)
    assert not any(
        calibration.paired_stage(name)
        for name, _, kind, *_ in calibration.THIRD_STAGES
        if kind == "contexts"
    )
    root = Path(sealed["root"]).expanduser()
    if not (root / "plan.json").is_file():
        pytest.skip("the sealed calibration root is not on this host")
    plan = json.loads((root / "plan.json").read_text())
    assert plan["order"] == sealed["order"] and plan["stages"] == sealed["stages"]
    assert plan["budget_seconds"] == 10800
    manifests: dict[str, list[Any]] = {}
    for name, stage in sealed["stages"].items():
        text = (root / f"{name}.jsonl").read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == stage["sha256"], name
        manifests[name] = [Job.model_validate_json(line).identity for line in text.splitlines()]
    seen: set[str] = set()
    for name, suite, kind, *payload in calibration.THIRD_STAGES:
        first, rest = manifests[name], manifests[f"{name}-refine"]
        assert {i.suite for i in (*first, *rest)} == {suite}
        by_context: dict[tuple[tuple[str, str], ...], list[list[Any]]] = {}
        for part, identities in enumerate((first, rest)):
            for identity in identities:
                members = calibration.mechanism_members(identity.faults)
                by_context.setdefault(members, [[], []])[part].append(identity)
        if kind == "arms":
            control, decoys = payload
            assert set(by_context) == {
                tuple(sorted(control)),
                *(tuple(sorted((*control, decoy))) for decoy in decoys),
            }
            # The control faults keep their identities in every arm.
            for identity in (*first, *rest):
                assert [(f.kind, f.fault_id) for f in identity.faults[: len(control)]] == [
                    (kind_, f"f{n + 1:02d}") for n, (kind_, _) in enumerate(control)
                ]
            sizes = (20, 30)
        else:
            assert set(by_context) == {tuple(sorted(members)) for members in payload}
            sizes = (10, 40)
        attempts = []
        for parts in by_context.values():
            for identities, size in zip(parts, sizes, strict=True):
                per_task = [sum(1 for i in identities if i.task_index == t) for t in range(10)]
                assert per_task == [size] * 10
            signatures = [supplement._task_signature(i) for part in parts for i in part]
            assert len(set(signatures)) == len(signatures) == 500  # fifty per task, none twice
            attempts.append((sorted(signatures[: 10 * sizes[0]]), sorted(signatures)))
        if kind == "arms":
            assert all(arm == attempts[0] for arm in attempts)  # the same attempts in every arm
    for identities in manifests.values():
        for identity in identities:
            assert identity.episode_id not in seen
            seen.add(identity.episode_id)
    assert len(seen) == sum(stage["jobs"] for stage in sealed["stages"].values()) == 8000


def test_a_stage_is_started_only_when_its_bound_fits_what_is_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import subprocess

    assert calibration.stage_fits(600, None)  # a stage run by hand has no clock
    assert calibration.stage_fits(100, 470) and not calibration.stage_fits(100, 469.9)
    # The slowest complete stage so far: 112 attempts in 335 seconds.
    assert calibration.stage_fits(112, 510) and not calibration.stage_fits(112, 335)

    (tmp_path / "plan.json").write_text('{"order":["one","two","three"]}')
    commands: list[list[str]] = []

    class Process:
        pid = 123456789

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            commands.append(command)
            # The second stage reports that it would not fit and does not run.
            self.returncode = 0 if len(commands) == 1 else calibration.STAGE_WOULD_NOT_FIT

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", Process)
    assert calibration.refinement_run(tmp_path, 6, 5400) == 124
    result = json.loads((tmp_path / "execution.json").read_text())
    assert [c[c.index("--stage") + 1] for c in commands] == ["one", "two"]  # never "three"
    assert all(0 < float(c[c.index("--seconds") + 1]) <= 5400 - 15 for c in commands)
    assert [stage["exit"] for stage in result["stages"]] == [0, calibration.STAGE_WOULD_NOT_FIT]
    assert "not started" in result["state"]


def test_the_driver_takes_its_cap_from_the_sealed_plan(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text('{"order":[]}')
    with pytest.raises(ValueError, match="budget"):
        calibration.refinement_run(tmp_path, 6, 5401)  # ninety minutes unless the plan says
    with pytest.raises(ValueError, match="budget"):
        calibration.refinement_run(tmp_path, 7, 60)
    assert not (tmp_path / "execution.json").exists()
    longer = tmp_path / "longer"
    longer.mkdir()
    (longer / "plan.json").write_text('{"order":[],"budget_seconds":10800}')
    with pytest.raises(ValueError, match="budget"):
        calibration.refinement_run(longer, 6, 10801)
    assert calibration.refinement_run(longer, 6, 10800) == 0
    assert json.loads((longer / "execution.json").read_text())["seconds_limit"] == 10800
    greedy = tmp_path / "greedy"
    greedy.mkdir()
    (greedy / "plan.json").write_text('{"order":[],"budget_seconds":10801}')
    with pytest.raises(ValueError, match="budget"):  # no plan may name more than three hours
        calibration.refinement_run(greedy, 6, 60)


def test_calibration_timeout_stops_the_only_admission_before_another_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os
    import signal
    import subprocess
    import time

    (tmp_path / "plan.json").write_text('{"order":["one","two"]}')
    clock = [0.0]
    commands: list[list[str]] = []
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 123456789  # never signalled: os.killpg is replaced below
        returncode: int | None = None

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            commands.append(command)

        def wait(self, timeout: float | None = None) -> int:
            if not signals:
                assert timeout is not None
                clock[0] += timeout
                raise subprocess.TimeoutExpired(commands[-1], timeout)
            self.returncode = -signal.SIGTERM
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert calibration.refinement_run(tmp_path, 6, 5400) == 124
    result = json.loads((tmp_path / "execution.json").read_text())
    assert len(commands) == 1 and signals == [(Process.pid, signal.SIGTERM)]
    assert result["elapsed_seconds"] <= 5400
    assert result["stages"] == [{"name": "one", "seconds": 5385.0, "exit": -signal.SIGTERM}]
    assert "unfinished stage not calibration" in result["state"]
