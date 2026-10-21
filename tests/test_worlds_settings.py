"""The three settings a recipe states, each at the rule as it stood and at its new value.

The orchestrating session decided them on 2026-10-10 for the fourth generation's
development worlds (README, "Round 6c: three settings"): the decoy rule's margin as two
points of fleet success, no retained forecast error at composition, a floor of 0.30.
They are fields of `WorldRecipe`; the fields' defaults are the rules as round 6 built
them. The last tests cover the calibration jobs a sealed recipe names.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import pytest
from test_worlds_measured import cell, context
from test_worlds_task_guards import cells, task_recipe

from fleetlab.contracts.identity import Task
from fleetlab.worlds import base, calibration, compose, curves, estimate
from fleetlab.worlds.compose import Composer, Draft, make_fault
from fleetlab.worlds.recipe import DEVELOPMENT, FOURTH_GENERATION, WorldRecipe, planned_recipe

# The record as of the first two calibration jobs: what these tests read of the real
# cells does not move when a later job is imported.
JOBS = ("20261010-task-cells-r6", "20261010-task-cells-r6c")
SEEDS = compose.ASSESSMENT_SEEDS
BLUR = ("camera_degradation", "3 px blur")
DELAY = ("action_latency", "17 steps")
BACKDROP = ("backdrop_change", "1 walls and floor")
DIMMER = ("illumination_change", "2 stops dimmer")
NEW_FIELDS = ("decoy_margin", "decoy_margin_of", "retained_error", "calibration_jobs")


def fourth(name: str = "D3", seed: int = SEEDS[0], **changes: Any) -> WorldRecipe:
    recipe = planned_recipe(name, world_id="w99", seed=seed)  # type: ignore[arg-type]
    return recipe.model_copy(update={"calibration_jobs": JOBS, **changes})


# ---- which recipe states what --------------------------------------------------------------


def test_the_fields_default_to_the_rules_as_they_stood() -> None:
    fields = WorldRecipe.model_fields
    assert fields["decoy_margin"].default == calibration.DECOY_MARGIN == 0.02
    assert fields["decoy_margin_of"].default == "carriers"
    assert fields["retained_error"].default == "generation-3"
    assert fields["calibration_jobs"].default is None
    assert fields["robot_success_floor"].default is None
    for name in ("D2", "D3"):  # the third generation's sealed recipes do not change
        old = DEVELOPMENT[name]
        assert old.robot_success_floor == 0.40 and old.estimator == "suite-v1"
        assert (old.decoy_margin, old.decoy_margin_of) == (0.02, "carriers")
        assert old.retained_error == "generation-3" and old.calibration_jobs is None
        assert not set(NEW_FIELDS) & set(old.model_dump())
        assert not set(NEW_FIELDS) & set(json.loads(old.model_dump_json()))
    for change in (
        {"decoy_margin_of": "fleet"},
        {"decoy_margin": 0.03},
        {"retained_error": "none"},
        {"calibration_jobs": JOBS},
    ):
        with pytest.raises(ValueError, match="measured-v3 settings"):
            WorldRecipe.model_validate({**DEVELOPMENT["D3"].model_dump(), **change})


def test_fourth_generation_recipes_state_all_three_settings() -> None:
    assert FOURTH_GENERATION["robot_success_floor"] == 0.30
    planned = [
        planned_recipe("D2", world_id="w99", seed=SEEDS[0]),
        planned_recipe("D3", world_id="w99", seed=SEEDS[0]),
        planned_recipe("planned", world_id="w99", seed=SEEDS[0]),
    ]
    for r in planned:
        assert r.estimator == "measured-v3" and r.band == (0.60, 0.75)  # the band is frozen
        assert r.robot_success_floor == 0.30
        assert (r.decoy_margin, r.decoy_margin_of) == (0.02, "fleet")
        assert r.retained_error == ("generation-4" if r.name == "planned" else "none")
        # The search's target keeps the reserve of 0.03 inside the band on both sides.
        assert 0.63 <= r.aim[0] < r.aim[1] <= 0.72
        dumped = r.model_dump()
        assert dumped["decoy_margin_of"] == "fleet"
        assert dumped["retained_error"] == r.retained_error
        # What section 1 of the design note asks at the planned difficulty.
        assert r.confounders == r.interactions == r.drifts == r.additive_pairs == 1
        assert r.decoys == 1 and r.min_detectable == 6
    d2, d3, template = planned
    assert (
        template.model_copy(
            update={"name": "D2", "retained_error": d2.retained_error, "aim": d2.aim}
        )
        == d2
    )  # the template keeps the development layout, with its own acceptance error
    assert d2.fleet_fault and not d3.fleet_fault
    assert d3.config_share == pytest.approx(2 / 3) and d2.config_share is None
    assert (d3.interaction_scope, d3.plain, d3.repeats) == ("robot", 4, 3)
    # Three plain faults of one mechanism per kind of scope for the fleet recipes too;
    # the third generation's D2 allowed one.
    assert (d2.interaction_scope, d2.plain, d2.repeats) == ("robot", 4, 3)
    assert DEVELOPMENT["D2"].repeats == 1 and DEVELOPMENT["D3"].repeats == 3


# ---- the decoy rule's margin ----------------------------------------------------------------


def test_decoy_margin_on_the_carriers_is_the_fleet_margin_over_their_share() -> None:
    site = make_fault("f01", *BACKDROP, "s1", 2)
    late = make_fault("f02", *BACKDROP, "s1", 9)  # reaches half of the scoring window
    # A decoy type sits at a site; the share is counted the same for whatever scope
    # carries a fault, shown here with a setting on one robot.
    robot = make_fault("f03", "action_latency", "8 steps", "r01", 2)
    old = Composer(fourth(decoy_margin_of="carriers"))
    assert [old.decoy_margin(f) for f in (site, late, robot)] == [0.02] * 3
    new = Composer(fourth())
    # Two points of the fleet: six on the four robots of one site among three, twelve
    # where they carry it for half of the window, twenty-four on one robot of twelve.
    assert new.decoy_margin(site) == pytest.approx(0.06)
    assert new.decoy_margin(late) == pytest.approx(0.12)
    assert new.decoy_margin(robot) == pytest.approx(0.24)
    never = site.model_copy(update={"schedule": site.schedule.model_copy(update={"onset_day": 10})})
    assert new.decoy_margin(never) == math.inf  # no scoring attempt carries it
    wider = Composer(fourth(decoy_margin=0.03))
    assert wider.decoy_margin(site) == pytest.approx(0.09)


def test_decoy_rule_at_two_points_on_the_carriers_and_at_two_points_of_the_fleet() -> None:
    """Counts are (both succeed, only with the decoy, only without it, both fail)."""
    # A loss of 1.5 points whose interval reaches about minus five.
    unsure = [{"task_index": 0, "n": 400, "paired_counts": [340, 20, 26, 14]}]
    at_two = calibration.decoy_mix_interval(unsure, {0: 10})
    at_six = calibration.decoy_mix_interval(unsure, {0: 10}, 0.06)
    assert at_two["difference"] == at_six["difference"]
    lower = at_two["difference"][1]  # type: ignore[index]
    assert -0.06 < lower < -0.02
    assert not at_two["flat"] and at_six["flat"]
    assert (at_two["margin"], at_six["margin"]) == (0.02, 0.06)
    # Zero outside the interval is a cause at any margin.
    harmful = [{"task_index": 0, "n": 400, "paired_counts": [340, 5, 41, 14]}]
    assert not calibration.decoy_mix_interval(harmful, {0: 10}, 0.06)["flat"]
    assert not calibration.decoy_mix_interval(harmful, {0: 10}, 0.5)["flat"]
    # Too few pairs to exclude six points.
    thin = [{"task_index": 0, "n": 40, "paired_counts": [34, 2, 3, 1]}]
    assert not calibration.decoy_mix_interval(thin, {0: 10}, 0.06)["flat"]


def test_the_measured_decoys_under_the_blur_at_both_margins() -> None:
    """The record: the backdrop passes at six points, dimming stays refused."""
    even = {task: 10 for task in range(10)}

    def flat(decoy: tuple[str, str], margin: float) -> bool:
        mixes = {(curves.SPATIAL, tuple(sorted((BLUR, decoy)))): even}
        return calibration.decoy_flat(*decoy, mixes, margin, JOBS)

    assert not flat(BACKDROP, 0.02) and flat(BACKDROP, 0.06)
    assert not flat(DIMMER, 0.02) and not flat(DIMMER, 0.06) and not flat(DIMMER, 0.5)
    # A context without a same-attempt comparison is refused whatever the margin.
    unmeasured = {(curves.SPATIAL, tuple(sorted((BLUR, DELAY, BACKDROP)))): even}
    assert not calibration.decoy_flat(*BACKDROP, unmeasured, 0.5, JOBS)


def test_the_composer_asks_the_decoy_rule_with_its_recipes_margin() -> None:
    """A decoy on a spatial site under a site-wide 17-step delay, on the twenty seeds.

    Two points on the carriers confirm it on few sites, two points of the fleet on
    all; the search's grouped check and `check`'s slot-by-slot one agree at both.
    """
    counts = {"carriers": 0, "fleet": 0}
    for seed in SEEDS:
        composers = {basis: Composer(fourth(seed=seed, decoy_margin_of=basis)) for basis in counts}
        for site in ("s1", "s3"):
            delay = make_fault("f01", *DELAY, site, 1)
            decoy = make_fault("f02", *BACKDROP, site, 2)
            schedule = [delay, decoy]
            found = {}
            for basis, composer in composers.items():
                found[basis] = composer.decoys_flat(schedule, [decoy.fault_id])
                assert found[basis] == compose.decoy_context_confirmed(
                    decoy,
                    schedule,
                    composer.slots,
                    measured=True,
                    margin=composer.decoy_margin(decoy),
                    jobs=JOBS,
                )
                counts[basis] += found[basis]
            assert found["fleet"] or not found["carriers"]  # the wider margin refuses no more
    # Of forty scoring-window task mixes: two at two points on the carriers,
    # all forty at two points of the fleet.
    assert counts == {"carriers": 2, "fleet": 40}


# ---- the retained error ---------------------------------------------------------------------


def test_the_retained_error_a_recipe_names_widens_its_interval_or_does_not() -> None:
    assert estimate.RETAINED_ERRORS["generation-3"] is estimate.MODEL_ERRORS
    assert estimate.MODEL_ERRORS == {
        "observed": (-0.007, 0.060),
        "experiment": (-0.037, 0.066),
        "scoring": (-0.048, 0.054),
    }
    assert set(estimate.RETAINED_ERRORS["none"].values()) == {(0.0, 0.0)}
    episodes = [estimate.Episode(curves.SPATIAL, (), 0)] * 1008
    old = context({(): 0.68}, 1000)
    new = replace(old, retained_error="none")
    for window, (down, up) in estimate.MODEL_ERRORS.items():
        with_error = estimate.measured_prediction(episodes, old, window=window)
        without = estimate.measured_prediction(episodes, new, window=window)
        a, b = without["logged"]["conditional_prediction_interval"]
        assert with_error["logged"]["conditional_prediction_interval"] == [a, b]
        assert without["logged"]["prediction_interval"] == pytest.approx([a, b], abs=0.00011)
        assert with_error["logged"]["prediction_interval"] == pytest.approx(
            [a + down, b + up], abs=0.00011
        )
        assert without["model_error_points"] == [0.0, 0.0]
        assert with_error["model_error_points"] == [round(100 * down, 3), round(100 * up, 3)]
        assert without["retained_error"] == "none" and "retained_error" not in with_error
    assert Composer(fourth()).context(Draft()).retained_error == "none"
    assert Composer(fourth(retained_error="generation-3")).context(Draft()).retained_error == (
        "generation-3"
    )


@pytest.mark.parametrize(
    ("retained_error", "rate", "refused"),
    (
        # 68%: the interval with its own spread is inside 60 to 75%; six points more are not.
        ("generation-3", 0.68, True),
        ("none", 0.68, False),
        # 65%: inside under either rule.
        ("generation-3", 0.65, False),
        ("none", 0.65, False),
        # 72%: the own spread of this fleet's 672 observed attempts alone leaves the band.
        ("none", 0.72, True),
    ),
)
def test_admission_holds_the_interval_with_the_named_error_inside_the_band(
    monkeypatch: pytest.MonkeyPatch, retained_error: str, rate: float, refused: bool
) -> None:
    comp = Composer(
        task_recipe(
            estimator="measured-v3",
            min_detectable=6,
            band=(0.60, 0.75),
            aim=(0.61, 0.74),
            retained_error=retained_error,
        )
    )
    assert comp.recipe.retained_error == retained_error
    ctx = replace(context({(): rate}, 100000), retained_error=retained_error)
    monkeypatch.setattr(comp, "context", lambda draft: ctx)
    monkeypatch.setattr(
        estimate,
        "measured_causes",
        lambda *a, **k: {"probability_at_least_minimum": 1.0, "faults": {}},
    )
    reasons, expected = comp.check(Draft(), [])
    assert any("observed prediction interval" in r for r in reasons) is refused
    assert not any("expected observed success" in r for r in reasons)  # inside the aim
    if not refused:
        assert not reasons
        windows = expected["window_forecasts"]
        assert list(windows) == ["observed", "experiment", "scoring"]
        assert windows["observed"] == expected["observed_forecast"]
        errors = estimate.RETAINED_ERRORS[retained_error]
        for name, forecast in windows.items():
            assert forecast["window"] == name
            assert forecast["model_error_points"] == [round(100 * x, 3) for x in errors[name]]
        causes = expected["measured_cause_forecast"]
        assert expected["forecast_seal"] == compose.forecast_seal(windows, causes)
        moved = json.loads(json.dumps(windows))
        moved["scoring"]["logged"]["expected"] += 0.0001
        assert compose.forecast_seal(moved, causes) != expected["forecast_seal"]


# ---- the floor -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rates", "floor", "holds"),
    (
        # A robot window at 36%: under 0.40 with its reserve of 0.03, above 0.30 with it.
        ((0.12, 0.60), 0.40, False),
        ((0.12, 0.60), 0.30, True),
        # At 32% the reserve refuses it under the new floor too.
        ((0.04, 0.60), 0.30, False),
        # At 44% both floors hold.
        ((0.28, 0.60), 0.40, True),
        ((0.28, 0.60), 0.30, True),
    ),
)
def test_the_floor_at_the_third_and_the_fourth_generations_value(
    rates: tuple[float, float], floor: float, holds: bool
) -> None:
    fault = make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0)
    the_curves = cells(*rates)
    comp = Composer(task_recipe(robot_success_floor=floor), the_curves)
    comp.slots = tuple(
        base.Slot("r01", "s1", day, 0, Task(curves.SPATIAL, day % 2), 0) for day in range(10)
    )
    ctx = comp.context(Draft())
    assert ctx.success_margin == 0.03  # the reserve is the same under both floors
    rows = estimate.robot_window_success([fault], comp.slots, ctx, comp.recipe.windows)
    assert all(row["true"] == pytest.approx(sum(rates) / 2) for row in rows)
    assert comp.floor_holds([fault], Draft()) is holds
    reasons, _ = comp.check(Draft(), [fault])
    assert any(f"below success floor {floor}" in r for r in reasons) is not holds


# ---- the calibration jobs a sealed recipe names ------------------------------------------------


def test_cells_as_of_named_jobs_are_the_record_without_the_other_jobs() -> None:
    raw = curves.refinement()
    names = [root.rsplit("/", 1)[-1] for root in raw["calibration_roots"]]
    assert names[:2] == list(JOBS)
    everything = curves.refinement(tuple(names))
    assert everything["cells"] == raw["cells"]
    assert everything["decoy_assays"] == raw["decoy_assays"]
    assert curves.measured_cells_of(tuple(names)) == curves.measured_cells()
    first = curves.refinement(JOBS[:1])
    other = raw["calibration_roots"][1]
    assert all(not path.startswith(other) for row in first["cells"] for path in row["sources"])
    for row in first["cells"]:
        for field in ("n", "successes", "logged_successes"):
            assert row[field] == sum(part[field] for part in row["sources"].values())
    # The second job measured the 17-step delay with the heavier batch (ten attempts on
    # this task, beside five that a calibration world happened to hold) and the two
    # decoys under the delay; the first job's record has neither.
    batch = (
        curves.SPATIAL,
        0,
        tuple(sorted((DELAY, ("object_batch_variant", "4 mass doublings")))),
    )
    assert curves.measured_cells_of(JOBS)[batch].n == 15
    assert curves.measured_cells_of(JOBS[:1])[batch].n == 5

    def contexts(jobs: tuple[str, ...]) -> set[tuple[tuple[str, ...], ...]]:
        return {
            tuple(sorted(map(tuple, a["members"])))
            for a in calibration.decoy_assays(*BACKDROP, jobs)
        }

    under_delay = tuple(sorted((DELAY, BACKDROP)))
    assert under_delay in contexts(JOBS) and under_delay not in contexts(JOBS[:1])
    assert (BACKDROP,) in contexts(JOBS[:1])  # the solo comparisons belong to no job
    with pytest.raises(ValueError, match="not imported"):
        curves.refinement(("20261010-task-cells-none",))
    assert curves.measured_sources(JOBS)[:-1] == curves.measured_sources()[:-1]
    assert curves.measured_sources(JOBS)[-1] != curves.measured_sources(JOBS[:1])[-1]
    pinned = Composer(fourth(calibration_jobs=JOBS[:1]))
    assert pinned.cells() is curves.measured_cells_of(JOBS[:1])
    assert Composer(fourth(calibration_jobs=None)).cells() is curves.measured_cells()


def test_a_later_job_moves_neither_the_named_cells_nor_their_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the pin is for: a recipe sealed on two jobs composes the same after a third."""
    text = curves._supplement_text()
    assert text is not None
    before = curves.refinement(JOBS)
    digest = curves.measured_sources(JOBS)
    whole = curves.measured_sources()
    raw = json.loads(text)
    record = raw["task_refinement"]
    later = record["calibration_roots"][0].rsplit("/", 1)[0] + "/20991231-task-cells-later"
    record["calibration_roots"].append(later)
    grown = record["cells"][0]
    part = {"n": 7, "successes": 3, "logged_successes": 3}
    grown["sources"][f"{later}/one/run"] = part
    for field, value in part.items():
        grown[field] += value
    record["cells"].append(
        {
            "suite": curves.SPATIAL,
            "task_index": 0,
            "members": [list(BLUR), list(DELAY), list(BACKDROP)],
            **part,
            "sources": {f"{later}/one/run": part},
        }
    )
    record["decoy_assays"].append({**record["decoy_assays"][0], "sources": [f"{later}/one/run"]})
    monkeypatch.setattr(curves, "_supplement_text", lambda: json.dumps(raw))
    assert curves.refinement(JOBS) == before
    assert curves.measured_sources(JOBS) == digest
    assert curves.measured_sources() != whole  # the record as a whole did change
    with_later = curves.refinement((*JOBS, "20991231-task-cells-later"))
    assert len(with_later["cells"]) == len(before["cells"]) + 1
    assert len(with_later["decoy_assays"]) == len(before["decoy_assays"]) + 1


def test_a_sealed_recipe_refuses_cells_other_than_those_it_names() -> None:
    sealed = fourth(calibration_sources=curves.measured_sources(JOBS))
    assert Composer(sealed).cells() is curves.measured_cells_of(JOBS)
    with pytest.raises(ValueError, match="sealed recipe calibration changed"):
        Composer(sealed.model_copy(update={"calibration_jobs": JOBS[:1]}))


def test_synthetic_cell_helper_is_what_these_tests_assume() -> None:
    assert cell(0.5, 10).sufficient is False and cell(0.5, 50).sufficient
