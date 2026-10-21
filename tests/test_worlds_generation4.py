"""The fourth generation's sealed development worlds recompose to their seals.

A world of this generation is sealed when it is composed, before it runs: its hash, and
the digest of its forecast for all three windows with their intervals and of its
measured causes. Its recipe names the calibration jobs whose cells it stands on, so
that the seal holds whatever is imported afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.contracts.ids import FLEET_SCOPE
from fleetlab.worlds import calibration, compose, curves, estimate
from fleetlab.worlds.compose import Composer
from fleetlab.worlds.recipe import DEVELOPMENT, FOURTH_GENERATION, FOURTH_GENERATION_JOBS
from fleetlab.worlds.world import generate, world_hash

SEALS = json.loads(Path(__file__).with_name("worlds_generation4_seals.json").read_text())
TEXTS = json.loads(Path(__file__).with_name("worlds_round3_seals.json").read_text())["task_texts"]


def sealed(key: str) -> tuple[Any, dict[str, Any]]:
    texts = {Task(row["suite"], row["task_index"]): row["text"] for row in TEXTS}
    world = generate(DEVELOPMENT[key], SoftwareVersion.model_validate(SEALS["software"]), texts)
    return world, SEALS["worlds"][key]


@pytest.mark.parametrize("key", sorted(SEALS["worlds"]))
def test_a_sealed_world_recomposes_to_its_hash_and_its_forecast(key: str) -> None:
    recipe = DEVELOPMENT[key]
    world, seal = sealed(key)
    assert recipe.world_id == key and recipe.name == seal["design"]
    assert recipe.sealed_hash == seal["hash"] == world_hash(world)
    assert recipe.calibration_jobs == tuple(seal["calibration_jobs"])
    assert recipe.calibration_sources == tuple(seal["calibration_sources"])
    assert recipe.calibration_sources == curves.measured_sources(recipe.calibration_jobs)
    comp = world.composition
    expected = comp.expected
    assert comp.attempt == seal["attempt"]
    windows, causes = expected["window_forecasts"], expected["measured_cause_forecast"]
    assert expected["forecast_seal"] == seal["forecast_seal"]
    assert compose.forecast_seal(windows, causes) == seal["forecast_seal"]
    assert list(windows) == ["observed", "experiment", "scoring"]
    for name, row in seal["forecast_logged"].items():
        logged = windows[name]["logged"]
        assert windows[name]["episodes"] == row["episodes"]
        assert logged["expected"] == row["expected"]
        assert logged["prediction_interval"] == row["interval"]
        # No retained error: the interval is the forecast's own spread, rounded outward.
        assert windows[name]["model_error_points"] == [0.0, 0.0]
        assert logged["prediction_interval"] == pytest.approx(
            logged["conditional_prediction_interval"], abs=0.00011
        )
    assert causes["count"] == seal["measured_causes"]["count"]
    assert causes["probability_at_least_minimum"] == pytest.approx(
        seal["measured_causes"]["probability_at_least_six"]
    )
    assert expected["lowest_robot_window"] == seal["lowest_robot_window"]
    assert len(world.spec.fault_schedule) == seal["faults"]
    assert {
        suite: sum(1 for job in world.jobs if job.identity.suite == suite)
        for suite in seal["attempts_by_suite"]
    } == seal["attempts_by_suite"]


@pytest.mark.parametrize("key", sorted(SEALS["worlds"]))
def test_a_sealed_world_meets_the_rules_under_the_settings_its_recipe_states(key: str) -> None:
    recipe = DEVELOPMENT[key]
    assert recipe.estimator == "measured-v3" and recipe.kind == "development"
    for field, value in FOURTH_GENERATION.items():
        assert getattr(recipe, field) == value, field
    assert recipe.calibration_jobs is not None
    assert set(FOURTH_GENERATION_JOBS) <= set(recipe.calibration_jobs)
    world, seal = sealed(key)
    comp, spec = world.composition, world.spec
    expected = comp.expected
    # The band, with the forecast's own spread and no retained error.
    low, high = expected["window_forecasts"]["observed"]["logged"]["prediction_interval"]
    assert recipe.band == (0.60, 0.75) and 0.60 <= low < high <= 0.75
    # Six measured causes in at least 95% of the resamples.
    assert expected["measured_cause_forecast"]["probability_at_least_minimum"] >= 0.95
    assert expected["measured_cause_forecast"]["minimum"] == 6
    # The floor of 0.30 with its reserve of 0.03, on every robot and window.
    composer = Composer(recipe)
    ctx = composer.context(compose.Draft(patterns=spec.patterns))
    rows = estimate.robot_window_success(spec.fault_schedule, composer.slots, ctx, recipe.windows)
    assert len(rows) == 36
    lowest = min(min(row["true"], row["logged"]) for row in rows)
    assert lowest == pytest.approx(min(seal["lowest_robot_window"][k] for k in ("true", "logged")))
    assert lowest >= 0.33
    # Whether the world would also stand on the third generation's floor of 0.40.
    assert (lowest < 0.43) is seal["below_the_third_generations_floor"]
    # Every decoy flat at two points of fleet success on what covers its carriers.
    assert spec.patterns.decoys
    for fault in spec.fault_schedule:
        if fault.fault_id not in spec.patterns.decoys:
            continue
        margin = composer.decoy_margin(fault)
        assert margin == pytest.approx(0.06)
        assert calibration.context_decoy_confirmed(
            fault, list(spec.fault_schedule), list(composer.slots), margin, recipe.calibration_jobs
        )
    # The patterns of the planned difficulty, and opaque fault identities.
    patterns = spec.patterns
    assert len(patterns.confounders) == len(patterns.interactions) == len(patterns.drifts) == 1
    assert len(patterns.additive_pairs) == 1 and 1 <= len(patterns.decoys) <= 2
    assert all(f.fault_id.startswith("fx") and len(f.fault_id) == 34 for f in spec.fault_schedule)
    if recipe.config_share is not None:
        counted = [f for f in spec.fault_schedule if f.fault_id not in patterns.decoys]
        config = [f for f in counted if compose.fault_class(f.kind) in compose.CONFIG_CLASSES]
        assert len(config) / len(counted) >= recipe.config_share
    fleet_wide = [f for f in spec.fault_schedule if f.target.scope == FLEET_SCOPE]
    assert len(fleet_wide) == int(recipe.fleet_fault)
    # No more plain faults of one mechanism at one kind of scope than the recipe allows.
    plain = [
        f for f in spec.fault_schedule if comp.roles.get(f.fault_id) in ("plain", "confounder")
    ]
    counts: dict[tuple[str, str], int] = {}
    for fault in plain:
        mechanism = fault.kind
        if fault.kind == "software_release":
            mechanism = fault.params["carries"][0]["kind"]
        place = (mechanism, "robot" if fault.target.scope.startswith("r") else "site")
        counts[place] = counts.get(place, 0) + 1
    assert max(counts.values()) <= recipe.repeats


def test_a_sealed_world_does_not_compose_on_other_cells() -> None:
    recipe = DEVELOPMENT["w49"]
    with pytest.raises(ValueError, match="sealed recipe calibration changed"):
        Composer(recipe.model_copy(update={"calibration_jobs": recipe.calibration_jobs[:1]}))  # type: ignore[index]
    with pytest.raises(ValueError, match="sealed recipe calibration changed"):
        Composer(recipe.model_copy(update={"calibration_jobs": None}))
