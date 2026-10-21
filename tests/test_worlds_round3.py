"""Recorded independent calibration and sealed development-world regressions."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import pytest

from fleetlab.contracts.actions import catalogue
from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.faults.records import health_shift
from fleetlab.worlds import curves, supplement
from fleetlab.worlds.compose import Composer
from fleetlab.worlds.recipe import DEVELOPMENT
from fleetlab.worlds.world import generate, generator_record, world_hash, write_world


def test_imported_pair_is_the_independent_confirmation_not_the_selected_screen() -> None:
    record = json.loads(resources.files("fleetlab.faults").joinpath("curves.json").read_text())
    proof = record["worlds_mild_pair_round3"]
    assert proof["selected_dose"] == [8, 15]
    assert proof["screen_used_only_for_selection"]
    assert proof["n_confirmation_per_arm"] == 100
    assert proof["screen"]["comparisons"] == proof["screen"]["identical"] == 200
    assert proof["confirmation"]["comparisons"] == proof["confirmation"]["identical"] == 81
    assert proof["confirmation"]["jobs"] == 800
    assert proof["confirmation"]["manifest_sha256"] == (
        "fb1ce285ddfb778dfa4963035aa3de6e266cac616843306a72eb9d9caf208ee2"
    )
    assert proof["eligible_suites"] == [curves.SPATIAL]
    usable = [p for entry in record["types"].values() for p in entry["points"]]
    usable += record["supplement"]["points"] + record["supplement"]["joints"]
    assert not any(
        "/worlds-v3/20261010-mild-pair/screen/" in path
        for point in usable
        for path in point.get("run_dirs", [])
    )
    assert all(j["dose"] == [8, 15] for j in record["supplement"]["joints"] if "dose" in j)
    c = curves.load()
    for suite, successes, singles, interaction, eligible in (
        (curves.SPATIAL, 55, (97, 99), (-0.43, -0.53, -0.33), True),
        (curves.OBJECT, 31, (95, 97), (-0.63, -0.74, -0.52), False),
    ):
        rows = [
            j
            for j in record["supplement"]["joints"]
            if j["name"] == "delay-8+chunk-15" and j["suite"] == suite
        ]
        assert len(rows) == 1
        raw = rows[0]
        assert raw["single_run_dirs"] == [raw["run_dirs"], raw["run_dirs"]]
        assert raw["healthy_run_dirs"] == raw["run_dirs"]
        assert supplement.mild_eligible(raw, 0.40) is eligible
        joint, moved = c.joint(
            ("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"), suite
        )
        assert not moved and joint.paired and joint.n == 100
        assert joint.successes == successes and joint.healthy_successes == 98
        assert joint.single_successes == singles
        assert joint.interaction == interaction
        assert joint.ratio == pytest.approx(successes / 98)
        assert joint.run == proof["confirmation"]["run"]
        for member, count in zip(joint.members, singles, strict=True):
            point, moved = c.point(*member, suite)
            assert not moved and point.successes == count and point.healthy_successes == 98
            assert point.run == joint.run
    # The smaller confirmation controls cannot replace the 1,000-attempt baselines.
    assert c.healthy_counts == {curves.SPATIAL: (968, 1000), curves.OBJECT: (985, 1000)}


def test_negative_fleet_blur_measurement_is_preserved_but_not_accepted() -> None:
    record = json.loads(resources.files("fleetlab.faults").joinpath("curves.json").read_text())
    proof = record["worlds_mild_pair_round3"]
    assert not proof["fleet_blur_accepted"]
    assert not proof["fleet_blur_effective_on_both_suites"]
    assert proof["fleet_blur"]["comparisons"] == proof["fleet_blur"]["identical"] == 20
    assert proof["fleet_blur"]["jobs"] == 200
    assert not proof["fleet_blur"]["incidents"]
    c = curves.load()
    for suite, successes, role in (
        (curves.SPATIAL, 95, "small_cause"),
        (curves.OBJECT, 80, "effect"),
    ):
        point, transferred = c.point("camera_degradation", "3 px blur", suite)
        assert not transferred and point.n == 100
        assert point.successes == successes and point.healthy_successes == 98
        assert point.role == role
    r = DEVELOPMENT["D3"].model_copy(
        update={
            "calibration_sources": None,
            "sealed_hash": None,
            "fleet_fault": True,
            "fleet_image_blur": 3,
            "site_suites": (("s1", curves.SPATIAL),),
            "interaction_single_loss_points": 10,
        }
    )
    assert "fleet" not in Composer(r, c).draft(0).roles.values()


@pytest.mark.parametrize("name", ["D2", "D3"])
def test_development_recipe_reproduces_its_real_sealed_world_and_predictions(
    name: str, tmp_path: Path
) -> None:
    """Real recorded curves, fixed software/texts; no cache, engine or synthetic curves."""
    fixture = json.loads(Path(__file__).with_name("worlds_round3_seals.json").read_text())
    expected = fixture["worlds"][name]
    recipe = DEVELOPMENT[name]
    assert recipe.sealed_hash == expected["hash"]
    texts = {Task(r["suite"], r["task_index"]): r["text"] for r in fixture["task_texts"]}
    world = generate(recipe, SoftwareVersion.model_validate(fixture["software"]), texts)
    record = generator_record(world)
    assert world_hash(world) == expected["hash"]
    assert record["predictions_sha256"] == expected["predictions_sha256"]
    assert world.composition.fault_id_map == expected["fault_id_map"]
    # Baseline digests were read from the immutable round-3 files before the ID
    # change, not generated by this implementation. Reverse only token references
    # and the consequent seal. All eleven files must recover their exact old bytes,
    # including authors, health, estimates, truth flags and all 1,680 job identities.
    write_world(world, tmp_path, base_runs=expected["round3_base_runs"])
    assert {p.name for p in tmp_path.iterdir()} == set(expected["round3_file_sha256"])
    for filename, digest in expected["round3_file_sha256"].items():
        content = (tmp_path / filename).read_text()
        for old, token in expected["fault_id_map"].items():
            content = content.replace(token, old)
        content = content.replace(expected["hash"], expected["round3_hash"])
        # The batch-4 merge changes source names to findable directories, never
        # measured cells. Reverse only that enumerated provenance migration.
        content = content.replace('  "estimator": "suite-v1",\n', "")
        migration = fixture["provenance_only_merge"]
        content = content.replace(migration["task_digest"], migration["old_task_digest"])
        for current, previous in migration["source_replacements"].items():
            content = content.replace(current, previous)
        content = content.replace(migration["library_digest"], migration["old_library_digest"])
        assert hashlib.sha256(content.encode()).hexdigest() == digest, filename
    assert tuple(record["curves"]) == recipe.calibration_sources
    assert not world.composition.assumed_joint and not recipe.provisional
    assert len(world.jobs) == 1680 and len(world.jobs_in("observed")) == 1008
    assert 0.60 <= world.predictions["observed"]["logged"]["expected"] <= 0.75
    counted = world.detect.observed_counted()
    assert [u.unit for u in counted] == expected["observed_causes"]
    assert len(counted) >= 6 and all(u.effect["observed"] >= 2 for u in counted)
    assert record["composition"]["lowest_robot_day"] == expected["lowest_robot_day"]
    assert min(expected["lowest_robot_day"][k] for k in ("true", "logged")) >= 0.40
    assert len(catalogue(world.fleet, world.task_classes)) >= 10 * len(world.spec.fault_schedule)
    patterns = world.spec.patterns
    assert patterns.confounders and patterns.decoys and patterns.interactions
    assert patterns.additive_pairs and patterns.drifts
    for fault in world.spec.fault_schedule:
        if fault.fault_id in patterns.drifts:
            assert fault.schedule.drift is not None and fault.schedule.drift.shape == "linear"
            assert fault.schedule.onset_day == 0 and health_shift(fault)
            assert world.fault_table.by_id()[fault.fault_id].drift_crossing_day == 8
    for job in world.jobs:
        if job.identity.fault_ids & set(patterns.decoys):
            assert job.identity.suite == curves.SPATIAL
