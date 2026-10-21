"""Regression evidence for paired qualification and the hidden additive mark."""

import json
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest

from fleetlab.contracts.identity import Task
from fleetlab.contracts.world import Patterns
from fleetlab.worlds import base, curves, detect, estimate
from fleetlab.worlds.compose import ADDITIVE_PAIR, INTERACTION_PAIRS, make_fault
from fleetlab.worlds.recipe import SMALL_WINDOWS


def test_library_joints_override_unpaired_world_supplement() -> None:
    c = curves.load()
    wear, moved = c.joint(*ADDITIVE_PAIR, curves.SPATIAL)
    assert not moved and wear.n == 300
    assert wear.single_successes == (281, 258)
    assert wear.successes == 234 and wear.healthy_successes == 296
    assert wear.interaction == (-0.03, -0.07, 0.01)
    assert wear.interaction_role == "additive"
    delay, _ = c.joint(*INTERACTION_PAIRS[0], curves.SPATIAL)
    assert delay.single_successes == (277, 273)
    assert delay.successes == 27 and delay.healthy_successes == 296
    assert (
        delay.successes - sum(delay.single_successes) + delay.healthy_successes
    ) / delay.n == pytest.approx(-227 / 300)
    assert delay.interaction == (-0.7567, -0.8067, -0.7033)
    assert delay.interaction_role == "interaction"


def test_object_pair_uses_the_handed_off_same_attempt_record() -> None:
    c = curves.load()
    joint, moved = c.joint(*INTERACTION_PAIRS[0], curves.OBJECT)
    assert not moved and joint.n == 300
    assert joint.healthy_successes == 294 and joint.single_successes == (188, 282)
    assert joint.successes == 0
    assert (
        joint.successes - sum(joint.single_successes) + joint.healthy_successes
    ) / joint.n == pytest.approx(-176 / 300)
    assert joint.interaction == (-0.5867, -0.6467, -0.5233)
    assert joint.interaction_role == "interaction"
    release = make_fault(
        "f01",
        "software_release",
        "action_latency|14 steps + policy_knob|10 actions per chunk",
        "r01",
        0,
        nominal_version="4.2.1",
        version="4.9.0",
    )
    ratio = c.ratio(release, curves.OBJECT)
    assert ratio.true == ratio.logged == 0.0
    assert ratio.effect and not ratio.transferred
    assert f"measured joint {joint.run}" in ratio.source


def test_library_source_paths_survive_point_and_joint_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = resources.files
    document = json.loads(files("fleetlab.faults").joinpath("curves.json").read_text())
    point = next(
        p
        for p in document["supplement"]["points"]
        if p["kind"] == "action_latency" and p["label"] == "14 steps"
    )
    point["run_dirs"] = ["/measurement/singles/run"]
    joint = next(
        j
        for j in document["supplement"]["joints"]
        if j["name"] == "delay+short-chunks" and j["suite"] == curves.OBJECT
    )
    joint["run_dirs"] = ["/measurement/paired/run"]
    (tmp_path / "curves.json").write_text(json.dumps(document))
    monkeypatch.setattr(
        resources,
        "files",
        lambda package: tmp_path if package == "fleetlab.faults" else files(package),
    )
    curves.load.cache_clear()
    try:
        c = curves.load()
        assert (
            c.point("action_latency", "14 steps", curves.OBJECT)[0].run
            == "/measurement/singles/run"
        )
        assert c.joint(*INTERACTION_PAIRS[0], curves.OBJECT)[0].run == "/measurement/paired/run"
    finally:
        curves.load.cache_clear()


def test_unpaired_ratios_cannot_qualify_an_interaction() -> None:
    c = curves.load()
    joint, _ = c.joint(*INTERACTION_PAIRS[0], curves.SPATIAL)
    unpaired = replace(joint, single_successes=None, interaction=None)
    assert unpaired.singles is not None  # legacy format had ratios, not paired evidence
    assert not unpaired.paired and unpaired.interaction_role == "unmeasured"


@pytest.mark.parametrize(
    "label",
    [
        "release with a 14-step delay and 10 actions per chunk",
        "action_latency|14 steps + policy_knob|10 actions per chunk",
    ],
)
def test_release_uses_measured_joint_even_when_renumbered(label: str) -> None:
    c = curves.load()
    release = make_fault(
        "f01", "software_release", label, "r01", 0, nominal_version="4.2.1", version="4.9.0"
    )
    ratio = c.ratio(release, curves.SPATIAL)
    assert ratio.true == pytest.approx(8 / 98)
    assert ratio.logged == ratio.true
    assert ratio.effect and not ratio.transferred
    assert "measured release" in ratio.source
    no_release = replace(
        c, points={key: p for key, p in c.points.items() if key[0] != "software_release"}
    )
    joint_ratio = no_release.ratio(release, curves.SPATIAL)
    assert joint_ratio.true == pytest.approx(27 / 296)
    assert "s3-joint-delay-chunks" in joint_ratio.source


def test_detect_uses_same_attempt_singles_and_percentage_point_denominator() -> None:
    c = curves.load()
    a = make_fault("f01", *ADDITIVE_PAIR[0], "r01", 0)
    b = make_fault("f02", *ADDITIVE_PAIR[1], "s1", 0)
    # Make unrelated singles absurdly harmful. Pair evidence must not change.
    points = dict(c.points)
    key = (a.kind, curves.SPATIAL, ADDITIVE_PAIR[0][1])
    points[key] = replace(points[key], successes=0)
    c = replace(c, points=points)
    ctx = estimate.Context(c, frozenset(), frozenset(), (("f01", "f02"),))
    slots = [base.Slot("r01", "s1", 8, 0, Task(curves.SPATIAL, 0), 0)]
    table = detect.detectability(
        [a, b], Patterns(interactions=[("f01", "f02")]), set(), slots, SMALL_WINDOWS, ctx
    )
    pair = table.pairs[0]
    evidence = pair["by_suite"][0]
    assert evidence["n"] == 300
    assert evidence["loss_points"] == {"f01": 5.0, "f02": 12.667, "joint": 20.667}
    assert pair["interaction_points"] == [-3.0, -7.0, 1.0]
    assert not pair["superadditive"]


def test_decoy_neutrality_is_suite_specific() -> None:
    c = curves.load()
    light = make_fault("f01", "illumination_change", "2 stops dimmer", "s1", 0)
    ctx = estimate.Context(c, frozenset({"f01"}), frozenset())
    assert estimate.probability([light], curves.SPATIAL, ctx) == c.healthy[curves.SPATIAL]
    assert estimate.probability([light], curves.OBJECT, ctx) == pytest.approx(
        c.healthy[curves.OBJECT] * c.point(light.kind, "2 stops dimmer", curves.OBJECT)[0].ratio
    )
    inert = make_fault("f01", "calibration_update", "scene calibration rewritten", "r01", 0)
    assert estimate.probability([inert], curves.OBJECT, ctx) == c.healthy[curves.OBJECT]


def test_additive_pattern_serialization_preserves_legacy_shape() -> None:
    assert "additive_pairs" not in Patterns().model_dump()
    marked = Patterns(additive_pairs=[("f01", "f02")])
    assert marked.fault_ids() == {"f01", "f02"}
    assert not marked.excluded_from_replay()
    assert Patterns.model_validate_json(marked.model_dump_json()) == marked
