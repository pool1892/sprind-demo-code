"""Rules for future worlds; synthetic cells test boundaries, not calibration evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_worlds_review import paired_table, recipe
from test_worlds_v1 import SOFTWARE, TEXTS

from fleetlab.contracts.faults import Drift
from fleetlab.contracts.identity import Task
from fleetlab.contracts.world import Patterns
from fleetlab.worlds import base, curves, detect, estimate, supplement
from fleetlab.worlds.compose import (
    Composer,
    Composition,
    Draft,
    decoy_context_confirmed,
    drift_precursor,
    identify,
    make_fault,
)
from fleetlab.worlds.recipe import WorldRecipe, external_recipe, private_root, stated_fields
from fleetlab.worlds.world import build, fault_table


def task_recipe(**changes: Any) -> WorldRecipe:
    return recipe(**{"estimator": "task-v2", "world_id": "w99", "seed": 2**62 + 937, **changes})


def cells(first: float, second: float, *, healthy: float = 0.95) -> curves.Curves:
    """One measured mechanism behaves differently on two tasks of the same suite."""
    c = curves.load()
    key = "camera_pose_offset", curves.SPATIAL, "3 deg scene pan"
    p = c.points[key]
    tasks = {}
    for t, rate in enumerate((first, second)):
        point = replace(
            p,
            n=1000,
            successes=rate * 1000,
            logged_successes=rate * 1000,
            healthy_successes=healthy * 1000,
            role="effect",
        )
        tasks[curves.SPATIAL, t] = curves.Curves(
            {key: point},
            (),
            {curves.SPATIAL: healthy},
            (),
            {curves.SPATIAL: (round(healthy * 1000), 1000)},
        )
    return replace(c, tasks=tasks)


def test_every_estimate_uses_task_index_including_prediction_groups() -> None:
    c = cells(0.90, 0.20)
    fault = make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0)
    ctx = estimate.Context(c, frozenset(), frozenset(), task_aware=True)
    good = [estimate.Episode(curves.SPATIAL, (fault,), 0)] * 20
    bad = [estimate.Episode(curves.SPATIAL, (fault,), 1)] * 20
    assert estimate.mean_success(good, ctx) == pytest.approx(0.90)
    assert estimate.mean_success(bad, ctx) == pytest.approx(0.20)
    assert estimate.removal_effect(["f01"], good, ctx, False) == pytest.approx(5)
    assert estimate.removal_effect(["f01"], bad, ctx, False) == pytest.approx(75)
    a = estimate.prediction(good, ctx, draws=32)
    b = estimate.prediction(bad, ctx, draws=32)
    mix = estimate.prediction(good + bad, ctx, draws=32)
    assert a["true"]["expected"] == 0.9 and b["true"]["expected"] == 0.2
    assert mix["true"]["expected"] == 0.55
    assert 0.4 < mix["true"]["prediction_interval"][0] < 0.55
    assert 0.55 < mix["true"]["prediction_interval"][1] < 0.7
    slots = [base.Slot("r01", "s1", 0, i, Task(curves.SPATIAL, t), 0) for i, t in enumerate([0, 1])]
    assert estimate.robot_day_success([fault], slots, ctx)[0]["true"] == pytest.approx(0.55)
    from fleetlab.worlds.recipe import SMALL_WINDOWS

    for episodes, gain in ((good, 5), (bad, 75)):
        table = detect.detectability(
            [fault],
            Patterns(),
            set(),
            [],
            SMALL_WINDOWS,
            ctx,
            by_window=dict.fromkeys(detect.WINDOWS, episodes),
        )
        assert table.units[0].effect["observed"] == pytest.approx(gain)
    with pytest.raises(KeyError, match="no task calibration"):
        estimate.mean_success([estimate.Episode(curves.SPATIAL, (fault,))], ctx)
    with pytest.raises(KeyError):
        estimate.probability(
            [make_fault("f02", "action_latency", "14 steps", "r01", 0)],
            curves.SPATIAL,
            ctx,
            task_index=0,
        )


def test_floor_has_a_margin_and_settling_uses_the_same_rule() -> None:
    r = task_recipe(robot_success_floor=0.4)
    fault = make_fault("f01", "camera_pose_offset", "3 deg scene pan", "r01", 0)
    for probability, allowed in ((0.429, False), (0.431, True)):
        composer = Composer(r, cells(probability, 0.9))
        draft = Draft(fixed=[fault])
        assert composer.floor_holds([fault], draft) is allowed
        reasons, _ = composer.check(draft, [fault])
        assert any("below success floor" in s for s in reasons) is not allowed


def test_observed_sizing_margin_is_additional_to_two_point_detectability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = paired_table(900, 900, 500).units[0]
    comp = Composer(task_recipe(min_detectable=1), cells(0.9, 0.2))
    for effect, allowed in ((2.49, False), (2.51, True)):
        cause = replace(template, effect=dict.fromkeys(detect.WINDOWS, effect))
        table = detect.DetectTable([cause], [], [], {})
        assert cause.detectable and cause.counts_observed
        monkeypatch.setattr(detect, "detectability", lambda *a, current=table, **kw: current)
        messages, _ = comp.check(Draft(), [])
        assert any("0 observed detectable causes" in s for s in messages) is not allowed


def test_fleet_band_has_a_three_point_reserve() -> None:
    c = cells(0.731, 0.731, healthy=0.731)
    comp = Composer(task_recipe(band=(0.6, 0.75), aim=(0.61, 0.74)), c)
    messages, _ = comp.check(Draft(), [])
    assert any("expected observed success" in s for s in messages)


def test_decoy_requires_a_measured_context_even_for_other_decoys_and_releases() -> None:
    decoy = make_fault("f01", "illumination_change", "2 stops dimmer", "s1", 0)
    slots = [base.Slot("r01", "s1", day, 0, Task(curves.SPATIAL, 0), 0) for day in (0, 5, 9)]
    assert decoy_context_confirmed(decoy, [decoy], slots)
    inert = make_fault("f02", "calibration_update", "scene calibration rewritten", "r01", 0)
    assert decoy_context_confirmed(decoy, [decoy, inert], slots)
    for companion in (
        make_fault(
            "f02",
            "software_release",
            "camera_degradation|4 px blur",
            "fleet",
            5,
            nominal_version="4.2.0",
            version="4.3.0",
        ),
        make_fault("f02", "backdrop_change", "1 walls and floor", "s1", 0),
        make_fault("f02", "action_latency", "8 steps", "r01", 8),
    ):
        assert not decoy_context_confirmed(decoy, [decoy, companion], slots)
        composer = Composer(task_recipe(tasks=(Task(curves.SPATIAL, 7),)))
        messages, _ = composer.check(Draft(patterns=Patterns(decoys=["f01"])), [decoy, companion])
        assert any("covering faults" in s for s in messages)
    elsewhere = inert.model_copy(
        update={
            "kind": "action_latency",
            "target": inert.target.model_copy(update={"scope": "r02"}),
        }
    )
    assert decoy_context_confirmed(decoy, [decoy, elsewhere], slots)


def test_precursor_is_two_noisy_observed_readings_before_its_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.faults import records

    fault = make_fault(
        "f01",
        "camera_pose_offset",
        "3 deg scene pan",
        "r01",
        0,
        level=0,
        drift=Drift(shape="linear", rate=1),
    )
    r = task_recipe(warnable_drift=True)
    for multiplier, visible in ((0.8, False), (8.2, False), (11.0, True)):
        monkeypatch.setattr(
            records,
            "health_shift",
            lambda f, scale=multiplier: {"calibration_residual": scale * f.level},
        )
        assert (drift_precursor(fault, r, 8) is not None) is visible
    assert drift_precursor(fault, r, 4) is None
    assert drift_precursor(fault, r, None) is None


def test_fault_table_enforces_observed_carriers_even_with_large_effect() -> None:
    table = paired_table(900, 900, 500, n=19)
    faults = [
        make_fault("f01", "action_latency", "14 steps", "r01", 0),
        make_fault("f02", "policy_knob", "10 actions per chunk", "r01", 0),
    ]
    comp = Composer(task_recipe())
    ctx = estimate.Context(curves.load(), frozenset(), frozenset(), (("f01", "f02"),))
    eps = [estimate.Episode(curves.SPATIAL, tuple(faults))] * 19
    truth = fault_table(
        comp.spec(faults, Patterns(interactions=[("f01", "f02")])),
        [],
        table,
        ctx,
        dict.fromkeys(detect.WINDOWS, eps),
    )
    for row in truth.faults:
        assert row.expected_loss_points["observed"] > 2
        assert row.qualified_by and row.episodes["observed"] == 19
        assert not row.detectable and row.small


def test_observed_effect_must_be_confirmed_in_observed_window() -> None:
    cause = replace(
        paired_table(900, 900, 500).units[0],
        qualified_by=("observed", "scoring"),
        confirmed_by=("scoring",),
    )
    table = detect.DetectTable([cause], [], [], {})
    assert cause.counts and len(table.counted()) == 1
    assert not cause.counts_observed and not table.observed_counted()
    assert not table.observed_counted(provisional=True)


def test_interaction_single_losses_are_an_acceptance_guard() -> None:
    a = make_fault("f01", "action_latency", "14 steps", "r01", 0)
    b = make_fault("f02", "policy_knob", "10 actions per chunk", "r01", 0)
    composer = Composer(
        task_recipe(tasks=(Task(curves.SPATIAL, 4),), interaction_single_loss_points=0.01)
    )
    problems, _ = composer.check(Draft(patterns=Patterns(interactions=[("f01", "f02")])), [a, b])
    assert "interaction singles cost more than the recipe allows" in problems


def test_private_paths_resolve_repositories_and_symlinks_before_writing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /somewhere")
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)
    for output in (repo, repo / "out", link / "out"):
        with pytest.raises(ValueError, match="outside the repository"):
            private_root(output)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "files").symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the repository"):
        private_root(outside)
    clean = tmp_path / "private"
    assert private_root(clean) == clean
    assert not clean.exists()


def test_development_cannot_enter_private_path_or_use_legacy_estimator(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"kind": "development", "name": "D2", "world_id": "w99"}))
    with pytest.raises(ValueError, match="test or reserve"):
        external_recipe(path)
    complete = stated_fields(task_recipe())
    path.write_text(json.dumps({**complete, "kind": "test", "name": "T1", "estimator": "suite-v1"}))
    with pytest.raises(ValueError, match=r"suite-v1 is restricted|must not leave empty"):
        external_recipe(path)


def test_private_report_and_failed_creation_never_print_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fleetlab.worlds import __main__ as cli
    from fleetlab.worlds import runner, world

    secret = "fx123abcabc123abcabc123abcabc123ab9"
    private = SimpleNamespace(recipe=SimpleNamespace(kind="test"))
    assert cli._report(private) == {"private": True}
    template = tmp_path / "template.json"
    template.write_text(
        task_recipe(kind="test", estimator="measured-v3").model_dump_json(exclude={"seed"})
    )
    monkeypatch.setattr(runner, "engine_software", lambda: SOFTWARE)

    def fail(*a: Any, **kw: Any) -> Any:
        raise ValueError(secret + " role kind scope level seed")

    monkeypatch.setattr(world, "generate", fail)
    assert cli.main(["generate", "--recipe", str(template), "--root", str(tmp_path / "out")]) == 2
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert "role kind scope level seed" not in output.out + output.err
    assert not (tmp_path / "out").exists()


def test_draw_handle_in_visible_text_is_a_rule_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.worlds import world

    r = task_recipe()
    f = make_fault("f01", "action_latency", "8 steps", "r01", 0)
    composition = identify(
        Composition(
            r, Composer(r).spec([f], Patterns()), 0, {"f01": "plain"}, frozenset(), False, {}
        )
    )
    original = world.maintenance

    def tainted(spec: Any) -> Any:
        rows = original(spec)
        assert rows
        return [rows[0].model_copy(update={"note": "check f01"}), *rows[1:]]

    monkeypatch.setattr(world, "maintenance", tainted)
    with pytest.raises(ValueError, match="private fault identity"):
        build(composition, SOFTWARE, TEXTS)


def test_task_record_counts_match_published_cells_and_keep_independent_confirmation() -> None:
    c = curves.load()
    data = json.loads(Path(curves.__file__).with_name("curves_supplement.json").read_text())[
        "task_calibration"
    ]
    singles = {
        (row["members"][0][0], row["suite"], row["members"][0][1])
        for row in data["points"]
        if len(row["members"]) == 1
    }
    excluded = {
        (row["members"][0][0], row["suite"], row["members"][0][1]) for row in data["excluded"]
    }
    assert singles | excluded == set(c.points)
    assert not singles & excluded
    for row in data["points"]:
        assert {p["task_index"] for p in row["tasks"]} == set(range(10))
        members = [tuple(m) for m in row["members"]]
        point: curves.CurvePoint | curves.JointPoint
        if len(members) == 1:
            point, _ = c.point(members[0][0], members[0][1], row["suite"])
        else:
            point, _ = c.joint(members[0], members[1], row["suite"])
        for field in ("n", "successes", "healthy_successes"):
            assert sum(p[field] for p in row["tasks"]) == getattr(point, field)
        assert all("/screen/" not in path for path in row["run_dirs"])
    joint, _ = c.for_task(curves.SPATIAL, 4).joint(
        ("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"), curves.SPATIAL
    )
    assert joint.n == 10 and joint.successes == 0 and joint.single_successes == (9, 10)
    assert "confirmation-100" in joint.run


def test_task_aggregation_refuses_mismatched_twins_and_conflicting_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.dose_response import Outcome
    from fleetlab.worlds.generate import policy_spec_nominal
    from fleetlab.worlds.supplement import identity

    i = identity(0, 0, curves.SPATIAL, policy_spec_nominal(), SOFTWARE)
    a = Outcome(i.episode_id, True, True, True, 1)
    b = replace(a, true_success=False, logged_success=False)
    monkeypatch.setattr(supplement, "_task_rows", lambda path: [(i, a if path == "a" else b)])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        supplement._task_collect(["a", "b"], curves.SPATIAL, ())
    assert supplement._task_signature(i) != supplement._task_signature(
        i.model_copy(update={"scene_seed": i.scene_seed + 1})
    )


def test_pair_placement_uses_task_cells_not_the_suite_average() -> None:
    c = curves.load()
    pair = (("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk"))
    tasks = dict(c.tasks)
    for index, rate in ((0, 0.65), (1, 0.2)):
        original = c.for_task(curves.SPATIAL, index)
        joint, _ = original.joint(*pair, curves.SPATIAL)
        tasks[curves.SPATIAL, index] = replace(
            original,
            joints=(
                replace(
                    joint,
                    successes=joint.healthy_successes * rate,
                    single_successes=(int(joint.healthy_successes), int(joint.healthy_successes)),
                    interaction=(-0.3, -0.4, -0.2),
                ),
            ),
        )
    composer = Composer(
        task_recipe(robot_success_floor=0.4, interaction_single_loss_points=10),
        replace(c, tasks=tasks),
    )
    composer.slots = tuple(
        base.Slot(f"r0{i + 1}", "s1", 1, 0, Task(curves.SPATIAL, i), 0) for i in range(2)
    )
    assert composer.pair_robot_choices(pair) == ["r01"]


@pytest.mark.parametrize("warnable", [False, True])
def test_composer_rejects_a_nonzero_but_subnoise_drift(
    monkeypatch: pytest.MonkeyPatch, warnable: bool
) -> None:
    from fleetlab.faults import records

    fault = make_fault(
        "f01",
        "camera_pose_offset",
        "3 deg scene pan",
        "r01",
        0,
        level=0,
        drift=Drift(shape="linear", rate=0.2),
    )
    composer = Composer(task_recipe(warnable_drift=warnable))
    monkeypatch.setattr(estimate, "drift_crossing_day", lambda *a, **kw: 8)
    monkeypatch.setattr(records, "health_shift", lambda f: {"calibration_residual": 4.0})
    messages, _ = composer.check(Draft(patterns=Patterns(drifts=["f01"])), [fault])
    assert "drift lacks a visible health precursor above daily noise" in messages


def test_private_cli_rejects_checkout_root_before_reading_recipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fleetlab.worlds import __main__ as cli
    from fleetlab.worlds import recipe as module

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /unneeded")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must validate output location before loading/drawing a recipe")

    monkeypatch.setattr(module, "external_recipe", forbidden)
    assert (
        cli.main(
            [
                "generate",
                "--recipe",
                str(tmp_path / "template.json"),
                "--root",
                str(checkout / "output"),
            ]
        )
        == 2
    )
    assert not (checkout / "output").exists()
