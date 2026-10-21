"""Noisy regression, onset, symptoms and experiment probes, independent of worlds."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_classical_baseline import DELAY, synthetic, write_export

from fleetlab.baselines.classical import Answer, analyse
from fleetlab.baselines.classical.analysis import _bh, change_points
from fleetlab.baselines.classical.data import VisibleData, load_export
from fleetlab.baselines.classical.experiments import choose, paired_estimate, plan, update
from fleetlab.baselines.classical.features import Design
from fleetlab.baselines.classical.health import health_series
from fleetlab.baselines.classical.regression import bootstrap_weights
from fleetlab.baselines.classical.selection import JointPrediction
from fleetlab.baselines.classical.signals import (
    event_signatures,
    outcome_check,
    stratified_contrast,
)
from fleetlab.contracts.faults import Target
from fleetlab.contracts.tools import ArmResult, ExperimentResult
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.toolserver.public_results import COMPLETED


def noisy(seed: int = 73) -> VisibleData:
    data = synthetic(harmful=False)
    rng = np.random.default_rng(seed)
    for row in data.episodes:
        success = rng.random() < (0.90 if row["task_text"] == "easy" else 0.60)
        row["outcome"] = "success" if success else "failure"
        row["steps"] = int(rng.integers(90, 230)) if success else 280
    for row in data.health:
        row["value"] = 20 + float(rng.normal(0, 2))
    return data


def set_rates(data: VisibleData, robot: str, onset: int, rate: float, seed: int = 81) -> None:
    rng = np.random.default_rng(seed)
    for row in data.episodes:
        if row["robot_id"] == robot and row["day"] >= onset:
            row["outcome"] = "success" if rng.random() < rate else "failure"
            row["steps"] = int(rng.integers(90, 230)) if row["outcome"] == "success" else 280


def record(answer: Answer, scope: str, thing: str) -> dict[str, Any]:
    return next(
        r
        for r in answer.evidence["actions"]
        if r["action"]["target"]["scope"] == scope and thing in r["action"]["target"].values()
    )


def test_late_health_step_uses_onset_and_component_repair() -> None:
    data = noisy()
    for row in data.health:
        if row["robot_id"] == "r05" and row["component"] == "scene_camera" and row["day"] >= 4:
            row["value"] += 70
    set_rates(data, "r05", 4, 0.06)
    design = Design(data)
    series = next(h for h in design.health if h.component == "scene_camera")
    assert series.robots["r05"].kind == "step"
    assert series.robots["r05"].onset == 4
    assert series.exposure("r05", 3) < 0.2 < series.exposure("r05", 4)
    action = next(
        a
        for a in design.candidates()
        if a.target == Target(scope="r05", component=Component.scene_camera)
        and a.verb == Verb.recalibrate
    )
    row = next(r for r in data.episodes if r["robot_id"] == "r05")
    column = next(i for i, n in enumerate(design.names) if n.startswith("health:scene_camera"))
    assert design.matrix([row], future=True, action=action)[0, column] == 0
    answer = analyse(data, bootstraps=40)
    assert answer.fix_set.actions[0].action == action
    assert answer.fix_set.actions[0].expected_gain > 9
    assert action.target in [c.target for c in answer.diagnosis.causes]


def test_noisy_drift_pools_noise_projects_and_warns() -> None:
    data = noisy()
    rng = np.random.default_rng(6)
    for row in data.health:
        if row["robot_id"] == "r05" and row["component"] == "scene_camera":
            row["value"] += 9 * row["day"] + float(rng.normal(0, 6))
    set_rates(data, "r05", 2, 0.1)
    series = next(h for h in health_series(data) if h.component == "scene_camera")
    assert series.noise < 4  # noisy faulted robot cannot determine everyone's noise
    assert series.robots["r05"].slope > 5
    assert series.exposure("r05", 8.5) > series.exposure("r05", 5) + 0.2
    assert series.exposure("r05", 8.5, flat=True) == series.exposure("r05", 5)
    design = Design(data)
    row = next(r for r in data.episodes if r["robot_id"] == "r05")
    column = next(
        i for i, name in enumerate(design.names) if name.startswith("health:scene_camera")
    )
    assert (
        design.matrix([row], future=True)[0, column]
        > design.matrix([row], forecast_day=5)[0, column]
    )
    answer = analyse(data, bootstraps=40)
    assert not any(w.scope == "r05" for w in answer.diagnosis.warnings)
    assert any(r["scope"] == "r05" for r in answer.evidence["warning_forecasts"])


def test_metric_directions_usage_and_unknown_tails() -> None:
    data = noisy()
    health = [
        {
            **r,
            "component": "gripper",
            "metric": "grip_force",
            "value": 60 - (45 if r["robot_id"] == "r04" else 0) + r["value"] / 10,
        }
        for r in data.health
        if r["component"] == "scene_camera"
    ]
    health += [{**r, "metric": "cycle_count", "value": r["day"] * 1000} for r in health]
    data = replace(data, health=health)
    series = health_series(data)
    assert len(series) == 1
    assert series[0].exposure("r04", 5) > 2
    assert series[0].exposure("r01", 5) < 0.2
    unknown = health_series(
        replace(data, health=[{**r, "metric": "novel_sensor"} for r in health[:36]])
    )
    assert {h.direction for h in unknown} == {-1, 1}
    set_rates(data, "r04", 0, 0.03)
    answer = analyse(data, bootstraps=40)
    assert answer.fix_set.actions[0].action.target == Target(
        scope="r04", component=Component.gripper
    )
    assert answer.fix_set.actions[0].expected_gain > 8


def test_all_exposure_cells_survive_hierarchical_resampling() -> None:
    clusters = [f"r{i // 6:02}:{i % 6}" for i in range(72) for _ in range(7)]
    sites = [f"s{i // 24}" for i in range(72) for _ in range(7)]
    rng = np.random.default_rng(42)
    weights = np.asarray([bootstrap_weights(clusters, rng, sites) for _ in range(2000)])
    assert np.all(weights > 0)
    assert np.std(weights[:, -14:].sum(axis=1)) > 5
    # All episodes of a robot-day have the same weight. Different days of the
    # same robot co-vary more than days of robots in different sites.
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    assert (
        np.corrcoef(weights[:, 0], weights[:, 7])[0, 1]
        > np.corrcoef(weights[:, 0], weights[:, -1])[0, 1]
    )
    same_robot = np.corrcoef(weights[:, 0], weights[:, 7])[0, 1]
    same_site = np.corrcoef(weights[:, 0], weights[:, 42])[0, 1]
    other_site = np.corrcoef(weights[:, 0], weights[:, 336])[0, 1]
    assert same_robot > same_site + 0.1
    assert same_site > other_site + 0.04


def test_two_breaks_retain_earlier_harmful_edit_and_validation_selects_best() -> None:
    data = noisy()
    data.changes[0]["day"] = 1
    set_rates(data, "r02", 1, 0.35)
    set_rates(data, "r02", 4, 0.01)
    answer = analyse(data, bootstraps=60)
    assert Target(scope="r02", config_key=DELAY) in [c.target for c in answer.diagnosis.causes]
    losses = answer.evidence["validation_log_loss"]
    key = f"{answer.evidence['selected_link']}:{answer.evidence['penalty_multiplier']}"
    assert losses[key] == min(losses.values())
    candidate = record(answer, "r02", str(DELAY))
    assert candidate["low"] < candidate["gain_points"] < candidate["high"]
    assert candidate["diagnosis_confidence"] >= 0.9


def test_fleet_release_is_judged_by_visible_cross_site_changes() -> None:
    data = noisy()
    data.changes[:] = [
        dict(
            day=3, scope="fleet", key=ConfigKey.software_version, old="old", new="new", author="ops"
        )
    ]
    for robot in data.fleet.robots():
        set_rates(data, robot.id, 3, 0.12)
    answer = analyse(data, bootstraps=40)
    candidate = record(answer, "fleet", str(ConfigKey.software_version))
    assert candidate["gain_points"] > 20
    assert Target(scope="fleet", config_key=ConfigKey.software_version) in [
        c.target for c in answer.diagnosis.causes
    ]
    # Same apparent release but only one site declines: time and local changes
    # cannot uniquely support a fleet-wide attribution.
    local = noisy()
    local.changes[:] = data.changes
    for robot in local.fleet.sites[0].robots:
        set_rates(local, robot.id, 3, 0.12)
    result = analyse(local, bootstraps=40)
    assert record(result, "fleet", str(ConfigKey.software_version))["gain_points"] > 0
    assert any(a.target.scope == "fleet" for a in choose(result))


def test_unlogged_deficit_nominates_repair_and_shared_drops_name_condition() -> None:
    data = noisy()
    set_rates(data, "r04", 0, 0.10)
    answer = analyse(data, bootstraps=40)
    assert any(
        r.action.target.scope == "r04" and r.expected_gain > 0 for r in answer.fix_set.actions
    )
    assert any(
        r["latent_candidate"]
        for r in answer.evidence["actions"]
        if r["action"]["target"]["scope"] == "r04"
    )
    sites = [
        s.model_copy(update={"conditions": {k: 0 for k in ConditionKey}}) for s in data.fleet.sites
    ]
    data = replace(data, fleet=data.fleet.model_copy(update={"sites": sites}))
    from fleetlab.contracts.actions import catalogue

    data = replace(data, actions=catalogue(data.fleet))
    rng = np.random.default_rng(414)
    for robot in sites[0].robots:
        set_rates(data, robot.id, 0, 0.25)
    data.events[:] = [
        dict(episode_id=r["episode_id"], step=50, kind="drop", detail="object fell")
        for r in data.episodes
        if rng.random() < (0.40 if r["site_id"] == "s1" else 0.015)
    ]
    signals = event_signatures(data)
    assert any(
        s.scope == "s1" and s.kind == "drop" and s.p < 0.01 and s.robots == 3 for s in signals
    )
    result = analyse(data, bootstraps=40)
    cause = next(
        c
        for c in result.diagnosis.causes
        if c.target == Target(scope="s1", condition_key=ConditionKey.object_batch)
    )
    assert str(cause.direction) == "changed"
    assert any(a.target.condition_key for a in choose(result))


def test_short_failure_check_and_handover_export(tmp_path: Path) -> None:
    data = noisy()
    # Flip formerly successful labels, preserving their visibly short durations.
    for row in data.episodes:
        if row["robot_id"] == "r05" and row["day"] >= 2 and row["outcome"] == "success":
            row["outcome"] = "failure"
    y, checks = outcome_check(data)
    assert [r["scope"] for r in checks] == ["r05"]
    indices = [
        i for i, r in enumerate(data.episodes) if r["episode_id"] in checks[0]["episode_ids"]
    ]
    assert min(y[indices]) > 0.90
    assert not outcome_check(noisy())[1]
    data.episodes[0]["outcome"] = "handed_over"
    write_export(tmp_path / "export", data)
    loaded = load_export(tmp_path / "export")
    result = analyse(loaded, bootstraps=20)
    assert result.evidence["handed_over"] == 1
    assert record(result, "r05", "controller")["gain_points"] < 1
    assert result.evidence["fitted_success_total"] > result.evidence["successes"]
    assert any(c.target == Target(scope="r05") for c in result.diagnosis.causes)


def test_failure_endpoint_ignores_peers_without_failures() -> None:
    data = noisy(814)
    rng = np.random.default_rng(91)
    for row in data.episodes:
        if row["robot_id"] in {"r01", "r02", "r03"}:
            row["outcome"] = "success"
            row["steps"] = int(rng.integers(90, 230))
        elif row["robot_id"] == "r05" and row["outcome"] == "success":
            row["outcome"] = "failure"
    # Only two peers have failure endpoints. Successful robots neither vote
    # for a horizon nor vote against the information those peers provide.
    _, checks = outcome_check(data)
    assert [r["scope"] for r in checks] == ["r05"]
    for row in data.episodes:
        if row["robot_id"] == "r06":
            row["outcome"] = "success"
    # A single peer cannot establish a shared endpoint.
    assert not outcome_check(data)[1]


def test_scan_controls_tasks_multiplicity_and_naming_screen() -> None:
    data = noisy()
    # Deliberately change task mix after day three, without changing within-task rates.
    rows = [r for r in data.episodes if (r["task_text"] == "hard") == (r["day"] >= 3)]
    scan = change_points(replace(data, episodes=rows))
    assert not any(r["detected"] for r in scan)
    # Direct noisy score has the analytical family correction, including all cuts.
    scan = change_points(data)
    own = next(r for r in scan if r["scope"] == "r01")
    expected = min(1, math.erfc(abs(own["score_z"]) / math.sqrt(2)) * len(scan) * 5)
    assert own["familywise_p"] == pytest.approx(expected)
    assert _bh([0.001, 0.03, 0.12, 0.8]) == {0, 1}
    assert _bh([0.07] + [0.9] * 20) == set()
    task_rows = [dict(task_text=t) for t in ("easy", "hard") for _ in range(100)]
    group = [i % 10 < (8 if i < 100 else 2) for i in range(200)]
    rng = np.random.default_rng(2)
    values = [float(rng.random() < (0.95 if i < 100 else 0.15)) for i in range(200)]
    z, contrast = stratified_contrast(task_rows, group, values)
    assert abs(z) < 2 and abs(contrast) < 0.15


def test_joint_optimum_matches_exhaustive_search_and_cost_ties() -> None:
    rng = np.random.default_rng(2)
    # Three jointly valuable repairs on a deeply impaired scope versus a larger
    # singleton elsewhere. A singleton sort chooses the wrong triple.
    x = np.array([[-7.0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0]])
    x[:, 0] += rng.normal(0, 0.02, 2)
    repairs = []
    for i in range(4):
        r = x.copy()
        r[0 if i < 3 else 1, i + 1] = 3 if i < 3 else 1
        repairs.append(r)
    joint = JointPrediction(x, np.ones(5), 5, "logistic", repairs, np.array([0.75, 0.25]))
    order, selection = joint.order([1.0] * 4, list("abcd"), 4)
    optimum = max(combinations(range(4), 3), key=joint.gain)
    assert set(order[:3]) == set(optimum) == {0, 1, 2}
    assert joint.gain((3,)) > joint.gain((0,))
    assert selection["predicted_prefix_gain_points"][2] == pytest.approx(joint.gain(optimum))
    tied = JointPrediction(
        x, np.ones(5), 5, "logistic", [repairs[0], repairs[0]], np.array([0.75, 0.25])
    )
    assert tied.order([2.0, 1.0], ["a", "b"], 2)[0][0] == 1


def test_actual_queue_contract_and_paired_update(tmp_path: Path) -> None:
    data = noisy()
    data.changes[0]["day"] = 4
    set_rates(data, "r02", 4, 0.02)
    export = tmp_path / "export"
    write_export(export, data)
    answer = analyse(data, bootstraps=20)
    root = tmp_path / "ab"
    queue = plan(export, answer, root, requests=1, episodes_per_arm=12)
    assert queue.is_file()
    from fleetlab.agents.sessions import Analysis, ExperimentQueue

    queued = ExperimentQueue.model_validate_json(queue.read_text())
    assert queued.experiments[0].request.paired_starts
    assert len(queued.experiments[0].request.arms) == 2
    assert queued.experiments[0].request.robots == ["r02"]
    experiment_id = queued.experiments[0].experiment_id
    arms = []
    rows = list(data.episodes)
    for arm in range(2):
        ids = []
        for i in range(12):
            eid = f"e{900000 + 12 * arm + i:06}"
            ids.append(eid)
            rows.append(
                dict(
                    episode_id=eid,
                    robot_id="r02",
                    site_id="s1",
                    day=6,
                    task_text="easy",
                    outcome="success" if arm == 1 or i < 2 else "failure",
                    steps=280,
                    experiment_id=experiment_id,
                )
            )
        arms.append(ArmResult(arm_index=arm, episode_ids=ids))
    pq.write_table(pa.Table.from_pylist(rows), export / "log/episodes.parquet")
    Analysis(root).record_results(
        [
            ExperimentResult(
                experiment_id=experiment_id,
                status="done",
                arms=arms,
                message=COMPLETED,
            )
        ]
    )
    updated = update(export, answer, root)
    assert updated.evidence["completed_requests"] == 1
    assert updated.evidence["experiments"][0]["pairs"] == 12
    combined = updated.evidence["experiments"][0]["combined"]
    assert 0 < combined["measurement_weight"] < 1
    assert updated.fix_set.actions[0].expected_gain == pytest.approx(combined["gain_points"])
    # No hidden oracle used by the updater; this is synthetic, not live queue evidence.
    assert updated.evidence["observed_episodes"] == len(data.episodes)


def test_paired_uncertainty_and_bad_pair_refusal() -> None:
    controls = [dict(robot_id="r01", task_text="t", outcome="failure") for _ in range(60)]
    treatments = [
        {**r, "outcome": "success" if i < 30 else "failure"} for i, r in enumerate(controls)
    ]
    estimate = paired_estimate(controls, treatments, {"r01": 0.1})
    assert estimate["gain_points"] == pytest.approx(5)
    assert 3 < estimate["low"] < 5 < estimate["high"] < 7
    treatments[0]["task_text"] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        paired_estimate(controls, treatments, {"r01": 0.1})


def test_nuisance_design_and_task_mix_noisy_negative_control() -> None:
    data = noisy(581)
    design = Design(data)
    for prefix, field, values in (
        ("robot", "robot_id", design.robots),
        ("site", "site_id", design.sites),
        ("task", "task_text", design.tasks),
    ):
        for value in values:
            column = design.names.index(f"{prefix}:{value}")
            expected = np.asarray([r[field] == value for r in data.episodes], dtype=float)
            np.testing.assert_array_equal(design.x[:, column], expected)
    for day in range(1, 6):
        np.testing.assert_array_equal(
            design.x[:, design.names.index(f"day:{day}")], [r["day"] == day for r in data.episodes]
        )
    # At the edited robot, task difficulty changes with the edit. Control
    # robots supply the common-task contrasts; omitting task effects blames it.
    rows = [
        r
        for r in data.episodes
        if r["robot_id"] != "r02" or (r["task_text"] == "hard") == (r["day"] >= 2)
    ]
    answer = analyse(replace(data, episodes=rows), bootstraps=40)
    assert abs(record(answer, "r02", str(DELAY))["gain_points"]) < 1.2


def test_scope_repairs_union_and_flat_noise_selection() -> None:
    data = noisy()
    design = Design(data)
    # A maintained component need not improve; preserve the intervention date
    # as an inspected boundary and let measured readings select the history.
    maintenance = [dict(day=2, robot_id="r01", component="scene_camera", verb="service")]
    series = next(
        h
        for h in health_series(replace(data, maintenance=maintenance))
        if h.component == "scene_camera"
    )
    assert series.robots["r01"].maintenance_days == (2,)
    assert series.robots["r01"].kind == "level"
    assert all(
        a.target.config_key is None
        or any(
            c["scope"] == a.target.scope and c["key"] == a.target.config_key for c in data.changes
        )
        for a in design.candidates()
    )
    actions = [a for a in design.candidates() if a.target.component == Component.scene_camera][:2]
    combined = design.matrix(data.episodes[:1], future=True, actions=tuple(actions))
    separately = design.matrix(data.episodes[:1], future=True, action=actions[0])
    np.testing.assert_array_equal(combined, separately)  # second robot cannot alter this one


@pytest.mark.parametrize("membership", ["absent", "complete", "incomplete"])
def test_evaluator_generic_root_and_informative_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, membership: str
) -> None:
    import json
    from types import SimpleNamespace

    from test_scoring import inputs, measured_fixture

    import fleetlab.oracle.table
    import fleetlab.oracle.world
    from fleetlab.baselines.classical.evaluate import evaluate

    values = inputs(confounders=[("f02", "f05")])
    assert values.fix_set is not None and values.diagnosis is not None
    world = SimpleNamespace(
        world_id="w01",
        world_hash="synthetic",
        fault_table=values.fault_table,
        scoring=lambda: values.oracle.identities,
        excluded=frozenset(),
        truth=lambda: values.truth,
        current=lambda: values.current,
        catalogue=lambda: values.catalogue,
        spec=SimpleNamespace(
            seed=values.seed, patterns=SimpleNamespace(confounders=values.confounders)
        ),
    )
    monkeypatch.setattr(fleetlab.oracle.world, "load_world", lambda _: world)
    monkeypatch.setattr(fleetlab.oracle.world, "all_runs", lambda *_: [])
    monkeypatch.setattr(
        fleetlab.oracle.table,
        "build_table",
        lambda *_: SimpleNamespace(table=values.oracle, problems=lambda: []),
    )
    root = tmp_path / "arbitrarily-named-world"
    outbox = tmp_path / "answer/outbox"
    outbox.mkdir(parents=True)
    (outbox / "fix_set.json").write_text(values.fix_set.model_dump_json())
    (outbox / "diagnosis.json").write_text(values.diagnosis.model_dump_json())
    evidence = dict(
        world_id="w01",
        actions=[
            dict(action=r.action.model_dump(mode="json"), informative_interval=i < 2)
            for i, r in enumerate(values.fix_set.actions)
        ],
    )
    (outbox.parent / "evidence.json").write_text(json.dumps(evidence))
    if membership != "absent":
        measured = measured_fixture(values.fault_table).model_copy(
            update={"world_hash": "synthetic", "complete": membership == "complete"}
        )
        # Give measured membership a different denominator from curve membership.
        removed = str(measured.causes[0]["unit"])
        measured.causes = measured.causes[1:]
        measured.cause_count = len(measured.causes)
        for fault in measured.faults:
            if fault.item == removed:
                fault.measured_class = "small"
        (root / "oracle").mkdir(parents=True)
        (root / "oracle/measured_causes.json").write_text(measured.model_dump_json())
    result = evaluate(root, outbox.parent, tmp_path / "score")
    report = json.loads((tmp_path / "score/score.json").read_text())
    assert result["claim1_basis"] == (
        "descriptive curves" if membership == "absent" else "measured causes"
    )
    selected = report["claim1_curve"] if membership == "absent" else report["claim1"]
    assert result["recall"] == selected["recall"]["value"]
    if membership == "complete":
        assert report["claim1"]["detectable"] != report["claim1_curve"]["detectable"]
    if membership != "complete":
        assert report["claim1"]["recall"]["value"] is None
    if membership == "incomplete":
        assert result["claim1_missing"]
        assert result["confounders_total"] == 0
        return
    assert result["world"] == "w01"
    assert result["informative_actions"] == 2
    assert result["informative_intervals_covered"] <= 2
    assert result["mae_points"] is not None
    assert result["calibration_slope"] is not None
    assert len(result["budgets"]) == 4
    assert result["confounders_total"] > 0


def test_one_late_health_spike_cannot_establish_a_hinge_drift() -> None:
    data = noisy()
    for row in data.health:
        if row["robot_id"] == "r03" and row["component"] == "scene_camera" and row["day"] == 5:
            row["value"] += 55
    series = next(h for h in health_series(data) if h.component == "scene_camera")
    trajectory = series.robots["r03"]
    assert trajectory.kind != "hinge" or trajectory.onset <= 3
    assert trajectory.slope < 20


def test_experiment_alarm_survives_a_correlated_edits_small_residual() -> None:
    from copy import deepcopy

    answer = analyse(noisy(), bootstraps=20)
    evidence = deepcopy(answer.evidence)
    # Visible alarm information must outrank a residual-only story: a fit can
    # absorb a site's decline into an observationally inseparable edit.
    from fleetlab.contracts.actions import Action
    from fleetlab.contracts.vocabulary import Verb

    for scope, gain in (("s1", 0.1), ("s2", 2.0)):
        for key in ConditionKey:
            action = Action(
                verb=Verb.change_condition, target=Target(scope=scope, condition_key=key), value=0
            )
            evidence["actions"].append(
                dict(action=action.model_dump(mode="json"), gain_points=gain, latent_candidate=True)
            )
    evidence["change_points"] = [
        dict(scope="s1", score_z=-8, detected=True, direction="decline"),
        dict(scope="s2", score_z=-1, detected=False, direction="decline"),
    ]
    chosen = choose(replace(answer, evidence=evidence))
    assert {a.target.condition_key for a in chosen if a.target.scope == "s1"} >= set(ConditionKey)
    assert not any(a.target.scope == "s2" and a.target.condition_key for a in chosen)
