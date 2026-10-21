"""Principle tests for the second review; synthetic observations, no world lookups."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from test_classical_statistics import noisy, record, set_rates

from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.analysis import _onset_evidence, change_points, warning_crossing
from fleetlab.baselines.classical.data import Row
from fleetlab.baselines.classical.experiments import arm_estimate, choose, combine_estimates, revise
from fleetlab.baselines.classical.features import Design
from fleetlab.baselines.classical.health import _trajectory
from fleetlab.baselines.classical.protocol import Z90, experiment_window, tie_key
from fleetlab.baselines.classical.signals import event_signatures, outcome_check
from fleetlab.contracts.actions import Action, catalogue
from fleetlab.contracts.faults import Target
from fleetlab.contracts.model import ConfigVersion
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb


def test_blind_entry_points_do_not_import_executor() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import fleetlab.baselines.classical; "
            "import fleetlab.baselines.classical.experiments; "
            "assert not any(n.startswith(('fleetlab.engine', 'fleetlab.oracle', "
            "'fleetlab.scoring', 'fleetlab.baselines.classical.driver')) for n in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("drop", [0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
def test_fleet_candidates_have_no_site_significance_veto(drop: float) -> None:
    data = noisy(943)
    data.changes[:] = [
        dict(day=3, scope="fleet", key=ConfigKey.software_version, old="v0", new="v1", author="ops")
    ]
    rng = np.random.default_rng(844)
    for row in data.episodes:
        p = (0.85 if row["task_text"] == "easy" else 0.65) - drop * (row["day"] >= 3)
        row["outcome"] = "success" if rng.random() < p else "failure"
    allowed, explanation = _onset_evidence(data, data.changes, change_points(data))
    assert allowed and "pooled site/task" in explanation
    answer = analyse(data, bootstraps=20)
    estimate = record(answer, "fleet", str(ConfigKey.software_version))
    assert estimate["gain_points"] > 0
    assert estimate["low"] <= estimate["gain_points"] <= estimate["high"]
    assert any(a.target.scope == "fleet" for a in choose(answer))


def test_precision_pooling_and_paired_arm_noise() -> None:
    observed = dict(
        gain_points=6.0, low=2.0, high=10.0, estimation_variance=4.0, informative_interval=True
    )
    measured = dict(gain_points=0.0, low=-Z90 * 3, high=Z90 * 3, variance=9.0)
    pooled = combine_estimates(observed, measured)
    assert pooled["measurement_weight"] == pytest.approx(4 / 13)
    assert pooled["gain_points"] == pytest.approx(54 / 13)
    assert pooled["variance"] == pytest.approx(36 / 13)
    # Identical pairs retain nonzero uncertainty, but shared starts reduce noise.
    control = [
        dict(robot_id="r01", task_text="t", outcome="success" if i % 2 else "failure")
        for i in range(60)
    ]
    estimate = arm_estimate(control, [dict(r) for r in control], {"r01": 1.0})
    assert estimate["gain_points"] == 0
    assert 0 < estimate["high"] < 5 and -5 < estimate["low"] < 0
    assert 1 < estimate["variance"] < 5


def test_nonsignificant_experiment_preserves_cause_and_joint_update() -> None:
    data = noisy()
    set_rates(data, "r02", 2, 0.05)
    answer = analyse(data, bootstraps=20)
    cause = next(c for c in answer.diagnosis.causes if c.target.scope == "r02")
    action = next(r.action for r in answer.fix_set.actions if r.action.target == cause.target)
    null = dict(
        action=action.model_dump(mode="json"),
        gain_points=0.0,
        low=-5.0,
        high=5.0,
        variance=1000.0,
        confidence=0.5,
    )
    updated = revise(answer, [null], data.costs)
    assert any(c.target == cause.target for c in updated.diagnosis.causes)
    low_rank = next(
        r
        for r in answer.evidence["actions"]
        if r["action"]["target"]["scope"] == "r06"
        and r["action"]["target"]["component"] == "controller"
    )
    action = Action.model_validate(low_rank["action"])
    assert action not in [r.action for r in answer.fix_set.actions[:3]]
    updated = revise(
        answer,
        [
            dict(
                action=low_rank["action"],
                gain_points=3.0,
                low=2.8,
                high=3.2,
                variance=0.01,
                confidence=1.0,
            )
        ],
        data.costs,
    )
    assert action in [r.action for r in updated.fix_set.actions[:3]]
    assert updated.evidence["selection"]["predicted_prefix_gain_points"][2] > 3


def test_robust_health_line_ignores_interior_spike() -> None:
    values = np.array([20, 22, 24, 60, 28, 30]) + np.random.default_rng(39).normal(0, 0.25, 6)
    fit = _trajectory([dict(day=d, value=float(v)) for d, v in enumerate(values)], 1.0, ())
    assert fit.kind == "linear" and 1.7 < fit.slope < 2.3


def test_label_check_controls_robot_multiplicity() -> None:
    data = noisy()
    rows: list[Row] = []
    rng = np.random.default_rng(91)
    for robot in data.fleet.robots():
        short, horizon = (5, 2) if robot.id == "r01" else (2, 12)
        for i in range(short + horizon + 20):
            failed = i < short + horizon
            rows.append(
                dict(
                    episode_id=f"e{len(rows) + 1:06}",
                    robot_id=robot.id,
                    site_id=robot.site_id,
                    task_text="task",
                    day=i % 6,
                    outcome="failure" if failed else "success",
                    steps=280 if short <= i < short + horizon else int(rng.integers(90, 180)),
                )
            )
    assert not outcome_check(replace(data, episodes=rows))[1]


def test_symptoms_control_both_scope_and_event_multiplicity() -> None:
    data = noisy()
    rows: list[Row] = []
    events: list[Row] = []
    rng = np.random.default_rng(233)
    for robot in data.fleet.robots():
        for i in range(100):
            row = dict(
                episode_id=f"e{len(rows) + 1:06}",
                robot_id=robot.id,
                site_id=robot.site_id,
                day=i % 6,
                task_text="hard" if i % 2 else "easy",
                outcome="failure" if rng.random() < 0.3 else "success",
            )
            rows.append(row)
            for kind, p in (("drop", 0.16 if robot.id == "r01" else 0.03), ("stall", 0.02)):
                if rng.random() < p:
                    events.append(dict(episode_id=row["episode_id"], kind=kind, step=5, detail=""))
    result = event_signatures(replace(data, episodes=rows, events=events))
    drop = next(r for r in result if r.scope == "r01" and r.kind == "drop")
    assert 0.08 < drop.p < 0.10


def test_actual_conflicting_scope_precedence() -> None:
    data = noisy()
    sites = list(data.fleet.sites)
    robots = list(sites[0].robots)
    robots[1] = robots[1].model_copy(
        update={
            "config": ConfigVersion(
                version=2, since_day=2, values={ConfigKey.control_action_delay_steps: 20}
            )
        }
    )
    sites[0] = sites[0].model_copy(
        update={
            "robots": robots,
            "config": ConfigVersion(
                version=1, since_day=1, values={ConfigKey.control_action_delay_steps: 7}
            ),
        }
    )
    changes = [
        dict(
            day=1, scope="s1", key=ConfigKey.control_action_delay_steps, old=0, new=7, author="ops"
        ),
        *data.changes,
    ]
    changes[1] = {**changes[1], "old": 7}
    design = Design(
        replace(data, fleet=data.fleet.model_copy(update={"sites": sites}), changes=changes)
    )
    key = str(ConfigKey.control_action_delay_steps)
    assert design.configuration("r02", 1)[key] == 7
    assert design.configuration("r02", 2)[key] == 20
    assert design.configuration("r01", 2)[key] == 7


def test_camera_repair_matches_geometry_or_image_reading() -> None:
    data = noisy()
    data = replace(data, health=[{**r, "metric": "image_noise"} for r in data.health])
    for row in data.health:
        if row["robot_id"] == "r05":
            row["value"] += 80
    set_rates(data, "r05", 0, 0.1)
    design = Design(data)
    actions = [
        a
        for a in design.candidates()
        if a.target == Target(scope="r05", component=Component.scene_camera)
    ]
    assert {a.verb for a in actions} == {Verb.clean, Verb.recalibrate}
    row = next(r for r in data.episodes if r["robot_id"] == "r05")
    column = next(
        i
        for i, name in enumerate(design.names)
        if name.startswith("health:scene_camera.image_noise:1")
    )
    values = {a.verb: design.matrix([row], future=True, action=a)[0, column] for a in actions}
    assert values[Verb.clean] == 0 < values[Verb.recalibrate]


def test_warning_requires_future_crossing_due_to_the_selected_trend() -> None:
    weight, covered = np.array([0.5, 0.5]), np.array([1.0, 0.0])
    daily = {5: np.array([0.78, 0.8]), 6: np.array([0.76, 0.8]), 7: np.array([0.68, 0.8])}
    frozen = {d: np.array([0.78, 0.8]) for d in daily}
    assert warning_crossing(daily, frozen, weight, covered, 5) == 7
    assert warning_crossing(daily, daily, weight, covered, 5) is None
    assert warning_crossing({**daily, 5: np.array([0.65, 0.8])}, frozen, weight, covered, 5) is None


def test_public_window_and_neutral_condition_ties() -> None:
    assert experiment_window(5) == (6, 7)
    assert experiment_window(12) == (13, 14)
    actions = [
        Action(verb=Verb.change_condition, target=Target(scope="s1", condition_key=k), value=0)
        for k in ConditionKey
    ]
    keyed = [tie_key(a) for a in actions]
    assert len(set(keyed)) == len(actions)
    assert sorted(actions, key=tie_key) != sorted(
        actions, key=lambda a: str(a.target.condition_key)
    )
    assert keyed == [tie_key(a) for a in actions]


def test_shared_drop_prior_changes_allocation_and_respects_catalogue() -> None:
    data = noisy()
    sites = [
        s.model_copy(update={"conditions": {k: 0 for k in ConditionKey}}) for s in data.fleet.sites
    ]
    data = replace(data, fleet=data.fleet.model_copy(update={"sites": sites}))
    data = replace(data, actions=catalogue(data.fleet))
    for robot in sites[0].robots:
        set_rates(data, robot.id, 0, 0.25)
    rng = np.random.default_rng(414)
    data.events[:] = [
        dict(episode_id=r["episode_id"], step=50, kind="drop", detail="")
        for r in data.episodes
        if rng.random() < (0.4 if r["site_id"] == "s1" else 0.015)
    ]
    result = analyse(data, bootstraps=20)
    object_gain = record(result, "s1", "object_batch")["gain_points"]
    other_gain = record(result, "s1", "clutter")["gain_points"]
    assert object_gain > 5 * other_gain > 0
    narrowed = replace(
        data,
        events=[],
        actions=[
            a
            for a in data.actions
            if a.target.condition_key not in {ConditionKey.object_batch, ConditionKey.lighting}
        ],
    )
    two = analyse(narrowed, bootstraps=20)
    assert record(two, "s1", "clutter")["gain_points"] > 4 * other_gain
    only_object = replace(
        data,
        actions=[
            a
            for a in data.actions
            if not a.target.condition_key or a.target.condition_key == ConditionKey.object_batch
        ],
    )
    one = analyse(only_object, bootstraps=20)
    assert record(one, "s1", "object_batch")["gain_points"] > object_gain
    no_object = replace(
        data,
        actions=[a for a in data.actions if a.target.condition_key != ConditionKey.object_batch],
    )
    absent = analyse(no_object, bootstraps=20)
    assert record(absent, "s1", "clutter")["gain_points"] > 3 * other_gain
    for rec in result.evidence["actions"]:
        population = 100 / 6 if rec["action"]["target"]["scope"].startswith("r") else 50
        assert (
            -population - 1e-12
            <= rec["low"]
            <= rec["gain_points"]
            <= rec["high"]
            <= population + 1e-12
        )
        if rec["action"]["target"]["scope"].startswith("r"):
            # Noise may come only from 1/6 of a two-day scoring panel (480 rows).
            assert rec["prediction_radius"] <= Z90 * math.sqrt((1 / 6) / 480) * 100 + 1e-8


def test_episode_resampling_retains_cell_sizes_and_event_identity() -> None:
    from collections import Counter

    from fleetlab.baselines.classical.resampling import resample

    data = noisy()
    sample = resample(data, np.random.default_rng(20261009))
    assert Counter((r["robot_id"], r["day"]) for r in sample.episodes) == Counter(
        (r["robot_id"], r["day"]) for r in data.episodes
    )
    assert len({r["episode_id"] for r in sample.episodes}) < len(sample.episodes)
    assert sample.health is data.health and sample.events is data.events


@pytest.mark.parametrize("link", ["logistic", "multiplicative"])
def test_compact_joint_search_equals_signed_feature_unions(link: str) -> None:
    from itertools import combinations

    from fleetlab.baselines.classical.selection import JointPrediction

    rng = np.random.default_rng(154)
    x = rng.normal(0, 0.3, (9, 7))
    x[:, 4:] = np.abs(x[:, 4:])
    beta = np.r_[rng.normal(size=4), -np.ones(3)]
    deltas = rng.choice([-0.8, -0.4, 0, 0.2, 0.6], size=(8, 9, 7))
    weight = rng.dirichlet(np.ones(9))
    joint = JointPrediction(x, beta, 4, link, [x + d for d in deltas], weight)
    for size in (1, 2, 3, 6):
        for indices in combinations(range(8), size):
            delta = deltas[list(indices)]
            union = np.minimum(delta, 0).min(axis=0) + np.maximum(delta, 0).max(axis=0)
            expected = float((joint._probability(x + union) - joint.base) @ weight * 100)
            assert joint.gain(indices) == pytest.approx(expected, abs=1e-10)


def test_null_camera_verb_does_not_erase_other_repair_evidence() -> None:
    data = noisy()
    for row in data.health:
        if row["robot_id"] == "r05" and row["component"] == "scene_camera":
            row["value"] += 70
    set_rates(data, "r05", 0, 0.05)
    answer = analyse(data, bootstraps=20)
    target = Target(scope="r05", component=Component.scene_camera)
    assert any(c.target == target for c in answer.diagnosis.causes)
    actions = [
        Action.model_validate(r["action"])
        for r in answer.evidence["actions"]
        if r["action"]["target"] == target.model_dump(mode="json")
    ]
    estimates = [
        dict(
            action=a.model_dump(mode="json"),
            gain_points=4.0 if a.verb == Verb.recalibrate else 0.0,
            low=4.0 if a.verb == Verb.recalibrate else 0.0,
            high=4.0 if a.verb == Verb.recalibrate else 0.0,
            confidence=1.0,
        )
        for a in actions
    ]
    for ordered in (estimates, estimates[::-1]):
        updated = revise(answer, ordered, data.costs, condition="oracle-proxy")
        assert any(c.target == target for c in updated.diagnosis.causes)


def test_compact_joint_empty_zero_nonfinite_and_numerical_ties() -> None:
    from fleetlab.baselines.classical.selection import JointPrediction

    x, beta, weight = np.zeros((2, 2)), np.ones(2), np.array([0.5, 0.5])
    empty = JointPrediction(x, beta, 2, "logistic", [], weight)
    assert empty.order([], [])[0] == [] and empty.gain(()) == 0
    no_rows = JointPrediction(
        np.zeros((0, 2)), beta, 2, "logistic", [np.zeros((0, 2))], np.zeros(0)
    )
    assert no_rows.gain((0,)) == 0
    zeros = JointPrediction(x, beta, 2, "logistic", [x.copy(), x.copy()], weight)
    assert zeros.gain((0, 1)) == 0 and zeros.order([2.0, 1.0], ["a", "b"])[0][0] == 1
    with pytest.raises(ValueError, match="finite"):
        JointPrediction(x, beta, 2, "logistic", [x + np.nan], weight)
    repairs = [x.copy(), x.copy()]
    repairs[0][0, 0] = 0.7
    repairs[1][1, 0] = 0.7 + 1e-13
    equal = JointPrediction(x, beta, 2, "logistic", repairs, weight)
    assert equal.order([1.0, 2.0], ["z", "a"])[0][0] == 0
