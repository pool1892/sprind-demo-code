"""Principle repairs verified without development-world data or oracle lookups."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from classical_planted import planted
from test_classical_statistics import noisy, set_rates

from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.analysis import _onset_evidence, change_points, named_actions
from fleetlab.baselines.classical.data import VisibleData
from fleetlab.baselines.classical.experiments import arm_estimate, choose, revise
from fleetlab.baselines.classical.onset import release_contrast
from fleetlab.baselines.classical.protocol import population_bound
from fleetlab.contracts.vocabulary import ConfigKey


def competing_release(drop: float, seed: int) -> VisibleData:
    """Four logged early robot faults, three unlogged earlier-than-release faults.

    The null release is last-day; genuine fleet faults start on day three.
    Rates are planted additively, with independent Bernoulli noise, not fitted.
    """
    onset = 5 if drop == 0 else 3
    data = planted(drop, onset, seed + 7300)
    known = {"r02", "r03", "r06", "r10"}
    unknown = {"r01", "r05", "r09"}
    data.changes.extend(
        dict(day=1, scope=r, key=ConfigKey.control_action_delay_steps, old=0, new=20, author="ops")
        for r in sorted(known)
    )
    data.changes.sort(key=lambda r: r["day"])
    rng = np.random.default_rng(seed + 8100)
    for row in data.episodes:
        rate = 0.85 if row["task_text"] == "easy" else 0.65
        rate -= 0.40 * (row["robot_id"] in known and row["day"] >= 1)
        rate -= 0.40 * (row["robot_id"] in unknown and row["day"] >= (3 if drop == 0 else 1))
        rate -= drop * (row["day"] >= onset)
        success = rng.random() < max(0, rate)
        row["outcome"] = "success" if success else "failure"
        row["steps"] = int(rng.integers(90, 230)) if success else 280
    return data


@pytest.mark.parametrize("drop", [0.0, 0.05, 0.10, 0.20, 0.40])
def test_release_rank_against_earlier_logged_and_unlogged_faults(drop: float) -> None:
    ranks = []
    for seed in range(12):
        answer = analyse(competing_release(drop, seed), bootstraps=20)
        rank = next(
            (
                i
                for i, r in enumerate(answer.fix_set.actions, 1)
                if r.action.target.scope == "fleet"
            ),
            99,
        )
        ranks.append(rank)
        assert any(a.target.scope == "fleet" for a in choose(answer))
    top_three = sum(r <= 3 for r in ranks)
    print({"drop": drop, "top_three": top_three, "ranks": ranks})
    assert top_three < 6 if drop == 0 else top_three > 6, (drop, ranks)


def test_release_estimate_is_invariant_to_remote_declines() -> None:
    data = competing_release(0, 7)
    reference = [
        dict(robot_id=r.id, task_text=t) for r in data.fleet.robots() for t in ("easy", "hard")
    ]
    weight = np.full(len(reference), 1 / len(reference))
    before, metadata = release_contrast(
        data, 5, reference, weight, np.ones(len(reference)), draws=100, seed=0
    )
    for row in data.episodes:
        if row["day"] < 3:
            row["outcome"] = "success"
    after, _ = release_contrast(
        data, 5, reference, weight, np.ones(len(reference)), draws=100, seed=0
    )
    assert np.array_equal(before, after)
    assert metadata["pre_days"] == [3, 4] and metadata["post_days"] == [5]


def test_pair_covariance_controls_uncertainty_and_keeps_nonzero_null_noise() -> None:
    control = [
        dict(robot_id="r01", task_text="t", outcome="success" if i % 2 else "failure")
        for i in range(60)
    ]
    agreement = arm_estimate(control, deepcopy(control), {"r01": 1.0})
    opposite = [
        dict(r, outcome="failure" if r["outcome"] == "success" else "success") for r in control
    ]
    disagreement = arm_estimate(control, opposite, {"r01": 1.0})
    assert agreement["gain_points"] == disagreement["gain_points"] == 0
    assert 1 < agreement["variance"] < 5
    assert disagreement["variance"] > 100
    assert -5 < agreement["low"] < 0 < agreement["high"] < 5
    assert disagreement["low"] < -15 < 15 < disagreement["high"]
    assert agreement["counts"]["r01"] == [30, 0, 0, 30]


def test_current_evidence_names_causes_in_both_conditions() -> None:
    data = noisy()
    set_rates(data, "r02", 2, 0.05)
    answer = analyse(data, bootstraps=40)
    cause = next(c for c in answer.diagnosis.causes if c.target.scope == "r02")
    action = next(
        r["action"]
        for r in answer.evidence["actions"]
        if r["action"]["target"] == cause.target.model_dump(mode="json")
    )
    for condition in ("experiments", "oracle-proxy"):
        for variance in (0.1, 1000.0):
            updated = revise(
                answer,
                [dict(action=action, gain_points=0.0, low=-1.0, high=1.0, variance=variance)],
                data.costs,
                condition=condition,
            )
            rec = next(r for r in updated.evidence["actions"] if r["action"] == action)
            accepted = named_actions(updated.evidence["actions"])
            supported = any(updated.evidence["actions"][i]["action"] == action for i in accepted)
            causes = [c for c in updated.diagnosis.causes if c.target == cause.target]
            assert bool(causes) == supported
            if variance < 1:
                assert not causes  # Precise null contradicts the observational attribution.
            else:
                assert causes  # A weak null measurement does not erase strong logs.
                assert causes[0].confidence == rec["diagnosis_confidence"]
                assert (
                    causes[0].confidence
                    == updated.evidence[
                        "experiments" if condition == "experiments" else "oracle_estimates"
                    ][0]["combined"]["confidence"]
                )


def test_population_interval_bound_is_outward_for_equivalent_fractions() -> None:
    weight = np.full(24, 1 / 24)
    covered = np.arange(24) < 2
    assert 100 / 12 <= population_bound(weight, covered.astype(float)) < 100 / 12 + 1e-12


def test_later_distinct_break_does_not_support_an_earlier_edit() -> None:
    data = noisy()
    set_rates(data, "r02", 4, 0.0)
    history = [dict(day=1, scope="r02", key=ConfigKey.software_version, old="a", new="b")]
    allowed, reason = _onset_evidence(data, history, change_points(data))
    assert not allowed and "later break" in reason


def test_improvement_alarm_does_not_replace_deficit_site_in_plan() -> None:
    data = noisy()
    sites = [
        s.model_copy(update={"conditions": {"placement": "standard", "object_batch": "standard"}})
        for s in data.fleet.sites
    ]
    answer = analyse(
        replace(data, fleet=data.fleet.model_copy(update={"sites": sites})), bootstraps=20
    )
    evidence = deepcopy(answer.evidence)
    for rec in evidence["actions"]:
        if rec["action"]["target"]["condition_key"]:
            rec.update(
                latent_candidate=True,
                gain_points=2.0 if rec["action"]["target"]["scope"] == "s2" else 0.0,
            )
    evidence["change_points"] = [
        dict(scope="s1", detected=True, direction="improvement", score_z=10.0)
    ]
    chosen = choose(replace(answer, evidence=evidence), requests=1)
    assert all(a.target.scope == "s2" and a.target.condition_key for a in chosen)


def test_release_effect_is_carried_into_joint_selection_and_experiment_transport() -> None:
    from fleetlab.baselines.classical.selection import JointPrediction

    data = competing_release(0.2, 2)
    answer = analyse(data, bootstraps=20)
    forecast = answer.evidence["forecast"]
    joint = JointPrediction(
        np.asarray(forecast["matrix"]),
        np.asarray(answer.evidence["coefficients"]),
        forecast["nuisance"],
        answer.evidence["selected_link"],
        [np.asarray(r) for r in forecast["repaired"]],
        np.asarray(forecast["weight"]),
        np.asarray(forecast["offsets"]),
    )
    for i, rec in enumerate(answer.evidence["actions"]):
        assert joint.gain((i,)) == pytest.approx(rec["gain_points"], abs=1e-8)
    updated = revise(answer, [], data.costs)
    assert updated.fix_set.actions[0].action.target.scope == "fleet"
    assert updated.evidence["selection"]["predicted_prefix_gain_points"][0] == pytest.approx(
        updated.fix_set.actions[0].expected_gain, abs=1e-8
    )


def test_pooled_release_contrast_handles_rotating_tasks_without_complete_cells() -> None:
    points = []
    covered_intervals = 0
    for seed in range(30):
        data = planted(0.0, 3, seed + 8700)
        rng = np.random.default_rng(8900 + seed)
        for row in data.episodes:
            robot = int(row["robot_id"][1:]) - 1
            task = (robot + row["day"]) % 4
            row["task_text"] = f"task-{task}"
            rate = 0.9 - 0.1 * task + 0.04 * (robot % 3 - 1) - 0.05 * (row["day"] >= 3)
            row["outcome"] = "success" if rng.random() < rate else "failure"
        reference = [
            dict(robot_id=r.id, task_text=t)
            for r in data.fleet.robots()
            for t in ("task-0", "task-1", "task-2", "task-3")
        ]
        gains, evidence = release_contrast(
            data, 3, reference, np.full(48, 1 / 48), np.ones(48), draws=200, seed=0
        )
        assert evidence["complete"]
        assert evidence["sampling_variance"] > 0
        points.append(float(gains[0]))
        covered_intervals += evidence["low"] <= 5 <= evidence["high"]
    assert covered_intervals >= 25
    assert 3 < np.mean(points) < 7
    assert sum(p > 0 for p in points) >= 25


def test_release_exactly_aliased_with_task_composition_is_not_identified() -> None:
    data = planted(0.05, 3, 9317)
    for row in data.episodes:
        row["task_text"] = "before-task" if row["day"] < 3 else "after-task"
    reference = [dict(robot_id=r.id) for r in data.fleet.robots()]
    gains, evidence = release_contrast(
        data, 3, reference, np.full(12, 1 / 12), np.ones(12), draws=200, seed=0
    )
    assert not evidence["complete"]
    assert np.all(gains == 0)
