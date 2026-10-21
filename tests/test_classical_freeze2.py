"""Boundary and integration guarantees found by the blind pre-freeze review."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from classical_planted import planted
from test_classical_baseline import write_export
from test_classical_statistics import noisy, record, set_rates

from fleetlab.agents.sessions import Analysis, ExperimentQueue
from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.analysis import fleet_diagnostic
from fleetlab.baselines.classical.experiments import choose, combine_estimates, plan, revise, update
from fleetlab.baselines.classical.features import Design
from fleetlab.baselines.classical.protocol import Z90, tie_key
from fleetlab.baselines.classical.regression import fit, fit_survival, predict
from fleetlab.contracts.actions import Action
from fleetlab.contracts.tools import ArmResult, ExperimentResult
from fleetlab.contracts.vocabulary import ConfigKey
from fleetlab.toolserver.public_results import COMPLETED


@pytest.mark.parametrize("link", ["logistic", "multiplicative"])
@pytest.mark.parametrize("onset", [1, 2, 3])
@pytest.mark.parametrize("drop", [0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
def test_positive_fleet_repair_is_representable_with_either_forced_link(
    link: str, onset: int, drop: float
) -> None:
    data = planted(drop, onset, 1000)
    design = Design(data)
    y = np.asarray([r["outcome"] == "success" for r in data.episodes], dtype=float)
    beta = (
        fit(design.x, y, design.penalty)
        if link == "logistic"
        else fit_survival(design.x, y, design.penalty, design.nuisance)
    )
    action = next(a for a in design.candidates() if a.target.scope == "fleet")
    base = design.matrix(data.episodes, future=True)
    repaired = design.matrix(data.episodes, future=True, action=action)
    gain = float(
        np.mean(
            predict(repaired, beta, design.nuisance, link)
            - predict(base, beta, design.nuisance, link)
        )
        * 100
    )
    assert gain > 1e-4, (onset, link, drop, gain)


def test_within_key_levels_are_not_aliases_and_robot_heterogeneity_is_visible() -> None:
    data = planted(0.30, 2, 1000)
    answer = analyse(data, bootstraps=20)
    effect = record(answer, "fleet", str(ConfigKey.software_version))
    assert effect["gain_points"] > 20 and not effect["aliases"]
    assert effect["observational_pooling"]  # conditional precision, like other regressions
    common = fleet_diagnostic(data, 2)
    local = fleet_diagnostic(planted(0.40, 2, 1000, True), 2)
    assert common["heterogeneity_p"] > local["heterogeneity_p"]
    assert local["heterogeneity_p"] < 0.05
    # Unlogged local changes cannot be uniquely attributed to the fleet release.
    unlogged = planted(0.40, 2, 1000, True)
    unlogged.changes[:] = unlogged.changes[:1]
    result = analyse(unlogged, bootstraps=20)
    candidate = record(result, "fleet", str(ConfigKey.software_version))
    # Heterogeneity is descriptive; the release is ranked from the local pooled
    # contrast, with no veto on genuine fleet effects that differ across robots.
    assert candidate["gain_points"] > 0
    assert candidate["onset_contrast"]["complete"]
    assert candidate["fleet_contrast"]["heterogeneity_p"] < 0.05
    assert candidate["observational_pooling"] == candidate["informative_interval"]
    # Unlogged coincident local causes remain a declared identification limit;
    # raw heterogeneity must not veto a regression estimate or its diagnosis.


def test_collapsed_or_conflicting_prior_does_not_overrule_randomization() -> None:
    prior = dict(
        gain_points=0.0, low=0.0, high=0.0, estimation_variance=0.0, informative_interval=True
    )
    measured = dict(gain_points=10.0, low=8.0, high=12.0, variance=1.0)
    assert combine_estimates(prior, measured)["gain_points"] == 10
    # Also below the separate 95% source-conflict threshold: this positive
    # 90% measurement still cannot inherit infinite precision from a boundary.
    assert (
        combine_estimates(prior, dict(gain_points=1.0, low=0.1, high=1.9, variance=0.3))[
            "gain_points"
        ]
        == 1
    )
    conflict = combine_estimates(
        {**prior, "gain_points": 7.0, "estimation_variance": 1.0},
        dict(gain_points=0.0, low=-1.2, high=1.2, variance=0.5),
    )
    assert conflict["source_conflict"]
    assert conflict["measurement_weight"] > 0.98
    assert abs(conflict["gain_points"]) < 0.1 and conflict["low"] < 0
    unresolved = combine_estimates(
        {**prior, "gain_points": 0.4, "informative_interval": False},
        dict(gain_points=3.3, low=-1.3, high=7.9, variance=8.0),
    )
    assert unresolved["gain_points"] == 0.4
    assert unresolved["precision_point"] == 3.3
    assert unresolved["unresolved_promotion_screened"]
    assert unresolved["low"] < 0 < unresolved["high"]


def test_fleet_reservation_is_independent_of_rank_and_respects_capacity() -> None:
    answer = analyse(planted(0.05, 1, 1000), bootstraps=20)
    fleet = next(r for r in answer.fix_set.actions if r.action.target.scope == "fleet")
    reordered = answer.fix_set.model_copy(
        update={"actions": [r for r in answer.fix_set.actions if r != fleet] + [fleet]}
    )
    chosen = choose(replace(answer, fix_set=reordered))
    assert len(chosen) == 8 and chosen[0] == fleet.action
    assert choose(replace(answer, fix_set=reordered), requests=1) == [fleet.action]


def test_equal_action_ties_reach_joint_ranking_and_negative_tests_remove_cause() -> None:
    data = noisy()
    set_rates(data, "r02", 2, 0.05)
    answer = analyse(data, bootstraps=20)
    cause = next(c for c in answer.diagnosis.causes if c.target.config_key)
    action = next(r.action for r in answer.fix_set.actions if r.action.target == cause.target)
    estimates = [
        dict(
            action=action.model_dump(mode="json"),
            gain_points=-3.0,
            low=-4.0,
            high=-2.0,
            variance=0.3,
            confidence=0.0,
        )
    ]
    updated = revise(answer, estimates, data.costs)
    assert cause.target not in [c.target for c in updated.diagnosis.causes]
    for r in updated.evidence["actions"]:
        if r.get("experimental"):
            assert r["prediction_radius"] > 0
            pooled = updated.evidence["experiments"][0]["combined"]
            assert r["low"] < pooled["low"] and r["high"] > pooled["high"]
            assert r["prediction_radius"] <= Z90 * np.sqrt((1 / 6) / 480) * 100
    # All repairs identical under an all-zero fit. The complete ranking must
    # use the declared stable hash after equal cost, not catalogue spelling.
    zero = deepcopy(answer.evidence)
    zero["coefficients"] = [0.0] * len(zero["coefficients"])
    from fleetlab.contracts.actions import cost_of

    actions = [Action.model_validate(r["action"]) for r in zero["actions"]]
    ranked = revise(replace(answer, evidence=zero), [], data.costs).fix_set.actions
    assert [r.action for r in ranked] == sorted(
        actions, key=lambda a: (cost_of(a, data.costs) or 0, tie_key(a))
    )[: len(ranked)]


def test_update_refuses_wrong_window_even_for_completed_result(tmp_path: Path) -> None:
    data = noisy()
    set_rates(data, "r02", 2, 0.01)
    answer = analyse(data, bootstraps=20)
    export, analysis_root = tmp_path / "export", tmp_path / "analysis"
    write_export(export, data)
    queue = ExperimentQueue.model_validate_json(
        plan(export, answer, analysis_root, requests=1, episodes_per_arm=1).read_text()
    )
    request = queue.experiments[0]
    robot = data.fleet.robot(request.request.robots[0])
    rows = list(data.episodes)
    arms = []
    for index in (0, 1):
        episode = f"e{900000 + index:06}"
        rows.append(
            dict(
                episode_id=episode,
                robot_id=robot.id,
                site_id=robot.site_id,
                day=8,
                task_text="easy",
                outcome="failure",
                steps=280,
                experiment_id=request.experiment_id,
            )
        )
        arms.append(ArmResult(arm_index=index, episode_ids=[episode]))
    pq.write_table(pa.Table.from_pylist(rows), export / "log/episodes.parquet")
    Analysis(analysis_root).record_results(
        [
            ExperimentResult(
                experiment_id=request.experiment_id, status="done", arms=arms, message=COMPLETED
            )
        ]
    )
    with pytest.raises(ValueError, match="outside the experiment window"):
        update(export, answer, analysis_root)


def test_pooled_heterogeneity_noise_has_reasonable_null_size() -> None:
    # Sparse day-one cells often contain all successes. Estimating each one's
    # own variance spuriously rejects a common five-point fault about 24% of
    # the time in this fixed independent study. Pooling phase/task noise avoids
    # that precision artifact; this is calibration evidence, not world tuning.
    for onset in (1, 3, 5):
        rejected = sum(
            fleet_diagnostic(planted(0.05, onset, 1000 + seed), onset)["heterogeneity_p"] < 0.05
            for seed in range(300)
        )
        assert rejected <= 30


@pytest.mark.parametrize("link", ["logistic", "multiplicative"])
def test_real_fleet_fault_with_logged_local_faults_is_not_vetoed(
    link: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from fleetlab.baselines.classical import analysis
    from fleetlab.baselines.classical.regression import Regression, bootstrap_weights

    data = planted(0.2, 2, 1000)
    local = {"r01", "r05", "r09"}
    data.changes.extend(
        dict(day=2, scope=r, key=ConfigKey.control_action_delay_steps, old=0, new=20, author="ops")
        for r in sorted(local)
    )
    rng = np.random.default_rng(8402)
    for row in data.episodes:
        probability = (0.85 if row["task_text"] == "easy" else 0.65) - (
            0.2 + 0.4 * (row["robot_id"] in local)
        ) * (row["day"] >= 2)
        success = rng.random() < probability
        row["outcome"] = "success" if success else "failure"
        row["steps"] = int(rng.integers(90, 230)) if success else 280

    def forced(
        x: Any, y: Any, penalty: Any, clusters: Any, folds: Any, **kwargs: Any
    ) -> Regression:
        def estimate(weights: Any = None) -> Any:
            return (
                fit(x, y, penalty, weights)
                if link == "logistic"
                else fit_survival(x, y, penalty, kwargs["nuisance"], weights)
            )

        noise = np.random.default_rng(0)
        draws = [estimate(bootstrap_weights(clusters, noise, kwargs["sites"])) for _ in range(40)]
        return Regression(link, estimate(), np.asarray(draws), 1.0, {link: 0.0})

    monkeypatch.setattr(analysis, "regress", forced)
    answer = analyse(data, bootstraps=40)
    rec = record(answer, "fleet", str(ConfigKey.software_version))
    assert rec["fleet_contrast"]["heterogeneity_p"] < 0.05
    assert rec["observational_pooling"] and rec["diagnosis_confidence"] > 0.9
    assert any(c.target.scope == "fleet" for c in answer.diagnosis.causes)
    # A noisy weak fleet measurement must be precision-pooled, not overwrite it.
    observed = dict(
        gain_points=10.0,
        low=6.0,
        high=14.0,
        estimation_variance=4.0,
        informative_interval=True,
        observational_pooling=True,
    )
    measured = dict(gain_points=2.0, low=-12.0, high=16.0, variance=70.0)
    pooled = combine_estimates(observed, measured)
    assert not pooled["source_conflict"] and pooled["gain_points"] > 9
