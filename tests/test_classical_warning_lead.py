"""A noisy prospective warning, generated without any development-world inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.special import expit, logit
from test_classical_statistics import noisy

from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.data import Row, VisibleData
from fleetlab.contracts.actions import catalogue
from fleetlab.contracts.model import ConfigVersion


def warning_fixture() -> VisibleData:
    data = noisy()
    sites = [
        site.model_copy(
            update={
                "robots": [
                    robot.model_copy(update={"config": ConfigVersion(version=0, values={})})
                    for robot in site.robots
                ]
            }
        )
        for site in data.fleet.sites
    ]
    fleet = data.fleet.model_copy(update={"sites": sites, "change_log": []})

    def dose(robot: str, day: int) -> float:
        # A static impaired donor identifies the health/outcome relationship;
        # only r05 has a changing health exposure.
        if robot == "r01":
            return 30.0
        return 4.0 * max(0, day - 3) if robot == "r05" else 0.0

    def probability(robot: str, task: str, day: int) -> float:
        baseline = (
            (0.975 if task == "easy" else 0.95)
            if robot == "r05"
            else (0.82 if task == "easy" else 0.74)
        )
        return float(expit(logit(baseline) - 0.7 * np.log1p(dose(robot, day) / 0.15)))

    tasks = ("easy", "hard")

    def planted_failure_gap(day: int) -> float:
        fleet_success = np.mean(
            [probability(robot.id, task, day) for robot in fleet.robots() for task in tasks]
        )
        robot_success = np.mean([probability("r05", task, day) for task in tasks])
        return float(fleet_success - robot_success)

    # The true population crosses after issuance, independent of sampling noise:
    # 1.67 percentage points on day 5, 7.13 points on day 6.
    assert planted_failure_gap(5) < 0.05 < planted_failure_gap(6)
    readings = np.random.default_rng(8021)
    outcomes = np.random.default_rng(8022)
    health: list[Row] = []
    episodes: list[Row] = []
    for robot in fleet.robots():
        for day in range(6):
            health.append(
                dict(
                    robot_id=robot.id,
                    day=day,
                    component="scene_camera",
                    metric="calibration_residual",
                    value=20 + dose(robot.id, day) + float(readings.normal(0, 0.15)),
                )
            )
            for task in tasks:
                for _ in range(100):
                    success = outcomes.random() < probability(robot.id, task, day)
                    episodes.append(
                        dict(
                            episode_id=f"e{len(episodes) + 1:06}",
                            robot_id=robot.id,
                            site_id=robot.site_id,
                            day=day,
                            task_text=task,
                            outcome="success" if success else "failure",
                            steps=int(outcomes.integers(90, 230)) if success else 280,
                            experiment_id=None,
                        )
                    )
    return replace(
        data,
        fleet=fleet,
        episodes=episodes,
        health=health,
        changes=[],
        events=[],
        actions=catalogue(fleet),
    )


def test_adverse_health_drift_warns_before_a_planted_threshold_crossing() -> None:
    data = warning_fixture()
    answer = analyse(data, bootstraps=20)
    assert len(data.episodes) == 7200
    assert [warning.scope for warning in answer.diagnosis.warnings] == ["r05"]
    warning = answer.diagnosis.warnings[0]
    assert warning.metric == "failure_rate" and 5 < warning.expected_day <= 9
    forecast = next(
        row
        for row in answer.evidence["warning_forecasts"]
        if row["scope"] == "r05" and row["metric"] == "calibration_residual"
    )
    assert forecast["issued_day"] == 5
    assert forecast["crossing_day"] == warning.expected_day
    assert forecast["lead_days"] == warning.expected_day - 5 > 0


def test_a_forecast_crossing_needs_support_for_the_health_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.baselines.classical import analysis
    from fleetlab.baselines.classical.features import Design
    from fleetlab.baselines.classical.regression import Regression, regress

    data = warning_fixture()
    design = Design(data)
    columns = [i for i, n in enumerate(design.names) if n.startswith("health:")]

    def unsupported(*args: Any, **kwargs: Any) -> Regression:
        fitted = regress(*args, **kwargs)
        draws = fitted.draws.copy()
        draws[:, columns] = 0
        return replace(fitted, draws=draws)

    monkeypatch.setattr(analysis, "regress", unsupported)
    answer = analyse(data, bootstraps=20)
    assert not answer.diagnosis.warnings
