"""Causal inputs, forecast labels, robot separation and portable model validation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fleetlab.contracts.live import StepTelemetry
from fleetlab.live.data import Dataset
from fleetlab.live.evaluation import alarm_metrics, auc, first_calls, forecast_metrics
from fleetlab.live.features import FEATURE_NAMES, FeatureWindow
from fleetlab.live.model import Head, RiskModel, TelemetryRiskEstimator
from fleetlab.live.training import fit


def telemetry(step: int, x: float = 0.0) -> StepTelemetry:
    return StepTelemetry(
        step,
        step / 20,
        np.zeros(7),
        np.zeros(7),
        np.zeros(2),
        np.asarray([x, 0, 1.0]),
        np.asarray([0.0, 0, 0, 1]),
        np.asarray([x, 0, 1, 0, 0, 0, -1], np.float32),
        0,
        step,
    )


def constant_model(probability: float = 0.7) -> RiskModel:
    n = len(FEATURE_NAMES)
    intercept = float(np.log(probability / (1 - probability)))
    return RiskModel(
        (0.0,) * n, (1.0,) * n, tuple(Head(h, (0.0,) * n, intercept) for h in (20, 40, 100))
    )


def dataset(robot: str = "r01") -> Dataset:
    # Two independently labelled episodes, one early failure and one success.
    steps = np.tile(np.arange(60, dtype=np.int64), 2)
    x = np.zeros((120, len(FEATURE_NAMES)), np.float64)
    x[:60, FEATURE_NAMES.index("opening")] = 0.05
    x[:, 0] = steps / 280
    return Dataset(
        x,
        steps,
        np.repeat(np.arange(2, dtype=np.int64), 60),
        [
            {
                "world": "toy",
                "run": "observed",
                "id": str(i),
                "robot": robot,
                "terminal": 60,
                "failure": i == 0,
            }
            for i in range(2)
        ],
        [],
    )


def test_features_are_causal_and_copy_retained_buffers() -> None:
    first, second = FeatureWindow(), FeatureWindow()
    t = telemetry(0)
    assert np.array_equal(first.update(t), second.update(telemetry(0)))
    t.eef_pos[0] = 100  # later caller mutation must not rewrite history
    for step in range(1, 30):
        a = first.update(telemetry(step, step / 100))
        b = second.update(telemetry(step, step / 100))
        assert np.array_equal(a, b)
    assert len(a) == len(FEATURE_NAMES)
    assert a[FEATURE_NAMES.index("window_path")] == pytest.approx(0.2)


def test_bad_or_discontinuous_telemetry_is_refused() -> None:
    with pytest.raises(ValueError, match="start at zero"):
        FeatureWindow().update(telemetry(10))
    with pytest.raises(ValueError, match="eef_pos"):
        FeatureWindow().update(replace(telemetry(0), eef_pos=np.full(3, np.nan)))
    window = FeatureWindow()
    window.update(telemetry(0))
    with pytest.raises(ValueError, match="consecutive"):
        window.update(telemetry(0))


def test_command_origin_does_not_create_a_pose_tracking_error() -> None:
    absolute, relative = FeatureWindow(), FeatureWindow()
    for step in range(30):
        a = telemetry(step, step / 32)
        command = a.commanded_action.copy()
        command[:3] -= np.asarray([0.5, 0.25, 1], np.float32)
        b = replace(a, commanded_action=command)
        np.testing.assert_array_equal(absolute.update(a), relative.update(b))


def test_horizon_labels_use_terminal_failure_and_never_success() -> None:
    data = dataset()
    assert np.flatnonzero(data.labels(20)).tolist() == list(range(40, 60))
    assert np.flatnonzero(data.labels(40)).tolist() == list(range(20, 60))
    assert data.labels(100).sum() == 60


def test_alarm_delay_and_no_call_on_terminal_state() -> None:
    data = dataset()
    fired = np.zeros(120, np.bool_)
    fired[59] = True  # score cannot act before this failure's terminal state
    fired[80] = True  # false alarm at step 21 of the success
    assert first_calls(data, fired).tolist() == [-1, 21]
    fired[40] = True
    metrics = alarm_metrics(data, fired, 20)
    assert metrics["caught"] == 1
    assert metrics["lead_seconds_median"] == 0.95
    assert metrics["false_alarms_per_episode"] == 0.5
    assert metrics["successful_episode_alarm_rate"] == 1.0


def test_auc_ties_and_single_class_and_reliability_denominator() -> None:
    y = np.asarray([False, False, True, True])
    assert auc(y, np.ones(4)) == 0.5
    assert auc(y, y.astype(np.float64)) == 1
    assert auc(y, 1 - y.astype(np.float64)) == 0
    assert auc(np.ones(4, np.bool_), np.arange(4, dtype=np.float64)) is None
    metrics = forecast_metrics(y, np.asarray([0, 0.1, 0.9, 1]))
    assert sum(row["rows"] for row in metrics["reliability"]) == 4


def test_model_roundtrip_is_exact_and_bad_digest_refused(tmp_path: Path) -> None:
    model = constant_model()
    path = tmp_path / "model.json"
    model.save(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = RiskModel.load(path, sha)
    a, b = TelemetryRiskEstimator(model), TelemetryRiskEstimator(restored)
    assert [a.update(telemetry(t)) for t in range(20)] == [
        b.update(telemetry(t)) for t in range(20)
    ]
    with pytest.raises(ValueError, match="digest"):
        RiskModel.load(path, "0" * 64)
    with pytest.raises(ValueError, match="scales"):
        replace(model, scale=(0.0,) * len(FEATURE_NAMES))
    with pytest.raises(ValueError, match="finite"):
        replace(model, mean=(float("nan"),) * len(FEATURE_NAMES))


def test_fit_calibrates_without_sharing_robots_and_cache_roundtrips(tmp_path: Path) -> None:
    train, calibration = dataset(), dataset("r03")
    with pytest.raises(ValueError, match="overlap"):
        fit(train, train)
    model, records = fit(train, calibration)
    probabilities = model.probabilities(calibration.x)
    assert np.all(np.diff(probabilities, axis=1) >= 0)
    assert probabilities[:60, 2].mean() > 0.95
    assert probabilities[60:, 2].mean() < 0.05
    assert records[0]["train_positive"] == 20
    path = tmp_path / "data.npz"
    train.save(path)
    restored = Dataset.load(path)
    assert np.array_equal(restored.x, train.x)
    assert restored.episodes == train.episodes


def test_every_exact_reliability_boundary_is_counted_once() -> None:
    p = np.arange(11, dtype=np.float64) / 10
    metrics = forecast_metrics(np.zeros(11, np.bool_), p)
    assert sum(row["rows"] for row in metrics["reliability"]) == len(p)


def test_horizon_false_alarm_is_measured_from_score_not_delayed_call() -> None:
    data = dataset()
    fired = np.zeros(120, np.bool_)
    fired[39] = True  # failure at 60 is 21 steps from this score, 20 from the call
    assert not data.labels(20)[39]
    assert alarm_metrics(data, fired, 20)["outside_horizon_alarms_per_episode"] == 0.5
