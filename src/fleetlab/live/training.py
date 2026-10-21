"""CPU logistic fitting and held-out-robot sigmoid calibration (no resampling)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from fleetlab.live.data import HORIZONS, Dataset
from fleetlab.live.features import FEATURE_NAMES, FloatArray
from fleetlab.live.model import Head, RiskModel


def _logistic_fit(
    x: FloatArray, y: FloatArray, penalty: float
) -> tuple[FloatArray, dict[str, Any]]:
    if len(np.unique(y)) < 2:
        raise ValueError("fitting needs both positive and negative labels")
    initial = np.zeros(x.shape[1] + 1, np.float64)
    initial[-1] = np.log(y.mean() / (1 - y.mean()))

    def objective(w: FloatArray) -> tuple[float, FloatArray]:
        scores = x @ w[:-1] + w[-1]
        loss = float(
            np.mean(np.logaddexp(0, scores) - y * scores) + penalty * (w[:-1] @ w[:-1]) / 2
        )
        error = 1 / (1 + np.exp(-np.clip(scores, -40, 40))) - y
        gradient = np.concatenate(
            [x.T @ error / len(y) + penalty * w[:-1], np.asarray([error.mean()])]
        )
        return loss, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"logistic optimizer did not converge: {result.message}")
    return np.asarray(result.x), {"iterations": int(result.nit), "loss": float(result.fun)}


def fit(
    train: Dataset, calibration: Dataset, *, penalty: float = 0.001, time_only: bool = False
) -> tuple[RiskModel, list[dict[str, Any]]]:
    """Fixed regularization, unweighted step population, no held-out test inputs."""
    train_robots = {(e["world"], e["robot"]) for e in train.episodes}
    cal_robots = {(e["world"], e["robot"]) for e in calibration.episodes}
    if train_robots & cal_robots:
        raise ValueError("training and calibration robots overlap")
    mean = train.x.mean(axis=0)
    scale = train.x.std(axis=0)
    scale[scale < 1e-8] = 1
    columns = [
        i
        for i, name in enumerate(FEATURE_NAMES)
        if not time_only or name == "elapsed" or name.startswith("time_after_")
    ]
    x = np.clip((train.x - mean) / scale, -20, 20)[:, columns].copy()
    xc = np.clip((calibration.x - mean) / scale, -20, 20)[:, columns].copy()
    heads: list[Head] = []
    records = []
    for horizon in HORIZONS:
        y = train.labels(horizon).astype(np.float64)
        yc = calibration.labels(horizon).astype(np.float64)
        weights, record = _logistic_fit(x, y, penalty)
        scores = xc @ weights[:-1] + weights[-1]
        calibrated, cal_record = _logistic_fit(scores[:, None], yc, 1e-6)
        # Constrain a calibration reversal to the constant prevalence predictor.
        if calibrated[0] < 0:
            calibrated = np.asarray([0.0, np.log(yc.mean() / (1 - yc.mean()))])
        coefficients = np.zeros(len(FEATURE_NAMES), np.float64)
        coefficients[columns] = weights[:-1]
        heads.append(
            Head(
                horizon,
                tuple(coefficients),
                float(weights[-1]),
                float(calibrated[0]),
                float(calibrated[-1]),
            )
        )
        records.append(
            {
                "horizon_steps": horizon,
                "fit": record,
                "calibration": cal_record,
                "train_rows": len(y),
                "train_positive": int(y.sum()),
                "calibration_rows": len(yc),
                "calibration_positive": int(yc.sum()),
                "penalty": penalty,
                "time_only": time_only,
            }
        )
    return RiskModel(tuple(mean), tuple(scale), tuple(heads)), records


def with_thresholds(model: RiskModel, thresholds: list[float]) -> RiskModel:
    return replace(
        model,
        heads=tuple(replace(h, threshold=t) for h, t in zip(model.heads, thresholds, strict=True)),
    )
