"""Portable, immutable logistic coefficients; NumPy-only deterministic inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts.live import StepTelemetry
from fleetlab.live.features import FEATURE_NAMES, FeatureWindow, FloatArray


@dataclass(frozen=True)
class Head:
    horizon_steps: int
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0
    threshold: float = 0.5


@dataclass(frozen=True)
class RiskModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    heads: tuple[Head, ...]

    def __post_init__(self) -> None:
        n = len(FEATURE_NAMES)
        if len(self.mean) != n or len(self.scale) != n or not self.heads:
            raise ValueError("model feature dimensions or heads invalid")
        horizons = [h.horizon_steps for h in self.heads]
        if horizons != sorted(set(horizons)) or horizons[0] <= 0:
            raise ValueError("horizons must be positive, increasing and unique")
        if any(s <= 0 for s in self.scale):
            raise ValueError("model scales must be positive")
        values = [*self.mean, *self.scale]
        for head in self.heads:
            if len(head.coefficients) != n or not 0 < head.threshold <= 1:
                raise ValueError("invalid model coefficients or threshold")
            if head.calibration_slope < 0:
                raise ValueError("calibration slope must be nonnegative")
            values += [
                *head.coefficients,
                head.intercept,
                head.calibration_slope,
                head.calibration_intercept,
                head.threshold,
            ]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("model parameters must be finite")

    def probabilities(self, features: FloatArray) -> FloatArray:
        """Final calibrated probabilities, monotone across horizons (cumulative max)."""
        z = np.clip((features - np.asarray(self.mean)) / np.asarray(self.scale), -20, 20)
        logits = np.stack(
            [
                (
                    (z @ np.asarray(h.coefficients) + h.intercept) * h.calibration_slope
                    + h.calibration_intercept
                )
                for h in self.heads
            ],
            axis=-1,
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        return np.minimum(np.maximum.accumulate(probabilities, axis=-1), np.nextafter(1.0, 0.0))

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "version": 1,
            "features": list(FEATURE_NAMES),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "heads": [{**asdict(h), "coefficients": list(h.coefficients)} for h in self.heads],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RiskModel:
        if value.get("version") != 1 or value.get("features") != list(FEATURE_NAMES):
            raise ValueError("incompatible risk model schema")
        return cls(
            tuple(value["mean"]),
            tuple(value["scale"]),
            tuple(Head(**{**h, "coefficients": tuple(h["coefficients"])}) for h in value["heads"]),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: Path, sha256: str | None = None) -> RiskModel:
        data = path.read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("risk model digest differs from SupervisorSpec")
        return cls.from_dict(json.loads(data))


class TelemetryRiskEstimator:
    """Implements contracts.live.RiskEstimator; fresh instance for every episode."""

    def __init__(self, model: RiskModel, horizon_steps: int = 100) -> None:
        self.model = model
        self.index = [h.horizon_steps for h in model.heads].index(horizon_steps)
        self.window = FeatureWindow()

    def update(self, telemetry: StepTelemetry) -> float:
        return float(self.model.probabilities(self.window.update(telemetry))[self.index])
