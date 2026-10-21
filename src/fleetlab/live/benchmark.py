"""Time the full telemetry update on recorded command rows; validate offline/live agreement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.live.data import telemetry_rows
from fleetlab.live.features import FeatureWindow
from fleetlab.live.model import RiskModel, TelemetryRiskEstimator


def benchmark(model: RiskModel, run: Path, episodes: int = 32) -> dict[str, Any]:
    durations: list[float] = []
    differences = []
    paths = sorted((run / "log/steps").glob("*.parquet"))[:episodes]
    if not paths:
        raise ValueError("no step files to benchmark")
    for path in paths:
        telemetry = telemetry_rows(path)
        # Warm the code path before measuring, without changing the measured episode state.
        if not durations:
            TelemetryRiskEstimator(model).update(telemetry[0])
        estimator = TelemetryRiskEstimator(model)
        scores = []
        for t in telemetry:
            started = time.perf_counter_ns()
            scores.append(estimator.update(t))
            durations.append((time.perf_counter_ns() - started) / 1e3)
        window = FeatureWindow()
        offline = model.probabilities(np.stack([window.update(t) for t in telemetry]))[:, -1]
        differences.append(float(np.abs(offline - np.asarray(scores)).max()))
    report = {
        "episodes": len(paths),
        "updates": len(durations),
        "microseconds_median": float(np.median(durations)),
        "microseconds_p95": float(np.quantile(durations, 0.95)),
        "microseconds_mean": float(np.mean(durations)),
        "max_absolute_offline_live_probability_difference": max(differences),
        "includes": "feature validation, input copies, window and all three calibrated heads",
    }
    if max(differences) > 1e-12:
        raise RuntimeError(f"offline/live probability mismatch: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(RiskModel.load(args.model), args.run, args.episodes)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
