"""Short health histories: pooled noise, explicit steps and segmented trends.

Engineering sign conventions are declared here, not inferred from fault code.
Unknown metrics get both tails. Usage counters never imply a repair benefit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from fleetlab.baselines.classical.data import Row, VisibleData
from fleetlab.baselines.classical.regression import Array

METRIC_DIRECTION = {
    "grip_force": -1.0,
    "calibration_residual": 1.0,
    "joint_torque_rms": 1.0,
    "cpu_load": 1.0,
}
USAGE_METRICS = {"cycle_count"}


@dataclass(frozen=True)
class Trajectory:
    kind: str
    onset: float
    intercept: float
    jump: float
    slope: float
    last_day: float
    slope_se: float
    maintenance_days: tuple[int, ...]

    def reading(self, day: float, flat: bool = False) -> float:
        day = min(day, self.last_day) if flat else day
        return (
            self.intercept
            + self.jump * (day >= self.onset)
            + self.slope * max(0.0, day - self.onset)
        )


@dataclass(frozen=True)
class HealthSeries:
    component: str
    metric: str
    center: float
    scale: float
    direction: float
    noise: float
    robots: dict[str, Trajectory]

    def exposure(self, robot: str, day: float, *, flat: bool = False) -> float:
        trajectory = self.robots.get(robot)
        if trajectory is None:
            return 0.0
        z = self.direction * (trajectory.reading(day, flat) - self.center) / self.scale
        # Soft compression avoids letting a single extreme check dominate the fit,
        # without saturating a real drift at an arbitrary maximum z-score.
        return float(np.log1p(max(z - 1.0, 0.0)))


def _trajectory(rows: list[Row], noise: float, maintenance: tuple[int, ...]) -> Trajectory:
    days = np.asarray([r["day"] for r in rows], dtype=float)
    values = np.asarray([r["value"] for r in rows], dtype=float)
    candidates: list[tuple[str, float, Array]] = [("level", 0, np.ones((len(rows), 1)))]
    if len(rows) >= 3:
        candidates.append(
            ("linear", float(days[0]), np.column_stack([np.ones(len(rows)), days - days[0]]))
        )
    # A change can begin at any observed boundary; maintenance dates are included
    # as ordinary candidates, never assumed successful or causal.
    for onset in sorted(set(days[1:].tolist()) | set(maintenance)):
        if np.sum(days < onset) < 1 or np.sum(days >= onset) < 2:
            continue
        candidates.append(("step", onset, np.column_stack([np.ones(len(rows)), days >= onset])))
        if len(rows) >= 4:
            candidates.append(
                ("hinge", onset, np.column_stack([np.ones(len(rows)), np.maximum(days - onset, 0)]))
            )
    choices = []
    for kind, onset, x in candidates:
        beta = np.linalg.lstsq(x, values, rcond=None)[0]
        # Student-t(4) errors tolerate isolated sensor spikes. Gaussian squared
        # error can invent an accelerating drift to chase one final reading.
        for _ in range(100):
            residual = (values - x @ beta) / noise
            weights = 5 / (4 + residual**2)
            updated = np.linalg.lstsq(
                x * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None
            )[0]
            if float(np.max(np.abs(updated - beta))) < 1e-7:
                beta = updated
                break
            beta = updated
        residual = values - x @ beta
        if kind != "level":
            departure = (values - beta[0]) * np.sign(beta[1])
            post = days >= onset if kind == "step" else days > onset
            # Two observed departures support a persistent change. One large
            # final reading cannot make an entire ramp by treating its healthy
            # predecessor as an outlier instead.
            if np.sum(post & (departure > 2 * noise)) < 2:
                continue
        # Per-series BIC; step/hinge searches pay for the scanned locations.
        # Pool noise, not the significance threshold: a larger fleet should not
        # force every modest real drift to an exactly flat forecast.
        search = 2 * np.log(max(1, len(rows) - 2)) if kind in {"step", "hinge"} else 0.0
        criterion = float(
            np.sum(5 * np.log1p(residual**2 / (4 * noise**2)))
            + (x.shape[1] - 1) * np.log(max(2, len(rows)))
            + search
        )
        choices.append((criterion, kind, onset, x, beta))
    _, kind, onset, x, beta = min(choices, key=lambda a: (a[0], a[1], a[2]))
    slope = float(beta[1]) if kind in {"linear", "hinge"} else 0.0
    slope_se = float(noise * np.sqrt(np.linalg.pinv(x.T @ x)[-1, -1])) if slope else 0.0
    return Trajectory(
        kind,
        onset,
        float(beta[0]),
        float(beta[1]) if kind == "step" else 0,
        slope,
        float(days[-1]),
        slope_se,
        maintenance,
    )


def health_series(data: VisibleData) -> list[HealthSeries]:
    grouped: dict[tuple[str, str], dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in data.health:
        key = row["robot_id"], row["component"], row["metric"], row["day"]
        if key in seen or not np.isfinite(row["value"]):
            raise ValueError("duplicate or non-finite health reading")
        seen.add(key)
        if row["metric"] not in USAGE_METRICS:
            grouped[row["component"], row["metric"]][row["robot_id"]].append(row)
    series = []
    for (component, metric), by_robot in sorted(grouped.items()):
        rows_by_robot = {r: sorted(rows, key=lambda v: v["day"]) for r, rows in by_robot.items()}
        # Second differences cancel linear drifts. Median pooling tolerates a
        # minority of jumps/noisy robots; sqrt(6) is their Gaussian noise scale.
        differences = [
            v for rows in rows_by_robot.values() for v in np.diff([r["value"] for r in rows], n=2)
        ]
        values = [r["value"] for rows in rows_by_robot.values() for r in rows]
        noise = max(
            1.4826 * float(np.median(np.abs(differences))) / np.sqrt(6)
            if differences
            else float(np.std(values)),
            1e-6,
        )
        robots = {}
        for robot, rows in rows_by_robot.items():
            maintenance = tuple(
                sorted(
                    m["day"]
                    for m in data.maintenance
                    if m["robot_id"] == robot and m["component"] == component
                )
            )
            robots[robot] = _trajectory(rows, noise, maintenance)
        baselines = np.asarray([r.reading(0) for r in robots.values()])
        center = float(np.median(baselines))
        scale = max(1.4826 * float(np.median(np.abs(baselines - center))), noise, 1e-6)
        directions = [METRIC_DIRECTION[metric]] if metric in METRIC_DIRECTION else [-1.0, 1.0]
        for direction in directions:
            series.append(HealthSeries(component, metric, center, scale, direction, noise, robots))
    return series
