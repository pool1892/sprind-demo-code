"""The daily health checks of a world, version 1.

One reading per robot, component, metric and day, as a fleet's periodic checks log them
(`METRICS`): per metric a baseline drawn per robot, day-to-day noise, a benign drift per
robot, and the faults that move it.

- `scene_camera` and `wrist_camera`, `calibration_residual` (px): baseline 10 to 35,
  noise 20 (about the residual of a 3 degree offset), drift N(0, 0.8) per day; a camera
  pose fault adds its residual at the day's level.
- `gripper`, `grip_force` (N): baseline 55 to 65, noise 4 (a tenth of the range a fleet
  sees), drift 0 to -0.25 per day (pads wear benignly); gripper wear takes its share.
- `gripper`, `cycle_count`: baseline 20,000 to 80,000, plus the day's episodes.
- `arm`, `joint_torque_rms` (Nm): baseline 10 to 14, noise 0.8, drift N(0, 0.03).
- `controller`, `cpu_load` (share): baseline 0.35 to 0.6, noise 0.04, drift N(0, 0.004).

Benign drift accumulates from day 0 and is taken back by the records that would take
it back in a real fleet: a calibration record of a camera (a benign `Change` of its
calibration key) resets that camera's residual drift, cleaning the gripper halves its
force drift, servicing the arm or the controller resets theirs. Configuration,
software, site and data faults never appear here. A residual is the size of an error,
so the reading is its absolute value; no reading goes below zero. Every draw is
seeded by world seed, robot, metric (and day for the noise).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.model import Change, Maintenance
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.worlds import base

RESIDUAL_NOISE_PX = 20.0

# component, metric, baseline range, noise sd, drift draw ("normal", sd) or ("down", max)
METRICS: tuple[tuple[Component, str, tuple[float, float], float, tuple[str, float]], ...] = (
    (
        Component.scene_camera,
        "calibration_residual",
        (10.0, 35.0),
        RESIDUAL_NOISE_PX,
        ("normal", 0.8),
    ),
    (
        Component.wrist_camera,
        "calibration_residual",
        (10.0, 35.0),
        RESIDUAL_NOISE_PX,
        ("normal", 0.8),
    ),
    (Component.gripper, "grip_force", (55.0, 65.0), 4.0, ("down", 0.25)),
    (Component.gripper, "cycle_count", (20_000.0, 80_000.0), 0.0, ("none", 0.0)),
    (Component.arm, "joint_torque_rms", (10.0, 14.0), 0.8, ("normal", 0.03)),
    (Component.controller, "cpu_load", (0.35, 0.6), 0.04, ("normal", 0.004)),
)
_CALIBRATION_KEYS = {
    ConfigKey.camera_scene_calibration: Component.scene_camera,
    ConfigKey.camera_wrist_calibration: Component.wrist_camera,
}


def _resets(
    robot: str,
    component: Component,
    metric: str,
    changes: Sequence[Change],
    maintenance: Sequence[Maintenance],
) -> list[tuple[int, float]]:
    """(day, share of the drift kept) for every record that takes benign drift back."""
    out: list[tuple[int, float]] = []
    if metric == "calibration_residual":
        for c in changes:
            if c.scope == robot and _CALIBRATION_KEYS.get(c.key) is component:
                out.append((c.day, 0.0))
    for m in maintenance:
        if m.robot_id != robot or m.component is not component:
            continue
        if metric == "grip_force" and m.verb is Verb.clean:
            out.append((m.day, 0.5))
        elif metric in ("joint_torque_rms", "cpu_load") and m.verb is Verb.service:
            out.append((m.day, 0.0))
    return sorted(out)


def rows(
    seed: int,
    robots: Sequence[tuple[str, str]],
    days: int,
    episodes_per_day: int,
    schedule: Sequence[FaultSpec],
    changes: Sequence[Change],
    maintenance: Sequence[Maintenance],
) -> list[dict[str, Any]]:
    """Every reading of the world, sorted by day, robot, component and metric."""
    from fleetlab.faults.records import health_shift

    out: list[dict[str, Any]] = []
    for robot, site in robots:
        for component, metric, span, sd, (drift_kind, drift_size) in METRICS:
            draw = base.rng(seed, "health-base", robot, component.value, metric)
            baseline = float(draw.uniform(*span))
            if drift_kind == "normal":
                slope = float(draw.normal(0.0, drift_size))
            elif drift_kind == "down":
                slope = -float(draw.uniform(0.0, drift_size))
            else:
                slope = 0.0
            resets = _resets(robot, component, metric, changes, maintenance)
            drift = 0.0
            for day in range(days):
                if day > 0:
                    drift += slope
                for reset_day, kept in resets:
                    if reset_day == day:
                        drift *= kept
                if metric == "cycle_count":
                    value = baseline + episodes_per_day * day
                else:
                    noise = base.rng(seed, "health-noise", robot, component.value, metric, day)
                    value = baseline + drift + (float(noise.normal(0.0, sd)) if sd else 0.0)
                    if metric == "calibration_residual":
                        value = abs(value)
                for fault in base.in_force(schedule, robot, site, day):
                    if fault.target.component is not component:
                        continue
                    shift = health_shift(fault)
                    if metric in shift:
                        value += shift[metric]
                    if metric == "grip_force" and "grip_force_loss" in shift:
                        value *= 1.0 - shift["grip_force_loss"]
                value = max(0.0, value)
                out.append(
                    {
                        "day": day,
                        "robot_id": robot,
                        "component": component.value,
                        "metric": metric,
                        "value": round(value, 3 if metric == "cpu_load" else 2),
                    }
                )
    out.sort(key=lambda r: (r["day"], r["robot_id"], r["component"], r["metric"]))
    return out
