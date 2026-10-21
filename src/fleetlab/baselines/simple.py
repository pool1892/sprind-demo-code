"""The uniform-random, fixed-prior fallback and worst-health methods."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from fleetlab.baselines.common import (
    BaselineAnswer,
    component_actions,
    export_catalogue,
    ranking,
    visible_fleet,
)
from fleetlab.baselines.prior import trivial_no_data_ranker
from fleetlab.baselines.random import random_fixes as random_fixes
from fleetlab.log.reader import LogReader

# Higher is worse except grip force. Benign metrics stay in the competition.
HEALTH_DIRECTIONS = {
    ("scene_camera", "calibration_residual"): 1,
    ("wrist_camera", "calibration_residual"): 1,
    ("gripper", "grip_force"): -1,
    ("gripper", "cycle_count"): 1,
    ("arm", "joint_torque_rms"): 1,
    ("controller", "cpu_load"): 1,
}


def fixed_prior(export: Path) -> BaselineAnswer:
    return BaselineAnswer(
        "fixed_prior_fallback",
        ranking(trivial_no_data_ranker(visible_fleet(export), export_catalogue(export))),
        metadata={"fallback": True, "model_call": False},
    )


def worst_health(export: Path) -> BaselineAnswer:
    """Median over days; bad-direction robust z within metric; maximum per component."""
    with LogReader(export) as reader:
        rows = reader.query("SELECT robot_id, component, metric, value FROM health").to_pylist()
    readings: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (row["component"], row["metric"]) not in HEALTH_DIRECTIONS:
            raise ValueError(f"unknown health metric: {row['component']}.{row['metric']}")
        value = float(row["value"])
        if np.isfinite(value):
            readings[row["robot_id"], row["component"], row["metric"]].append(value)
    medians = {key: float(np.median(values)) for key, values in readings.items()}
    populations: dict[str, list[float]] = defaultdict(list)
    for (_, _, metric), value in medians.items():
        populations[metric].append(value)
    normalizers = {}
    for metric, values in populations.items():
        center = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(np.array(values) - center)))
        if scale == 0:
            scale = float(np.std(values)) or 1.0
        normalizers[metric] = center, scale
    scores: dict[tuple[str, str], float] = {}
    for (robot, component, metric), value in medians.items():
        center, scale = normalizers[metric]
        z = HEALTH_DIRECTIONS[component, metric] * (value - center) / scale
        scores[robot, component] = max(scores.get((robot, component), -float("inf")), z)
    actions = component_actions(export)
    # Unobserved components rank after every measured component, then by neutral IDs.
    actions.sort(
        key=lambda a: (
            -scores.get((a.target.scope, str(a.target.component)), -float("inf")),
            a.target.scope,
            str(a.target.component),
        )
    )
    return BaselineAnswer(
        "worst_health",
        ranking(actions),
        metadata={
            "scores": [
                {"robot": robot, "component": component, "severity": score}
                for (robot, component), score in sorted(scores.items())
            ],
            "missing": [
                a.model_dump(mode="json")
                for a in actions
                if (a.target.scope, str(a.target.component)) not in scores
            ],
        },
    )
