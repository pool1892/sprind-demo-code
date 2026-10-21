"""Local release contrasts: no effect can be borrowed from remote prehistory."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.special import expit, logit

from fleetlab.baselines.classical.data import Row, VisibleData
from fleetlab.baselines.classical.regression import Array


def release_contrast(
    data: VisibleData,
    onset: int,
    reference: list[Row],
    weight: Array,
    covered: Array,
    *,
    draws: int,
    seed: int,
) -> tuple[Array, dict[str, Any]]:
    """Local weighted level contrast, with unpenalized robot and task effects.

    Shared task effects permit a rotating task schedule. This does
    not require every robot/task to appear on both sides. A release exactly
    confounded with robot/task composition remains unidentified. Bernoulli
    noise is pooled within task/phase, with Jeffreys half-count smoothing.
    """
    shares: dict[str, float] = defaultdict(float)
    for row, share, affected in zip(reference, weight, covered, strict=True):
        if affected:
            shares[row["robot_id"]] += float(share)
    rows = [
        r for r in data.episodes if r["robot_id"] in shares and onset - 2 <= r["day"] < onset + 2
    ]
    robots = sorted(shares)
    tasks = sorted({r["task_text"] for r in rows})
    post = np.asarray([r["day"] >= onset for r in rows], dtype=float)
    outcome = np.asarray([r["outcome"] == "success" for r in rows], dtype=float)
    # The constant local step is shared. Nuisance contrasts are unpenalized,
    # so shrinkage cannot assign a task or robot difference to the release.
    nuisance = np.asarray(
        [
            [float(r["robot_id"] == robot) for robot in robots]
            + [float(r["task_text"] == task) for task in tasks]
            for r in rows
        ]
    )
    counts = {robot: sum(r["robot_id"] == robot for r in rows) for robot in robots}
    row_weight = np.asarray([shares[r["robot_id"]] / counts[r["robot_id"]] for r in rows])
    complete = bool(rows) and all(counts.values())
    residual = np.zeros(len(rows))
    if complete:
        root_weight = np.sqrt(row_weight)
        coefficient = np.linalg.lstsq(
            nuisance * root_weight[:, None], post * root_weight, rcond=None
        )[0]
        residual = post - nuisance @ coefficient
    information = float(row_weight @ residual**2)
    complete = complete and information > 1e-12 * sum(shares.values())
    point, variance = 0.0, 0.0
    if complete:
        influence = row_weight * residual / information
        scale = 100 * sum(shares.values())
        point = -float(influence @ outcome) * scale
        cells: dict[tuple[str, bool], list[float]] = defaultdict(list)
        for row, y in zip(rows, outcome, strict=True):
            cells[row["task_text"], row["day"] >= onset].append(float(y))
        rates = {key: (sum(values) + 0.5) / (len(values) + 1) for key, values in cells.items()}
        noise = np.asarray([rates[r["task_text"], r["day"] >= onset] for r in rows])
        variance = float(influence**2 @ (noise * (1 - noise))) * scale**2
    rng = np.random.default_rng(seed)
    gains = np.r_[point, rng.normal(point, np.sqrt(variance), size=draws)]
    days = sorted({r["day"] for r in data.episodes})
    return gains, dict(
        rule="local two-day pre/post pooled contrast, unpenalized robot/task adjustment",
        pre_days=[d for d in days if onset - 2 <= d < onset],
        post_days=[d for d in days if onset <= d < onset + 2],
        complete=complete,
        information=information,
        gain_points=point,
        sampling_variance=variance,
        low=float(np.quantile(gains[1:], 0.05)),
        high=float(np.quantile(gains[1:], 0.95)),
        limitation=(
            "shared task effects and a constant local step assumed; "
            "within-window changes remain confounded"
        ),
    )


def shifted(probability: Array, offset: Array) -> Array:
    return np.asarray(expit(logit(np.clip(probability, 1e-12, 1 - 1e-12)) + offset))


def match_gain(probability: Array, weight: Array, covered: Array, gains: Array) -> Array:
    """Common success-log-odds shift, bounded and calibrated for every draw."""
    lower, upper = np.full(len(gains), -30.0), np.full(len(gains), 30.0)
    for _ in range(60):
        middle = (lower + upper) / 2
        effect = weight @ (shifted(probability, covered[:, None] * middle) - probability) * 100
        lower = np.where(effect < gains, middle, lower)
        upper = np.where(effect >= gains, middle, upper)
    return np.asarray(covered[:, None] * ((lower + upper) / 2), dtype=float)
