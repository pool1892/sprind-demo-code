"""Forecast scores and first-call opportunity metrics, never claims of rescue."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from fleetlab.live.data import BoolArray, Dataset, IntArray
from fleetlab.live.features import FEATURE_NAMES, HZ, FloatArray


def auc(y: BoolArray, scores: FloatArray) -> float | None:
    """Rank AUROC with tied scores receiving half credit."""
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float(
        (ranks[y[order]].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def forecast_metrics(y: BoolArray, p: FloatArray) -> dict[str, Any]:
    if len(y) == 0:
        return {"rows": 0, "positives": 0, "brier": None, "auroc": None, "reliability": []}
    reliability = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & ((p < hi) if hi < 0.999 else (p <= 1))
        reliability.append(
            {
                "low": round(float(lo), 1),
                "high": round(hi, 1),
                "rows": int(mask.sum()),
                "predicted": float(p[mask].mean()) if mask.any() else None,
                "observed": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": auc(y, p),
        "reliability": reliability,
    }


def first_calls(data: Dataset, fired: BoolArray) -> IntArray:
    """The score at step t calls at t+1, only if the unassisted episode is still active."""
    calls = np.full(len(data.episodes), -1, np.int64)
    eligible = fired & (data.steps + 1 < data.terminal)
    # Preserve the first call per episode, regardless of a caller's row ordering.
    sentinel = np.iinfo(np.int64).max
    calls[:] = sentinel
    np.minimum.at(calls, data.episode[eligible], data.steps[eligible] + 1)
    calls[calls == sentinel] = -1
    return calls


def alarm_metrics(data: Dataset, fired: BoolArray, horizon: int) -> dict[str, Any]:
    calls = first_calls(data, fired)
    failed = np.asarray([e["failure"] for e in data.episodes], np.bool_)
    terminal = np.asarray([e["terminal"] for e in data.episodes], np.int64)
    caught = failed & (calls >= 0)
    false = ~failed & (calls >= 0)
    lead = (terminal[caught] - calls[caught]) / HZ
    temporal_false = (calls >= 0) & (~failed | ((terminal - (calls - 1)) > horizon))
    return {
        "episodes": len(calls),
        "failures": int(failed.sum()),
        "successes": int((~failed).sum()),
        "calls": int((calls >= 0).sum()),
        "caught": int(caught.sum()),
        "failure_recall": float(caught.sum() / failed.sum()) if failed.any() else None,
        "false_alarms": int(false.sum()),
        "false_alarms_per_episode": float(false.mean()),
        "successful_episode_alarm_rate": float(false.sum() / (~failed).sum())
        if (~failed).any()
        else None,
        "outside_horizon_alarms_per_episode": float(temporal_false.mean()),
        "lead_seconds_median": float(np.median(lead)) if len(lead) else None,
        "lead_seconds_p10": float(np.quantile(lead, 0.1)) if len(lead) else None,
        "lead_seconds_mean": float(lead.mean()) if len(lead) else None,
        "failure_lead_seconds_per_failure": float(lead.sum() / failed.sum())
        if failed.any()
        else None,
        "caught_with_0_5s_to_decide": int((lead > 0.5).sum()),
        "caught_with_2s_to_decide": int((lead > 2).sum()),
        "first_call_steps": calls.tolist(),
    }


def select_threshold(
    data: Dataset, p: FloatArray, horizon: int, false_alarm_cap: float = 0.05
) -> tuple[float, dict[str, Any]]:
    """Calibration only: most failure lead per failure subject to successful-call cap."""
    if not any(not e["failure"] for e in data.episodes):
        raise ValueError("threshold selection needs successful calibration episodes")
    choices = []
    for threshold in np.linspace(0.01, 1, 100):
        metrics = alarm_metrics(data, p >= threshold, horizon)
        if metrics["successful_episode_alarm_rate"] <= false_alarm_cap:
            # Failure lead integrates how many failures were caught and how early.
            choices.append(
                (
                    metrics["failure_lead_seconds_per_failure"] or 0.0,
                    -metrics["false_alarms"],
                    float(threshold),
                    metrics,
                )
            )
    if not choices:
        raise ValueError("no threshold meets false alarm cap")
    _, _, threshold, chosen = max(choices, key=lambda item: item[:3])
    return threshold, chosen


def baseline_fired(data: Dataset, kind: str, parameter: int) -> BoolArray:
    if kind == "timeout":
        return data.steps >= parameter
    if kind == "stall":
        # Same movement predicate as the features: <1 mm on every consecutive step.
        column = FEATURE_NAMES.index("stall_steps")
        return (data.steps >= 40) & (data.x[:, column] * HZ >= parameter)
    raise ValueError(f"unknown baseline {kind}")


def select_baseline(data: Dataset, kind: str, horizon: int) -> tuple[int, dict[str, Any]]:
    values = range(1, 281) if kind == "stall" else range(0, 281)
    choices = []
    for parameter in values:
        metrics = alarm_metrics(data, baseline_fired(data, kind, parameter), horizon)
        if metrics["successful_episode_alarm_rate"] <= 0.05:
            choices.append(
                (
                    metrics["failure_lead_seconds_per_failure"] or 0.0,
                    -metrics["false_alarms"],
                    parameter,
                    metrics,
                )
            )
    _, _, parameter, metrics = max(choices, key=lambda item: item[:3])
    return parameter, metrics


def evaluate(
    data: Dataset, p: FloatArray, threshold: float, horizon: int, *, fired: BoolArray | None = None
) -> dict[str, Any]:
    y = data.labels(horizon)
    result = forecast_metrics(y, p)
    fired = p >= threshold if fired is None else fired
    result["alarms"] = alarm_metrics(data, fired, horizon)
    result["robot_bootstrap_95"] = cluster_intervals(data, p, threshold, horizon, fired=fired)
    # Absolute-time strata expose discrimination hidden by terminal-time predictability.
    result["time_strata"] = [
        {
            "first_step": start,
            "last_step_exclusive": start + 40,
            **forecast_metrics(
                y[(data.steps >= start) & (data.steps < start + 40)],
                p[(data.steps >= start) & (data.steps < start + 40)],
            ),
        }
        for start in range(0, 280, 40)
    ]
    result["per_robot"] = []
    for world, robot in sorted({(e["world"], e["robot"]) for e in data.episodes}):
        mask = np.asarray(
            [e["world"] == world and e["robot"] == robot for e in data.episodes], np.bool_
        )[data.episode]
        subset = data.subset(mask)
        result["per_robot"].append(
            {
                "world": world,
                "robot": robot,
                **forecast_metrics(subset.labels(horizon), p[mask]),
                "alarms": alarm_metrics(subset, fired[mask], horizon),
            }
        )
    return result


def cluster_intervals(
    data: Dataset, p: FloatArray, threshold: float, horizon: int, *, fired: BoolArray | None = None
) -> dict[str, Any]:
    """500 deterministic robot-cluster resamples, stratified by development world.

    Steps and episodes of a robot stay together. These intervals describe the
    few sampled robots; they do not capture unobserved worlds or fault classes.
    """
    keys = sorted({(e["world"], e["robot"]) for e in data.episodes})
    calls = first_calls(data, p >= threshold if fired is None else fired)
    row_loss = (p - data.labels(horizon)) ** 2
    totals = []
    for world, robot in keys:
        episodes = np.asarray(
            [e["world"] == world and e["robot"] == robot for e in data.episodes], np.bool_
        )
        rows = episodes[data.episode]
        failed = np.asarray([e["failure"] for e in data.episodes], np.bool_)
        caught = episodes & failed & (calls >= 0)
        terminal = np.asarray([e["terminal"] for e in data.episodes], np.int64)
        totals.append(
            [
                row_loss[rows].sum(),
                rows.sum(),
                (failed & episodes).sum(),
                caught.sum(),
                ((~failed) & episodes & (calls >= 0)).sum(),
                episodes.sum(),
                ((terminal[caught] - calls[caught]) / HZ).sum(),
            ]
        )
    values = np.asarray(totals, np.float64)
    rng = np.random.Generator(np.random.PCG64(61401))
    samples = []
    for _ in range(500):
        selected: list[int] = []
        for world in sorted({w for w, _ in keys}):
            group = [i for i, (w, _) in enumerate(keys) if w == world]
            selected.extend(rng.choice(group, len(group), replace=True))
        total = values[selected].sum(axis=0)
        samples.append(
            [
                total[0] / total[1],
                total[3] / total[2] if total[2] else np.nan,
                total[4] / total[5],
                total[6] / total[2] if total[2] else np.nan,
            ]
        )
    sample = np.asarray(samples)
    intervals = {}
    for i, name in enumerate(
        ("brier", "failure_recall", "false_alarms_per_episode", "failure_lead_seconds_per_failure")
    ):
        valid = sample[:, i][np.isfinite(sample[:, i])]
        intervals[name] = np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None
    return {"clusters": len(keys), "resamples": 500, "seed": 61401, "intervals": intervals}
