"""Visible label checks and symptom signatures; no fault-registry knowledge."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import fisher_exact

from fleetlab.baselines.classical.data import Row, VisibleData
from fleetlab.baselines.classical.regression import Array


def stratified_contrast(
    rows: list[Row], exposed: list[bool], values: list[float]
) -> tuple[float, float]:
    """CMH score and task-standardized exposed-minus-reference rate."""
    cells: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
    for row, group, value in zip(rows, exposed, values, strict=True):
        cells[row["task_text"]][int(group)].append(value)
    score = variance = 0.0
    differences = []
    for control, treated in cells.values():
        n0, n1 = len(control), len(treated)
        n = n0 + n1
        successes = sum(control) + sum(treated)
        if not n0 or not n1:
            continue
        score += sum(treated) - n1 * successes / n
        variance += n0 * n1 * successes * (n - successes) / (n * n * max(1, n - 1))
        differences.append(float(np.mean(treated) - np.mean(control)))
    return (
        score / math.sqrt(variance) if variance > 0 else 0.0,
        float(np.mean(differences)) if differences else 0.0,
    )


def outcome_check(data: VisibleData) -> tuple[Array, list[dict[str, Any]]]:
    """Detect concentrations of short logged failures before outcome regression.

    A task horizon needs two robots and at least half the failure-bearing
    robots agreeing; each votes only when 60% of its failures terminate there.
    Robots without failures supply no endpoint information. Require a familywise
    1% excess of short failures at a robot,
    at least five examples and a majority of its failures. Replace suspect labels
    by the peer probability of success among short episodes, with Jeffreys halves.
    This is a label-error sensitivity model, not knowledge of the true outcome.
    """
    rows = data.episodes
    y = np.asarray([r["outcome"] == "success" for r in rows], dtype=float)
    horizons: dict[str, int] = {}
    robots = sorted({r["robot_id"] for r in rows})
    for task in sorted({r["task_text"] for r in rows}):
        votes: Counter[int] = Counter()
        eligible = 0
        for robot in robots:
            lengths = Counter(
                int(r["steps"])
                for r in rows
                if r["task_text"] == task
                and r["robot_id"] == robot
                and r["outcome"] == "failure"
                and r.get("steps")
            )
            if lengths:
                eligible += 1
                horizon, count = max(lengths.items(), key=lambda p: (p[1], p[0]))
                if count / sum(lengths.values()) >= 0.6:
                    votes[horizon] += 1
        if votes:
            horizon, count = max(votes.items(), key=lambda p: (p[1], p[0]))
            if count >= max(2, eligible / 2):
                horizons[task] = horizon
    short = np.asarray(
        [
            bool(
                r.get("steps")
                and r["task_text"] in horizons
                and r["steps"] < 0.95 * horizons[r["task_text"]]
            )
            for r in rows
        ]
    )
    failed = np.asarray([r["outcome"] == "failure" for r in rows])
    records = []
    robots = sorted({r["robot_id"] for r in rows})
    for robot in robots:
        own = np.asarray([r["robot_id"] == robot for r in rows])
        a, b = int(np.sum(own & failed & short)), int(np.sum(own & failed & ~short))
        c, d = int(np.sum(~own & failed & short)), int(np.sum(~own & failed & ~short))
        p = min(
            1.0, float(fisher_exact([[a, b], [c, d]], alternative="greater").pvalue) * len(robots)
        )
        if a < 5 or a <= b or p > 0.01:
            continue
        ids = []
        for i in np.flatnonzero(own & failed & short):
            peers = (
                np.asarray(
                    [
                        r["task_text"] == rows[i]["task_text"] and r["outcome"] != "handed_over"
                        for r in rows
                    ]
                )
                & ~own
                & short
            )
            probability = (float(y[peers].sum()) + 0.5) / (int(peers.sum()) + 1)
            y[i] = probability
            ids.append(rows[i]["episode_id"])
        records.append(
            dict(
                scope=robot,
                familywise_p=p,
                short_failures=a,
                other_failures=b,
                episode_ids=ids,
                onset=min(rows[i]["day"] for i in np.flatnonzero(own & failed & short)),
                interpretation=(
                    "possible outcome-recording fault; duration resembles peer successes"
                ),
            )
        )
    return y, records


@dataclass(frozen=True)
class Symptom:
    scope: str
    kind: str
    excess: float
    p: float
    robots: int
    episodes: tuple[str, ...]


def event_signatures(data: VisibleData) -> list[Symptom]:
    kinds: dict[str, set[str]] = defaultdict(set)
    for event in data.events:
        kinds[event["episode_id"]].add(event["kind"])
    # Commands to close/open a gripper are not success evidence. All other
    # event kinds are tested, including unknown detector kinds.
    tested = sorted({e["kind"] for e in data.events} - {"grasp", "release"})
    scopes = [s.id for s in data.fleet.sites] + [r.id for r in data.fleet.robots()]
    results = []
    for scope in scopes:
        exposed = [scope in (r["site_id"], r["robot_id"]) for r in data.episodes]
        for kind in tested:
            values = [float(kind in kinds[r["episode_id"]]) for r in data.episodes]
            z, excess = stratified_contrast(data.episodes, exposed, values)
            p = min(1.0, math.erfc(max(z, 0) / math.sqrt(2)) * len(scopes) * len(tested))
            ids = tuple(
                r["episode_id"]
                for r, e, v in zip(data.episodes, exposed, values, strict=True)
                if e and v
            )
            robots = {r["robot_id"] for r in data.episodes if r["episode_id"] in ids}
            if excess > 0:
                results.append(Symptom(scope, kind, excess, p, len(robots), ids))
    return results
