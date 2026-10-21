"""Claim 2: the agent's expected gains against the true gains of its ranked actions,
and the statistics it uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fleetlab.contracts.actions import Action, Budget, FixSet
from fleetlab.contracts.faults import TruthTable
from fleetlab.contracts.oracle import CurrentValues, OracleTable, ReplayNeed
from fleetlab.contracts.tools import RankedFixSet
from fleetlab.scoring.models import TOP_K, ActionValue, Metric, RankingScore
from fleetlab.scoring.valuation import points, value_or_missing


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, ties given the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        mean = (start + end) / 2 + 1
        for j in range(start, end + 1):
            ranks[order[j]] = mean
        start = end + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson's correlation; None when either side has no variance or fewer than two points."""
    if len(x) != len(y):
        raise ValueError("two series of the same length")
    if len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rank correlation: Pearson's on the average ranks."""
    return pearson(average_ranks(x), average_ranks(y))


def slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    """The least-squares slope of y on x; None when x has no variance."""
    if len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sxx


def single(action: Action) -> FixSet:
    """The fix set of one action."""
    return FixSet(actions=[action], budget=Budget(kind="count", amount=1))


def score_ranking(
    fix_set: RankedFixSet | None,
    oracle: OracleTable,
    truth: TruthTable,
    current: CurrentValues | None,
) -> tuple[RankingScore, list[tuple[str, list[ReplayNeed]]]]:
    """Each ranked action's true gain, and the correlation, calibration and coverage.

    Returns the score and the replays each missing valuation waits for. The
    shares of the achievable gain by the top k are filled in by claim 3, which
    values the same cuts.
    """
    if fix_set is None:
        why = "no ranked fix set submitted"
        absent = Metric.absent(why)
        return RankingScore(
            missing=why,
            spearman=absent,
            mae_points=absent,
            slope=absent,
            coverage=absent,
            top_k_share={str(k): absent for k in TOP_K},
        ), []
    episodes = len(oracle.identities)
    values: list[ActionValue] = []
    needs: list[tuple[str, list[ReplayNeed]]] = []
    for rank, ranked in enumerate(fix_set.actions, start=1):
        row = ActionValue(
            rank=rank,
            action=ranked.action,
            expected_gain=ranked.expected_gain,
            low=ranked.uncertainty.low,
            high=ranked.uncertainty.high,
        )
        if not ranked.action.has_replay:
            row.missing = f"{ranked.action.verb.value} has no replay: scored under claim 5"
        elif episodes == 0:
            row.missing = "the oracle holds no scoring episode"
        else:
            valued = value_or_missing(single(ranked.action), oracle, truth, current)
            if valued.gain is None:
                row.missing = valued.missing
                needs.append((f"claim 2: ranked action {rank}", valued.needs))
            else:
                true_points = points(valued.gain.net, episodes)
                row.true = valued.gain
                row.true_points = true_points
                row.covered = row.low <= true_points <= row.high
        values.append(row)

    done = [v for v in values if v.true_points is not None]
    expected = [v.expected_gain for v in done]
    true = [v.true_points for v in done if v.true_points is not None]
    unvalued = len(values) - len(done)
    why_few = f"{len(done)} actions valued ({unvalued} not)"
    if len(done) < 2:
        rho = mae = beta = coverage = Metric.absent(f"fewer than two actions valued: {why_few}")
        if done:
            mae = Metric.of(abs(expected[0] - true[0]))
            coverage = Metric.of(float(bool(done[0].covered)))
    else:
        rho_value = spearman(expected, true)
        rho = (
            Metric.of(rho_value)
            if rho_value is not None
            else Metric.absent("the expected or the true gains are all equal")
        )
        mae = Metric.of(sum(abs(e - t) for e, t in zip(expected, true, strict=True)) / len(done))
        beta_value = slope(expected, true)
        beta = (
            Metric.of(beta_value)
            if beta_value is not None
            else Metric.absent("the expected gains are all equal")
        )
        coverage = Metric.of(sum(bool(v.covered) for v in done) / len(done))
    return (
        RankingScore(
            actions=values,
            valued=len(done),
            spearman=rho,
            mae_points=mae,
            slope=beta,
            coverage=coverage,
            complete=all(v.true_points is not None or not v.action.has_replay for v in values),
        ),
        needs,
    )
