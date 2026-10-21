"""Independent worlds' mean-share floor, using only oracle population histograms."""

from collections import Counter
from fractions import Fraction
from typing import Any

from fleetlab.scoring.comparisons import exact_problem
from fleetlab.scoring.models import BudgetScore


def percentile(hist: Counter[int]) -> Fraction:
    total = sum(hist.values())
    index, remainder = divmod((total - 1) * 95, 100)

    def at(position: int) -> int:
        cumulative = 0
        for value, weight in sorted(hist.items()):
            cumulative += weight
            if position < cumulative:
                return value
        raise ValueError("empty population")

    return Fraction(at(index)) + Fraction(remainder, 100) * (
        at(min(index + 1, total - 1)) - at(index)
    )


def pooled_floor(budgets: dict[str, BudgetScore]) -> dict[str, Any]:
    """Integer-weight convolution with a rigorous share discretisation bound.

    Each world's share is rounded down to 1/4000 before taking the mean.
    Thus every mean moves down by less than 0.00025, as does its quantile.
    Missing sets are placed at the oracle's net extremes, never dropped.
    """
    missing = []
    for world, at in budgets.items():
        exact = at.exact_chance or {}
        if (
            exact_problem(exact) is not None
            or at.achievable.gain is None
            or at.achievable.gain.net <= 0
            or (
                exact.get("unvalued_sets", 0)
                and (exact.get("net_low") is None or exact.get("net_high") is None)
            )
        ):
            missing.append(world)
    if missing:
        return {
            "p95_low": None,
            "p95_high": None,
            "missing_worlds": missing,
            "commands": {world: budgets[world].chance_command for world in missing},
            "missing": "oracle exact histogram or achievable denominator unavailable",
        }
    endpoints = []
    for bound in ("net_low", "net_high"):
        distribution: Counter[int] = Counter({0: 1})
        for at in budgets.values():
            exact = at.exact_chance or {}
            assert exact is not None and at.achievable.gain is not None
            histogram = Counter({int(net): int(weight) for net, weight in exact["histogram"]})
            unknown = int(exact.get("unvalued_sets", 0))
            if unknown:
                histogram[int(exact[bound])] += unknown
            bins: Counter[int] = Counter()
            for net, weight in histogram.items():
                bins[net * 4000 // at.achievable.gain.net] += weight
            convolved: Counter[int] = Counter()
            for a, wa in distribution.items():
                for b, wb in bins.items():
                    convolved[a + b] += wa * wb
            distribution = convolved
        endpoints.append(float(percentile(distribution) / (4000 * len(budgets))))
    return {
        "p95_low": endpoints[0],
        "p95_high": endpoints[1] + 0.00025,
        "percentile_error_share": 0.00025,
        "missing_worlds": [],
        "basis": "oracle exact histograms, independent worlds; integer-weight convolution",
    }
