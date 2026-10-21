"""The fixed opponents and the oracle-owned chance floor (design changes 5-7)."""

from collections import Counter
from fractions import Fraction
from typing import Any

from fleetlab.scoring.models import BudgetScore, FixSetValue, Metric

# Section 6, baselines 2-5. Caller labels cannot narrow this set. Bare model
# labels mean repeat 1; an explicit ' repeat 2' is displayed but never scored.
BASELINE_METHODS = {
    "no-data ranker": ("no_data",),
    "worst health metric": ("worst_health",),
    "dramatic videos": ("dramatic_videos",),
    "classical statistics": ("classical-binomial", "classical-binomial-v2"),
}
REQUIRED_BASELINES = tuple(BASELINE_METHODS)
MODEL_BASELINES = frozenset({REQUIRED_BASELINES[0], REQUIRED_BASELINES[2]})
EXPERIMENTAL_BASELINE = "classical statistics with experiments"
BASELINE_METHODS[EXPERIMENTAL_BASELINE] = BASELINE_METHODS["classical statistics"]


def required_baselines(condition: str) -> tuple[str, ...]:
    return (
        (*REQUIRED_BASELINES, EXPERIMENTAL_BASELINE)
        if condition == "with experiments"
        else REQUIRED_BASELINES
    )


def required_values(at: BudgetScore) -> tuple[dict[str, FixSetValue], list[str]]:
    if at.valuation_missing or at.analysis_binding_missing:
        return {}, [at.valuation_missing or at.analysis_binding_missing or "comparison unavailable"]
    values = {}
    absent = []
    sources: set[str] = set()
    for name in required_baselines(at.condition):
        label = name + " repeat 1" if name + " repeat 1" in at.baselines else name
        value = at.baselines.get(label)
        provenance = at.baseline_sources.get(label, {})
        record = provenance.get("record") or {}
        source = provenance.get("source")
        # The miniature chain's deterministic prior is not the required model run.
        if provenance.get("lost_call"):
            absent.append(name + " (both model call starts lost)")
        elif value is None or value.missing == "required baseline absent":
            absent.append(name + " (absent)")
        elif record.get("method") not in BASELINE_METHODS[name] or not source:
            absent.append(name + " (source method record absent or mismatched)")
        elif name in MODEL_BASELINES and record.get("repeat") != 1:
            absent.append(name + " (source repeat 1 record absent or mismatched)")
        elif source in sources:
            absent.append(name + " (source shared by required baselines)")
        elif not provenance.get("binding") or record.get("world_id") != at.world_id:
            absent.append(name + " (producer world/output binding absent or mismatched)")
        elif value.points is None:
            absent.append(name + " (unvalued)")
        else:
            values[name] = value
            sources.add(source)
    return values, absent


def exact_problem(exact: dict[str, Any] | None) -> str | None:
    """A population statistic is usable only with complete set accounting."""
    if not exact:
        return "oracle exact chance absent"
    histogram = exact.get("histogram")
    sets, unknown = exact.get("sets"), exact.get("unvalued_sets")
    if (
        not isinstance(histogram, list)
        or type(sets) is not int
        or sets <= 0
        or type(unknown) is not int
        or unknown < 0
    ):
        return "oracle exact chance lacks valid histogram/set accounting"
    nets: set[int] = set()
    counted = unknown
    for row in histogram:
        if (
            not isinstance(row, (list, tuple))
            or len(row) != 2
            or type(row[0]) is not int
            or type(row[1]) is not int
            or row[1] <= 0
            or row[0] in nets
        ):
            return "oracle exact chance histogram is invalid"
        nets.add(row[0])
        counted += row[1]
    if counted != sets:
        return "oracle exact histogram sets plus unvalued sets differ from total sets"
    if unknown and (
        type(exact.get("net_low")) is not int
        or type(exact.get("net_high")) is not int
        or exact["net_low"] > exact["net_high"]
    ):
        return "oracle exact chance lacks unvalued-set net bounds"
    return None


def chance_points(at: BudgetScore, statistic: str = "p95") -> Metric:
    exact = at.exact_chance
    if problem := exact_problem(exact):
        return Metric.absent(
            problem + "; " + (at.chance_missing or "produce it with: " + at.chance_command)
        )
    assert exact is not None
    key = statistic + "_points"
    value = exact.get(key)
    if value is None:
        bounds = exact.get("bounds") or {}
        interval = bounds.get(key)
        value = interval[1] if interval is not None else None
    return Metric.of(value, at.chance_missing or exact.get("missing") or "exact statistic absent")


def chance_share(at: BudgetScore) -> Metric:
    value = chance_points(at)
    if value.value is None:
        return value
    if at.achievable.points is None or at.achievable.points == 0:
        return Metric.absent("achievable gain missing or zero")
    return Metric.of(value.value / at.achievable.points)


def chance_net(at: BudgetScore) -> Fraction | None:
    """Unrounded linear p95, taking unvalued sets at their upper net bound."""
    exact = at.exact_chance
    if exact_problem(exact):
        return None
    assert exact is not None
    histogram = Counter({net: weight for net, weight in exact["histogram"]})
    if exact["unvalued_sets"]:
        histogram[exact["net_high"]] += exact["unvalued_sets"]
    index, remainder = divmod((exact["sets"] - 1) * 95, 100)

    def value(position: int) -> int:
        cumulative = 0
        for net, weight in sorted(histogram.items()):
            cumulative += weight
            if position < cumulative:
                return int(net)
        raise ValueError("order statistic outside exact population")

    lo, hi = value(index), value(min(index + 1, exact["sets"] - 1))
    return Fraction(lo) + Fraction(remainder, 100) * (hi - lo)
