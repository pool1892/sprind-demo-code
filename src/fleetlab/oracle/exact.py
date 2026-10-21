"""Exact random-fix distributions when the existing table suffices.

Count budgets are uniform over distinct complete catalogue entries. Entries with the
same per-episode resolution form a class: choosing m of c entries contributes C(c,m),
and resolving that class once has the same effect as resolving it m times. This counts
every set, including no-change entries, without enumerating millions of sets.

Missing replay values retain their multiplicity and receive worst/best bounds.
Resolution errors prevent valuation. The state limit retains uncounted mass in bounds.
Cost budgets use greedy permutations,
not uniform feasible sets; only small catalogues can be enumerated here.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from fleetlab.contracts.actions import (
    COSTS,
    Action,
    Budget,
    CostTable,
    Resolution,
    cut,
)
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.oracle import Gain
from fleetlab.oracle.valuation import FRESH, DrawValuer, complete_actions

Signature = tuple[tuple[int, tuple[str, ...], tuple[str, ...]], ...]

EXACT_STATE_LIMIT = 100_000
EXACT_PERMUTATION_LIMIT = 40_320


class ExactChance(BaseModel):
    """Population statistics beside the seeded draws, or an explicit unavailable reason."""

    model_config = ConfigDict(extra="forbid")
    method: str
    sets: int | None = None  # count sets, or permutations for a cost budget
    mean_points: float | None = None
    p5_points: float | None = None
    p95_points: float | None = None
    histogram: list[tuple[int, int]] = []
    unvalued_sets: int = 0
    net_low: int | None = None
    net_high: int | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    missing: str | None = None
    counting_limited: str | None = None
    uncomputed_sets: int = 0  # included in unvalued_sets when enumeration is bounded


ExactValuer = DrawValuer  # identical physical replay lookup, including added changes


def _signature(effects: dict[int, Resolution]) -> Signature:
    return tuple(
        (i, tuple(sorted(e.removed)), tuple(sorted(f.model_dump_json() for f in e.added)))
        for i, e in sorted(effects.items())
    )


class _ClassValuer:
    """Cache by each episode group's projection of the selected resolution classes.

    This is the same union and actual lookup as DrawValuer. Disjoint scopes can
    share their partial valuations; a global set is still unknown if any part is.
    """

    def __init__(self, valuer: DrawValuer, groups: list[list[Action]]) -> None:
        self.valuer = valuer
        self.effects: list[dict[int, Resolution]] = []
        incidence: dict[int, int] = defaultdict(int)
        for bit, group in enumerate(groups):
            effects = valuer.effects(group[0])
            assert isinstance(effects, dict)
            self.effects.append(effects)
            for index in effects:
                incidence[index] |= 1 << bit
        self.members: dict[int, list[int]] = defaultdict(list)
        for index, mask in incidence.items():
            self.members[mask].append(index)
        self.cache: dict[tuple[int, int], int | str] = {}

    def value(self, mask: int) -> int | str:
        net = 0
        for incidence, members in self.members.items():
            projection = mask & incidence
            if not projection:
                continue
            key = (incidence, projection)
            if key not in self.cache:
                subtotal = 0
                bits = [i for i in range(len(self.effects)) if projection & (1 << i)]
                for index in members:
                    identity = self.valuer.oracle.identities[index]
                    gone: set[str] = set()
                    added: dict[str, FaultSpec] = {}
                    for bit in bits:
                        change = self.effects[bit][index]
                        gone.update(change.removed)
                        for fault in change.added:
                            added.setdefault(fault.fault_id, fault)
                    remaining = [f for f in identity.faults if f.fault_id not in gone]
                    before = self.valuer.oracle.success(identity.episode_id, identity.faults)
                    after = self.valuer.oracle.success(
                        identity.episode_id, [*remaining, *added.values()]
                    )
                    if before is None or after is None:
                        self.cache[key] = FRESH
                        break
                    subtotal += int(after) - int(before)
                else:
                    self.cache[key] = subtotal
            value = self.cache[key]
            if isinstance(value, str):
                return value
            net += value
        return net


def _percentile(histogram: Counter[int], total: int, percentile: int) -> float:
    # Numpy's linear percentile of the expanded integer-weight population, without
    # float rounding of its (potentially enormous) order-statistic positions.
    numerator = (total - 1) * percentile
    index, remainder = divmod(numerator, 100)

    def at(position: int) -> int:
        cumulative = 0
        for net, weight in sorted(histogram.items()):
            cumulative += weight
            if position < cumulative:
                return net
        raise ValueError("order statistic outside exact population")

    lo, hi = at(index), at(min(index + 1, total - 1))
    return lo + (hi - lo) * remainder / 100


def _statistics(histogram: Counter[int], episodes: int) -> dict[str, float]:
    total = sum(histogram.values())
    mean = 100 * sum(net * w for net, w in histogram.items()) / total / episodes
    return dict(
        mean_points=mean,
        p5_points=100 * _percentile(histogram, total, 5) / episodes,
        p95_points=100 * _percentile(histogram, total, 95) / episodes,
    )


def _finish(result: ExactChance, histogram: Counter[int], valuer: DrawValuer) -> ExactChance:
    result.histogram = sorted(histogram.items())
    result.sets = sum(histogram.values()) + result.unvalued_sets
    n = len(valuer.oracle.identities)
    baseline = [valuer.oracle.success(i.episode_id, i.faults) for i in valuer.oracle.identities]
    result.net_low = -sum(s is not False for s in baseline)
    result.net_high = sum(s is not True for s in baseline)
    if result.unvalued_sets:
        low, high = histogram.copy(), histogram.copy()
        low[result.net_low] += result.unvalued_sets
        high[result.net_high] += result.unvalued_sets
        lower, upper = _statistics(low, n), _statistics(high, n)
        bounds = {}
        for key in lower:
            if lower[key] == upper[key]:
                setattr(result, key, round(lower[key], 3))
            else:
                bounds[key] = (
                    math.floor(lower[key] * 1000) / 1000,
                    math.ceil(upper[key] * 1000) / 1000,
                )
        result.bounds = bounds or None
    else:
        for key, value in _statistics(histogram, n).items():
            setattr(result, key, round(value, 3))
    return result


def exact_chance(
    valuer: DrawValuer, entries: Sequence[Action], budget: Budget, costs: CostTable = COSTS
) -> ExactChance:
    """Every count set (or small cost permutation), valued from the available table."""
    exact = valuer
    pool = list({a.model_dump_json(): a for a in complete_actions(entries)}.values())
    result = ExactChance(
        method="uniform count sets" if budget.kind == "count" else "greedy permutations"
    )
    if not pool or not exact.oracle.identities:
        result.missing = "no catalogue or scoring episodes"
        return result
    histogram: Counter[int] = Counter()
    if budget.kind == "cost":
        if math.factorial(len(pool)) > EXACT_PERMUTATION_LIMIT:
            result.missing = (
                "greedy cost draws require order-dependent permutations; "
                f"exceed {EXACT_PERMUTATION_LIMIT} enumeration limit"
            )
            return result
        for perm in itertools.permutations(pool):
            gain = exact.value(cut(perm, budget, costs).actions)
            if gain == FRESH:
                result.unvalued_sets += 1
            elif not isinstance(gain, Gain):
                result.missing = f"catalogue set cannot resolve: {gain}"
                return result
            else:
                histogram[gain.net] += 1
        return _finish(result, histogram, exact)
    groups: dict[Signature, list[Action]] = defaultdict(list)
    for action in pool:
        effects = exact.effects(action)
        if isinstance(effects, str):
            result.missing = f"catalogue action cannot resolve: {effects}"
            return result
        groups[_signature(effects)].append(action)
    k = min(int(budget.amount), len(pool))
    noops = len(groups.pop((), []))
    active = list(groups.values())
    class_valuer = _ClassValuer(exact, active)
    states = 0

    try:
        # Count smaller class subsets first: their multiplicities tend to dominate
        # when most catalogue entries are no-change. A cap then bounds the rare tail
        # rather than losing the large mass of unvisited one-class subsets.
        for size in range(min(k, len(active)) + 1):
            for selected in itertools.combinations(range(len(active)), size):
                states += 1
                if states > EXACT_STATE_LIMIT:
                    raise CountingLimit
                polynomial = {0: 1}
                mask = 0
                for index in selected:
                    mask |= 1 << index
                    following: dict[int, int] = defaultdict(int)
                    for degree, ways in polynomial.items():
                        for count in range(1, min(len(active[index]), k - degree) + 1):
                            following[degree + count] += ways * math.comb(len(active[index]), count)
                    polynomial = following
                weight = sum(
                    ways * math.comb(noops, k - degree)
                    for degree, ways in polynomial.items()
                    if 0 <= k - degree <= noops
                )
                if weight:
                    net = class_valuer.value(mask)
                    if net == FRESH:
                        result.unvalued_sets += weight
                    elif isinstance(net, int):
                        histogram[net] += weight
                    else:
                        raise ValueError(f"catalogue set cannot resolve: {net}")
    except CountingLimit:
        result.counting_limited = f"resolution classes exceed {EXACT_STATE_LIMIT} counting states"
        result.uncomputed_sets = (
            math.comb(len(pool), k) - sum(histogram.values()) - result.unvalued_sets
        )
        result.unvalued_sets += result.uncomputed_sets
    except ValueError as exc:
        return ExactChance(method=result.method, missing=str(exc))
    assert sum(histogram.values()) + result.unvalued_sets == math.comb(len(pool), k)
    return _finish(result, histogram, exact)


class CountingLimit(Exception):
    """The deterministic enumeration cap; unvisited sets retain bounded mass."""
