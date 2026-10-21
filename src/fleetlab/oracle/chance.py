"""The chance level of random fixes (experimental design, section 6): 200 draws from the world's
filled catalogue at every scoring budget, valued through the oracle table.

The draws are `fleetlab.oracle.valuation.random_draws`, shared with the scorer, from the
world seed. The oracle requires every draw to have a value; callers must apply the same
denominator to compare their baseline. A draw
that needs a fresh replay (an action that adds a change) or cannot be resolved makes
the distribution unavailable, with bounds over ALL draws. Percentiles are numpy's
linear ones; the sample interval is the 5th to 95th percentile of net in points.
Count-budget scoring uses the exact population (or its missing-replay bounds).
Cost budgets retain these draws with a separate population-quantile confidence interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from scipy.stats import binom

from fleetlab.contracts.actions import COSTS, SCORING_BUDGETS, Action, Budget, CostTable
from fleetlab.contracts.faults import TruthTable
from fleetlab.contracts.oracle import CurrentValues, Gain, OracleTable
from fleetlab.oracle.exact import ExactChance, exact_chance
from fleetlab.oracle.valuation import FRESH, DrawValuer, random_draws

DRAWS = 200
PERCENTILE_COVERAGE = 0.95
PERCENTILE_PROBABILITY = 0.95


def percentile_ranks(draws: int) -> tuple[int, int, float]:
    """One-based equal-tail binomial ranks, 0/n+1 denoting the known support ends.

    These bracket the population's lower 95th quantile with at least the stated
    coverage, including distributions with atoms. No smoothness assumption.
    """
    tail = (1 - PERCENTILE_COVERAGE) / 2
    low = int(binom.ppf(tail, draws, PERCENTILE_PROBABILITY))
    high = int(binom.ppf(1 - tail, draws, PERCENTILE_PROBABILITY)) + 1
    coverage = float(
        binom.cdf(high - 1, draws, PERCENTILE_PROBABILITY)
        - binom.cdf(low - 1, draws, PERCENTILE_PROBABILITY)
    )
    return low, high, coverage


def budget_label(budget: Budget) -> str:
    """ "k = 3" for a count budget, "4 h" for a cost budget (the scorer's labels)."""
    if budget.kind == "count":
        return f"k = {int(budget.amount)}"
    return f"{budget.amount:g} h"


class Chance(BaseModel):
    """Random fixes at one budget."""

    model_config = ConfigDict(extra="forbid")

    budget: str
    draws: int
    valued: int
    skipped_fresh_replay: int
    skipped_error: int
    mean_points: float | None = None
    p5_points: float | None = None
    p95_points: float | None = None
    above_zero: float | None = None  # share of all draws with a positive net
    achievable_net: int | None = None
    mean_share: float | None = None
    p5_share: float | None = None
    p95_share: float | None = None
    missing: str | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    exact: ExactChance | None = None
    p95_points_interval: tuple[float, float] | None = None
    p95_interval_coverage: float | None = None
    p95_interval_order_stats: tuple[int, int] | None = None


def chance_intervals(
    table: OracleTable,
    truth: TruthTable,
    current: CurrentValues | None,
    entries: Sequence[Action],
    *,
    seed: int,
    achievable: dict[str, int | None],
    draws: int = DRAWS,
    budgets: Sequence[Budget] = SCORING_BUDGETS,
    costs: CostTable = COSTS,
) -> list[Chance]:
    """The chance level at every budget; `achievable` maps a budget label to the net the
    oracle's best reaches there (for the shares)."""
    n = len(table.identities)
    results = random_draws(
        table, truth, current, entries, budgets, seed=seed, draws=draws, costs=costs
    )
    out: list[Chance] = []
    valuer = DrawValuer(table, truth, current)
    for budget, drawn in zip(budgets, results, strict=True):
        label = budget_label(budget)
        gains = [r for r in drawn if isinstance(r, Gain)]
        fresh = sum(1 for r in drawn if r == FRESH)
        chance = Chance(
            budget=label,
            draws=len(drawn),
            valued=len(gains),
            skipped_fresh_replay=fresh,
            skipped_error=len(drawn) - len(gains) - fresh,
            achievable_net=achievable.get(label),
            exact=exact_chance(valuer, entries, budget, costs),
        )
        if not entries:
            chance.missing = "no catalogue"
        elif not drawn or n == 0:
            chance.missing = f"no draw could be valued ({fresh} need a fresh replay)"
        elif chance.valued != chance.draws:
            unknown = chance.draws - chance.valued
            chance.missing = (
                f"unavailable: {unknown} of {chance.draws} draws lack values "
                f"({fresh} need replays, {chance.skipped_error} resolution errors)"
            )
            # A fix can break at most the baseline successes and rescue at most the
            # baseline failures. Unknown baseline outcomes allow either possibility.
            baseline = [table.success(i.episode_id, i.faults) for i in table.identities]
            break_limit = -sum(s is not False for s in baseline)
            rescue_limit = sum(s is not True for s in baseline)
            lower = np.array(
                [r.net if isinstance(r, Gain) else break_limit for r in drawn], dtype=float
            )
            upper = np.array(
                [r.net if isinstance(r, Gain) else rescue_limit for r in drawn], dtype=float
            )
            chance.bounds = _bounds(lower, upper, n, chance.achievable_net)
        else:
            nets = np.array([g.net for g in gains], dtype=float)
            pts = 100.0 * nets / n
            p5, p95 = np.percentile(pts, [5, 95])
            chance.mean_points = round(float(pts.mean()), 3)
            chance.p5_points = round(float(p5), 3)
            chance.p95_points = round(float(p95), 3)
            chance.above_zero = round(float((nets > 0).mean()), 3)
            best = chance.achievable_net
            if best:
                shares = nets / best
                s5, s95 = np.percentile(shares, [5, 95])
                chance.mean_share = round(float(shares.mean()), 4)
                chance.p5_share = round(float(s5), 4)
                chance.p95_share = round(float(s95), 4)
        if budget.kind == "cost" and drawn and n:
            baseline = [table.success(i.episode_id, i.faults) for i in table.identities]
            support_low = -sum(s is not False for s in baseline)
            support_high = sum(s is not True for s in baseline)
            low_sample = sorted(r.net if isinstance(r, Gain) else support_low for r in drawn)
            high_sample = sorted(r.net if isinstance(r, Gain) else support_high for r in drawn)
            low, high, coverage = percentile_ranks(len(drawn))
            lo = support_low if low == 0 else low_sample[low - 1]
            hi = support_high if high > len(drawn) else high_sample[high - 1]
            chance.p95_points_interval = (
                math.floor(100_000 * lo / n) / 1000,
                math.ceil(100_000 * hi / n) / 1000,
            )
            chance.p95_interval_order_stats = (low, high)
            chance.p95_interval_coverage = coverage
        out.append(chance)
    return out


def _bounds(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    episodes: int,
    achievable: int | None,
) -> dict[str, tuple[float, float]]:
    """Worst-case bounds on statistics of the complete draw population, not intervals
    from bootstrap sampling; monotonicity of means and percentiles makes them valid."""
    out: dict[str, tuple[float, float]] = {}
    for suffix, scale, digits in [
        ("points", 100.0 / episodes, 3),
        *([("share", 1.0 / achievable, 4)] if achievable else []),
    ]:
        lo, hi = lower * scale, upper * scale
        lp, hp = np.percentile(lo, [5, 95]), np.percentile(hi, [5, 95])
        for name, a, b in [
            ("mean", lo.mean(), hi.mean()),
            ("p5", lp[0], hp[0]),
            ("p95", lp[1], hp[1]),
        ]:
            out[f"{name}_{suffix}"] = (round(float(a), digits), round(float(b), digits))
    out["above_zero"] = (round(float((lower > 0).mean()), 3), round(float((upper > 0).mean()), 3))
    return out
