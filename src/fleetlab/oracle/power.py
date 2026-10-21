"""A design-only sufficient power bound for the actual additive verdict.

Carrier counterfactual vectors must be independent (not necessarily identically
distributed), and bootstrap draws independent of those vectors. No assumption about
their observed spread or outcome coupling is used. This is deliberately conservative,
not a minimum detectable excess or a calibration of the physical robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from fleetlab.oracle.gains import LOW

POWER = 0.80
SAMPLE_ERROR = 0.15
BOOTSTRAP_ERROR = 0.05
ROUNDING_GUARD_POINTS = 0.001  # exceeds the interval's nearest-rounding error
METHOD = "Cantelli sample bound; Bernstein bootstrap bound; binomial Chernoff bound"
ASSUMPTIONS = (
    "Independent binary counterfactual vectors across carriers (not necessarily iid); "
    "independent episode-bootstrap draws; noncarriers contribute zero. "
    "Probability is over carrier samples and bootstrap draws, not a fixed replay table."
)


@dataclass(frozen=True)
class PowerBound:
    points: float | None
    sample_net: float | None
    bootstrap_tail_probability: float | None
    unavailable: str | None = None


def bootstrap_tail_limit(boot: int) -> float:
    """Per-draw tail probability giving at most BOOTSTRAP_ERROR quantile failures.

    If fewer than r bootstrap sums lie at/below the guarded band, the linear LOW
    percentile lies above it. P(Bin(boot,p) >= r) <= exp(-boot KL(r/boot || p)).
    Bisection retains the safe lower endpoint, avoiding an optimistic numeric root.
    """
    r = math.floor((boot - 1) * LOW / 100) + 1
    q = r / boot
    target = -math.log(BOOTSTRAP_ERROR) / boot
    low, high = 0.0, q
    for _ in range(60):
        p = (low + high) / 2
        kl = q * math.log(q / p)
        if q < 1:
            kl += (1 - q) * math.log((1 - q) / (1 - p))
        if kl >= target:
            low = p
        else:
            high = p
    return low


@lru_cache(maxsize=256)
def sufficient_power_bound(n: int, carriers: int, boot: int, tolerance: float) -> PowerBound:
    """At this expected excess or greater, contradiction probability is at least 80%.

    Write X=-T for each carrier, H for standalone harm, S=sum X, and D=E S.
    Binary outcomes imply X in [-2,2], H+X=joint gain <=1. Thus the observed
    band is at most tolerance*max(0,m-S). Conditional on S>=c, the bootstrap
    sum has variance <=4m and centered lower increments <=2+2m/n. Bernstein
    bounds its lower tail; a binomial Chernoff bound handles the finite bootstrap.
    Choose c so even nearest rounding leaves the LOW percentile above the band.

    Independence gives Var S <=4m-D^2/m. Cantelli bounds P(S<c) by 0.15
    at the returned D. The bootstrap failure bound is 0.05. A union bound
    gives power >=1-0.15-0.05=0.80 under the stated assumptions.
    """
    m = carriers
    if not 0 < m <= n or boot < 1 or tolerance < 0:
        return PowerBound(None, None, None, "needs carriers, scoring episodes and bootstrap")
    p = bootstrap_tail_limit(boot)
    log_tail = -math.log(p)
    a = (2 / 3) * (2 + 2 * m / n) * log_tail
    gap = (a + math.sqrt(a * a + 32 * m * log_tail)) / 2
    guarded = gap + ROUNDING_GUARD_POINTS * n / 100
    c = guarded if guarded >= m else (guarded + tolerance * m) / (1 + tolerance)
    if c >= 2 * m:
        return PowerBound(None, c, p, "conservative bound exceeds attainable binary excess")
    q = (1 - SAMPLE_ERROR) / SAMPLE_ERROR
    d = (c + math.sqrt(q * (4 * m + 4 * q - c * c / m))) / (1 + q / m)
    # Round upward so quoting the field still satisfies the analytic inequality.
    points = math.ceil(100 * d / n * 1000) / 1000
    return PowerBound(points, c, p)
