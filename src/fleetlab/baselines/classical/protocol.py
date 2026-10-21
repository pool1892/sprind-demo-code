"""Public design constants and neutral deterministic tie breaking.

Window lengths come from the experiment design, not world truth.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from fleetlab.baselines.classical.regression import Array
from fleetlab.contracts.actions import Action

EXPERIMENT_LENGTH = 2
SCORING_LENGTH = 2
Z90 = 1.6448536269514722


def experiment_window(observed_last: int) -> tuple[int, int]:
    return observed_last + 1, observed_last + EXPERIMENT_LENGTH


def tie_key(action: Action) -> str:
    """Fixed lottery among equal-cost/equal-gain hypotheses, without lexical priority."""
    return hashlib.sha256(b"classical-ties-v2:" + action.model_dump_json().encode()).hexdigest()


def population_bound(weight: Array, covered: Array) -> float:
    """Outward guard for a physical percentage bound, independent of sum order.

    Eight ulps cover the sum, product and equivalent fraction representations;
    this changes no practically measurable interval width.
    """
    value = 100 * math.fsum(float(w) for w, keep in zip(weight, covered, strict=True) if keep)
    return min(100.0, value + 8 * float(np.spacing(value)))
