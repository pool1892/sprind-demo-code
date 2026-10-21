"""Low-memory statistical simulation of the design-only sufficient power bound.

The two binary-counterfactual laws have the printed expected excess: maximal +/-2
spread, and independent counterfactual outcomes. Neither is physical robot evidence.
The analytic bound covers all independent carrier vectors, not only these laws.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.oracle.gains import BOOT, Estimate
from fleetlab.oracle.power import ASSUMPTIONS, METHOD, POWER, sufficient_power_bound
from fleetlab.oracle.report import ADDITIVE_RELATIVE_TOLERANCE, additive_verdict


def calibrate(worlds: int = 1000, seed: int = 20261010) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cells = []
    designs = [("w42 f01+f02", 28), ("w42 f01+f03", 28), ("w43 f01+f02", 28), ("112 carriers", 112)]
    for label, m in designs:
        bound = sufficient_power_bound(336, m, BOOT, ADDITIVE_RELATIVE_TOLERANCE)
        assert bound.points is not None
        mean = bound.points * 336 / (100 * m)
        for model in ("maximal +/-2 spread", "independent counterfactuals"):
            counts = dict.fromkeys(("contradicted", "below the sum", "not contradicted"), 0)
            for _ in range(worlds):
                if model == "maximal +/-2 spread":
                    positive = rng.random(m) < (mean + 2) / 4
                    both = healthy = (~positive).astype(np.int16)
                    a = b = positive.astype(np.int16)
                else:
                    # E[a+b-healthy-both] = mean, with all four independent.
                    probabilities = np.array(
                        [0.5 - mean / 4, 0.5 + mean / 4, 0.5 + mean / 4, 0.5 - mean / 4]
                    )
                    both, a, b, healthy = (rng.random((m, 4)) < probabilities).astype(np.int16).T
                term, harm = healthy + both - a - b, 2 * healthy - a - b
                vector = np.vstack(
                    (np.column_stack((term, harm)), np.zeros((336 - m, 2), dtype=np.int16))
                )
                values, frequencies = np.unique(vector, axis=0, return_counts=True)
                weights = rng.multinomial(336, frequencies / 336, size=BOOT)
                low, high = np.percentile(100 * (weights @ values) / 336, [2.5, 97.5], axis=0)
                estimate = Estimate(
                    item=label,
                    item_kind="interaction",
                    episodes=m,
                    net=int(term.sum()),
                    low=round(float(low[0]), 3),
                    high=round(float(high[0]), 3),
                    detail={"summed_harm_net": int(harm.sum())},
                )
                counts[additive_verdict(estimate, 336)[0]] += 1
            cells.append(
                {
                    "pair": label,
                    "carriers": m,
                    "model": model,
                    "sufficient_excess_power_bound_points": bound.points,
                    "expected_excess_net": bound.points * 336 / 100,
                    "counts": counts,
                    "contradiction_rate": counts["contradicted"] / worlds,
                }
            )
    # All 16 possible binary vectors obey the inequalities used in the proof.
    for both_i, a_i, b_i, healthy_i in itertools.product((0, 1), repeat=4):
        excess_i = a_i + b_i - healthy_i - both_i
        harm_i = 2 * healthy_i - a_i - b_i
        assert -2 <= excess_i <= 2 and harm_i + excess_i <= 1
    return {
        "seed": seed,
        "worlds_per_cell": worlds,
        "boot": BOOT,
        "scoring_attempts": 336,
        "guaranteed_power": POWER,
        "method": METHOD,
        "assumptions": ASSUMPTIONS,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1000)
    args = parser.parse_args()
    result = calibrate(args.worlds)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for cell in result["cells"]:
        print(cell)


if __name__ == "__main__":
    main()
