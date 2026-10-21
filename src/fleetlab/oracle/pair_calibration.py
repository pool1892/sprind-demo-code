"""Synthetic power study of the additive verdict; no robot simulation or GPU.

Two outcome couplings bracket noise as in the review: one uniform difficulty shared
by all four counterfactuals, or independent difficulties. Marginal healthy success
is 0.9; standalone harms are 0.05 and 0.127. Resampling is the report's episode
bootstrap over 336 scoring attempts, with non-carrying attempts contributing zero.
This studies statistical resolution under these assumptions, not physical robots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.oracle.gains import Estimate
from fleetlab.oracle.report import ADDITIVE_RELATIVE_TOLERANCE, additive_verdict


def calibrate(worlds: int = 300, boot: int = 1000, seed: int = 20261009) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cells = []
    for model in ("shared difficulty", "independent difficulties"):
        for carrying in (28, 112):
            for excess in (0, 10, 20):
                counts = dict.fromkeys(("contradicted", "below the sum", "not contradicted"), 0)
                probabilities = np.array([0.9 - 0.177 - excess / 100, 0.85, 0.773, 0.9])
                for _ in range(worlds):
                    difficulty = rng.random((carrying, 1 if model == "shared difficulty" else 4))
                    outcomes = (difficulty < probabilities).astype(np.int16)
                    both, a, b, healthy = outcomes.T
                    term = healthy + both - a - b
                    harm = 2 * healthy - a - b
                    # Collapse equal (term, harm) pairs, plus non-carriers: exactly
                    # the multinomial episode bootstrap, with fewer sampling bins.
                    vectors = np.column_stack((term, harm))
                    vectors = np.vstack((vectors, np.zeros((336 - carrying, 2), dtype=np.int16)))
                    values, frequencies = np.unique(vectors, axis=0, return_counts=True)
                    weights = rng.multinomial(336, frequencies / 336, size=boot)
                    low, high = np.percentile(100 * (weights @ values) / 336, [2.5, 97.5], axis=0)
                    estimate = Estimate(
                        item="pair",
                        item_kind="interaction",
                        net=int(term.sum()),
                        low=round(float(low[0]), 3),
                        high=round(float(high[0]), 3),
                        detail={
                            "summed_harm_net": int(harm.sum()),
                            "summed_harm_low": round(float(low[1]), 3),
                        },
                    )
                    verdict, _ = additive_verdict(estimate, 336)
                    counts[verdict] += 1
                cells.append(
                    {
                        "model": model,
                        "carrying": carrying,
                        "excess_points": excess,
                        "counts": counts,
                        "percent": {k: round(100 * v / worlds, 1) for k, v in counts.items()},
                    }
                )
    return {
        "seed": seed,
        "worlds_per_cell": worlds,
        "boot": boot,
        "scoring_attempts": 336,
        "tolerance": ADDITIVE_RELATIVE_TOLERANCE,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=300)
    args = parser.parse_args()
    result = calibrate(args.worlds)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for cell in result["cells"]:
        print(cell)


if __name__ == "__main__":
    main()
