"""Uniform catalogue permutations; chance-table seed read only when called."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fleetlab.baselines.common import BaselineAnswer, export_catalogue, ranking


def random_fixes(
    export: Path, *, seed: int | None = None, draws: int = 200, chance_table: Path | None = None
) -> list[BaselineAnswer]:
    """One uniform permutation per draw; prefix counts and greedy cost cuts match stage 2."""
    if draws < 1:
        raise ValueError("draws must be positive")
    if seed is None:
        from fleetlab.baselines._seed import default_seed

        seed = default_seed(export, chance_table)
    pool = export_catalogue(export)
    rng = np.random.default_rng(seed)
    return [
        BaselineAnswer(
            "random",
            ranking([pool[int(i)] for i in rng.permutation(len(pool))]),
            metadata={"seed": seed, "draw": draw, "draws": draws},
        )
        for draw in range(draws)
    ]
