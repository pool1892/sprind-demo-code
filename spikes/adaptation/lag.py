"""How late do a robot's commands take effect? A lag estimate from the visible telemetry only.

The visible `steps` table of every episode logs the commanded action before
the action hook (its first three numbers are the commanded end-effector
position, in the world frame, under absolute control) and the measured
end-effector position after the step. For each lag L in 0..MAX_LAG the
estimate takes the mean distance between the position commanded L steps ago
and the position measured now, over the same steps for every L, and returns
the L with the smallest mean distance. A healthy robot shows the arm
controller's own tracking lag; commands that arrive d steps late add d.

Nothing hidden is read: no fault, no truth table, no setting. The same
computation is a self-join of the steps table on `step = step' + L` in SQL.

    uv run --locked python spikes/adaptation/lag.py RUN_DIR [RUN_DIR ...] [--by block|cell]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

MAX_LAG = 40
FIRST = MAX_LAG  # the steps every lag is scored on start here
MIN_STEPS = 100  # shorter episodes say too little


def lag_curve(path: Path) -> np.ndarray | None:
    """Mean distance in metres between commanded and measured position, per lag; or None."""
    table = pq.read_table(path, columns=["step", "eef_pos", "commanded_action"]).to_pydict()
    commanded = [c for c in table["commanded_action"] if c is not None]
    if len(commanded) < MIN_STEPS:
        return None
    cmd = np.asarray([c[:3] for c in commanded], np.float64)
    eef = np.asarray(table["eef_pos"][1 : len(commanded) + 1], np.float64)  # after each step
    steps = np.arange(FIRST, len(cmd))
    return np.asarray(
        [np.linalg.norm(cmd[steps - lag] - eef[steps], axis=1).mean() for lag in range(MAX_LAG + 1)]
    )


def episode_lag(path: Path) -> int | None:
    curve = lag_curve(path)
    return None if curve is None else int(np.argmin(curve))


def group_of(episode_id: str, by: str) -> str:
    number = int(episode_id[1:])
    block = number // 1000
    if by == "block":
        return f"block {block}"
    # the adaptation phases' cells (fleetlab.adaptation), read without importing anything hidden
    from fleetlab.adaptation import PHASES, cell_of

    for phase in PHASES:
        try:
            return cell_of(phase, episode_id)[0].name
        except (IndexError, ValueError):
            continue
    return f"block {block}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--by", choices=("block", "cell"), default="block")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    lags: dict[str, list[int]] = {}
    for run in args.runs:
        for path in sorted((run / "log" / "steps").glob("*.parquet")):
            lag = episode_lag(path)
            if lag is not None:
                lags.setdefault(group_of(path.stem, args.by), []).append(lag)
    out: dict[str, Any] = {}
    for group in sorted(lags):
        values = lags[group]
        out[group] = {
            "episodes": len(values),
            "median_lag": statistics.median(values),
            "quartiles": [float(q) for q in np.percentile(values, [25, 75])],
        }
        print(
            f"{group}: {len(values)} episodes, median lag {statistics.median(values):g} steps "
            f"(quartiles {out[group]['quartiles'][0]:g} to {out[group]['quartiles'][1]:g})"
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
