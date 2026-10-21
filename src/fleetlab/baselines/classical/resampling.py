"""Blind stability measurement: resample episodes within robot/day cells.

No truth lookup. Full answers are retained for one later, prespecified scoring
pass. Health histories and change logs stay fixed; this assesses sensitivity to
the observed episode sample, not repeated fleet histories or experiment noise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.data import Row, VisibleData, load_export
from fleetlab.baselines.classical.experiments import choose


def resample(data: VisibleData, rng: np.random.Generator) -> VisibleData:
    cells: dict[tuple[str, int], list[Row]] = {}
    for row in data.episodes:
        cells.setdefault((row["robot_id"], row["day"]), []).append(row)
    rows: list[Row] = []
    for key in sorted(cells):
        cell = cells[key]
        rows.extend(cell[i] for i in rng.integers(0, len(cell), len(cell)))
    # Repeated IDs deliberately carry their original event evidence. The engine
    # does not see these resamples, and they never impersonate new real attempts.
    return replace(data, episodes=sorted(rows, key=lambda r: r["episode_id"]))


def measure(export: Path, output: Path, *, draws: int = 100, seed: int = 20261009) -> None:
    if draws < 1:
        raise ValueError("positive draw count required")
    data = load_export(export)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    protocol = dict(
        world=data.fleet.id,
        seed=seed,
        draws=draws,
        input_sha256=data.hashes,
        rule="within robot/day episode resampling; fixed health and changes",
        bootstrap_draws_per_refit=200,
        note="200 uncertainty refits, identical to the canonical analysis",
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for index in range(draws):
        answer = analyse(resample(data, rng), bootstraps=200, seed=0)
        root = output / f"sample-{index:03}"
        answer.write(root)
        (root / "planned_actions.json").write_text(
            json.dumps([a.model_dump(mode="json") for a in choose(answer)], indent=2) + "\n"
        )
        print(f"{data.fleet.id} {index + 1}/{draws}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    measure(args.export, args.out, draws=args.draws, seed=args.seed)


if __name__ == "__main__":
    main()
