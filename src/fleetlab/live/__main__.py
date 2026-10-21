"""Prepare features, fit on development robots, and render the measured tables."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fleetlab.live.data import build_dataset, development_runs
from fleetlab.live.measure import measure, tables
from fleetlab.worlds.layout import WorldPaths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--world", required=True)
    prepare.add_argument("--run", type=Path, action="append", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    world = sub.add_parser("prepare-world")
    world.add_argument("root", type=Path)
    world.add_argument("--out", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, action="append", required=True)
    train.add_argument("--mini", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--models-out", type=Path, help="separate runtime model directory")
    report = sub.add_parser("tables")
    report.add_argument("metrics", type=Path)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "prepare-world"):
        started = time.perf_counter()
        name = args.world if args.command == "prepare" else WorldPaths.from_root(args.root).world_id
        runs = args.run if args.command == "prepare" else development_runs(args.root)
        data = build_dataset(name, runs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data.save(args.out)
        print(
            json.dumps(
                {
                    "episodes": len(data.episodes),
                    "rows": len(data.steps),
                    "seconds": time.perf_counter() - started,
                    "provenance": data.provenance,
                },
                indent=2,
            )
        )
    elif args.command == "train":
        results = measure(args.data, args.mini, args.out, models_out=args.models_out)
        (args.out / "tables.md").write_text(tables(results))
        print(
            json.dumps(
                {
                    "populations": results["populations"],
                    "fit_seconds": {
                        name: results["selection"][name]["fit_seconds"]
                        for name in ("telemetry", "time_only")
                    },
                },
                indent=2,
            )
        )
    else:
        args.out.write_text(tables(json.loads(args.metrics.read_text())))


if __name__ == "__main__":
    main()
