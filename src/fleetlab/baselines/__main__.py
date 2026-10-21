"""Run all four methods, preserving existing exports and prior results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fleetlab.baselines.common import (
    WORDING_VERSION,
    validate_export,
    write_answer,
    write_json,
)
from fleetlab.baselines.models import BaselineModelError, no_data_ranker
from fleetlab.baselines.random import random_fixes
from fleetlab.baselines.simple import fixed_prior, worst_health
from fleetlab.baselines.video import dramatic_videos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="must be a new directory")
    parser.add_argument(
        "--seed", type=int, help="default: seed recorded in the world oracle chance table"
    )
    parser.add_argument("--chance-table", type=Path, help="default: EXPORT/../oracle/oracle.json")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--offline", action="store_true", help="skip model methods explicitly")
    args = parser.parse_args()
    export, out = args.export.expanduser().resolve(), args.out.expanduser().resolve()
    if out.is_relative_to(export) or export.is_relative_to(out):
        parser.error("output must be separate from the export")
    try:
        validate_export(export)
        random_answers = random_fixes(
            export, seed=args.seed, draws=args.draws, chance_table=args.chance_table
        )
        seed = random_answers[0].metadata["seed"]
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "export": str(export),
        "seed": seed,
        "seed_source": "explicit --seed"
        if args.seed is not None
        else str(
            (args.chance_table or export.parent / "oracle/oracle.json").expanduser().resolve()
        ),
        "wording_version": WORDING_VERSION,
        "draws": args.draws,
        "methods": {},
        "offline": args.offline,
        "missing": {},
    }
    for draw in random_answers:
        root = out / "random" / f"draw-{draw.metadata['draw']:03}"
        write_answer(root, draw)
    manifest["methods"]["random"] = "random/draw-NNN/outbox/fix_set.json"
    for answer in (worst_health(export), fixed_prior(export)):
        write_answer(out / answer.method, answer)
        manifest["methods"][answer.method] = f"{answer.method}/outbox/fix_set.json"
    manifest["model_repeats"] = []
    for repeat in (1, 2):
        record: dict[str, Any] = {
            "repeat": repeat,
            "scored": repeat == 1,
            "methods": {},
            "missing": {},
        }
        manifest["model_repeats"].append(record)
        repeat_root = out if repeat == 1 else out / "repeats/repeat-2"
        for method in ("no_data", "dramatic_videos"):
            failure: dict[str, Any] | None = None
            if args.offline:
                failure = {"error": "offline", "detail": "model calls skipped"}
            elif "stopped" in manifest:
                failure = {"error": "not_attempted", "detail": "subscription stopped"}
            else:
                try:
                    answer = (
                        no_data_ranker(export, run_root=out / "runtime")
                        if method == "no_data"
                        else dramatic_videos(
                            export, repeat_root / "video-inputs", run_root=out / "runtime"
                        )
                    )
                    answer.metadata.update(repeat=repeat, scored=repeat == 1)
                    write_answer(repeat_root / method, answer)
                    path = str((repeat_root / method / "outbox/fix_set.json").relative_to(out))
                    record["methods"][method] = path
                    if repeat == 1:
                        manifest["methods"][method] = path
                except (BaselineModelError, ValueError) as exc:
                    failure = (
                        {
                            "error": exc.result.error,
                            "detail": exc.result.error_detail,
                            "run_dir": exc.result.run_dir,
                            "attempts": exc.attempts,
                        }
                        if isinstance(exc, BaselineModelError)
                        else {"error": "invalid_output", "detail": str(exc)}
                    )
                    print(str(exc))
            if failure:
                # Missing is an unavailable comparison, never an observed zero.
                record["comparison_available"] = False
                record["margin"] = record["ratio"] = None
                record["missing"][method] = failure
                if repeat == 1:
                    manifest["comparison_available"] = False
                    manifest["margin"] = manifest["ratio"] = None
                    manifest["missing"][method] = failure
                if failure["error"] in {"auth", "rate_limit"}:
                    manifest["stopped"] = failure
    write_json(out / "manifest.json", manifest)
    print(out / "manifest.json")
    return 2 if any(r["missing"] for r in manifest["model_repeats"]) and not args.offline else 0


if __name__ == "__main__":
    raise SystemExit(main())
