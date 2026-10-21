"""The oracle on a world root (src/fleetlab/oracle/README.md).

```sh
python -m fleetlab.oracle plan ROOT            # the manifest and its counts
python -m fleetlab.oracle estimate SPEC.json   # an upper bound from a spec alone
python -m fleetlab.oracle check ROOT           # the stored manifest, without the simulator
python -m fleetlab.oracle run ROOT             # the replays: one heavy job under tools/heavy-run
python -m fleetlab.oracle report ROOT          # table, gains, intervals, chance, checks
python -m fleetlab.oracle all ROOT             # plan, run, report
python -m fleetlab.oracle fresh ROOT --name N --fix-set FILE   # fresh replays: plan, run, value
```

(each through `uv run --locked`)

Common options: `--oracle-dir DIR` (default ROOT/oracle), `--runs RUN` (more run
directories with rows, repeatable). `--here` runs a heavy step in this process (what the
heavy job itself is started with).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import fleetlab  # noqa: F401  (torch before anything that imports mujoco)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", type=Path, help="the world root (with world/ and runs/)")
    parser.add_argument("--oracle-dir", type=Path, default=None)
    parser.add_argument(
        "--runs", type=Path, action="append", default=[], help="more runs with rows"
    )
    parser.add_argument(
        "--confirm",
        choices=("full", "single", "none"),
        default="full",
        help="confirmation replays of decoys and inert entries (plan.py)",
    )
    parser.add_argument(
        "--window",
        choices=("observed", "experiment", "scoring"),
        action="append",
        default=[],
        help="claim-bearing window; scoring is always verified",
    )


def _heavy(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers", type=int, default=10, help="0: as many as the free memory admits"
    )
    parser.add_argument("--here", action="store_true", help="run in this process (heavy)")


def _report(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--draws", type=int, default=200, help="random-fix draws")
    parser.add_argument("--parquet-into", type=Path, default=None)
    parser.add_argument(
        "--no-truth-table", action="store_true", help="write no truth/oracle.parquet into a run"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.oracle")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "check", "run", "report", "all", "fresh", "verify", "causes"):
        p = sub.add_parser(name)
        _common(p)
        if name in ("run", "all", "fresh", "verify", "causes"):
            _heavy(p)
        if name in ("report", "all"):
            _report(p)
        if name == "plan" or name == "all":
            p.add_argument("--replace", action="store_true", help="replace a stored plan")
        if name == "causes":
            p.add_argument(
                "--estimate", action="store_true", help="provisional counts from curves only"
            )
            p.add_argument("--plan-only", action="store_true")
            p.add_argument(
                "--run-only", action="store_true", help="execute only the stored observed stage"
            )
        if name == "verify":
            p.add_argument("--plan-only", action="store_true", help="count missing seconds")
        if name == "fresh":
            p.add_argument("--name", required=True, help="the fresh set's name")
            p.add_argument("--fix-set", type=Path, action="append", default=[])
            p.add_argument(
                "--request",
                type=Path,
                action="append",
                default=[],
                help="scorer needs_fresh_replay requests, any world window",
            )
            p.add_argument("--catalogue", action="store_true", help="every catalogue entry alone")
            p.add_argument("--oracle-items", action="store_true", help="the oracle's own items")
            p.add_argument(
                "--chance", action="store_true", help="all seeded draws at every scoring budget"
            )
            p.add_argument("--draws", type=int, default=200, help="chance draws (default 200)")
            p.add_argument("--plan-only", action="store_true")
            p.add_argument("--run-only", action="store_true", help="run the stored fresh set")
    estimate = sub.add_parser("estimate")
    estimate.add_argument("spec", type=Path)
    estimate.add_argument("--confirm", choices=("full", "single", "none"), default="full")
    args = parser.parse_args(argv)

    if args.command == "estimate":
        from fleetlab.contracts.world import WorldSpec
        from fleetlab.oracle.plan import estimate_from_spec

        spec = WorldSpec.model_validate_json(args.spec.read_text())
        print(json.dumps(estimate_from_spec(spec, confirm=args.confirm), indent=1))
        return 0

    from fleetlab.oracle.steps import Oracle

    oracle = Oracle.open(
        args.root, args.oracle_dir, args.runs, confirm=args.confirm, windows=args.window
    )
    if args.command == "verify":
        if args.plan_only:
            print(json.dumps(oracle.step_verification_plan(), indent=1))
            return 0
        print(json.dumps(oracle.step_verify(workers=args.workers, here=args.here), indent=1))
        return 0
    if args.command == "causes":
        if args.estimate:
            print(json.dumps(oracle.step_causes_plan(estimate=True), indent=1))
            return 0
        if args.run_only:
            print(
                json.dumps(oracle.step_causes_run(workers=args.workers, here=args.here), indent=1)
            )
            return 0
        # Each adaptive stage is planned in the caller, before a heavy job queues.
        for _ in range(3):
            observed_plan = oracle.step_causes_plan()
            print(json.dumps(observed_plan, indent=1), flush=True)
            if args.plan_only or not observed_plan["jobs_remaining"]:
                if not args.plan_only:
                    oracle.step_causes_run(workers=args.workers, here=args.here)
                return 0 if args.plan_only or oracle.measured_causes().complete else 2
            print(
                json.dumps(oracle.step_causes_run(workers=args.workers, here=args.here), indent=1),
                flush=True,
            )
        print(
            "cause measurements still incomplete after standalone and joint stages", file=sys.stderr
        )
        return 2
    if args.command == "plan":
        plan, written, _ = oracle.step_plan(replace=args.replace)
        print(
            json.dumps(
                {"written": written, "manifest": str(oracle.layout.jobs), **plan.counts()}, indent=1
            )
        )
        return 0
    if args.command == "check":
        from fleetlab.oracle.plan import check_jobs

        problems = check_jobs(oracle.world, oracle.stored_manifest())
        print(json.dumps({"jobs": len(oracle.stored_manifest()), "problems": problems}, indent=1))
        return 1 if problems else 0
    if args.command in ("run", "all"):
        if args.command == "all" or not oracle.layout.jobs.is_file():
            oracle.step_plan(replace=getattr(args, "replace", False))
        info = oracle.step_run(workers=args.workers, here=args.here)
        print(json.dumps(info, indent=1, default=str))
        if info.get("chain_ok") is False:
            return 1
        if args.command == "run":
            return 0
    if args.command in ("report", "all"):
        from fleetlab.oracle.report import ReportRefused, report_status

        try:
            report = oracle.step_report(
                boot=args.boot,
                draws=args.draws,
                parquet_into=args.parquet_into,
                truth_table=not args.no_truth_table,
            )
        except ReportRefused as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(oracle.layout.summary.read_text())
        return report_status(report)
    if args.command == "fresh":
        from fleetlab.oracle.fresh import read_fix_sets, read_requests

        if not args.run_only:
            fix_sets = read_fix_sets(args.fix_set)
            record = oracle.step_fresh_plan(
                args.name,
                fix_sets,
                catalogue=args.catalogue,
                oracle_items=args.oracle_items,
                chance=args.chance,
                draws=args.draws,
                requests=read_requests(args.request),
            )
            print(json.dumps({k: v for k, v in record.items() if k != "fix_sets"}, indent=1))
            if args.plan_only:
                return int(bool(record["refused"] or record["errors"]))
        info = oracle.step_fresh_run(args.name, workers=args.workers, here=args.here)
        print(json.dumps(info, indent=1, default=str))
        if info.get("chain_ok") is False:
            return 1
        if args.here and args.run_only:
            return 0
        values = oracle.step_fresh_value(args.name)
        print(json.dumps(values, indent=1))
        return int(any(v.get("missing") or v.get("error") for v in values))
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
