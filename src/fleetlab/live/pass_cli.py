"""Claim-4 passes, stored scoring and cost/timeout sweeps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import SupervisorSpec
from fleetlab.engine.fleet import RunConfig
from fleetlab.live.figure import draw
from fleetlab.live.pass_run import cohort, digest, economic_tape, execute
from fleetlab.live.scoring import score_files
from fleetlab.policies.teleop.human import HumanSettings


def load_supervisor(path: Path) -> SupervisorSpec:
    """Accept an engine spec or the supervisor's tracked VLM settings profile."""
    value = json.loads(path.read_text())
    if "name" in value:
        return SupervisorSpec.model_validate(value)
    from fleetlab.live.vlm.supervisor import VLMSettings

    return SupervisorSpec(name="vlm", params=asdict(VLMSettings(**value)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("cohort")
    prepare.add_argument("world", type=Path)
    run = sub.add_parser("pass")
    sweep = sub.add_parser("sweep")
    for p in (run, sweep):
        p.add_argument("--world", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument(
            "--rule", choices=("none", "timeout", "random", "decision", "supervisor"), required=True
        )
        p.add_argument("--operators", type=int, default=2)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--after", type=int, default=174)
        p.add_argument("--cost-ratio", type=float, default=2)
        p.add_argument("--idle-cost", type=float, default=0)
        p.add_argument("--max-wait-s", type=float)
        p.add_argument("--avoidable-fraction", type=float, default=1)
        p.add_argument("--reference", type=Path)
        p.add_argument(
            "--supervisor", type=Path, help="JSON SupervisorSpec or VLM settings profile"
        )
        p.add_argument("--factory", default="fleetlab.engine.supervisors:make")
        p.add_argument("--latency-route", choices=("subscription", "model"), default="subscription")
    sweep.add_argument("--ratios", type=float, nargs="+", default=[1, 2, 4])
    sweep.add_argument("--timeouts", type=int, nargs="+", default=[100, 174, 240])
    sweep.add_argument("--matched-random", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("records", type=Path, nargs="+")
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--resamples", type=int, default=1000)
    figure = sub.add_parser("figure")
    figure.add_argument("table", type=Path)
    figure.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cohort":
        _, population = cohort(args.world)
        print(json.dumps({k: v for k, v in population.items() if k != "episodes"}, indent=2))
        return
    if args.command == "score":
        score_files(args.records, args.out, args.resamples)
        return
    if args.command == "figure":
        draw(args.table, args.out)
        return
    jobs, population = cohort(args.world)
    config = RunConfig(
        out="",
        workers=args.workers,
        batch_size=jobs[0].identity.batch_size,
        precision=jobs[0].identity.policy.precision,
        life=8,
    )
    settings = HumanSettings(operators=args.operators)
    options: dict[str, Any] = {
        "after": args.after,
        "cost_ratio": args.cost_ratio,
        "idle_cost": args.idle_cost,
        "max_wait_s": args.max_wait_s,
        "avoidable_fraction": args.avoidable_fraction,
        "reference": args.reference,
        "supervisor": load_supervisor(args.supervisor) if args.supervisor else None,
        "factory": args.factory,
        "config": config,
        "latency_route": args.latency_route,
    }
    if args.command == "pass":
        result = execute(args.world, args.out, args.rule, settings, **options)
        print(
            json.dumps(
                {
                    "rule": result["rule"],
                    "cohort": len(population["episodes"]),
                    "record": str(args.out / "record.json"),
                },
                indent=2,
            )
        )
        return
    if args.rule == "random":
        parser.error("sweep random through the comparison rule's --matched-random")
    points = args.timeouts if args.rule == "timeout" else args.ratios
    records = []
    tape = None
    if args.rule == "decision":
        points = [p for p in points if p * args.avoidable_fraction > 1]
        tape = economic_tape(args.world, args.out, list(args.ratios), args.avoidable_fraction)
    for point in points:
        key = "after" if args.rule == "timeout" else "cost_ratio"
        out = args.out / f"{key}-{point:g}"
        point_options = {**options, key: point}
        if tape is not None:
            point_options.update(
                economic_points=tape["points"][f"{point:g}"],
                economic_tape_digest=digest(args.out / "economic-tape.json"),
            )
        execute(args.world, out, args.rule, settings, **point_options)
        records.append(out / "record.json")
        if args.matched_random:
            random_out = args.out / f"random-{key}-{point:g}"
            execute(
                args.world,
                random_out,
                "random",
                settings,
                **{
                    **options,
                    key: point,
                    "reference": out,
                    "supervisor": None,
                    "factory": "fleetlab.engine.supervisors:make",
                },
            )
            records.append(random_out / "record.json")
    score_files(records, args.out / "scores")
    draw(args.out / "scores/table.json", args.out / "scores/success.png")


if __name__ == "__main__":
    main()
