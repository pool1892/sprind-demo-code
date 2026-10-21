"""Budgeted object-task calibration; run through tools/heavy-run from a fixed checkout."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from fleetlab import dose_response_stage3 as s3

# One or two useful levels around the spatial curve's knee; object effects decide
# which 100-pair measurements need the prespecified extension to 300.
LEVELS = (
    ("lens_contamination", ("0.75 scene film", "1 scene film")),
    ("camera_latency", ("12 steps scene lag", "24 steps scene lag")),
    ("gripper_force_loss", ("0.25 force lost", "0.5 force lost")),
    ("gripper_sticky_release", ("10 steps late open", "20 steps late open")),
    ("arm_torque_loss", ("0.2 torque lost", "0.3 torque lost")),
    ("controller_overload", ("0.1 cycles missed", "0.2 cycles missed")),
    ("calibration_update", ("scene calibration rewritten",)),
    ("image_pipeline_release", ("10% cropped", "20% cropped")),
    ("table_height_change", ("2 cm lower", "2 cm higher")),
    # light_color_shift is already measured as the third decoy.
    ("success_false_positive", ("0.5 on a 6 deg camera", "1 on a 6 deg camera")),
)
DECOYS = (
    ("illumination_change", "2 stops dimmer"),
    ("backdrop_change", "1 walls and floor"),
    s3.DECOY_LABEL,
)


def run(root: Path, budget_seconds: float, workers: int = 8) -> int:
    start = time.monotonic()
    setup = s3.xvla_setup()
    completed: list[s3.PlanRun] = []
    timings: list[dict[str, object]] = []
    seconds_per_episode = 1.5  # conservative until this invocation measures its throughput
    root.mkdir(parents=True, exist_ok=True)

    def perform(item: s3.PlanRun) -> bool:
        nonlocal seconds_per_episode
        count = len(s3.jobs_for(item, setup.policy, setup.software))
        remaining = budget_seconds - (time.monotonic() - start)
        if count * seconds_per_episode + 30 > remaining:
            return False
        before = time.monotonic()
        if s3.run(root=root, workers=workers, setup=setup, runs=[item]) != 0:
            raise ValueError(f"{item.name}: calibration failed validation")
        elapsed = time.monotonic() - before
        timings.append({"name": item.name, "episodes": count, "seconds": elapsed})
        seconds_per_episode = 1.15 * max(0.8, elapsed / max(1, count))
        completed.append(item)
        save()
        return True

    def save() -> None:
        result = s3.analyse_object(root, completed)
        (root / "analysis.json").write_text(json.dumps(result, indent=1) + "\n")
        (root / "receipt.json").write_text(
            json.dumps(
                {
                    "budget_seconds": budget_seconds,
                    "elapsed_seconds": time.monotonic() - start,
                    "completed": [item.__dict__ for item in completed],
                    "timings": timings,
                },
                indent=1,
            )
            + "\n"
        )

    healthy = s3.PlanRun("object-healthy", count=300, suite=s3.OBJECT)
    if not perform(healthy):
        save()
        return 0
    for kind, label in DECOYS:
        item = s3.PlanRun(f"object-{kind}-300", kind, 300, s3.labels(kind, label), suite=s3.OBJECT)
        if not perform(item):
            return 0
    healthy_count = 300
    for count in (600, 1000):
        for kind, label in DECOYS:
            result = s3.analyse_object(root, completed)["object"][kind]
            point = next(p for p in result if p["label"] == label)
            if max(abs(v) for v in point["difference"][1:]) <= 0.015:
                continue
            if healthy_count < count:
                extension = s3.PlanRun(
                    f"object-healthy-{count}", count=count, first=healthy_count, suite=s3.OBJECT
                )
                if not perform(extension):
                    return 0
                healthy_count = count
            extension = s3.PlanRun(
                f"object-{kind}-{count}",
                kind,
                count,
                s3.labels(kind, label),
                first=point["n"],
                suite=s3.OBJECT,
            )
            if not perform(extension):
                return 0
    primary = {
        "camera_latency": "12 steps scene lag",
        "arm_torque_loss": "0.2 torque lost",
        "controller_overload": "0.1 cycles missed",
        "image_pipeline_release": "10% cropped",
        "table_height_change": "2 cm lower",
        "success_false_positive": "0.5 on a 6 deg camera",
    }
    first_levels = [(kind, primary.get(kind, names[-1])) for kind, names in LEVELS]
    second_levels = [
        (kind, label)
        for kind, names in LEVELS
        for label in names
        if (kind, label) not in first_levels
    ]
    # Cover every type before spending the remaining queue on extra levels.
    for level_group in (first_levels, second_levels):
        extensions = []
        for kind, label in level_group:
            p = s3.labels(kind, label)
            item = s3.PlanRun(f"object-{kind}-{p[0]}-100", kind, 100, p, suite=s3.OBJECT)
            if not perform(item):
                return 0
            result = s3.analyse_object(root, completed)["object"][kind]
            point = next(p for p in result if p["label"] == label)
            if 0.02 <= abs(point["difference"][0]) <= 0.15:
                extensions.append(
                    s3.PlanRun(
                        f"object-{kind}-{p[0]}-300", kind, 300, p, first=100, suite=s3.OBJECT
                    )
                )
        for extension in extensions:
            if not perform(extension):
                return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("budget must be positive")

    def expired(signum: int, frame: object) -> None:
        raise TimeoutError("calibration heavy-queue budget exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
    started = time.monotonic()
    try:
        return run(args.root, args.budget_seconds, args.workers)
    except TimeoutError:
        args.root.mkdir(parents=True, exist_ok=True)
        (args.root / "budget-stop.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "budget_seconds": args.budget_seconds,
                    "note": (
                        "Only completed validated runs in analysis.json are publishable; "
                        "interrupted runs remain on disk."
                    ),
                },
                indent=1,
            )
            + "\n"
        )
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
