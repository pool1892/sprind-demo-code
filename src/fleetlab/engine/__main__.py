"""The engine's command line.

    # run a manifest in the worker layout (heavy: loads the policy onto the GPU)
    tools/heavy-run uv run --locked python -m fleetlab.engine run MANIFEST RUN_DIR \\
        [--workers 10] [--batch-size 4] [--precision fp16_res32] [--video] [--faults standin] \\
        [--snapshot-at 37] [--no-chain-sample]

    # compare the chains of two runs, job by job (cheap)
    uv run --locked python -m fleetlab.engine compare RUN_A RUN_B [--manifest MANIFEST]

    # replay every snapshot of a run from where it was taken, in the same run directory,
    # and compare each continuation with its uninterrupted episode (heavy)
    tools/heavy-run uv run --locked python -m fleetlab.engine restore RUN_DIR MANIFEST [...]

`--check libero` ends episodes on LIBERO's own check, as the reference
evaluation does; the default is the fleet's check, "placed".
`--faults standin` uses the engine's stand-ins for the camera and delay faults
(`engine.standin_faults`); the default is `fleetlab.faults`. `--snapshot-at S`
takes a snapshot of every run job at step S + (number of the job modulo 41),
so that the snapshots fall at different places within a chunk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SnapshotRef
from fleetlab.engine.chain import Chain, compare
from fleetlab.engine.manifests import read

CHECKS = {
    "fleet": "fleetlab.sim.libero_success:FleetCheck",
    "libero": "fleetlab.sim.libero_sim:LiberoCheck",
}
REGISTRIES = {
    "standin": "fleetlab.engine.standin_faults:STANDIN",
    "fleetlab": "fleetlab.faults:REGISTRY",
}


def _config(args: argparse.Namespace, out: Path, **extra: Any) -> Any:
    from fleetlab.engine.fleet import RunConfig

    return RunConfig(
        out=str(out),
        batch_size=args.batch_size,
        workers=args.workers,
        precision=args.precision,
        video=args.video,
        life=args.life,
        chain_sample=not args.no_chain_sample,
        fault_registry=REGISTRIES[args.faults],
        check_factory=CHECKS[args.check],
        mem_floor_gib=args.mem_floor,
        teleoperator=args.teleoperator,
        continuation_steps=args.continuation_steps,
        episodes_per_robot_day=args.episodes_per_robot_day,
        allow_manifest_day_length=args.allow_manifest_day_length,
        **extra,
    )


def snapshot_steps(jobs: list[Job], first: int) -> dict[str, tuple[int, ...]]:
    return {j.key: (first + n % 41,) for n, j in enumerate(jobs) if j.mode is JobMode.run}


def cmd_run(args: argparse.Namespace) -> int:
    from fleetlab.engine.fleet import run

    jobs = read(Path(args.manifest))
    if args.limit:
        jobs = jobs[: args.limit]
    extra: dict[str, Any] = {}
    if args.snapshot_at is not None:
        steps = snapshot_steps(jobs, args.snapshot_at)
        jobs = [
            j.model_copy(update={"snapshot_steps": list(steps[j.key])}) if j.key in steps else j
            for j in jobs
        ]
    result = run(_config(args, Path(args.out), **extra), jobs)
    extra_record = result.record.extra
    summary = {
        k: extra_record.get(k)
        for k in (
            "jobs",
            "jobs_run_now",
            "successes",
            "workers",
            "throughput",
            "total_s",
            "share_of_rows_in_use",
            "inference_s_per_call_median",
            "peaks",
        )
    }
    summary["chain_ok"] = result.record.chain_ok
    print(json.dumps(summary, indent=1))
    return 0 if result.record.chain_ok is not False else 1


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = RunLayout(Path(args.a)), RunLayout(Path(args.b))
    if args.manifest:
        keys = [j.key for j in read(Path(args.manifest))]
    else:
        keys = sorted(p.stem for p in (a.truth_dir / "chain").glob("*.npz"))
    same, differ, missing = 0, [], []
    for key in keys:
        pa, pb = a.truth_dir / "chain" / f"{key}.npz", b.truth_dir / "chain" / f"{key}.npz"
        if not (pa.is_file() and pb.is_file()):
            missing.append(key)
            continue
        result = compare(Chain.load(pb), Chain.load(pa))
        if result.identical:
            same += 1
        else:
            differ.append({"key": key, "step": result.first_difference, "what": result.what})
    print(
        json.dumps(
            {
                "compared": same + len(differ),
                "identical": same,
                "differ": differ,
                "missing": missing,
            },
            indent=1,
        )
    )
    return 0 if not differ and not missing else 1


def cmd_restore(args: argparse.Namespace) -> int:
    """Replay each snapshot of a run in the same run directory and compare the chains."""
    from fleetlab.engine.fleet import run

    layout = RunLayout(Path(args.run_dir))
    jobs = read(Path(args.manifest))
    if args.limit:
        jobs = jobs[: args.limit]
    by_episode = {j.identity.episode_id: j for j in jobs if j.mode is JobMode.run}
    replays = []
    for path in sorted((layout.truth_dir / "snapshots").glob("*/*.pkl")):
        episode_id = path.parent.name
        job = by_episode.get(episode_id)
        if job is None:
            continue
        step = int(path.stem.rsplit(".", 1)[1])
        ref = SnapshotRef(
            episode_id=episode_id, step=step, path=path.relative_to(layout.root).as_posix()
        )
        replays.append(Job(identity=job.identity, mode=JobMode.replay, from_snapshot=ref))
    replays = replays[: args.count]
    result = run(_config(args, layout.root), jobs + replays)
    out: list[dict[str, Any]] = []
    for replay in replays:
        original = by_episode[replay.identity.episode_id]
        c = compare(Chain.load(layout.chain(replay)), Chain.load(layout.chain(original)))
        out.append(
            {
                "episode_id": replay.identity.episode_id,
                "step": replay.from_snapshot.step if replay.from_snapshot else None,
                "identical": c.identical,
                "what": c.what,
                "steps": c.compared_steps,
            }
        )
    report: dict[str, Any] = {
        "restores": len(out),
        "identical": sum(bool(r["identical"]) for r in out),
        "chain_ok": result.record.chain_ok,
        "detail": out,
    }
    (layout.root / "restore_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("restores", "identical", "chain_ok")}, indent=1))
    return 0 if report["identical"] == report["restores"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workers", type=int, default=10)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--precision", default="fp16_res32")
        p.add_argument("--video", action="store_true")
        p.add_argument("--life", type=int, default=16)
        p.add_argument("--faults", choices=sorted(REGISTRIES), default="fleetlab")
        p.add_argument("--check", choices=sorted(CHECKS), default="fleet")
        p.add_argument("--teleoperator", default=None)
        p.add_argument("--continuation-steps", type=int, default=None)
        p.add_argument(
            "--episodes-per-robot-day",
            type=int,
            default=None,
            help="the world's, which fixes the visible start times",
        )
        p.add_argument(
            "--allow-manifest-day-length",
            action="store_true",
            help="without the world's: one more than the manifest's largest episode index",
        )
        p.add_argument("--no-chain-sample", action="store_true")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument(
            "--mem-floor",
            type=float,
            default=8.0,
            help="GiB of memory that must stay available beside the run's estimate",
        )

    p = sub.add_parser("run", help="run a manifest")
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--snapshot-at", type=int, default=None)
    run_options(p)
    c = sub.add_parser("compare", help="compare the chains of two runs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--manifest", default=None)
    r = sub.add_parser("restore", help="replay a run's snapshots and compare")
    r.add_argument("run_dir")
    r.add_argument("manifest")
    r.add_argument("--count", type=int, default=20)
    run_options(r)
    args = parser.parse_args(argv)
    if args.command != "compare":
        from fleetlab.runtime import configure_process

        configure_process()
        import fleetlab  # noqa: F401  (torch before mujoco)
    handler = {"run": cmd_run, "compare": cmd_compare, "restore": cmd_restore}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
