"""Fleet-layout proofs and development-world measurement for the human teleoperator.

Run simulation commands under tools/heavy-run. The report command alone reads
truth tables; the live pass reads scheduled identities and visible logs only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fleetlab.contracts.identity import noise_for
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SupervisorSpec
from fleetlab.engine import manifests, software
from fleetlab.engine.chain import Chain, compare
from fleetlab.engine.fleet import RunConfig, persist_followup, run
from fleetlab.engine.scripted import SWAY_SPEC
from fleetlab.policies.teleop.human import HumanSettings, admit
from fleetlab.policies.teleoperator import EXPERT_STEPS

HUMAN = "fleetlab.policies.teleop.human:HumanTeleoperator"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def copy_snapshots(source: RunLayout, target: RunLayout) -> None:
    for path in source.snapshots_dir.rglob("*.pkl"):
        destination = target.root / path.relative_to(source.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)


def human_pass(config: RunConfig, settings: HumanSettings) -> list[Job]:
    layout = RunLayout(Path(config.out))
    jobs = admit(layout.continuations(), settings, config.episodes_per_robot_day)
    # The admission replaces the raw continuation next to the same snapshot.
    # A resumed fleet invocation now finds the seeded settings and queue slot.
    for job in jobs:
        persist_followup(layout, job)
    manifests.write(layout.root / "admitted.jsonl", jobs)
    completed = run(
        replace(config, teleoperator=HUMAN, defer_continuations=True, chain_sample=False), jobs
    )
    metrics_path = layout.root / "human-metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update(
        {s["key"]: s["human"] for s in completed.record.extra["jobs_detail"] if "human" in s}
    )
    dump(metrics_path, metrics)
    dump(layout.root / "human-run.json", completed.record.model_dump(mode="json"))
    return jobs


def repeat(config: RunConfig, jobs: list[Job], out: Path) -> dict[str, Any]:
    source, target = RunLayout(Path(config.out)), RunLayout(out)
    copy_snapshots(source, target)
    run(
        replace(
            config,
            out=str(out),
            workers=max(1, config.workers - 1),
            teleoperator=HUMAN,
            defer_continuations=True,
            chain_sample=False,
        ),
        list(reversed(jobs)),
    )
    comparisons = {
        j.key: asdict(compare(Chain.load(target.chain(j)), Chain.load(source.chain(j))))
        for j in jobs
    }
    return {
        "episodes": len(jobs),
        "identical": sum(c["identical"] for c in comparisons.values()),
        "comparisons": comparisons,
    }


def cpu_proof(root: Path, workers: int, settings: HumanSettings) -> dict[str, Any]:
    jobs = []
    for n, task in enumerate((0, 1, 4, 7), 1):
        identity = manifests.identity(
            world_seed=604,
            episode_number=n,
            robot=n,
            site=1,
            day=0,
            episode_index=0,
            suite="libero_spatial",
            task_index=task,
            start_state=2,
            policy=SWAY_SPEC,
            batch_size=1,
            software=software.current("none"),
            image_size=64,
            max_steps=280,
        )
        jobs.append(
            Job(
                identity=identity,
                supervisor=SupervisorSpec(name="timeout", params={"after": 20, "decision_ms": 500}),
            )
        )
    cfg = RunConfig(
        out=str(root / "fleet"),
        batch_size=1,
        workers=workers,
        precision="fp32",
        device="cpu",
        miopen=None,
        policy_factory="fleetlab.engine.scripted:SwayPolicy",
        teleoperator=HUMAN,
        continuation_steps=settings.active_budget,
        episodes_per_robot_day=14,
        defer_continuations=True,
        chain_sample=True,
        life=4,
    )
    started = time.perf_counter()
    collected = run(cfg, jobs)
    supervised_ok = collected.record.chain_ok
    admitted = human_pass(cfg, settings)
    proof = repeat(cfg, admitted, root / "repeat")
    result = {
        "settings": settings.model_dump(mode="json"),
        "supervised_chain_ok": supervised_ok,
        "replay": proof,
        "wall_s": time.perf_counter() - started,
    }
    dump(root / "proof.json", result)
    return result


def pace_proof(root: Path, workers: int) -> dict[str, Any]:
    """Paired cadence check from the CPU proof's snapshots, no queue/reaction/fumble."""
    source = RunLayout(root / "fleet")
    originals = source.continuations()
    results = {}
    for pace in (1, 2):
        settings = HumanSettings(
            operators=len(originals),
            reaction_min_s=0,
            reaction_max_s=0,
            fumble_probability=0,
            pace=pace,
        )
        out = root / f"pace-{pace}"
        copy_snapshots(source, RunLayout(out))
        jobs = admit(originals, settings, 14)
        cfg = RunConfig(
            out=str(out),
            batch_size=1,
            workers=workers,
            precision="fp32",
            device="cpu",
            miopen=None,
            policy_factory="fleetlab.engine.scripted:SwayPolicy",
            teleoperator=HUMAN,
            episodes_per_robot_day=14,
            defer_continuations=True,
            chain_sample=False,
            life=4,
        )
        completed = run(cfg, jobs)
        results[str(pace)] = {
            s["episode_id"]: {"placed": s["success"], "active_steps": s["human"]["active_steps"]}
            for s in completed.record.extra["jobs_detail"]
            if "human" in s
        }
    result = {
        "episodes": len(originals),
        "cadence": results,
        "ratios": {
            e: results["2"][e]["active_steps"] / results["1"][e]["active_steps"]
            for e in results["1"]
        },
    }
    dump(root / "pace-proof.json", result)
    return result


def require_ready(world: Path) -> dict[str, Any]:
    from fleetlab.worlds.layout import WorldPaths

    problem = WorldPaths.from_root(world).readiness_problem
    if problem:
        raise ValueError(f"{world}: no current checked export; world is not read: {problem}")
    files = world / "files"
    if not files.is_dir():
        files = world / "world"
    spec = json.loads((files / "spec.json").read_text())
    return {"files": files, "spec": spec}


def expert_only(root: Path, latest_start: str | None = None) -> dict[str, Any]:
    """Run the unchanged expert from both methods' previously admitted snapshots."""
    if latest_start is not None:
        deadline = datetime.fromisoformat(latest_start)
        if deadline.tzinfo is None or datetime.now(deadline.tzinfo) >= deadline:
            raise ValueError("latest permitted heavy-job start has passed or lacks a timezone")
    measurement = json.loads((root / "measurement.json").read_text())
    if measurement["settings"]["expert_steps"] != EXPERT_STEPS:
        raise ValueError("the expert-alone factory has a fixed 600-command budget")
    os.environ["FLEETLAB_MIOPEN_DIR"] = str(Path.home() / ".cache/robots-probes/engine/miopen")
    started = time.perf_counter()
    phases = {}
    for name in ("timeout", "random"):
        source = RunLayout(root / name)
        target = RunLayout(root / "expert-only" / name)
        copy_snapshots(source, target)
        jobs = [
            job.model_copy(update={"teleoperator_params": None})
            for job in manifests.read(source.root / "admitted.jsonl")
        ]
        config = json.loads((source.root / "human-run.json").read_text())["extra"]["config"]
        config.update(
            out=str(target.root),
            teleoperator="fleetlab.policies.teleoperator:Teleoperator",
            defer_continuations=True,
            chain_sample=False,
        )
        manifests.write(target.root / "manifest.jsonl", jobs)
        result = run(RunConfig(**config), jobs)
        phases[name] = result.record.model_dump(mode="json")
    result = {
        "source_measurement": measurement["code"],
        "human_disabled": ["queue", "reaction", "pace", "fumble"],
        "phases": phases,
        "wall_s": time.perf_counter() - started,
    }
    dump(root / "expert-only" / "measurement.json", result)
    return result


def world_pass(
    world: Path,
    root: Path,
    workers: int,
    limit: int,
    after: int,
    settings: HumanSettings,
    window: str = "scoring",
    latest_start: str | None = None,
) -> dict[str, Any]:
    if latest_start is not None:
        deadline = datetime.fromisoformat(latest_start)
        if deadline.tzinfo is None:
            raise ValueError("latest_start needs a timezone offset")
        if datetime.now(deadline.tzinfo) >= deadline:
            raise ValueError("latest permitted heavy-job start has passed")
    ready = require_ready(world)
    files, spec = ready["files"], ready["spec"]
    days = spec["windows"][window]
    jobs = [
        j
        for j in manifests.read(files / "manifest.jsonl")
        if days["first"] <= j.identity.day <= days["last"]
    ]
    # A seeded sample of the entire window, not a sample chosen from failures or fault labels.
    jobs = sorted(jobs, key=lambda j: float(noise_for(j.identity, 0, (1,), sample=606)[0]))[:limit]
    jobs.sort(key=lambda j: j.identity.episode_id)
    miopen_dir = Path.home() / ".cache/robots-probes/engine/miopen"
    os.environ["FLEETLAB_MIOPEN_DIR"] = str(miopen_dir)
    cfg = RunConfig(
        out=str(root / "timeout"),
        workers=workers,
        batch_size=jobs[0].identity.batch_size,
        precision=jobs[0].identity.policy.precision,
        teleoperator=HUMAN,
        continuation_steps=settings.active_budget,
        episodes_per_robot_day=spec["episodes_per_robot_day"],
        defer_continuations=True,
        chain_sample=True,
        life=8,
    )
    started = time.perf_counter()
    timeout = [
        j.model_copy(
            update={
                "supervisor": SupervisorSpec(
                    name="timeout", params={"after": after, "decision_ms": 500}
                )
            }
        )
        for j in jobs
    ]
    manifests.write(root / "sample.jsonl", jobs)
    manifests.write(root / "timeout.jsonl", timeout)
    collected = run(cfg, timeout)
    dump(root / "timeout-collect-run.json", collected.record.model_dump(mode="json"))
    timeout_chain_ok = collected.record.chain_ok
    handovers = human_pass(cfg, settings)
    # Random episode selection is seeded and blind. The early fixed trigger prevents
    # successes ending before selected hand-overs; rate is matched to actual timeout hand-overs.
    chosen = {
        j.key
        for j in sorted(jobs, key=lambda j: float(noise_for(j.identity, 0, (1,), sample=605)[0]))[
            : len(handovers)
        ]
    }
    random_jobs = [
        j.model_copy(
            update={
                "supervisor": SupervisorSpec(
                    name="timeout", params={"after": 20, "decision_ms": 500}
                )
            }
        )
        if j.key in chosen
        else j
        for j in jobs
    ]
    random_cfg = replace(cfg, out=str(root / "random"))
    manifests.write(root / "random.jsonl", random_jobs)
    random_collected = run(random_cfg, random_jobs)
    dump(root / "random-collect-run.json", random_collected.record.model_dump(mode="json"))
    random_chain_ok = random_collected.record.chain_ok
    random_handovers = human_pass(random_cfg, settings)
    # Counterfactuals retain every fault and the original policy's remaining chunk.
    for c, handed in ((cfg, handovers), (random_cfg, random_handovers)):
        counter = [
            Job(identity=j.identity, mode=JobMode.replay, from_snapshot=j.from_snapshot)
            for j in handed
        ]
        manifests.write(Path(c.out) / "counterfactual.jsonl", counter)
        counter_result = run(replace(c, chain_sample=False), counter)
        dump(Path(c.out) / "counterfactual-run.json", counter_result.record.model_dump(mode="json"))
    proof = repeat(cfg, handovers[:4], root / "xvla-repeat")
    result = {
        "world": str(world),
        "window": window,
        "sample": len(jobs),
        "timeout_after": after,
        "random_after": 20,
        "timeout_handovers": len(handovers),
        "random_handovers": len(random_handovers),
        "settings": settings.model_dump(mode="json"),
        "timeout_chain_ok": timeout_chain_ok,
        "random_chain_ok": random_chain_ok,
        "xvla_continuation_replay": proof,
        "wall_s": time.perf_counter() - started,
        "code": __import__("fleetlab.engine.fleet", fromlist=["code_version"]).code_version(),
    }
    dump(root / "measurement.json", result)
    return result


def read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def report(root: Path) -> dict[str, Any]:
    """Truth is accessed only here, after simulation, to score and stratify."""
    measurement = json.loads((root / "measurement.json").read_text())
    world = Path(measurement["world"])
    jobs = manifests.read(root / "sample.jsonl")
    ids = {j.identity.episode_id for j in jobs}
    baseline = {}
    for path in (world / "runs").glob("*/truth/job_outcomes.parquet"):
        for row in read_rows(path):
            if row["mode"] == "run" and row["episode_id"] in ids:
                baseline[row["episode_id"]] = bool(row["success"])
    if set(baseline) != ids:
        raise ValueError("baseline outcomes missing; no denominator is silently dropped")
    result = {
        "measurement": measurement,
        "no_handover": {"placed": sum(baseline.values()), "episodes": len(baseline)},
        "methods": {},
    }
    by_identity = {j.identity.episode_id: j.identity for j in jobs}
    for name in ("timeout", "random"):
        layout = RunLayout(root / name)
        handed = manifests.read(layout.root / "admitted.jsonl")
        outcomes = {r["job_key"]: r for r in read_rows(layout.table("job_outcomes"))}
        decisions = (
            read_rows(layout.table("decisions")) if layout.table("decisions").exists() else []
        )
        metrics = json.loads((layout.root / "human-metrics.json").read_text())
        detail = []
        for job in handed:
            ident = job.identity
            counter = Job(identity=ident, mode=JobMode.replay, from_snapshot=job.from_snapshot)
            taken, without = outcomes[job.key], outcomes[counter.key]
            detail.append(
                {
                    "episode_id": ident.episode_id,
                    "placed": bool(taken["success"]),
                    "snapshot_without": bool(without["success"]),
                    "as_run": baseline[ident.episode_id],
                    "fault_types": sorted({f.kind for f in by_identity[ident.episode_id].faults})
                    or ["healthy"],
                    "params": job.teleoperator_params,
                    "human": metrics[job.key],
                }
            )

        def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
            return {
                "handovers": len(rows),
                "placed": sum(r["placed"] for r in rows),
                "snapshot_would_succeed": sum(r["snapshot_without"] for r in rows),
                "as_run_would_succeed": sum(r["as_run"] for r in rows),
                "rescued_snapshot": sum(r["placed"] and not r["snapshot_without"] for r in rows),
                "broken_snapshot": sum(not r["placed"] and r["snapshot_without"] for r in rows),
                "snapshot_would_fail": sum(not r["snapshot_without"] for r in rows),
                "rescued_as_run": sum(r["placed"] and not r["as_run"] for r in rows),
                "as_run_would_fail": sum(not r["as_run"] for r in rows),
            }

        groups = {}
        for kind in sorted({kind for row in detail for kind in row["fault_types"]}):
            groups[kind] = counts([row for row in detail if kind in row["fault_types"]])
        by_ep = {r["episode_id"]: r for r in detail}
        placed = sum(
            by_ep[e]["placed"] if e in by_ep else bool(outcomes[e]["success"]) for e in ids
        )
        result["methods"][name] = {
            **counts(detail),
            "all_placed": placed,
            "placed_while_waiting": sum(
                r["placed"] and r["human"].get("active_steps") == 0 for r in detail
            ),
            "supervisor_calls": len(decisions),
            "supervisor_decision_s": sum(d["decision_ms"] for d in decisions) / 1000,
            "supervisor_hold_s": sum(
                min(d["hold_steps"], max(0, outcomes[d["episode_id"]]["steps"] - d["step"]))
                for d in decisions
            )
            / 20,
            "by_fault_type": groups,
            "operator_reserved_s": len(detail) * measurement["settings"]["service_s"],
            "operator_used_s": sum(r["human"]["operator_used_s"] for r in detail),
            "reaction_s": sum(
                max(0, r["human"]["idle_steps"] - r["params"]["queue_steps"]) for r in detail
            )
            / 20,
            "fumble_draws": sum(r["human"]["fumble_draw"] for r in detail),
            "fumbles_applied": sum(r["human"]["fumble_applied"] for r in detail),
            "queue_s": sum(
                min(r["human"]["idle_steps"], r["params"]["queue_steps"]) for r in detail
            )
            / 20,
            "scheduled_queue_s": sum(r["params"]["queue_steps"] for r in detail) / 20,
            "detail": detail,
        }
        expert_layout = RunLayout(root / "expert-only" / name)
        if expert_layout.table("job_outcomes").exists():
            expert_outcomes = {
                row["job_key"]: row for row in read_rows(expert_layout.table("job_outcomes"))
            }
            expert_jobs = manifests.read(expert_layout.root / "manifest.jsonl")
            if {j.identity.episode_id for j in expert_jobs} != set(by_ep):
                raise ValueError("expert-only denominator differs from the hand-over cohort")
            expert_rows = [
                {**by_ep[j.identity.episode_id], "placed": bool(expert_outcomes[j.key]["success"])}
                for j in expert_jobs
            ]
            result["methods"][name]["expert_alone"] = {
                **counts(expert_rows),
                "by_fault_type": {
                    kind: counts([row for row in expert_rows if kind in row["fault_types"]])
                    for kind in groups
                },
            }
    dump(root / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("cpu-proof", "pace-proof", "expert-only", "world", "report")
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=480)
    parser.add_argument("--after", type=int, default=200)
    parser.add_argument("--window", choices=("observed", "scoring"), default="scoring")
    parser.add_argument(
        "--latest-start",
        help="ISO datetime with timezone; checked after heavy-run acquires its lock",
    )
    parser.add_argument("--settings", type=Path, help="JSON HumanSettings, optional")
    args = parser.parse_args()
    settings = (
        HumanSettings.model_validate_json(args.settings.read_text())
        if args.settings
        else HumanSettings()
    )
    if args.command == "cpu-proof":
        result = cpu_proof(args.root, args.workers, settings)
    elif args.command == "pace-proof":
        result = pace_proof(args.root, args.workers)
    elif args.command == "expert-only":
        result = expert_only(args.root, args.latest_start)
    elif args.command == "world":
        if args.world is None:
            parser.error("world requires --world")
        result = world_pass(
            args.world,
            args.root,
            args.workers,
            args.limit,
            args.after,
            settings,
            args.window,
            args.latest_start,
        )
    else:
        result = report(args.root)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("methods", "comparisons", "phases")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
