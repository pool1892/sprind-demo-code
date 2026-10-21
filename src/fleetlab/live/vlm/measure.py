"""Small development-only paired measurement. Simulation runs via tools/heavy-run.

Selection and scoring may read truth. The supervisor receives only live
SupervisorContext. This is not the full cost-ratio/queue sweep of claim 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.agents.cli import Backend
from fleetlab.contracts.live import OperatorQueue
from fleetlab.contracts.rollout import Job, SupervisorSpec
from fleetlab.engine import manifests
from fleetlab.engine.episode import SupervisorParts
from fleetlab.engine.fleet import RunConfig, run
from fleetlab.engine.supervisors import make
from fleetlab.live.vlm.supervisor import VLMSettings, instructions, supervisor_env
from fleetlab.log.runfiles import read_identities, read_table
from fleetlab.policies.teleop.human import HumanSettings
from fleetlab.policies.teleop.measure import HUMAN, human_pass


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def gate(backends: list[Any]) -> None:
    from fleetlab.agents.batch import Batch
    from fleetlab.agents.cli import agent_root
    from fleetlab.agents.toy import toy_world

    world = toy_world(agent_root() / "supervisor-gate-toy")
    Batch.after_gate(
        supervisor_env(),
        backends,
        world.export,
        [world.run_dir],
        fleet=world.fleet,
        texts={"system": instructions()},
        rerun=False,
    )
    print("runtime gates passed: " + ", ".join(backends), flush=True)


def select(world: Path, out: Path, failures: int, successes: int, window: str) -> None:
    """Freeze one stratified seeded sample before making any calls."""
    if out.exists():
        raise FileExistsError("sample directory already exists; preserve the original selection")
    files = world / ("files" if (world / "files").is_dir() else "world")
    spec = json.loads((files / "spec.json").read_text())
    days = spec["windows"][window]
    pool = []
    for source in sorted((world / "runs").iterdir()):
        if not (source / "run.json").is_file():
            continue
        verdicts = {
            r["episode_id"]: r["true_success"] for r in read_table(source, "verdicts").to_pylist()
        }
        run_record = json.loads((source / "run.json").read_text())
        for identity in read_identities(source):
            if days["first"] <= identity.day <= days["last"]:
                pool.append(
                    (
                        identity,
                        bool(verdicts[identity.episode_id]),
                        source,
                        run_record["extra"]["episodes_per_robot_day"],
                    )
                )
    rng = random.Random(602)
    chosen = []
    for success, count in ((False, failures), (True, successes)):
        group = sorted((r for r in pool if r[1] == success), key=lambda r: r[0].episode_id)
        chosen.extend(rng.sample(group, count))
    chosen.sort(key=lambda r: r[0].episode_id)
    out.mkdir(parents=True)
    jobs = [Job(identity=r[0], record_video=True) for r in chosen]
    manifests.write(out / "sample.jsonl", jobs)
    if len({r[3] for r in chosen}) != 1:
        raise ValueError("source day lengths differ")
    dump(
        out / "selection.json",
        {
            "world": str(world),
            "window": window,
            "seed": 602,
            "failures": failures,
            "successes": successes,
            "episodes_per_robot_day": chosen[0][3],
            "sample_sha256": hashlib.sha256((out / "sample.jsonl").read_bytes()).hexdigest(),
            "episodes": {r[0].episode_id: {"success": r[1], "source": str(r[2])} for r in chosen},
        },
    )


def measurement_supervisor(name: str, params: Mapping[str, Any]) -> SupervisorParts:
    """Static, explicitly labelled queue observation for this small isolated probe.

    The full live-layer builder supplies a dynamic queue. Actual operator
    admission below still uses the existing two-operator reservation model.
    """
    return replace(make(name, params), operator_queue=lambda: OperatorQueue(0, 2))


def measure(
    sample: Path,
    out: Path,
    method: str,
    backend: Backend,
    workers: int,
    limit: int | None,
    effort: str,
    *,
    settings_file: Path | None = None,
    dry_start: bool = False,
) -> None:
    if (out / "measurement.json").exists():
        raise FileExistsError("completed measurement exists; use a new output directory")
    selection = json.loads((sample / "selection.json").read_text())
    jobs = manifests.read(sample / "sample.jsonl")
    if limit is not None:
        jobs = jobs[:limit]
    if method == "vlm":
        gate([backend])
        supplied = json.loads(settings_file.read_text()) if settings_file else {}
        settings = VLMSettings(
            **{
                "backend": backend,
                "effort": effort,
                **supplied,
                "stop_file": str(out / "route-stopped.json"),
            }
        )
        if settings.backend != backend:
            raise ValueError("settings backend must match the gated backend")
        spec = SupervisorSpec(name="vlm", params=asdict(settings))
    else:
        spec = SupervisorSpec(name="timeout", params={"after": 200, "decision_ms": 0})
    watched = [j.model_copy(update={"supervisor": spec}) for j in jobs]
    out.mkdir(parents=True, exist_ok=True)
    manifests.write(out / "manifest.jsonl", watched)
    settings_human = HumanSettings()
    cfg = RunConfig(
        out=str(out / "run"),
        workers=workers,
        batch_size=jobs[0].identity.batch_size,
        precision=jobs[0].identity.policy.precision,
        episodes_per_robot_day=selection["episodes_per_robot_day"],
        teleoperator=HUMAN,
        defer_continuations=True,
        continuation_steps=settings_human.active_budget,
        supervisors="fleetlab.live.vlm.measure:measurement_supervisor",
        chain_sample=True,
    )
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selection": selection,
        "supervisor": spec.model_dump(mode="json"),
        "queue_observation": "static: zero waiting, two free; admission is post-pass FIFO",
        "human": settings_human.model_dump(mode="json"),
    }
    dump(out / "started.json", metadata)
    if dry_start:
        # Exactly the same inputs/factory/config as a real run, before GPU loading.
        measurement_supervisor(spec.name, spec.params)
        dump(
            out / "dry-start.json",
            {
                "boundary": "immediately before engine.fleet.run; no GPU or model called",
                "config": asdict(cfg),
                "metadata": metadata,
            },
        )
        print(f"dry start reached fleet.run: {out}", flush=True)
        return
    completed = run(cfg, watched)
    metadata["chain_ok"] = completed.record.chain_ok
    dump(out / "supervised-run.json", completed.record.model_dump(mode="json"))
    if not completed.record.chain_ok:
        raise RuntimeError("supervised replay proof failed; measurement stopped")
    require_measurement_calls(out, read_table(out / "run", "decisions").to_pylist())
    if completed.layout.continuations():
        human_pass(cfg, settings_human)
    dump(out / "measurement.json", metadata)
    report(out)


def require_measurement_calls(out: Path, decisions: list[dict[str, Any]]) -> None:
    """Keep failure traces, but never score a run that silently lost its model route."""
    skipped = [d for d in decisions if d["call_status"] == "skipped"]
    if skipped:
        dump(
            out / "measurement-refused.json",
            {
                "reason": "model route skipped calls; this run is not a measurement",
                "skipped_calls": len(skipped),
                "episodes": sorted({d["episode_id"] for d in skipped}),
            },
        )
        raise RuntimeError("measurement refused: model route skipped calls")


def call_timing(decisions: list[dict[str, Any]], *, is_vlm: bool) -> dict[str, Any]:
    """Subscription wall time and labelled model estimates share the actual-call denominator."""
    calls = [d for d in decisions if is_vlm and d["call_status"] in ("ok", "failed")]
    times = [d["decision_ms"] / 1000 for d in calls]
    return {
        "model_call_attempts": len(calls),
        "decision_s_total": sum(times),
        "decision_s_median": float(np.median(times)) if times else None,
        "decision_s_p95": float(np.percentile(times, 95)) if times else None,
        "model_ms_production_estimate_total_s": sum(
            d["model_ms"] / 1000 for d in calls if d["model_ms"] is not None
        ),
        "model_ms_unavailable": sum(d["model_ms"] is None for d in calls),
        # All charged time remains visible, including skipped-call overhead and rule decisions.
        "all_decisions_s_total": sum(d["decision_ms"] / 1000 for d in decisions),
    }


def report(out: Path) -> dict[str, Any]:
    metadata = json.loads((out / "measurement.json").read_text())
    baseline = metadata["selection"]["episodes"]
    jobs = manifests.read(out / "manifest.jsonl")
    rows = read_table(out / "run", "job_outcomes").to_pylist()
    originals = {r["episode_id"]: r for r in rows if r["mode"] == "run"}
    continuations = {r["episode_id"]: r for r in rows if r["mode"] == "continue"}
    decisions = read_table(out / "run", "decisions").to_pylist()
    require_measurement_calls(out, decisions)
    metric_file = out / "run/human-metrics.json"
    human = json.loads(metric_file.read_text()) if metric_file.exists() else {}
    record_file = out / "supervised-run.json"
    detail_stats = (
        json.loads(record_file.read_text()).get("extra", {}).get("jobs_detail", [])
        if record_file.exists()
        else []
    )
    robot_jobs = [r for r in detail_stats if r.get("mode") == "run"]
    robot_timing_available = bool(robot_jobs) and all("robot_time_s" in r for r in robot_jobs)
    detail = []
    for job in jobs:
        eid = job.identity.episode_id
        original = originals[eid]
        follow = continuations.get(eid)
        placed = bool(follow["success"] if follow is not None else original["success"])
        detail.append(
            {
                "episode_id": eid,
                "baseline_success": baseline[eid]["success"],
                "autonomous_success": bool(original["success"]),
                "handed_over": follow is not None,
                "final_success": placed,
                "calls": [d for d in decisions if d["episode_id"] == eid],
            }
        )
    result = {
        "episodes": len(detail),
        "baseline_failures": sum(not d["baseline_success"] for d in detail),
        "baseline_successes": sum(d["baseline_success"] for d in detail),
        "rescued_without_person": sum(
            not d["baseline_success"] and d["autonomous_success"] for d in detail
        ),
        "rescued_with_person": sum(
            not d["baseline_success"] and d["handed_over"] and d["final_success"] for d in detail
        ),
        "handed_over": sum(d["handed_over"] for d in detail),
        "successful_attempts_called": sum(
            d["baseline_success"] and bool(d["calls"]) for d in detail
        ),
        "successes_broken": sum(d["baseline_success"] and not d["final_success"] for d in detail),
        "calls": len(decisions),  # historical field: all trace records, including rules
        **call_timing(decisions, is_vlm=metadata.get("supervisor", {}).get("name") == "vlm"),
        "robot_time_s": sum(r["robot_time_s"] for r in robot_jobs)
        if robot_timing_available
        else None,
        "robot_active_s": sum(r["robot_active_steps"] for r in robot_jobs) / 20
        if robot_timing_available
        else None,
        "retry_active_s": sum(r["retry_active_steps"] for r in robot_jobs) / 20
        if robot_timing_available
        else None,
        "retry_attempts": sum(r["retry_started_at"] is not None for r in robot_jobs)
        if robot_timing_available
        else None,
        "failed_calls": sum(d["call_status"] == "failed" for d in decisions),
        "skipped_calls": sum(d["call_status"] == "skipped" for d in decisions),
        "rule_decisions": sum(d["call_status"] == "rule" for d in decisions),
        "operator_used_s": sum(h["operator_used_s"] for h in human.values()),
        "queue_wait_s": sum(h["queue_steps"] / 20 for h in human.values()),
        "chain_ok": metadata["chain_ok"],
        "detail": detail,
    }
    dump(out / "report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "detail"}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gate")
    g.add_argument("backends", nargs="+", choices=["codex", "claude"])
    s = sub.add_parser("select")
    s.add_argument("world", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--failures", type=int, default=60)
    s.add_argument("--successes", type=int, default=20)
    s.add_argument("--window", default="scoring")
    r = sub.add_parser("run")
    r.add_argument("sample", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--method", choices=["vlm", "timeout"], default="vlm")
    r.add_argument("--backend", choices=["codex", "claude"], default="codex")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--limit", type=int)
    r.add_argument("--effort", default="xhigh")
    r.add_argument("--settings", type=Path)
    r.add_argument("--dry-start", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "gate":
        gate(args.backends)
    elif args.command == "select":
        select(args.world, args.out, args.failures, args.successes, args.window)
    elif args.command == "report":
        report(args.out)
    else:
        measure(
            args.sample,
            args.out,
            args.method,
            args.backend,
            args.workers,
            args.limit,
            args.effort,
            settings_file=args.settings,
            dry_start=args.dry_start,
        )


if __name__ == "__main__":
    main()
