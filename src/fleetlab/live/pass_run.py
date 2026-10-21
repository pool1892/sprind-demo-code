"""A resumable claim-4 pass: all as-run failures plus 50 sampled successes.

Cohort selection and scoring may read truth. The actual Supervisor receives
only the standard live context; labels and source paths never enter its spec.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.identity import noise_for
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SupervisorSpec
from fleetlab.engine import manifests
from fleetlab.engine.fleet import RunConfig, code_version, run
from fleetlab.live.counterfactual import capture
from fleetlab.live.rules import CostRule, QueuePolicy, gate_requests, start_tick
from fleetlab.log.writer import run_status
from fleetlab.policies.teleop.human import HumanSettings
from fleetlab.policies.teleop.measure import dump, read_rows, require_ready

HUMAN = "fleetlab.live.human:RecordedHuman"


def economic_proposal(rule: CostRule) -> SupervisorSpec | None:
    """First possible positive-cost offer; the real queue is priced at admission."""
    effective_ratio = rule.ratio * rule.avoidable_fraction
    if effective_ratio <= 1:
        return None  # Even certain failure cannot justify the strict operator cost.
    return SupervisorSpec(
        name="risk",
        params={
            "model": json.loads((Path(__file__).parent / "results/telemetry.json").read_text()),
            "horizon_steps": 100,
            "threshold": math.nextafter(1 / effective_ratio, 1.0),
            "decision": "hand_over",
            "decision_ms": 0.0,
        },
    )


def economic_tape(
    world: Path, root: Path, ratios: list[float], fraction: float = 1.0
) -> dict[str, Any]:
    """Apply every economic ratio to one recorded unassisted telemetry run.

    New physical continuations are still required at each distinct request point.
    No supervisor or estimator is rerun separately for each ratio.
    """
    from fleetlab.live.data import telemetry_rows
    from fleetlab.live.model import RiskModel, TelemetryRiskEstimator

    selected, population = cohort(world)
    model_path = Path(__file__).parent / "results/telemetry.json"
    meta = {
        "population": population,
        "model_sha256": digest(model_path),
        "ratios": ratios,
        "avoidable_fraction": fraction,
    }
    frozen(root / "economic-tape-config.json", meta)
    source_paths: dict[str, Path] = {}
    for source in sorted((world / "runs").glob("*/truth/job_outcomes.parquet")):
        ids = {r["episode_id"] for r in read_rows(source) if r["mode"] == "run"}
        for job in selected:
            if job.key in ids:
                source_paths[job.key] = source.parent.parent / "log/steps" / f"{job.key}.parquet"
    hashes = {ep: digest(path) for ep, path in source_paths.items()}
    frozen(root / "economic-telemetry-digests.json", hashes)
    tape_path = root / "economic-tape.json"
    if tape_path.exists():
        value: dict[str, Any] = json.loads(tape_path.read_text())
        return value
    model = RiskModel.load(model_path)
    points: dict[str, dict[str, Any]] = {f"{ratio:g}": {} for ratio in ratios}
    for job in selected:
        telemetry = telemetry_rows(source_paths[job.key])
        estimator = TelemetryRiskEstimator(model, 100)
        for tel in telemetry:
            risk = estimator.update(tel)
            # A final command's estimate cannot trigger after the attempt has ended.
            if tel.step + 1 >= len(telemetry):
                continue
            for ratio in ratios:
                if job.key not in points[f"{ratio:g}"] and risk * ratio * fraction > 1:
                    points[f"{ratio:g}"][job.key] = {"step": tel.step + 1, "risk": risk}
    result = {
        "source": "one as-run telemetry record per attempt",
        "points": points,
        "source_digests": hashes,
        "model_sha256": meta["model_sha256"],
    }
    dump(tape_path, result)
    return result


def frozen(path: Path, value: Any) -> None:
    """A resume cannot change the cohort, rule, software or human assumptions."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"resume configuration/provenance changed: {path}")
    else:
        dump(path, value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(world: Path, successes: int = 50, seed: int = 650) -> tuple[list[Job], dict[str, Any]]:
    ready = require_ready(world)
    files, spec = ready["files"], ready["spec"]
    window = spec["windows"]["scoring"]
    jobs = [
        j
        for j in manifests.read(files / "manifest.jsonl")
        if window["first"] <= j.identity.day <= window["last"]
    ]
    if not jobs or len({j.key for j in jobs}) != len(jobs):
        raise ValueError("empty or duplicate scoring manifest")
    ids = {j.key for j in jobs}
    outcomes: dict[str, dict[str, Any]] = {}
    provenance = {str(files / "manifest.jsonl"): digest(files / "manifest.jsonl")}
    for path in sorted((world / "runs").glob("*/truth/job_outcomes.parquet")):
        source_rows = [r for r in read_rows(path) if r["mode"] == "run" and r["episode_id"] in ids]
        if not source_rows:
            continue
        record_path = path.parent.parent / "run.json"
        if json.loads(record_path.read_text()).get("chain_ok") is not True:
            raise ValueError(f"as-run source lacks a passing chain sample: {record_path}")
        provenance[str(path)] = digest(path)
        provenance[str(record_path)] = digest(record_path)
        for row in source_rows:
            if row["episode_id"] in outcomes:
                raise ValueError("duplicate as-run scoring outcome")
            outcomes[row["episode_id"]] = row
    if set(outcomes) != ids:
        raise ValueError("missing as-run outcomes; no denominator is silently dropped")
    good = [j for j in jobs if outcomes[j.key]["success"]]
    n = min(successes, len(good))
    if n < 1 and good:
        raise ValueError("at least one success must be sampled")
    chosen = {
        j.key
        for j in sorted(good, key=lambda j: float(noise_for(j.identity, 0, (1,), sample=seed)[0]))[
            :n
        ]
    }
    selected = sorted(
        [j for j in jobs if not outcomes[j.key]["success"] or j.key in chosen], key=lambda j: j.key
    )
    meta = {
        "version": 1,
        "world": spec["world_id"],
        "world_root": str(world.resolve()),
        "window": window,
        "window_episodes": len(jobs),
        "window_successes": len(good),
        "window_failures": len(jobs) - len(good),
        "sampled_successes": n,
        "sample_seed": seed,
        "episodes_per_robot_day": spec["episodes_per_robot_day"],
        "identities": hash_json([j.identity.model_dump(mode="json") for j in selected]),
        "source_digests": provenance,
        "episodes": [
            {
                "episode_id": j.key,
                "robot": j.identity.robot_id,
                "as_run": bool(outcomes[j.key]["success"]),
                "weight": len(good) / n if j.key in chosen else 1.0,
            }
            for j in selected
        ],
    }
    return selected, meta


def wrap(
    job: Job, spec: SupervisorSpec, root: Path, factory: str, latency_route: str = "subscription"
) -> Job:
    return job.model_copy(
        update={
            "supervisor": SupervisorSpec(
                name="claim4",
                params={
                    "inner": spec.model_dump(mode="json"),
                    "factory": factory,
                    "latency_route": latency_route,
                    "events": str((root / "decisions" / f"{job.key}.json").resolve()),
                },
            ),
        }
    )


def execute(
    world: Path,
    root: Path,
    rule: str,
    settings: HumanSettings,
    *,
    after: int = 174,
    cost_ratio: float = 2.0,
    idle_cost: float = 0.0,
    avoidable_fraction: float = 1.0,
    reference: Path | None = None,
    supervisor: SupervisorSpec | None = None,
    factory: str = "fleetlab.engine.supervisors:make",
    config: RunConfig | None = None,
    latency_route: str = "subscription",
    max_wait_s: float | None = None,
    economic_points: dict[str, Any] | None = None,
    economic_tape_digest: str | None = None,
    dry_start: bool = False,
) -> dict[str, Any]:
    """One command per world/rule. Completed jobs and phases survive interruption."""
    if rule not in {"none", "timeout", "random", "decision", "supervisor", "holds"}:
        raise ValueError("unknown live rule")
    if after < 1:
        raise ValueError("timeout must follow at least one command")
    if latency_route not in {"subscription", "model"}:
        raise ValueError("latency route must be subscription or model")
    queue_policy = QueuePolicy(max_wait_s)
    economic = CostRule(cost_ratio, settings.service_s, idle_cost, avoidable_fraction)
    selected, population = cohort(world)
    frozen(root / "cohort.json", population)
    manifests.write(root / "cohort.jsonl", selected)
    reference_digest = None
    chosen: set[str] = set()
    hold_traces: dict[str, Any] = {}
    if rule == "holds":
        if reference is None:
            raise ValueError("hold control requires its source record")
        receipt = json.loads((reference / "record.json").read_text())
        if receipt["rule"] != "supervisor":
            raise ValueError("hold control is only defined for a supervisor pass")
        if receipt["population"] != population or receipt["config"]["code"] != code_version():
            raise ValueError("hold control requires identical cohort and measured code")
        reference_digest = digest(reference / "record.json")
        hold_traces = {r["episode_id"]: r["decisions"] for r in receipt["episodes"]}
    if rule == "random":
        if reference is None:
            raise ValueError("random requires a completed reference pass")
        receipt = json.loads((reference / "record.json").read_text())
        if receipt["population"] != population or receipt["settings"] != settings.model_dump(
            mode="json"
        ):
            raise ValueError("random reference must use the same cohort and human model")
        reference_digest = digest(reference / "record.json")
        count = sum(r["handover"] for r in receipt["episodes"])
        chosen = {
            j.key
            for j in sorted(
                selected, key=lambda j: float(noise_for(j.identity, 0, (1,), sample=651)[0])
            )[:count]
        }
    if rule == "supervisor" and supervisor is None:
        raise ValueError("supervisor requires a SupervisorSpec")
    if rule == "decision":
        supervisor = economic_proposal(economic)
    frozen(
        root / "pass.json",
        {
            "rule": rule,
            "after": after,
            "cost_ratio": cost_ratio,
            "idle_cost": idle_cost,
            "max_wait_s": max_wait_s,
            "economic_points": economic_points,
            "economic_tape_sha256": economic_tape_digest,
            "avoidable_fraction": avoidable_fraction,
            "reference_sha256": reference_digest,
            "settings": settings.model_dump(mode="json"),
            "factory": factory,
            "latency_route": latency_route,
            "supervisor": supervisor.model_dump(mode="json") if supervisor else None,
            "code": code_version(),
        },
    )
    jobs = []
    for j in selected:
        spec = supervisor
        if rule == "holds":
            spec = (
                SupervisorSpec(name="holds", params={"trace": hold_traces[j.key]})
                if any(d["hold_steps"] for d in hold_traces[j.key])
                else None
            )
        elif rule == "decision" and economic_points is not None:
            point = economic_points.get(j.key)
            spec = SupervisorSpec(name="economic_point", params=point) if point else None
        elif rule == "timeout":
            spec = SupervisorSpec(name="timeout", params={"after": after, "decision_ms": 0.0})
        elif rule == "random":
            # Random selects episodes without their outcomes, then calls at step 1.
            spec = (
                SupervisorSpec(name="timeout", params={"after": 0, "decision_ms": 0.0})
                if j.key in chosen
                else None
            )
        job_factory = (
            "fleetlab.engine.supervisors:make"
            if rule in {"timeout", "random", "decision", "holds"}
            else factory
        )
        if rule == "holds" or (rule == "decision" and economic_points is not None):
            job_factory = "fleetlab.live.rules:make"
        wrapped = wrap(j, spec, root, job_factory, latency_route) if spec else j
        if wrapped.supervisor is not None:
            wrapped.supervisor.params["queue"] = {
                "fleet": str((root / "fleet").resolve()),
                "job": j.model_dump(mode="json"),
                "settings": settings.model_dump(mode="json"),
                "per_day": population["episodes_per_robot_day"],
                "cost_ratio": cost_ratio,
                "max_wait_s": max_wait_s,
            }
        jobs.append(wrapped)
    jobs.sort(key=lambda j: (start_tick(j, population["episodes_per_robot_day"]), j.key))
    manifests.write(root / "proposals.jsonl", jobs)
    miopen = Path.home() / ".cache/robots-probes/engine/miopen"
    os.environ["FLEETLAB_MIOPEN_DIR"] = str(miopen)
    cfg = config or RunConfig(
        out=str(root / "fleet"),
        workers=8,
        batch_size=jobs[0].identity.batch_size,
        precision=jobs[0].identity.policy.precision,
        life=8,
    )
    cfg = replace(
        cfg,
        out=str(root / "fleet"),
        workers=1 if rule != "none" else cfg.workers,
        supervisors="fleetlab.live.rules:make",
        episodes_per_robot_day=population["episodes_per_robot_day"],
        defer_continuations=True,
        teleoperator=HUMAN,
        continuation_steps=settings.active_budget,
    )
    frozen(root / "config.json", asdict(cfg))
    if dry_start:
        receipt = {
            "boundary": "immediately before engine.fleet.run; no GPU job or model call",
            "config": asdict(cfg),
            "jobs": len(jobs),
            "supervisor": supervisor.model_dump(mode="json") if supervisor else None,
        }
        frozen(root / "dry-start.json", receipt)
        return receipt
    phases: dict[str, Any] = {}

    def phase(name: str, phase_jobs: list[Job], phase_cfg: RunConfig) -> None:
        path = root / f"{name}-run.json"
        if path.exists():
            phases[name] = json.loads(path.read_text())
            return
        if name == "proposals":
            # A completed attempt owns its follow-ups; an abandoned start owns none.
            layout = RunLayout(Path(phase_cfg.out))
            pending = set(run_status(layout.root, phase_jobs).pending)
            for job in phase_jobs:
                if job.key not in pending:
                    continue
                directory = layout.snapshots_dir / job.identity.episode_id
                for followup in directory.glob("*.continue.json"):
                    followup.unlink()
                for sample in (layout.truth_dir / "samples").glob(
                    f"{job.identity.episode_id}.*.json"
                ):
                    sample.unlink()
        result = run(phase_cfg, phase_jobs)
        if phase_cfg.chain_sample and result.record.chain_ok is not True:
            raise ValueError(f"{name}: failed supervised replay sample")
        phases[name] = result.record.model_dump(mode="json")
        dump(path, phases[name])

    phase("proposals", jobs, cfg)
    layout = RunLayout(Path(cfg.out))
    # Never replace raw request jobs with admitted ones: both manifests survive resume.
    request_path = root / "requests.jsonl"
    if not request_path.exists():
        manifests.write(request_path, layout.continuations())
    requests = manifests.read(request_path)
    decisions = read_rows(layout.table("decisions")) if layout.table("decisions").exists() else []
    risks: dict[str, float] = {}
    for job in requests:
        ep = job.identity.episode_id
        trace = [d for d in decisions if d["episode_id"] == ep]
        terminal = next((d for d in reversed(trace) if d["decision"] == "hand_over"), None)
        if terminal is None:
            raise ValueError(f"request has no canonical engine decision: {ep}")
        events_file = root / "decisions" / f"{ep}.json"
        events = json.loads(events_file.read_text()) if events_file.exists() else []
        event = next((e for e in reversed(events) if e["step"] == terminal["step"]), None)
        risks[ep] = (
            1.0 if terminal.get("call_status") == "rule" else event["risk"] if event else 1.0
        )
    admitted, denied, queue_events = gate_requests(
        requests, risks, settings, population["episodes_per_robot_day"], queue_policy
    )
    # Nonterminal refusals are handled at the call itself and stay supervised.
    # A terminal backstop has exhausted the robot's active budget: its as-policy
    # continuation still supplies the physical outcome, never the stopped proposal.
    if any(
        next(
            d
            for d in reversed(decisions)
            if d["episode_id"] == j.identity.episode_id and d["decision"] == "hand_over"
        ).get("call_status")
        != "rule"
        for j in denied
    ):
        raise ValueError("queue refusal escaped the retained supervisor")
    admitted = [
        j.model_copy(
            update={
                "teleoperator_params": {
                    **(j.teleoperator_params or {}),
                    "metrics_path": str(
                        (root / "human-metrics" / f"{j.identity.episode_id}.json").resolve()
                    ),
                }
            }
        )
        for j in admitted
    ]
    if rule == "random" and len(admitted) != len(chosen):
        raise ValueError("random did not achieve the reference hand-over count")
    frozen(root / "queue.json", queue_events)
    manifests.write(root / "admitted.jsonl", admitted)
    counter = [
        Job(identity=j.identity, mode=JobMode.replay, from_snapshot=j.from_snapshot)
        for j in requests
    ]
    manifests.write(root / "counterfactual.jsonl", counter)
    follow_cfg = replace(cfg, chain_sample=False)
    if counter:
        phase("counterfactual", counter, follow_cfg)
    if admitted:
        phase("human", admitted, follow_cfg)
    outcomes = {r["job_key"]: r for r in read_rows(layout.table("job_outcomes"))}
    human = {
        j.key: json.loads((root / "human-metrics" / f"{j.identity.episode_id}.json").read_text())
        for j in admitted
    }
    by_request = {j.identity.episode_id: j for j in requests}
    by_admitted = {j.identity.episode_id: j for j in admitted}
    by_counter = {j.identity.episode_id: j for j in counter}
    autonomous = capture(root, cfg, jobs, decisions) if rule == "supervisor" else {}
    details_by_key = {
        s["key"]: s
        for phase_record in phases.values()
        for s in phase_record.get("extra", {}).get("jobs_detail", [])
    }
    detail = []
    identities = {j.key: j.identity for j in selected}
    for base in population["episodes"]:
        ep = base["episode_id"]
        request, accepted = by_request.get(ep), by_admitted.get(ep)
        final = (
            outcomes[accepted.key]
            if accepted
            else outcomes[by_counter[ep].key]
            if request
            else outcomes[ep]
        )
        if rule == "none" and bool(final["success"]) != base["as_run"]:
            raise ValueError(f"no-supervisor replay changed the as-run outcome: {ep}")
        metric = human[accepted.key] if accepted else {}
        trace = sorted([d for d in decisions if d["episode_id"] == ep], key=lambda d: d["step"])
        interventions = autonomous.get(ep, [])
        if request:
            interventions = [
                *interventions,
                {
                    "decision_index": len(trace) - 1,
                    "decision": "hand_over",
                    "call_step": trace[-1]["step"],
                    "snapshot_step": request.from_snapshot.step if request.from_snapshot else None,
                    "without": bool(outcomes[by_counter[ep].key]["success"]),
                    "counter_key": by_counter[ep].key,
                    "applied": True,
                    "root": str(layout.root.resolve()),
                },
            ]
        interventions.sort(key=lambda d: d["decision_index"])
        applied = [d for d in interventions if d["applied"]]
        event_file = root / "decisions" / f"{ep}.json"
        calls = json.loads(event_file.read_text()) if event_file.exists() else []
        called_trace = [d for d in trace if d.get("call_status", "ok") != "rule"]
        if any(d.get("call_status") == "skipped" for d in called_trace):
            raise ValueError("skipped model route cannot be a measured supervisor pass")
        if [(c["step"], c["decision"]) for c in calls] != [
            (d["step"], d["decision"]) for d in called_trace
        ]:
            raise ValueError("recorded decision calls do not match the engine trace")
        detail.append(
            {
                **base,
                "success": bool(final["success"]),
                "handover": accepted is not None,
                "task": f"{identities[ep].suite}/{identities[ep].task_index}",
                "fault_types": sorted({f.kind for f in identities[ep].faults}) or ["healthy"],
                "request": request is not None or any(c.get("requested") for c in calls),
                "refused_requests": sum(bool(c.get("refused")) for c in calls),
                "calls": calls,
                "actual_calls": len(calls) if rule == "supervisor" else 0,
                "rule_decisions": len(trace) - len(called_trace),
                "snapshot_without": applied[0]["without"] if applied else None,
                "interventions": interventions,
                "handover_snapshot_without": bool(outcomes[by_counter[ep].key]["success"])
                if request
                else None,
                "snapshot_step": applied[0]["snapshot_step"] if applied else None,
                "continuation_key": accepted.key if accepted else None,
                "human": metric,
                "status": metric.get("status"),
                "decisions": [
                    {
                        k: d[k]
                        for k in (
                            "step",
                            "decision",
                            "decision_ms",
                            "model_ms",
                            "hold_steps",
                            "call_status",
                            "error",
                            "retry_budget_steps",
                        )
                    }
                    for d in trace
                ],
                "subscription_s": sum(d["decision_ms"] for d in calls) / 1000,
                "model_s": sum(d["model_ms"] for d in calls) / 1000
                if all(d["model_ms"] is not None for d in calls)
                else None,
                "decision_hold_s": sum(
                    min(d["hold_steps"], max(0, outcomes[ep]["steps"] - d["step"])) for d in trace
                )
                / 20,
                "queue_s": min(metric.get("idle_steps", 0), metric.get("queue_steps", 0)) / 20,
                "operator_s": metric.get("operator_used_s", 0),
                "reserved_s": metric.get("operator_reserved_s", 0),
                "robot_active_steps": details_by_key.get(ep, {}).get("robot_active_steps", 0)
                + (
                    details_by_key.get(final["job_key"], {}).get("robot_active_steps", 0)
                    if final["job_key"] != ep
                    else 0
                ),
                "retry_active_steps": details_by_key.get(ep, {}).get("retry_active_steps", 0),
                "robot_time_s": details_by_key.get(ep, {}).get("robot_time_s", 0)
                + (
                    details_by_key.get(final["job_key"], {}).get("robot_time_s", 0)
                    if final["job_key"] != ep
                    else 0
                ),
            }
        )
    record = {
        "version": 1,
        "population": population,
        "rule": rule,
        "after": after,
        "cost_ratio": cost_ratio,
        "latency_route": latency_route,
        "settings": settings.model_dump(mode="json"),
        "config": json.loads((root / "pass.json").read_text()),
        "queue_events": queue_events,
        "episodes": detail,
        "evidence": {
            name: {"chain_ok": rec["chain_ok"], "jobs_done": rec["extra"]["jobs_done"]}
            for name, rec in phases.items()
        },
        "limitations": [
            "Human settings are assumptions; D2 and furniture-shift recoveries are provisional.",
            "Proposal queue contains persisted requests in fixed fleet-start order; "
            "overlapping future calls are not known.",
            "Model-latency runs are simulated production estimates, not a production route.",
            "Snapshot comparison uses first applied intervention; per-action counters retained.",
            "Queue is simulated on the cohort; window estimates condition on that demand.",
        ],
    }
    dump(root / "record.json", record)
    if rule == "supervisor" and any(d["hold_steps"] > 0 for r in detail for d in r["decisions"]):
        # The parent receives its control receipt afterwards; its input must not
        # point at a file whose digest changes when that receipt is attached.
        source = root / "hold-source"
        frozen(source / "record.json", record)
        control = execute(
            world,
            root / "hold-control",
            "holds",
            settings,
            reference=source,
            config=config,
            latency_route=latency_route,
            cost_ratio=cost_ratio,
            max_wait_s=max_wait_s,
        )
        from fleetlab.live.scoring import summarize

        record["hold_control"] = {
            "source_record": str((source / "record.json").resolve()),
            "source_sha256": digest(source / "record.json"),
            "record": str((root / "hold-control/record.json").resolve()),
            "sha256": digest(root / "hold-control/record.json"),
            "net": summarize(control)["estimated_window_net_gain"],
            "episodes": [
                {"episode_id": r["episode_id"], "success": r["success"]}
                for r in control["episodes"]
            ],
        }
        dump(root / "record.json", record)
    return record
