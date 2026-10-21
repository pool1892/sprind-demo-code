"""Capture autonomous intervention states by exact recorded-decision prefixes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SupervisorSpec
from fleetlab.engine import manifests
from fleetlab.engine.chain import Chain, compare
from fleetlab.engine.fleet import RunConfig, run
from fleetlab.policies.teleop.measure import dump, read_rows


def capture(
    root: Path, config: RunConfig, jobs: list[Job], decisions: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Continue policy before each autonomous intervention's primitive.

    A prefix includes earlier decisions and the current call's physical hold.
    Its chain must equal the real run up to the captured hand-over snapshot.
    No model is called, and this is a separate counterfactual run. A decision
    that never applied before completion is explicitly labelled unapplied.
    """
    traces = {
        j.key: sorted([d for d in decisions if d["episode_id"] == j.key], key=lambda d: d["step"])
        for j in jobs
    }
    result: dict[str, list[dict[str, Any]]] = {j.key: [] for j in jobs}
    for index in range(max((len(t) for t in traces.values()), default=0)):
        selected = [
            j
            for j in jobs
            if len(traces[j.key]) > index
            and traces[j.key][index]["decision"] not in {"hand_over", "continue"}
        ]
        if not selected:
            continue
        layout = RunLayout(root / "autonomous-counters" / str(index))
        cfg = replace(
            config,
            out=str(layout.root),
            supervisors="fleetlab.live.rules:make",
            chain_sample=False,
            defer_continuations=True,
        )
        prefixes = [
            j.model_copy(
                update={
                    "supervisor": SupervisorSpec(
                        name="prefix",
                        params={
                            "trace": [
                                {
                                    k: d[k]
                                    for k in (
                                        "step",
                                        "decision",
                                        "decision_ms",
                                        "model_ms",
                                        "hold_steps",
                                    )
                                }
                                for d in traces[j.key][: index + 1]
                            ]
                        },
                    ),
                }
            )
            for j in selected
        ]
        manifests.write(layout.root / "prefixes.jsonl", prefixes)
        if not (layout.root / "prefix-run.json").exists():
            prefix_run = run(cfg, prefixes)
            dump(layout.root / "prefix-run.json", prefix_run.record.model_dump(mode="json"))
        requests = layout.continuations()
        by_ep = {j.identity.episode_id: j for j in requests}
        counters = [
            Job(identity=j.identity, mode=JobMode.replay, from_snapshot=j.from_snapshot)
            for j in requests
        ]
        manifests.write(layout.root / "counters.jsonl", counters)
        if counters and not (layout.root / "counter-run.json").exists():
            counter_run = run(cfg, counters)
            dump(layout.root / "counter-run.json", counter_run.record.model_dump(mode="json"))
        outcomes = {r["job_key"]: r for r in read_rows(layout.table("job_outcomes"))}
        counter_by_ep = {j.identity.episode_id: j for j in counters}
        for job in selected:
            decision = traces[job.key][index]
            original = Chain.load(RunLayout(Path(config.out)).chain(job))
            prefix = Chain.load(layout.chain(job))
            n = len(prefix)
            clipped = replace(
                original,
                steps=original.steps[:n],
                state=original.state[:n],
                scene=original.scene[:n],
                wrist=original.wrist[:n],
                executed=original.executed[:n],
            )
            comparison = compare(prefix, clipped)
            if not comparison.identical:
                raise ValueError(f"autonomous prefix differs for {job.key}: {comparison}")
            request = by_ep.get(job.key)
            result[job.key].append(
                {
                    "decision_index": index,
                    "decision": decision["decision"],
                    "call_step": decision["step"],
                    "snapshot_step": request.from_snapshot.step
                    if request and request.from_snapshot
                    else None,
                    "without": bool(outcomes[counter_by_ep[job.key].key]["success"])
                    if request
                    else None,
                    "counter_key": counter_by_ep[job.key].key if request else None,
                    "applied": request is not None,
                    "prefix_chain_identical": True,
                    "root": str(layout.root.resolve()),
                }
            )
    return result
