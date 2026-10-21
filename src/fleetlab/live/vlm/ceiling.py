"""Three predeclared, no-model retry probes on the frozen D2 failures.

Run once under heavy-run with an outer 1,490-second execution timeout.
This measures a three-point trigger ceiling, not an oracle over every tick.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fleetlab.contracts.live import Decision, SupervisorContext, SupervisorDecision
from fleetlab.contracts.rollout import SupervisorSpec
from fleetlab.engine import manifests
from fleetlab.engine.episode import SupervisorParts
from fleetlab.engine.fleet import RunConfig, run
from fleetlab.engine.supervisors import AfterSteps
from fleetlab.live.vlm.deferred import check_sample
from fleetlab.live.vlm.measure import dump
from fleetlab.log.runfiles import read_table

STEPS = (120, 160, 200)


class ForcedRetry:
    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        eligible = Decision.retry in context.options
        return SupervisorDecision(
            Decision.retry if eligible else Decision.continue_,
            "fixed-step ceiling: retry available" if eligible else "fixed-step ceiling: ineligible",
            0.0,
            0.0,
        )


def factory(name: str, params: Mapping[str, Any]) -> SupervisorParts:
    if name != "retry-ceiling":
        raise ValueError("expected retry-ceiling")
    step = int(params["step"])
    return SupervisorParts(
        AfterSteps(0),
        ForcedRetry(),
        threshold=0.0,
        supports_snapshot_start=False,
        should_call=lambda c: c.active_step == step and not c.previous_decisions,
    )


def summarize(out: Path, expected: set[str]) -> dict[str, Any]:
    arms = {}
    rescued: set[str] = set()
    for step in STEPS:
        root = out / str(step)
        record = json.loads((root / "run.json").read_text())
        if not record["chain_ok"]:
            raise RuntimeError("retry ceiling replay proof failed")
        rows = [r for r in read_table(root, "job_outcomes").to_pylist() if r["mode"] == "run"]
        if len(rows) != len(expected) or {r["episode_id"] for r in rows} != expected:
            raise ValueError("incomplete retry ceiling arm")
        decisions = read_table(root, "decisions").to_pylist()
        retried = {r["episode_id"] for r in decisions if r["decision"] == "retry"}
        successes = {r["episode_id"] for r in rows if r["success"]}
        rescued |= successes & retried
        arms[str(step)] = {
            "attempts": len(rows),
            "eligible_retries": len(retried),
            "rescued": len(successes & retried),
            "rescued_ids": sorted(successes & retried),
            "success_without_retry_ids": sorted(successes - retried),
            "ineligible_ids": sorted(expected - retried),
            "chain_ok": True,
            "robot_time_s": sum(
                r["robot_time_s"] for r in record["extra"]["jobs_detail"] if r["mode"] == "run"
            ),
            "wall_s": record["extra"]["total_s"],
        }
    result = {"arms": arms, "best_of_three_rescued": len(rescued), "rescued_ids": sorted(rescued)}
    dump(out / "ceiling.json", result)
    return result


def measure(sample: Path, out: Path) -> None:
    check_sample(sample)
    if out.exists():
        raise FileExistsError("use a new ceiling directory")
    selection = json.loads((sample / "selection.json").read_text())
    jobs = [
        j.model_copy(update={"record_video": False})
        for j in manifests.read(sample / "sample.jsonl")
        if not selection["episodes"][j.identity.episode_id]["success"]
    ]
    dump(
        out / "protocol.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "selection": selection,
            "steps": STEPS,
            "call_ms": 0,
            "minimum_engine_hold_ticks": 1,
            "operators": 0,
            "eligibility": "unchanged engine guards; ineligible stays on policy; no later trigger",
            "ceiling_scope": "best of these three ticks with today's guarded primitive",
        },
    )
    for step in STEPS:
        watched = [
            j.model_copy(
                update={"supervisor": SupervisorSpec(name="retry-ceiling", params={"step": step})}
            )
            for j in jobs
        ]
        cfg = RunConfig(
            out=str(out / str(step)),
            workers=2,
            batch_size=jobs[0].identity.batch_size,
            precision=jobs[0].identity.policy.precision,
            episodes_per_robot_day=selection["episodes_per_robot_day"],
            supervisors="fleetlab.live.vlm.ceiling:factory",
            chain_sample=True,
        )
        completed = run(cfg, watched)
        if not completed.record.chain_ok:
            raise RuntimeError(f"replay proof failed at {step}")
    print(json.dumps(summarize(out, {j.identity.episode_id for j in jobs}), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    measure(args.sample, args.out)


if __name__ == "__main__":
    main()
