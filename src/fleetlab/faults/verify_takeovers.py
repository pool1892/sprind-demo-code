"""Repeat a fixed sample of D2's moved-table takeover failures without policy inference.

Run under tools/heavy-run --light from a committed checkout. Stored source
snapshots and outcomes are read only. The saved reaction, queue, cadence,
fumble and every active fault are retained. This is a paired repair probe,
not a new measurement of the entire live supervisor.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from fleetlab.contracts.rollout import Job
from fleetlab.engine import manifests
from fleetlab.engine.chain import Chain, compare
from fleetlab.engine.episode import DirectoryStore, EpisodeContext, run_job
from fleetlab.faults import get
from fleetlab.policies.teleop.human import HumanTeleoperator
from fleetlab.sim.libero_sim import LiberoSimEnv
from fleetlab.sim.libero_success import FleetCheck


class ObservedHuman(HumanTeleoperator):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[Any] = []

    def act_one(self, observation: Any, env: Any) -> Any:
        action = super().act_one(observation, env)
        self.targets.append(action.copy())
        return action


def selected_jobs(source: Path, count: int) -> list[Job]:
    outcomes = {
        r["job_key"]: r for r in pq.read_table(source / "truth/job_outcomes.parquet").to_pylist()
    }
    metrics = json.loads((source / "human-metrics.json").read_text())
    required = {"backdrop_change", "furniture_shift", "policy_knob"}
    eligible = []
    for job in manifests.read(source / "admitted.jsonl"):
        kinds = {f.kind for f in job.effective_faults}
        if (
            required <= kinds <= required | {"camera_degradation"}
            and not outcomes[job.key]["success"]
            and metrics[job.key]["status"] == "fail:attempts_exhausted"
        ):
            eligible.append(job)
    if len(eligible) != 63:
        raise ValueError(f"expected the preserved 63-failure cohort, found {len(eligible)}")
    if not 20 <= count <= len(eligible):
        raise ValueError("sample must contain 20 to 63 failures")
    eligible.sort(key=lambda job: job.key)
    return [eligible[i] for i in np.linspace(0, len(eligible) - 1, count).round().astype(int)]


def probe(source: Path, out: Path, count: int = 20) -> dict[str, Any]:
    if out.exists():
        raise ValueError("use a new output directory")
    out.mkdir(parents=True)
    jobs = selected_jobs(source, count)
    manifests.write(out / "manifest.jsonl", jobs)
    rows = []
    started = time.monotonic()
    for job in jobs:
        ident = job.identity
        observer = ObservedHuman()

        def produce(policy: ObservedHuman = observer) -> ObservedHuman:
            return policy

        env = LiberoSimEnv(ident.suite, ident.task_index, ident.image_size)
        try:
            record = run_job(
                job,
                env,
                EpisodeContext(
                    fault_types=get,
                    make_check=FleetCheck,
                    teleoperator=produce,
                    teleoperator_name=HumanTeleoperator.spec.name,
                    store=DirectoryStore(source),
                ),
            )
        finally:
            env.close()
        reference = Chain.load(source / "truth/chain" / (job.key + ".npz"))
        comparison = compare(record.chain, reference)
        idle = record.stats["human"]["idle_steps"]
        assert comparison.first_difference == record.chain.first_step + idle + 1, comparison
        delta = np.array(
            [
                record.chain.executed[i + 1] - target
                for i, target in enumerate(observer.targets)
                if i >= idle
            ]
        )
        assert np.max(np.abs(delta)) <= 1e-7, (
            "executed targets must match the expert's world targets"
        )
        record.chain.save(out / "chain" / (job.key + ".npz"))
        row = {
            "job_key": job.key,
            "episode_id": ident.episode_id,
            "suite": ident.suite,
            "robot_id": ident.robot_id,
            "day": ident.day,
            "stored_success": False,
            "repaired_success": record.true_success,
            "human": record.stats["human"],
            "comparison": asdict(comparison),
            "active_command_rows": len(delta),
            "active_command_max_error_m": float(np.abs(delta[:, :3]).max()),
        }
        rows.append(row)
        print(json.dumps({"episode": ident.episode_id, "placed": record.true_success}), flush=True)
        (out / "result.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "cohort": 63,
                    "n": count,
                    "completed": len(rows),
                    "rescued": sum(r["repaired_success"] for r in rows),
                    "elapsed_seconds": time.monotonic() - started,
                    "rows": rows,
                },
                indent=1,
            )
            + "\n"
        )
    result: dict[str, Any] = json.loads((out / "result.json").read_text())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    probe(args.source, args.out, args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
