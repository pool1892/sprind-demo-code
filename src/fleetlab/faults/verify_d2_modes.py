"""D2's stale-mode regression and absolute-policy control, through the real fleet."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from fleetlab import dose_response_stage3 as s3
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import JobMode, RunLayout
from fleetlab.engine import manifests
from fleetlab.engine.chain import Chain, compare
from fleetlab.engine.episode import effective_spec
from fleetlab.engine.fleet import RunConfig, run
from fleetlab.faults import get


def verify(source: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise ValueError("use a new output directory")
    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    setup = s3.xvla_setup()
    changed, controls = [], []
    source_by_id = {}
    old_outcomes = {}
    for name in ("w42-observed", "w42-later"):
        path = source / "runs" / name
        old_outcomes.update(
            {
                r["episode_id"]: r["true_success"]
                for r in pq.read_table(path / "truth/verdicts.parquet").to_pylist()
            }
        )
        for eid in old_outcomes:
            if (path / "truth/chain" / (eid + ".npz")).is_file():
                source_by_id[eid] = path
    for job in manifests.read(source / "files/manifest.jsonl"):
        ident = job.identity
        if not any(f.kind == "furniture_shift" for f in job.effective_faults):
            continue
        pairs = [(get(f.kind), get(f.kind).new_episode(f, ident)) for f in job.effective_faults]
        mode = effective_spec(ident, pairs).control_mode
        if mode != ident.policy.control_mode:
            changed.append(job)
        elif mode is ControlMode.absolute:
            controls.append(job)
    if len(changed) != 98 or any(old_outcomes[j.identity.episode_id] for j in changed):
        raise ValueError("the preserved 98-failure stale-mode cohort does not match")
    controls.sort(key=lambda j: j.key)
    controls = [controls[i] for i in np.linspace(0, len(controls) - 1, 8).round().astype(int)]
    selected = changed + controls
    # The run mode enables the engine's automatic replay sample. Replay-only
    # jobs intentionally have chain_ok=None, preventing recursive validation.
    jobs = [j.model_copy(update={"mode": JobMode.run}) for j in selected]
    config = RunConfig(out=str(out), workers=4, episodes_per_robot_day=14, **setup.config)
    out.mkdir(parents=True)
    manifests.write(out / "manifest.jsonl", jobs)
    started = time.monotonic()
    result = run(config, jobs)
    outcomes = {
        r["job_key"]: r["success"]
        for r in pq.read_table(out / "truth/job_outcomes.parquet").to_pylist()
    }
    rows = []
    for job in jobs:
        eid = job.identity.episode_id
        comparison = compare(
            Chain.load(RunLayout(out).chain(job)),
            Chain.load(source_by_id[eid] / "truth/chain" / (eid + ".npz")),
        )
        rows.append(
            {
                "episode_id": eid,
                "control": eid in {j.identity.episode_id for j in controls},
                "old_success": old_outcomes[eid],
                "new_success": outcomes[job.key],
                "chain_comparison": asdict(comparison),
            }
        )
    proof = {
        "source": str(source),
        "elapsed_seconds": time.monotonic() - started,
        "changed_mode_n": 98,
        "control_n": 8,
        "chain_ok": result.record.chain_ok,
        "changed_mode_outcomes_moved": sum(
            r["old_success"] != r["new_success"] for r in rows if not r["control"]
        ),
        "absolute_controls_identical": sum(
            r["chain_comparison"]["identical"] for r in rows if r["control"]
        ),
        "rows": rows,
    }
    (out / "proof.json").write_text(json.dumps(proof, indent=1) + "\n")
    if (
        proof["changed_mode_outcomes_moved"]
        or proof["absolute_controls_identical"] != 8
        or proof["chain_ok"] is not True
    ):
        raise ValueError("D2 mode or absolute-policy regression failed; see proof.json")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    verify(args.source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
