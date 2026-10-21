"""CPU toy chain references for the intentional stage-5 supervised transition.

Regenerate from an immutable detached checkout, via heavy-run --light:
    uv run --locked python tests/test_supervisor_references.py NEW_JSON
Never updates a stored run or the smoke world's reference.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from engine_helpers import context, ident

from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import run_job
from fleetlab.engine.ladder import chain_digest
from fleetlab.engine.toy import ToyEnv, ToyTeleoperator

REFERENCE = Path(__file__).parent / "data/supervisor_rule_reference.json"


def chains(directory: Path) -> dict[str, Any]:
    result = {}
    for choice, number, budget, timing in itertools.product(
        ("continue", "retry", "stop", "hand_over"),
        (1, 2, 3, 5, 8),
        (30, 60, 100),
        ((4, 500), (10, 500), (10, 0), (20, 2000)),
    ):
        after, latency = timing
        identity = ident(number, max_steps=budget)
        spec = SupervisorSpec(
            name="timeout",
            params={
                "after": after,
                "decision": choice,
                "decision_ms": latency,
            },
        )
        ctx = context(teleop=ToyTeleoperator)
        record = run_job(Job(identity=identity, supervisor=spec), ToyEnv("toy", 0, 32), ctx)
        key = f"{choice}/{number}/{budget}/after{after}/ms{latency}"
        path = directory / "chain.npz"
        record.chain.save(path)
        result[key] = {
            "chain": chain_digest(path),
            "steps": record.n_steps,
            "success": record.true_success,
            "choices": [d.decision.value for d in record.decisions],
        }
        if record.decisions:
            replay = run_job(
                Job(identity=identity, mode=JobMode.scripted, decision_trace=record.decisions),
                ToyEnv("toy", 0, 32),
                context(),
            )
            assert compare(record.chain, replay.chain).identical, key
    return result


def test_rule_supervised_reference(tmp_path: Path) -> None:
    reference = json.loads(REFERENCE.read_text())
    assert chains(tmp_path) == reference["chains"]


if __name__ == "__main__":
    out = Path(sys.argv[1])
    if out.exists():
        raise FileExistsError("new output only; old references are records")
    with tempfile.TemporaryDirectory() as tmp:
        observed = chains(Path(tmp))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "format": "supervised-toy-chains/2",
                "purpose": "fresh retry budget, first tracked grasp pose, held check clock",
                "chains": observed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
