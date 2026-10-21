"""The policy adapter with the model loaded (`make check-model`, a heavy job).

The rows check in both precisions, the same chunk bits in another process,
the noise handed in and used, and a short run whose episodes are
the same in the worker layout and in one process. Needs the pinned
convolution database (`python -m fleetlab.engine.miopen pin`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sim_helpers import in_fresh_process

pytestmark = pytest.mark.model

BATCH = 4


def _setup(precision: str) -> Any:
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.engine import miopen
    from fleetlab.policies.xvla import XVLAPolicy

    miopen.activate()
    miopen.require(BATCH, precision)
    policy = XVLAPolicy()
    policy.load("cuda", precision)
    return policy


def rows(precision: str) -> dict[str, Any]:
    from fleetlab.policies.rows_check import rows_check
    from fleetlab.policies.xvla import sample_observations

    contents = sample_observations(16)
    policy = _setup(precision)
    return rows_check(policy, contents, BATCH)


@pytest.mark.parametrize("precision", ["fp32", "fp16_res32"])
def test_a_row_does_not_depend_on_its_place_or_its_mates(precision: str) -> None:
    result = in_fresh_process(rows, precision)
    assert result["independent"], result


def chunks(precision: str, seeds: tuple[int, ...]) -> list[bytes]:
    from fleetlab.contracts.policy import NoiseSample
    from fleetlab.policies.xvla import sample_observations

    contents = sample_observations(BATCH)
    policy = _setup(precision)
    out = []
    for seed in seeds:
        rng = np.random.Generator(np.random.PCG64(seed))
        noise = [
            NoiseSample(seed, 0, rng.standard_normal((30, 20), dtype=np.float32))
            for _ in range(BATCH)
        ]
        result = policy.act([c[0] for c in contents], [c[1] for c in contents], noise, policy.spec)
        out.append(b"".join(np.ascontiguousarray(r.actions).tobytes() for r in result))
    return out


def test_the_same_call_gives_the_same_bits_in_another_process() -> None:
    first = in_fresh_process(chunks, "fp16_res32", (1, 1, 2))
    second = in_fresh_process(chunks, "fp16_res32", (1,))
    assert first[0] == first[1] == second[0]  # the same noise: the same chunks
    assert first[0] != first[2]  # other noise: other chunks, so the noise handed in is used


def short_run(out: str, workers: int, in_process: bool) -> dict[str, Any]:
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401
    from fleetlab.engine import manifests, miopen, software
    from fleetlab.engine.fleet import RunConfig, run, run_in_process
    from fleetlab.policies.xvla import nominal_spec

    miopen.activate()
    soft = software.current(miopen.require(BATCH, "fp16_res32"))
    jobs = manifests.mix200(nominal_spec("fp16_res32"), BATCH, soft, starts=2, tasks=2)
    config = RunConfig(
        out=out,
        workers=workers,
        batch_size=BATCH,
        precision="fp16_res32",
        fault_registry="fleetlab.engine.standin_faults:STANDIN",
        chain_sample=False,
        allow_manifest_day_length=True,
    )
    result = (run_in_process if in_process else run)(config, jobs)
    return {"keys": [j.key for j in jobs], "successes": sum(s["success"] for s in result.summaries)}


def test_the_worker_layout_runs_the_same_episodes_as_one_process(tmp_path: Path) -> None:
    from fleetlab.contracts.rollout import RunLayout
    from fleetlab.engine.chain import Chain, compare

    fleet = in_fresh_process(short_run, str(tmp_path / "fleet"), 3, False)
    alone = in_fresh_process(short_run, str(tmp_path / "alone"), 1, True)
    for key in fleet["keys"]:
        a = Chain.load(RunLayout(tmp_path / "fleet").truth_dir / "chain" / f"{key}.npz")
        b = Chain.load(RunLayout(tmp_path / "alone").truth_dir / "chain" / f"{key}.npz")
        assert compare(a, b).identical, key
    assert fleet["successes"] == alone["successes"]
