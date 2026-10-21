"""Frozen grid preflight: free runtime gates and explicit GPU/model boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts.env import Camera
from fleetlab.contracts.live import Decision, OperatorQueue, StepTelemetry, SupervisorContext
from fleetlab.engine import manifests, miopen, software
from fleetlab.engine.fleet import RunConfig, code_version
from fleetlab.live.pass_cli import load_supervisor
from fleetlab.live.pass_run import cohort, execute, frozen
from fleetlab.live.rules import RecordingSupervisor, make
from fleetlab.live.vlm.measure import gate
from fleetlab.live.vlm.supervisor import VLMSettings, VLMSupervisor
from fleetlab.policies.teleop.human import HumanSettings
from fleetlab.policies.teleop.measure import read_rows


class ModelBoundary(BaseException):
    """Stop before subscription invocation, outside the failed-call fallback."""


def model_boundary(supervisor: VLMSupervisor, context: SupervisorContext) -> dict[str, Any]:
    """Exercise request encoding with a labelled fixture, never invoke the provider."""
    receipt: dict[str, Any] = {}

    def stop(*args: Any, **kwargs: Any) -> Any:
        receipt.update(
            backend=args[0],
            model=args[1],
            effort=kwargs["effort"],
            service_tier=kwargs["service_tier"],
            images=len(kwargs["images"]),
            boundary="immediately before run_agent; intercepted, zero model calls",
            input_scope="synthetic startup fixture; real live observations require GPU execution",
        )
        raise ModelBoundary

    previous = supervisor.caller
    supervisor.caller = stop
    try:
        supervisor.decide(context)
    except ModelBoundary:
        pass
    finally:
        supervisor.caller = previous
    if not receipt:
        raise RuntimeError("model startup did not reach the call boundary")
    return receipt


def prepare(
    world: Path,
    out: Path,
    profile: Path,
    factory: str,
    workers: int,
    mode: str,
    dry_start: bool,
) -> dict[str, Any]:
    selected, population = cohort(world)
    spec = load_supervisor(profile)
    cfg = RunConfig(
        out="",
        workers=workers,
        batch_size=selected[0].identity.batch_size,
        precision=selected[0].identity.policy.precision,
        life=8,
    )
    hashes = set()
    for job in selected:
        digest = miopen.require(job.identity.batch_size, job.identity.policy.precision)
        if job.identity.software != software.current(digest):
            raise ValueError("cohort identity software/database does not match this checkout")
        hashes.add(digest)
    if spec.name == "vlm":
        params: dict[str, Any] = dict(spec.params)
        settings = VLMSettings(**params)
        gate([settings.backend])
    else:
        settings = None
    ids = {j.key for j in selected}
    rows = [
        r
        for path in (world / "runs").glob("*/truth/job_outcomes.parquet")
        for r in read_rows(path)
        if r["mode"] == "run" and r["episode_id"] in ids
    ]
    expected_calls = 3 * sum(r["steps"] > settings.first_step for r in rows) if settings else None
    passes = 17 if mode == "development" else 10
    receipt = {
        "code": code_version(),
        "population": population,
        "supervisor": spec.model_dump(mode="json"),
        "miopen_hashes": sorted(hashes),
        "full_cohort_passes": passes,
        "base_attempts": len(selected) * passes,
        "expected_model_calls": expected_calls,
        "model_call_basis": "three ratio passes; source attempts still active after the first look",
        "maximum_model_calls": 3 * len(selected) * settings.max_calls if settings else None,
        "memory": {
            "workers": workers,
            "estimated_resident_gib": workers * cfg.worker_gib + cfg.server_gib,
            "required_available_gib": workers * cfg.worker_gib + cfg.server_gib + cfg.mem_floor_gib,
            "scope_ram_limit_gib": 24,
            "gpu_start_limit_gib": cfg.gtt_start_limit_gib,
            "gpu_abort_gib": cfg.gtt_abort_gib,
        },
        "gpu_hours": {
            "base_estimate": passes / 10,
            "model_wait_estimate": expected_calls * 9.15 / 3600 if expected_calls else 0,
            "basis": "test-grid estimate: about one hour per ten passes; "
            "historical offline low median 9.15s",
            "additional": "prefix proofs, counterfactuals, hold stepping, "
            "operator continuations; unmeasured",
        },
    }
    frozen(out / "preflight.json", receipt)
    if dry_start:
        execute(
            world,
            out / "gpu-boundary",
            "none" if mode == "development" else "timeout",
            HumanSettings(),
            after=1,
            config=cfg,
            dry_start=True,
        )
        execute(
            world,
            out / "model-boundary",
            "supervisor",
            HumanSettings(),
            supervisor=spec,
            factory=factory,
            cost_ratio=1,
            config=cfg,
            dry_start=True,
        )
        watched = manifests.read(out / "model-boundary/proposals.jsonl")[0]
        assert watched.supervisor is not None
        parts = make(watched.supervisor.name, watched.supervisor.params)
        assert isinstance(parts.supervisor, RecordingSupervisor)
        inner = parts.supervisor.inner
        if isinstance(inner, VLMSupervisor):
            assert settings is not None
            tel = StepTelemetry(
                settings.first_step - 1,
                (settings.first_step - 1) / 20,
                np.zeros(7),
                np.zeros(7),
                np.zeros(2),
                np.zeros(3),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.zeros(7, dtype=np.float32),
                0,
                0,
            )
            context = SupervisorContext(
                {c: [np.zeros((8, 8, 3), dtype=np.uint8)] for c in Camera},
                [tel],
                0,
                "Synthetic startup fixture, not a world observation",
                list(Decision),
                OperatorQueue(0, 2),
                active_step=settings.first_step,
                remaining_active_steps=watched.identity.max_steps - settings.first_step,
            )
            if parts.should_call is not None and not parts.should_call(context):
                raise RuntimeError("configured first-look schedule refused its startup boundary")
            frozen(out / "model-boundary/call-boundary.json", model_boundary(inner, context))
        print("DRY START: reached fleet.run and model request boundaries; no GPU/model call.")
    else:
        print(
            f"Preflight passed: {receipt['base_attempts']} base attempts; "
            f"about {expected_calls} model calls."
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=("development", "test"), default="development")
    parser.add_argument("--dry-start", action="store_true")
    args = parser.parse_args()
    prepare(
        args.world, args.out, args.supervisor, args.factory, args.workers, args.mode, args.dry_start
    )


if __name__ == "__main__":
    main()
