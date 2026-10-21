"""The estimator in the real supervisor hook: probability, decisions and exact replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from engine_helpers import context, ident
from log_toy import record as run_record
from test_live_model import constant_model

from fleetlab.contracts.live import Decision, StepTelemetry, SupervisorContext, SupervisorDecision
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import SupervisorParts, run_job
from fleetlab.engine.records import LogSink
from fleetlab.engine.supervisors import make
from fleetlab.live.data import build_dataset
from fleetlab.live.model import RiskModel, TelemetryRiskEstimator
from fleetlab.log import RunWriter


def specification(threshold: float = 0.5) -> SupervisorSpec:
    return SupervisorSpec(
        name="risk",
        params={
            "model": constant_model().to_dict(),
            "threshold": threshold,
            "decision": "stop",
            "decision_ms": 150,
        },
    )


def test_active_risk_rule_replays_exactly_from_decisions() -> None:
    job = Job(identity=ident(), supervisor=specification())
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(job, env, context())
    assert record.decisions[0].step == 1  # score at zero reaches next step's decision
    assert record.decisions[0].hold_steps == 3
    assert record.n_steps == 4
    repeated = run_job(job, env, context())
    replay = run_job(
        Job(identity=job.identity, mode=JobMode.scripted, decision_trace=record.decisions),
        env,
        context(),
    )
    assert compare(record.chain, repeated.chain).identical
    assert compare(record.chain, replay.chain).identical


def test_inactive_estimator_does_not_change_a_single_chain_bit() -> None:
    identity = ident()
    env = toy.ToyEnv("toy", 0, 32)
    plain = run_job(Job(identity=identity), env, context())
    watched = run_job(Job(identity=identity, supervisor=specification(0.9)), env, context())
    assert not watched.decisions
    assert compare(plain.chain, watched.chain).identical


def test_supervisor_receives_the_estimated_probability() -> None:
    seen: list[float] = []

    class Spy:
        def decide(self, c: SupervisorContext) -> SupervisorDecision:
            seen.append(c.risk)
            assert c.telemetry_window[-1].step == 0
            return SupervisorDecision(Decision.stop, "test", 0, 0)

    def factory(name: str, params: Any) -> SupervisorParts:
        parts = make(name, params)
        return SupervisorParts(parts.estimator, Spy(), parts.threshold)

    ctx = context()
    ctx.supervisors = factory
    run_job(Job(identity=ident(), supervisor=specification()), toy.ToyEnv("toy", 0, 32), ctx)
    assert seen == pytest.approx([0.7])


def test_path_models_require_pin_and_fixed_decision_is_explicit() -> None:
    with pytest.raises(ValueError, match="sha256"):
        make("risk", {"model_path": "anything"})
    with pytest.raises(KeyError, match="decision"):
        make("risk", {"model": constant_model().to_dict()})


@pytest.mark.parametrize("name", ["risk", "custom_estimator"])
def test_nonzero_snapshot_with_learned_risk_is_refused_before_reset(name: str) -> None:
    from fleetlab.contracts.rollout import SnapshotRef
    from fleetlab.engine.episode import EngineError

    class NoReset(toy.ToyEnv):
        def reset(self, identity: Any, cell_hooks: Any) -> Any:
            raise AssertionError("unsupported job must not reach simulation")

    identity = ident()
    job = Job(
        identity=identity,
        mode=JobMode.replay,
        supervisor=specification().model_copy(update={"name": name}),
        from_snapshot=SnapshotRef(episode_id=identity.episode_id, step=10, path="unused"),
    )
    ctx = context()
    ctx.supervisors = lambda name, params: make("risk", params)
    with pytest.raises(EngineError, match="history is not saved"):
        run_job(job, NoReset("toy", 0, 32), ctx)


@pytest.mark.parametrize("holds", [False, True])
def test_fitted_model_hook_matches_written_steps(tmp_path: Path, holds: bool) -> None:
    """Exercise engine telemetry construction, Parquet writing and offline extraction."""
    model = RiskModel.load(Path(__file__).parents[1] / "src/fleetlab/live/results/telemetry.json")
    seen: list[float] = []
    phases: list[int] = []

    class Capture:
        def __init__(self, estimator: TelemetryRiskEstimator) -> None:
            self.estimator = estimator

        def update(self, telemetry: StepTelemetry) -> float:
            probability = self.estimator.update(telemetry)
            seen.append(probability)
            phases.append(telemetry.chunk_step)
            return probability

    def factory(name: str, params: Any) -> SupervisorParts:
        parts = make(name, params)
        assert isinstance(parts.estimator, TelemetryRiskEstimator)
        return SupervisorParts(
            Capture(parts.estimator),
            parts.supervisor,
            parts.threshold,
            supports_snapshot_start=parts.supports_snapshot_start,
        )

    ctx = context()
    ctx.supervisors = factory
    spec = SupervisorSpec(
        name="risk",
        params={
            "model": model.to_dict(),
            "decision": "continue",
            "decision_ms": 150,
            "threshold": 1e-20 if holds else 1.0,
        },
    )
    job = Job(identity=ident(max_steps=280), supervisor=spec)
    record = run_job(job, toy.ToyEnv("toy", 0, 32), ctx)
    assert len(seen) == record.n_steps
    assert np.ptp(seen) > 1e-5  # Nonzero coefficients must affect actual probabilities.
    assert bool(record.decisions) is holds
    assert (-1 in phases) is holds  # The live hold sentinel must match Parquet nulls.
    run_dir = tmp_path / "run"
    with RunWriter(run_dir, run_record(), episodes_per_robot_day=50) as writer:
        LogSink(run_dir, 50).write(record, "toy task")
        writer.finalize()
    data = build_dataset("toy", [run_dir])
    assert data.steps.tolist() == list(range(record.n_steps))
    np.testing.assert_allclose(seen, model.probabilities(data.x)[:, -1], rtol=0, atol=1e-12)
    replay = run_job(
        Job(
            identity=job.identity,
            mode=JobMode.scripted if holds else JobMode.replay,
            decision_trace=record.decisions or None,
        ),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert compare(record.chain, replay.chain).identical
