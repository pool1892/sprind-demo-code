"""The supervisor and placement repairs share one canonical policy-command stream.

CPU numerical cell, real observation/action/config hooks; no scene-physics claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from engine_helpers import ident
from fault_toys import spec_for
from test_faults_engine_commands import placement
from test_supervisor_repairs import AtFirstEligible, NeverDone

from fleetlab.contracts.live import Decision
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import EpisodeContext, MemoryStore, SupervisorParts, run_job
from fleetlab.engine.recovery import Recovery
from fleetlab.engine.supervisors import AfterSteps
from fleetlab.faults import get
from fleetlab.faults.condition_change import ConditionChange
from fleetlab.faults.furniture_shift import FurnitureShift


@pytest.fixture(autouse=True)
def numerical_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FurnitureShift, "cell_hook", lambda *args: None)
    monkeypatch.setattr(ConditionChange, "cell_hook", lambda *args: None)


@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change", "condition_change"])
@pytest.mark.parametrize(
    ("world_policy", "decision"),
    [
        (world, decision)
        for world in (False, True)
        for decision in Decision
        if not (world and decision is Decision.degrade)
    ],
)
def test_supervised_placement_stream_and_scripted_proof(
    kind: str, world_policy: bool, decision: Decision
) -> None:
    shift, offset = placement(kind)
    world_goal = np.array([0.125, 0.25, 0, 0.1, 0.2, 0.3, -1], np.float32)
    policy_goal = world_goal.copy()
    policy_goal[:3] += offset

    class Approach(toy.ToyTeleoperator):
        # This numerical expert emits world targets, like the real expert adapter.
        commands_in_world_frame = True

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def command(self, goal: Any) -> Any:
            cmd = goal.copy()
            cmd[6] = -1 if self.calls < 20 else 1
            self.calls += 1
            return cmd

        def act_one(self, observation: Any, env: Any) -> Any:
            return self.command(world_goal)

        def chunk(self, request: Any) -> Any:
            return self.command(policy_goal)[None, :]

    def context() -> EpisodeContext:
        return EpisodeContext(
            fault_types=get,
            source=Approach(),
            teleoperator=Approach,
            make_check=NeverDone,
            store=MemoryStore(),
            supervisors=lambda *_: SupervisorParts(
                AfterSteps(25),
                AtFirstEligible(decision, milliseconds=500),
                should_call=lambda c: not c.previous_decisions and decision in c.options,
            ),
        )

    identity = ident(
        policy=toy.TELEOP_SPEC if world_policy else toy.TOY_SPEC,
        faults=[shift],
        max_steps=100,
    )
    job = Job(identity=identity, supervisor=SupervisorSpec(name="fixture"))
    ctx = context()
    record = run_job(job, toy.ToyEnv("toy", 0, 32), ctx)
    assert len(record.decisions) == 1
    d = record.decisions[0]
    assert d.decision is decision and d.hold_steps == 10
    start = d.step + d.hold_steps
    before = record.steps.commanded[d.step - 1]
    assert before is not None
    np.testing.assert_array_equal(before[:3], policy_goal[:3])
    for tick in range(d.step, start):
        held = record.steps.commanded[tick]
        assert held is not None and held.tobytes() == before.tobytes()
    # Every action sees placement once, after its canonical command was recorded.
    for commanded, executed in zip(
        record.steps.commanded[:-1], record.chain.executed[1:], strict=True
    ):
        assert commanded is not None
        expected = commanded.copy()
        expected[:3] -= offset
        assert executed.tobytes() == expected.tobytes()
    if decision in (Decision.retry, Decision.recover):
        assert Recovery.commands_in_world_frame is False
        first = record.steps.commanded[start]
        assert first is not None
        np.testing.assert_array_equal(first[:3], before[:3])
        length = 70 if decision is Decision.retry else 34
        target = policy_goal[:3].copy()
        if decision is Decision.recover:
            target[2] += 0.08
        last = record.steps.commanded[start + length - 1]
        assert last is not None
        np.testing.assert_allclose(last[:3], target, atol=1e-7)
    if decision is Decision.retry:
        assert d.retry_budget_steps == 100
        assert record.n_steps == start + 100
        assert record.stats["retry_active_steps"] == 100
    if decision is Decision.hand_over:
        assert ctx.store is not None
        snapshot = ctx.store.load(record.snapshots[0])
        assert snapshot.remaining_active_steps == 100 - d.step
        assert snapshot.last_commanded is not None
        assert snapshot.last_commanded.tobytes() == before.tobytes()
    proof = run_job(
        Job(identity=identity, mode=JobMode.scripted, decision_trace=record.decisions),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert compare(record.chain, proof.chain).identical


@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change", "condition_change"])
def test_relative_recovery_uses_effective_mode_without_translating_deltas(kind: str) -> None:
    shift, _ = placement(kind)
    ft: Any = get("control_mode_change")
    switch = spec_for("control_mode_change", ft.calibration[0], "f02").model_copy(
        update={"params": {"value": "relative"}, "nominal": "absolute"}
    )

    class Source:
        def chunk(self, request: Any) -> Any:
            return np.array([[0.01, 0, 0, 0, 0, 0, -1]], np.float32)

    def context() -> EpisodeContext:
        return EpisodeContext(
            fault_types=get,
            source=Source(),
            make_check=NeverDone,
            supervisors=lambda *_: SupervisorParts(
                AfterSteps(2),
                AtFirstEligible(Decision.recover),
                should_call=lambda c: not c.previous_decisions,
            ),
        )

    identity = ident(faults=[shift, switch], max_steps=60)
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(
        Job(identity=identity, supervisor=SupervisorSpec(name="fixture")), env, context()
    )
    assert env.spec.control_mode is ControlMode.relative
    assert record.decisions[0].decision is Decision.recover
    for commanded, executed in zip(
        record.steps.commanded[:-1], record.chain.executed[1:], strict=True
    ):
        assert commanded is not None
        assert executed.tobytes() == commanded.tobytes()
    proof = run_job(
        Job(identity=identity, mode=JobMode.scripted, decision_trace=record.decisions),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert compare(record.chain, proof.chain).identical
