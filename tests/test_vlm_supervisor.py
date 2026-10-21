"""Observation boundary, timing, bounded calls and physical/replay consequences."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from engine_helpers import context, ident
from PIL import Image

from fleetlab.agents.run import AgentResult
from fleetlab.contracts.env import Camera
from fleetlab.contracts.live import Decision, OperatorQueue, SupervisorContext, SupervisorDecision
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import SupervisorParts, _IdleCheck, run_job
from fleetlab.engine.recovery import primitive
from fleetlab.engine.supervisors import AfterSteps
from fleetlab.live.vlm.supervisor import CallSchedule, VLMSettings, VLMSupervisor, model_time


def observation(**changes: Any) -> SupervisorContext:
    frame = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    return replace(
        SupervisorContext(
            recent_frames={Camera.scene: [frame]},
            telemetry_window=[],
            risk=0.8,
            task_text="place the block",
            options=list(Decision),
            active_step=160,
            remaining_active_steps=120,
            frame_steps=(160,),
            operator_queue=OperatorQueue(2, 1),
        ),
        **changes,
    )


def result(decision: str = "retry", **changes: Any) -> AgentResult:
    return replace(
        AgentResult(
            ok=True,
            result={"decision": decision, "rationale": "The fingers closed beside the block."},
            usage={},
            seconds=1.2,
            meta={"phases": {"ready_s": 0.2, "answer_s": 1.1}},
        ),
        **changes,
    )


def test_model_receives_only_observed_context_and_rotated_images() -> None:
    seen = []
    c = observation()

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        seen.append((args, kwargs))
        payload = json.loads(args[3])
        assert payload["task"] == c.task_text
        assert payload["operator_queue"] == {"waiting": 2, "free_operators": 1}
        assert payload["remaining_active_steps"] == 120
        assert set(payload["options"]) == {d.value for d in Decision}
        assert "fault" not in payload and "identity" not in payload
        assert "tools" not in kwargs and kwargs["auth"] == "subscription"
        np.testing.assert_array_equal(
            np.asarray(Image.open(kwargs["images"][0])),
            c.recent_frames[Camera.scene][0][::-1, ::-1],
        )
        return result()

    answer = VLMSupervisor(caller=caller).decide(c)
    assert seen[0][0][:2] == ("codex", "gpt-6-astra")
    assert answer.decision is Decision.retry and answer.decision_ms > 0
    assert answer.model_ms == pytest.approx(900)
    assert VLMSettings(backend="claude").model == "claude-opus-5-5"


def test_failure_or_unavailable_decision_is_not_a_synthetic_success() -> None:
    with pytest.raises(RuntimeError, match="auth"):
        VLMSupervisor(caller=lambda *a, **k: result(ok=False, error="auth")).decide(observation())
    with pytest.raises(RuntimeError, match="unavailable"):
        VLMSupervisor(caller=lambda *a, **k: result()).decide(
            observation(previous_decisions=(Decision.retry,))
        )
    with pytest.raises(RuntimeError, match="timing"):
        model_time(result(meta={}))
    assert model_time(result(meta={"api_ms": 220}))[0] == 220


def test_schedule_counts_active_time_and_caps_calls() -> None:
    schedule = CallSchedule(VLMSettings())
    assert not schedule(observation(active_step=159))
    assert schedule(observation(active_step=160))
    assert not schedule(observation(active_step=230, previous_decisions=(Decision.retry,)))
    assert not schedule(observation(active_step=269, previous_decisions=(Decision.retry,)))
    assert schedule(observation(active_step=270, previous_decisions=(Decision.retry,)))
    assert not schedule(
        observation(active_step=900, previous_decisions=(Decision.retry, Decision.continue_))
    )


def test_failed_call_stops_engine_retries_without_more_subscription_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fleetlab.live.vlm.supervisor as module

    calls: list[int] = []

    def failed(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append(1)
        return result(ok=False, error="timeout")

    monkeypatch.setattr(module, "supervisor_env", lambda: None)
    monkeypatch.setattr(module, "require_verified", lambda *args: None)
    monkeypatch.setattr(module, "run_agent", failed)
    settings = VLMSettings(stop_file=str(tmp_path / "stopped.json"))
    with pytest.raises(RuntimeError, match="timeout"):
        VLMSupervisor(settings).decide(observation())
    with pytest.raises(RuntimeError, match="previously stopped"):
        VLMSupervisor(settings).decide(observation())
    assert calls == [1]


@pytest.mark.parametrize(
    "decision", [Decision.retry, Decision.recover, Decision.degrade, Decision.continue_]
)
def test_physical_interventions_replay_with_long_charged_holds(decision: Decision) -> None:
    class Once:
        def decide(self, c: SupervisorContext) -> SupervisorDecision:
            assert c.operator_queue == OperatorQueue(3, 2)
            assert c.remaining_active_steps == 100 - c.active_step
            return SupervisorDecision(decision, "observed test evidence", 1000, 700)

    ctx = context()
    ctx.supervisors = lambda name, params: SupervisorParts(
        AfterSteps(2),
        Once(),
        should_call=lambda c: not c.previous_decisions and decision in c.options,
        operator_queue=lambda: OperatorQueue(3, 2),
    )
    identity = ident(3, max_steps=100)
    original = run_job(
        Job(identity=identity, supervisor=SupervisorSpec(name="test")),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    replay = run_job(
        Job(identity=identity, mode=JobMode.scripted, decision_trace=original.decisions),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert compare(original.chain, replay.chain).identical
    assert original.decisions[0].rationale == "observed test evidence"
    assert original.stats["decision_hold_steps"] == 20
    assert original.n_steps <= 120 + original.decisions[0].step  # fresh budget on retry
    if decision in (Decision.retry, Decision.recover):
        action = original.steps.commanded[original.decisions[0].step + 20]
        assert action is not None and action[6] == -1
    if decision is Decision.degrade:
        # After the hold, replanning advances the chunk index at least every four ticks.
        chunks = [c for c in original.steps.chunk_index[23:] if c is not None]
        assert max(chunks.count(c) for c in set(chunks)) <= 4


def test_retry_actually_returns_to_pregrasp_and_relative_control_is_bounded() -> None:
    last = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], np.float32)
    before = np.array([-0.1, 0.1, 0.2, 0, 0, 0, -1], np.float32)
    absolute = primitive(Decision.retry, ControlMode.absolute, last, last[:3].astype(float), before)
    actions = [absolute.action(last[:3].astype(float)) for _ in range(len(absolute.targets))]
    np.testing.assert_array_equal(actions[-1], before)
    assert max(a[2] for a in actions) >= last[2] + 0.079
    with pytest.raises(ValueError, match="absolute pre-grasp"):
        primitive(Decision.retry, ControlMode.relative, last, last[:3].astype(float), before)
    relative = primitive(Decision.recover, ControlMode.relative, last, last[:3].astype(float), None)
    while relative:
        action = relative.action(last[:3].astype(float))
        assert np.max(np.abs(action[:3])) <= 0.300001
        assert not action[3:6].any() and action[6] == -1


def test_remaining_budget_excludes_prior_decision_holds() -> None:
    seen: list[SupervisorContext] = []

    class Continue:
        def decide(self, c: SupervisorContext) -> SupervisorDecision:
            seen.append(c)
            assert len(seen) <= 2, "repeated holds must not starve the policy of its next tick"
            return SupervisorDecision(Decision.continue_, "still moving", 1000, 700)

    ctx = context()
    ctx.supervisors = lambda name, params: SupervisorParts(
        AfterSteps(2), Continue(), should_call=lambda c: c.active_step in (3, 8)
    )
    run_job(
        Job(identity=ident(3, max_steps=12), supervisor=SupervisorSpec(name="test")),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    assert [c.active_step for c in seen] == [3, 8]
    assert [c.remaining_active_steps for c in seen] == [9, 4]
    assert [c.telemetry_window[-1].step for c in seen] == [2, 27]


def test_hold_clock_survives_snapshot_restore() -> None:
    class Check:
        def __init__(self) -> None:
            self.last = 0

        def update(self, state: Any, step: int) -> Any:
            self.last = step

        def snapshot(self) -> bytes:
            return b"underlying"

        def restore(self, blob: bytes) -> None:
            assert blob == b"underlying"

    first, second = _IdleCheck(Check()), _IdleCheck(Check())
    first.idle = True
    for t in range(1, 21):
        first.update(None, t)
    second.restore(first.snapshot())
    second.update(None, 21)
    assert second.check.last == 1


def test_rationale_roundtrips_on_truth_side(tmp_path: Path) -> None:
    from log_toy import record as run_record

    from fleetlab.engine.records import LogSink
    from fleetlab.log import RunWriter
    from fleetlab.log.runfiles import read_table

    job = Job(
        identity=ident(),
        supervisor=SupervisorSpec(
            name="timeout",
            params={
                "after": 2,
                "decision": "stop",
                "decision_ms": 100,
            },
        ),
    )
    record = run_job(job, toy.ToyEnv("toy", 0, 32), context())
    with RunWriter(tmp_path, run_record(), episodes_per_robot_day=50) as writer:
        LogSink(tmp_path, 50).write(record, "toy task")
        writer.finalize()
    assert read_table(tmp_path, "decisions").to_pylist()[0]["rationale"] == "timeout rule: stop"
