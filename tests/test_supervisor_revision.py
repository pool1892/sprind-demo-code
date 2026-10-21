"""One early assessment keeps a retry's budget and the terminal hand-over rule."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from engine_helpers import context, fault, ident
from test_supervisor_repairs import NeverDone
from test_vlm_supervisor import observation, result

from fleetlab.contracts.live import Decision
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import run_job
from fleetlab.live.vlm.supervisor import CallSchedule, VLMSettings, VLMSupervisor, make_vlm

PROFILE = Path(__file__).parents[1] / "src/fleetlab/live/vlm/profiles/one_early_look.json"


def test_declared_profile_calls_once_at_120_including_after_a_retry() -> None:
    settings = VLMSettings(**json.loads(PROFILE.read_text()))
    assert (settings.model, settings.effort, settings.first_step, settings.max_calls) == (
        "gpt-6-astra",
        "low",
        120,
        1,
    )
    schedule = CallSchedule(settings)
    assert not schedule(observation(active_step=119))
    assert schedule(observation(active_step=120))
    for step in range(121, 401):
        assert not schedule(observation(active_step=step, previous_decisions=(Decision.retry,)))


@pytest.mark.parametrize("single_initial", [True, False])
def test_only_assessment_offers_retry_but_default_final_reassessment_does_not(
    single_initial: bool,
) -> None:
    settings = VLMSettings(max_calls=1 if single_initial else 2)
    previous = () if single_initial else (Decision.continue_,)

    def caller(*args: Any, **kwargs: Any) -> Any:
        payload = json.loads(args[3])
        assert payload["remaining_calls_after_this"] == 0
        assert ("retry" in payload["options"]) is single_initial
        if not single_initial:
            assert set(payload["options"]) == {"continue", "hand_over", "stop"}
        return result("retry" if single_initial else "continue")

    answer = VLMSupervisor(settings, caller=caller).decide(observation(previous_decisions=previous))
    assert answer.decision is (Decision.retry if single_initial else Decision.continue_)


@pytest.mark.parametrize("choice", [Decision.retry, Decision.continue_])
def test_one_model_call_keeps_full_budget_terminal_handover_and_scripted_chain(
    choice: Decision,
) -> None:
    params = json.loads(PROFILE.read_text())
    settings = VLMSettings(**params)
    calls = []

    def caller(*args: Any, **kwargs: Any) -> Any:
        payload = json.loads(args[3])
        calls.append(payload)
        assert len(calls) == 1, "the revision must not interrupt a retry for reassessment"
        assert payload["active_step"] == 120 and payload["remaining_active_steps"] == 160
        assert payload["remaining_calls_after_this"] == 0
        assert choice.value in payload["options"]
        return result(choice.value)

    parts = replace(make_vlm(params), supervisor=VLMSupervisor(settings, caller=caller))
    ctx = context()
    ctx.make_check = NeverDone
    ctx.supervisors = lambda *_: parts
    identity = ident(3, max_steps=280, faults=[fault("toy_camera_shift", 3, pixels=8)])
    original = run_job(
        Job(identity=identity, supervisor=SupervisorSpec(name="vlm", params=params)),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    assert len(calls) == 1
    assert [d.decision for d in original.decisions] == [choice, Decision.hand_over]
    assert original.decisions[-1].call_status == "rule"
    assert original.decisions[-1].decision_ms == 0
    assert original.stats["robot_active_steps"] == (400 if choice is Decision.retry else 280)
    assert original.stats["retry_active_steps"] == (280 if choice is Decision.retry else 0)
    replay_context = context()
    replay_context.make_check = NeverDone
    replay = run_job(
        Job(identity=identity, mode=JobMode.scripted, decision_trace=original.decisions),
        toy.ToyEnv("toy", 0, 32),
        replay_context,
    )
    assert compare(original.chain, replay.chain).identical


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
def test_profile_model_and_effort_reach_the_subscription_call(model: str) -> None:
    params = json.loads(PROFILE.read_text())
    params["model"] = model
    settings = VLMSettings(**params)
    calls = []

    def caller(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        assert args[:2] == ("codex", model)
        assert kwargs["effort"] == "low"
        assert kwargs["service_tier"] == "fast"
        assert kwargs["auth"] == "subscription"
        return result("continue")

    VLMSupervisor(settings, caller=caller).decide(observation(active_step=120))
    assert len(calls) == 1


def test_engine_queue_callback_changes_the_models_input() -> None:
    from fleetlab.contracts.live import OperatorQueue

    queues = [
        OperatorQueue(waiting=0, free_operators=2),
        OperatorQueue(waiting=3, free_operators=0),
    ]
    payloads = []
    for queue in queues:
        params = json.loads(PROFILE.read_text())

        def caller(*args: Any, **kwargs: Any) -> Any:
            payloads.append(json.loads(args[3]))
            return result("stop")

        def current_queue(queue: OperatorQueue = queue) -> OperatorQueue:
            return queue

        parts = replace(
            make_vlm(params),
            supervisor=VLMSupervisor(VLMSettings(**params), caller=caller),
            operator_queue=current_queue,
        )
        ctx = context()
        ctx.make_check = NeverDone
        ctx.supervisors = lambda *_, parts=parts: parts
        run_job(
            Job(
                identity=ident(3, max_steps=280),
                supervisor=SupervisorSpec(name="vlm", params=params),
            ),
            toy.ToyEnv("toy", 0, 32),
            ctx,
        )
    assert [p["operator_queue"] for p in payloads] == [
        {"waiting": 0, "free_operators": 2},
        {"waiting": 3, "free_operators": 0},
    ]
    assert {k: v for k, v in payloads[0].items() if k != "operator_queue"} == {
        k: v for k, v in payloads[1].items() if k != "operator_queue"
    }
