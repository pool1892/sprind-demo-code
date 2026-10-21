"""Offline runtime fixtures exercise the real VLM registry, errors and report arithmetic."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
from engine_helpers import context, ident
from test_supervisor_repairs import NeverDone, replay
from test_vlm_supervisor import observation, result

from fleetlab.contracts.env import Camera
from fleetlab.contracts.live import Decision, SupervisorCallFailure
from fleetlab.contracts.rollout import Job, SupervisorSpec
from fleetlab.engine import manifests, toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import run_job
from fleetlab.live.vlm import supervisor as module
from fleetlab.live.vlm.supervisor import ActiveRisk, VLMSupervisor


def fake_runtime(monkeypatch: pytest.MonkeyPatch, caller: Any, timestamps: list[float]) -> None:
    clock = iter(timestamps)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    monkeypatch.setattr(module, "supervisor_env", lambda: None)
    monkeypatch.setattr(module, "require_verified", lambda *a: None)
    monkeypatch.setattr(module, "run_agent", caller)


def test_prompt_exact_allowlist_and_primary_time_is_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def call(*args: Any, **kwargs: Any) -> Any:
        payload = json.loads(args[3])
        assert set(payload) == {
            "task",
            "risk",
            "risk_caveat",
            "active_step",
            "remaining_active_steps",
            "active_budget_note",
            "control_hz",
            "previous_decisions",
            "remaining_calls_after_this",
            "options",
            "operator_queue",
            "costs",
            "telemetry",
            "images",
        }
        assert set(payload["costs"]) == {
            "operator_cost_per_minute",
            "expected_operator_seconds",
            "robot_cost_per_minute",
            "units",
        }
        assert set(payload["operator_queue"]) == {"waiting", "free_operators"}
        assert all(set(image) == {"camera", "index", "step"} for image in payload["images"])
        assert set(payload["options"]) == {"continue", "stop", "hand_over"}
        assert payload["active_budget_note"] == (
            "At control_hz ticks per second, these are the remaining motion ticks before "
            "the attempt times out. Recover spends these ticks and decision holds do not, "
            "while retry gets its own nominal attempt budget once per attempt. "
            "Judge whether continued progress can finish within this budget."
        )
        return result("continue")

    fake_runtime(monkeypatch, call, [100, 102.125])
    answer = VLMSupervisor(caller=call).decide(
        observation(previous_decisions=(Decision.continue_,))
    )
    assert answer.decision_ms == 2125
    assert answer.model_ms == pytest.approx(
        900
    )  # distinguish provider proxy from subscription wall time


def test_real_registry_continue_reaches_terminal_rule_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def call(*args: Any, **kwargs: Any) -> Any:
        calls.append(json.loads(args[3]))
        return result("continue")

    fake_runtime(monkeypatch, call, [10, 10.1, 20, 20.1])
    ctx = context()
    ctx.make_check = NeverDone
    job = Job(
        identity=ident(3, max_steps=280),
        supervisor=SupervisorSpec(
            name="vlm",
            params={"stop_file": str(tmp_path / "route.json")},
        ),
    )
    record = run_job(job, toy.ToyEnv("toy", 0, 32), ctx)
    assert [p["active_step"] for p in calls] == [160, 230]
    assert [d.decision for d in record.decisions] == [
        Decision.continue_,
        Decision.continue_,
        Decision.hand_over,
    ]
    terminal = record.decisions[-1]
    assert (terminal.call_status, terminal.hold_steps, terminal.decision_ms) == ("rule", 0, 0)
    assert terminal.step == record.n_steps == 280 + record.stats["decision_hold_steps"]
    assert len(record.spawned) == 1
    assert compare(record.chain, replay(record).chain).identical
    assert len(calls) == 2  # neither fallback nor replay calls a model
    assert ctx.store is not None
    assert ctx.store.load(record.snapshots[-1]).remaining_active_steps == 0


def test_failure_trace_retains_earlier_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def call(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return result("continue") if len(calls) == 1 else result(ok=False, error="timeout", meta={})

    fake_runtime(monkeypatch, call, [10, 10.1, 20, 22.125, 30, 30.025])
    stop_file = tmp_path / "stopped.json"
    ctx = context()
    ctx.make_check = NeverDone
    settings: dict[str, Any] = {"first_step": 3, "cooldown_steps": 2, "stop_file": str(stop_file)}
    record = run_job(
        Job(
            identity=ident(3, max_steps=100),
            supervisor=SupervisorSpec(
                name="vlm",
                params=settings,
            ),
        ),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    assert len(record.decisions) == 2
    first, failure = record.decisions
    assert first.decision == Decision.continue_
    assert failure.error is not None
    assert failure.call_status == "failed" and "timeout" in failure.error
    assert failure.decision == Decision.hand_over
    assert failure.decision_ms == 2125 and failure.model_ms is None
    assert failure.hold_steps == 43
    assert (
        record.stats["robot_time_s"]
        == 43 / 20 + sum(d.decision_ms for d in record.decisions) / 1000
    )
    from log_toy import record as run_record

    from fleetlab.contracts.live import DecisionRecord
    from fleetlab.engine.records import LogSink
    from fleetlab.log import RunWriter
    from fleetlab.log.runfiles import read_table

    persisted = tmp_path / "record"
    with RunWriter(persisted, run_record(), episodes_per_robot_day=50) as writer:
        LogSink(persisted, 50).write(record, "fixture")
        writer.finalize()
    trace = [
        DecisionRecord.model_validate({k: v for k, v in row.items() if k != "episode_id"})
        for row in read_table(persisted, "decisions").to_pylist()
    ]
    assert trace == record.decisions
    assert compare(record.chain, replay(record, trace).chain).identical
    with pytest.raises(SupervisorCallFailure) as caught:
        VLMSupervisor(module.VLMSettings(**settings)).decide(observation())
    assert not caught.value.called and caught.value.decision_ms == pytest.approx(25)
    assert calls == [1, 1]  # disabled route does not invoke the subscription again
    assert "timeout" in stop_file.read_text()


def test_context_frames_and_risk_clock_skip_idle_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.contracts.live import SupervisorDecision
    from fleetlab.engine.episode import SupervisorParts
    from fleetlab.engine.supervisors import AfterSteps

    seen = []

    class Stop:
        def decide(self, c: Any) -> Any:
            seen.append(c)
            return SupervisorDecision(Decision.stop, "fixture", 1, 1)

    ctx = context()
    ctx.make_check = NeverDone
    ctx.supervisors = lambda *_: SupervisorParts(AfterSteps(16), Stop())
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(Job(identity=ident(), supervisor=SupervisorSpec(name="fixture")), env, ctx)
    c = seen[0]
    assert c.active_step == 17 and c.frame_steps == (0, 4, 8, 12, 16)
    for camera in Camera:
        assert len(c.recent_frames[camera]) == 5
    assert c.telemetry_window[-1].step == 16
    assert set(vars(c.telemetry_window[-1])) == {
        "step",
        "time_s",
        "joint_pos",
        "joint_vel",
        "gripper_qpos",
        "eef_pos",
        "eef_quat",
        "commanded_action",
        "chunk_index",
        "chunk_step",
    }
    risk = ActiveRisk()
    observations = []

    def observe(tel: Any) -> float:
        observations.append(tel)
        return 0.5

    monkeypatch.setattr(risk.estimator, "update", observe)
    for chunk in [0, -1, -1, 1]:
        risk.update(replace(c.telemetry_window[0], chunk_index=chunk))
    assert [(row.step, row.time_s) for row in observations] == [(0, 0), (1, 0.05)]
    assert record.decisions[0].step == 17


def test_report_counts_original_failures_and_operator_time_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.live.vlm import measure

    jobs = [Job(identity=ident(n)) for n in range(1, 7)]
    manifests.write(tmp_path / "manifest.jsonl", jobs)
    baseline = {job.identity.episode_id: {"success": n >= 3} for n, job in enumerate(jobs)}
    (tmp_path / "measurement.json").write_text(
        json.dumps(
            {
                "selection": {"episodes": baseline},
                "chain_ok": True,
                "supervisor": {"name": "vlm"},
            }
        )
    )
    rows = [
        {"episode_id": j.identity.episode_id, "mode": "run", "success": n in (0, 3)}
        for n, j in enumerate(jobs)
    ]
    rows += [
        {"episode_id": jobs[n].identity.episode_id, "mode": "continue", "success": success}
        for n, success in [(1, True), (4, False), (5, True)]
    ]
    decisions = [
        {
            "episode_id": jobs[n].identity.episode_id,
            "decision_ms": 2100.0,
            "model_ms": 900.0 if n == 1 else None,
            "call_status": "ok" if n == 1 else "failed",
        }
        for n in (1, 4)
    ]
    decisions += [
        {
            "episode_id": jobs[n].identity.episode_id,
            "decision_ms": 0,
            "model_ms": 0,
            "call_status": "rule",
        }
        for n in (0, 2)
    ]
    monkeypatch.setattr(
        measure,
        "read_table",
        lambda root, name: pa.Table.from_pylist(rows if name == "job_outcomes" else decisions),
    )
    (tmp_path / "run").mkdir()
    (tmp_path / "run/human-metrics.json").write_text(
        json.dumps(
            {
                str(n): {"operator_used_s": used, "operator_reserved_s": 90, "queue_steps": 20}
                for n, used in enumerate([3, 4, 5])
            }
        )
    )
    (tmp_path / "supervised-run.json").write_text(
        json.dumps(
            {
                "extra": {
                    "jobs_detail": [
                        {
                            "mode": "run",
                            "robot_time_s": 17.1,
                            "robot_active_steps": 300,
                            "retry_active_steps": 280,
                            "retry_started_at": 20,
                        },
                        {"mode": "scripted", "robot_time_s": 17.1},
                    ]
                }
            }
        )
    )
    report = measure.report(tmp_path)
    assert report["baseline_failures"] == report["baseline_successes"] == 3
    assert report["rescued_without_person"] == report["rescued_with_person"] == 1
    assert report["handed_over"] == 3 and report["successes_broken"] == 1
    assert report["successful_attempts_called"] == 1
    assert report["operator_used_s"] == 12 and report["queue_wait_s"] == 3
    assert report["robot_time_s"] == 17.1
    assert report["robot_active_s"] == 15 and report["retry_active_s"] == 14
    assert report["retry_attempts"] == 1
    assert report["decision_s_total"] == 4.2
    assert report["decision_s_median"] == report["decision_s_p95"] == 2.1
    assert report["model_call_attempts"] == report["rule_decisions"] == 2
    assert report["calls"] == 4
    assert report["model_ms_production_estimate_total_s"] == 0.9
    assert report["model_ms_unavailable"] == report["failed_calls"] == 1


@pytest.mark.parametrize(
    "phases",
    [[], {"ready_s": 0, "answer_s": float("inf")}, {"ready_s": float("nan"), "answer_s": 1}],
)
def test_malformed_runtime_timer_becomes_recordable_failure(
    phases: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def call(*args: Any, **kwargs: Any) -> Any:
        return result("continue", meta={"phases": phases})

    fake_runtime(monkeypatch, call, [1, 1.051])
    with pytest.raises(SupervisorCallFailure) as caught:
        VLMSupervisor(caller=call).decide(observation())
    assert caught.value.called
    assert caught.value.decision_ms == pytest.approx(51)
    assert caught.value.model_ms is None
    assert "timing" in str(caught.value)
