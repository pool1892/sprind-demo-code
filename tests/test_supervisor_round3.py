"""Bounded offline regressions for the second independent supervisor review."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from engine_helpers import context, ident
from log_toy import record as run_record
from test_supervisor_repairs import NeverDone, execute, watched_context
from test_vlm_repairs import fake_runtime
from test_vlm_supervisor import result

from fleetlab.contracts.live import Decision, SupervisorDecision
from fleetlab.contracts.log import EventKind
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import manifests, toy
from fleetlab.engine.episode import MemoryStore, SupervisorParts, run_job
from fleetlab.engine.records import LogSink
from fleetlab.engine.recovery import Pregrasp
from fleetlab.engine.supervisors import AfterSteps
from fleetlab.live.vlm import measure
from fleetlab.live.vlm.supervisor import VLMSettings
from fleetlab.log import RunWriter
from fleetlab.log.runfiles import read_table


@pytest.mark.parametrize("failure", ["timeout", "unoffered"])
def test_measurement_refuses_skipped_route_before_operator_pass_or_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Real registry/engine/persistence; only the subscription and fleet launcher are fixtures."""
    calls = []

    def call(*args: Any, **kwargs: Any) -> Any:
        calls.append(json.loads(args[3]))
        if len(calls) == 3:
            if failure == "timeout":
                return result(ok=False, error="timeout", meta={})
            assert "retry" not in calls[-1]["options"]
            return result("retry")
        return result("continue")

    fake_runtime(monkeypatch, call, [i / 100 for i in range(40)])
    monkeypatch.setattr(measure, "gate", lambda _: None)
    monkeypatch.setattr(
        measure, "VLMSettings", lambda **kw: VLMSettings(first_step=3, cooldown_steps=2, **kw)
    )
    out = tmp_path / "measurement"

    def run_fixture(cfg: Any, jobs: Any) -> Any:
        ctx = context()
        ctx.make_check = NeverDone
        with RunWriter(out / "run", run_record(), episodes_per_robot_day=50) as writer:
            for job in jobs:
                record = run_job(job, toy.ToyEnv("toy", 0, 32), ctx)
                LogSink(out / "run", 50).write(record, "fixture")
            writer.finalize()
        return SimpleNamespace(
            record=run_record().model_copy(update={"chain_ok": True}),
            layout=SimpleNamespace(continuations=lambda: [object()]),
        )

    def forbidden(*args: Any) -> None:
        pytest.fail("invalid measurements must stop before operator service")

    monkeypatch.setattr(measure, "run", run_fixture)
    monkeypatch.setattr(measure, "human_pass", forbidden)
    sample = tmp_path / "sample"
    sample.mkdir()
    jobs = [Job(identity=ident(n, max_steps=100)) for n in range(1, 9)]
    manifests.write(sample / "sample.jsonl", jobs)
    measure.dump(sample / "selection.json", {"episodes_per_robot_day": 50, "episodes": {}})
    with pytest.raises(RuntimeError, match="measurement refused: model route skipped calls"):
        measure.measure(sample, out, "vlm", "codex", 1, None, "xhigh")
    assert len(calls) == 3
    decisions = read_table(out / "run", "decisions").to_pylist()
    assert sum(d["call_status"] == "failed" for d in decisions) == 1
    assert sum(d["call_status"] == "skipped" for d in decisions) == 6
    assert sum(d["call_status"] == "ok" for d in decisions) == 2
    assert (out / "route-stopped.json").is_file()
    assert not (out / "measurement.json").exists() and not (out / "report.json").exists()
    assert json.loads((out / "measurement-refused.json").read_text())["skipped_calls"] == 6
    # An old completed record with this problem cannot bypass the guard through `report`.
    measure.dump(out / "measurement.json", json.loads((out / "started.json").read_text()))
    with pytest.raises(RuntimeError, match="measurement refused"):
        measure.report(out)
    assert not (out / "report.json").exists()


def test_timing_denominator_is_actual_model_calls_only() -> None:
    calls = [
        {"call_status": status, "decision_ms": ms, "model_ms": model}
        for status, ms, model in [
            ("ok", 20000, 4000),
            ("failed", 21000, None),
            ("ok", 22000, 6000),
            ("rule", 0, 0),
            ("rule", 0, 0),
            ("skipped", 2, None),
            ("skipped", 3, None),
        ]
    ]
    timings = measure.call_timing(calls, is_vlm=True)
    assert timings["model_call_attempts"] == 3
    assert timings["decision_s_total"] == 63
    assert timings["decision_s_median"] == 21
    assert timings["decision_s_p95"] == pytest.approx(21.9)
    assert timings["model_ms_production_estimate_total_s"] == 10
    assert timings["model_ms_unavailable"] == 1
    assert timings["all_decisions_s_total"] == pytest.approx(63.005)
    rule = measure.call_timing(calls[:1], is_vlm=False)
    assert rule["model_call_attempts"] == rule["decision_s_total"] == 0
    assert rule["decision_s_median"] is rule["decision_s_p95"] is None
    assert rule["all_decisions_s_total"] == 20


@pytest.mark.parametrize("height", [0.258, 0.322])
@pytest.mark.parametrize("low_before_close", [True, False])
def test_midair_close_cannot_be_a_retry_anchor(height: float, low_before_close: bool) -> None:
    approach = Pregrasp()
    low = np.array([0.1, 0.2, 0.1, 0, 0, 0, -1], np.float32)
    high = low.copy()
    high[2] += height
    if low_before_close:
        approach.observe(low, low[:3].astype(float))
    approach.observe(high, high[:3].astype(float))
    closed = high.copy()
    closed[6] = 1
    approach.observe(closed, closed[:3].astype(float))
    if not low_before_close:
        approach.observe(low, low[:3].astype(float))
    assert not approach.available(ControlMode.absolute, high, high[:3].astype(float))
    # A later close at the low point must not replace the rejected first anchor.
    low_closed = low.copy()
    low_closed[6] = 1
    approach.observe(low, low[:3].astype(float))
    approach.observe(low_closed, low[:3].astype(float))
    assert not approach.available(ControlMode.absolute, low, low[:3].astype(float))


def test_recorded_d2_midair_closes_are_refused_and_low_approach_is_retained() -> None:
    fixtures = json.loads((Path(__file__).parent / "data/supervisor_pregrasp_d2.json").read_text())
    assert {r["episode_id"] for r in fixtures} == {"e000661", "e001295", "e000148"}
    for row in fixtures:
        approach = Pregrasp()
        for observed in row["observations"]:
            approach.observe(
                np.array(observed["commanded_action"], np.float32), np.array(observed["eef_pos"])
            )
        assert (
            approach.available(
                ControlMode.absolute,
                np.array(row["last_command"]),
                np.array(row["position_at_call"]),
            )
            == row["expected_available"]
        ), row["episode_id"]
    assert [r["episode_id"] for r in fixtures if r["expected_available"]] == ["e000148"]


@pytest.mark.parametrize("mode", [JobMode.replay, JobMode.continue_])
@pytest.mark.parametrize("step", [1, 12, 40])
@pytest.mark.parametrize("change_gripper", [False, True])
def test_snapshot_gripper_events_match_actual_command_transitions(
    mode: JobMode, step: int, change_gripper: bool
) -> None:
    ctx = context(steps=[step], teleop=toy.ToyTeleoperator)
    ctx.make_check = NeverDone
    original = run_job(Job(identity=ident(3, max_steps=100)), toy.ToyEnv("toy", 0, 32), ctx)
    ref = original.snapshots[0]
    assert isinstance(ctx.store, MemoryStore)
    snapshot = ctx.store.load(ref)
    previous = snapshot.last_commanded
    assert previous is not None and previous[6] != 0
    command = previous.copy()
    command[6] *= -1 if change_gripper else 1
    # Fix the first command in either route, so both real transitions and non-events are tested.
    ctx.store.snapshots[ref.path] = replace(snapshot, queued_actions=[command])

    class Operator(toy.ToyTeleoperator):
        commands_in_world_frame = False  # reissues the snapshot's policy-frame command

        def act_one(self, observation: Any, env: Any) -> Any:
            return command

    ctx.teleoperator = Operator
    ctx.snapshot_steps = ()
    resumed = run_job(
        Job(identity=original.job.identity, mode=mode, from_snapshot=ref),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    kinds = [
        k for t, k, _ in resumed.events if t == step and k in (EventKind.grasp, EventKind.release)
    ]
    expected = [EventKind.grasp if command[6] > 0 else EventKind.release] if change_gripper else []
    assert kinds == expected


def test_snapshot_retains_and_restores_retry_used(monkeypatch: pytest.MonkeyPatch) -> None:
    first = execute(Decision.retry)
    d = first.decisions[0]
    ctx = watched_context(Decision.retry)
    ctx.snapshot_steps = (d.step + d.hold_steps + 75,)  # after the primitive
    original = execute(Decision.retry, ctx=ctx)
    ref = original.snapshots[0]
    assert ctx.store.load(ref).retry_used is True  # catches a lost field at save
    ctx.snapshot_steps = ()
    offered = []

    class Observe:
        def decide(self, c: Any) -> SupervisorDecision:
            offered.append(c.options)
            return SupervisorDecision(Decision.continue_, "fixture", 0, 0)

    # Isolate the once-only guard from the independent geometry guard on a new approach.
    monkeypatch.setattr(Pregrasp, "available", lambda *args: True)
    ctx.supervisors = lambda *_: SupervisorParts(
        AfterSteps(0), Observe(), should_call=lambda c: not c.previous_decisions
    )
    run_job(
        Job(
            identity=original.job.identity,
            mode=JobMode.replay,
            from_snapshot=ref,
            supervisor=SupervisorSpec(name="fixture"),
        ),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    assert offered and all(Decision.retry not in options for options in offered)
