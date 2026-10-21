"""Review regressions: budget ownership, recorded failures and complete replay traces."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from engine_helpers import context, ident
from log_toy import record as run_record

from fleetlab.contracts.env import SuccessVerdict
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.episode import EngineError, MemoryStore, run_job
from fleetlab.engine.records import LogSink
from fleetlab.log import RunWriter
from fleetlab.log._rows import RowError, decode_part, encode_part, to_arrow
from fleetlab.log.runfiles import open_run, read_table


class NeverDone(toy.ToyCheck):
    def update(self, state: Any, step: int) -> SuccessVerdict:
        return SuccessVerdict(False, False, "test: keep running")


@pytest.mark.parametrize("hold_ticks", [10, 100, 600])
@pytest.mark.parametrize("legacy", [False, True])
def test_snapshot_policy_keeps_remaining_active_budget(hold_ticks: int, legacy: bool) -> None:
    ctx = context(store=MemoryStore(), teleop=toy.ToyTeleoperator)
    ctx.make_check = NeverDone
    job = Job(
        identity=ident(3, max_steps=60),
        supervisor=SupervisorSpec(
            name="timeout", params={"after": 9, "decision_ms": hold_ticks * 50}
        ),
    )
    original = run_job(job, toy.ToyEnv("toy", 0, 32), ctx)
    ref = original.snapshots[0]
    assert ref.step == 10 + hold_ticks
    assert ctx.store is not None
    snapshot = ctx.store.load(ref)
    assert snapshot.remaining_active_steps == 50
    if legacy:
        assert isinstance(ctx.store, MemoryStore)
        ctx.store.snapshots[ref.path] = replace(snapshot, remaining_active_steps=None)
    resumed = run_job(
        Job(identity=job.identity, mode=JobMode.replay, from_snapshot=ref),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    assert resumed.n_steps - ref.step == 50
    operator = run_job(
        Job(identity=job.identity, mode=JobMode.continue_, from_snapshot=ref, max_steps=17),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )
    assert operator.n_steps - ref.step == 17
    with pytest.raises(ValueError, match="mid-attempt"):
        Job(
            identity=job.identity,
            mode=JobMode.scripted,
            from_snapshot=ref,
            decision_trace=original.decisions,
        )
    # Engine also protects callers that use pydantic's unvalidated model_copy.
    forged = job.model_copy(
        update={
            "mode": JobMode.scripted,
            "from_snapshot": ref,
            "supervisor": None,
            "decision_trace": original.decisions,
        }
    )
    with pytest.raises(EngineError, match="mid-attempt"):
        run_job(forged, toy.ToyEnv("toy", 0, 32), ctx)


def test_old_decision_table_and_parts_can_be_read_and_extended(tmp_path: Path) -> None:
    job = Job(
        identity=ident(3),
        supervisor=SupervisorSpec(name="timeout", params={"after": 2, "decision": "stop"}),
    )
    original = run_job(job, toy.ToyEnv("toy", 0, 32), context())
    with RunWriter(tmp_path, run_record(), episodes_per_robot_day=50) as writer:
        LogSink(tmp_path, 50).write(original, "toy")
        writer.finalize()
    table_path = tmp_path / "truth/decisions.parquet"
    before = pq.read_table(table_path).drop(
        ["rationale", "call_status", "error", "retry_budget_steps"]
    )
    pq.write_table(before, table_path)
    part_path = tmp_path / "truth/parts/e000003.json"
    part = decode_part(part_path.read_text())
    for row in part["tables"]["decisions"]:
        for key in ("rationale", "call_status", "error", "retry_budget_steps"):
            del row[key]
    part_path.write_text(encode_part(part))
    original_bytes = table_path.read_bytes()
    assert read_table(tmp_path, "decisions")["rationale"].to_pylist() == [""]
    with open_run(tmp_path) as reader:
        assert reader.con.execute(
            "SELECT rationale, call_status, error, retry_budget_steps FROM decisions"
        ).fetchall() == [("", "ok", None, None)]
    assert table_path.read_bytes() == original_bytes  # reading never migrates the record on disk
    with RunWriter(tmp_path, run_record(), episodes_per_robot_day=50) as writer:
        another = run_job(Job(identity=ident(4)), toy.ToyEnv("toy", 0, 32), context())
        LogSink(tmp_path, 50).write(another, "toy")
        writer.finalize()
    assert read_table(tmp_path, "decisions")["rationale"].to_pylist() == [""]
    assert set(read_table(tmp_path, "identities")["episode_id"].to_pylist()) == {
        "e000003",
        "e000004",
    }
    # Compatibility is narrow: it does not permit missing core fields or arbitrary columns.
    with pytest.raises(RowError):
        to_arrow("decisions", before.drop(["decision_ms"]))
    with pytest.raises(RowError):
        to_arrow("decisions", before.append_column("unexpected", pa.array([1])))


# These policies/timers are test fixtures, not evidence of model or D2 performance.
class AtFirstEligible:
    def __init__(self, decision: Any, milliseconds: float = 51) -> None:
        self.decision = decision
        self.milliseconds = milliseconds

    def decide(self, observation: Any) -> Any:
        from fleetlab.contracts.live import SupervisorDecision

        return SupervisorDecision(self.decision, "fixture", self.milliseconds, 1)


def watched_context(decision: Any, *, step: int = 2, milliseconds: float = 51) -> Any:
    from fleetlab.engine.episode import SupervisorParts
    from fleetlab.engine.supervisors import AfterSteps

    ctx = context()
    ctx.make_check = NeverDone
    ctx.supervisors = lambda *_: SupervisorParts(
        AfterSteps(step),
        AtFirstEligible(decision, milliseconds),
        should_call=lambda c: not c.previous_decisions and decision in c.options,
    )
    return ctx


def execute(decision: Any, *, steps: int = 100, ctx: Any = None) -> Any:
    return run_job(
        Job(identity=ident(3, max_steps=steps), supervisor=SupervisorSpec(name="fixture")),
        toy.ToyEnv("toy", 0, 32),
        ctx or watched_context(decision),
    )


def replay(record: Any, trace: Any = None) -> Any:
    ctx = context()
    ctx.make_check = NeverDone
    return run_job(
        Job(
            identity=record.job.identity,
            mode=JobMode.scripted,
            decision_trace=record.decisions if trace is None else trace,
        ),
        toy.ToyEnv("toy", 0, 32),
        ctx,
    )


def test_retry_budget_is_nominal_once_and_all_active_time_is_robot_time() -> None:
    import numpy as np

    from fleetlab.contracts.live import Decision
    from fleetlab.engine.chain import compare

    record = execute(Decision.retry)
    assert len(record.decisions) == 1
    decision = record.decisions[0]
    start = decision.step + decision.hold_steps
    assert decision.retry_budget_steps == 100
    assert decision.hold_steps == 2  # ceil(51 ms at 20 Hz), never truncation
    assert record.n_steps == start + 100
    assert record.stats["retry_active_steps"] == 100
    assert record.stats["robot_active_steps"] == decision.step + 100
    assert record.stats["robot_time_s"] == (decision.step + 100) / 20 + 0.051
    assert compare(record.chain, replay(record).chain).identical
    # The primitive returns to the FIRST open approach, not a later carry/close.
    first_close = next(
        i for i, a in enumerate(record.steps.commanded) if a is not None and a[6] > 0
    )
    anchor = record.steps.commanded[first_close - 1]
    np.testing.assert_allclose(record.steps.eef_pos[start + 70], anchor[:3], atol=0.003)
    assert record.steps.chunk_step[start + 70] == 0  # stale queued actions discarded
    assert record.steps.chunk_index[start + 70] > record.steps.chunk_index[decision.step - 1]
    # A second retry is rejected even when a forged trace asks for it.
    second = decision.model_copy(update={"step": start + 75})
    with pytest.raises(EngineError, match="retry is unavailable"):
        replay(record, [decision, second])
    with pytest.raises(EngineError, match="legacy retry trace"):
        replay(record, [decision.model_copy(update={"retry_budget_steps": None})])


def test_short_rotation_arc_and_first_tracked_approach() -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    from fleetlab.contracts.live import Decision
    from fleetlab.contracts.policy import ControlMode
    from fleetlab.engine.recovery import Pregrasp, primitive

    start = np.array([0.2, 0, 0.1, 0, 0, np.deg2rad(179), 1], np.float32)
    goal = np.array([0.25, 0, 0, 0, 0, np.deg2rad(-178), -1], np.float32)
    motion = primitive(Decision.retry, ControlMode.absolute, start, start[:3].astype(float), goal)
    rotations = Rotation.from_rotvec(np.stack(list(motion.targets))[:, 3:6])
    start_rotation = Rotation.from_rotvec(start[3:6])
    distances = (start_rotation.inv() * rotations).magnitude()
    assert np.max(distances) == pytest.approx(np.deg2rad(3), abs=1e-6)
    assert np.all(np.diff(distances) >= -1e-6)
    approach = Pregrasp()
    approach.observe(goal, goal[:3].astype(float))
    approach.observe(goal * np.array([1, 1, 1, 1, 1, 1, -1]), goal[:3].astype(float))
    other = start.copy()
    other[6] = -1
    approach.observe(other, other[:3].astype(float))
    approach.observe(start, start[:3].astype(float))
    assert approach.command is not None
    np.testing.assert_array_equal(approach.command, goal)
    assert approach.available(ControlMode.absolute, goal, goal[:3].astype(float))
    assert not approach.available(ControlMode.relative, goal, goal[:3].astype(float))
    assert not approach.available(ControlMode.absolute, goal, goal[:3].astype(float) + 0.1)
    far = goal.copy()
    far[0] += 0.4
    assert not approach.available(ControlMode.absolute, far, far[:3].astype(float))
    untracked = Pregrasp()
    untracked.observe(goal, goal[:3].astype(float))
    untracked.observe(start, goal[:3].astype(float) + 0.1)
    assert untracked.command is None  # no initial-pose or untracked-close fallback


@pytest.mark.parametrize("decision", ["recover", "degrade"])
def test_intervention_discards_the_old_chunk_and_recover_lifts(decision: str) -> None:
    import numpy as np

    from fleetlab.contracts.live import Decision
    from fleetlab.engine.chain import compare

    record = execute(Decision(decision))
    d = record.decisions[0]
    start = d.step + d.hold_steps
    resume = start + (34 if decision == "recover" else 0)
    assert record.steps.chunk_step[resume] == 0
    assert record.steps.chunk_index[resume] == record.steps.chunk_index[d.step - 1] + 1
    if decision == "recover":
        commands = np.stack(record.steps.commanded[start:resume])
        assert np.all(commands[:, 6] == -1)
        assert commands[-1, 2] == pytest.approx(record.steps.commanded[d.step - 1][2] + 0.08)
        assert record.steps.eef_pos[resume][2] > record.steps.eef_pos[start][2] + 0.07
    assert compare(record.chain, replay(record).chain).identical


@pytest.mark.parametrize("location", ["hold", "primitive", "after", "duplicate", "early_rule"])
def test_unconsumed_or_misplaced_trace_is_refused(location: str) -> None:
    from fleetlab.contracts.live import Decision

    record = execute(Decision.recover)
    first = record.decisions[0]
    at = {
        "hold": first.step + 1,
        "primitive": first.step + first.hold_steps + 1,
        "after": record.n_steps + 1,
        "duplicate": first.step,
        "early_rule": 1,
    }[location]
    extra = first.model_copy(update={"step": at, "decision": Decision.continue_})
    if location == "early_rule":
        extra = extra.model_copy(update={"call_status": "rule", "hold_steps": 0})
    with pytest.raises(EngineError, match="trace"):
        replay(record, [first, extra])


def test_success_check_clock_freezes_during_holds() -> None:
    from fleetlab.contracts.live import Decision

    class Deadline(NeverDone):
        def update(self, state: Any, step: int) -> SuccessVerdict:
            return SuccessVerdict(False, step >= 7, "deadline")

    ctx = watched_context(Decision.continue_, milliseconds=500)
    ctx.make_check = Deadline
    record = execute(Decision.continue_, ctx=ctx)
    assert record.n_steps == 17  # 7 active plus 10 held ticks
    assert record.ended_by == "check"


@pytest.mark.parametrize("field", ["decision_ms", "model_ms"])
@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_engine_rejects_invalid_call_times(field: str, bad: float) -> None:
    from fleetlab.contracts.live import Decision, SupervisorDecision
    from fleetlab.engine.episode import SupervisorParts
    from fleetlab.engine.supervisors import AfterSteps

    class Invalid:
        def decide(self, c: Any) -> Any:
            return SupervisorDecision(
                Decision.continue_,
                "fixture",
                bad if field == "decision_ms" else 1,
                bad if field == "model_ms" else 1,
            )

    ctx = context()
    ctx.supervisors = lambda *_: SupervisorParts(AfterSteps(2), Invalid())
    with pytest.raises(EngineError, match="timing"):
        execute(Decision.continue_, ctx=ctx)


def test_engine_rejects_unavailable_decision() -> None:
    from fleetlab.contracts.live import Decision
    from fleetlab.engine.episode import SupervisorParts
    from fleetlab.engine.supervisors import AfterSteps

    ctx = context()
    ctx.supervisors = lambda *_: SupervisorParts(AfterSteps(2), AtFirstEligible(Decision.retry))
    with pytest.raises(EngineError, match=r"^supervisor chose an unavailable decision$"):
        execute(Decision.retry, ctx=ctx)


@pytest.mark.parametrize("budget", [30, 60, 70])
def test_retry_not_offered_when_its_nominal_budget_cannot_finish_the_primitive(budget: int) -> None:
    from fleetlab.contracts.live import Decision

    record = execute(Decision.retry, steps=budget)
    assert record.decisions == []
    assert record.stats["retry_active_steps"] == 0
    assert record.n_steps == budget
