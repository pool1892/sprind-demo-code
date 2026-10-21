"""Replay-rule regressions using real CPU toy policy answers and recorded chains."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fleetlab.contracts.env import PrivilegedState, SuccessVerdict
from fleetlab.contracts.rollout import Job, JobMode, RunRecord
from fleetlab.engine import software
from fleetlab.engine import toy as cell
from fleetlab.engine.episode import ChunkRequest, EpisodeContext, call_policy, run_job
from fleetlab.engine.records import LogSink
from fleetlab.log import RunWriter
from fleetlab.oracle import toy
from fleetlab.oracle.gains import Gains
from fleetlab.oracle.proof import compare_proofs
from fleetlab.oracle.run import RunRefused, manifest_sha
from fleetlab.oracle.table import TableBuild, build_table
from fleetlab.oracle.verification import (
    POLICY_CALL_RATE_BOUND,
    Execution,
    ReplayVerifier,
    event_ceiling,
    exact_interval,
    once_replay_risk,
)


class GripperCheck(cell.ToyCheck):
    """Toy marginal outcome: the first command must leave the gripper open."""

    end_step = 32

    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict:
        if step == 1:
            self.first_success = state.gripper_opening > 0
        done = step >= self.end_step
        success = done and self.first_success
        return SuccessVerdict(success, done, "placed" if success else "step limit")


class RowPolicy(cell.ToyPolicy):
    def act(self, *args: Any, **kwargs: Any) -> Any:
        chunks = super().act(*args, **kwargs)
        for row, chunk in enumerate(chunks):
            chunk.actions[0, 6] = -1.0 if row == 0 else 1.0
        return chunks


class MarginalPolicy(cell.ToyPolicy):
    """One first command is just below the toy cell's gripper sign threshold."""

    def act(self, *args: Any, **kwargs: Any) -> Any:
        chunks = super().act(*args, **kwargs)
        if len(self.calls) == 1:
            for chunk in chunks:
                chunk.actions[0, 6] = -3e-5
        return chunks


class Source:
    def __init__(
        self, delta: float = 0, *, row: int = 0, row_dependent: bool = False, flip: bool = False
    ) -> None:
        self.policy = RowPolicy() if row_dependent else MarginalPolicy()
        self.delta, self.row, self.flip = delta, row, flip

    def chunk(self, request: ChunkRequest) -> Any:
        answer = call_policy(self.policy, [replace(request, row=self.row)])[0]
        if len(self.policy.calls) == 1:
            answer[0, 0] += np.float32(self.delta)
            if self.flip:
                answer[0, 6] += np.float32(7e-5)  # one number; the marginal outcome flips
        return answer


class ChangedState(cell.ToyEnv):
    def step(self, action: Any) -> Any:
        result = super().step(action)
        self.data.eef[0] += 1e-5  # first differing state, same executed action
        return result


def execute(
    path: Path,
    job: Job,
    *,
    delta: float = 0,
    flip: bool = False,
    row: int = 0,
    row_dependent: bool = False,
    changed_state: bool = False,
    end_step: int = 32,
) -> Execution:
    path.mkdir(parents=True, exist_ok=True)
    text = job.model_dump_json() + "\n"
    (path / "jobs.jsonl").write_text(text)
    record = RunRecord(
        software=job.identity.software,
        batch_size=4,
        replay_rules=[],
        chain_ok=True,
        extra={
            "oracle": {
                "world_hash": None,
                "manifest": "jobs.jsonl",
                "manifest_sha256": manifest_sha(text),
                "jobs": 1,
            }
        },
    )
    if (path / "run.json").is_file():
        previous = RunRecord.model_validate_json((path / "run.json").read_text())
        record.extra["jobs_detail"] = previous.extra.get("jobs_detail", [])
    env = (ChangedState if changed_state else cell.ToyEnv)(
        job.identity.suite, job.identity.task_index, end_step
    )
    check = GripperCheck()
    check.end_step = end_step
    ctx = EpisodeContext(
        toy.TOY_ORACLE_FAULTS.get,
        lambda: check,
        source=Source(delta, row=row, row_dependent=row_dependent, flip=flip),
        noise_shape=(8, 3),
        noise_dtype="float32",
    )
    produced = run_job(job, env, ctx)
    with RunWriter(path, record, episodes_per_robot_day=4) as writer:
        detail = LogSink(path, 4).write(produced, env.task_description)
        writer.finalize()
        earlier: Any = record.extra.get("jobs_detail", [])
        writer.update_record(
            extra={"jobs_detail": [d for d in earlier if d["key"] != job.key] + [detail]}
        )
    env.close()
    return Execution(path, job, produced.true_success)


@pytest.fixture
def job() -> Job:
    return toy.jobs(software.current("none"))[24]  # scoring r01, one decoy


def verifier(tmp: Path, job: Job, third: Any) -> ReplayVerifier:
    return ReplayVerifier(tmp / "verification", {job.identity.episode_id: "scoring"}, None, third)


@pytest.mark.parametrize("odd", ["run", "replay"])
def test_one_policy_answer_is_settled_and_the_majority_outcome_is_used(
    tmp_path: Path, job: Job, odd: str
) -> None:
    replay = job.model_copy(update={"mode": JobMode.replay})
    left = execute(tmp_path / "as-run", job, flip=odd == "run")
    before = {
        p.relative_to(left.run): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in left.run.rglob("*")
        if p.is_file()
    }
    right = execute(tmp_path / "second", replay, delta=7e-5 if odd == "replay" else 0)
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    ok, details = compare_proofs(right.run, [replay], [left.run], v.proof)
    assert ok and not details[0]["identical"]
    assert details[0]["odd"] == ("left" if odd == "run" else "right")
    assert details[0]["outcome_differs"] == (odd == "run")
    assert v.canonical(left) is True
    assert v.canonical(right) is True
    v.save()
    restored = verifier(tmp_path, job, None)
    restored.validate()
    removed = job.model_copy(update={"mode": JobMode.replay, "remove": ["f03"]})
    after = execute(tmp_path / "removed", removed)
    built = build_table([job.identity], (), [left.run, right.run, after.run], verifier=restored)
    # Without majority replacement the run-odd case falsely reports one rescue.
    assert built.table.success(job.identity.episode_id, job.identity.faults) is True
    assert Gains(built.table, boot=0).fault_effect("f03").net == 0
    assert {
        p.relative_to(left.run): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in left.run.rglob("*")
        if p.is_file()
    } == before
    window = next(w for w in restored.summary()["windows"] if w["events"])
    assert window["attempts_executed_twice"] == 1
    assert window["compared_calls"] == 7 and window["event_ceiling"] == 1
    assert window["events"][0]["odd_side"] == ("as_run" if odd == "run" else "replay")


def test_changed_rows_use_the_same_third_execution_rule(tmp_path: Path, job: Job) -> None:
    changed = job.model_copy(update={"mode": JobMode.replay, "remove": ["f03"]})
    left = execute(tmp_path / "first", changed)
    right = execute(tmp_path / "second", changed, flip=True)
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    built = build_table([job.identity], (), [left.run, right.run], verifier=v)
    assert built.table.success(job.identity.episode_id, ()) is True
    assert (
        next(w for w in v.summary()["windows"] if w["events"])["events"][0]["odd_side"] == "replay"
    )


def test_row_dependence_can_be_an_event_when_both_scratch_jobs_run_alone(
    tmp_path: Path, job: Job
) -> None:
    from fleetlab.oracle.run import execute_third

    # Both actual engine scratch jobs run alone in row zero. The original's
    # other batch row is not recreated; the resulting 3:1 split is an event.
    left = execute(tmp_path / "as-run", job, row=1, row_dependent=True)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), row_dependent=True
    )
    overrides = {
        **toy.engine_overrides(),
        "policy_factory": "test_oracle_verification:RowPolicy",
        "check_factory": "test_oracle_verification:GripperCheck",
    }
    v = verifier(tmp_path, job, lambda folder, j: execute_third(folder, j, 4, None, 1, overrides))
    record = v.settle(left, right)
    assert record["odd"] == "left"
    assert Job.model_validate(record["third"]["job"]).mode is JobMode.run
    assert Job.model_validate(record["fourth"]["job"]).mode is JobMode.replay
    window = v.summary()["windows"][0]
    assert len(window["events"]) == window["event_ceiling"] == 1


def test_changed_state_is_not_an_event(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), changed_state=True
    )
    v = verifier(
        tmp_path, job, lambda *args: pytest.fail("a state difference must not execute a third")
    )
    with pytest.raises(RunRefused, match="first difference is state"):
        v.settle(left, right)


def test_three_executions_pairwise_different_refuse(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j, delta=1.4e-4))
    with pytest.raises(RunRefused, match="matches neither"):
        v.settle(left, right)


def test_third_execution_cannot_be_made_refuses(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    v = verifier(tmp_path, job, None)
    with pytest.raises(RunRefused, match="third execution required"):
        v.settle(left, right)


def test_more_one_off_events_than_ceiling_refuse(tmp_path: Path, job: Job) -> None:
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    for number in range(3):
        identity = job.identity.model_copy(update={"episode_id": f"e{number + 1:06}"})
        v.windows[identity.episode_id] = "scoring"
        left = execute(tmp_path / "as-run", Job(identity=identity))
        right = execute(
            tmp_path / "second", Job(identity=identity, mode=JobMode.replay), delta=7e-5
        )
        v.settle(left, right)
    with pytest.raises(RunRefused, match="events exceed ceiling"):
        v.save()
    import json

    saved = json.loads(v.path.read_text())
    assert len(saved["comparisons"]) == 3
    assert all(r["third"] and r["fourth"] for r in saved["comparisons"].values())


def test_interval_and_fixed_ceiling_do_not_use_observed_deviations() -> None:
    assert exact_interval(0, 100) == pytest.approx([0, 36216.692645])
    assert event_ceiling(1000) == 3
    assert [event_ceiling(n) for n in (3300, 3400, 3500)] == [5, 5, 6]
    assert exact_interval(0, 0) == [None, None]


def test_ceiling_is_checked_after_the_named_window_is_compared(tmp_path: Path, job: Job) -> None:
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    v.required = {"scoring": [f"e{n + 1:06}" for n in range(4)]}
    for number in range(3):
        identity = job.identity.model_copy(update={"episode_id": f"e{number + 1:06}"})
        v.windows[identity.episode_id] = "scoring"
        left = execute(tmp_path / "as-run", Job(identity=identity))
        right = execute(
            tmp_path / "second", Job(identity=identity, mode=JobMode.replay), delta=7e-5
        )
        v.settle(left, right)
    v.save()  # a partial proof sample is not the named-window denominator
    assert v.summary()["windows"][0]["ceiling_pending"]
    identity = job.identity.model_copy(update={"episode_id": "e000004"})
    v.windows[identity.episode_id] = "scoring"
    v.settle(
        execute(tmp_path / "as-run", Job(identity=identity)),
        execute(tmp_path / "second", Job(identity=identity, mode=JobMode.replay)),
    )
    with pytest.raises(RunRefused, match="events exceed ceiling"):
        v.save()


def test_two_unmodified_replays_do_not_prove_the_as_run_baseline(tmp_path: Path, job: Job) -> None:
    replay = job.model_copy(update={"mode": JobMode.replay})
    v = verifier(tmp_path, job, None)
    v.required = {"scoring": [job.identity.episode_id]}
    v.settle(execute(tmp_path / "first", replay), execute(tmp_path / "second", replay))
    coverage = v.summary()["coverage"][0]
    assert coverage["attempts_executed_twice"] == 0
    assert coverage["missing"] == [job.identity.episode_id]


def test_residual_bounds_are_conservative_and_named(job: Job) -> None:
    risk = once_replay_risk(
        [Execution(Path("absent"), job.model_copy(update={"remove": ["f03"]}), False)], 10
    )
    assert risk["changed_replays"] == 1
    assert risk["expected_affected_replays_upper"] == pytest.approx(60 * POLICY_CALL_RATE_BOUND)
    assert risk["expected_wrong_outcome_points_upper_per_table"] == pytest.approx(
        600 * POLICY_CALL_RATE_BOUND
    )


def test_missing_second_call_total_never_widens_the_ceiling(tmp_path: Path, job: Job) -> None:
    import json

    left = execute(tmp_path / "as-run", job)
    right = execute(tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}))
    path = right.run / "run.json"
    data = json.loads(path.read_text())
    data["extra"]["jobs_detail"][0].pop("policy_calls")
    path.write_text(json.dumps(data))
    v = verifier(tmp_path, job, None)
    v.settle(left, right)
    with pytest.raises(RunRefused, match=r"missing policy call total.*second"):
        v.save()


def test_saved_missing_call_total_is_recovered_from_the_engine_record(
    tmp_path: Path, job: Job
) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}))
    v = verifier(tmp_path, job, None)
    record = v.settle(left, right)
    record["execution_calls"][1] = record["compared_calls"] = None
    window = v.summary()["windows"][0]
    assert window["compared_calls"] == 7
    assert window["ceiling_unit"] == "policy request executions"


def test_two_events_exceed_a_one_event_call_ceiling(tmp_path: Path, job: Job) -> None:
    job = Job(identity=job.identity.model_copy(update={"max_steps": 80}))
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j, end_step=80))
    for number in range(2):
        original = Job(identity=job.identity.model_copy(update={"episode_id": f"e{number + 1:06}"}))
        v.windows[original.identity.episode_id] = "scoring"
        v.settle(
            execute(tmp_path / "as-run", original, end_step=80),
            execute(
                tmp_path / "second",
                original.model_copy(update={"mode": JobMode.replay}),
                delta=7e-5,
                end_step=80,
            ),
        )
    assert event_ceiling(32) == 1 and event_ceiling(160) == 2
    with pytest.raises(RunRefused, match="2 events exceed ceiling 1"):
        v.save()


@pytest.mark.parametrize("tamper", ["chain", "outcome"])
def test_saved_compared_execution_evidence_is_rechecked(
    tmp_path: Path, job: Job, tamper: str
) -> None:
    import pyarrow.parquet as pq

    from fleetlab.engine.chain import Chain

    left = execute(tmp_path / "as-run", job)
    right = execute(tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}))
    v = verifier(tmp_path, job, None)
    v.settle(left, right)
    v.save()
    if tamper == "chain":
        changed = Chain.load(right.chain_path)
        changed.wrist[-1] = "changed"
        changed.save(right.chain_path)
    else:
        path = right.run / "truth/job_outcomes.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["success"] = not rows[0]["success"]
        pq.write_table(type(table).from_pylist(rows, schema=table.schema), path)
    with pytest.raises(RunRefused, match="compared execution's evidence changed"):
        verifier(tmp_path, job, None).validate()


def test_identical_chains_with_different_outcomes_refuse(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}))
    right = replace(right, success=not left.success)
    with pytest.raises(RunRefused, match="identical chains have disagreeing outcomes"):
        verifier(tmp_path, job, None).settle(left, right)


def test_majority_requires_states_and_images_as_well_as_actions(tmp_path: Path, job: Job) -> None:
    from fleetlab.engine.chain import Chain

    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )

    def further(folder: Path, j: Job) -> Execution:
        execution = execute(folder, j)
        if j.mode is JobMode.run:
            chain = Chain.load(execution.chain_path)
            chain.wrist[-1] = "different image, same actions"
            chain.save(execution.chain_path)
        return execution

    with pytest.raises(RunRefused, match="no three-to-one"):
        verifier(tmp_path, job, further).settle(left, right)


def test_residual_uses_recorded_calls_and_labels_only_missing_totals(
    tmp_path: Path, job: Job
) -> None:
    changed = job.model_copy(update={"mode": JobMode.replay, "remove": ["f03"]})
    execution = execute(tmp_path / "changed", changed)
    risk = once_replay_risk([execution], 100)
    assert risk["expected_affected_replays_upper"] == pytest.approx(7 * POLICY_CALL_RATE_BOUND)
    assert risk["recorded_policy_calls"] == 7 and risk["step_budget_fallback_replays"] == 0
    import json

    path = execution.run / "run.json"
    data = json.loads(path.read_text())
    data["extra"]["jobs_detail"][0].pop("policy_calls")
    path.write_text(json.dumps(data))
    risk = once_replay_risk([execution], 100)
    assert risk["expected_affected_replays_upper"] == pytest.approx(60 * POLICY_CALL_RATE_BOUND)
    assert risk["step_budget_fallback_replays"] == 1
    assert "1 changed replay" in risk["text"] and "step budget" in risk["text"]


def test_all_windows_resume_reuses_every_existing_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.oracle.steps import Oracle

    root = tmp_path / "w99"
    manifest = toy.write_world(root, software.current("none"))
    run(
        RunConfig(
            out=str(root / "runs/toy"),
            chain_sample=False,
            batch_size=4,
            episodes_per_robot_day=4,
            **toy.engine_overrides(),
        ),
        manifest,
    )
    # Construct a proven source with no stored unmodified replays, to exercise
    # adding every absent second execution rather than reusing the world's sample.
    source = root / "runs/toy"
    with RunWriter(
        source,
        RunRecord.model_validate_json((source / "run.json").read_text()),
        episodes_per_robot_day=4,
    ) as writer:
        writer.update_record(chain_ok=True)
    oracle = Oracle.open(root, registry=toy.REGISTRY)
    oracle.step_plan()
    oracle.step_run(here=True, overrides=toy.engine_overrides())
    original_files = {p: p.read_bytes() for p in oracle.layout.run.rglob("*.npz")}
    oracle.claim_windows = ("scoring", "observed", "experiment")
    result = oracle.step_verify(here=True, overrides=toy.engine_overrides())
    assert result["jobs_run_now"] == 24  # 12 scoring proofs reused, 24 other-window attempts added
    assert [r["attempts_executed_twice"] for r in result["coverage"]] == [12, 12, 12]
    assert len({j.key for j in oracle.stored_manifest()}) == len(oracle.stored_manifest())
    again = oracle.step_verify(here=True, overrides=toy.engine_overrides())
    assert again["jobs_run_now"] == 0
    assert all(p.read_bytes() == value for p, value in original_files.items())
    from fleetlab.oracle import steps

    monkeypatch.setattr(steps, "run_heavy", lambda *args: pytest.fail("verification is complete"))
    assert oracle.step_verify()["jobs_run_now"] == 0
    report = oracle.step_report(boot=0, draws=0)
    assert report.replay_verification["coverage"] == result["coverage"]
    assert "windows were not executed twice" not in oracle.layout.summary.read_text()


def test_legacy_report_explains_missing_second_executions() -> None:
    from test_oracle import toy_table, toy_world

    from fleetlab.oracle.report import OracleReport, compute_report, summary_markdown

    report, _ = compute_report(
        toy_world(),
        TableBuild(toy_table(), []),
        boot=0,
        draws=0,
    )
    old = report.model_dump()
    old.pop("replay_verification")
    restored = OracleReport.model_validate(old)
    assert "windows were not executed twice (legacy run)" in summary_markdown(restored)


def test_actual_third_engine_execution_resumes_without_rerunning(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine import fleet
    from fleetlab.oracle.run import execute_third

    replay = job.model_copy(update={"mode": JobMode.replay})
    third = execute_third(tmp_path / "third", replay, 4, None, 2, toy.engine_overrides())
    before = third.chain_path.read_bytes()
    monkeypatch.setattr(
        fleet, "run", lambda *args: pytest.fail("the third execution already exists")
    )
    again = execute_third(tmp_path / "third", replay, 4, None, 2, toy.engine_overrides())
    assert again.success == third.success and again.chain_path.read_bytes() == before


@pytest.mark.parametrize("field", ["scene", "wrist", "state", "fixture_hash"])
def test_preceding_recorded_difference_refuses_before_any_third(
    tmp_path: Path, job: Job, field: str
) -> None:
    from fleetlab.engine.chain import Chain

    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    changed = Chain.load(right.chain_path)
    if field == "fixture_hash":
        changed.fixture_hash = "different"
    else:
        getattr(changed, field)[0] = "different"
    changed.save(right.chain_path)
    v = verifier(
        tmp_path, job, lambda *args: pytest.fail("prefix differs before the policy answer")
    )
    with pytest.raises(RunRefused, match="first difference"):
        v.settle(left, right)


def test_saved_majority_evidence_is_rechecked_on_read(tmp_path: Path, job: Job) -> None:
    from fleetlab.engine.chain import Chain

    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    result = v.settle(left, right)
    v.save()
    path = Execution(
        Path(result["third"]["run"]), Job.model_validate(result["third"]["job"]), left.success
    ).chain_path
    changed = Chain.load(path)
    changed.executed[1][0] += np.float32(7e-5)
    changed.save(path)
    with pytest.raises(RunRefused, match="third execution's evidence changed"):
        verifier(tmp_path, job, None).validate()


def test_pre_rule_failed_proof_can_be_classified_without_reexecuting_jobs(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.contracts.rollout import RunLayout
    from fleetlab.oracle.run import proof_failed

    replay = job.model_copy(update={"mode": JobMode.replay})
    record = RunRecord(
        software=job.identity.software,
        batch_size=4,
        replay_rules=[],
        chain_ok=False,
        extra={"chain_proof": {"details": [{"identical": False, "what": "state_hash"}]}},
    )
    (tmp_path / "run.json").write_text(record.model_dump_json())
    monkeypatch.setattr(RunLayout, "is_done", lambda *args: True)
    assert proof_failed(tmp_path, [replay]) is not None
    assert proof_failed(tmp_path, [replay], allow_adjudication=True) is None
    record.extra["replay_refusal"] = "three distinct chains"
    (tmp_path / "run.json").write_text(record.model_dump_json())
    assert proof_failed(tmp_path, [replay], allow_adjudication=True) is not None


def test_one_systematic_run_replay_path_difference_is_a_defect(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    kinds = []

    def additional(folder: Path, j: Job) -> Execution:
        kinds.append(j.mode)
        return execute(folder, j, delta=7e-5 if j.mode is JobMode.replay else 0)

    v = verifier(tmp_path, job, additional)
    with pytest.raises(RunRefused, match="no three-to-one"):
        v.settle(left, right)
    assert kinds == [JobMode.run, JobMode.replay]


def test_fourth_execution_unavailable_refuses(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )

    def additional(folder: Path, j: Job) -> Execution:
        if j.mode is JobMode.replay:
            raise OSError("cannot issue fourth")
        return execute(folder, j)

    with pytest.raises(RunRefused, match="fourth execution could not be made"):
        verifier(tmp_path, job, additional).settle(left, right)


def test_scratch_executions_use_both_real_engine_job_kinds(tmp_path: Path, job: Job) -> None:
    from fleetlab.oracle.run import execute_third

    for mode in (JobMode.run, JobMode.replay):
        execution = execute_third(
            tmp_path / mode.value,
            Job(identity=job.identity, mode=mode),
            4,
            None,
            2,
            toy.engine_overrides(),
        )
        assert execution.job.mode is mode and execution.chain_path.is_file()


def test_ceiling_is_the_strict_poisson_tail_at_the_unrounded_upper_rate() -> None:
    from scipy.stats import chi2, poisson

    assert pytest.approx(chi2.ppf(0.975, 8) / 2 / 26928, rel=1e-15) == POLICY_CALL_RATE_BOUND
    for calls in (1, 1000, 3300, 3400, 3500, 26928):
        ceiling = event_ceiling(calls)
        assert poisson.sf(ceiling, calls * POLICY_CALL_RATE_BOUND) < 0.001
        if ceiling:
            assert poisson.sf(ceiling - 1, calls * POLICY_CALL_RATE_BOUND) >= 0.001


def test_ceiling_corrects_a_quantile_at_the_strict_tail_boundary() -> None:
    import math

    from scipy.stats import poisson

    from fleetlab.oracle.verification import EVENT_CEILING_COVERAGE

    # The quantile rounds to zero here, although its survival probability is
    # just above the required strict tail. That zero is not a valid ceiling.
    rate = math.nextafter(-math.log(EVENT_CEILING_COVERAGE), math.inf)
    assert poisson.ppf(EVENT_CEILING_COVERAGE, rate) == 0
    assert poisson.sf(0, rate) >= 1 - EVENT_CEILING_COVERAGE
    assert event_ceiling(1, rate) == 1


def test_saved_round_six_third_is_reused_and_a_run_kind_execution_added(
    tmp_path: Path,
    job: Job,
) -> None:
    import json

    from fleetlab.oracle.verification import LEGACY_RULE

    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    v.settle(left, right)
    # Shape and actual kind of the round-six ledger, with its recorded replay third.
    v.save()
    stored = json.loads(v.path.read_text())
    saved_record = next(iter(stored["comparisons"].values()))
    saved_record["third"] = saved_record.pop("fourth")
    stored["rule"] = LEGACY_RULE
    v.path.write_text(json.dumps(stored))
    issued = []

    def additional(folder: Path, j: Job) -> Execution:
        issued.append(j.mode)
        # Avoid this constructed v2 fixture's old fourth path, which genuine v1 lacks.
        return execute(tmp_path / "new-as-run", j)

    upgraded = verifier(tmp_path, job, additional)
    upgraded.validate()
    assert issued == [JobMode.run]
    result = next(iter(upgraded.records.values()))
    assert Job.model_validate(result["third"]["job"]).mode is JobMode.replay
    assert Job.model_validate(result["fourth"]["job"]).mode is JobMode.run


def test_verify_resumes_a_legacy_event_before_saving_coverage(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from fleetlab.oracle import steps
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.verification import LEGACY_RULE
    from fleetlab.oracle.world import OracleLayout, OracleWorld

    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    v.required = {"scoring": [job.identity.episode_id]}
    v.settle(left, right)
    v.save()
    old = json.loads(v.path.read_text())
    record = next(iter(old["comparisons"].values()))
    record["third"] = record.pop("fourth")
    old["rule"] = LEGACY_RULE
    v.path.write_text(json.dumps(old))
    issued = []

    def further(folder: Path, j: Job) -> Execution:
        issued.append(j.mode)
        return execute(tmp_path / "new-as-run", j)

    world = OracleWorld(spec=toy.spec(), identities=(job.identity,), registry=toy.REGISTRY)
    oracle = Oracle(tmp_path, world, OracleLayout(tmp_path / "oracle", world.world_id))
    monkeypatch.setattr(steps, "world_ready", lambda *args: [])
    monkeypatch.setattr(Oracle, "runs", lambda *args: [left.run, right.run])
    monkeypatch.setattr(
        Oracle, "verifier", lambda *args, **kwargs: verifier(tmp_path, job, further)
    )
    result = oracle.step_verify(here=True)
    assert result["jobs_run_now"] == 0 and issued == [JobMode.run]
    assert not result["coverage"][0]["missing"]
    assert json.loads(v.path.read_text())["rule"] != LEGACY_RULE


def test_a_growing_replay_run_refreshes_its_cached_outcomes(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from fleetlab.contracts.rollout import RunLayout
    from fleetlab.engine import fleet
    from fleetlab.oracle.run import run_here

    second_identity = job.identity.model_copy(update={"episode_id": "e000099"})
    originals = [Job(identity=job.identity), Job(identity=second_identity)]
    source = tmp_path / "source"
    for original in originals:
        execute(source, original)
    replays = [j.model_copy(update={"mode": JobMode.replay}) for j in originals]
    v = ReplayVerifier(
        tmp_path / "verification", {j.identity.episode_id: "scoring" for j in originals}, None
    )

    def engine(config: Any, manifest: Any) -> Any:
        for replay in manifest:
            if not RunLayout(Path(config.out)).is_done(replay):
                execute(Path(config.out), replay)
        return SimpleNamespace(
            record=RunRecord.model_validate_json((Path(config.out) / "run.json").read_text())
        )

    monkeypatch.setattr(fleet, "run", engine)
    folder = tmp_path / "replays"
    options = dict(
        manifest_file=tmp_path / "jobs.jsonl",
        world_hash=None,
        extend=True,
        verifier=v,
        overrides=toy.engine_overrides(),
    )
    assert run_here(folder, replays[:1], [source], 4, **options)["chain_ok"]  # type: ignore[arg-type]
    before = RunLayout(folder).chain(replays[0]).read_bytes()
    assert v.recorded_outcome(folder, replays[0])
    assert run_here(folder, replays, [source], 4, **options)["chain_ok"]  # type: ignore[arg-type]
    assert v.recorded_outcome(folder, replays[1])
    assert RunLayout(folder).chain(replays[0]).read_bytes() == before


def oracle_with_executions(tmp: Path, job: Job, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fleetlab.oracle import steps
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.world import OracleLayout, OracleWorld

    world = OracleWorld(spec=toy.spec(), identities=(job.identity,), registry=toy.REGISTRY)
    oracle = Oracle(tmp, world, OracleLayout(tmp / "oracle", world.world_id))
    monkeypatch.setattr(steps, "world_ready", lambda *args: [])
    return oracle


def test_report_explains_incomplete_double_execution(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = oracle_with_executions(tmp_path, job, monkeypatch)
    left = execute(tmp_path / "source", job)
    monkeypatch.setattr(type(oracle), "runs", lambda *args: [left.run])
    v = oracle.verifier()
    v.required = {"scoring": [job.identity.episode_id]}
    v.save()
    report = oracle.step_report(boot=0, draws=0, truth_table=False)
    assert "incomplete double execution of scoring: 1 attempts" in report.problems
    assert "0 of 1 attempts executed twice" in oracle.layout.summary.read_text()


def test_verify_does_not_return_done_with_a_missing_attempt(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.oracle import steps

    oracle = oracle_with_executions(tmp_path, job, monkeypatch)
    left = execute(tmp_path / "source", job)
    monkeypatch.setattr(type(oracle), "runs", lambda *args: [left.run])
    v = oracle.verifier()
    v.required = {"scoring": [job.identity.episode_id]}
    v.save()
    queued = []

    def queue(*args: Any) -> int:
        queued.append(args)
        return 17

    monkeypatch.setattr(steps, "run_heavy", queue)
    with pytest.raises(RunRefused, match="exit status 17"):
        oracle.step_verify()
    assert len(queued) == 1


def test_report_reloads_saved_evidence_before_using_it(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine.chain import Chain

    oracle = oracle_with_executions(tmp_path, job, monkeypatch)
    left = execute(tmp_path / "source", job)
    right = execute(tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}))
    monkeypatch.setattr(type(oracle), "runs", lambda *args: [left.run, right.run])
    v = oracle.verifier()
    v.required = {"scoring": [job.identity.episode_id]}
    v.settle(left, right)
    v.save()
    chain = Chain.load(right.chain_path)
    chain.wrist[-1] = "tampered after verification"
    chain.save(right.chain_path)
    with pytest.raises(RunRefused, match="compared execution's evidence changed"):
        oracle.step_report(boot=0, draws=0, truth_table=False)


def test_refused_ledger_is_named_in_the_report(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    oracle = oracle_with_executions(tmp_path, job, monkeypatch)
    left = execute(tmp_path / "source", job)
    monkeypatch.setattr(type(oracle), "runs", lambda *args: [left.run])
    v = oracle.verifier()
    v.save()
    data = json.loads(v.path.read_text())
    data["rule"] = "unknown-rule"
    v.path.write_text(json.dumps(data))
    report = oracle.step_report(boot=0, draws=0, truth_table=False)
    assert any("verification belongs to another world or rule" in p for p in report.problems)


def test_further_execution_of_the_wrong_kind_refuses(tmp_path: Path, job: Job) -> None:
    left = execute(tmp_path / "as-run", job)
    right = execute(
        tmp_path / "second", job.model_copy(update={"mode": JobMode.replay}), delta=7e-5
    )
    v = verifier(
        tmp_path,
        job,
        lambda folder, j: execute(folder, j.model_copy(update={"mode": JobMode.replay})),
    )
    with pytest.raises(RunRefused, match="changed the kind"):
        v.settle(left, right)


def test_legacy_replay_third_is_replaced_for_two_changed_replays(tmp_path: Path, job: Job) -> None:
    import json

    from fleetlab.oracle.verification import LEGACY_RULE

    changed = job.model_copy(update={"mode": JobMode.replay, "remove": ["f03"]})
    left = execute(tmp_path / "first", changed)
    right = execute(tmp_path / "second", changed, delta=7e-5)
    v = verifier(tmp_path, job, lambda folder, j: execute(folder, j))
    v.settle(left, right)
    v.save()
    stored = json.loads(v.path.read_text())
    record = next(iter(stored["comparisons"].values()))
    record["third"] = left.evidence()  # v1 used a replay third, even for this pair
    record.pop("fourth")
    stored["rule"] = LEGACY_RULE
    v.path.write_text(json.dumps(stored))
    issued = []

    def further(folder: Path, j: Job) -> Execution:
        issued.append(j.mode)
        return execute(tmp_path / f"replacement-{len(issued)}", j)

    verifier(tmp_path, job, further).validate()
    assert issued == [JobMode.run, JobMode.run]


def test_changed_replay_is_replaced_by_an_unmodified_second(
    tmp_path: Path, job: Job, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.oracle import steps

    oracle = oracle_with_executions(tmp_path, job, monkeypatch)
    left = execute(tmp_path / "source", job)
    changed = execute(
        tmp_path / "changed", job.model_copy(update={"mode": JobMode.replay, "remove": ["f03"]})
    )
    second_run = oracle.layout.root / "verification/run"
    monkeypatch.setattr(
        type(oracle),
        "runs",
        lambda *args: [left.run, changed.run] + ([second_run] if second_run.is_dir() else []),
    )
    issued = []

    def run_seconds(folder: Path, manifest: list[Job], *args: Any, **kwargs: Any) -> dict[str, int]:
        for replay in manifest:
            assert not (replay.remove or replay.add)
            issued.append(replay)
            execute(folder, replay)
        return {"jobs_run_now": len(manifest)}

    monkeypatch.setattr(steps, "run_here", run_seconds)
    result = oracle.step_verify(here=True)
    assert len(issued) == result["jobs_run_now"] == 1
    assert not result["coverage"][0]["missing"]
