"""Synthetic queue-driver unit tests; engine, arm reader and run status are mocked.

These prove orchestration and refusal checks, not simulation or GPU execution.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from engine_helpers import ident
from test_engine_experiments import toy_world

from fleetlab.agents.sessions import Analysis, EngineReply, ExperimentQueue, QueuedExperiment
from fleetlab.baselines.classical import driver
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.tools import ArmResult, ExperimentArm, ExperimentRequest, ExperimentResult
from fleetlab.engine.experiments import WorldTruth, queue_dirs
from fleetlab.engine.fleet import RunConfig
from fleetlab.log.writer import RunStatus
from fleetlab.runtime import repo_root
from fleetlab.toolserver.files import EXPERIMENTS_FILE, RESULTS_FILE, append_line
from fleetlab.toolserver.public_results import COMPLETED, EXECUTION_FAILED, PUBLIC_REFUSALS


@dataclass
class Synthetic:
    queue: Path
    truth: Path
    root: Path
    analysis: Analysis
    engine: Mock
    arms: Mock
    status: RunStatus


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    analysis = Analysis.create("w01", base=tmp_path / "analyses", analysis_id="w01-synthetic")
    export = tmp_path / "export"
    export.mkdir()
    (export / "synthetic.txt").write_text("synthetic observed data")
    import hashlib

    from fleetlab.contracts.log import DESCRIPTION_FILE
    from fleetlab.log.leaks import audit_identifier, export_digest

    (export / DESCRIPTION_FILE).parent.mkdir(parents=True)
    (export / DESCRIPTION_FILE).write_text(json.dumps({"fleet": {"id": "w01"}}))
    (tmp_path / "spec.json").write_text(json.dumps({"world_id": "w01"}))
    receipts = tmp_path / "export-checks"
    receipts.mkdir()
    (receipts / "check-synthetic.json").write_text(
        json.dumps(
            {
                "clean": True,
                "export": str(export.resolve()),
                "export_digest": export_digest(export),
                "check_id": audit_identifier(()),
                "terms": [],
                "terms_digest": hashlib.sha256(b"[]").hexdigest(),
            }
        )
    )
    # Synthetic receipt tests byte binding, not the leak auditor's coverage.
    queued = QueuedExperiment(
        experiment_id="x0001",
        session=1,
        request=ExperimentRequest(
            arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[])],
            robots=["r01"],
            episodes_per_arm=2,
            paired_starts=True,
            power_note="synthetic unit test only",
        ),
    )
    append_line(analysis.outbox / EXPERIMENTS_FILE, queued.model_dump(mode="json"))
    written = analysis.write_queue(1, export)
    assert written is not None
    queue, _ = written
    base_run = tmp_path / "original-run"
    (base_run / "log").mkdir(parents=True)
    (base_run / "log/synthetic.txt").write_text("synthetic base log")
    world = toy_world([str(base_run)], 1)
    world = world.model_copy(
        update={"policy": world.policy.model_copy(update={"precision": "fp16_res32"})}
    )
    truth = tmp_path / "world_truth.json"
    truth.write_text(world.model_dump_json())
    root = tmp_path / "new-engine-output"
    episodes = [ident(n).episode_id for n in range(1, 5)]
    status = RunStatus(done=tuple(episodes), partial=(), merged=tuple(episodes))
    arms = Mock(return_value=[])
    monkeypatch.setattr(driver, "_in_heavy_scope", lambda: True)
    monkeypatch.setattr(
        driver, "_source", lambda: {"commit": "synthetic", "sha256": "synthetic-source"}
    )
    monkeypatch.setattr(driver, "configure_process", lambda: None)
    monkeypatch.setattr(driver, "arm_problems", arms)
    monkeypatch.setattr(driver, "run_status", lambda *args: status)
    # Track the environment writes so they cannot leak into other test modules.
    monkeypatch.setenv("FLEETLAB_DATA_DIR", str(tmp_path / "synthetic-runtime"))
    monkeypatch.setenv("FLEETLAB_MIOPEN_DIR", str(tmp_path / "synthetic-miopen"))

    def engine_reply(queue_path: Path, truth: WorldTruth, root: Path, **kwargs: Any) -> EngineReply:
        received = ExperimentQueue.model_validate_json(queue_path.read_text())
        assert received.world_id == truth.world_id
        run_dir, export_dir = queue_dirs(root, received.analysis_id, received.session)
        run_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "synthetic.txt").write_text("synthetic extended export")
        result = ExperimentResult(
            experiment_id="x0001",
            status="done",
            message=COMPLETED,
            arms=[
                ArmResult(arm_index=i, episode_ids=episodes[2 * i : 2 * i + 2]) for i in range(2)
            ],
        )
        (run_dir / "run.json").write_text(json.dumps({"chain_ok": True}))
        (run_dir.parent / "session-01.plan.json").write_text(
            json.dumps(
                {
                    "jobs": [Job(identity=ident(n)).model_dump(mode="json") for n in range(1, 5)],
                    "results": [result.model_dump(mode="json")],
                }
            )
        )
        return EngineReply(results=[result], export=export_dir)

    engine = Mock(side_effect=engine_reply)
    monkeypatch.setattr(driver, "run_queue", engine)
    return Synthetic(queue, truth, root, analysis, engine, arms, status)


def run(s: Synthetic, **kwargs: Any) -> EngineReply:
    kwargs.setdefault("miopen_dir", s.truth.parent / "synthetic-miopen")
    return driver.run(s.queue, s.truth, s.root, **kwargs)


def test_engine_queue_reply_provenance_and_analysis_results(synthetic: Synthetic) -> None:
    s = synthetic
    original = s.queue.read_bytes(), s.truth.read_bytes()
    reply = run(s, workers=8, batch_size=4, record_video=False)
    args, kwargs = s.engine.call_args
    assert args[0] == s.root / "engine-queue.json" and args[2] == s.root
    config: RunConfig = kwargs["config"]
    assert (config.workers, config.batch_size, config.precision) == (8, 4, "fp16_res32")
    assert config.check_factory == "fleetlab.sim.libero_success:FleetCheck"
    assert kwargs["record_video"] is False
    s.arms.assert_called_once_with(reply)
    assert s.analysis.results_by_id() == {"x0001": reply.results[0]}
    assert (s.queue.read_bytes(), s.truth.read_bytes()) == original
    manifest = json.loads((s.root / "driver.json").read_text())
    assert manifest["inputs"]["queue"]["sha256"] == driver._hash(s.queue)
    assert manifest["inputs"]["truth"]["sha256"] == driver._hash(s.truth)
    assert manifest["source"]["commit"] == "synthetic"
    attempt = json.loads((s.root / "attempts/0001.json").read_text())
    assert attempt["complete"] and attempt["completion"]["jobs"] == 4
    assert attempt["completion"]["chain_ok"] is True
    assert attempt["reply_sha256"] == driver._hash(s.root / "reply.json")
    assert attempt["inputs_verified"] is True
    assert attempt["inputs_sha256"] == driver._digest(manifest["inputs"])


@pytest.mark.parametrize(
    "changed_input",
    [
        "queue",
        "truth",
        "analysis",
        "requests",
        "export",
        "base-log",
        "queue-snapshot",
        "truth-snapshot",
    ],
)
def test_completion_rejects_input_changes_during_engine_execution(
    synthetic: Synthetic,
    changed_input: str,
) -> None:
    original_engine = synthetic.engine.side_effect

    def mutating_engine(*args: Any, **kwargs: Any) -> EngineReply:
        reply: EngineReply = original_engine(*args, **kwargs)
        paths = {
            "queue": synthetic.queue,
            "truth": synthetic.truth,
            "analysis": synthetic.analysis.root / "analysis.json",
            "export": synthetic.truth.parent / "export/synthetic.txt",
            "base-log": synthetic.truth.parent / "original-run/log/synthetic.txt",
            "queue-snapshot": synthetic.root / "queue.json",
            "truth-snapshot": synthetic.root / "world_truth.json",
        }
        if changed_input == "requests":
            request = synthetic.analysis.requests()[0]
            changed = request.request.model_copy(
                update={"power_note": "changed during synthetic run"}
            )
            updated = request.model_copy(update={"request": changed})
            (synthetic.analysis.outbox / EXPERIMENTS_FILE).write_text(
                updated.model_dump_json() + "\n"
            )
        else:
            path = paths[changed_input]
            path.write_text(path.read_text() + "\n")
        return reply

    synthetic.engine.side_effect = mutating_engine
    with pytest.raises(driver.DriverError, match=r"inputs? (changed|snapshots changed)"):
        run(synthetic)
    assert synthetic.analysis.results_by_id() == {}
    assert not (synthetic.root / "reply.json").exists()
    assert not (synthetic.root / "reply-completion.json").exists()
    attempt = json.loads((synthetic.root / "attempts/0001.json").read_text())
    assert attempt["complete"] is False and attempt["inputs_verified"] is False
    assert "reply" in attempt and "error" in attempt


def test_requires_heavy_scope_before_writing(
    synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "_in_heavy_scope", lambda: False)
    with pytest.raises(driver.DriverError, match="heavy-run"):
        run(synthetic)
    assert not synthetic.root.exists()
    synthetic.engine.assert_not_called()


def test_cli_relaunches_through_wrapper_without_running_engine(
    synthetic: Synthetic,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_in_heavy_scope", lambda: False)
    command = Mock(return_value=subprocess.CompletedProcess([], 7))
    monkeypatch.setattr(subprocess, "run", command)
    argv = [
        "--queue",
        str(synthetic.queue),
        "--truth",
        str(synthetic.truth),
        "--root",
        str(synthetic.root),
    ]
    assert driver.main(argv) == 7
    assert command.call_args.args[0][0] == str(repo_root() / "tools/heavy-run")
    assert command.call_args.args[0][-len(argv) :] == argv
    synthetic.engine.assert_not_called()


@pytest.mark.parametrize(
    "cgroup, wrapped",
    [
        ("0::/user.slice/app.slice/robots-heavy-12345.scope\n", True),
        ("0::/user.slice/app.slice/robots-heavy-12345.scope/child\n", True),
        ("0::/user.slice/app.slice/run-light.scope\n", False),
        ("0::/user.slice/app.slice/robots-heavy-other.scope\n", False),
    ],
)
def test_heavy_scope_detection(cgroup: str, wrapped: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: cgroup)
    assert driver._in_heavy_scope() is wrapped


def test_completed_resume_is_idempotent_and_does_not_rerun_engine(synthetic: Synthetic) -> None:
    reply = run(synthetic)
    recorded = (synthetic.analysis.results / RESULTS_FILE).read_bytes()
    assert run(synthetic, resume=True) == reply
    synthetic.engine.assert_called_once()
    assert (synthetic.analysis.results / RESULTS_FILE).read_bytes() == recorded
    assert len(list((synthetic.root / "attempts").glob("*.json"))) == 2


def test_resume_after_interrupted_result_recording(
    synthetic: Synthetic,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_results = Analysis.record_results
    interrupted = Mock(side_effect=OSError("synthetic interruption before recording"))
    monkeypatch.setattr(Analysis, "record_results", interrupted)
    with pytest.raises(OSError, match="interruption"):
        run(synthetic)
    assert (synthetic.root / "reply.json").is_file()
    monkeypatch.setattr(Analysis, "record_results", record_results)
    reply = run(synthetic, resume=True)
    assert synthetic.analysis.results_by_id() == {"x0001": reply.results[0]}
    synthetic.engine.assert_called_once()


def test_resume_after_engine_interruption_reuses_exact_queue(synthetic: Synthetic) -> None:
    engine = synthetic.engine.side_effect

    def interrupted(*args: Any, **kwargs: Any) -> EngineReply:
        engine(*args, **kwargs)
        raise OSError("synthetic interrupted engine")

    synthetic.engine.side_effect = interrupted
    with pytest.raises(OSError, match="interrupted engine"):
        run(synthetic)
    snapshot = (synthetic.root / "queue.json").read_bytes()
    assert not (synthetic.root / "reply.json").exists()
    assert synthetic.analysis.results_by_id() == {}
    assert not json.loads((synthetic.root / "attempts/0001.json").read_text())["complete"]
    synthetic.engine.side_effect = engine
    run(synthetic, resume=True)
    assert synthetic.engine.call_count == 2
    assert (synthetic.root / "queue.json").read_bytes() == snapshot
    assert synthetic.analysis.results_by_id()["x0001"].status == "done"


def test_resume_rejects_changed_completed_export(synthetic: Synthetic) -> None:
    reply = run(synthetic)
    (reply.export / "synthetic.txt").write_text("changed output data")
    with pytest.raises(driver.DriverError, match="completion artifacts changed"):
        run(synthetic, resume=True)
    synthetic.engine.assert_called_once()


def test_cli_prints_only_verified_reply_export(
    synthetic: Synthetic,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        driver.main(
            [
                "--queue",
                str(synthetic.queue),
                "--truth",
                str(synthetic.truth),
                "--root",
                str(synthetic.root),
                "--miopen-dir",
                str(synthetic.truth.parent / "synthetic-miopen"),
            ]
        )
        == 0
    )
    _, export_dir = queue_dirs(synthetic.root, synthetic.analysis.analysis_id, 1)
    assert capsys.readouterr().out.strip() == str(export_dir)


@pytest.mark.parametrize("change", ["workers", "queue", "truth", "source", "base-log"])
def test_resume_rejects_changed_inputs_or_configuration(
    synthetic: Synthetic,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    run(synthetic)
    kwargs: dict[str, Any] = {"resume": True}
    if change == "workers":
        kwargs["workers"] = 3
    elif change in ("queue", "truth"):
        path = getattr(synthetic, change)
        path.write_text(path.read_text() + "\n")
    elif change == "source":
        monkeypatch.setattr(driver, "_source", lambda: {"commit": "different"})
    else:
        (synthetic.truth.parent / "original-run/log/synthetic.txt").write_text("changed")
    with pytest.raises(driver.DriverError, match="resume inputs"):
        run(synthetic, **kwargs)
    synthetic.engine.assert_called_once()


def test_existing_cache_is_preserved(synthetic: Synthetic) -> None:
    synthetic.root.mkdir()
    sentinel = synthetic.root / "old-record.txt"
    sentinel.write_text("old cache must remain")
    with pytest.raises(FileExistsError):
        run(synthetic)
    with pytest.raises(driver.DriverError, match="created by this driver"):
        run(synthetic, resume=True)
    assert sentinel.read_text() == "old cache must remain"
    synthetic.engine.assert_not_called()


def test_root_must_not_overlap_inputs(synthetic: Synthetic) -> None:
    with pytest.raises(driver.DriverError, match="separate"):
        driver.run(synthetic.queue, synthetic.truth, synthetic.analysis.root / "engine")
    synthetic.engine.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "missing-result",
        "wrong-count",
        "reused-episode",
        "missing-arm",
        "queued",
        "arm-export",
        "pending",
        "chain",
        "free-text-message",
        "wrong-status-message",
        "refusal-with-arms",
    ],
)
def test_does_not_fabricate_completion(
    synthetic: Synthetic,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    original_engine = synthetic.engine.side_effect

    def invalid(*args: Any, **kwargs: Any) -> EngineReply:
        reply: EngineReply = original_engine(*args, **kwargs)
        if failure == "missing-result":
            reply.results = []
        elif failure == "wrong-count":
            reply.results[0].arms[0].episode_ids.pop()
        elif failure == "reused-episode":
            reply.results[0].arms[1].episode_ids[0] = reply.results[0].arms[0].episode_ids[0]
        elif failure == "missing-arm":
            reply.results[0].arms.pop()
        elif failure == "queued":
            reply.results[0] = reply.results[0].model_copy(update={"status": "queued"})
        elif failure == "arm-export":
            synthetic.arms.return_value = ["synthetic missing arm episode"]
        elif failure == "pending":
            monkeypatch.setattr(
                driver, "run_status", lambda *args: replace(synthetic.status, pending=("e000001",))
            )
        elif failure == "chain":
            run_dir, _ = queue_dirs(synthetic.root, "w01-synthetic", 1)
            (run_dir / "run.json").write_text(json.dumps({"chain_ok": False}))
        elif failure == "free-text-message":
            reply.results[0] = reply.results[0].model_copy(
                update={"message": "synthetic hidden runner detail"}
            )
        elif failure == "wrong-status-message":
            reply.results[0] = reply.results[0].model_copy(
                update={"message": next(iter(PUBLIC_REFUSALS))}
            )
        elif failure == "refusal-with-arms":
            reply.results[0] = reply.results[0].model_copy(
                update={"status": "refused", "message": next(iter(PUBLIC_REFUSALS))}
            )
        return reply

    synthetic.engine.side_effect = invalid
    with pytest.raises(driver.DriverError):
        run(synthetic)
    assert synthetic.analysis.results_by_id() == {}
    record = json.loads((synthetic.root / "attempts/0001.json").read_text())
    assert record["complete"] is False and "reply" in record and "error" in record


@pytest.mark.parametrize("message", sorted(PUBLIC_REFUSALS))
def test_refusal_is_saved_and_recorded_but_is_not_success(
    synthetic: Synthetic, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    original_engine = synthetic.engine.side_effect

    def refused(*args: Any, **kwargs: Any) -> EngineReply:
        reply: EngineReply = original_engine(*args, **kwargs)
        reply.results[0] = ExperimentResult(
            experiment_id="x0001",
            status="refused",
            arms=[],
            message=message,
        )
        run_dir, _ = queue_dirs(synthetic.root, "w01-synthetic", 1)
        (run_dir.parent / "session-01.plan.json").write_text(
            json.dumps(
                {
                    "jobs": [],
                    "results": [reply.results[0].model_dump(mode="json")],
                }
            )
        )
        return reply

    synthetic.engine.side_effect = refused
    if message == EXECUTION_FAILED:
        with pytest.raises(driver.DriverError, match="execution failed; resume"):
            run(synthetic)
        assert synthetic.analysis.results_by_id() == {}
        assert not (synthetic.root / "reply.json").exists()
        assert not json.loads((synthetic.root / "attempts/0001.json").read_text())["complete"]
        synthetic.engine.side_effect = original_engine
        assert run(synthetic, resume=True).results[0].status == "done"
        assert synthetic.engine.call_count == 2
        return
    reply = run(synthetic)
    assert synthetic.analysis.results_by_id()["x0001"].status == "refused"
    assert reply.results[0].message == message and reply.results[0].arms == []
    assert (synthetic.root / "reply.json").is_file()
    attempt = json.loads((synthetic.root / "attempts/0001.json").read_text())
    assert attempt["complete"]  # The reply was delivered, not a successful experiment.
    assert attempt["completed_requests"] == attempt["completion"]["jobs"] == 0
    assert attempt["refused_requests"] == ["x0001"]
    assert run(synthetic, resume=True) == reply
    assert (
        driver.main(
            [
                "--queue",
                str(synthetic.queue),
                "--truth",
                str(synthetic.truth),
                "--root",
                str(synthetic.root),
                "--resume",
                "--miopen-dir",
                str(synthetic.truth.parent / "synthetic-miopen"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(reply.export)
    synthetic.engine.assert_called_once()


def test_public_name_queue_binds_certified_world_records(synthetic: Synthetic) -> None:
    s = synthetic
    world = WorldTruth.model_validate_json(s.truth.read_text()).model_copy(
        update={"world_id": "w47"}
    )
    s.truth.write_text(world.model_dump_json())
    (s.truth.parent / "spec.json").write_text(json.dumps({"world_id": "w47"}))
    run(s)
    original = ExperimentQueue.model_validate_json(s.queue.read_text())
    executed = ExperimentQueue.model_validate_json((s.root / "engine-queue.json").read_text())
    assert original.world_id == "w01"
    assert executed.world_id == "w47"
    assert executed.experiments == original.experiments
    assert s.engine.call_args.args[1].world_id == "w47"


@pytest.mark.parametrize("public_id", ["w02", "w01"])
def test_other_world_queue_refused_even_with_same_public_name(
    synthetic: Synthetic, public_id: str
) -> None:
    from fleetlab.contracts.log import DESCRIPTION_FILE

    s = synthetic
    other = s.truth.parent / "other-export"
    other.mkdir()
    (other / DESCRIPTION_FILE).parent.mkdir(parents=True)
    (other / DESCRIPTION_FILE).write_text(json.dumps({"fleet": {"id": public_id}}))
    (other / "synthetic.txt").write_text("another world's observed data")
    queue = ExperimentQueue.model_validate_json(s.queue.read_text()).model_copy(
        update={"world_id": public_id, "export": str(other)}
    )
    s.queue.write_text(queue.model_dump_json())
    with pytest.raises(driver.DriverError, match="queue export belongs to another world"):
        run(s)
    assert not s.root.exists()
    s.engine.assert_not_called()


def test_missing_certification_refused_before_engine(synthetic: Synthetic) -> None:
    (synthetic.truth.parent / "export-checks/check-synthetic.json").unlink()
    with pytest.raises(ValueError, match="not certified"):
        run(synthetic)
    synthetic.engine.assert_not_called()


def test_evaluator_public_name_answer_uses_certified_producer_binding(
    synthetic: Synthetic,
) -> None:
    from fleetlab.baselines.classical.evaluate import _check_answer_world
    from fleetlab.contracts.log import DESCRIPTION_FILE
    from fleetlab.log.leaks import export_digest

    s = synthetic
    export = s.truth.parent / "export"
    (export / "log/episodes.parquet").write_text("synthetic episodes bytes")
    receipt_path = s.truth.parent / "export-checks/check-synthetic.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["export_digest"] = export_digest(export)
    receipt_path.write_text(json.dumps(receipt))
    (s.truth.parent / "spec.json").write_text(json.dumps({"world_id": "w47"}))
    answer = s.truth.parent / "answer"
    (answer / "outbox").mkdir(parents=True)
    (answer / "outbox/fix_set.json").write_text("synthetic output pointer")
    evidence: dict[str, Any] = {
        "world_id": "w01",
        "input_sha256": {
            str(DESCRIPTION_FILE): driver._hash(export / DESCRIPTION_FILE),
            "log/episodes.parquet": driver._hash(export / "log/episodes.parquet"),
        },
    }
    (answer / "evidence.json").write_text(json.dumps(evidence))
    _check_answer_world(answer, evidence, s.truth.parent, "w47")
    evidence["input_sha256"]["log/episodes.parquet"] = "foreign bytes"
    (answer / "evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="input hashes match no export"):
        _check_answer_world(answer, evidence, s.truth.parent, "w47")
