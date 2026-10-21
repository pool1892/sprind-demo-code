"""The development loop with synthetic logs; model, heavy dispatch and scorer are stubs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import TASK_CLASSES, write_toy_run

from fleetlab.agents.audit import tree_manifest
from fleetlab.agents.batch import Batch
from fleetlab.agents.run import AgentResult
from fleetlab.agents.sessions import Analysis, EngineReply, ExperimentQueue, SessionRecord
from fleetlab.agents.toolenv import ToolEnv, build_skills
from fleetlab.contracts.actions import catalogue
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.tools import ArmResult, ExperimentResult
from fleetlab.contracts.world import DayRange, Windows
from fleetlab.integration import accounting
from fleetlab.integration import analyse as runner
from fleetlab.integration.accounting import (
    PRICES,
    collect_accounting,
    price_equivalent,
    repeat_group,
    reserve_repeat,
    verify_repeat,
)
from fleetlab.log import export_runs
from fleetlab.skill_library import ANALYSIS_SKILLS, library_dir
from fleetlab.toolserver.files import EXPERIMENTS_FILE, append_line
from fleetlab.worlds.mini import mini_spec


@pytest.fixture
def prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[argparse.Namespace, Fleet, Mock]:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    monkeypatch.setattr(accounting, "REPEAT_REGISTER", tmp_path / "fixed-register")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "synthetic-commit")
    fleet = write_toy_run(tmp_path / "observed")
    export = tmp_path / "export"
    export_runs([tmp_path / "observed"], export, fleet=fleet, task_classes=TASK_CLASSES)
    files = tmp_path / "stored" / "files"
    files.mkdir(parents=True)
    (files / "hash").write_text("fixed-world-hash")
    (files / "world_truth.json").write_text(
        json.dumps(
            {
                "fleet": fleet.model_dump(mode="json"),
                "task_classes": TASK_CLASSES,
            }
        )
    )
    spec = mini_spec().model_copy(
        update={
            "world_id": "w99",
            "windows": Windows(
                observed=DayRange(first=0, last=2),
                experiment=DayRange(first=3, last=3),
                scoring=DayRange(first=4, last=4),
            ),
        }
    )
    monkeypatch.setattr(
        runner, "inputs", lambda root: (files, spec, fleet, [tmp_path / "observed"])
    )
    empty = tmp_path / "empty"
    empty.mkdir()

    def environment(**kwargs: Any) -> ToolEnv:
        skills = build_skills(
            tmp_path / "bundles",
            library_dir(),
            ANALYSIS_SKILLS,
            ablate_fault_classes=kwargs["ablate_fault_classes"],
        )
        return ToolEnv(empty, empty, empty, skills)

    monkeypatch.setattr(runner, "build_tool_env", environment)
    monkeypatch.setattr(Batch, "after_gate", Mock(return_value=SimpleNamespace(records={})))
    monkeypatch.setattr(
        runner,
        "source_audit",
        lambda *a, **kw: SimpleNamespace(passed=True, as_json=lambda: {"passed": True}),
    )
    score = Mock()
    monkeypatch.setattr(runner, "score_analysis", score)
    args = argparse.Namespace(
        world_root=files.parent,
        export=export,
        out=tmp_path / "result",
        condition="logs",
        model="gpt-6-luna",
        effort="high",
        workers=2,
        resume_from=None,
        base_run=[],
        ablate_fault_classes=False,
        max_sessions=17,
    )
    return args, fleet, score


def save_session(
    analysis: Analysis,
    export: Path,
    agent: Any,
    *,
    queued: bool = False,
    error: str | None = None,
    fleet: Fleet,
    on_start: Any = None,
) -> SessionRecord:
    number = analysis.sessions_done() + 1
    call = analysis.root.parent / "agent-runs" / f"call-{number}"
    call.mkdir(parents=True)
    if on_start is not None:
        on_start(call)
    usage = {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}
    (call / "result.json").write_text(
        json.dumps({"model": agent.model, "usage": usage, "seconds": 2, "error": error})
    )
    (call / "audit.json").write_text(
        json.dumps({"passed": True, "manifest": {"export": tree_manifest(export)}})
    )
    (call / "audit-time.json").write_text('{"seconds": 0.25}')
    if queued:
        action = next(a for a in catalogue(fleet) if a.is_complete and a.has_replay)
        append_line(
            analysis.outbox / EXPERIMENTS_FILE,
            {
                "experiment_id": "x0001",
                "session": number,
                "request": {
                    "arms": [{"actions": []}, {"actions": [action.model_dump(mode="json")]}],
                    "robots": ["r01"],
                    "episodes_per_arm": 1,
                    "paired_starts": True,
                    "power_note": "public calculation",
                },
            },
        )
    elif error is None:
        (analysis.outbox / "diagnosis.json").write_text(
            json.dumps({"causes": [], "mode_summaries": [], "warnings": [], "retrain": []})
        )
        actions = [a for a in catalogue(fleet) if a.is_complete and a.has_replay][:10]
        (analysis.outbox / "fix_set.json").write_text(
            json.dumps(
                {
                    "actions": [
                        {
                            "action": a.model_dump(mode="json"),
                            "expected_gain": 0,
                            "uncertainty": {"low": 0, "high": 0},
                            "rationale": "test",
                        }
                        for a in actions
                    ]
                }
            )
        )
    session = SessionRecord(
        number=number,
        state="waiting" if queued else "failed" if error else "done",
        export=str(export),
        run_dir=str(call),
        ok=error is None,
        error=error,
        seconds=2,
        usage=usage,
        queued=["x0001"] if queued else [],
        submitted=[] if queued else ["diagnosis", "fix_set"],
        report=None,
        tool_calls=1,
    )
    (analysis.root / "sessions" / f"session-{number:02d}.json").write_text(
        json.dumps(asdict(session))
    )
    return session


def test_logs_loop_assigns_repeats_at_first_call_and_records_configured_model(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fleet, score = prepared
    calls = []

    def session(
        analysis: Analysis,
        export: Path,
        agent: Any,
        system: str,
        task: str,
        env: ToolEnv,
        **kwargs: Any,
    ) -> SessionRecord:
        record = json.loads((analysis.root.parent / "run.json").read_text())
        assert record["repeat_reservation"] is None and record["repeat"] is None
        kwargs["on_start"](analysis.root.parent / "agent-runs/call-1")
        record = json.loads((analysis.root.parent / "run.json").read_text())
        assert Path(record["repeat_reservation"]["reservation"]).is_file()
        assert "w99" not in system + task
        assert agent.model == args.model
        assert kwargs["audit"].experiment_days == (3, 3)
        assert "request_experiment" not in kwargs["tools"]
        calls.append(record["repeat"])
        return save_session(analysis, export, agent, fleet=fleet)

    monkeypatch.setattr(runner, "run_session", session)
    for name in ("one", "two"):
        args.out = args.out.parent / name
        assert runner.analyse(args) == 0
        record = json.loads((args.out / "run.json").read_text())
        assert record["model"] == "gpt-6-luna"
        assert record["lost_twice"] is False
        account = record["accounting"]
        assert account["model_seconds"] == 2 and account["launch_audit_seconds"] == 0.25
        assert account["price"]["low_usd"] > 0
        assert account["queue_seconds"] == account["simulation_seconds"] == 0
    assert calls == [1, 2]
    assert [call.args[-1] for call in score.call_args_list] == [1, 2]


@pytest.mark.parametrize("error", ["auth", "rate_limit"])
def test_loop_stops_before_queue_after_auth_or_rate_error(
    prepared: Any, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    args, fleet, score = prepared
    args.condition = "experiments"
    monkeypatch.setattr(
        runner,
        "run_session",
        lambda analysis, export, agent, *a, **kw: save_session(
            analysis, export, agent, queued=True, error=error, fleet=fleet, on_start=kw["on_start"]
        ),
    )
    dispatch = Mock(side_effect=AssertionError("must not dispatch"))
    monkeypatch.setattr(runner.DevelopmentEngine, "run_queue", dispatch)
    assert runner.analyse(args) == 1
    assert not dispatch.called and score.call_count == 1
    assert (
        json.loads((args.out / "run.json").read_text())["accounting"]["usage"]["input_tokens"]
        == 100
    )


@pytest.mark.parametrize(
    ("errors", "starts", "lost_twice"),
    [
        (["timeout", "timeout"], 2, True),
        (["timeout", "overloaded"], 2, True),
        (["timeout", "malformed_output"], 2, False),
        (["malformed_output"], 1, False),
    ],
)
def test_analysis_record_exposes_two_lost_starts_without_reading_session_files(
    prepared: Any,
    monkeypatch: pytest.MonkeyPatch,
    errors: list[str],
    starts: int,
    lost_twice: bool,
) -> None:
    args, _, score = prepared
    calls = []

    def fake_agent(*a: Any, **kw: Any) -> AgentResult:
        calls.append((a, kw))
        number = len(calls)
        call = kw["run_root"] / f"start-{number}"
        call.mkdir(parents=True)
        kw["on_start"](call)
        result = AgentResult(
            ok=False,
            result=None,
            usage={"input_tokens": 100, "output_tokens": 2},
            seconds=3,
            error=errors[number - 1],
            error_detail=f"start {number} failed",
            text="not JSON" if errors[number - 1] == "malformed_output" else "",
            run_dir=str(call),
        )
        (call / "result.json").write_text(json.dumps(asdict(result)))
        return result

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    assert runner.analyse(args) == 1
    assert len(calls) == starts and all(call == calls[0] for call in calls)
    run = json.loads((args.out / "run.json").read_text())
    assert run["state"] == "failed" and run["lost_twice"] is lost_twice
    assert run["repeat"] == 1 and len(list(accounting.REPEAT_REGISTER.rglob("*.json"))) == 1
    assert run["accounting"]["model_seconds"] == 3 * starts
    assert run["accounting"]["usage"]["input_tokens"] == 100 * starts
    assert len(run["accounting"]["calls"]) == starts
    assert score.call_count == 1  # evaluator sees the finalized run.json
    saved = json.loads((Path(run["analysis"]) / "sessions/session-01.json").read_text())
    assert saved["lost_twice"] is lost_twice and saved["rerun"] is (starts == 2)
    assert [a["error"] for a in saved["attempts"]] == errors
    assert [a["seconds"] for a in saved["attempts"]] == [3] * starts


def test_resume_after_completed_queue_keeps_results_and_never_dispatches_twice(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fleet, score = prepared
    args.condition = "experiments"
    interrupted = False

    def session(analysis: Analysis, export: Path, agent: Any, *a: Any, **kw: Any) -> SessionRecord:
        nonlocal interrupted
        if not analysis.sessions_done():
            return save_session(
                analysis, export, agent, queued=True, fleet=fleet, on_start=kw["on_start"]
            )
        if not interrupted:
            interrupted = True
            raise RuntimeError("synthetic interruption after simulation")
        assert analysis.results_by_id()["x0001"].status == "done"
        return save_session(analysis, export, agent, fleet=fleet, on_start=kw["on_start"])

    def dispatch(command: list[str], **kwargs: Any) -> None:
        queue_file = Path(command[command.index("--queue") + 1])
        q = ExperimentQueue.model_validate_json(queue_file.read_text())
        engine = Path(command[command.index("--engine-root") + 1])
        result_export = engine / "completed-export"
        shutil.copytree(q.export, result_export)
        table = pq.read_table(result_export / "log" / "episodes.parquet")
        original = table.to_pylist()[0]
        arm_ids = ["e010001", "e010002"]
        new_rows = [
            {**original, "episode_id": e, "robot_id": "r01", "experiment_id": "x0001", "day": 3}
            for e in arm_ids
        ]
        pq.write_table(
            pa.concat_tables([table, pa.Table.from_pylist(new_rows, schema=table.schema)]),
            result_export / "log" / "episodes.parquet",
        )
        stored_run = engine / "experiments" / q.analysis_id / "session-01"
        stored_run.mkdir(parents=True)
        (stored_run / "run.json").write_text('{"synthetic_completed_queue": true}')
        reply = EngineReply(
            export=result_export,
            results=[
                ExperimentResult(
                    experiment_id="x0001",
                    status="done",
                    message="done",
                    arms=[ArmResult(arm_index=i, episode_ids=[e]) for i, e in enumerate(arm_ids)],
                )
            ],
        )
        queue_file.with_suffix(".reply.json").write_text(reply.model_dump_json())
        timing_file = queue_file.with_suffix(".timing.json")
        timing = json.loads(timing_file.read_text())
        timing.update(queue_seconds=5, simulation_seconds=7, state="done", ended=1)
        timing_file.write_text(json.dumps(timing))

    dispatch_mock = Mock(side_effect=dispatch)
    monkeypatch.setattr(runner, "run_session", session)
    monkeypatch.setattr(subprocess, "run", dispatch_mock)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.analyse(args)
    source = args.out
    before = tree_manifest(source)
    args.resume_from = source
    args.out = source.parent / "continued"
    assert runner.analyse(args) == 0
    assert dispatch_mock.call_count == 1
    assert tree_manifest(source) == before
    record = json.loads((args.out / "run.json").read_text())
    assert record["repeat"] == 1
    assert record["experiments"] == {
        "requested": 1,
        "completed": 1,
        "refused": 0,
        "pending": 0,
        "delivered_episodes": 2,
    }
    account = record["accounting"]
    assert account["queue_seconds"] == 5 and account["simulation_seconds"] == 7
    assert account["model_seconds"] == 4 and account["launch_audit_seconds"] == 0.5
    assert len(account["queue_attempts"]) == 1 and len(account["calls"]) == 2
    assert score.call_count == 2


def test_repeat_register_is_atomic_fixed_and_keys_model_and_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accounting, "REPEAT_REGISTER", tmp_path / "fixed-register")
    group = repeat_group("world", "logs", "library", "gpt-6-astra", "xhigh")
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(
            pool.map(
                lambda i: reserve_repeat(group, tmp_path / str(i), tmp_path / str(i) / "call"),
                range(6),
            )
        )
    assert sorted(r["repeat"] for r in records) == list(range(1, 7))
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "another-agent-root"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "another-cache"))
    assert reserve_repeat(group, tmp_path, tmp_path / "call")["repeat"] == 7
    for key in ("world_hash", "condition", "library_revision", "model", "effort"):
        assert (
            reserve_repeat({**group, key: "different"}, tmp_path, tmp_path / "call")["repeat"] == 1
        )
    for receipt in records:
        verify_repeat(receipt, group)
        with pytest.raises(ValueError):
            verify_repeat({**receipt, "repeat": 42}, group)
        with pytest.raises(ValueError):
            verify_repeat(receipt, {**group, "effort": "medium"})
    copied = tmp_path / "alternate.json"
    copied.write_bytes(Path(records[0]["reservation"]).read_bytes())
    with pytest.raises(ValueError, match="fixed register"):
        verify_repeat({**records[0], "reservation": str(copied)}, group)


def test_price_bounds_include_cache_and_never_double_count_reasoning() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "cached_input_tokens": 200_000,
        "output_tokens": 100_000,
        "reasoning_tokens": 80_000,
    }
    price = price_equivalent(usage, PRICES["gpt-6-astra"])
    assert price["low_usd"] == pytest.approx(26.4)
    assert price["high_usd"] == pytest.approx(55.8)
    assert price_equivalent({}, PRICES["gpt-6-astra"])["status"] == "usage_unavailable"
    with pytest.raises(ValueError, match="invalid token"):
        price_equivalent({**usage, "cached_input_tokens": 2_000_000}, PRICES["gpt-6-astra"])


def test_refused_queue_still_gets_a_final_answer(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fleet, score = prepared
    args.condition = "experiments"

    def session(analysis: Analysis, export: Path, agent: Any, *a: Any, **kw: Any) -> SessionRecord:
        if not analysis.sessions_done():
            return save_session(
                analysis, export, agent, queued=True, fleet=fleet, on_start=kw["on_start"]
            )
        assert analysis.results_by_id()["x0001"].message == "invalid_action_value"
        return save_session(analysis, export, agent, fleet=fleet, on_start=kw["on_start"])

    monkeypatch.setattr(runner, "run_session", session)
    monkeypatch.setattr(
        runner.DevelopmentEngine,
        "run_queue",
        lambda self, path, q: EngineReply(
            export=Path(q.export),
            results=[
                ExperimentResult(
                    experiment_id="x0001", status="refused", message="invalid_action_value", arms=[]
                )
            ],
        ),
    )
    assert runner.analyse(args) == 0
    assert score.call_count == 1


def test_failed_dispatch_records_wait_and_preserves_analysis_checkpoint(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fleet, score = prepared
    args.condition = "experiments"
    monkeypatch.setattr(
        runner,
        "run_session",
        lambda analysis, export, agent, *a, **kw: save_session(
            analysis, export, agent, queued=True, fleet=fleet, on_start=kw["on_start"]
        ),
    )
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "stub"))
    )
    with pytest.raises(subprocess.CalledProcessError):
        runner.analyse(args)
    record = json.loads((args.out / "run.json").read_text())
    assert record["state"] == "waiting" and record["ended"]
    account = record["accounting"]
    assert account["queue_seconds"] >= 0 and account["simulation_seconds"] == 0
    assert account["queue_attempts"][0]["state"] == "dispatch_failed"
    assert score.call_count == 1


def test_missing_cli_result_marks_cost_and_duration_incomplete(tmp_path: Path) -> None:
    call = tmp_path / "agent-runs" / "interrupted"
    call.mkdir(parents=True)
    (call / "prompt.json").write_text("{}")
    account = collect_accounting(
        tmp_path,
        {"model": "gpt-6-astra", "prices": PRICES["gpt-6-astra"], "started": 1, "ended": 2},
        10,
    )
    assert account["price"]["status"] == "incomplete_usage"
    assert account["price"]["low_usd"] is None
    assert account["model_seconds"] is None
    assert account["calls"][0]["error"] == "missing_result"


def test_interrupted_caller_cannot_declare_the_queue_ended(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fleet, _ = prepared
    args.condition = "experiments"
    monkeypatch.setattr(
        runner,
        "run_session",
        lambda analysis, export, agent, *a, **kw: save_session(
            analysis, export, agent, queued=True, fleet=fleet, on_start=kw["on_start"]
        ),
    )
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        runner.analyse(args)
    record = json.loads((args.out / "run.json").read_text())
    timing = record["accounting"]["queue_attempts"][0]
    assert timing["state"] == "interrupted" and "ended" not in timing
    assert record["accounting"]["queue_seconds"] is None
    assert record["accounting"]["simulation_seconds"] is None
    with pytest.raises(ValueError, match="queue has not ended"):
        runner.copy_waiting_analysis(args.out, args.out.parent / "unsafe-copy", record)


def test_analyse_cli_routes_the_ablation_switch(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, fleet, _ = prepared
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyse",
            "run",
            "--world-root",
            str(args.world_root),
            "--export",
            str(args.export),
            "--out",
            str(args.out),
            "--condition",
            "logs",
            "--ablate-fault-classes",
        ],
    )

    def session(
        analysis: Analysis,
        export: Path,
        agent: Any,
        system: str,
        task: str,
        env: ToolEnv,
        **kw: Any,
    ) -> SessionRecord:
        assert env.skills is not None
        text = "\n".join(p.read_text() for p in env.skills.rglob("SKILL.md"))
        assert "fault-class-knowledge" not in text
        assert "Public action vocabulary" in text
        assert "Public vocabulary for hypotheses" not in text
        return save_session(analysis, export, agent, fleet=fleet, on_start=kw["on_start"])

    monkeypatch.setattr(runner, "run_session", session)
    assert runner.main() == 0
    record = json.loads((args.out / "run.json").read_text())
    assert record["ablate_fault_classes"] and record["model"] == "gpt-6-astra"


def test_queue_worker_records_each_phase_without_running_a_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine import experiments

    queue = tmp_path / "queue.json"
    timing = queue.with_suffix(".timing.json")
    timing.write_text('{"dispatch_id": "synthetic", "state": "queued"}')
    args = argparse.Namespace(
        queue=queue,
        queued_at=time.time() - 10,
        truth=tmp_path / "truth.json",
        base_run=[],
        workers=2,
        engine_root=tmp_path / "engine",
        reply=tmp_path / "reply.json",
    )
    monkeypatch.setattr(experiments.WorldTruth, "load", Mock(return_value=None))

    def simulated(*a: Any, **kw: Any) -> EngineReply:
        kw["timings"].update(prepare_seconds=1, simulation_seconds=2, export_seconds=3)
        return EngineReply(export=tmp_path, results=[])

    monkeypatch.setattr(experiments, "run_queue", simulated)
    runner.queue_main(args)
    record = json.loads(timing.read_text())
    assert record["state"] == "done" and record["queue_seconds"] >= 10
    phases = [record[k] for k in ("prepare_seconds", "simulation_seconds", "export_seconds")]
    assert phases == [1, 2, 3]
    assert record["ended"] >= record["started"] and args.reply.is_file()


def test_refused_pipeline_launch_does_not_consume_repeat_one(
    prepared: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.agents.checks import GateError

    args, fleet, score = prepared
    monkeypatch.setattr(Batch, "after_gate", Mock(side_effect=GateError("gate refuses")))
    with pytest.raises(GateError):
        runner.analyse(args)
    metadata = json.loads((args.out / "run.json").read_text())
    assert metadata["repeat"] is None and metadata["repeat_reservation"] is None
    assert not accounting.REPEAT_REGISTER.exists() and not score.called
    monkeypatch.setattr(Batch, "after_gate", Mock(return_value=SimpleNamespace(records={})))
    monkeypatch.setattr(runner, "run_session", Mock(side_effect=GateError("audit refuses")))
    args.out = args.out.parent / "audit-refused"
    with pytest.raises(GateError):
        runner.analyse(args)
    metadata = json.loads((args.out / "run.json").read_text())
    assert metadata["repeat"] is None and not accounting.REPEAT_REGISTER.exists()
    assert not score.called
    monkeypatch.setattr(
        runner,
        "run_session",
        lambda analysis, export, agent, *a, **kw: save_session(
            analysis,
            export,
            agent,
            fleet=fleet,
            on_start=kw["on_start"],
        ),
    )
    args.out = args.out.parent / "first-started"
    assert runner.analyse(args) == 0
    metadata = json.loads((args.out / "run.json").read_text())
    assert metadata["repeat"] == 1
    assert metadata["repeat_reservation"]["first_call"].endswith("agent-runs/call-1")


def test_machine_failure_is_resumable_and_counts_only_delivered_experiments(
    prepared: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.engine.episode import EngineError

    args, fleet, _ = prepared
    args.condition = "experiments"
    session = Mock(
        side_effect=lambda analysis, export, agent, *a, **kw: save_session(
            analysis,
            export,
            agent,
            fleet=fleet,
            queued=True,
            on_start=kw["on_start"],
        )
    )
    monkeypatch.setattr(runner, "run_session", session)
    failed = Mock(side_effect=EngineError("lost worker"))
    monkeypatch.setattr(runner.DevelopmentEngine, "run_queue", failed)
    with pytest.raises(EngineError, match="lost worker"):
        runner.analyse(args)
    metadata = json.loads((args.out / "run.json").read_text())
    assert (
        metadata["state"] == "waiting" and metadata["stop_reason"] == "experiment_execution_failed"
    )
    assert metadata["experiments"] == {
        "requested": 1,
        "completed": 0,
        "refused": 0,
        "pending": 1,
        "delivered_episodes": 0,
    }
    assert failed.call_count == session.call_count == 1
    source = args.out
    original = tree_manifest(source)
    monkeypatch.setattr(
        runner.DevelopmentEngine,
        "run_queue",
        lambda self, path, q: EngineReply(
            export=Path(q.export),
            results=[
                ExperimentResult(
                    experiment_id="x0001",
                    status="refused",
                    arms=[],
                    message="invalid_action_value",
                )
            ],
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_session",
        lambda analysis, export, agent, *a, **kw: save_session(
            analysis,
            export,
            agent,
            fleet=fleet,
            on_start=kw["on_start"],
        ),
    )
    args.resume_from, args.out = source, source.parent / "resumed"
    assert runner.analyse(args) == 0
    metadata = json.loads((args.out / "run.json").read_text())
    assert metadata["repeat"] == 1 and tree_manifest(source) == original
    assert metadata["experiments"] == {
        "requested": 1,
        "completed": 0,
        "refused": 1,
        "pending": 0,
        "delivered_episodes": 0,
    }
