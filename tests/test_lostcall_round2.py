"""Lost-call review repairs, using CPU stand-ins rather than model sessions."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from test_agents import World

from fleetlab.agents.audit import tree_manifest
from fleetlab.agents.run import AgentResult, ToolAccess, lost_runtime_answer, parse_codex
from fleetlab.agents.sessions import AgentSpec, Analysis, EngineReply, ExperimentQueue, run_analysis
from fleetlab.baselines.models import BaselineModelError, model_call
from fleetlab.contracts.actions import catalogue
from fleetlab.contracts.tools import ExperimentResult
from fleetlab.integration import analyse as runner
from fleetlab.toolserver.files import EXPERIMENTS_FILE, append_line

pytest_plugins = ["test_agents", "test_analysis_pipeline"]


def test_replacement_restores_memory_and_working_files(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis = Analysis.create("w01", base=root / "analyses")
    (analysis.memory / "notes.md").write_text("before")
    (analysis.memory / "scratch").mkdir()
    (analysis.memory / "scratch" / "old.txt").write_text("original")
    (analysis.outbox / "work.txt").write_text("original work")
    before = {p: tree_manifest(analysis.root / p) for p in ("memory", "outbox")}
    calls: list[Any] = []

    def agent(*a: Any, tools: ToolAccess, **kw: Any) -> AgentResult:
        calls.append((a, kw["timeout"]))
        assert {p: tree_manifest(analysis.root / p) for p in before} == before
        if len(calls) == 1:
            (tools.memory / "notes.md").write_text("lost work")
            (tools.memory / "scratch" / "old.txt").unlink()
            (tools.memory / "new.txt").write_text("extra work")
            (tools.outbox / "work.txt").write_text("lost working file")
            return AgentResult(False, None, {}, 1800, error="timeout")
        return AgentResult(True, {"status": "done", "summary": "finished"}, {}, 5)

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", agent)
    record = run_analysis(
        analysis, world.export, AgentSpec("codex", "gpt-6-astra"), "S", "T", world.env, None
    )
    assert len(calls) == 2 and calls[0] == calls[1]
    assert record.sessions[0].seconds == 1805
    discarded = analysis.root / "session-state" / "session-01" / "lost-start-1"
    assert (discarded / "memory" / "notes.md").read_text() == "lost work"
    assert (discarded / "memory" / "new.txt").read_text() == "extra work"
    assert (discarded / "outbox" / "work.txt").read_text() == "lost working file"


@pytest.mark.parametrize("completion", ["cut", "last_file", "completed_turn", "final_channel"])
def test_codex_commentary_is_not_an_answer_in_a_tool_session(
    tmp_path: Path, completion: str
) -> None:
    last = tmp_path / "last.txt"
    report = '{"status":"done","summary":"finished"}'
    events: list[dict[str, Any]] = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Checking."}},
    ]
    if completion == "last_file":
        last.write_text(report)
    elif completion == "completed_turn":
        events += [
            {"type": "item.completed", "item": {"type": "agent_message", "text": report}},
            {"type": "turn.completed", "usage": {}},
        ]
    elif completion == "final_channel":
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "channel": "final",
                    "text": report,
                },
            }
        )
    result = AgentResult(False, None, {}, 1, error="timeout")
    parse_codex([json.dumps(e) for e in events], result, last)
    assert lost_runtime_answer(result, with_tools=True) is (completion == "cut")
    assert result.text == ("Checking." if completion == "cut" else report)
    # No-tool baseline interpretation is unchanged, even on a cut stream.
    assert not lost_runtime_answer(result)


def test_twice_lost_analysis_cannot_be_called_again(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = Mock(return_value=AgentResult(False, None, {}, 1, error="timeout"))
    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    args = (analysis, world.export, AgentSpec("codex", "gpt-6-astra"), "S", "T", world.env, None)
    first = run_analysis(*args)
    second = run_analysis(*args)
    assert agent.call_count == 2
    assert second == first and second.sessions[0].lost_twice


TRANSPORT_PHRASES = [
    "transport error",
    "connection reset",
    "connection refused",
    "connection closed",
    "stream disconnected",
    "error sending request",
    "network is unreachable",
    "dns error",
    "tls handshake",
]


@pytest.mark.parametrize("phrase", TRANSPORT_PHRASES)
def test_all_nine_transport_phrases_get_one_replacement(phrase: str) -> None:
    result = AgentResult(
        False, None, {}, 1, error="cli_error", error_detail=f"CLI: {phrase.upper()}!"
    )
    assert lost_runtime_answer(result, with_tools=True)
    caller = Mock(return_value=result)
    with pytest.raises(BaselineModelError) as failure:
        model_call("same prompt", caller=caller, label="test")
    assert caller.call_count == 2 and len(failure.value.attempts) == 2
    assert caller.call_args_list[0] == caller.call_args_list[1]


@pytest.mark.parametrize("dispatched", [False, True])
def test_resume_counts_every_start_and_never_grants_a_third(
    prepared: Any, monkeypatch: pytest.MonkeyPatch, dispatched: bool
) -> None:
    args, fleet, _ = prepared
    args.condition = "experiments"
    calls: list[int] = []
    interrupted = False
    counter = itertools.count(1)

    def agent(*a: Any, tools: ToolAccess, **kw: Any) -> AgentResult:
        nonlocal interrupted
        number = next(counter)
        call = kw["run_root"] / f"start-{number}"
        if tools.session == 2 and len(calls) == 2 and not interrupted:
            interrupted = True
            saved = json.loads((tools.memory.parent / "sessions/session-02.json").read_text())
            assert len(saved["attempts"]) == 1
            assert saved["attempts"][0]["run_dir"] == str(args.out / "agent-runs/start-2")
            if dispatched:
                call.mkdir(parents=True)
                (call / "prompt.json").write_text("{}")
                kw["on_start"](call)
                calls.append(tools.session)
            raise RuntimeError("interruption before result")
        call.mkdir(parents=True)
        (call / "prompt.json").write_text("{}")
        kw["on_start"](call)
        calls.append(tools.session)
        (call / "audit.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "manifest": {"export": tree_manifest(tools.export)},
                }
            )
        )
        if tools.session == 1:
            action = next(x for x in catalogue(fleet) if x.is_complete and x.has_replay)
            append_line(
                tools.outbox / EXPERIMENTS_FILE,
                {
                    "experiment_id": "x0001",
                    "session": 1,
                    "request": {
                        "arms": [{"actions": []}, {"actions": [action.model_dump(mode="json")]}],
                        "robots": ["r01"],
                        "episodes_per_arm": 1,
                        "paired_starts": True,
                        "power_note": "test",
                    },
                },
            )
            result = AgentResult(True, {"status": "waiting", "summary": "queued"}, {}, 2)
        else:
            result = AgentResult(False, None, {}, 1800, error="timeout")
        result.run_dir = str(call)
        (call / "result.json").write_text(json.dumps(asdict(result)))
        return result

    def dispatch(command: list[str], **kw: Any) -> None:
        queue_file = Path(command[command.index("--queue") + 1])
        queue = ExperimentQueue.model_validate_json(queue_file.read_text())
        reply = EngineReply(
            export=Path(queue.export),
            results=[
                ExperimentResult(
                    experiment_id=q.experiment_id,
                    status="refused",
                    message="request_not_runnable",
                    arms=[],
                )
                for q in queue.experiments
            ],
        )
        queue_file.with_suffix(".reply.json").write_text(reply.model_dump_json())
        timing_file = queue_file.with_suffix(".timing.json")
        timing = json.loads(timing_file.read_text())
        timing.update(state="done", ended=1, queue_seconds=1, simulation_seconds=0)
        timing_file.write_text(json.dumps(timing))

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", agent)
    dispatch_mock = Mock(side_effect=dispatch)
    monkeypatch.setattr("subprocess.run", dispatch_mock)
    with pytest.raises(RuntimeError, match="interruption before result"):
        runner.analyse(args)
    source = args.out
    original = tree_manifest(source)
    args.resume_from = source
    args.out = source.parent / "continued"
    assert runner.analyse(args) == 1
    assert calls == [1, 2, 2]
    assert dispatch_mock.call_count == 1
    assert tree_manifest(source) == original
    run = json.loads((args.out / "run.json").read_text())
    saved = json.loads((Path(run["analysis"]) / "sessions/session-02.json").read_text())
    assert saved["lost_twice"] and run["lost_twice"]
    assert saved["state"] == run["state"] == "failed"
    assert len(saved["attempts"]) == 2 and saved["rerun"]
    assert saved["attempts"][0]["run_dir"] == str(source / "agent-runs/start-2")
    assert len(run["accounting"]["calls"]) == 3
    assert run["accounting"]["model_seconds"] == (None if dispatched else 3602)
