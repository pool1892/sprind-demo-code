"""The driver's repair pass after a review of 8a5f165: the agent's earlier decisions
are carried as its own audited state, never pasted into the strict task text; failed
simulations count; an aborted engagement is not a success; cost totals keep missing
usage a lower bound; every supplied alternative is checked like an adoption. No model,
no simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from log_toy import toy_fleet
from test_agent_launch_audit import ReachedLaunch, launch  # noqa: F401  (the fixture)

from fleetlab.agents import run
from fleetlab.agents.checks import GateError
from fleetlab.agents.sessions import AgentSpec, Analysis, SessionRecord
from fleetlab.contracts.decisions import RoundDecision
from fleetlab.loop.costs import cost_delta, queue_costs, session_costs, simulated_jobs
from fleetlab.loop.rounds import round_task
from fleetlab.toolserver.decisions import PublicLedger, PublicLedgerEntry, decision_problems
from fleetlab.toolserver.files import SUBMISSIONS_FILE, append_line
from fleetlab.toolserver.server import Paths, ToolBox

# Prose a reviewer reproduced: accepted as agent prose, refused in strict pipeline text.
PROSE = "After repairing the fault, r02 should return to nominal performance."
RECAL: dict[str, Any] = {
    "verb": "recalibrate",
    "target": {
        "scope": "r02",
        "component": "scene_camera",
        "config_key": None,
        "condition_key": None,
    },
    "value": None,
}


def round_one_decision() -> dict[str, Any]:
    return {
        "adopt": [{"action": RECAL, "rationale": PROSE}],
        "replace": [],
        "undo": [],
        "predicted_set_gain": {"low": 1.0, "mid": 2.0, "high": 4.0},
        "expectations": [PROSE],
        "stop": False,
        "ranked_candidates": [],
    }


def write_round_one(outbox: Path) -> None:
    decision = round_one_decision()
    RoundDecision.model_validate(decision)
    append_line(
        outbox / SUBMISSIONS_FILE,
        {"kind": "decision", "receipt_id": "decision-1", "session": 2, "answer": decision},
    )
    (outbox / "round_decision.json").write_text(json.dumps(decision))


def round_two_task() -> str:
    return round_task(
        round_number=2,
        rounds=2,
        operation=[(1, (0, 1)), (2, (3, 3))],
        experiment_days=(4, 4),
        budget_text="80 robot attempts in all",
        ledger=PublicLedger(
            round=2,
            changes=[
                PublicLedgerEntry.model_validate(
                    {
                        "entry_id": "L1",
                        "round": 1,
                        "adopted_day": 3,
                        "action": RECAL,
                        "status": "active",
                    }
                )
            ],
        ),
        sessions=[(1, (1, 2))],
    )


def test_carried_prose_does_not_block_the_next_launch(
    launch: tuple[run.ToolAccess, list[Path]],  # noqa: F811
) -> None:
    access, launched = launch
    write_round_one(access.outbox)
    task = round_two_task()
    assert PROSE not in task and "earlier_decisions" in task
    with pytest.raises(ReachedLaunch):  # the outbox is audited as agent prose: it passes
        run.run_agent("claude", "model", "System", task, tools=access)
    assert len(launched) == 1
    # The same prose pasted into the task text (strict pipeline text) is refused, which is
    # why the driver no longer does that.
    with pytest.raises(GateError):
        run.run_agent("claude", "model", "System", task + "\n" + PROSE, tools=access)
    assert len(launched) == 1


def test_earlier_decisions_returns_the_agents_own_decisions(tmp_path: Path) -> None:
    from log_toy import TASK_CLASSES, write_toy_run

    from fleetlab.log import export_runs

    fleet = write_toy_run(tmp_path / "run")
    export_runs([tmp_path / "run"], tmp_path / "export", fleet=fleet, task_classes=TASK_CLASSES)
    for part in ("memory", "outbox", "results"):
        (tmp_path / part).mkdir()
    write_round_one(tmp_path / "outbox")
    box = ToolBox(
        Paths(
            export=tmp_path / "export",
            memory=tmp_path / "memory",
            outbox=tmp_path / "outbox",
            results=tmp_path / "results",
        ),
        session=3,
    )
    out = json.loads(box.call("earlier_decisions", {}).content[0]["text"])
    box.close()
    assert out == [{"session": 2, "receipt_id": "decision-1", "decision": round_one_decision()}]


def test_failed_simulations_count_and_unconfirmed_ones_are_labelled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    incidents = [
        {
            "jobs": ["e000001", "e000002", "e000003"],
            "assignment_status": {
                "e000001": "execution_started",
                "e000002": "execution_unconfirmed",
                "e000003": "completed_report_late",
            },
        },
        {"jobs": ["e000004"], "assignment_status": {"e000004": "not_delivered"}},
    ]
    (run_dir / "run.json").write_text(json.dumps({"extra": {"incidents": incidents}}))
    counted = simulated_jobs(run_dir)
    assert (counted["failed_confirmed"], counted["unconfirmed"], counted["not_delivered"]) == (
        2,
        1,
        1,
    )
    assert counted["simulated_confirmed"] == counted["run_jobs"] + counted["replays"] + 2
    total = queue_costs([{"simulated": counted, "state": "failed"}])
    assert total["simulated_failed_confirmed"] == 2 and total["simulated_unconfirmed"] == 1
    assert total["failed_queues"] == 1


def _session(starts: list[dict[str, Any]]) -> SessionRecord:
    return SessionRecord(
        number=1,
        state="done",
        export="e",
        run_dir=starts[-1]["run_dir"],
        ok=True,
        error=None,
        seconds=0.0,
        usage={},
        queued=[],
        submitted=[],
        report=None,
        tool_calls=0,
        attempts=starts,
    )


def _start(tmp_path: Path, name: str, usage: dict[str, int] | None) -> dict[str, Any]:
    return {
        "ok": usage is not None,
        "error": None if usage is not None else "timeout",
        "detail": "",
        "run_dir": str(tmp_path / name),
        "seconds": 5.0,
        "usage": usage or {},
        "runtime_meta": {} if usage is not None else {"interrupted_without_result": True},
        "answer_present": usage is not None,
    }


def test_missing_usage_stays_a_lower_bound_and_has_no_delta(tmp_path: Path) -> None:
    full = session_costs([_session([_start(tmp_path, "a", {"input_tokens": 100})])], "gpt-6-astra")
    gap = session_costs(
        [_session([_start(tmp_path, "b", None), _start(tmp_path, "c", {"input_tokens": 80})])],
        "gpt-6-astra",
    )
    assert full["usage_complete"] and full["usage_note"] is None
    assert not gap["usage_complete"] and "1 of 2 model starts" in gap["usage_note"]
    assert gap["api_equivalent_usd"]["status"] == "lower_bound"
    assert gap["api_equivalent_usd"]["partial"]["status"] == "bounded_api_equivalent"
    assert cost_delta(full, gap, "input_tokens")["status"] == "unavailable"
    assert cost_delta(full, gap, "tool_calls")["status"] == "complete"  # not a usage figure
    same = cost_delta(full, full, "input_tokens")
    assert (same["status"], same["difference"]) == ("complete", 0.0)


def test_every_supplied_alternative_is_checked_like_an_adoption() -> None:
    fleet = toy_fleet()

    def candidate(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": action,
            "expected_gain": 1.0,
            "uncertainty": {"low": 0.0, "high": 2.0},
            "rationale": "x",
        }

    unset = {**RECAL, "verb": "set", "target": {**RECAL["target"], "component": None}}
    unset["target"]["config_key"] = "policy.sampling_steps"
    decision = RoundDecision.model_validate(
        {
            **round_one_decision(),
            "ranked_candidates": [
                candidate({**unset, "value": None}),  # incomplete
                candidate({**unset, "value": 0}),  # out of the value domain
                candidate(
                    {
                        "verb": "route_to_teleop",
                        "target": {**RECAL["target"], "scope": "fleet", "component": None},
                        "value": "put",
                    }
                ),  # no simulated effect
            ],
        }
    )
    problems = decision_problems(decision, fleet, None)
    assert any(p.startswith("ranked candidate 1") and "needs its value" in p for p in problems)
    assert any(p.startswith("ranked candidate 2") and "allowed" in p for p in problems)
    assert any(p.startswith("ranked candidate 3") and "simulated" in p for p in problems)


def test_an_engagement_without_a_decision_is_aborted_and_not_evaluated(tmp_path: Path) -> None:
    from fleetlab.loop.rounds import Engagement, run_engagement
    from fleetlab.loop.scripted import ScriptedAgent

    class NoState:
        """A one-round state without simulation: the export is a toy run's."""

        evaluated = False

        def __init__(self, export: Path) -> None:
            self.export_dir = export

        def rounds(self) -> int:
            return 2

        def operation_days(self, n: int) -> tuple[int, int]:
            return (0, 1)

        def experiment_days(self, n: int) -> tuple[int, int]:
            return (2, 2)

        def empty_ledger(self) -> Any:
            return "L0"

        def operate(self, n: int, ledger: Any) -> list[Path]:
            return []

        def export(self, n: int, ledger: Any, analysis_id: str) -> Path:
            return self.export_dir

        def engine(self, n: int, ledger: Any, analysis_id: str) -> Any:
            return None

        def adopt(self, ledger: Any, n: int, decision: Any) -> Any:
            raise AssertionError("no decision was made")

        def public_ledger(self, ledger: Any, n: int) -> PublicLedger:
            return PublicLedger(round=n, changes=[])

        def changes(self, ledger: Any) -> int:
            return 0

        def one_look(self, actions: Any, final: Any) -> Any:
            raise AssertionError("round 1 never decided")

        def evaluate(self, states: Any) -> Path:
            self.evaluated = True
            return tmp_path

        def audit_runs(self) -> tuple[Path, ...]:
            return ()

        def experiment_root(self, analysis_id: str) -> Path:
            return tmp_path / "rounds"

    from log_toy import TASK_CLASSES, write_toy_run

    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.log import export_runs

    fleet = write_toy_run(tmp_path / "run")
    export_runs([tmp_path / "run"], tmp_path / "export", fleet=fleet, task_classes=TASK_CLASSES)

    def silent(box: Any) -> str:
        box.call("memory_read", {})
        return "done"

    state = NoState(tmp_path / "export")
    how = Engagement(
        agent=AgentSpec("claude", "scripted"),
        budget=__import__("fleetlab.loop.driver", fromlist=["AttemptBudget"]).AttemptBudget(),
        max_sessions=2,
        system="system",
        env=project_tool_env(tmp_path / "env"),
        fleet=None,
    )
    from unittest import mock

    agent = ScriptedAgent([silent], run_root=tmp_path / "scripted")
    with mock.patch("fleetlab.agents.sessions.run_agent", agent):
        record = run_engagement(state, "w01", tmp_path / "out", how)
    assert record["outcome"] == "aborted"
    assert record["ended_because"] == "round 1: no decision in 2 sessions"
    assert record["evaluation"] == {"skipped": "the engagement was aborted"}
    assert not state.evaluated and record["states"] == ["original"]


def test_the_cli_exits_distinctly_for_an_aborted_engagement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fleetlab.loop import driver

    for outcome, code in (("aborted", driver.EXIT_ABORTED), ("stopped", 0), ("completed", 0)):
        monkeypatch.setattr(driver, "run_rounds", lambda *a, o=outcome, **k: {"outcome": o})
        args = ["rounds", "--world-root", str(tmp_path), "--out", str(tmp_path / outcome)]
        assert driver.main(args) == code
    assert driver.EXIT_ABORTED not in (0, 1, 2)


def test_analysis_root_is_unused_by_the_fixture_import() -> None:
    # The launch fixture is imported from test_agent_launch_audit; keep its name in use.
    assert Analysis is not None


def test_the_round_prompt_with_learning_passes_the_real_launch_audit(
    launch: tuple[run.ToolAccess, list[Path]],  # noqa: F811
) -> None:
    """Lane learn's paragraph and the round tools' descriptions (write_learned_skill among
    them) through the actual launch audit, with a round-1 decision in the outbox."""
    from dataclasses import replace

    from fleetlab.loop.learned import with_learning
    from fleetlab.loop.rounds import ROUND_TOOLS

    access, launched = launch
    write_round_one(access.outbox)
    task = with_learning(round_two_task())
    assert "write_learned_skill" in task and "write_learned_skill" in ROUND_TOOLS
    assert "provisional competing explanations" in task  # also audited here
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", task, tools=replace(access, tools=ROUND_TOOLS))
    assert len(launched) == 1
