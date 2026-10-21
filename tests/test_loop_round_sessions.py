"""A round's sessions on the toy export with the scripted agent and a stand-in engine
(fleetlab.loop.rounds.run_round_sessions): a request, its (refused) result in the next
session, the budget's status, and a decision that adopts and undoes against the public
ledger. The tool server is the real one; no model, no simulator."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from log_toy import TASK_CLASSES, write_toy_run

from fleetlab.agents.audit import AuditContext
from fleetlab.agents.sessions import AgentSpec, Analysis, EngineReply, ExperimentQueue
from fleetlab.agents.toolenv import project_tool_env
from fleetlab.contracts.tools import ExperimentResult
from fleetlab.log import export_runs
from fleetlab.loop.costs import experiment_costs, session_costs
from fleetlab.loop.driver import AttemptBudget
from fleetlab.loop.rounds import ROUND_TOOLS, publish_ledger, run_round_sessions
from fleetlab.loop.scripted import ScriptedAgent, decide_step
from fleetlab.toolserver.decisions import PublicLedger, PublicLedgerEntry
from fleetlab.toolserver.public_results import REFUSED

RECAL = {
    "verb": "recalibrate",
    "target": {
        "scope": "r02",
        "component": "scene_camera",
        "config_key": None,
        "condition_key": None,
    },
    "value": None,
}
REVERT = {
    "verb": "revert",
    "target": {
        "scope": "r02",
        "component": None,
        "config_key": "control.action_delay_steps",
        "condition_key": None,
    },
    "value": None,
}
REQUEST = {
    "arms": [{"actions": []}, {"actions": [RECAL]}],
    "robots": ["r02"],
    "episodes_per_arm": 4,
    "paired_starts": True,
    "power_note": "a toy",
}


class RefusingEngine:
    """Answers every request "refused" (no arms), on the same export."""

    def __init__(self, export: Path) -> None:
        self.export = export
        self.queues: list[ExperimentQueue] = []

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        self.queues.append(queue)
        return EngineReply(
            results=[
                ExperimentResult(
                    experiment_id=q.experiment_id, status="refused", arms=[], message=REFUSED
                )
                for q in queue.experiments
            ],
            export=self.export,
        )


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Any]]:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    fleet = write_toy_run(tmp_path / "run")
    export_runs([tmp_path / "run"], tmp_path / "export", fleet=fleet, task_classes=TASK_CLASSES)
    yield tmp_path / "export", project_tool_env(tmp_path / "env")


def test_a_round_requests_reads_and_decides(tmp_path: Path, toy: tuple[Path, Any]) -> None:
    export, env = toy
    analysis = Analysis.create("w01", base=tmp_path / "analyses")
    publish_ledger(
        analysis,
        PublicLedger(
            round=2,
            changes=[
                PublicLedgerEntry.model_validate(
                    {
                        "entry_id": "L1",
                        "round": 1,
                        "adopted_day": 2,
                        "action": REVERT,
                        "status": "active",
                    }
                )
            ],
        ),
    )
    agent = ScriptedAgent(
        [decide_step([RECAL], undo=[{"entry_id": "L1"}], request_first=REQUEST)],
        run_root=tmp_path / "scripted",
    )
    engine = RefusingEngine(export)
    budget = AttemptBudget(attempts=40, per_arm=4)
    from unittest import mock

    with mock.patch("fleetlab.agents.sessions.run_agent", agent):
        outcome = run_round_sessions(
            analysis,
            export,
            AgentSpec("claude", "scripted"),
            "system",
            "the task",
            env,
            engine,
            round_number=2,
            limits=budget.limits(first_session=1),
            attempts=budget.attempts,
            max_sessions=3,
            audit=AuditContext((), None, None, (2, 3)),
            run_root=tmp_path / "runs",
            tools=ROUND_TOOLS,
        )
    assert [s.state for s in outcome.sessions] == ["waiting", "done"]
    assert outcome.decision is not None and outcome.stopped is None
    assert [c.action.verb.value for c in outcome.decision.adopt] == ["recalibrate"]
    assert [u.entry_id for u in outcome.decision.undo] == ["L1"]
    assert [q.experiment_id for q in engine.queues[0].experiments] == ["x0001"]
    # The second session saw the refusal and the budget: a refused request costs nothing.
    calls = [
        json.loads(line)
        for line in (Path(outcome.sessions[1].run_dir) / "box" / "toolbox" / "tool_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    budget_call = next(c for c in calls if c["tool"] == "experiment_budget")
    status = json.loads(budget_call["result"][0])
    assert (status["attempts"], status["attempts_used"], status["attempts_left"]) == (40, 0, 40)
    assert "round 2" in agent.seen[1]["prompt"] and "experiments" in agent.seen[1]["prompt"]
    stored = json.loads((analysis.outbox / "round_decision.json").read_text())
    assert stored["undo"][0]["entry_id"] == "L1"
    first = Path(outcome.sessions[0].run_dir) / "box" / "toolbox" / "tool_calls.jsonl"
    costs = {**session_costs(outcome.sessions), **experiment_costs(analysis)}
    assert costs["model_starts"] == 2
    assert costs["tool_calls"] == len(calls) + len(first.read_text().splitlines()) == len(calls) + 5
    assert (costs["requests"], costs["requests_refused"], costs["arm_attempts"]) == (1, 1, 0)


def test_a_session_without_a_decision_is_reminded(tmp_path: Path, toy: tuple[Path, Any]) -> None:
    export, env = toy
    analysis = Analysis.create("w01", base=tmp_path / "analyses")

    def nothing(box: Any) -> str:
        box.call("memory_read", {})
        return "done"

    agent = ScriptedAgent([nothing, decide_step([RECAL])], run_root=tmp_path / "scripted")
    from unittest import mock

    with mock.patch("fleetlab.agents.sessions.run_agent", agent):
        outcome = run_round_sessions(
            analysis,
            export,
            AgentSpec("claude", "scripted"),
            "system",
            "the task",
            env,
            None,
            round_number=1,
            limits=AttemptBudget().limits(),
            attempts=AttemptBudget().attempts,
            max_sessions=2,
            audit=AuditContext((), None, None, (2, 3)),
            run_root=tmp_path / "runs",
        )
    assert [s.state for s in outcome.sessions] == ["ended", "done"]
    assert "without an accepted decision" in agent.seen[1]["prompt"]
    assert outcome.decision is not None
