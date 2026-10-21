"""A two-round toy engagement in the driver's real record format, for the exhibit lane.

Runs lane driver's `run_rounds` (snapshot of loop/driver at ef72804) on lane state's
CPU toy world with a scripted agent that writes notes, submits a diagnosis before
and after its experiments (so the working table has confidence before and after),
withdraws one suspicion, adopts one change per round and stops in round 2.
No model, no simulator, no GPU. Usage (from the driver snapshot's venv):

    python toy_engagement.py OUT
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import AgentSpec
from fleetlab.agents.toolenv import project_tool_env
from fleetlab.loop.driver import AttemptBudget, run_rounds
from fleetlab.loop.scripted import ScriptedAgent, _data
from fleetlab.loop.toy import write_toy_world
from fleetlab.toolserver.server import ToolBox

RECAL = {
    "verb": "recalibrate",
    "target": {"scope": "r02", "component": "scene_camera", "config_key": None, "condition_key": None},
    "value": None,
}
REVERT = {
    "verb": "revert",
    "target": {"scope": "r03", "component": None, "config_key": "control.action_delay_steps", "condition_key": None},
    "value": None,
}
CLEAN = {
    "verb": "clean",
    "target": {"scope": "r02", "component": "scene_camera", "config_key": None, "condition_key": None},
    "value": None,
}


def cause(target: dict[str, Any], direction: str, confidence: float, mode: str, episode: str | None = None) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = [
        {"episode_id": None, "step": None, "table": "episodes",
         "query": "SELECT robot_id, count(*) FROM episodes GROUP BY robot_id"}
    ]
    if episode:
        evidence.append({"episode_id": episode, "step": 0, "table": None, "query": None})
    return {"target": target, "direction": direction, "confidence": confidence,
            "evidence": evidence, "failure_mode": mode}


def diagnose(box: ToolBox, causes: list[dict[str, Any]], summaries: list[str]) -> None:
    receipt = _data(box.call("submit_diagnosis", {"causes": causes, "mode_summaries": summaries,
                                                   "warnings": [], "retrain": []}))
    if not receipt["accepted"]:
        raise RuntimeError(receipt["problems"])


def first_episode(box: ToolBox, robot: str) -> str:
    rows = _data(box.call("query", {"sql": f"SELECT episode_id FROM episodes WHERE robot_id = '{robot}' "
                                            "AND outcome = 'failure' ORDER BY episode_id LIMIT 1"}))["rows"]
    return str(rows[0][0]) if rows else ""


def r1_s1(box: ToolBox) -> str:
    _data(box.call("query", {"sql": "SELECT robot_id, avg(CAST(outcome = 'success' AS INT)) FROM episodes GROUP BY robot_id"}))
    ep = first_episode(box, "r02")
    diagnose(box, [
        cause(RECAL["target"], "degraded", 0.55, "toy: grasps miss on r02", ep or None),
        cause(REVERT["target"], "misconfigured", 0.35, "toy: late motion on r03"),
    ], ["toy: two suspicions before testing"])
    for arms, robot in (([[], [RECAL], [CLEAN]], "r02"), ([[], [REVERT]], "r03")):
        receipt = _data(box.call("request_experiment", {
            "arms": [{"actions": a} for a in arms], "robots": [robot], "episodes_per_arm": 4,
            "paired_starts": True, "power_note": f"toy power note for {robot}"}))
        if not receipt["accepted"]:
            raise RuntimeError(receipt["problems"])
    _data(box.call("memory_write", {"notes": "Round 1, session 1. Suspect r02 scene camera (0.55) and r03 delay (0.35). Requested two experiments."}))
    return "waiting"


def r1_s2(box: ToolBox) -> str:
    notes = _data(box.call("memory_read", {}))["notes"]
    _data(box.call("experiment_results", {}))
    _data(box.call("experiment_budget", {}))
    ep = first_episode(box, "r02")
    diagnose(box, [cause(RECAL["target"], "degraded", 0.85, "toy: grasps miss on r02", ep or None)],
             ["toy: the r03 delay suspicion is withdrawn after its experiment"])
    ranked = [
        {"action": CLEAN, "expected_gain": 1.0, "uncertainty": {"low": 0.0, "high": 3.0},
         "rationale": "toy: cleaning did not help in x0001"},
        {"action": REVERT, "expected_gain": 0.5, "uncertainty": {"low": 0.0, "high": 4.0},
         "rationale": "toy: the delay revert stays an alternative"},
    ]
    decision = {"adopt": [{"action": RECAL, "rationale": "toy: recalibration rescued r02 in x0001"}],
                "replace": [], "undo": [],
                "predicted_set_gain": {"low": 2.0, "mid": 5.0, "high": 9.0},
                "expectations": ["toy: r02 succeeds at least as often as the other robots"],
                "stop": False, "ranked_candidates": ranked}
    receipt = _data(box.call("submit_decision", decision))
    if not receipt["accepted"]:
        raise RuntimeError(receipt["problems"])
    _data(box.call("memory_write", {"notes": notes + "\nRound 1, session 2. Recalibration of r02 supported; r03 delay withdrawn. Decided."}))
    return "done"


def r2_s3(box: ToolBox) -> str:
    notes = _data(box.call("memory_read", {}))["notes"]
    diagnose(box, [cause(REVERT["target"], "misconfigured", 0.6, "toy: late motion on r03 remains")],
             ["toy: round 2 looks again at r03"])
    decision = {"adopt": [{"action": REVERT, "rationale": "toy: r03 still lags in round 2's records"}],
                "replace": [], "undo": [],
                "predicted_set_gain": {"low": 0.0, "mid": 3.0, "high": 6.0},
                "expectations": ["toy: r03 succeeds more often"], "stop": True, "ranked_candidates": []}
    receipt = _data(box.call("submit_decision", decision))
    if not receipt["accepted"]:
        raise RuntimeError(receipt["problems"])
    _data(box.call("memory_write", {"notes": notes + "\nRound 2, session 3. Reverted r03 delay; stop."}))
    return "done"


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["ROBOTS_AGENT_ROOT"] = str(root / "agent-root")
    world, setup = write_toy_world(root / "world-root")
    agent = ScriptedAgent([r1_s1, r1_s2, r2_s3], run_root=root / "scripted")
    record = run_rounds(world, root / "out", agent=AgentSpec("codex", "scripted"),
                        budget=AttemptBudget(attempts=40, per_arm=4), max_sessions=3,
                        scripted=agent, heavy=False, setup=setup, env=project_tool_env(root / "env"))
    print(json.dumps({"ended_because": record.get("ended_because"), "states": record.get("states")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
