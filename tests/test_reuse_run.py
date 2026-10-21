"""The reuse run end to end on the CPU, with scripted agents (fleetlab.loop.reuse_run).

Lane state's toy world stands in for both fleets. A source engagement of the round
driver (which learns by default) writes a learned skill in round 1, and the driver
checks and freezes it after the rounds; the reuse comparison then runs the toy world as
two engagements in fresh analyses, one with that library mounted, and scores them with
the registered metrics. The scripted agents prove the mechanics: what each arm spends is
scripted, not learned. No model, no simulator, no GPU.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fleetlab.agents.sessions import AgentSpec, Analysis
from fleetlab.loop.driver import AttemptBudget, run_rounds
from fleetlab.loop.learned import LEARNING_INSTRUCTIONS, FrozenLibrary
from fleetlab.loop.reuse import Budget, LaterCase, compare
from fleetlab.loop.reuse_run import (
    ADAPT_KEY,
    DELAY_KEY,
    ConditionTruth,
    EngagementRunner,
    ReuseScorer,
    adoption,
    engagement_problems,
    score_comparison,
)
from fleetlab.loop.scripted import ScriptedAgent, Step, decide_step, request_step
from fleetlab.toolserver.server import ToolBox

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
    "target": {"scope": "r03", "component": None, "config_key": DELAY_KEY, "condition_key": None},
    "value": None,
}  # fmt: skip
REQUEST = {
    "arms": [{"actions": []}, {"actions": [REVERT]}],
    "robots": ["r03"],
    "episodes_per_arm": 4,
    "paired_starts": True,
    "power_note": "a toy",
}
DIAGNOSIS: dict[str, Any] = {
    "causes": [
        {
            "target": {
                "scope": "r03",
                "component": None,
                "config_key": DELAY_KEY,
                "condition_key": None,
            },
            "direction": "misconfigured",
            "confidence": 0.8,
            "evidence": [{"episode_id": None, "step": None, "table": "episodes", "query": "toy"}],
            "failure_mode": "commands executed late",
        }
    ],
    "mode_summaries": ["one robot's commands run late"],
    "warnings": [],
    "retrain": [],
}
SKILL = {
    "name": "one-robot-runs-its-commands-late",
    "description": "Tell a robot whose command buffer setting changed from other causes.",
    "applies_when": "One robot fails more than the others of its site from the day a "
    "control setting of it changed.",
    "parameters": [
        {"name": "robot_id", "meaning": "the robot that dropped"},
        {"name": "from_day", "meaning": "the day its setting changed"},
    ],
    "evidence": [
        {
            "claim": "success of {robot_id} before and after day {from_day}, against the rest",
            "sql": "SELECT robot_id = $robot_id AS suspect, day >= $from_day AS after_change, "
            "count(*) AS n, avg(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS rate "
            "FROM episodes WHERE experiment_id IS NULL GROUP BY 1, 2 ORDER BY 1, 2",
            "result": "{robot_id} was lower than the rest after day {from_day}",
        }
    ],
    "alternatives": [
        {
            "explanation": "a camera of that robot",
            "how_to_tell": "revert the setting alone in an experiment",
        }
    ],
    "next_experiment": "On {robot_id}, two arms: as it is, and with the setting reverted.",
    "check_with": [{"name": "robot_id", "value": "r03"}, {"name": "from_day", "value": "0"}],
}


def _data(result: Any) -> Any:
    assert not result.is_error, result.content
    return json.loads(result.content[0]["text"])


def learn_then(step: Step) -> Step:
    def run(box: ToolBox) -> str:
        receipt = _data(box.call("write_learned_skill", SKILL))
        assert receipt["accepted"], receipt["problems"]
        return step(box)

    return run


def diagnose_then(step: Step) -> Step:
    def run(box: ToolBox) -> str:
        assert _data(box.call("submit_diagnosis", DIAGNOSIS))["accepted"]
        return step(box)

    return run


def use_library_then(step: Step) -> Step:
    """Read the learned skill, run its query on this fleet, then go on."""

    def run(box: ToolBox) -> str:
        listed = _data(box.call("list_skills", {}))
        learned = [m for m in listed if m.get("origin") == "learned"]
        assert learned, "the library is not mounted"
        text = box.call("read_skill", {"name": learned[0]["name"]}).content[0]["text"]
        sql = text.split("```sql\n", 1)[1].split("```", 1)[0]
        _data(
            box.call("query", {"sql": sql.replace("$robot_id", "'r03'").replace("$from_day", "0")})
        )
        return step(box)

    return run


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    from toolenv_fixed import fixed_digest

    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.loop.toy import write_toy_world

    root = tmp_path_factory.mktemp("reuse-run")
    previous = os.environ.get("ROBOTS_AGENT_ROOT")
    os.environ["ROBOTS_AGENT_ROOT"] = str(root / "agent-root")
    try:
        world, setup = write_toy_world(root / "world-root")
        env = fixed_digest(project_tool_env(root / "env"))
        budget = AttemptBudget(attempts=40, per_arm=4)
        source_agent = ScriptedAgent(
            [
                learn_then(request_step(REQUEST)),
                decide_step([RECAL]),
                decide_step([REVERT], stop=True),
            ],
            run_root=root / "scripted-source",
        )
        source = run_rounds(
            world,
            root / "source",
            agent=AgentSpec("codex", "scripted"),
            budget=budget,
            max_sessions=3,
            scripted=source_agent,
            heavy=False,
            setup=setup,
            env=env,
            fleet_label="the toy fleet",
        )
        frozen = source["learned"]
        assert frozen["library"] is not None, frozen
        library = FrozenLibrary.load(Path(frozen["library"]))
        steps = {
            "with": [use_library_then(diagnose_then(decide_step([REVERT], stop=True)))],
            "without": [
                request_step(REQUEST),
                diagnose_then(decide_step([REVERT])),
                decide_step([], stop=True, alternatives=0),
            ],
        }
        runner = EngagementRunner(
            world,
            budget,
            3,
            root / "reuse" / "engagements",
            heavy=False,
            setup=setup,
            scripted=lambda arm: ScriptedAgent(steps[arm], run_root=root / f"scripted-{arm}"),
        )
        truth = ConditionTruth(
            site="s2",
            robots=("r03",),
            components=(),
            config_keys=(DELAY_KEY,),
            adapt_verbs=("revert",),
            adapt_keys=(DELAY_KEY,),
            effective_points=-100.0,  # the toy panel is too small for an effect; mechanics only
            effective_p=1.01,
        )
        case = LaterCase(world.world_id, None, "SYSTEM", "round task", world_root=world.root)
        comparison = compare(
            case,
            AgentSpec("codex", "scripted"),
            env,
            library,
            root / "reuse",
            budget=Budget(max_sessions=3),
            runner=runner,
            score=ReuseScorer(truth, "scripted"),
        )
        yield {
            "root": root,
            "world": world,
            "source": source,
            "source_agent": source_agent,
            "library": library,
            "comparison": comparison,
            "truth": truth,
        }
    finally:
        if previous is None:
            os.environ.pop("ROBOTS_AGENT_ROOT", None)
        else:
            os.environ["ROBOTS_AGENT_ROOT"] = previous


def test_the_source_engagement_learned_and_the_driver_froze_its_skill(
    run: dict[str, Any],
) -> None:
    source, library = run["source"], run["library"]
    assert source["ended_because"] == "round 2: the agent stopped"
    prompt = json.loads(next((run["root"] / "scripted-source").rglob("prompt.json")).read_text())
    assert prompt["prompt"].count(LEARNING_INSTRUCTIONS) == 1  # the round task, once
    assert LEARNING_INSTRUCTIONS not in prompt["system"]
    assert library.names == [SKILL["name"]]
    text = (library.skills / SKILL["name"] / "SKILL.md").read_text()
    assert "learned by the analysis agent on the toy fleet" in text
    assert library.record["skills"][0]["source"]["world_id"] == run["world"].world_id


def test_the_arms_are_two_fresh_engagements_identical_but_for_the_library(
    run: dict[str, Any],
) -> None:
    comparison = run["comparison"]
    without, with_ = comparison.arms["without"], comparison.arms["with"]
    assert comparison.validity == []  # round 1's export is the same bytes in both arms
    assert without.runner["round_exports"]["1"] == with_.runner["round_exports"]["1"]
    for key in ("world", "base_skills", "tool_env", "agent", "runner", "system_sha256"):
        assert without.inputs[key] == with_.inputs[key], key
    assert without.inputs["runner"]["kind"] == "engagement"
    assert without.inputs["library"] is None and with_.inputs["library"] == run["library"].digest
    with_record = json.loads(Path(with_.runner["engagement"]).read_text())
    without_record = json.loads(Path(without.runner["engagement"]).read_text())
    assert with_record["analysis"] == with_.analysis  # the harness's fresh analysis
    assert with_record["learned_mounted"]["digest"] == run["library"].digest
    assert "learned_mounted" not in without_record
    assert with_record["outcome"] == without_record["outcome"] == "stopped"
    assert comparison.arms["with"].runner["validity"] == []
    assert Analysis(Path(with_.analysis)).learned is not None


def test_the_registered_metrics_of_each_arm(run: dict[str, Any]) -> None:
    comparison = run["comparison"]
    with_ = comparison.arms["with"].outcome
    without = comparison.arms["without"].outcome
    first_with = with_["identification"]["first"]
    first_without = without["identification"]["first"]
    assert with_["identification"]["targeted"] and first_with["session"] == 1
    assert first_with["attempts_before"] == 0
    assert first_with["diagnosis"] == DIAGNOSIS  # verbatim, beside the proxy
    assert first_with["trace"] and all(Path(t).is_file() for t in first_with["trace"])
    assert first_without["session"] == 2 and first_without["round"] == 1
    assert first_without["attempts_before"] == 8
    assert with_["adoption"]["first_effective"]["round"] == 1
    assert with_["adoption"]["attempts_until_effective"] == 0
    assert without["adoption"]["attempts_until_effective"] == 8
    assert with_["adoption"]["first_effective"]["effect_at_condition"]["attempts"] > 0
    assert [q["block"] for q in with_["procedural_use"]["learned_queries_run"]] == ["evidence-1"]
    assert with_["procedural_use"]["skill_reads"] == [{"session": 1, "skill": SKILL["name"]}]
    assert without["procedural_use"]["mounted"] == []
    assert with_["outcome"]["final_state"] == "round-1"
    assert without["outcome"]["final_state"] == "round-2"
    assert isinstance(with_["score"], float)
    costs = comparison.arms["with"].costs
    assert costs["learned_skill_reads"] == {SKILL["name"]: 1}
    assert comparison.arms["without"].costs["experiments"]["attempts_simulated"] == 8
    assert comparison.differences["experiments_requested"]["with_minus_without"] == -1
    scored = score_comparison(run["root"] / "reuse", run["truth"])
    assert scored["arms"]["with"]["metrics"]["identification"]["first"]["session"] == 1
    assert (run["root"] / "reuse" / "metrics.json").is_file()
    between = scored["between_arms"]
    assert between["paired"] and between["states"] == {"without": "round-2", "with": "round-1"}
    assert between["attempts"] > 0 and 0.0 <= between["p_mcnemar"] <= 1.0


def test_the_truth_key_reads_a_loop_fleets_lasting_condition() -> None:
    from types import SimpleNamespace

    from fleetlab.contracts.faults import FaultSpec, Schedule, Target
    from fleetlab.contracts.vocabulary import Component

    def fault(fault_id: str, scope: str) -> FaultSpec:
        return FaultSpec(
            fault_id=fault_id,
            kind="remote_link_latency",
            params={"steps": 1},
            level=1.0,
            target=Target(scope=scope, component=Component.controller),
            schedule=Schedule(onset_day=0),
        )

    robots = {"s1": ["r01", "r02"], "s2": ["r03", "r04"]}
    world = SimpleNamespace(
        model=SimpleNamespace(
            sites=[
                SimpleNamespace(robots=[SimpleNamespace(id=r, site_id=s) for r in rs])
                for s, rs in robots.items()
            ]
        ),
        spec=SimpleNamespace(
            patterns=SimpleNamespace(decoys=["f01", "f02"]),
            fault_schedule=[fault("f01", "r01"), fault("f02", "r02"), fault("f03", "r03"),
                            fault("f04", "r04")],
        ),
    )  # fmt: skip
    truth = ConditionTruth.from_world(world)
    assert truth.site == "s2" and truth.robots == ("r03", "r04")
    assert truth.lookalike_sites == ("s1",) and truth.lookalikes == (("s1", ("r01", "r02")),)


def _set(scope: str, value: int) -> dict[str, Any]:
    target = {"scope": scope, "component": None, "config_key": ADAPT_KEY, "condition_key": None}
    return {"verb": "set", "target": target, "value": value}


def _rows(state: str, robot: str, wins: int, n: int = 20) -> list[dict[str, Any]]:
    return [
        {"state": state, "episode_id": f"e{robot}{i:03d}", "robot_id": robot, "success": i < wins}
        for i in range(n)
    ]


def test_adoption_skips_an_ineffective_choice_and_records_settings_elsewhere() -> None:
    """Round 1 sets a value that does not help; round 2 replaces it with one that does and
    sets the key fleet-wide, which costs the lookalike: the headline counts attempts
    through round 2, both attempts are listed, and the loss is measured harm."""
    truth = ConditionTruth(site="s2", robots=("r03",), lookalikes=(("s1", ("r01",)),))
    engagement = {
        "rounds": [
            {"round": 1, "costs": {"arm_attempts": 40},
             "decision": {"adopt": [{"action": _set("s2", 2)}], "replace": []}},
            {"round": 2, "costs": {"arm_attempts": 60},
             "decision": {"adopt": [{"action": _set("fleet", 9)}],
                          "replace": [{"entry_id": "L1", "action": _set("r03", 9)}]}},
        ]
    }  # fmt: skip
    rows = (
        _rows("original", "r03", 5) + _rows("round-1", "r03", 6) + _rows("round-2", "r03", 19)
        + _rows("original", "r01", 19) + _rows("round-1", "r01", 19) + _rows("round-2", "r01", 2)
    )  # fmt: skip
    result = adoption(truth, engagement, rows)
    assert [a["round"] for a in result["attempts"]] == [1, 2]
    assert [a["effective"] for a in result["attempts"]] == [False, True]
    assert result["first_effective"]["round"] == 2
    assert result["attempts_until_effective"] == 100
    [elsewhere] = result["adaptation_elsewhere"]
    assert elsewhere["where"] == "fleet" and elsewhere["round"] == 2
    assert elsewhere["at_lookalikes"][0]["measured_harm"] is True
    none = adoption(truth, {"rounds": engagement["rounds"][:1]}, rows)
    assert none["first_effective"] is None and none["attempts_until_effective"] is None
    assert len(none["attempts"]) == 1  # the ineffective attempt is still reported


def test_a_lookalike_diagnosis_is_recorded_on_its_own() -> None:
    from fleetlab.contracts.tools import Diagnosis

    truth = ConditionTruth(site="s2", robots=("r03", "r04"), lookalikes=(("s1", ("r01", "r02")),))
    cause: dict[str, Any] = DIAGNOSIS["causes"][0]
    on_lookalike = {**cause, "target": {**cause["target"], "scope": "r01"}}
    diagnosis = Diagnosis.model_validate({**DIAGNOSIS, "causes": [on_lookalike]})
    assert truth.targeting(diagnosis, "s1", ("r01", "r02"))
    assert not truth.targeting(diagnosis, "s2", ("r03", "r04"))


def test_an_aborted_or_unevaluated_engagement_makes_the_comparison_invalid() -> None:
    aborted = {"outcome": "aborted", "ended_because": "round 1: no decision",
               "evaluation": {"skipped": "the engagement was aborted"}}  # fmt: skip
    problems = engagement_problems(aborted, "with")
    assert len(problems) == 2 and "did not complete" in problems[0]
    unevaluated = {"outcome": "completed", "evaluation": {}}
    assert engagement_problems(unevaluated, "without") == [
        "the without engagement has no evaluation: no summary"
    ]
    done = {"outcome": "stopped", "evaluation": {"summary": {"states": []}}}
    assert engagement_problems(done, "x") == []
