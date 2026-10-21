"""Two rounds of the driver end to end on the CPU (fleetlab.loop.driver.run_rounds over
lane state's toy world): a scripted agent requests an experiment in round 1, reads its
public result in the next session and decides; the ledger takes the decision; round 2
operates under it, its export holds the history with the adoption as the fleet's own
record, the agent decides again and stops; the states, the one-look plan included, are
evaluated on the reserved panel. No model, no simulator, no GPU."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fleetlab.agents.sessions import AgentSpec
from fleetlab.log import LogReader
from fleetlab.loop.driver import AttemptBudget, run_rounds
from fleetlab.loop.scripted import ScriptedAgent, decide_step, request_step
from fleetlab.loop.state import Ledger

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
        "scope": "r03",
        "component": None,
        "config_key": "control.action_delay_steps",
        "condition_key": None,
    },
    "value": None,
}
# A learned skill as lane learn's tests write one (no identifier of this fleet in it).
SKILL: dict[str, Any] = {
    "name": "site-step-after-control-timing",
    "description": (
        "Find a site whose robots all lose success from one day on, and tell a control "
        "timing setting apart from other causes shared by the site."
    ),
    "applies_when": (
        "Every robot of one site drops together from one day on while the other sites "
        "hold, and no single robot stands out."
    ),
    "parameters": [
        {"name": "site_id", "meaning": "the site whose robots dropped together"},
        {"name": "from_day", "meaning": "the first day of the drop"},
    ],
    "evidence": [
        {
            "claim": "success at {site_id} before and after day {from_day}, against the rest",
            "sql": (
                "SELECT site_id = $site_id AS at_site, day >= $from_day AS after_step, "
                "count(*) AS n, avg(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS rate "
                "FROM episodes WHERE experiment_id IS NULL GROUP BY 1, 2 ORDER BY 1, 2"
            ),
            "result": "at {site_id} the rate fell from day {from_day}; the other site held",
        }
    ],
    "alternatives": [
        {
            "explanation": "a software update logged on the same day at the same site",
            "how_to_tell": "revert one of the two alone in an experiment",
        }
    ],
    "next_experiment": (
        "At {site_id}, two arms on the same robots and start states: as it is, and with the "
        "control timing setting reverted; 30 attempts each."
    ),
    "check_with": [{"name": "site_id", "value": "s2"}, {"name": "from_day", "value": "1"}],
}
REQUEST = {
    "arms": [{"actions": []}, {"actions": [RECAL]}],
    "robots": ["r02"],
    "episodes_per_arm": 4,
    "paired_starts": True,
    "power_note": "a toy",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.loop.toy import write_toy_world

    root = tmp_path_factory.mktemp("driver")
    previous = os.environ.get("ROBOTS_AGENT_ROOT")
    os.environ["ROBOTS_AGENT_ROOT"] = str(root / "agent-root")
    try:
        world, setup = write_toy_world(root / "world-root")
        agent = ScriptedAgent(
            [
                request_step(REQUEST),
                decide_step([RECAL], learned=SKILL),
                decide_step([REVERT], stop=True),
            ],
            run_root=root / "scripted",
        )
        record = run_rounds(
            world,
            root / "out",
            agent=AgentSpec("codex", "scripted"),
            budget=AttemptBudget(attempts=40, per_arm=4),
            max_sessions=3,
            scripted=agent,
            heavy=False,
            setup=setup,
            env=project_tool_env(root / "env"),
            fleet_label="the miniature fleet",
        )
        yield {"root": root, "record": record, "agent": agent, "world": world}
    finally:
        if previous is None:
            os.environ.pop("ROBOTS_AGENT_ROOT", None)
        else:
            os.environ["ROBOTS_AGENT_ROOT"] = previous


def test_two_rounds_run_and_the_agent_stops(run: dict[str, Any]) -> None:
    record = run["record"]
    assert record["outcome"] == "stopped"
    assert record["ended_because"] == "round 2: the agent stopped"
    first, second = record["rounds"]
    assert [s["state"] for s in first["sessions"]] == ["waiting", "done"]
    assert [s["state"] for s in second["sessions"]] == ["done"]
    assert first["costs"]["arm_attempts"] == 8 and second["costs"]["arm_attempts"] == 0
    assert first["costs"]["experiments"][0]["attempts_per_arm"] == [4, 4]
    assert first["costs"]["tool_calls"] > 0 and first["costs"]["model_starts"] == 2
    assert [c["action"]["verb"] for c in first["decision"]["adopt"]] == ["recalibrate"]
    assert second["decision"]["stop"] is True


def test_the_ledger_keeps_every_version(run: dict[str, Any]) -> None:
    ledgers = Path(run["record"]["loop"]) / "ledgers"
    versions = [Ledger.load(ledgers / f"ledger-r{n}.json") for n in (0, 1, 2)]
    assert [len(v.entries) for v in versions] == [0, 1, 2]
    l1, l2 = versions[1].entries[0], versions[2].entries[1]
    assert (l1.entry_id, l1.round, l1.action.verb.value) == ("L1", 1, "recalibrate")
    assert (l2.entry_id, l2.round, l2.action.verb.value) == ("L2", 2, "revert")
    assert l1.adopted_day == 3 and l2.adopted_day == 5  # the next round's first day; evaluation


def test_round_two_sees_its_operation_its_own_arms_and_the_adoption(run: dict[str, Any]) -> None:
    second = run["record"]["rounds"][1]
    with LogReader(Path(second["export"]["path"])) as reader:
        days = reader.query(
            "select day, count(*) n, count(experiment_id) arms from episodes group by day "
            "order by day"
        ).to_pylist()
        upkeep = reader.query("select day, robot_id, verb from maintenance").to_pylist()
    assert [(d["day"], d["arms"] > 0) for d in days] == [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
    ]
    assert {"day": 3, "robot_id": "r02", "verb": "recalibrate"} in upkeep
    task = second["task"]
    assert "round 2 of an ongoing engagement" in task
    assert "L1 (round 1, from day 3, in force): recalibrate scene_camera at r02" in task
    assert "round 1: sessions 1 to 2" in task and "earlier_decisions" in task
    assert "scripted" not in task  # the agent's own prose is never pasted into the task
    # What the agent read in round 2 carries nothing of the evaluation.
    prompts = " ".join(s["prompt"] for s in run["agent"].seen).lower()
    assert "evaluation" not in prompts and "panel" not in prompts


def test_the_states_and_the_one_look_plan_are_evaluated(run: dict[str, Any]) -> None:
    record = run["record"]
    assert record["states"] == ["one-look", "original", "round-1", "round-2"]
    summary = record["evaluation"]["summary"]
    names = {s["name"]: s for s in summary["states"]}
    assert set(names) == {"one-look", "original", "round-1", "round-2"}
    assert names["round-2"]["changes"] == 2 and names["round-1"]["changes"] == 1
    # The one-look plan: round 1's adoptions, then its alternatives, cut at the final
    # number of changes (2): the adopted recalibration and the first alternative.
    listed = json.loads((Path(record["analysis"]).parent / "one-look-list.json").read_text())
    assert listed[0] == RECAL and len(listed) == 5  # 1 adopted + 4 alternatives
    plan = Ledger.load(Path(record["loop"]) / "one-look" / "ledger-r1.json")
    assert [e.action.model_dump(mode="json") for e in plan.entries] == listed[:2]
    assert names["one-look"]["changes"] == 2
    assert record["one_look"] == {
        "requested_cutoff": 2,
        "listed": 5,
        "adoptable": 5,
        "applied": 2,
    }
    assert len({s["attempts"] for s in summary["states"]}) == 1  # the same panel
    simulated = record["simulated"]
    assert simulated["experiments"]["run_jobs"] == 8 and simulated["operation"]["run_jobs"] > 0


def test_the_learned_skill_is_frozen_after_the_rounds(run: dict[str, Any]) -> None:
    """The round task carries lane learn's paragraph, the agent's draft is checked and
    frozen by lane learn's freeze, and the run record holds the library's digest."""
    from fleetlab.loop.learned import LEARNING_INSTRUCTIONS, FrozenLibrary

    record = run["record"]
    assert LEARNING_INSTRUCTIONS.split(".")[0] in record["rounds"][0]["task"]
    learned = record["learned"]
    assert learned["drafts"] == [SKILL["name"]] and learned["kept"] == [SKILL["name"]]
    assert learned["refused"] == {}
    library = FrozenLibrary.load(Path(learned["library"]))
    assert library.digest == learned["digest"] and library.names == [SKILL["name"]]
    text = (library.skills / SKILL["name"] / "SKILL.md").read_text()
    assert "origin: learned" in text and "the miniature fleet" in text


def test_a_skill_keeps_the_round_it_was_written_in(run: dict[str, Any]) -> None:
    """Lane learn's session-to-round map: the draft written in session 2 is round 1's."""
    from fleetlab.loop.learned import FrozenLibrary

    learned = run["record"]["learned"]
    assert learned["rounds"] == {SKILL["name"]: 1}
    library = FrozenLibrary.load(Path(learned["library"]))
    (entry,) = library.record["skills"]
    assert entry["source"]["round"] == 1 and entry["source"]["session"] == 2
    text = (library.skills / SKILL["name"] / "SKILL.md").read_text()
    assert "the miniature fleet, round 1, session 2" in text


class ReachedLaunch(Exception):
    pass


def test_the_recorded_library_launches_a_later_engagement(
    run: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round trip: the path the first engagement recorded is loaded and verified, only
    its skills tree is mounted, and a later engagement's first session passes the real
    launch audit with it (the call stops at the process start; no model)."""
    from fleetlab.agents import checks
    from fleetlab.agents import run as runtime
    from fleetlab.agents.audit import tree_digest
    from fleetlab.agents.sessions import Analysis
    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.learned import FrozenLibrary
    from fleetlab.loop.rounds import Engagement, run_engagement
    from fleetlab.loop.toy import write_toy_world
    from fleetlab.loop.world_loop import WorldLoop

    library = FrozenLibrary.load(Path(run["record"]["learned"]["library"]))
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    world, setup = write_toy_world(tmp_path / "later-world")
    state = WorldLoop.create(world, layout_for_world(world.spec), tmp_path / "loop", setup)
    audited: list[Path] = []

    def prepare(*args: Any, **kwargs: Any) -> Any:
        directory = args[4]
        box = runtime.box_dir(directory)
        (box / "cwd").mkdir(parents=True)
        (box / "toolbox").mkdir()
        (box / "system.txt").write_text(args[2])
        (box / "schema.json").write_text(json.dumps(args[7]))
        return runtime.PreparedCall([], {}, args[3], directory, None, 0)

    def execute(*args: Any, **kwargs: Any) -> None:
        directory = args[3]
        assert json.loads((directory / "audit.json").read_text())["passed"]
        audited.append(directory)
        raise ReachedLaunch

    monkeypatch.setattr(runtime, "prepare_call", prepare)
    monkeypatch.setattr(runtime, "_execute", execute)
    monkeypatch.setattr(checks, "require_verified", lambda *args: {"passed": True})
    how = Engagement(
        agent=AgentSpec("claude", "model"),
        budget=AttemptBudget(attempts=40),
        max_sessions=1,
        system="You analyse a fleet of robot arms.",
        env=project_tool_env(tmp_path / "env"),
        fleet=world.model,
        learned=library.root,
    )
    with pytest.raises(ReachedLaunch):
        run_engagement(state, world.world_id, tmp_path / "later", how)
    assert len(audited) == 1  # the launch audit passed with the library mounted
    record = json.loads((tmp_path / "later" / "engagement.json").read_text())
    assert record["learned_mounted"]["digest"] == library.digest
    analysis = Analysis(next((tmp_path / "later").glob("w01-*")))
    assert analysis.learned is not None
    assert tree_digest(analysis.learned) == library.digest  # the skills tree alone
    assert not (analysis.learned / "library.json").exists()
    # A path that is not a frozen library's root is refused before anything is created.
    wrong = replace(how, learned=library.skills)
    with pytest.raises(ValueError):
        run_engagement(state, world.world_id, tmp_path / "wrong", wrong)
