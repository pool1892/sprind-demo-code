"""The plain search, the adaptation comparator (fleetlab.loop.plain_search): its fixed rule
on synthetic outcomes, its request from the public description, and one engagement on lane
state's toy world through the driver: the request goes through the round's queue and engine,
the decision follows from the experiment's visible outcomes only, the ledger takes it, the
states are evaluated on the reserved panel and the costs are the driver's. No model, no
simulator, no GPU."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fleetlab.contracts.vocabulary import ConfigKey
from fleetlab.loop.plain_search import (
    ALPHA,
    GRID,
    PER_ARM,
    PlainSearch,
    choose,
    current_value,
    request_for,
    run_plain_search,
)
from fleetlab.loop.state import Ledger

DELAY = ConfigKey.control_action_delay_steps
COMPENSATION = ConfigKey.control_latency_compensation_steps


# ---- the rule -------------------------------------------------------------------------------


def test_the_design_is_fixed_before_any_result() -> None:
    assert GRID == (0, 5, 10, 15, 20, 25) and PER_ARM == 24 and ALPHA == 0.05
    search = PlainSearch("s1", COMPENSATION)
    assert (search.grid, search.per_arm, search.alpha) == (GRID, PER_ARM, ALPHA)
    with pytest.raises(ValueError):
        PlainSearch("fleet", COMPENSATION)


def test_the_most_successes_win_and_a_clear_gain_is_adopted() -> None:
    control = [True] * 12 + [False] * 12  # 12 of 24
    better = [True] * 23 + [False]  # 23 of 24: rescues 11, breaks 0
    worse = [False] * 24
    choice = choose([0, 5, 10], [control, better, worse])
    assert (choice.chosen, choice.adopt, choice.rescued, choice.broken) == (5, True, 11, 0)
    assert choice.threshold == pytest.approx(0.025)
    assert choice.p_value == pytest.approx(0.5**11)


def test_a_tie_goes_to_the_value_nearest_the_current_one() -> None:
    same = [True] * 20 + [False] * 4
    choice = choose([10, 0, 5, 15, 20], [[False] * 24, same, same, same, same])
    assert choice.chosen == 5  # 5 and 15 tie on successes and distance; the smaller wins
    assert choose([10, 0, 20], [[False] * 24, same, same]).chosen == 0


def test_nothing_is_adopted_without_a_clear_paired_gain() -> None:
    control = [True] * 18 + [False] * 6
    slightly = [True] * 21 + [False] * 3  # rescues 3, breaks 0: p = 1/8
    choice = choose([0, 5, 10, 15, 20, 25], [control, slightly] + [control] * 4)
    assert (choice.chosen, choice.adopt) == (5, False)
    assert "not below" in choice.reason
    kept = choose([0, 5], [control, control])
    assert (kept.chosen, kept.adopt) == (0, False)
    with pytest.raises(ValueError):
        choose([0, 5], [control, control[:-1]])


def test_the_request_comes_from_the_public_description() -> None:
    fleet = {
        "config": {"values": {COMPENSATION.value: 0, DELAY.value: 0}},
        "sites": [
            {
                "id": "s1",
                "config": {"values": {}},
                "robots": [
                    {"id": "r01", "config": {"values": {}}},
                    {"id": "r02", "config": {"values": {DELAY.value: 3}}},
                ],
            },
            {
                "id": "s2",
                "config": {"values": {COMPENSATION.value: 10}},
                "robots": [{"id": "r03", "config": {"values": {}}}],
            },
        ],
    }
    assert current_value(fleet, "s1", COMPENSATION) == 0
    assert current_value(fleet, "s2", COMPENSATION) == 10
    assert current_value(fleet, "r02", DELAY) == 3
    request, values = request_for(PlainSearch("s2", COMPENSATION), fleet)
    assert values == [10, 0, 5, 15, 20, 25]
    assert request["robots"] == ["r03"] and request["paired_starts"] is True
    assert request["arms"][0] == {"actions": []}
    assert [a["actions"][0]["value"] for a in request["arms"][1:]] == [0, 5, 15, 20, 25]
    assert {a["actions"][0]["target"]["scope"] for a in request["arms"][1:]} == {"s2"}
    assert request["episodes_per_arm"] == PER_ARM


# ---- one engagement on the toy world -----------------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.loop.toy import write_toy_world

    root = tmp_path_factory.mktemp("plain-search")
    previous = os.environ.get("ROBOTS_AGENT_ROOT")
    os.environ["ROBOTS_AGENT_ROOT"] = str(root / "agent-root")
    try:
        world, setup = write_toy_world(root / "world-root")
        # The toy fleet has no compensation key; its r03 runs 3 steps late through
        # control.action_delay_steps, which the toy registry simulates.
        search = PlainSearch("r03", DELAY, grid=(0, 3, 6), per_arm=24)
        report = run_plain_search(
            world,
            root / "out",
            search,
            heavy=False,
            setup=setup,
            env=project_tool_env(root / "env"),
        )
        yield {"root": root, "report": report, "search": search}
    finally:
        if previous is None:
            os.environ.pop("ROBOTS_AGENT_ROOT", None)
        else:
            os.environ["ROBOTS_AGENT_ROOT"] = previous


def test_one_experiment_through_the_agents_path_and_its_costs(run: dict[str, Any]) -> None:
    report = run["report"]
    assert report.extra["outcome"] == "stopped"
    costs = report.costs
    assert costs["requests"] == 1 and costs["requests_refused"] == 0
    (experiment,) = costs["experiments"]
    assert experiment["attempts_per_arm"] == [24, 24, 24] and experiment["robots"] == ["r03"]
    assert costs["arm_attempts"] == 72
    # Scripted sessions on the CPU, no model: a complete receipt with no model use.
    assert costs["model_starts"] == 0 and costs["model_seconds"] == 0.0
    assert costs["usage_complete"] is True and costs["starts_without_usage"] == 0
    assert set(costs["tokens"].values()) == {0}
    assert costs["scripted_sessions"] == 2 and costs["compute_seconds"] > 0
    assert costs["api_equivalent_usd"]["low_usd"] == 0.0 and costs["cost_usd_list_price"] == 0.0
    assert costs["tool_calls"] > 0
    # The driver's raw record of the round is kept beside the receipt.
    raw = report.raw_costs
    assert raw["arm_attempts"] == 72 and raw["sessions"] == 2
    # The arms: the current state (3 steps), then the grid without it.
    assert experiment["actions"][0] == []
    assert [a[0]["value"] for a in experiment["actions"][1:]] == [0, 6]


def test_the_decision_follows_from_the_visible_outcomes_only(run: dict[str, Any]) -> None:
    import pyarrow.parquet as pq

    from fleetlab.toolserver.files import RESULTS_FILE

    report = run["report"]
    engagement = json.loads(Path(report.engagement).read_text())
    results = [
        json.loads(line)
        for line in (Path(engagement["analysis"]) / "results" / RESULTS_FILE)
        .read_text()
        .splitlines()
        if line.strip()
    ]
    (result,) = [r for r in results if r["status"] == "done"]
    # The arms' visible records, as the engine wrote them for the round's queue.
    loop = Path(engagement["analysis"]).parents[1] / "loop"
    tables = sorted(loop.glob("rounds/r1/experiments/*/session-*/log/episodes.parquet"))
    assert tables
    rows = [r for t in tables for r in pq.read_table(t).to_pylist()]
    outcome = {r["episode_id"]: r["outcome"] == "success" for r in rows}
    arms = sorted(result["arms"], key=lambda a: a["arm_index"])
    expected = choose([3, 0, 6], [[outcome[e] for e in a["episode_ids"]] for a in arms])
    assert report.choice is not None
    assert report.choice["chosen"] == expected.chosen
    assert report.choice["adopt"] == expected.adopt
    assert report.choice["successes"] == expected.successes
    ledger = Ledger.load(Path(report.ledger))
    if expected.adopt:
        (entry,) = ledger.entries
        assert entry.action.target.scope == "r03" and entry.action.value == expected.chosen
        assert report.adopted[0]["value"] == expected.chosen
    else:
        assert not ledger.entries and report.adopted == []


def test_the_states_are_evaluated_on_the_reserved_panel(run: dict[str, Any]) -> None:
    import pyarrow.parquet as pq

    report = run["report"]
    summary = report.evaluation
    assert summary is not None
    names = {s["name"] for s in summary["states"]}
    assert {"original", "round-1"} <= names
    # The toy's r03 is 3 steps late; 24 paired attempts per arm make the search adopt 0
    # there, and the evaluated state runs it in place of the delay (truth side).
    assert report.choice is not None and report.choice["adopt"] is True
    loop = Path(json.loads(Path(report.engagement).read_text())["analysis"]).parents[1] / "loop"
    rows = pq.read_table(loop / "evaluation" / "round-1" / "truth" / "faults_active.parquet")
    at_r03 = [r for r in rows.to_pylist() if r["target_scope"] == "r03"]
    assert at_r03 and {(r["kind"], r["params"]) for r in at_r03} == {
        ("config_change", '{"value":0}')
    }
    stored = json.loads((run["root"] / "out" / "plain_search.json").read_text())
    assert stored["search"]["grid"] == [0, 3, 6] and stored["search"]["key"] == DELAY.value
