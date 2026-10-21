"""The round decision and the attempt budget (fleetlab.contracts.decisions,
fleetlab.toolserver.decisions): the schema, the public checks of a decision against the
fleet and its ledger, and the budget's status. No model, no simulator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from log_toy import toy_fleet
from pydantic import ValidationError

from fleetlab.contracts.decisions import MAX_ADOPTED_PER_ROUND, RoundDecision
from fleetlab.contracts.tools import json_schema_strict
from fleetlab.toolserver.decisions import (
    PublicLedger,
    PublicLedgerEntry,
    budget_status,
    decision_problems,
)


def act(verb: str, scope: str, **target: Any) -> dict[str, Any]:
    value = target.pop("value", None)
    full = {"scope": scope, "component": None, "config_key": None, "condition_key": None}
    return {"verb": verb, "target": {**full, **target}, "value": value}


RECAL = act("recalibrate", "r02", component="scene_camera")
REVERT = act("revert", "r02", config_key="control.action_delay_steps")
CLEAN = act("clean", "r01", component="gripper")
FILLERS = [
    act("recalibrate", "r01", component="scene_camera"),
    act("recalibrate", "r01", component="wrist_camera"),
    act("clean", "r02", component="scene_camera"),
    act("service", "r01", component="arm"),
    act("service", "r02", component="arm"),
    act("replace", "r02", component="gripper"),
]


def ranked(*first: dict[str, Any]) -> list[dict[str, Any]]:
    actions = list(first) + [a for a in FILLERS if a not in first]
    return [
        {
            "action": a,
            "expected_gain": 1.0,
            "uncertainty": {"low": 0.0, "high": 2.0},
            "rationale": "a test",
        }
        for a in actions[:8]
    ]


def decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "adopt": [{"action": RECAL, "rationale": "the camera"}],
        "replace": [],
        "undo": [],
        "predicted_set_gain": {"low": 1.0, "mid": 3.0, "high": 6.0},
        "expectations": ["r02 above 90% on its tasks"],
        "stop": False,
        "ranked_candidates": ranked(RECAL),
    }
    return {**base, **overrides}


def ledger(*entries: tuple[str, dict[str, Any], str]) -> PublicLedger:
    return PublicLedger(
        round=2,
        changes=[
            PublicLedgerEntry.model_validate(
                {
                    "entry_id": entry_id,
                    "round": 1,
                    "adopted_day": 4,
                    "action": action,
                    "status": status,
                    "replaced_by": "L9" if status == "replaced" else None,
                }
            )
            for entry_id, action, status in entries
        ],
    )


def test_the_decision_schema_is_strict_and_bounded() -> None:
    json_schema_strict(RoundDecision)  # OpenAI's strict mode takes it
    RoundDecision.model_validate(decision())
    four = [{"action": a, "rationale": "x"} for a in [RECAL, *FILLERS[:3]]]
    assert len(four) > MAX_ADOPTED_PER_ROUND
    with pytest.raises(ValidationError):
        RoundDecision.model_validate(decision(adopt=four))
    with pytest.raises(ValidationError):
        RoundDecision.model_validate(
            decision(predicted_set_gain={"low": 3.0, "mid": 1.0, "high": 6.0})
        )
    with pytest.raises(ValidationError):  # an entry named twice
        RoundDecision.model_validate(
            decision(undo=[{"entry_id": "L1", "rationale": "a"}] * 2, adopt=[])
        )
    with pytest.raises(ValidationError):  # a set without its value
        RoundDecision.model_validate(
            decision(
                adopt=[
                    {
                        "action": act("set", "r01", config_key="policy.sampling_steps"),
                        "rationale": "",
                    }
                ]
            )
        )


def test_a_good_decision_has_no_problems_and_a_bad_one_hears_why() -> None:
    fleet = toy_fleet()
    assert decision_problems(RoundDecision.model_validate(decision()), fleet, None) == []
    # The ranked alternatives are optional: none at all, or ones that leave out the adopted
    # action, are both a valid decision.
    for alternatives in ([], ranked(CLEAN)):
        plain = RoundDecision.model_validate(decision(ranked_candidates=alternatives))
        assert decision_problems(plain, fleet, None) == []
    # Public applicability: a revert needs a recorded change at that scope; values in domain.
    wrong = RoundDecision.model_validate(
        decision(
            adopt=[
                {
                    "action": act("revert", "s1", config_key="control.action_delay_steps"),
                    "rationale": "x",
                },
                {
                    "action": act("set", "r01", config_key="policy.sampling_steps", value=0),
                    "rationale": "x",
                },
                {"action": act("route_to_teleop", "fleet", value="put"), "rationale": "x"},
            ],
            ranked_candidates=ranked(
                act("revert", "s1", config_key="control.action_delay_steps"),
                act("set", "r01", config_key="policy.sampling_steps", value=0),
                act("route_to_teleop", "fleet", value="put"),
            ),
        )
    )
    problems = decision_problems(wrong, fleet, None)
    assert any("no recorded change" in p for p in problems)
    assert any("allowed experiment values" in p for p in problems)
    assert any("no simulated effect" in p for p in problems)


def test_replacements_and_undos_follow_the_ledger() -> None:
    fleet = toy_fleet()
    book = ledger(("L1", REVERT, "active"), ("L2", RECAL, "active"), ("L3", CLEAN, "undone"))
    undo_revert = RoundDecision.model_validate(
        decision(adopt=[], undo=[{"entry_id": "L1", "rationale": "no effect"}])
    )
    assert decision_problems(undo_revert, fleet, book) == []
    physical = RoundDecision.model_validate(
        decision(adopt=[], undo=[{"entry_id": "L2", "rationale": "x"}])
    )
    assert any("cannot be taken back" in p for p in decision_problems(physical, fleet, book))
    gone = RoundDecision.model_validate(
        decision(adopt=[], undo=[{"entry_id": "L3", "rationale": "x"}])
    )
    assert any("no longer in force" in p for p in decision_problems(gone, fleet, book))
    unknown = RoundDecision.model_validate(
        decision(adopt=[], undo=[{"entry_id": "L7", "rationale": "x"}])
    )
    assert any("no ledger entry L7" in p for p in decision_problems(unknown, fleet, book))
    again = RoundDecision.model_validate(decision())  # RECAL is L2, in force
    assert any("already in force" in p for p in decision_problems(again, fleet, book))
    # adopt and replace count together against the cap; an undo is free.
    many = RoundDecision.model_validate(
        decision(
            adopt=[{"action": a, "rationale": "x"} for a in FILLERS[:3]],
            replace=[{"entry_id": "L1", "action": CLEAN, "rationale": "x"}],
            ranked_candidates=ranked(*FILLERS[:3], CLEAN),
        )
    )
    assert any("at most 3 changes" in p for p in decision_problems(many, fleet, book))


def test_the_budget_status_charges_what_ran_from_the_rounds_first_session() -> None:
    limits = SimpleNamespace(
        max_experiment_episodes=100,
        max_episodes_per_arm=20,
        max_experiment_requests=50,
        budget_first_session=3,
    )

    def request(experiment_id: str, session: int, n: int, arms: int) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "session": session,
            "request": {"episodes_per_arm": n, "arms": [{"actions": []}] * arms},
        }

    requests = [
        request("x0001", 1, 20, 2),  # an earlier round: not counted
        request("x0002", 3, 20, 3),  # 60
        request("x0003", 4, 10, 2),  # refused: free
        request("x0004", 4, 5, 2),  # 10
    ]
    status = budget_status(limits, requests, refused={"x0003"})
    assert (status.attempts_used, status.attempts_left, status.requests_used) == (70, 30, 2)
    assert (status.counted_from_session, status.max_per_arm) == (3, 20)


def test_zero_alternatives_and_a_stop_are_valid_decisions() -> None:
    fleet = toy_fleet()
    stop = RoundDecision.model_validate(
        decision(
            adopt=[],
            predicted_set_gain={"low": 0.0, "mid": 0.0, "high": 0.0},
            expectations=[],
            stop=True,
            ranked_candidates=[],
        )
    )
    assert stop.stop and not stop.adopt and not stop.ranked_candidates
    assert decision_problems(stop, fleet, None) == []
    adopt_only = RoundDecision.model_validate(decision(ranked_candidates=[]))
    assert decision_problems(adopt_only, fleet, None) == []


def test_the_one_look_plan_holds_round_ones_adoptions_without_alternatives() -> None:
    from fleetlab.loop.rounds import one_look_actions
    from fleetlab.loop.state import changes_in_force, one_look_ledger

    adopt_only = RoundDecision.model_validate(
        decision(
            adopt=[{"action": RECAL, "rationale": "a"}, {"action": CLEAN, "rationale": "b"}],
            ranked_candidates=[],
        )
    )
    actions = one_look_actions(adopt_only)
    assert [a.model_dump(mode="json") for a in actions] == [RECAL, CLEAN]
    plan = one_look_ledger("w01", actions, 2, 4)  # lane state's cut at the final count
    assert [e.action.model_dump(mode="json") for e in plan.entries] == [RECAL, CLEAN]
    assert changes_in_force(plan) == 2
    # Alternatives follow the adoptions, each action once, and the cut takes them in order.
    with_alternatives = RoundDecision.model_validate(
        decision(ranked_candidates=ranked(CLEAN, RECAL))
    )
    listed = [a.model_dump(mode="json") for a in one_look_actions(with_alternatives)]
    assert listed[:2] == [RECAL, CLEAN] and listed.count(RECAL) == 1
    plan = one_look_ledger("w01", one_look_actions(with_alternatives), 3, 4)
    assert [e.action.model_dump(mode="json") for e in plan.entries] == listed[:3]
