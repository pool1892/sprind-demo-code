"""The round driver's prompts and carry-over (fleetlab.loop.rounds): what the agent is told
in round N, and that nothing but its own records crosses between rounds. No model."""

from __future__ import annotations

from fleetlab.loop.driver import AttemptBudget
from fleetlab.loop.rounds import decision_reminder, round_continuation, round_task
from fleetlab.toolserver.decisions import PublicLedger, PublicLedgerEntry

REVERT = {
    "verb": "revert",
    "target": {
        "scope": "s2",
        "component": None,
        "config_key": "control.action_delay_steps",
        "condition_key": None,
    },
    "value": None,
}
SET = {
    "verb": "set",
    "target": {
        "scope": "r03",
        "component": None,
        "config_key": "policy.actions_executed_per_chunk",
        "condition_key": None,
    },
    "value": 10,
}

# Words that would carry the evaluation or the truth into a prompt.
FORBIDDEN = ("evaluation", "reserved", "oracle", "fault", "truth", "panel", "scor")


def test_round_one_says_what_an_engagement_is_and_that_stopping_is_valid() -> None:
    budget = AttemptBudget(attempts=480, per_arm=60)
    task = round_task(
        round_number=1,
        rounds=2,
        operation=[(1, (0, 3))],
        experiment_days=(4, 5),
        budget_text=budget.text(),
        ledger=PublicLedger(round=1, changes=[]),
    )
    assert "round 1 of an ongoing engagement" in task and "at most 2 rounds" in task
    assert "Stopping is a valid\nchoice" in task or "Stopping is a valid choice" in task
    assert "480 robot attempts in all" in task and "at most 60 attempts per arm" in task
    assert "days 4 to 5" in task and "round 1: days 0 to 3" in task
    assert "submit_decision" in task and "experiment_budget" in task
    # Provisional explanations with a rough confidence before testing; light
    # (no tool-order gate, nothing refused when it is skipped).
    assert "Before you request experiments, write\ndown your provisional competing" in task
    assert "each with a rough confidence" in task
    assert "The fleet's ledger" not in task  # nothing adopted yet
    lowered = task.lower()
    assert not [w for w in FORBIDDEN if w in lowered]


def test_round_two_carries_the_ledger_and_points_to_the_agents_own_decisions() -> None:
    ledger = PublicLedger(
        round=2,
        changes=[
            PublicLedgerEntry.model_validate(
                {
                    "entry_id": "L1",
                    "round": 1,
                    "adopted_day": 6,
                    "action": REVERT,
                    "status": "active",
                }
            ),
            PublicLedgerEntry.model_validate(
                {
                    "entry_id": "L2",
                    "round": 1,
                    "adopted_day": 6,
                    "action": SET,
                    "status": "active",
                }
            ),
        ],
    )
    task = round_task(
        round_number=2,
        rounds=2,
        operation=[(1, (0, 3)), (2, (6, 7))],
        experiment_days=(8, 8),
        budget_text=AttemptBudget().text(),
        ledger=ledger,
        sessions=[(1, (1, 2))],
    )
    assert "round 2 of an ongoing engagement" in task
    assert "- L1 (round 1, from day 6, in force): revert control.action_delay_steps at s2" in task
    assert "set policy.actions_executed_per_chunk at r03 to 10" in task
    assert "round 1: sessions 1 to 2" in task and "earlier_decisions returns every" in task
    assert "Nobody has told you whether your changes were right" in task
    assert "round 2: days 6 to 7" in task
    lowered = task.lower()
    assert not [w for w in FORBIDDEN if w in lowered]


def test_the_loop_budget_limits_an_arm_only_when_a_run_says_so() -> None:
    free = AttemptBudget(attempts=480)
    assert free.limits()["max_episodes_per_arm"] == 480
    assert "an arm may take any part of it" in free.text()
    assert "attempts per arm" not in free.text()
    mini = AttemptBudget(attempts=120, per_arm=20, requests=2)
    assert mini.limits(3) == {
        "max_experiment_requests": 2,
        "max_episodes_per_arm": 20,
        "budget_first_session": 3,
    }
    assert "at most 20 attempts per arm" in mini.text()
    assert AttemptBudget(attempts=480, per_arm=200).limits()["max_episodes_per_arm"] == 200


def test_continuations_name_the_round_and_session() -> None:
    assert "session 3 of round 2" in round_continuation(2, 3, "the task")
    reminder = decision_reminder(1, 2, "the task")
    assert "without an accepted decision" in reminder and "to\nstop" not in reminder
    assert reminder.endswith("The task:\nthe task")
