"""The supervisor skills: the telemetry signs of assess-risk, the rule of choose-intervention."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import skills_fleet as sf
from skills_support import helpers

from fleetlab.contracts.live import Decision, StepTelemetry

assess = helpers("assess-risk")
choose_module = helpers("choose-intervention")
Situation = choose_module.Situation


def episode_window(failure: str | None) -> list[StepTelemetry]:
    planned = next(p for p in sf.plan() if p.failure == failure)
    rows, _ = sf.telemetry(planned)
    return rows


def test_signs_see_a_drop() -> None:
    signs = assess.signs(episode_window("drop"))
    assert (
        signs["lost_object_at"] == sf.DROP_STEP and signs["phase_hint"] == "carry: lost the object"
    )
    assert not signs["holding_now"] and not signs["closed_on_nothing"]


def test_signs_see_a_missed_grasp() -> None:
    signs = assess.signs(episode_window("miss"))
    assert signs["closed_on_nothing"] and signs["phase_hint"] == "grasp: closed on nothing"
    assert signs["lost_object_at"] is None


def test_signs_see_a_stall_and_a_success() -> None:
    stalled = assess.signs(episode_window("wander"), still_speed=0.1)
    assert stalled["longest_still_run"] >= 8 and stalled["holding_now"]
    done = assess.signs(episode_window(None))
    assert done["phase_hint"] == "place or retreat: opened again" and done["lift_m"] > 0


def test_signs_count_reversals_and_read_dictionaries() -> None:
    window = [
        {
            "step": k,
            "gripper_qpos": [0.04, -0.04],
            "joint_vel": [0.3] * 7,
            "eef_pos": [0.1 * (k % 2), 0.0, 0.2],
        }
        for k in range(8)
    ]
    signs = assess.signs(window)
    assert signs["reversals"] == 6 and signs["phase_hint"] == "approach: gripper open"
    with pytest.raises(ValueError):
        assess.signs([])


def test_signs_use_a_known_open_width() -> None:
    window = episode_window("miss")[
        5:
    ]  # starts after the grasp: the window alone has no open width
    assert not assess.signs(window)["closed_on_nothing"]
    assert assess.signs(window, open_width=2 * sf.OPEN)["closed_on_nothing"]


def situation(**changes: Any) -> Any:
    base: dict[str, Any] = {
        "p_fail_continue": 0.1,
        "p_fail_retry": 0.4,
        "p_fail_hand_over": 0.1,
        "operator_minutes": 2.0,
        "operator_cost_per_minute": 0.1,
    }
    return Situation(**{**base, **changes})


def test_the_rule_continues_a_healthy_attempt() -> None:
    assert choose_module.choose(situation())[0] == "continue"


def test_the_rule_retries_a_recoverable_failure_once() -> None:
    failing = situation(p_fail_continue=0.95, p_fail_retry=0.3, operator_cost_per_minute=0.3)
    assert choose_module.choose(failing)[0] == "retry"
    again = situation(
        p_fail_continue=0.95, p_fail_retry=0.3, operator_cost_per_minute=0.3, retry_used=True
    )
    decision, costs = choose_module.choose(again)
    assert "retry" not in costs and decision == "hand_over"


def test_the_rule_hands_over_when_operator_time_is_cheap() -> None:
    lost = situation(p_fail_continue=0.95, p_fail_retry=0.9, operator_cost_per_minute=0.05)
    assert choose_module.choose(lost)[0] == "hand_over"


def test_the_rule_stops_when_help_is_dear_or_far() -> None:
    dear = situation(
        p_fail_continue=0.95, p_fail_retry=0.95, operator_cost_per_minute=0.6, doomed_waste=0.5
    )
    assert choose_module.choose(dear)[0] == "stop"
    far = situation(
        p_fail_continue=0.95,
        p_fail_retry=0.95,
        operators_free=0,
        waiting_cost=1.0,
        doomed_waste=0.5,
    )
    assert choose_module.choose(far)[0] == "stop"
    wasteful = situation(p_fail_continue=0.8, doomed_waste=0.5, options=("continue", "stop"))
    assert choose_module.choose(wasteful)[0] == "stop"


def test_without_waste_stopping_never_beats_carrying_on() -> None:
    doomed = situation(p_fail_continue=1.0, options=("continue", "stop"))
    assert choose_module.choose(doomed)[0] == "continue"  # a tie goes to the gentler option


def test_the_rule_respects_the_options_and_the_decision_names() -> None:
    only = situation(p_fail_continue=0.95, doomed_waste=0.5, options=("continue", "stop"))
    decision, costs = choose_module.choose(only)
    assert set(costs) == {"continue", "stop"} and decision == "stop"
    assert set(choose_module.OPTIONS) == {d.value for d in Decision}
    with pytest.raises(ValueError):
        situation(p_fail_continue=1.5)
    with pytest.raises(ValueError):
        situation(options=("invented",))


def test_signs_take_contract_telemetry() -> None:
    row = StepTelemetry(
        step=0,
        time_s=0.0,
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        gripper_qpos=np.array([0.04, -0.04]),
        eef_pos=np.array([0.0, 0.0, 0.3]),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        commanded_action=np.zeros(7, dtype=np.float32),
        chunk_index=0,
        chunk_step=0,
    )
    signs = assess.signs([row])
    assert signs["from_step"] == signs["to_step"] == 0 and signs["longest_still_run"] == 1
