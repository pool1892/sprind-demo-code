"""The decision rule of choose-intervention as arithmetic, for a harness that checks or logs it.

Costs are in failed attempts. The caller supplies every number: the chances
come from the supervisor's assessment, the cost of an operator minute from
the run's cost ratio. Nothing here is a value of a particular fleet.

    from choose_intervention_helpers import Situation, choose
    choose(Situation(p_fail_continue=0.9, p_fail_retry=0.5, p_fail_hand_over=0.1,
                     operator_minutes=2.0, operator_cost_per_minute=0.2,
                     options=("continue", "retry", "stop", "hand_over")))
"""

from __future__ import annotations

from dataclasses import dataclass

OPTIONS = ("continue", "retry", "recover", "degrade", "stop", "hand_over")


@dataclass(frozen=True)
class Situation:
    """What the comparison needs, all from the assessment and the run's settings.

    `doomed_waste` is the time a failing attempt still wastes if it runs on,
    as a share of an attempt (a failing attempt runs to the step limit, a
    success ends early); without it, stopping never beats carrying on, since
    a stop is a failure too. `retry_overhead` is the time a retry costs,
    `waiting_cost` the cost of waiting for an operator when none is free.
    Safety (an arm pressing against the cell) is not in the arithmetic; the
    rule in words stops for it.
    """

    p_fail_continue: float
    p_fail_retry: float
    p_fail_hand_over: float
    operator_minutes: float
    operator_cost_per_minute: float
    options: tuple[str, ...] = ("continue", "retry", "stop", "hand_over")
    p_fail_recover: float = 1.0
    p_fail_degrade: float = 1.0
    recover_overhead: float = 0.0
    degrade_overhead: float = 0.0
    retry_used: bool = False
    operators_free: int = 1
    waiting_cost: float = 0.0
    doomed_waste: float = 0.0
    retry_overhead: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "p_fail_continue",
            "p_fail_retry",
            "p_fail_hand_over",
            "p_fail_recover",
            "p_fail_degrade",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} is a probability, not {value}")
        unknown = sorted(set(self.options) - set(OPTIONS))
        if unknown or not self.options:
            raise ValueError(f"options are some of {OPTIONS}, not {self.options}")


def expected_costs(situation: Situation) -> dict[str, float]:
    """The expected cost of each available option, in failed attempts."""
    s = situation
    costs = {
        "continue": s.p_fail_continue * (1.0 + s.doomed_waste),
        "retry": s.p_fail_retry * (1.0 + s.doomed_waste) + s.retry_overhead,
        "recover": s.p_fail_recover * (1.0 + s.doomed_waste) + s.recover_overhead,
        "degrade": s.p_fail_degrade * (1.0 + s.doomed_waste) + s.degrade_overhead,
        "stop": 1.0,
        "hand_over": s.p_fail_hand_over
        + s.operator_minutes * s.operator_cost_per_minute
        + (s.waiting_cost if s.operators_free <= 0 else 0.0),
    }
    # A second retry for the same symptom is not offered: it rarely works.
    return {
        option: round(cost, 6)
        for option, cost in costs.items()
        if option in s.options and not (option == "retry" and s.retry_used)
    }


def choose(situation: Situation) -> tuple[str, dict[str, float]]:
    """The cheapest available option and every option's cost; ties go to the gentler option."""
    costs = expected_costs(situation)
    if not costs:
        raise ValueError("no option is available")
    order = {option: i for i, option in enumerate(OPTIONS)}
    best = min(costs, key=lambda option: (costs[option], order[option]))
    return best, costs
