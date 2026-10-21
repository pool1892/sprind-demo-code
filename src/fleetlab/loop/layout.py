"""The chronology of a loop: the rounds' operation and experiment days, and the evaluation
(docs/design/loop.md, section 1).

    layout = layout_for_world(spec)            # an existing world, two rounds
    layout.adoption_day(1)                     # the day round 1's decision takes effect

A round is an operation window (the fleet's regular work, whose records the round's agent
reads) followed by an experiment window (days given to the agent's arms). The decision a
round ends with takes effect on the next day, its *adoption day*: the first day of the
next round's operation, or of the evaluation after the last round. The windows follow
each other strictly: round 1's operation, round 1's experiments, round 2's operation,
and so on, then the evaluation.

On a stored world (`layout_for_world`): round 1 is the world's observed and experiment
windows; round 2's operation is the world's scheduled regular work on its old scoring
days; round 2's experiment days and the evaluation days are appended after the world's
last day (by default as many as the world's experiment and scoring windows hold).

Choices this module makes where the contract is silent:

- Experiment days carry only the agent's arms. The world's own regular work on round 1's
  experiment days (the as-run "later" run) is not part of the loop's history: no round
  operates on those days, so no export shows them. They are test days.
- Days past the stored world's horizon (`world_days`) are allowed for experiments and the
  evaluation; `fleetlab.loop.world` says how faults and identities extend there.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.world import DayRange, WorldSpec

__all__ = ["LoopLayout", "RoundWindows", "days_of", "layout_for_world"]


def days_of(window: DayRange) -> list[int]:
    """The days of an inclusive range, in order."""
    return list(range(window.first, window.last + 1))


class RoundWindows(ContractModel):
    """One round: its operation days, then its experiment days."""

    ops: DayRange
    experiment: DayRange

    @model_validator(mode="after")
    def _ordered(self) -> RoundWindows:
        if not self.ops.last < self.experiment.first:
            raise ValueError("a round's experiment days follow its operation days")
        return self


class LoopLayout(ContractModel):
    """The rounds in order, then the evaluation, in strictly increasing days.

    `world_days` is the stored world's horizon (its days 0 .. world_days - 1); a window
    may reach past it.
    """

    rounds: list[RoundWindows] = Field(min_length=1)
    evaluation: DayRange
    world_days: int = Field(ge=1)

    @model_validator(mode="after")
    def _in_order(self) -> LoopLayout:
        last = -1
        for number, windows in enumerate(self.rounds, start=1):
            if windows.ops.first <= last:
                raise ValueError(f"round {number} starts on or before day {last}")
            last = windows.experiment.last
        if self.evaluation.first <= last:
            raise ValueError("the evaluation follows the last round")
        return self

    def round(self, number: int) -> RoundWindows:
        """Round `number`, counted from 1."""
        if not 1 <= number <= len(self.rounds):
            raise KeyError(f"no round {number}; the layout has {len(self.rounds)}")
        return self.rounds[number - 1]

    def adoption_day(self, number: int) -> int:
        """The day round `number`'s decision takes effect: the next round's first
        operation day, or the evaluation's first day after the last round."""
        self.round(number)
        if number < len(self.rounds):
            return self.rounds[number].ops.first
        return self.evaluation.first

    def round_of_day(self, day: int) -> int | None:
        """The round whose operation or experiment days hold `day`, or None."""
        for number, windows in enumerate(self.rounds, start=1):
            if day in windows.ops or day in windows.experiment:
                return number
        return None

    @property
    def last_day(self) -> int:
        return self.evaluation.last

    def past_horizon(self, day: int) -> bool:
        """Whether a day lies after the stored world's last day."""
        return day >= self.world_days


def layout_for_world(
    spec: WorldSpec,
    *,
    experiment_days: int | None = None,
    evaluation_days: int | None = None,
) -> LoopLayout:
    """Two rounds on a stored world (loop.md, section 1).

    Round 1: the observed window, then the experiment window. Round 2: the scoring window
    as its operation, then `experiment_days` appended days (default: the experiment
    window's length). The evaluation: `evaluation_days` appended days after those
    (default: the scoring window's length).
    """
    windows = spec.windows
    exp_len = experiment_days or (windows.experiment.last - windows.experiment.first + 1)
    eval_len = evaluation_days or (windows.scoring.last - windows.scoring.first + 1)
    if exp_len < 1 or eval_len < 1:
        raise ValueError("a round needs at least one experiment day, the evaluation one day")
    second_exp = DayRange(first=windows.scoring.last + 1, last=windows.scoring.last + exp_len)
    evaluation = DayRange(first=second_exp.last + 1, last=second_exp.last + eval_len)
    return LoopLayout(
        rounds=[
            RoundWindows(ops=windows.observed, experiment=windows.experiment),
            RoundWindows(ops=windows.scoring, experiment=second_exp),
        ],
        evaluation=evaluation,
        world_days=spec.days,
    )
