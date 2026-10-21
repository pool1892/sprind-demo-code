"""The round decision and the attempt budget of the improvement loop (docs/design/loop.md, 8).

An engagement of the analysis agent with a fleet runs in rounds. A round ends with
the agent's `RoundDecision`, submitted with the tool `submit_decision`: at most
`MAX_ADOPTED_PER_ROUND` new changes, replacements and undos of earlier ones (named
by their ledger entry id, `L1`, `L2`, ...), the predicted gain of the whole set over
the fleet's current state, whether to stop, and optionally the further candidates it
considered, best first (`ranked_candidates`, which may be empty). The plan fixed after
one look is round 1's adopted actions in order, then those alternatives in order, cut
at the final number of changes. Expectations are concrete statements about the next
round's records that the agent can check itself then.

The experiment budget of a round is counted in robot attempts: the episodes over
every arm of every request (`episodes_per_arm` times arms). An arm is limited only by
what is left of the round's attempts, unless a run sets a limit per arm (the
miniature run's 20); the historical stage-4 default of 60 per arm applies only to the
old pipeline's limits. A refused request is not charged. The tool `experiment_budget`
reports the budget in force and what is left of it (`ExperimentBudgetStatus`); the
description of `request_experiment` and every refusal state the limits as well.

How replacements and undos count against the number of changes, and the cut of the
one-look plan, follow the one convention of `fleetlab.loop.state`, which the driver
and the evaluator share; this module only bounds what one decision may hold.

The models are defined in `contracts.tools` beside `RankedAction`, which the decision
holds (an import cycle otherwise), and re-exported here, the documented home of the
names. Both are mounted into the tool sandbox with the rest of the contracts: they name
no fault kind and no world.
"""

from __future__ import annotations

from fleetlab.contracts.tools import (
    LEDGER_ENTRY_PATTERN,
    MAX_ADOPTED_PER_ROUND,
    AdoptedChange,
    ExperimentBudgetStatus,
    LedgerEntryId,
    Replacement,
    RoundDecision,
    SetGain,
    Undo,
)

__all__ = [
    "LEDGER_ENTRY_PATTERN",
    "MAX_ADOPTED_PER_ROUND",
    "AdoptedChange",
    "ExperimentBudgetStatus",
    "LedgerEntryId",
    "Replacement",
    "RoundDecision",
    "SetGain",
    "Undo",
]
