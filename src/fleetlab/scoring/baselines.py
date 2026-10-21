"""The stage-2 fixed prior, retained as an explicitly labelled fallback.

The live no-data model and other export-only baselines live in fleetlab.baselines.
read_baseline_fix_set loads their authoritative budget answers for claim 3.
"""

from __future__ import annotations

from pathlib import Path

from fleetlab.baselines.prior import PRIOR as PRIOR
from fleetlab.baselines.prior import trivial_no_data_ranker as trivial_no_data_ranker
from fleetlab.contracts.actions import Budget, FixSet, fits_budget


def read_baseline_fix_set(root: str | Path, budget: Budget) -> FixSet:
    """Read an export-only baseline's authoritative, contract-valid budget answer.

    Explicit FixSets preserve repeated video nominations (fewer distinct fixes),
    instead of selecting the padding required by RankedFixSet's ten-entry minimum.
    Other methods' explicit files are exactly the usual cut of their ranking.
    """
    label = f"{budget.kind}-{budget.amount:g}"
    result = FixSet.model_validate_json((Path(root) / "fix_sets" / f"{label}.json").read_text())
    if result.budget != budget or not fits_budget(result):
        raise ValueError("baseline FixSet does not match or fit the requested budget")
    return result
