"""Valuing fix sets with the oracle. The implementation lives in `fleetlab.oracle.valuation`,
so that the oracle's chance interval and the scorer's random baseline are
one computation; this module keeps the scorer's names.
"""

from __future__ import annotations

from fleetlab.oracle.valuation import (
    FRESH,
    HOURS_SLACK,
    DrawResult,
    DrawValuer,
    RemovalOptions,
    Valued,
    best_removal,
    cheapest_fix,
    complete_actions,
    missing_text,
    points,
    removal_options,
    value_or_missing,
)

__all__ = [
    "FRESH",
    "HOURS_SLACK",
    "DrawResult",
    "DrawValuer",
    "RemovalOptions",
    "Valued",
    "best_removal",
    "cheapest_fix",
    "complete_actions",
    "missing_text",
    "points",
    "removal_options",
    "value_or_missing",
]
