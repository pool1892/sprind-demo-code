"""The oracle (docs/design/contracts.md sections 8 and 9, the experimental design's
sections 1 and 4 to 6): per-episode replays of a world's
scoring window under subsets of its faults, the table they fill, and every gain that
follows, with its uncertainty. See README.md in this directory.

Modules: `world` (a finished world as the oracle reads it, and where it writes), `plan`
(the replay manifest and its counts), `run` and `proof` (the replays as one heavy job,
proven against the world's as-run chains), `table` (the `OracleTable` from
`job_outcomes`), `gains` (fault effects, interactions, achievable gains, every fault
removed, with a bootstrap over scoring episodes), `chance` (random fixes), `detect` (the
generator's detectability list against the oracle), `fresh` (added changes and episodes
above the cap), `valuation` (fix sets through the table; the scorer uses it), `report`
(oracle.json, summary.md, episodes.parquet, truth/oracle.parquet), `steps` (the steps on
a world root) and `__main__` (the command line).

Importing the package imports nothing heavy; the engine is imported only by a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fleetlab.oracle.plan import OraclePlan, estimate_from_spec, plan_world
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.table import build_table
    from fleetlab.oracle.world import OracleWorld, load_world

_EXPORTS = {
    "Oracle": "steps",
    "OraclePlan": "plan",
    "OracleWorld": "world",
    "build_table": "table",
    "estimate_from_spec": "plan",
    "load_world": "world",
    "plan_world": "plan",
}

__all__ = [
    "Oracle",
    "OraclePlan",
    "OracleWorld",
    "build_table",
    "estimate_from_spec",
    "load_world",
    "plan_world",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
