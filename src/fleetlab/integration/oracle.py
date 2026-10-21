"""The miniature fleet's oracle, through `fleetlab.oracle`, which grew out of
this module.

`oracle_jobs` makes the replay jobs of the world's scoring episodes
(`fleetlab.oracle.plan.plan_replays`): the contract's power set of each episode's
effective faults, then the confirmation replays with its decoys (and inert confounder
entries) removed as well; for the miniature fleet 65 and 65. The chain runs them in its
`w10-later` run beside the scoring window as run, whose chain sample proves them, so no
proof replays are added here.

`oracle_table` fills an `OracleTable` from the runs' `job_outcomes`
(`fleetlab.oracle.table.build_table`), only from runs whose chain sample held: a run that
gives no rows raises `ChainBroken`, because the chain names only runs it expects rows
from. The gains, their intervals and the files come from `fleetlab.oracle.report`
(`chain.step_oracle`).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fleetlab.contracts.oracle import OracleTable
from fleetlab.contracts.rollout import RunRecord
from fleetlab.oracle.plan import OraclePlan, plan_replays
from fleetlab.oracle.table import build_table
from fleetlab.worlds.generate import World

__all__ = [
    "ChainBroken",
    "OraclePlan",
    "check_chain",
    "oracle_jobs",
    "oracle_table",
    "oracle_table_and_count",
    "run_record",
]


def oracle_jobs(world: World) -> OraclePlan:
    """The replay jobs of the world's scoring episodes: the power set, then the confirmations."""
    identities = [j.identity for j in world.jobs_in("scoring")]
    return plan_replays(identities, world.spec.patterns.excluded_from_replay(), confirm=True)


class ChainBroken(RuntimeError):
    """A run gives no oracle rows: its chain sample did not hold, or it is unfinished."""


def run_record(run_dir: Path) -> RunRecord:
    return RunRecord.model_validate_json((run_dir / "run.json").read_text())


def check_chain(run_dir: Path) -> RunRecord:
    """The run's record, if its chain sample held; ChainBroken otherwise."""
    record = run_record(run_dir)
    if record.chain_ok is not True:
        raise ChainBroken(
            f"{run_dir}: chain_ok is {record.chain_ok!r}; the run gives no oracle rows"
        )
    return record


def oracle_table(world: World, run_dirs: Iterable[Path]) -> OracleTable:
    """The scoring episodes with every outcome the runs recorded for them."""
    return oracle_table_and_count(world, run_dirs)[0]


def oracle_table_and_count(world: World, run_dirs: Iterable[Path]) -> tuple[OracleTable, int]:
    """The table, and how many distinct (episode, fault list) rows it holds."""
    build = build_table(
        world.identities("scoring"), world.spec.patterns.excluded_from_replay(), run_dirs
    )
    refused = [f"{r.run_dir}: {r.refused}" for r in build.runs if r.refused]
    if refused:
        raise ChainBroken("; ".join(refused))
    return build.table, build.rows_held
