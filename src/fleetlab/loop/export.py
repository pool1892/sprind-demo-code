"""The export a round's agent reads (docs/design/loop.md, section 5).

    export = export_round(world, layout, 2, ledger_r1, paths, out)

The fleet's whole visible history up to the round, as the fleet's own records hold it:

- the operation runs of rounds 1 .. K (round 1's is the world's observed run);
- the agent's own experiment arms of rounds 1 .. K so far (`rounds/r<k>/experiments/`
  of the round's analysis), whose episodes carry their experiment id;
- the description, the change log, the maintenance log and the catalogue from the
  public model after the ledger on the round's last operation day
  (`LoopWorld.public_fleet`), with the adopted actions as the fleet's own `ops` entries;
- the daily health readings of every day up to the round's last operation day, each day
  under the ledger in force that day;
- the probes the round's analyses composed (`fleetlab.loop.probes`), as the export's own
  tables `probe_episodes`, `probe_segments` and `probe_steps`, their runs audited too.

Before anything is returned, the pre-export checks of contracts section 11 run on the
built tree (`fleetlab.log.export_runs` with its leak audit): the visible allow-list, the
neutral identifiers, and the word search for truth terms and canaries over every file,
with the truth vocabulary taken from the loop's runs *and* the world's own runs and
files, and the world's full model (traits) as the audit fleet. No evaluation result,
oracle value or fault name enters an export: the evaluation's runs are never among the
runs exported, and its identities are never minted before the evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fleetlab.loop.layout import LoopLayout
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.state import Ledger
from fleetlab.loop.world import LoopWorld

__all__ = ["export_round", "health_history", "round_runs"]


def round_runs(
    world: LoopWorld, paths: LoopPaths, round_number: int, analyses: Sequence[str] = ()
) -> list[Path]:
    """The finished runs round `round_number`'s export holds, in order: operation runs of
    rounds 1 .. K (round 1's is the world's observed run unless the loop ran its own),
    then the finished arm runs of each round's analysis in `analyses` (one per round, in
    round order; an empty string for a round without one)."""
    from fleetlab.log import run_status

    runs: list[Path] = []
    for number in range(1, round_number + 1):
        own = paths.ops(number)
        if (own / "run.json").is_file():
            runs.append(own)
        elif number == 1:
            runs.append(world.observed_run)
        else:
            raise FileNotFoundError(f"round {number} has not operated yet ({own})")
    for number, analysis in enumerate(analyses, start=1):
        if number > round_number or not analysis:
            continue
        root = paths.experiments(number, analysis)
        runs += sorted(
            d
            for d in root.glob("session-*")
            if d.is_dir() and (d / "run.json").is_file() and run_status(d).finalized
        )
    return runs


def health_history(
    world: LoopWorld, layout: LoopLayout, round_number: int, ledger: Ledger
) -> list[dict[str, Any]]:
    """The health readings of every day up to round `round_number`'s last operation day,
    each day under the ledger in force on it (a removed fault stops showing on the day
    its entry took effect)."""
    last = layout.round(round_number).ops.last
    return world.health_rows(list(range(last + 1)), ledger)


def export_round(
    world: LoopWorld,
    layout: LoopLayout,
    round_number: int,
    ledger: Ledger,
    paths: LoopPaths,
    out: Path,
    *,
    analyses: Sequence[str] = (),
    overwrite: bool = True,
) -> Path:
    """Build and leak-check round `round_number`'s export in `out`; returns `out`.

    `ledger` is the ledger in force during the round (version `round_number - 1`).
    Raises `fleetlab.log.ExportRefused` on any problem, and nothing is published then.
    """
    from fleetlab.log import export_runs
    from fleetlab.loop.probes import probe_runs, write_probe_tables

    late = [
        e.entry_id for e in ledger.entries if e.adopted_day > layout.round(round_number).ops.first
    ]
    if late:
        raise ValueError(f"entries {late} take effect after round {round_number} starts")
    runs = round_runs(world, paths, round_number, analyses)
    probes = probe_runs(paths, round_number, analyses)
    last = layout.round(round_number).ops.last
    public = world.public_fleet(last, ledger)
    export_runs(
        runs,
        out,
        fleet=public,
        audit_fleet=world.model,
        task_classes=list(world.task_classes),
        health=health_history(world, layout, round_number, ledger),
        overwrite=overwrite,
        extra_tables=(lambda stage: write_probe_tables(stage, probes)) if probes else None,
        audit_runs=probes,
    )
    return Path(out)
