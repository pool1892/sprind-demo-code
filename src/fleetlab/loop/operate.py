"""Operation of the fleet's regular work under a ledger (docs/design/loop.md, section 3).

    run_dir = operate(world, layout, 2, ledger_r1, paths, setup)

A round's operation is the world's scheduled regular work on the round's operation days:
the world's own as-run identities (`LoopWorld.regular_identities`: the same episode ids,
tasks, start states and seeds as the world's manifest), with the current software and
each attempt's faults materialized under the ledger in force on its day
(`LoopWorld.state_on`). It goes through the normal engine (`fleetlab.engine.fleet.run`)
and log writer, and then the fleet's own records go into the finished run, as the world
runner does for the world's own runs:

- the description and the change and maintenance logs as they stand on the round's last
  operation day after the ledger (`LoopWorld.public_fleet`): a `set` or `revert` is a
  change-log entry by `ops` on its adoption day, a component verb a maintenance entry,
  a condition change the site's believed condition (there is no condition log);
- the health readings of the operation days under the ledger (`LoopWorld.health_rows`);
- the derived events (`fleetlab.worlds.events.add_events`: drop, stall, near miss, and
  with the contact pass, collisions), of the kinds the world's own runs carry
  (`LoopWorld.event_setup`): a run of the loop extends the world's history with records
  of the same kinds, so a kind of event never appears only from round 2 on.

Round 1 under the original state is the world's stored observed run itself: that is the
operation the original state had, and it was run, so nothing is simulated again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import Job
from fleetlab.loop.layout import LoopLayout, days_of
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.state import Ledger
from fleetlab.loop.world import LoopWorld

__all__ = ["EngineSetup", "add_events", "operate", "operation_jobs", "write_round_records"]


@dataclass(frozen=True)
class EngineSetup:
    """How the engine runs the loop's jobs.

    `config` is a `fleetlab.engine.fleet.RunConfig` template (workers, batch size,
    precision, the factories and the fault registry); each run sets its own `out` and the
    world's day length. `contacts` adds the contact replay to the derived events (LIBERO
    runs only); `events` switches the derived events off altogether (the toy cell).

    `panel_video_indices`: the reserved panel's attempts that record their two videos in
    every evaluated state, by their index within each robot and day (M3: `(0,)`, one
    attempt per robot and day, docs/design/m3-recipe.md, section 3); empty, none does.
    It travels with the setup into every heavy step, so `fleetlab.loop.evaluate` honours
    it wherever the evaluation runs. `video` is the operation runs' and is unrelated.
    """

    config: Any
    contacts: bool = True
    events: bool = True
    video: bool = True  # every attempt records its two videos, as the world's own runs do
    contact_workers: int = 4
    panel_video_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # A setup read back from JSON brings a list; keep one immutable, sorted form.
        indices = tuple(sorted({int(i) for i in self.panel_video_indices}))
        if any(i < 0 for i in indices):
            raise ValueError(f"a panel attempt's index is not negative: {indices}")
        object.__setattr__(self, "panel_video_indices", indices)

    def for_run(self, out: Path, per_day: int) -> Any:
        return replace(
            self.config,
            out=str(out),
            episodes_per_robot_day=per_day,
            allow_manifest_day_length=False,
        )


def operation_jobs(
    world: LoopWorld,
    layout: LoopLayout,
    round_number: int,
    ledger: Ledger,
    software: Any,
    record_video: bool = True,
) -> list[Job]:
    """The run jobs of round `round_number`'s operation under `ledger`."""
    days = days_of(layout.round(round_number).ops)
    return [
        Job(identity=identity, record_video=record_video)
        for identity in world.regular_identities(days, software, ledger)
    ]


def _check_ledger(layout: LoopLayout, round_number: int, ledger: Ledger) -> None:
    first = layout.round(round_number).ops.first
    late = [e.entry_id for e in ledger.entries if e.adopted_day > first]
    if late:
        raise ValueError(f"entries {late} take effect after round {round_number} starts")


def operate(
    world: LoopWorld,
    layout: LoopLayout,
    round_number: int,
    ledger: Ledger,
    paths: LoopPaths,
    setup: EngineSetup,
) -> Path:
    """Round `round_number`'s operation under `ledger`; returns the finished run directory.

    Resumable: a run whose jobs are all done is not run again (the engine resumes a
    killed run by itself); the records are written again, which changes no byte when
    nothing changed.
    """
    from fleetlab.engine import fleet

    _check_ledger(layout, round_number, ledger)
    if round_number == 1 and not ledger.entries:
        return world.observed_run
    ledger.save(paths.ledgers)
    run_dir = paths.ops(round_number)
    software = fleet.prepare(setup.config)
    jobs = operation_jobs(world, layout, round_number, ledger, software, setup.video)
    if not _complete(run_dir, jobs):
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        fleet.run(setup.for_run(run_dir, world.truth.episodes_per_robot_day), jobs)
    write_round_records(world, layout, round_number, ledger, run_dir)
    events = world.event_setup(setup)
    if events.events:
        add_events(run_dir, events)
    return run_dir


def _complete(run_dir: Path, jobs: Sequence[Job]) -> bool:
    from fleetlab.contracts.rollout import RunLayout
    from fleetlab.log import run_status

    if not (run_dir / "run.json").is_file():
        return False
    layout = RunLayout(run_dir)
    return all(layout.is_done(j) for j in jobs) and run_status(run_dir).finalized


def add_events(run_dir: Path, setup: EngineSetup) -> dict[str, int]:
    """The derived events into a finished run, as the world runner adds them: from the
    telemetry, and with `setup.contacts` from the contact replay of every episode."""
    import importlib

    events: Any = importlib.import_module("fleetlab.worlds.events")
    contacts = None
    if setup.contacts:
        found = events.contact_pass(run_dir, workers=setup.contact_workers)
        if found.mismatched:
            raise RuntimeError(
                f"{run_dir}: {len(found.mismatched)} episodes did not replay to their chains "
                f"in the contact pass: {sorted(found.mismatched)[:5]}"
            )
        contacts = found.events
    counts: dict[str, int] = events.add_events(run_dir, contacts)
    return counts


def write_round_records(
    world: LoopWorld, layout: LoopLayout, round_number: int, ledger: Ledger, run_dir: Path
) -> None:
    """The fleet's records of the round into its finished operation run: description and
    logs after the ledger on the round's last operation day, health of its days."""
    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.log import RunWriter

    ops = layout.round(round_number).ops
    public = world.public_fleet(ops.last, ledger)
    rows = world.health_rows(days_of(ops), ledger)
    record = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    with RunWriter(
        run_dir, record, episodes_per_robot_day=world.truth.episodes_per_robot_day
    ) as writer:
        writer.write_model(public, list(world.task_classes))
        writer.write_table("health", rows)
        writer.finalize()
