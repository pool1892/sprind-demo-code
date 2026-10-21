"""Does some ordinary setting beat the default under a lasting condition? The adaptation
measurement of docs/design/loop.md, section 7 (lane adapt).

The lasting condition is `remote_link_latency`: every command of a robot reaches
it a fixed number of control steps late, through a delay line of its own that
no action removes. The settings are the generic ones every robot has:
`control.latency_compensation_steps` (skip that many actions at the head of
each chunk after an attempt's first) and `policy.actions_executed_per_chunk`.
Every effect is simulated; nothing here plants or reads a response curve.

The cut changed between the phases. `p1` ran at 611448c, where the skip also
applied to an attempt's first chunk: during the delay line's warm-up the robot
then already moved toward the plan's pose `d` steps ahead, a head start a real
link would not give. From `p2` on (the cut of `engine.episode.call_policy`
now), the first chunk is cut as without the setting, so the warm-up is the
same in every cell. A default cell is the same episodes in either, chain for
chain, so a later phase's settings are paired with `p1`'s default cells.

A measurement is a list of cells. A cell is one suite, one delay (0: a healthy
site) and one setting, run on `episodes` paired attempts: attempt i is task
i % 10 at start state i // 10, with the scene and noise seeds of the
calibration's world (`dose_response.WORLD_SEED`, robot r01, day 1, index i),
so attempt i of every cell is the same start, and the delay-only cells repeat
the calibration's `action_latency` episodes of the same index. A setting is
applied as the loop applies an adopted action: `contracts.actions.resolve` of
`set KEY = v at s1` against the attempt's identity, which adds a
`config_change` and removes nothing.

    # one heavy job: the phase's manifest, then the engine
    tools/heavy-run uv run --locked python -m fleetlab.adaptation run p2 --workers 8

    # the tables: success with Wilson intervals, and each setting against the
    # default of its suite and delay, paired (Newcombe's interval, exact McNemar),
    # the default taken from the same phase or else from the earliest that has it
    uv run --locked python -m fleetlab.adaptation analyse p1 p2 [--json OUT]

Run directories live under `/var/tmp/robots-loop/adapt/runs/<phase>`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import Action, resolve_all
from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.ids import IdKind, id_number, neutral_id
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.dose_response import (
    BLOCK_WIDTH,
    DAY,
    IMAGE_SIZE,
    MAX_STEPS,
    ROBOT_ID,
    SITE_ID,
    WORLD_ID,
    WORLD_SEED,
    RunOutcomes,
    mcnemar_exact,
    newcombe_paired,
    read_run,
    wilson,
)

CONDITION_KIND = "remote_link_latency"
CONDITION_ID = "f01"
# Blocks of episode ids above every block the calibration uses (`dose_response`).
FIRST_BLOCK = 3000
RUN_ROOT = Path("/var/tmp/robots-loop/adapt/runs")
COMPENSATION = ConfigKey.control_latency_compensation_steps
PER_CHUNK = ConfigKey.policy_actions_executed_per_chunk
# The fleet's values of the two settings (worlds.generate.NOMINAL_CONFIG).
NOMINAL: dict[ConfigKey | ConditionKey, ConfigValue] = {COMPENSATION: 0, PER_CHUNK: 30}


@dataclass(frozen=True)
class Cell:
    """One suite, one delay of the lasting condition (0: none) and one setting."""

    suite: str
    delay: int
    setting: tuple[tuple[ConfigKey, int], ...] = ()
    episodes: int = 60

    @property
    def name(self) -> str:
        values = dict(self.setting)
        parts = [self.suite.removeprefix("libero_"), f"d{self.delay}"]
        parts.append(f"c{values.get(COMPENSATION, 0)}")
        if PER_CHUNK in values:
            parts.append(f"k{values[PER_CHUNK]}")
        return "-".join(parts)

    @property
    def is_default(self) -> bool:
        return all(NOMINAL[key] == value for key, value in self.setting)

    def default(self) -> Cell:
        """The cell with the fleet's default setting, same suite and delay."""
        return Cell(self.suite, self.delay, (), self.episodes)


def _cells(suite: str, delay: int, compensations: Sequence[int], n: int) -> list[Cell]:
    return [Cell(suite, delay, ((COMPENSATION, c),) if c else (), n) for c in compensations]


# The phases, in the order they ran. Append only: a cell's block is its place in its phase,
# and a phase is never changed after it ran.
PHASES: dict[str, list[Cell]] = {
    # The decisive first look (about 520 attempts): the two delays the calibration
    # measured large losses at (spatial 17 steps, object 14 steps), no compensation, half
    # and full; and the full compensation on a healthy site.
    "p1": [
        *_cells("libero_spatial", 17, (0, 8, 17), 60),
        *_cells("libero_object", 14, (0, 7, 14), 60),
        *_cells("libero_spatial", 0, (0, 17), 40),
        *_cells("libero_object", 0, (0, 14), 40),
    ],
    # The startup check (a review of 611448c): the four compensated delay cells and
    # the two winners on a healthy site again, with the first chunk never cut, paired
    # with p1's default cells (about 320 attempts).
    "p2": [
        *_cells("libero_spatial", 17, (8, 17), 60),
        *_cells("libero_object", 14, (7, 14), 60),
        *_cells("libero_spatial", 0, (8,), 40),
        *_cells("libero_object", 0, (14,), 40),
    ],
    # The recipe's questions, with the first chunk never cut (about 500 attempts):
    # compensating the telemetry lag instead of the delay (3 steps too many); each suite
    # at the other delay; replanning as often as compensation 17 does, without the skip;
    # and full compensation on a healthy spatial site.
    "p3": [
        *_cells("libero_spatial", 17, (20,), 60),
        *_cells("libero_object", 14, (17,), 60),
        *_cells("libero_spatial", 14, (0, 14), 60),
        *_cells("libero_object", 17, (0, 17), 60),
        Cell("libero_spatial", 17, ((PER_CHUNK, 13),), 60),
        *_cells("libero_spatial", 0, (17, 14), 40),
    ],
}


def condition(delay: int, robot_id: str = ROBOT_ID) -> FaultSpec:
    """The lasting condition on one robot: its commands `delay` steps late, from day 0."""
    return FaultSpec(
        fault_id=CONDITION_ID,
        kind=CONDITION_KIND,
        level=float(delay),
        target=Target(scope=robot_id, component=Component.controller),
        schedule=Schedule(onset_day=0),
    )


def setting_actions(cell: Cell, scope: str = SITE_ID) -> list[Action]:
    """The cell's setting as the actions an operator adopts: `set KEY = v` at the site."""
    return [
        Action(verb=Verb.set, target=Target(scope=scope, config_key=key), value=value)
        for key, value in cell.setting
    ]


def identity(
    cell: Cell,
    block: int,
    index: int,
    policy: PolicySpec,
    software: SoftwareVersion,
    batch_size: int = 4,
) -> EpisodeIdentity:
    """Attempt `index` of a cell: its paired start, the condition, the adopted setting."""
    import fleetlab.faults as faults

    scene, noise = seeds_for(WORLD_SEED, ROBOT_ID, DAY, index)
    base = EpisodeIdentity(
        world_id=WORLD_ID,
        episode_id=neutral_id(IdKind.episode, block * BLOCK_WIDTH + index),
        robot_id=ROBOT_ID,
        site_id=SITE_ID,
        day=DAY,
        episode_index=index,
        suite=cell.suite,
        task_index=index % 10,
        start_state=index // 10,
        scene_seed=scene,
        noise_seed=noise,
        policy=policy,
        batch_size=batch_size,
        row=None,
        image_size=IMAGE_SIZE,
        max_steps=MAX_STEPS,
        faults=[condition(cell.delay)] if cell.delay else [],
        software=software,
    )
    resolution = resolve_all(
        setting_actions(cell), base, base.faults, faults.truth_table(), NOMINAL
    )
    if resolution.removed:
        raise AssertionError(f"a setting removed {sorted(resolution.removed)}: the condition lasts")
    return base.model_copy(update={"faults": [*base.faults, *resolution.added]})


def plan(
    phase: str, policy: PolicySpec, software: SoftwareVersion, batch_size: int = 4
) -> list[Job]:
    """The jobs of a phase: every cell's attempts, cell by cell."""
    jobs: list[Job] = []
    for number, cell in enumerate(PHASES[phase]):
        block = FIRST_BLOCK + 100 * list(PHASES).index(phase) + number
        jobs.extend(
            Job(identity=identity(cell, block, i, policy, software, batch_size))
            for i in range(cell.episodes)
        )
    return jobs


def cell_of(phase: str, episode_id: str) -> tuple[Cell, int]:
    """The cell and attempt index of one of a phase's episodes."""
    number = id_number(IdKind.episode, episode_id)
    block, index = divmod(number, BLOCK_WIDTH)
    offset = block - FIRST_BLOCK - 100 * list(PHASES).index(phase)
    return PHASES[phase][offset], index


# ---------------------------------------------------------------------------- the tables


@dataclass(frozen=True)
class CellResult:
    phase: str
    cell: str
    suite: str
    delay: int
    setting: dict[str, int]
    n: int
    successes: int
    interval: tuple[float, float]
    mean_steps: float
    against: str | None = None  # the default cell it is paired with, as phase/cell
    pairs: int = 0
    rescued: int = 0  # default failed, this setting succeeded
    broken: int = 0  # default succeeded, this setting failed
    difference: tuple[float, float, float] | None = None  # setting minus default, interval
    p_value: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")


def outcomes_by_cell(phase: str, run: RunOutcomes) -> dict[Cell, dict[int, tuple[bool, int]]]:
    """Each cell's true outcome (the fleet's placed check) and steps, by attempt index."""
    out: dict[Cell, dict[int, tuple[bool, int]]] = {}
    for eid, outcome in run.outcomes.items():
        cell, index = cell_of(phase, eid)
        out.setdefault(cell, {})[index] = (outcome.true_success, outcome.steps)
    return out


def analyse(runs: dict[str, RunOutcomes]) -> list[CellResult]:
    """Every cell of the given phases, and each setting against its default, paired by attempt.

    A setting's default is the default cell of its suite, delay and size in its own phase,
    or else in the earliest given phase that has one.
    """
    by_phase = {phase: outcomes_by_cell(phase, run) for phase, run in runs.items()}
    results = []
    for phase in runs:
        for cell in PHASES[phase]:
            got = by_phase[phase].get(cell, {})
            n, s = len(got), sum(ok for ok, _ in got.values())
            result = CellResult(
                phase=phase,
                cell=cell.name,
                suite=cell.suite,
                delay=cell.delay,
                setting={key.value: value for key, value in cell.setting},
                n=n,
                successes=s,
                interval=wilson(s, n),
                mean_steps=sum(st for _, st in got.values()) / n if n else float("nan"),
            )
            homes = [phase, *(p for p in runs if p != phase)]
            home = next((p for p in homes if by_phase[p].get(cell.default())), None)
            if not cell.is_default and home is not None:
                reference = by_phase[home][cell.default()]
                common = sorted(set(got) & set(reference))
                a = sum(got[i][0] and reference[i][0] for i in common)
                rescued = sum(got[i][0] and not reference[i][0] for i in common)
                broken = sum(reference[i][0] and not got[i][0] for i in common)
                d = len(common) - a - rescued - broken
                result = replace(
                    result,
                    against=f"{home}/{cell.default().name}",
                    pairs=len(common),
                    rescued=rescued,
                    broken=broken,
                    # the setting first: "only the first" is a rescue, "only the second" a break
                    difference=newcombe_paired(a, rescued, broken, d),
                    p_value=mcnemar_exact(rescued, broken),
                )
            results.append(result)
    return results


def markdown(results: Sequence[CellResult], runs: dict[str, RunOutcomes]) -> str:
    """The table of the phases, one row per cell."""

    def pct(x: float) -> str:
        return f"{100 * x:.1f}"

    lines = [
        *(f"Run `{run.run_id}`, replay sample held: {run.chain_ok}." for run in runs.values()),
        "",
        "| Cell | Suite | Delay, steps | Setting | Attempts | Successes | 95% interval | "
        "Against the default: rescued / broken | Difference, points (95% interval) | "
        "McNemar p | Mean steps |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        setting = ", ".join(f"{k} = {v}" for k, v in r.setting.items()) or "default"
        if r.difference is None:
            versus, diff, p = "", "", ""
        else:
            d, lo, hi = r.difference
            versus = f"{r.rescued} / {r.broken} of {r.pairs}"
            diff = f"{100 * d:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})"
            p = f"{r.p_value:.2g}"
        lines.append(
            f"| {r.phase}/{r.cell} | {r.suite} | {r.delay} | {setting} | {r.n} | {r.successes} "
            f"({pct(r.rate)}%) | {pct(r.interval[0])} to {pct(r.interval[1])} | {versus} | "
            f"{diff} | {p} | {r.mean_steps:.0f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------- running


def run_phase(phase: str, workers: int = 8, root: Path = RUN_ROOT) -> int:
    """Write the phase's manifest and run it through the engine (heavy: loads X-VLA)."""
    import os

    from fleetlab.dose_response import _engine, _nominal_policy

    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    miopen = _engine("fleetlab.engine.miopen")
    fleet = _engine("fleetlab.engine.fleet")
    miopen.activate()
    software = _engine("fleetlab.engine.software").current(miopen.require(4, "fp16_res32"))
    jobs = plan(phase, _nominal_policy("fp16_res32"), software, 4)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{phase}.jsonl"
    manifest.write_text("".join(job.model_dump_json() + "\n" for job in jobs))
    config = fleet.RunConfig(
        out=str(root / phase),
        workers=workers,
        batch_size=4,
        precision="fp16_res32",
        episodes_per_robot_day=BLOCK_WIDTH,
    )
    result = fleet.run(config, jobs)
    extra = result.record.extra
    print(
        json.dumps(
            {
                "phase": phase,
                "jobs": len(jobs),
                "chain_ok": result.record.chain_ok,
                "successes": extra.get("successes"),
                "throughput": extra.get("throughput"),
            }
        ),
        flush=True,
    )
    return 0 if result.record.chain_ok is not False else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.adaptation")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="run a phase through the engine (heavy)")
    run_cmd.add_argument("phase", choices=sorted(PHASES))
    run_cmd.add_argument("--workers", type=int, default=8)
    run_cmd.add_argument("--root", type=Path, default=RUN_ROOT)
    ana = sub.add_parser("analyse", help="print the phases' table")
    ana.add_argument("phases", nargs="+", choices=sorted(PHASES))
    ana.add_argument("--root", type=Path, default=RUN_ROOT)
    ana.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_phase(args.phase, args.workers, args.root)
    runs = {phase: read_run(args.root / phase) for phase in args.phases}
    results = analyse(runs)
    print(markdown(results, runs))
    if args.json:
        args.json.write_text(json.dumps([r.__dict__ | {"rate": r.rate} for r in results], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
