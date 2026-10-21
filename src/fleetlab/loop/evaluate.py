"""The reserved evaluation (docs/design/loop.md, section 6).

    summary = evaluate(world, layout, {"original": L0, "round1": L1, "round2": L2,
                                       "one_look": plan}, paths, setup)

The panel's identities are minted once, deterministically, from the world seed, on the
layout's evaluation days (`LoopWorld.panel_identities`), and written to
`evaluation/panel.json` before any state runs; they are never exported. Every state is
the same identities with each attempt's faults materialized under that state's ledger
on the attempt's day (`LoopWorld.state_on`), so the states differ only by what their
ledgers change; state `original` is the empty ledger. Each state is one engine run in
`evaluation/<state>/`.

The output, all on the truth side:

- `evaluation/<state>/` the engine's run (episodes, truth with the faults in force);
- `evaluation/attempts.parquet`: one row per state and attempt: state, episode id,
  robot, site, day, task, start state, success, steps;
- `evaluation/summary.json` (`EvaluationSummary`): success per state with a 95% Wilson
  interval, and for every pair of states the attempts rescued (failure under the first,
  success under the second), broken (the reverse) and the net, with McNemar's exact
  two-sided p on the discordant pairs.

A panel day inside the stored world (a test fixture, or a world without days to spare)
reuses the world's own as-run identities, so state `original` must reproduce the as-run
outcomes there; `reproduces_as_run` checks it.

Video: `video=True` records both cameras for every attempt; otherwise only the attempts
whose index within their robot and day is in `video_indices` (default: the setup's
`panel_video_indices`, M3's `(0,)`) record them, the same identities in every state, so
that each recorded attempt is a before-and-after pair (`fleetlab.loop.clips` chooses
among them). Recording changes no outcome: the chain is the same with or without video.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from fleetlab.contracts._base import ContractModel
from fleetlab.loop.layout import LoopLayout, days_of
from fleetlab.loop.operate import EngineSetup
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.state import Ledger
from fleetlab.loop.world import LoopWorld, _with

__all__ = [
    "EvaluationSummary",
    "PairResult",
    "StateResult",
    "evaluate",
    "mcnemar_exact",
    "panel_jobs",
    "reproduces_as_run",
    "summarize",
    "wilson",
]


class StateResult(ContractModel):
    """One state on the panel."""

    name: str
    ledger_version: int
    changes: int = Field(ge=0)  # entries in force (`state.changes_in_force`)
    attempts: int = Field(ge=0)
    successes: int = Field(ge=0)
    rate: float
    interval: tuple[float, float]  # 95% Wilson


class PairResult(ContractModel):
    """Two states on the same attempts: what the second rescued and broke."""

    first: str
    second: str
    rescued: int = Field(ge=0)
    broken: int = Field(ge=0)
    net: int
    p_mcnemar: float


class EvaluationSummary(ContractModel):
    world_id: str
    days: list[int]
    attempts: int
    panel_sha256: str
    states: list[StateResult]
    pairs: list[PairResult]


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval of a proportion (95% by default); (0, 1) for no attempt."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(rescued: int, broken: int) -> float:
    """McNemar's exact two-sided p: the discordant pairs against a fair coin."""
    n = rescued + broken
    if n == 0:
        return 1.0
    tail: float = sum(math.comb(n, i) for i in range(min(rescued, broken) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


STATE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")


def panel_jobs(
    world: LoopWorld,
    panel: Sequence[Any],
    ledger: Ledger,
    software: Any,
    filmed: bool | Collection[int] = False,
) -> list[Any]:
    """The run jobs of one state on the panel: each identity with its faults materialized
    under `ledger` on its day; `filmed` is True (every attempt records its videos) or the
    indices within a robot and day whose attempts do."""
    from fleetlab.contracts.rollout import Job

    faults: dict[tuple[str, int], list[Any]] = {}
    jobs = []
    for identity in panel:
        key = (identity.robot_id, identity.day)
        if key not in faults:
            faults[key] = list(world.state_on(identity.robot_id, identity.day, ledger).faults)
        video = filmed if isinstance(filmed, bool) else identity.episode_index in filmed
        jobs.append(Job(identity=_with(identity, software, faults[key]), record_video=video))
    return jobs


def evaluate(
    world: LoopWorld,
    layout: LoopLayout,
    states: Mapping[str, Ledger],
    paths: LoopPaths,
    setup: EngineSetup,
    *,
    days: list[int] | None = None,
    per_robot_day: int | None = None,
    video: bool = False,
    video_indices: Collection[int] | None = None,
) -> EvaluationSummary:
    """Simulate every state on the reserved panel (the layout's evaluation days unless
    `days` is given) and summarize. Resumable: a state whose run is complete is not run
    again. The panel is minted once and stored first; a later call must mint the same."""
    from fleetlab.engine import fleet
    from fleetlab.loop.operate import _complete

    for name in states:
        if not STATE_NAME.match(name):
            raise ValueError(f"a state's name is lower-case letters, digits, - and _: {name!r}")
    days = days if days is not None else days_of(layout.evaluation)
    late = [
        (name, e.entry_id)
        for name, ledger in states.items()
        for e in ledger.entries
        if e.adopted_day > min(days)
    ]
    if late:
        raise ValueError(f"entries take effect after the evaluation starts: {late}")
    chosen = setup.panel_video_indices if video_indices is None else video_indices
    filmed: bool | frozenset[int] = True if video else frozenset(int(i) for i in chosen)
    root = paths.evaluation
    root.mkdir(parents=True, exist_ok=True)
    software = fleet.prepare(setup.config)
    panel = world.panel_identities(days, software, setup.config.batch_size, per_robot_day)
    text = json.dumps([i.model_dump(mode="json") for i in panel], indent=0) + "\n"
    panel_file = root / "panel.json"
    if panel_file.is_file() and panel_file.read_text() != text:
        raise ValueError(f"{panel_file} holds another panel; the panel is minted once")
    panel_file.write_text(text)
    for name, ledger in states.items():
        stored = root / f"{name}.ledger.json"
        ledger_text = ledger.model_dump_json(indent=1) + "\n"
        if stored.is_file() and stored.read_text() != ledger_text:
            raise ValueError(f"state {name} was evaluated under another ledger")
        stored.write_text(ledger_text)
        jobs = panel_jobs(world, panel, ledger, software, filmed)
        run_dir = root / name
        if not _complete(run_dir, jobs):
            fleet.run(setup.for_run(run_dir, world.truth.episodes_per_robot_day), jobs)
    return summarize(world, paths, states, panel_sha256=_sha(text))


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _outcomes(run_dir: Path) -> dict[str, bool]:
    """The true outcome of every attempt of a run (`truth/job_outcomes`, run jobs)."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        run_dir / "truth" / "job_outcomes.parquet", columns=["episode_id", "mode", "success"]
    ).to_pylist()
    out: dict[str, bool] = {}
    for row in table:
        if row["mode"] == "run":
            out[row["episode_id"]] = bool(row["success"])
    return out


def summarize(
    world: LoopWorld,
    paths: LoopPaths,
    states: Mapping[str, Ledger],
    panel_sha256: str | None = None,
) -> EvaluationSummary:
    """The summary of finished state runs; writes `evaluation/attempts.parquet` and
    `evaluation/summary.json`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.loop.state import changes_in_force

    root = paths.evaluation
    text = (root / "panel.json").read_text()
    panel = [EpisodeIdentity.model_validate(i) for i in json.loads(text)]
    outcomes = {name: _outcomes(root / name) for name in states}
    rows = []
    results = []
    for name, ledger in states.items():
        missing = [i.episode_id for i in panel if i.episode_id not in outcomes[name]]
        if missing:
            raise ValueError(f"state {name} lacks {len(missing)} panel attempts")
        wins = sum(outcomes[name][i.episode_id] for i in panel)
        results.append(
            StateResult(
                name=name,
                ledger_version=ledger.version,
                changes=changes_in_force(ledger),
                attempts=len(panel),
                successes=wins,
                rate=wins / len(panel) if panel else 0.0,
                interval=wilson(wins, len(panel)),
            )
        )
        rows += [
            {
                "state": name,
                "episode_id": i.episode_id,
                "robot_id": i.robot_id,
                "site_id": i.site_id,
                "day": i.day,
                "suite": i.suite,
                "task_index": i.task_index,
                "start_state": i.start_state,
                "success": outcomes[name][i.episode_id],
            }
            for i in panel
        ]
    names = list(states)
    pairs = []
    for a, first in enumerate(names):
        for second in names[a + 1 :]:
            rescued = sum(
                1
                for i in panel
                if not outcomes[first][i.episode_id] and outcomes[second][i.episode_id]
            )
            broken = sum(
                1
                for i in panel
                if outcomes[first][i.episode_id] and not outcomes[second][i.episode_id]
            )
            pairs.append(
                PairResult(
                    first=first,
                    second=second,
                    rescued=rescued,
                    broken=broken,
                    net=rescued - broken,
                    p_mcnemar=mcnemar_exact(rescued, broken),
                )
            )
    summary = EvaluationSummary(
        world_id=world.world_id,
        days=sorted({i.day for i in panel}),
        attempts=len(panel),
        panel_sha256=panel_sha256 or _sha(text),
        states=results,
        pairs=pairs,
    )
    pq.write_table(pa.Table.from_pylist(rows), root / "attempts.parquet", compression="zstd")
    (root / "summary.json").write_text(summary.model_dump_json(indent=1) + "\n")
    return summary


def reproduces_as_run(world: LoopWorld, eval_dir: Path) -> dict[str, int]:
    """For panel attempts that are as-run identities of the world: how many of state
    `original`'s outcomes equal the stored as-run outcome ("same") and how many differ."""
    stored: dict[str, bool] = {}
    for run in world.runs:
        if (run / "truth" / "job_outcomes.parquet").is_file():
            stored.update(_outcomes(run))
    mine = _outcomes(Path(eval_dir) / "original")
    shared = [e for e in mine if e in stored]
    same = sum(1 for e in shared if mine[e] == stored[e])
    return {"compared": len(shared), "same": same, "differ": len(shared) - same}


def main(argv: list[str] | None = None) -> int:
    """Evaluate states on the reserved panel, for a caller under `tools/heavy-run`:

        tools/heavy-run uv run --locked python -m fleetlab.loop.evaluate --world ROOT \\
            --loop DIR --state original=DIR/ledgers/ledger-r0.json \\
            --state round1=DIR/ledgers/ledger-r1.json [--per-robot-day 3] [--workers 6] \\
            [--video | --video-index 0]
    """
    import argparse

    from fleetlab.engine.fleet import RunConfig

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.evaluate")
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--state", action="append", required=True, help="NAME=LEDGER.json")
    parser.add_argument("--per-robot-day", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--video", action="store_true", help="every attempt records video")
    parser.add_argument(
        "--video-index",
        type=int,
        action="append",
        default=[],
        help="only the attempts with this index within their robot and day record video",
    )
    args = parser.parse_args(argv)
    paths = LoopPaths(args.loop)
    world = LoopWorld.load(args.world)
    layout = LoopLayout.model_validate_json(paths.layout.read_text())
    states = {}
    for item in args.state:
        name, _, file = item.partition("=")
        states[name] = Ledger.load(file)
    config = RunConfig(out="", batch_size=4, workers=args.workers, precision="fp16_res32")
    summary = evaluate(
        world,
        layout,
        states,
        paths,
        EngineSetup(config=config, panel_video_indices=tuple(args.video_index)),
        per_robot_day=args.per_robot_day,
        video=args.video,
    )
    print(summary.model_dump_json(indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
