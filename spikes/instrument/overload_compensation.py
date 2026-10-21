"""Does latency compensation also rescue an overloaded controller? (lane P, an open question)

    # write and check the manifest only (light, no policy)
    uv run --locked python spikes/instrument/overload_compensation.py plan OUT
    # one small heavy job: 180 X-VLA attempts, four workers
    tools/heavy-run uv run --locked python spikes/instrument/overload_compensation.py run OUT
    # the table (light): each cell, paired with the overload's default and with the
    # adaptation spike's 17-step link on the same starts
    uv run --locked python spikes/instrument/overload_compensation.py analyse OUT

Story B's decision beat is that the remote-link delay gets compensation and the robot whose
controller falls behind gets service. That holds only if compensation does not also rescue
the overloaded robot. The cells repeat the adaptation spike's design
(`fleetlab.adaptation`): libero_spatial, 60 paired attempts, attempt i is task i % 10 at
start state i // 10 with the calibration world's seeds (robot r01, day 1, index i), so
attempt i here is the same start as attempt i of the adaptation cells. The fault is
`controller_overload` at 0.2 on r01's controller (the kept level, -42 points at 100 pairs);
the setting is applied as the loop applies an adopted action (`set
control.latency_compensation_steps = c at s1`, resolved against the identity). Values: 0
(the default), 21 (what an agent would set if it read the overload's alignment lag of 24
as a delay plus the controller's 3) and 10 (half of it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LEVEL = 0.2
COMPENSATIONS = (0, 21, 10)
EPISODES = 60
SUITE = "libero_spatial"
FIRST_BLOCK = 3900  # above the adaptation phases' blocks (3000 on, 100 per phase)
ADAPT_RUNS = Path("/var/tmp/robots-loop/adapt/runs")


def cells() -> list[Any]:
    from fleetlab.adaptation import COMPENSATION, Cell

    return [Cell(SUITE, 0, ((COMPENSATION, c),) if c else (), EPISODES) for c in COMPENSATIONS]


def identity(cell: Any, block: int, index: int, policy: Any, software: Any) -> Any:
    """`fleetlab.adaptation.identity` with the overloaded controller instead of the link."""
    import fleetlab.faults as faults
    from fleetlab.adaptation import NOMINAL, setting_actions
    from fleetlab.contracts.actions import resolve_all
    from fleetlab.contracts.faults import FaultSpec, Schedule, Target
    from fleetlab.contracts.identity import EpisodeIdentity, seeds_for
    from fleetlab.contracts.ids import IdKind, neutral_id
    from fleetlab.contracts.vocabulary import Component
    from fleetlab.dose_response import (
        BLOCK_WIDTH,
        DAY,
        IMAGE_SIZE,
        MAX_STEPS,
        ROBOT_ID,
        SITE_ID,
        WORLD_ID,
        WORLD_SEED,
    )

    scene, noise = seeds_for(WORLD_SEED, ROBOT_ID, DAY, index)
    overload = FaultSpec(
        fault_id="f01",
        kind="controller_overload",
        level=LEVEL,
        target=Target(scope=ROBOT_ID, component=Component.controller),
        schedule=Schedule(onset_day=0),
    )
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
        batch_size=4,
        row=None,
        image_size=IMAGE_SIZE,
        max_steps=MAX_STEPS,
        faults=[overload],
        software=software,
    )
    resolution = resolve_all(
        setting_actions(cell), base, base.faults, faults.truth_table(), NOMINAL
    )
    if resolution.removed:
        raise AssertionError(f"a setting removed {sorted(resolution.removed)}")
    return base.model_copy(update={"faults": [*base.faults, *resolution.added]})


def plan(policy: Any, software: Any) -> list[Any]:
    from fleetlab.contracts.rollout import Job

    return [
        Job(identity=identity(cell, FIRST_BLOCK + number, i, policy, software))
        for number, cell in enumerate(cells())
        for i in range(cell.episodes)
    ]


def run(out: Path, workers: int) -> int:
    import os

    from fleetlab.dose_response import BLOCK_WIDTH, _engine, _nominal_policy

    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    miopen = _engine("fleetlab.engine.miopen")
    fleet = _engine("fleetlab.engine.fleet")
    miopen.activate()
    software = _engine("fleetlab.engine.software").current(miopen.require(4, "fp16_res32"))
    jobs = plan(_nominal_policy("fp16_res32"), software)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    config = fleet.RunConfig(
        out=str(out / "run"),
        workers=workers,
        batch_size=4,
        precision="fp16_res32",
        episodes_per_robot_day=BLOCK_WIDTH,
    )
    result = fleet.run(config, jobs)
    print(json.dumps({"jobs": len(jobs), "chain_ok": result.record.chain_ok}), flush=True)
    return 0 if result.record.chain_ok is not False else 1


def check_plan(out: Path) -> int:
    """The manifest without a policy: every identity validates, the setting resolves."""
    from fleetlab.dose_response import _nominal_policy
    from fleetlab.engine import software as soft

    jobs = plan(_nominal_policy("fp16_res32"), soft.current("none"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest-check.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    kinds = sorted({tuple(f.kind for f in j.identity.faults) for j in jobs})
    print(json.dumps({"jobs": len(jobs), "fault_sets": kinds}), flush=True)
    return 0


def analyse(out: Path) -> int:
    from fleetlab.adaptation import cell_of
    from fleetlab.contracts.ids import IdKind, id_number
    from fleetlab.dose_response import BLOCK_WIDTH, mcnemar_exact, read_run, wilson

    ours = read_run(out / "run")
    by_cell: dict[int, dict[int, bool]] = {}
    for eid, outcome in ours.outcomes.items():
        block, index = divmod(id_number(IdKind.episode, eid), BLOCK_WIDTH)
        by_cell.setdefault(block - FIRST_BLOCK, {})[index] = outcome.true_success
    # the adaptation spike's 17-step link without compensation, on the same starts
    link: dict[int, bool] = {}
    p1 = read_run(ADAPT_RUNS / "p1")
    for eid, outcome in p1.outcomes.items():
        cell, index = cell_of("p1", eid)
        if cell.suite == SUITE and cell.delay == 17 and cell.is_default:
            link[index] = outcome.true_success
    rows = []
    default = by_cell.get(0, {})
    for number, compensation in enumerate(COMPENSATIONS):
        got = by_cell.get(number, {})
        n, k = len(got), sum(got.values())
        low, high = wilson(k, n) if n else (float("nan"), float("nan"))
        pairs = [i for i in got if i in default]
        rescued = sum(got[i] and not default[i] for i in pairs)
        broken = sum(default[i] and not got[i] for i in pairs)
        rows.append(
            {
                "compensation": compensation,
                "attempts": n,
                "successes": k,
                "wilson_95": [round(low, 3), round(high, 3)],
                "vs_default_rescued_broken": [rescued, broken],
                "vs_default_mcnemar_p": mcnemar_exact(rescued, broken) if number else None,
            }
        )
    both = [i for i in default if i in link]
    summary = {
        "level": LEVEL,
        "chain_ok": ours.chain_ok,
        "cells": rows,
        "link17_default_same_starts": [sum(link[i] for i in both), len(both)],
        "overload_default_same_starts": [sum(default[i] for i in both), len(both)],
    }
    print(json.dumps(summary, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run", "analyse"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)

    if args.command == "plan":
        return check_plan(args.out)
    if args.command == "run":
        return run(args.out, args.workers)
    return analyse(args.out)


if __name__ == "__main__":
    sys.exit(main())
