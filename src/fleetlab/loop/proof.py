"""The state lane's path on a stored world in one command, for a small real run:

    tools/heavy-run uv run --locked python -m fleetlab.loop.proof \\
        --world ~/.cache/robots-probes/mini/w10 --loop /var/tmp/robots-loop/state/proof-w10 \\
        --adopt recalibrate:r04:scene_camera --adopt revert:s2:control.action_delay_steps \\
        --arm-robot r05 --arm revert:s2:control.action_delay_steps --per-arm 3 \\
        --panel-per-robot-day 3 --reproduce-day 3 --workers 6

1. Round 1's export under the original state (the world's observed run), compared with
   the world's own stored export where there is one.
2. The ledger of round 1's decision (the `--adopt` actions, on round 1's adoption day).
3. Round 2's operation under it (`operate`), and round 2's export (`export_round`).
4. One experiment of round 2 through the round's engine (`RoundEngine`): a control arm,
   which is the current state, and the `--arm` actions as a delta on it.
5. The reserved evaluation of the states `original` and `round1` on the panel
   (`evaluate`), and with `--reproduce-day` state `original` on as-run identities of that
   stored day, whose outcomes must equal the world's own.

Writes `report.json` in the loop directory. No model is called; the decision is given.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main", "parse_action"]


def parse_action(text: str) -> Any:
    """`verb:scope[:thing[:value]]`, the thing a component, configuration key or condition,
    the value JSON (`set:r03:policy.actions_executed_per_chunk:10`)."""
    from fleetlab.contracts.actions import Action
    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb

    parts = text.split(":", 3)
    verb, scope = Verb(parts[0]), parts[1]
    target: dict[str, Any] = {"scope": scope}
    if len(parts) > 2:
        thing = parts[2]
        if thing in {c.value for c in Component}:
            target["component"] = Component(thing)
        elif thing in {k.value for k in ConfigKey}:
            target["config_key"] = ConfigKey(thing)
        else:
            target["condition_key"] = ConditionKey(thing)
    value = json.loads(parts[3]) if len(parts) > 3 else None
    return Action(verb=verb, target=Target(**target), value=value)


def _episodes(export: Path) -> list[dict[str, Any]]:
    from fleetlab.log import LogReader

    with LogReader(export) as reader:
        rows: list[dict[str, Any]] = reader.query(
            "select episode_id, robot_id, day, experiment_id, outcome from episodes"
        ).to_pylist()
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.agents.audit import tree_manifest
    from fleetlab.agents.sessions import ExperimentQueue, QueuedExperiment
    from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
    from fleetlab.engine.fleet import RunConfig
    from fleetlab.loop.evaluate import evaluate, reproduces_as_run
    from fleetlab.loop.experiments import RoundEngine
    from fleetlab.loop.export import export_round
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.operate import EngineSetup, operate
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.state import Ledger
    from fleetlab.loop.world import LoopWorld

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.proof")
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--loop", type=Path, required=True)
    parser.add_argument("--adopt", action="append", default=[])
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--arm-robot", action="append", default=[])
    parser.add_argument("--per-arm", type=int, default=3)
    parser.add_argument("--panel-per-robot-day", type=int, default=3)
    parser.add_argument("--reproduce-day", type=int, default=None)
    parser.add_argument("--reproduce-per-robot-day", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-contacts", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    timings: dict[str, float] = {}
    report: dict[str, Any] = {"started": datetime.datetime.now().astimezone().isoformat()}
    world = LoopWorld.load(args.world)
    layout = layout_for_world(world.spec)
    paths = LoopPaths(args.loop.expanduser().resolve())
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.layout.is_file() and paths.layout.read_text() != layout.model_dump_json(indent=1):
        raise ValueError(f"{paths.layout} holds another layout")
    paths.layout.write_text(layout.model_dump_json(indent=1))
    config = RunConfig(out="", batch_size=4, workers=args.workers, precision="fp16_res32")
    setup = EngineSetup(config=config, contacts=not args.no_contacts)
    report.update(world=str(world.root), world_id=world.world_id, layout=layout.model_dump())

    def mark(name: str, since: float) -> None:
        timings[name] = round(time.time() - since, 1)

    # 1. Round 1's export, under the original state.
    t = time.time()
    l0 = Ledger.empty(world.world_id)
    l0.save(paths.ledgers)
    first = export_round(world, layout, 1, l0, paths, paths.exports(1) / "start")
    stored = world.root / "export"
    report["round1_export"] = {
        "episodes": len(_episodes(first)),
        "equals_stored_export": (
            tree_manifest(first) == tree_manifest(stored) if stored.is_dir() else None
        ),
    }
    mark("round1_export", t)

    # 2. The decision of round 1, retained.
    l1 = l0.adopt(1, layout.adoption_day(1), adopt=[parse_action(a) for a in args.adopt])
    l1.save(paths.ledgers)
    report["ledger_r1"] = json.loads(l1.model_dump_json())

    # 3. Round 2's operation under the ledger, and round 2's export.
    t = time.time()
    run2 = operate(world, layout, 2, l1, paths, setup)
    mark("round2_operation", t)
    t = time.time()
    second = export_round(world, layout, 2, l1, paths, paths.exports(2) / "start")
    rows = _episodes(second)
    by_day: dict[int, list[int]] = {}
    for row in rows:
        counts = by_day.setdefault(row["day"], [0, 0])
        counts[0] += 1
        counts[1] += row["outcome"] == "success"
    report["round2_operation"] = {"run": str(run2)}
    report["round2_export"] = {
        "episodes_by_day": {
            d: {"attempts": n, "successes": s} for d, (n, s) in sorted(by_day.items())
        }
    }
    mark("round2_export", t)

    # 4. One experiment of round 2, from the current state.
    if args.arm_robot:
        t = time.time()
        request = ExperimentRequest(
            arms=[
                ExperimentArm(actions=[]),
                ExperimentArm(actions=[parse_action(a) for a in args.arm]),
            ],
            robots=args.arm_robot,
            episodes_per_arm=args.per_arm,
            paired_starts=True,
            power_note="state lane proof: control is the current state, the treatment a delta",
        )
        analysis = f"{world.world_id}-proof-r2"
        queue_file = paths.round(2) / "queue-session-01.json"
        if queue_file.is_file():  # a resumed proof answers the same queue again
            queue = ExperimentQueue.model_validate_json(queue_file.read_text())
            if queue.experiments[0].request != request or queue.export != str(second):
                raise ValueError(f"{queue_file} holds another experiment")
        else:
            queue = ExperimentQueue(
                world_id=world.world_id,
                analysis_id=analysis,
                session=1,
                created_at=datetime.datetime.now().astimezone().isoformat(),
                export=str(second),
                experiments=[QueuedExperiment(experiment_id="x0001", session=1, request=request)],
            )
            queue_file.write_text(queue.model_dump_json(indent=1))
        engine = RoundEngine(world, layout, 2, l1, paths, setup)
        reply = engine.run_queue(queue_file, queue)
        latest = {r["episode_id"]: r for r in _episodes(reply.export)}
        arms = []
        for result in reply.results:
            for arm in result.arms:
                outcomes = [latest[e]["outcome"] == "success" for e in arm.episode_ids]
                arms.append(
                    {"arm": arm.arm_index, "attempts": len(outcomes), "successes": sum(outcomes)}
                )
        report["round2_experiment"] = {
            "status": [r.status for r in reply.results],
            "messages": [r.message for r in reply.results],
            "arms": arms,
            "export": str(reply.export),
        }
        mark("round2_experiment", t)

    # 5. The reserved evaluation, and the reproduction of the as-run outcomes.
    t = time.time()
    summary = evaluate(
        world,
        layout,
        {"original": l0, "round1": l1},
        paths,
        setup,
        per_robot_day=args.panel_per_robot_day,
    )
    report["evaluation"] = json.loads(summary.model_dump_json())
    mark("evaluation", t)
    if args.reproduce_day is not None:
        t = time.time()
        inside = LoopPaths(paths.root / "reproduce")
        evaluate(
            world,
            layout,
            {"original": l0},
            inside,
            setup,
            days=[args.reproduce_day],
            per_robot_day=args.reproduce_per_robot_day,
        )
        report["reproduction"] = reproduces_as_run(world, inside.evaluation)
        mark("reproduction", t)
    timings["total"] = round(time.time() - started, 1)
    report["seconds"] = timings
    (paths.root / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
