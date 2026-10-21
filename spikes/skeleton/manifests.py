"""Build the episode lists ("manifests") the fleet runner executes. No simulator needed.

    python manifests.py dose   OUT.jsonl --robot 2 --kind scene_camera_pan --param degrees --levels 0 1 2 4 8
    python manifests.py fleet  OUT.jsonl --cam-degrees 4 --delay-steps 10
    python manifests.py replay OUT.jsonl --from RUN_DIR [--keep-faults-every 5]
    python manifests.py healthy OUT.jsonl            # 200 episodes without any fault
    python manifests.py smoke  OUT.jsonl             # 24 episodes for a first test of the runner

An episode's identity is everything in a manifest line except `episode_id` and
`variant`: suite, task, start state, scene seed, noise seed, the size of the
inference batch and the episode's row in it, the step limit and the fault set.
Two lines with the same identity are the same episode, whichever run, worker or
order they appear in.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROWS = 8
TASKS = range(10)
STARTS = range(4)


def identity(robot: int, task: int, init: int) -> dict:
    return {
        "robot": f"robot_{robot}",
        "suite": "libero_spatial",
        "task": task,
        "init": init,
        # every robot stands in its own cell: its own fixture placement and its own policy noise
        "scene_seed": 1000 + 10000 * robot + 100 * task + init,
        "noise_seed": 100000 * robot + 100 * task + init,
        "rows": ROWS,
        "row": (4 * task + init) % ROWS,
        "max_steps": 280,
    }


def write(path: str, jobs: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(j) + "\n" for j in jobs))
    per_row = [sum(j["row"] == r for j in jobs) for r in range(ROWS)]
    print(f"{path}: {len(jobs)} episodes, per row {per_row}")


def dose(args) -> list[dict]:
    jobs = []
    for task in TASKS:
        for init in STARTS:
            for level in args.levels:
                level = int(level) if float(level).is_integer() else float(level)
                faults = [] if level == 0 else [{"kind": args.kind, args.param: level}]
                jobs.append({
                    "episode_id": f"d{args.robot}-{args.param}{level}-t{task}-i{init}",
                    "variant": f"{args.param}={level}", **identity(args.robot, task, init), "faults": faults,
                })
    return jobs


def fleet(args) -> list[dict]:
    fault_of = {
        1: [],
        2: [{"kind": "scene_camera_pan", "degrees": args.cam_degrees}],
        3: [{"kind": "action_delay", "steps": args.delay_steps}],
    }
    # Neutral episode numbers: each robot works through the 40 episodes in its own shuffled
    # order, and the numbers follow the fleet's time order (robots take turns).
    rng = random.Random(20261006)
    order = {}
    for robot in (1, 2, 3):
        todo = [(t, i) for t in TASKS for i in STARTS]
        rng.shuffle(todo)
        for seq, (t, i) in enumerate(todo):
            order[(robot, t, i)] = seq * 3 + (robot - 1)
    jobs = []
    for task in TASKS:
        for init in STARTS:
            for robot in (1, 2, 3):
                jobs.append({
                    "episode_id": f"e{order[(robot, task, init)] + 1:04d}", "variant": "as_run",
                    **identity(robot, task, init), "faults": fault_of[robot],
                })
    return jobs


def replay(args) -> list[dict]:
    """Every episode of a run again with its fault set emptied, read from the run's truth records.

    With --keep-faults-every N, every Nth faulted episode is also replayed with its
    fault left in (id suffix "+f"): a replay with nothing removed has to reproduce
    the original, and it makes the workers switch faults on and off between the
    fault-free replays. The order is the reverse of the original run's.
    """
    keys = ("robot", "suite", "task", "init", "scene_seed", "noise_seed", "rows", "row", "max_steps")
    src = [json.loads(line) for line in (Path(args.source) / "truth" / "episodes.jsonl").read_text().splitlines()]
    jobs, n_faulted = [], 0
    for e in reversed(src):
        ident = {k: e[k] for k in keys}
        if e["faults"]:
            n_faulted += 1
            if args.keep_faults_every and n_faulted % args.keep_faults_every == 0:
                jobs.append({"episode_id": e["episode_id"] + "+f", "variant": "replay_as_run", **ident,
                             "faults": e["faults"]})
        jobs.append({"episode_id": e["episode_id"], "variant": "replay_fault_removed", **ident, "faults": []})
    return jobs


def healthy(args) -> list[dict]:
    """200 episodes with no fault anywhere: the three robots' 40 each, and 80 more of robot 1."""
    jobs = []
    for task in TASKS:
        for init in range(12):
            for robot in (1, 2, 3):
                if init >= 4 and robot != 1:
                    continue
                jobs.append({"episode_id": f"h{robot}-t{task}-i{init}", "variant": "healthy",
                             **identity(robot, task, init), "faults": []})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dose")
    p.add_argument("out")
    p.add_argument("--robot", type=int, required=True)
    p.add_argument("--kind", required=True)
    p.add_argument("--param", required=True)
    p.add_argument("--levels", type=float, nargs="+", required=True)
    p = sub.add_parser("fleet")
    p.add_argument("out")
    p.add_argument("--cam-degrees", type=float, required=True)
    p.add_argument("--delay-steps", type=int, required=True)
    p = sub.add_parser("replay")
    p.add_argument("out")
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--keep-faults-every", type=int, default=0)
    p = sub.add_parser("healthy")
    p.add_argument("out")
    p = sub.add_parser("smoke")
    p.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "smoke":
        jobs = []
        for task, init in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)):
            jobs.append({"episode_id": f"s1-t{task}-i{init}", "variant": "healthy", **identity(1, task, init), "faults": []})
            jobs.append({"episode_id": f"s2-t{task}-i{init}", "variant": "cam", **identity(2, task, init),
                         "faults": [{"kind": "scene_camera_pan", "degrees": 4}]})
            jobs.append({"episode_id": f"s3-t{task}-i{init}", "variant": "delay", **identity(3, task, init),
                         "faults": [{"kind": "action_delay", "steps": 10}]})
    else:
        jobs = {"dose": dose, "fleet": fleet, "replay": replay, "healthy": healthy}[args.cmd](args)
    write(args.out, jobs)


if __name__ == "__main__":
    main()
