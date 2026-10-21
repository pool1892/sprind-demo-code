"""Build the episode lists ("manifests") the fleet runner executes. No simulator needed.

    python manifests.py spatial100 OUT.jsonl   # 10 tasks x start states 0 to 9, no fault: the success-rate sample
    python manifests.py healthy200 OUT.jsonl   # the walking skeleton's 200 healthy episodes
    python manifests.py mix200 OUT.jsonl       # 40 healthy, 80 with the camera fault, 80 with the action delay
    python manifests.py smoke OUT.jsonl        # 24 episodes for a first test of the runner

For the paired comparison of two number formats (README section 8), each kind run
once per format:

    python manifests.py object200 OUT.jsonl    # libero_object, 10 tasks x start states 0 to 19, no fault
    python manifests.py cam6_a OUT.jsonl       # libero_spatial, camera turned 6 degrees, start states 0 to 19
    python manifests.py cam6_b OUT.jsonl       # the same, start states 20 to 39
    python manifests.py delay17 OUT.jsonl      # actions 17 steps late, start states 0 to 19
    python manifests.py cam3 OUT.jsonl         # camera turned 3 degrees, start states 0 to 19
    python manifests.py cam9 OUT.jsonl         # camera turned 9 degrees, start states 0 to 19

Adapted from spikes/skeleton/manifests.py. The identity function is the
skeleton's without the row: scene seed and noise seed are a pure function of
(robot, task, start state). The batch size and the inference settings are added
by the runner, because they are settings of a run.

The two faults and their levels are the skeleton's: robot 2's scene camera turned
by 6 degrees (24 of 40 there), robot 3's actions executed 17 control steps late
(28 of 40 there).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = range(10)
CAM = [{"kind": "scene_camera_pan", "degrees": 6}]
DELAY = [{"kind": "action_delay", "steps": 17}]


def identity(robot: int, task: int, init: int, suite: str = "libero_spatial") -> dict:
    return {
        "robot": f"robot_{robot}",
        "suite": suite,
        "task": task,
        "init": init,
        # every robot stands in its own cell: its own fixture placement and its own policy noise
        "scene_seed": 1000 + 10000 * robot + 100 * task + init,
        "noise_seed": 100000 * robot + 100 * task + init,
        "max_steps": 280,
    }


def episode(prefix: str, robot: int, task: int, init: int, faults: list, variant: str) -> dict:
    return {"episode_id": f"{prefix}{robot}-t{task}-i{init}", "variant": variant, **identity(robot, task, init),
            "faults": faults}


def spatial100() -> list[dict]:
    return [episode("s", 1, t, i, [], "healthy") for t in TASKS for i in range(10)]


def healthy200() -> list[dict]:
    """The three robots' 40 each, and 80 more of robot 1: the same 200 as the skeleton's healthy batch."""
    return [episode("h", r, t, i, [], "healthy") for t in TASKS for i in range(12) for r in (1, 2, 3)
            if i < 4 or r == 1]


def mix200() -> list[dict]:
    jobs = []
    for t in TASKS:
        for i in range(8):
            if i < 4:
                jobs.append(episode("m", 1, t, i, [], "healthy"))
            jobs.append(episode("m", 2, t, i, CAM, "cam"))
            jobs.append(episode("m", 3, t, i, DELAY, "delay"))
    return jobs


def object200() -> list[dict]:
    return [{"episode_id": f"o1-t{t}-i{i}", "variant": "healthy", **identity(1, t, i, "libero_object"), "faults": []}
            for t in TASKS for i in range(20)]


def camera(degrees: float, inits: range) -> list[dict]:
    """Robot 2's episodes, as in the skeleton: the same scene and noise at every level of the fault."""
    faults = [{"kind": "scene_camera_pan", "degrees": degrees}]
    return [{"episode_id": f"c{degrees:g}-t{t}-i{i}", "variant": f"cam{degrees:g}", **identity(2, t, i), "faults": faults}
            for t in TASKS for i in inits]


def delay17() -> list[dict]:
    return [{"episode_id": f"d17-t{t}-i{i}", "variant": "delay", **identity(3, t, i), "faults": DELAY}
            for t in TASKS for i in range(20)]


def smoke() -> list[dict]:
    jobs = []
    for t, i in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)):
        jobs.append(episode("k", 1, t, i, [], "healthy"))
        jobs.append(episode("k", 2, t, i, CAM, "cam"))
        jobs.append(episode("k", 3, t, i, DELAY, "delay"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    kinds = {
        "spatial100": spatial100, "healthy200": healthy200, "mix200": mix200, "smoke": smoke,
        "object200": object200, "delay17": delay17,
        "cam6_a": lambda: camera(6, range(20)), "cam6_b": lambda: camera(6, range(20, 40)),
        "cam3": lambda: camera(3, range(20)), "cam9": lambda: camera(9, range(20)),
    }
    ap.add_argument("kind", choices=sorted(kinds))
    ap.add_argument("out")
    args = ap.parse_args()
    jobs = kinds[args.kind]()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(j) + "\n" for j in jobs))
    by = {}
    for j in jobs:
        by[j["variant"]] = by.get(j["variant"], 0) + 1
    print(f"{args.out}: {len(jobs)} episodes, {by}")


if __name__ == "__main__":
    main()
