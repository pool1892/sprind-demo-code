"""Simulator speed without a policy: one LIBERO environment, fixed small actions.

Used to compare MuJoCo versions and camera resolutions on equal terms. Prints
one JSON line: resets per second and environment steps per second (physics plus
rendering of both cameras, as in an evaluation). It reports wall-clock time and
the CPU time of this process; on a busy machine the CPU time is the steadier of
the two.

    python bench_sim.py [--suite libero_spatial] [--task 0] [--size 360] [--steps 200] [--resets 3]
"""

from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np
import robosuite

from lerobot.envs.libero import LiberoEnv, _get_suite

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--task", type=int, default=0)
ap.add_argument("--size", type=int, default=360)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--resets", type=int, default=3)
args = ap.parse_args()

suite = _get_suite(args.suite)
env = LiberoEnv(
    task_suite=suite,
    task_id=args.task,
    task_suite_name=args.suite,
    obs_type="pixels_agent_pos",
    observation_width=args.size,
    observation_height=args.size,
)
env.reset(seed=0)  # first reset builds the simulation; not timed

rng = np.random.default_rng(0)
actions = (rng.uniform(-0.2, 0.2, size=(args.steps, 7))).astype(np.float32)
actions[:, 6] = -1.0

reset_s, step_s, step_cpu, n_steps = [], 0.0, 0.0, 0
for r in range(args.resets):
    t = time.perf_counter()
    env.reset(seed=r)
    reset_s.append(time.perf_counter() - t)
    t, c = time.perf_counter(), time.process_time()
    for a in actions:
        _, _, terminated, _, _ = env.step(a)
        n_steps += 1
        if terminated:
            break
    step_s += time.perf_counter() - t
    step_cpu += time.process_time() - c
env.close()
print(
    "BENCH "
    + json.dumps(
        {
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
            "suite": args.suite,
            "task": args.task,
            "size": args.size,
            "steps": n_steps,
            "steps_per_s": round(n_steps / step_s, 1),
            "ms_per_step": round(1000 * step_s / n_steps, 2),
            "cpu_ms_per_step": round(1000 * step_cpu / n_steps, 2),
            "reset_s_mean": round(float(np.mean(reset_s)), 2),
        }
    )
)
