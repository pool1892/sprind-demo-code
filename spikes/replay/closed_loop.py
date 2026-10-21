"""Question 6 and 7: full episodes with SmolVLA in the loop, recorded step by step.

    python closed_loop.py --out NAME --tasks 0,1 --inits 0,1 [--repeats 2]
        [--batching alone|together] [--amp fp32|bf16|fp16] [--device cuda]
        [--order forward|reverse] [--noise-offset 0] [--save-obs]

Episodes are (task, init) pairs; the scene seed and the noise seed are derived
from them, so the same pair means the same episode in every run. Compare the
outputs with compare_runs.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import common as C
import engine as E
from policy_lib import Policy


def specs_for(tasks, inits, noise_offset=0, suite="libero_spatial"):
    # The slot (the episode's row in a slotted inference batch) depends only on the episode:
    # even tasks use rows 0-4, odd tasks rows 5-9, one row per initial state.
    return [
        E.Spec(
            task=t,
            init=i,
            seed=1000 + 100 * t + i,
            noise_seed=noise_offset + 100 * t + i,
            suite=suite,
            slot=5 * (t % 2) + i % 5,
        )
        for t in tasks
        for i in inits
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", default="0")
    p.add_argument("--inits", default="0")
    p.add_argument("--pairs", default=None, help="explicit episodes as task:init,task:init,...")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--batching", default="alone")
    p.add_argument("--amp", default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--order", default="forward")
    p.add_argument("--noise-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--group", type=int, default=0, help="episodes per rollout call; 0 = all at once")
    p.add_argument("--save-obs", action="store_true", help="keep the observations at each inference")
    p.add_argument("--one-env", action="store_true", help="close simulators after each group (saves memory)")
    args = p.parse_args()

    if args.pairs:
        # explicit episodes "task:init,task:init,...", in this order
        specs = []
        for item in args.pairs.split(","):
            t, i = (int(x) for x in item.split(":"))
            specs += specs_for([t], [i], args.noise_offset, args.suite)
    else:
        tasks = [int(x) for x in args.tasks.split(",")]
        inits = [int(x) for x in args.inits.split(",")]
        specs = specs_for(tasks, inits, args.noise_offset, args.suite)
    if args.batching.startswith("slots") and int(args.batching.removeprefix("slots")) == 5:
        for s in specs:
            s.slot = s.init % 5  # batches of five rows: the row is the initial-state index
    if args.order == "reverse":
        specs = specs[::-1]

    t0 = time.time()
    policy = Policy(device=args.device, suite=args.suite)
    print(f"policy loaded in {time.time() - t0:.1f} s", flush=True)
    pool = E.EnvPool()
    out = {}
    for rep in range(args.repeats):
        t1 = time.time()
        runs = []
        step = args.group or len(specs)
        for k in range(0, len(specs), step):
            runs += E.rollout(
                policy,
                specs[k : k + step],
                pool,
                batching=args.batching,
                amp=args.amp,
                max_steps=args.max_steps,
                keep_infer_obs=args.save_obs and rep == 0,
            )
            if args.one_env:
                pool.close()
        if args.order == "reverse":
            runs = runs[::-1]
        out.update(E.pack(runs, prefix=f"r{rep}_"))
        if args.save_obs and rep == 0:
            obs = [o for r in runs for o in r.infer_obs]
            tasks_txt = [r.env.task_description for r in runs for _ in r.infer_obs]
            out["obs_image"] = np.stack([o["pixels"]["image"] for o in obs])
            out["obs_image2"] = np.stack([o["pixels"]["image2"] for o in obs])
            out["obs_eef_pos"] = np.stack([o["robot_state"]["eef"]["pos"] for o in obs])
            out["obs_eef_quat"] = np.stack([o["robot_state"]["eef"]["quat"] for o in obs])
            out["obs_eef_mat"] = np.stack([o["robot_state"]["eef"]["mat"] for o in obs])
            out["obs_gripper_qpos"] = np.stack([o["robot_state"]["gripper"]["qpos"] for o in obs])
            out["obs_gripper_qvel"] = np.stack([o["robot_state"]["gripper"]["qvel"] for o in obs])
            out["obs_joint_pos"] = np.stack([o["robot_state"]["joints"]["pos"] for o in obs])
            out["obs_joint_vel"] = np.stack([o["robot_state"]["joints"]["vel"] for o in obs])
            out["obs_task"] = np.array(tasks_txt)
        succ = [int(r.success) for r in runs]
        steps = [r.t for r in runs]
        print(
            f"repeat {rep}: {sum(succ)}/{len(succ)} successful, steps {steps}, "
            f"{time.time() - t1:.1f} s, outcomes {''.join(map(str, succ))}",
            flush=True,
        )
    out["n_episodes"] = np.array(len(specs))
    out["n_repeats"] = np.array(args.repeats)
    C.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.OUT / f"{args.out}.npz", **out)
    pool.close()
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
