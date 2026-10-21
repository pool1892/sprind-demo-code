"""Question 1 and 2: run a fixed action script in the simulator and record everything.

For each repeat the environment is reset the way lerobot-eval resets it (hard
reset, fixed LIBERO initial state, ten settling steps) and the action script is
played open loop. Recorded per step: MuJoCo's full integration state, the
proprioceptive observation and both camera images.

    python sim_trace.py --out NAME [--actions FILE.npy] [--repeats 3]
                        [--task 0] [--init 0] [--seeds 1000,1000,7]

Compare the files with sim_compare.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import common as C


def run_once(env, actions: np.ndarray, seed, np_seed=False, clear=False):
    if np_seed:
        # What robosuite 1.4 did inside env.seed(): LIBERO's placement samplers
        # draw the fixture positions from numpy's global generator. The wrapper
        # builds the simulator lazily with a reset of its own, which draws from
        # the same generator, so build first and seed afterwards.
        env._ensure_env()
        if clear and C.rs_env_of(env).hard_reset:
            # LIBERO keeps appending joint-setting samplers at every hard reset;
            # each one draws a number before the fixtures are placed.
            C.rs_env_of(env).object_property_initializers.clear()
        np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    states = [C.get_mj_state(env)]
    proprio = [C.flat_proprio(obs)]
    imgs = {k: [obs["pixels"][k].copy()] for k in C.CAMERAS}
    success = []
    for a in actions:
        obs, _, terminated, _, info = env.step(a)
        states.append(C.get_mj_state(env))
        proprio.append(C.flat_proprio(obs))
        for k in C.CAMERAS:
            imgs[k].append(obs["pixels"][k].copy())
        success.append(bool(info["is_success"]))
        if terminated:
            break
    model = C.sim_of(env).model._model
    return (
        np.stack(states),
        np.stack(proprio),
        {k: np.stack(v) for k, v in imgs.items()},
        np.array(success),
        np.concatenate([model.body_pos.ravel(), model.body_quat.ravel()]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--actions", default=None)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init", type=int, default=0)
    p.add_argument("--seeds", default="1000")
    p.add_argument("--soft-reset", action="store_true")
    p.add_argument("--size", type=int, default=360, help="camera resolution; lerobot-eval uses 360")
    p.add_argument("--np-seed", action="store_true", help="seed numpy's global generator before each reset")
    p.add_argument("--clear", action="store_true", help="also empty LIBERO's accumulating sampler list")
    args = p.parse_args()

    actions = np.load(args.actions) if args.actions else C.synthetic_actions()
    seeds = [None if s == "none" else int(s) for s in args.seeds.split(",")]
    env = C.make_env(
        args.suite,
        args.task,
        episode_index=args.init,
        hard_reset=not args.soft_reset,
        observation_height=args.size,
        observation_width=args.size,
    )
    out = {"actions": actions}
    t0 = time.time()
    for r in range(args.repeats):
        env.init_state_id = args.init  # the wrapper advances it at every reset
        seed = seeds[r % len(seeds)]
        st, pr, im, su, bodies = run_once(env, actions, seed, np_seed=args.np_seed, clear=args.clear)
        out[f"bodies_{r}"] = bodies
        out[f"state_{r}"] = st
        out[f"proprio_{r}"] = pr
        out[f"success_{r}"] = su
        for k in C.CAMERAS:
            out[f"{k}_{r}"] = im[k]
        print(
            f"repeat {r} seed {seed}: steps {len(st) - 1} state {C.sha(st)} proprio {C.sha(pr)} "
            f"images {C.sha(im['image'])} {C.sha(im['image2'])} fixtures {C.sha(bodies)} "
            f"success {bool(su.any())}",
            flush=True,
        )
    print(f"{time.time() - t0:.1f} s")
    C.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.OUT / f"{args.out}.npz", **out)
    env.close()


if __name__ == "__main__":
    main()
