"""Make a contact-rich action script without a policy.

A small state-feedback routine drives the gripper onto a bowl, closes, lifts,
carries, releases and then sweeps the closed gripper at table height through
the other objects and on to the stove. The actions it produces are saved and afterwards only
ever replayed open loop, so the script is just a fixed list of 280 actions
that happens to grasp, drop and push things.

    python scripted_actions.py --out actions_pick [--task 0] [--init 0]
"""

from __future__ import annotations

import argparse

import numpy as np

import common as C
import engine as E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="actions_pick")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init", type=int, default=0)
    p.add_argument("--steps", type=int, default=280)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    env = C.make_env(args.suite, args.task, observation_height=128, observation_width=128)
    spec = E.Spec(task=args.task, init=args.init, seed=args.seed, noise_seed=0, suite=args.suite)
    obs = E.reset_episode(env, spec)
    e = C.rs_env_of(env)
    sim = C.sim_of(env)
    m, d = sim.model._model, sim.data._data
    bowl = next((n for n in e.objects_dict if "bowl" in n), next(iter(e.objects_dict)))
    bowl_id = m.body(e.objects_dict[bowl].root_body).id
    stove = next((n for n in e.fixtures_dict if "stove" in n), None)
    stove_pos = d.xpos[m.body(e.fixtures_dict[stove].root_body).id].copy() if stove else None
    start_objs = {n: d.xpos[m.body(o.root_body).id].copy() for n, o in e.objects_dict.items()}
    bowl0 = d.xpos[bowl_id].copy()
    eef0 = np.array(obs["robot_state"]["eef"]["pos"])
    rim = bowl0 + np.array([0.0, 0.045, 0.0])  # straddle the rim of the bowl

    def target(t):
        if t < 25:
            return rim + [0, 0, 0.12], -1
        if t < 50:
            return rim + [0, 0, 0.005], -1
        if t < 65:
            return rim + [0, 0, 0.005], 1
        if t < 95:
            return rim + [0, 0, 0.20], 1
        if t < 125:
            return rim + [0.10, -0.12, 0.20], 1
        if t < 140:
            return rim + [0.10, -0.12, 0.20], -1
        if t < 160:
            return bowl0 + [0.12, 0.10, 0.03], -1
        # sweep at table height through the other objects and on to the stove, gripper closed
        z = bowl0[2] + 0.025
        others = [v for n, v in start_objs.items() if n != bowl]
        legs = [*others, stove_pos if stove_pos is not None else eef0]
        leg = min((t - 160) // 30, len(legs) - 1)
        return np.array([legs[leg][0], legs[leg][1], z]), 1

    acts, ncon = [], []
    for t in range(args.steps):
        tgt, grip = target(t)
        eef = np.array(obs["robot_state"]["eef"]["pos"])
        a = np.zeros(7, np.float32)
        a[:3] = np.clip((np.asarray(tgt) - eef) / 0.05, -1, 1)
        a[6] = grip
        obs, _, terminated, _, info = env.step(a)
        acts.append(a)
        ncon.append(d.ncon)
        if terminated:
            break
    moved = {n: float(np.linalg.norm(d.xpos[m.body(o.root_body).id] - start_objs[n])) for n, o in e.objects_dict.items()}
    print(f"task '{env.task_description}', grasp target {bowl}")
    print(f"steps {len(acts)}, contacts per step: min {min(ncon)} median {int(np.median(ncon))} max {max(ncon)}")
    print("object displacement at the end (m):", {k: round(v, 3) for k, v in moved.items()})
    np.save(C.OUT / f"{args.out}.npy", np.stack(acts))
    env.close()


if __name__ == "__main__":
    main()
