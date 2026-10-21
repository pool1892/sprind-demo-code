"""Record sixteen observations from the simulator for policy_probe.py (no policy needed).

    python make_obs.py --out obs_sim

Tasks 0 to 7 of libero_spatial, each at the start of an episode and 40 steps
into it. Kept separate from the probe so that no simulation runs while the
probe holds the GPU.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

import common as C
import engine as E


def sim_obs():
    obs, tasks = [], []
    for task in range(8):
        env = C.make_env("libero_spatial", task, observation_height=360, observation_width=360)
        o = E.reset_episode(env, E.Spec(task=task, init=0, seed=1000 + 100 * task, noise_seed=0))
        for _ in range(2):
            obs.append(copy.deepcopy({"pixels": o["pixels"], "robot_state": o["robot_state"]}))
            tasks.append(env.task_description)
            for a in C.synthetic_actions(40, seed=task):
                o, *_ = env.step(a)
        env.close()
    return obs, tasks


def _get(d, path):
    for k in path:
        d = d[k]
    return np.asarray(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="obs_sim")
    args = p.parse_args()
    obs, tasks = sim_obs()

    def rs(*path):
        return np.stack([_get(o["robot_state"], path) for o in obs])

    np.savez_compressed(
        C.OUT / f"{args.out}.npz",
        obs_image=np.stack([o["pixels"]["image"] for o in obs]),
        obs_image2=np.stack([o["pixels"]["image2"] for o in obs]),
        obs_eef_pos=rs("eef", "pos"),
        obs_eef_quat=rs("eef", "quat"),
        obs_eef_mat=rs("eef", "mat"),
        obs_gripper_qpos=rs("gripper", "qpos"),
        obs_gripper_qvel=rs("gripper", "qvel"),
        obs_joint_pos=rs("joints", "pos"),
        obs_joint_vel=rs("joints", "vel"),
        obs_task=np.array(tasks),
    )
    print(f"{len(obs)} observations written, {len(set(tasks))} distinct task texts")


if __name__ == "__main__":
    main()
