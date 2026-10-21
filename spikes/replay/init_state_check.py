"""Does an episode's starting point depend on what the process did before?

    python init_state_check.py history stock|pinned first|after-other   # one line of hashes
    python init_state_check.py layout                                   # all ten libero_spatial tasks

'history' builds libero_spatial task 0 and resets it to initial state 0 with
seed 1000, either as the first thing in the process or after another
environment (task 5) was built, reset and closed. 'stock' resets the way
lerobot-eval does (numpy seeded once at program start, then env.reset(seed));
'pinned' uses engine.reset_episode. It prints hashes of the simulator state
after the reset, of the fixture layout and of the first image.

'layout' resets every libero_spatial task with two different scene seeds and
reports whether the simulator state after the reset (not just the picture)
differs between the two fixture layouts.
"""

from __future__ import annotations

import sys

import numpy as np

import common as C
import engine as E


def fingerprint(env, obs):
    m = C.sim_of(env).model._model
    st = C.get_mj_state(env)
    nphys = 1 + m.nq + m.nv
    return st[:nphys], st, np.concatenate([m.body_pos.ravel(), m.body_quat.ravel()]), obs["pixels"]["image"]


def history(how: str, when: str):
    np.random.seed(1000)  # lerobot-eval's set_seed(), once at program start
    if when == "after-other":
        other = C.make_env("libero_spatial", 5, observation_height=360, observation_width=360)
        if how == "stock":
            other.reset(seed=1005)
        else:
            E.reset_episode(other, E.Spec(task=5, init=0, seed=1005, noise_seed=0))
        other.close()
    env = C.make_env("libero_spatial", 0, observation_height=360, observation_width=360)
    if how == "stock":
        obs, _ = env.reset(seed=1000)
    else:
        obs = E.reset_episode(env, E.Spec(task=0, init=0, seed=1000, noise_seed=0))
    phys, full, fix, img = fingerprint(env, obs)
    print(
        f"{how:6} {when:11}: positions+velocities {C.sha(phys)}  full state {C.sha(full)}  "
        f"fixtures {C.sha(fix)}  first image {C.sha(img)}"
    )


def layout():
    differ = 0
    for task in range(10):
        env = C.make_env("libero_spatial", task, observation_height=128, observation_width=128)
        out = []
        for seed in (1, 2):
            obs = E.reset_episode(env, E.Spec(task=task, init=0, seed=seed, noise_seed=0))
            out.append(fingerprint(env, obs))
        d = np.abs(out[0][0] - out[1][0]).max()
        differ += d > 0
        print(
            f"task {task} ({env.task_description}): state after reset "
            f"{'identical' if d == 0 else f'DIFFERS, largest difference {d:.2e}'} between two fixture layouts"
        )
        env.close()
    print(f"{differ}/10 tasks start from a different simulator state when the fixtures sit elsewhere")


if __name__ == "__main__":
    if sys.argv[1] == "history":
        history(sys.argv[2], sys.argv[3])
    else:
        layout()
