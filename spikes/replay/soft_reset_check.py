"""Does exact replay need hard resets?

    python soft_reset_check.py SUITE TASK hard|soft plain|history

Plays a contact-rich action script (made by scripted_actions.py for this task)
from initial state 0 and prints hashes of the whole state trace, the image
stream and the fixture layout. With 'history' the same environment first runs
an unrelated episode (another initial state, another scene seed, 120 steps of
other actions that end with the gripper closed and objects displaced), so the
reset in between has something to clean up. 'hard plain' in a fresh process is
the reference; the other three combinations must print the same hashes.
"""

from __future__ import annotations

import sys

import numpy as np

import common as C
import engine as E


def main():
    suite, task, reset, hist = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    name = "actions_pick" if (suite, task) == ("libero_spatial", 0) else f"sweep_{suite}_{task}"
    actions = np.load(C.OUT / f"{name}.npy")
    env = C.make_env(suite, task, observation_height=360, observation_width=360, hard_reset=reset == "hard")
    if hist == "history":
        E.reset_episode(env, E.Spec(task=task, init=7, seed=4242, noise_seed=0, suite=suite))
        junk = C.synthetic_actions(120, seed=9)
        junk[:, 6] = 1.0
        for a in junk:
            env.step(a)
    obs = E.reset_episode(env, E.Spec(task=task, init=0, seed=1000, noise_seed=0, suite=suite))
    states, imgs = [C.get_mj_state(env)], [E.img_digest(obs)]
    for a in actions:
        obs, _, terminated, _, _ = env.step(a)
        states.append(C.get_mj_state(env))
        imgs.append(E.img_digest(obs))
        if terminated:
            break
    m = C.sim_of(env).model._model
    fix = np.concatenate([m.body_pos.ravel(), m.body_quat.ravel()])
    print(f"{suite} {task} {reset:4} {hist:7}: state {C.sha(np.stack(states))} images {C.sha(np.stack(imgs))} fixtures {C.sha(fix)}")


if __name__ == "__main__":
    main()
