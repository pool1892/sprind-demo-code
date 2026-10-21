"""Is a reset repeatable, and is a whole expert episode?  run.sh probes/check_reset_repeatable.py SUITE TASK pin|nopin

Resets the same task to the same initial state three times and prints a fingerprint of
(a) the model's fixture positions plus all joint positions right after the reset,
(b) the state after 60 fixed actions, (c) the state after a full scripted-expert episode.
Run it twice to compare across processes. With `pin` the fingerprints agree everywhere;
with `nopin` (what LeRobot's wrapper does) the fixtures land somewhere else on every reset.
"""
import hashlib
import sys

import mujoco
import numpy as np

from common import make_env, reset_to_init
from runner import run_expert

suite, tid, pin = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "pin"
env, task, inits = make_env(suite, tid, renderer=False)


def fp(*arrays):
    return hashlib.md5(np.concatenate([np.ravel(a) for a in arrays]).tobytes()).hexdigest()[:8]


for rep in range(3):
    reset_to_init(env, inits[0], seed=0, pin_fixtures=pin)
    m, d = env.sim.model._model, env.sim.data._data
    after_reset = fp(m.body_pos, m.body_quat, d.qpos)
    fixtures = {n: np.round(m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.root_body)], 4).tolist()
                for n, o in env.env.fixtures_dict.items()}
    for _ in range(60):
        env.step(np.array([0.3, 0.2, -0.2, 0, 0, 0.1, -1.0]))
    after_fixed = fp(d.qpos, d.qvel)
    reset_to_init(env, inits[0], seed=0, pin_fixtures=pin)
    res = run_expert(env)
    d = env.sim.data._data
    print(f"reset {rep} ({'pinned' if pin else 'as LeRobot does it'}): after reset {after_reset}, after 60 fixed actions {after_fixed}, "
          f"after expert episode {fp(d.qpos, d.qvel)} ({res['status']}, {res['steps']} steps); fixtures at {fixtures}")
