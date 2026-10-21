"""An arm with a straight elbow is hard to pull back. Which commands free it?

    run.sh probes/straight_elbow.py

Starts from a state the 'closed_empty' disturbance produced (libero_spatial task 6, state 1:
gripper 79 cm out, elbow joint at its limit) and tries several constant commands for 60 steps.
"""
import numpy as np

from common import make_env, reset_to_init
from disturb import apply_scenario
from expert import ScriptedExpert

suite, tid, iid, scen = "libero_spatial", 6, 1, "closed_empty"
env, task, inits = make_env(suite, tid, renderer=False)


def setup():
    reset_to_init(env, inits[iid], seed=iid)
    rng = np.random.default_rng([0, tid, iid, sum(map(ord, suite + scen))])
    apply_scenario(env, scen, rng)
    return ScriptedExpert(env)


tests = {
    "pull back, full command": [-1, 0, 0, 0, 0, 0, -1],
    "pull back, small command": [-0.3, 0, 0, 0, 0, 0, -1],
    "straight up, full": [0, 0, 1, 0, 0, 0, -1],
    "straight down, full": [0, 0, -1, 0, 0, 0, -1],
    "back and up": [-1, 0, 1, 0, 0, 0, -1],
    "back and down": [-1, 0, -1, 0, 0, 0, -1],
    "sideways": [0, 1, 0, 0, 0, 0, -1],
    "pitch the hand (rotate about y, +)": [0, 0, 0, 0, 1, 0, -1],
    "pitch the hand (rotate about y, -)": [0, 0, 0, 0, -1, 0, -1],
    "back + pitch +": [-1, 0, 0, 0, 1, 0, -1],
    "back + pitch -": [-1, 0, 0, 0, -1, 0, -1],
}
for name, a in tests.items():
    ex = setup()
    sc = ex.sc
    q0 = sc.d.qpos[sc.arm_qpos_idx].copy()
    p0 = sc.eef_pos()
    # the direction 'back' is toward the robot base
    back = ex.base_xy - p0[:2]
    back /= np.linalg.norm(back)
    act = np.array(a, dtype=float)
    if act[0] != 0:
        act[:2] = -act[0] * back
    for _ in range(60):
        env.step(act)
    q = sc.d.qpos[sc.arm_qpos_idx]
    p = sc.eef_pos()
    print(f"{name:36s}: elbow j4 {q0[3]:+.2f} -> {q[3]:+.2f}, gripper moved {np.round(p - p0, 3)}, "
          f"distance from base {np.linalg.norm(p0[:2]-ex.base_xy):.3f} -> {np.linalg.norm(p[:2]-ex.base_xy):.3f}, lean {np.degrees(ex._tilt()):.0f} deg")
