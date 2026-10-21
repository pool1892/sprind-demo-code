"""Where can the arm put the gripper, fingers pointing straight down? run.sh probes/reach.py SUITE

Sweeps the grip site outward from the robot along several directions and heights and prints
the largest horizontal distance from the robot base at which it still arrives within 5 mm.
"""
import sys

import numpy as np

from common import make_env, reset_to_init
from expert import HOME_YAW, OPEN, ScriptedExpert

suite = sys.argv[1]
env, task, inits = make_env(suite, 0, use_camera_obs=False, horizon=100000)
reset_to_init(env, inits[0])
ex = ScriptedExpert(env)
base = np.array(env.env.robots[0].base_pos)
print("base", base, "support", ex.support_z)


def go(target, yaw, steps=80):
    gen = ex._move(np.array(target), yaw, OPEN, tol=0.004, rtol=0.05, max_steps=steps)
    try:
        while True:
            env.step(next(gen))
    except StopIteration as s:
        return s.value


rows = []
for dz in ((0.05, 0.20, 0.30, 0.40) if suite == "libero_object" else (0.33, 0.42)):
    z = ex.support_z + dz
    for ang in (-45, 0, 45):
        for yaw in (HOME_YAW, 0.0):
            reset_to_init(env, inits[0])
            ex = ScriptedExpert(env)
            go([base[0] + 0.40, base[1], ex.support_z + 0.30], HOME_YAW, 80)
            c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            go([base[0] + 0.40 * c, base[1] + 0.40 * s_, z], yaw, 120)
            best, comfy = None, None
            for r in np.arange(0.40, 0.95, 0.02):
                ok = go([base[0] + r * c, base[1] + r * s_, z], yaw, 60)
                q4 = ex.sc.d.qpos[ex.sc.arm_qpos_idx][3]
                if not ok:
                    break
                best = r
                if q4 < -0.35:
                    comfy = r
            print(f"height +{dz:.2f} m above the support, direction {ang:4d} deg, closing-line yaw {np.degrees(yaw):3.0f}: "
                  f"arrives up to r = {best and round(best, 2)}, elbow still bent (j4 < -0.35) up to r = {comfy and round(comfy, 2)}")
# Can the arm come back once the elbow is straight?
reset_to_init(env, inits[0])
ex = ScriptedExpert(env)
z = ex.support_z + 0.30
go([base[0] + 0.95, base[1], z], HOME_YAW, 150)
q = ex.sc.d.qpos[ex.sc.arm_qpos_idx]
print("stretched out: eef", np.round(ex.sc.eef_pos(), 3), "joints", np.round(q, 2))
for label, tgt, n in (("pull straight back 20 cm", [base[0] + 0.55, base[1], z], 150),
                      ("pull back and up", [base[0] + 0.50, base[1], z + 0.15], 150),
                      ("pull back and down", [base[0] + 0.50, base[1], z - 0.15], 150),
                      ("sideways", [base[0] + 0.50, base[1] + 0.3, z], 150)):
    ok = go(tgt, HOME_YAW, n)
    q = ex.sc.d.qpos[ex.sc.arm_qpos_idx]
    print(f"  {label}: arrived {ok}, eef {np.round(ex.sc.eef_pos(), 3)} joints {np.round(q, 2)}")
