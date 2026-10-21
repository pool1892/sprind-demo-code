"""Step-by-step trace of one episode: run.sh probes/trace.py SUITE TASK INIT SCENARIO LAST_STEP"""
import sys, numpy as np
from common import make_env, reset_to_init
from disturb import apply_scenario
from expert import ScriptedExpert
suite, tid, iid, scen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
T = int(sys.argv[5])
env, task, inits = make_env(suite, tid, renderer=False)
reset_to_init(env, inits[iid], seed=iid)
rng = np.random.default_rng([0, tid, iid, sum(map(ord, suite + scen))])
print(apply_scenario(env, scen, rng))
ex = ScriptedExpert(env, verbose=True)
sc = ex.sc
while not ex.done and ex.t < T:
    a = ex.act(); env.step(a)
    if ex.t % 5 == 0:
        q = sc.d.qpos[sc.arm_qpos_idx]
        print(f"t {ex.t} a {np.round(a,2)} eef {np.round(sc.eef_pos(),3)} r {np.linalg.norm(sc.eef_pos()[:2]-ex.base_xy):.3f} yaw {np.degrees(ex._yaw()):6.1f} tilt {np.degrees(ex._tilt()):5.1f} q {np.round(q,2)}")
