"""Why did the planner find no grasp after a disturbance? Compares the fixtures' joints and the
bowl's surroundings before and after the set-up.  run.sh probes/why_no_grasp.py SUITE TASK INIT SCENARIO"""
import sys

import numpy as np

from common import make_env, reset_to_init
from disturb import apply_scenario
from expert import HOME_YAW, ScriptedExpert
from grasp import plan_grasps

suite, tid, iid, scen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
env, task, inits = make_env(suite, tid, renderer=False)
reset_to_init(env, inits[iid], seed=iid)
e = env.env


def joints():
    out = {}
    for name, fx in e.fixtures_dict.items():
        for j in fx.joints:
            out[j] = round(float(e.sim.data.qpos[e.sim.model.get_joint_qpos_addr(j)]), 4)
    return out


def count(label):
    ex = ScriptedExpert(env)
    sc = ex.sc
    P = sc.points(ex.obj)
    Q = sc.obstacle_points(exclude=(ex.obj,), near=sc.body_pos(ex.obj), radius=0.4)
    com = sc.d.xipos[sc.root[ex.obj]].copy()
    kw = dict(ref_xy=sc.body_pos(ex.obj)[:2], place_r=float(np.linalg.norm(ex._goal_xy() - ex.base_xy)))
    n_all = len(plan_grasps(P, Q, ex.support_z, com, HOME_YAW, ex.base_xy, **kw))
    print(f"{label}: object at {np.round(sc.body_pos(ex.obj), 3)}, fixture joints {joints()}, {n_all} grasp candidates")


count("before")
rng = np.random.default_rng([0, tid, iid, sum(map(ord, suite + scen))])
print("setup:", apply_scenario(env, scen, rng))
count("after ")
