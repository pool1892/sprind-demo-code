"""Time the grasp planner and print its best candidates: run.sh probes/time_planner.py SUITE TASK INIT"""
import sys, time
import numpy as np
from common import make_env, reset_to_init
from expert import ScriptedExpert, HOME_YAW
from grasp import plan_grasps
suite, tid, iid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env, task, inits = make_env(suite, tid, use_camera_obs=False)
reset_to_init(env, inits[iid])
ex = ScriptedExpert(env)
sc = ex.sc
P = sc.points(ex.obj); Q = sc.obstacle_points(exclude=(ex.obj,), near=sc.body_pos(ex.obj), radius=0.4)
t0 = time.time()
gs = plan_grasps(P, Q, ex.support_z, sc.d.xipos[sc.root[ex.obj]].copy(), HOME_YAW, ex.base_xy)
print(f"{len(P)} object points, {len(Q)} obstacle points, {len(gs)} grasps, {time.time()-t0:.2f} s; object top {P[:,2].max():.3f} centre {np.round(sc.body_pos(ex.obj),3)}")
for g in gs[:6] + gs[-2:]:
    print(f"  score {g.score:+.4f} pos {np.round(g.pos,3)} yaw {np.degrees(g.yaw):4.0f} width {g.width*1000:4.0f} overlap {g.overlap*1000:4.0f} tilt {np.degrees(g.tilt):5.1f} " + " ".join(f"{k} {v*1000:.0f}" for k, v in g.info.items()))
