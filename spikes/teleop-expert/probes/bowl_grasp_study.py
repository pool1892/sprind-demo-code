"""Which bowl-rim grasps hold? Try many planner candidates, lift, shake, and record.

    run.sh probes/bowl_grasp_study.py TASK INIT N_TRIALS OUT.jsonl [TILT_DEG]
"""
import json
import sys

import numpy as np

from common import make_env_bddl, bddl_path, load_init_states, suite_tasks, reset_to_init
from expert import HOME_YAW, CLOSE, ScriptedExpert
from grasp import plan_grasps

tid, iid, n, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tilt_deg = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
suite, tasks = suite_tasks("libero_spatial")
inits = load_init_states(suite, tid)
env = make_env_bddl(bddl_path(suite, tid), renderer=False)
rng = np.random.default_rng(iid)


def drive(gen):
    try:
        while True:
            env.step(next(gen))
    except StopIteration as stop:
        return stop.value


rows = []
reset_to_init(env, inits[iid])
ex = ScriptedExpert(env)
sc = ex.sc
P = sc.points(ex.obj)
Q = sc.obstacle_points(exclude=(ex.obj,), near=sc.body_pos(ex.obj), radius=0.4)
gs = plan_grasps(P, Q, ex.support_z, sc.d.xipos[sc.root[ex.obj]].copy(), HOME_YAW, ex.base_xy,
                 depths=(0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024))
gs = [g for g in gs if g.info["slant"] < 0.004]
pick = rng.choice(len(gs), size=min(n, len(gs)), replace=False)
centre0 = sc.body_pos(ex.obj).copy()
rim_top = P[:, 2].max()
for j in pick:
    g = gs[j]
    if tilt_deg:
        rad0 = g.pos[:2] - centre0[:2]
        g.tilt = float(np.radians(tilt_deg) * np.sign(g.u @ rad0))
    reset_to_init(env, inits[iid])
    ex = ScriptedExpert(env)
    sc = ex.sc
    ok = drive(ex._pick(g))
    gap_after_lift = sc.finger_gap()
    held = False
    if ok:
        p0 = sc.eef_pos()
        yaw = sc.eef_mat()
        for target in (p0 + [0.12, 0.12, 0.05], p0 + [0, 0, 0.05], p0 + [0.12, -0.12, 0.05], p0):
            drive(ex._move(np.array(target), yaw, CLOSE, tol=0.01, max_steps=60))
        held = ex._held()
    rad = g.pos[:2] - centre0[:2]
    phi = float(np.degrees(np.arctan2(rad[1], rad[0])))
    radial_yaw = float((np.degrees(g.yaw) - phi + 90) % 180 - 90)
    row = dict(task=tid, init=iid, lifted=bool(ok), held=bool(held), below_rim=float(rim_top - g.pos[2]),
               radius=float(np.linalg.norm(rad)), phi=phi, yaw=float(np.degrees(g.yaw)), yaw_vs_radial=radial_yaw,
               width=g.width, overlap=g.overlap, tilt_deg=tilt_deg, gap=gap_after_lift, **g.info)
    rows.append(row)
with open(out, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "trials", sum(r["lifted"] for r in rows), "lifted", sum(r["held"] for r in rows), "held after shaking")
