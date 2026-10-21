"""Exploration: object sizes per task, gripper timing, arm response, step rate."""
import sys
import time

import numpy as np

from common import DUMMY, make_env, reset_to_init, suite_tasks
from scene import Scene

mode = sys.argv[1] if len(sys.argv) > 1 else "sizes"

if mode == "sizes":
    for suite_name in ("libero_spatial", "libero_object"):
        suite, tasks = suite_tasks(suite_name)
        for tid in range(len(tasks)):
            env, task, inits = make_env(suite_name, tid, use_camera_obs=False)
            reset_to_init(env, inits[0])
            sc = Scene(env)
            e = env.env
            goal = e.parsed_problem["goal_state"]
            name = e.obj_of_interest[0]
            p = sc.points(name)
            lo, hi = p.min(0), p.max(0)
            print(f"{suite_name} {tid} goal {goal} obj {name}: pos {np.round(sc.body_pos(name),3)} "
                  f"bbox min {np.round(lo,3)} max {np.round(hi,3)} size {np.round(hi-lo,3)} npts {len(p)} "
                  f"support {sc.support_height():.3f} mass {sc.m.body_subtreemass[sc.root[name]]*1000:.1f} g")
            if tid == 0:
                for other in sc.names:
                    q = sc.points(other, 0.006)
                    print(f"     {other}: pos {np.round(sc.body_pos(other),3)} bbox min {np.round(q.min(0),3)} max {np.round(q.max(0),3)}")
                tgt = goal[0][2]
                if tgt in e.object_sites_dict:
                    pos, mat, size = sc.site_pos(tgt)
                    print("     target site", tgt, "pos", np.round(pos, 3), "size", np.round(size, 3), "obj size attr",
                          getattr(e.object_sites_dict[tgt], "size", None))
            env.close()

if mode == "dyn":
    env, task, inits = make_env("libero_spatial", 0, use_camera_obs=False)
    reset_to_init(env, inits[0])
    sc = Scene(env)
    print("gap open", sc.finger_gap(), "gripper qpos", sc.gripper_qpos())
    print("pads", sc.pad_pos(), "eef", sc.eef_pos())
    a = np.zeros(7); a[6] = 1
    for i in range(20):
        env.step(a)
        print("close step", i, "gap %.4f" % sc.finger_gap(), sc.gripper_qpos().round(4))
    a[6] = -1
    for i in range(20):
        env.step(a)
        if i % 3 == 0:
            print("open step", i, "gap %.4f" % sc.finger_gap())
    # arm response to a constant +x action
    p0 = sc.eef_pos()
    a = np.zeros(7); a[0] = 1; a[6] = -1
    for i in range(20):
        env.step(a)
        if i % 4 == 3:
            print("x step", i, np.round(sc.eef_pos() - p0, 4))
    for i in range(10):
        env.step(DUMMY)
    print("after stop", np.round(sc.eef_pos() - p0, 4))
    # yaw response
    m0 = sc.eef_mat()
    a = np.zeros(7); a[5] = 1; a[6] = -1
    for i in range(20):
        env.step(a)
        m1 = sc.eef_mat()
        yaw = np.arctan2((m1 @ m0.T)[1, 0], (m1 @ m0.T)[0, 0])
        if i % 4 == 3:
            print("yaw step", i, "world-z rotation %.3f rad" % yaw, "pos drift", np.round(sc.eef_pos() - p0, 3))
    for i in range(10):
        env.step(DUMMY)
    m1 = sc.eef_mat()
    print("after stop yaw %.3f" % np.arctan2((m1 @ m0.T)[1, 0], (m1 @ m0.T)[0, 0]))
    # reach test: go up as far as possible
    a = np.zeros(7); a[2] = 1; a[6] = -1
    for i in range(60):
        env.step(a)
    print("max z reached straight up", sc.eef_pos(), "joints", np.round(sc.d.qpos[sc.arm_qpos_idx], 2))

if mode == "speed":
    for cam in (True, False):
        t0 = time.time()
        env, task, inits = make_env("libero_spatial", 0, use_camera_obs=cam)
        t1 = time.time()
        reset_to_init(env, inits[0])
        t2 = time.time()
        for i in range(200):
            env.step(DUMMY)
        t3 = time.time()
        print(f"camera_obs={cam}: make {t1-t0:.2f}s reset {t2-t1:.2f}s 200 steps {t3-t2:.2f}s -> {200/(t3-t2):.0f} steps/s")
        env.close()
