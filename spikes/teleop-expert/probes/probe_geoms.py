"""Exploration: list the collision geoms of every object, and the gripper's bodies/geoms."""
import sys

import mujoco
import numpy as np

from common import make_env, reset_to_init

suite_name = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
task_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
env, task, init_states = make_env(suite_name, task_id)
reset_to_init(env, init_states[0])
e = env.env
m = e.sim.model._model
d = e.sim.data._data
GT = {v: k for k, v in vars(mujoco.mjtGeom).items() if k.startswith("mjGEOM_")}


def body_subtree(root):
    ids = [root]
    for b in range(m.nbody):
        p = b
        while p != 0:
            p = m.body_parentid[p]
            if p == root:
                ids.append(b)
                break
    return ids


def dump(root_name):
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    bodies = body_subtree(root)
    n = 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in bodies:
            col = m.geom_contype[g] or m.geom_conaffinity[g]
            if not col:
                continue
            n += 1
            typ = m.geom_type[g]
            extra = ""
            if typ == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
                extra = f"nvert {len(v)} world min {np.round(w.min(0),3)} max {np.round(w.max(0),3)}"
            if n <= 6:
                print(f"    geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)} type {typ} size {np.round(m.geom_size[g],3)}"
                      f" grp {m.geom_group[g]} con {m.geom_contype[g]}/{m.geom_conaffinity[g]} fr {m.geom_friction[g]}"
                      f" solref {m.geom_solref[g]} condim {m.geom_condim[g]} {extra}")
    print(f"  -> {root_name}: {len(bodies)} bodies, {n} collision geoms, mass {m.body_subtreemass[root]:.4f}")


for name, obj in list(e.objects_dict.items()) + list(e.fixtures_dict.items()):
    print(name, obj.root_body)
    dump(obj.root_body)

print("--- robot/gripper bodies")
for b in range(m.nbody):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if nm and ("gripper" in nm or "robot0_right_hand" in nm or "link7" in nm):
        print(b, nm, np.round(d.xpos[b], 4))
for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if nm and "gripper" in nm and (m.geom_contype[g] or m.geom_conaffinity[g]):
        print("  ", nm, "type", m.geom_type[g], "size", np.round(m.geom_size[g], 4), "pos", np.round(d.geom_xpos[g], 4),
              "fr", m.geom_friction[g], "body", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
print("grip site", np.round(d.site_xpos[sid], 4), "\n", np.round(d.site_xmat[sid].reshape(3, 3), 3))
print("eef obs pos", e._get_observations()["robot0_eef_pos"])
robot = e.robots[0]
print("actuators:", [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a), m.actuator_ctrlrange[a], m.actuator_gainprm[a][:3], m.actuator_biasprm[a][:3], m.actuator_forcerange[a]) for a in range(m.nu)])
print("table/floor geoms:")
for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    if m.geom_bodyid[g] == 0 or (nm and ("table" in nm or "floor" in nm)):
        print("  ", nm, "type", m.geom_type[g], "size", np.round(m.geom_size[g], 3), "pos", np.round(d.geom_xpos[g], 3),
              "con", m.geom_contype[g], m.geom_conaffinity[g], "body", m.geom_bodyid[g])
print("joint ranges", np.round(m.jnt_range[:7], 3))
print("qpos", np.round(d.qpos[:9], 3))
print("gravity", m.opt.gravity, "cone", m.opt.cone, "impratio", m.opt.impratio, "noslip", m.opt.noslip_iterations)
