"""Privileged view of a LIBERO scene: true poses and collision geometry from MuJoCo.

Everything here only *reads* the simulator. The expert uses it the way a
teleoperator uses their eyes: where is the object, what shape is it, what is in
the way. Geometry is represented as points sampled on the surfaces of the
collision geoms (LIBERO objects are unions of boxes), transformed to the world
frame with the current poses.

Copied unchanged from spikes/teleop-expert/scene.py.
"""

from __future__ import annotations

import mujoco
import numpy as np

G = mujoco.mjtGeom


def _box_points(half, spacing):
    """Points on the surface of an axis-aligned box with the given half sizes."""
    pts = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        na = max(2, int(np.ceil(2 * half[a] / spacing)) + 1)
        nb = max(2, int(np.ceil(2 * half[b] / spacing)) + 1)
        ga, gb = np.meshgrid(np.linspace(-half[a], half[a], na), np.linspace(-half[b], half[b], nb))
        for sign in (-1.0, 1.0):
            p = np.zeros((ga.size, 3))
            p[:, a] = ga.ravel()
            p[:, b] = gb.ravel()
            p[:, ax] = sign * half[ax]
            pts.append(p)
    return np.concatenate(pts)


def _local_points(m, g, spacing):
    """Surface samples of geom g in its own frame. Round shapes use their bounding box
    for the lateral surface, which errs on the side of 'occupied'."""
    typ = m.geom_type[g]
    size = m.geom_size[g]
    if typ == G.mjGEOM_BOX:
        return _box_points(size, spacing)
    if typ == G.mjGEOM_SPHERE:
        return _box_points(np.array([size[0]] * 3), spacing)
    if typ == G.mjGEOM_ELLIPSOID:
        return _box_points(size, spacing)
    if typ == G.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        n = max(8, int(np.ceil(2 * np.pi * r / spacing)))
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        zs = np.linspace(-h, h, max(2, int(np.ceil(2 * h / spacing)) + 1))
        side = np.array([[r * np.cos(a), r * np.sin(a), z] for a in ang for z in zs])
        rr = np.linspace(0, r, max(2, int(np.ceil(r / spacing)) + 1))
        caps = np.array(
            [[q * np.cos(a), q * np.sin(a), s * h] for a in ang for q in rr for s in (-1, 1)]
        )
        return np.concatenate([side, caps])
    if typ == G.mjGEOM_CAPSULE:
        return _box_points(np.array([size[0], size[0], size[1] + size[0]]), spacing)
    if typ == G.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        rng = np.random.default_rng(0)
        tri = v[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = int(np.clip(area.sum() / spacing**2, 50, 20000))
        idx = rng.choice(len(f), size=n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        p = (
            (1 - s)[:, None] * tri[idx, 0]
            + (s * (1 - r2))[:, None] * tri[idx, 1]
            + (s * r2)[:, None] * tri[idx, 2]
        )
        return np.concatenate([v, p])
    return np.zeros((0, 3))  # planes and height fields are handled as support heights


class Scene:
    def __init__(self, env, obj_spacing=0.003, obstacle_spacing=0.006):
        self.env = env.env if hasattr(env, "env") and hasattr(env.env, "sim") else env
        e = self.env
        self.m = e.sim.model._model
        self.d = e.sim.data._data
        m = self.m
        self.names = list(e.objects_dict) + list(e.fixtures_dict)
        self.root = {}
        self.geoms = {}
        owned = set()
        for name in self.names:
            obj = e.objects_dict.get(name) or e.fixtures_dict.get(name)
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            self.root[name] = root
            bodies = {b for b in range(m.nbody) if self._descends(b, root)}
            gs = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]
            self.geoms[name] = gs
            owned.update(gs)
        self._local = {}
        self._spacing = (obj_spacing, obstacle_spacing)
        # Robot geoms: everything under the robot's base body.
        self.robot_geoms = [
            g
            for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                ("robot0", "gripper0")
            )
        ]
        self.pad_geoms = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in (
                "gripper0_right_finger1_pad_collision",
                "gripper0_right_finger2_pad_collision",
            )
        ]
        self.finger_geoms = [
            [
                g
                for g in self.robot_geoms
                if f"finger{i}" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]
            for i in (1, 2)
        ]
        self.grip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
        # World collision geoms that belong to no named object: table top, floor.
        self.world_geoms = [
            g
            for g in range(m.ngeom)
            if g not in owned
            and g not in self.robot_geoms
            and "mount" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        robot = e.robots[0]
        self.arm_qpos_idx = np.array(robot._ref_joint_pos_indexes)
        self.arm_qvel_idx = np.array(robot._ref_joint_vel_indexes)
        self.grip_qpos_idx = np.array(robot._ref_gripper_joint_pos_indexes["right"])

    def _descends(self, b, root):
        while True:
            if b == root:
                return True
            if b == 0:
                return False
            b = self.m.body_parentid[b]

    # ---- poses -----------------------------------------------------------
    def body_pos(self, name):
        return self.d.xpos[self.root[name]].copy()

    def body_mat(self, name):
        return self.d.xmat[self.root[name]].reshape(3, 3).copy()

    def body_vel(self, name):
        """Linear and angular speed of the object's root body (world frame)."""
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.root[name], v, 0)
        return v[3:], v[:3]

    def eef_pos(self):
        return self.d.site_xpos[self.grip_site].copy()

    def eef_mat(self):
        return self.d.site_xmat[self.grip_site].reshape(3, 3).copy()

    def pad_pos(self):
        return self.d.geom_xpos[self.pad_geoms].copy()

    def finger_gap(self):
        """Distance between the inner faces of the two finger pads (m)."""
        p = self.pad_pos()
        return float(np.linalg.norm(p[0] - p[1]) - 2 * self.m.geom_size[self.pad_geoms[0]][1])

    def gripper_qpos(self):
        return self.d.qpos[self.grip_qpos_idx].copy()

    def site_pos(self, site_name):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return (
            self.d.site_xpos[sid].copy(),
            self.d.site_xmat[sid].reshape(3, 3).copy(),
            self.m.site_size[sid].copy(),
        )

    # ---- geometry --------------------------------------------------------
    def geom_points(self, g, spacing):
        key = (g, spacing)
        if key not in self._local:
            self._local[key] = _local_points(self.m, g, spacing)
        loc = self._local[key]
        return self.d.geom_xpos[g] + loc @ self.d.geom_xmat[g].reshape(3, 3).T

    def points(self, name, spacing=None):
        spacing = spacing or self._spacing[0]
        return np.concatenate([self.geom_points(g, spacing) for g in self.geoms[name]])

    def obstacle_points(self, exclude=(), near=None, radius=0.35):
        """Surface points of every named object and fixture except `exclude`."""
        out = []
        for name in self.names:
            if name in exclude:
                continue
            for g in self.geoms[name]:
                if near is not None:
                    if (
                        np.linalg.norm(self.d.geom_xpos[g][:2] - near[:2]) - self.m.geom_rbound[g]
                        > radius
                    ):
                        continue
                out.append(self.geom_points(g, self._spacing[1]))
        return np.concatenate(out) if out else np.zeros((0, 3))

    def support_height(self, xy=None):
        """Height of the table top (or floor) the scene stands on."""
        best = -np.inf
        for g in self.world_geoms:
            typ = self.m.geom_type[g]
            if typ == G.mjGEOM_PLANE:
                best = max(best, self.d.geom_xpos[g][2])
            elif typ == G.mjGEOM_BOX:
                pos, half = self.d.geom_xpos[g], self.m.geom_size[g]
                if xy is None or np.all(np.abs(np.asarray(xy) - pos[:2]) <= half[:2]):
                    best = max(best, pos[2] + half[2])
        return float(best)

    def table_extent(self):
        """(xmin, xmax, ymin, ymax) of the table top, or None on a floor arena."""
        for g in self.world_geoms:
            if self.m.geom_type[g] == G.mjGEOM_BOX:
                pos, half = self.d.geom_xpos[g], self.m.geom_size[g]
                return (pos[0] - half[0], pos[0] + half[0], pos[1] - half[1], pos[1] + half[1])
        return None

    # ---- contacts --------------------------------------------------------
    def contacts_between(self, geoms_a, geoms_b):
        a, b = set(geoms_a), set(geoms_b)
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist > 0.0:
                continue
            if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
                n += 1
        return n

    def fingers_touching(self, name):
        """(finger 1 touches object, finger 2 touches object)."""
        gs = self.geoms[name]
        return tuple(self.contacts_between(f, gs) > 0 for f in self.finger_geoms)

    def robot_contact_pairs(self):
        """(robot geom name, other geom name) for every current contact of the robot."""
        rob = set(self.robot_geoms)
        name = lambda g: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)  # noqa: E731
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist > 0.0:
                continue
            if c.geom1 in rob and c.geom2 not in rob:
                out.add((name(c.geom1), name(c.geom2)))
            elif c.geom2 in rob and c.geom1 not in rob:
                out.add((name(c.geom2), name(c.geom1)))
        return sorted(out)

    def robot_contacts(self, exclude=()):
        """Names of objects/fixtures/world geoms the robot currently touches."""
        rob = set(self.robot_geoms)
        hit = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist > 0.0:
                continue
            other = c.geom2 if c.geom1 in rob else c.geom1 if c.geom2 in rob else None
            if other is None or other in rob:
                continue
            for name, gs in self.geoms.items():
                if other in gs:
                    hit.add(name)
                    break
            else:
                hit.add("world")
        return hit - set(exclude)
