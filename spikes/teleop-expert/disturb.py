"""Disturbed start states and model faults for testing the scripted expert.

`apply_scenario` is called after the environment has been reset to one of LIBERO's
initial states. It changes the scene into the kind of state a failing policy
leaves behind, and returns a small description of what it did. The expert under
test is created afterwards and knows nothing about how the state came about.

Two ways are used to create a state:
  * teleport: the object's pose is written into the simulator and left to settle
    (displaced, on_side, inverted);
  * act it out: a throw-away expert moves the arm through the normal action
    interface, e.g. picks the object up and lets go of it in mid-air (dropped,
    held, closed_empty, awkward).

`apply_fault` edits MuJoCo model parameters: gripper friction and object mass.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from expert import CLOSE, HOLD, HOME_YAW, OPEN, ScriptedExpert, _top_down

SCENARIOS = ("init", "displaced", "dropped", "on_side", "inverted", "closed_empty", "held", "awkward")


# ----------------------------------------------------------------- helpers
def _drive(env, gen):
    """Run a generator of actions to its end and return its return value."""
    try:
        while True:
            env.step(next(gen))
    except StopIteration as stop:
        return stop.value


def _steps(env, n, action):
    for _ in range(n):
        env.step(np.asarray(action, dtype=float))


def _stay(grip):
    return [0, 0, 0, 0, 0, 0, grip]


def _free_joint(env, name):
    return env.env.objects_dict[name].joints[-1]


def _set_pose(env, name, pos, R):
    """Write an object's pose (world frame) into the simulator, at rest."""
    q = Rotation.from_matrix(R).as_quat()          # x, y, z, w
    e = env.env
    e.sim.data.set_joint_qpos(_free_joint(env, name), np.concatenate([pos, [q[3], q[0], q[1], q[2]]]))
    e.sim.data.set_joint_qvel(_free_joint(env, name), np.zeros(6))
    e.sim.forward()


def _up_axis(ex):
    """The object's 'up' direction in its own frame, taken from how it stands right now."""
    return ex.sc.body_mat(ex.obj).T @ np.array([0.0, 0.0, 1.0])


def _lean_deg(ex, up_body):
    """Angle between the object's own up direction and the world's up, in degrees."""
    return float(np.degrees(np.arccos(np.clip((ex.sc.body_mat(ex.obj) @ up_body)[2], -1, 1))))


def _free_spot(ex, rng, radius, away_from=None, r_range=(0.40, 0.70), tries=300):
    """A place on the table or floor, within the arm's reach, with nothing within `radius`."""
    sc = ex.sc
    Q = sc.obstacle_points(exclude=(ex.obj,))
    Q = Q[Q[:, 2] > ex.support_z + 0.004]
    table = sc.table_extent()
    for _ in range(tries):
        r = rng.uniform(*r_range)
        ang = rng.uniform(-0.8, 0.8)
        xy = ex.base_xy + r * np.array([np.cos(ang), np.sin(ang)])
        if table is not None and not (table[0] + 0.08 < xy[0] < table[1] - 0.08 and table[2] + 0.08 < xy[1] < table[3] - 0.08):
            continue
        if len(Q) and np.linalg.norm(Q[:, :2] - xy, axis=1).min() < radius + 0.04:
            continue
        if away_from is not None and np.linalg.norm(xy - away_from) < 0.10:
            continue
        return xy
    return None


def _describe(ex, up_body, start_xy):
    sc = ex.sc
    pos = sc.body_pos(ex.obj)
    return dict(
        object_moved_m=round(float(np.linalg.norm(pos[:2] - start_xy)), 3),
        object_height_m=round(float(pos[2] - ex.support_z), 3),
        object_lean_deg=round(_lean_deg(ex, up_body), 1),
        gripper_gap_mm=round(1000 * sc.finger_gap(), 1),
        gripper_lean_deg=round(float(np.degrees(ex._tilt())), 1),
        touching=[bool(t) for t in sc.fingers_touching(ex.obj)],
        already_success=bool(ex.env.check_success()),
    )


# --------------------------------------------------------------- scenarios
def apply_scenario(env, scenario, rng):
    if scenario == "init":
        return {}
    ex = ScriptedExpert(env)            # throw-away: used for its motion primitives and its view of the scene
    sc = ex.sc
    up_body = _up_axis(ex)
    start_xy = sc.body_pos(ex.obj)[:2].copy()
    P = sc.points(ex.obj)
    centre = sc.body_pos(ex.obj)
    radius = float(np.linalg.norm(P[:, :2] - centre[:2], axis=1).max())
    longest = float(np.linalg.norm(P - centre, axis=1).max())
    info = dict(scenario=scenario)

    if scenario in ("displaced", "on_side", "inverted"):
        spot = _free_spot(ex, rng, radius if scenario == "displaced" else longest, away_from=start_xy)
        if spot is None:
            return dict(info, invalid="no free spot")
        R = sc.body_mat(ex.obj)
        yaw = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix()
        if scenario == "displaced":
            pos = np.array([spot[0], spot[1], ex.support_z + (centre[2] - P[:, 2].min()) + 0.01])
            R_new = yaw @ R
        else:
            tip = Rotation.from_euler("x", np.pi / 2 if scenario == "on_side" else np.pi).as_matrix()
            pos = np.array([spot[0], spot[1], ex.support_z + longest + 0.01])
            R_new = yaw @ tip @ R
        _set_pose(env, ex.obj, pos, R_new)
        _steps(env, 60, _stay(OPEN))

    elif scenario in ("dropped", "held"):
        g = ex._plan()
        if g is None or not _drive(env, ex._pick(g)):
            return dict(info, invalid="setup pick failed")
        ori = sc.eef_mat()
        spot = _free_spot(ex, rng, 0.0, r_range=(0.42, 0.66))
        if spot is None:
            spot = sc.eef_pos()[:2]
        hang = float(sc.eef_pos()[2] - sc.points(ex.obj)[:, 2].min())
        z_top = ex._safe_z(sc.eef_pos()[:2], spot, hang, exclude=(ex.obj,))
        _drive(env, ex._move(np.array([*sc.eef_pos()[:2], z_top]), ori, CLOSE, tol=0.02, max_steps=60))
        _drive(env, ex._move(np.array([spot[0], spot[1], z_top]), ori, CLOSE, tol=0.015, max_steps=150))
        if scenario == "held":
            # stop somewhere between carrying height and just above the support
            z = rng.uniform(ex.support_z + hang + 0.03, z_top)
            _drive(env, ex._move(np.array([spot[0], spot[1], z]), ori, CLOSE, tol=0.015, max_steps=80))
            _steps(env, 5, _stay(CLOSE))
            if not ex._held():
                return dict(info, invalid="object lost during setup")
        else:
            _steps(env, 14, _stay(OPEN))   # let go in mid-air
            _steps(env, 50, _stay(OPEN))   # and let it land
            info["drop_height_m"] = round(float(z_top - hang - ex.support_z), 3)

    elif scenario == "closed_empty":
        # A missed grasp: gripper next to the object, low, closed on nothing.
        ang = rng.uniform(-np.pi, np.pi)
        off = (radius + rng.uniform(0.03, 0.07)) * np.array([np.cos(ang), np.sin(ang)])
        xy = sc.body_pos(ex.obj)[:2] + off
        Q = sc.obstacle_points(exclude=())
        near = Q[np.linalg.norm(Q[:, :2] - xy, axis=1) < 0.11]
        z_low = (near[:, 2].max() if len(near) else ex.support_z) + rng.uniform(0.0, 0.03)
        z_low = max(z_low, sc.points(ex.obj)[:, 2].max() - 0.02)
        yaw = HOME_YAW + rng.uniform(-1.0, 1.0)
        z_top = ex._safe_z(sc.eef_pos()[:2], xy, 0.0)
        _drive(env, ex._move(np.array([*sc.eef_pos()[:2], z_top]), None, OPEN, tol=0.02, max_steps=60))
        _drive(env, ex._move(np.array([xy[0], xy[1], z_top]), yaw, OPEN, tol=0.015, max_steps=150))
        _drive(env, ex._move(np.array([xy[0], xy[1], z_low]), yaw, OPEN, tol=0.01, max_steps=80, stall=15))
        _steps(env, 18, _stay(CLOSE))
        if all(sc.fingers_touching(ex.obj)):
            return dict(info, invalid="gripper closed on the object")

    elif scenario == "awkward":
        kind = ("tilted", "stretched", "pressing", "folded", "twisted")[int(rng.integers(5))]
        info["kind"] = kind
        grip = OPEN if rng.random() < 0.5 else CLOSE
        if kind == "tilted":
            # wrist rolled and pitched 25-50 degrees, somewhere above the scene
            spot = _free_spot(ex, rng, 0.0, r_range=(0.40, 0.65))
            spot = sc.eef_pos()[:2] if spot is None else spot
            z = ex._safe_z(sc.eef_pos()[:2], spot, 0.0) + rng.uniform(0.0, 0.08)
            _drive(env, ex._move(np.array([spot[0], spot[1], z]), None, OPEN, tol=0.02, max_steps=120))
            rot = Rotation.from_rotvec(rng.uniform(0.45, 0.9) * _unit(rng, horizontal=True)).as_matrix()
            target = rot @ _top_down(HOME_YAW + rng.uniform(-1.2, 1.2))
            _drive(env, ex._move(sc.eef_pos(), target, grip, tol=0.05, rtol=0.1, max_steps=60, stall=15))
        elif kind == "stretched":
            # arm driven out toward the edge of its reach until the elbow is nearly straight
            ang = rng.uniform(-0.7, 0.7)
            out = np.array([np.cos(ang), np.sin(ang), 0.0])
            z = ex._safe_z(sc.eef_pos()[:2], sc.eef_pos()[:2], 0.0) + 0.10
            _drive(env, ex._move(np.array([*sc.eef_pos()[:2], z]), None, OPEN, tol=0.02, max_steps=60))
            for _ in range(70):
                a = np.zeros(7)
                a[:3] = out + np.array([0, 0, 0.3])
                a[6] = grip
                env.step(np.clip(a, -1, 1))
        elif kind == "pressing":
            # gripper pushed down onto the support next to the object and still pushing
            ang = rng.uniform(-np.pi, np.pi)
            xy = sc.body_pos(ex.obj)[:2] + (radius + rng.uniform(0.06, 0.10)) * np.array([np.cos(ang), np.sin(ang)])
            spot = _free_spot(ex, rng, 0.03, r_range=(0.40, 0.68))
            Q = sc.obstacle_points(exclude=())
            if len(Q) and np.linalg.norm(Q[:, :2] - xy, axis=1).min() < 0.08 and spot is not None:
                xy = spot
            z_top = ex._safe_z(sc.eef_pos()[:2], xy, 0.0)
            _drive(env, ex._move(np.array([*sc.eef_pos()[:2], z_top]), None, OPEN, tol=0.02, max_steps=60))
            _drive(env, ex._move(np.array([xy[0], xy[1], z_top]), HOME_YAW + rng.uniform(-1, 1), grip, tol=0.02, max_steps=150))
            for _ in range(60):
                env.step(np.array([0, 0, -0.6, 0, 0, 0, grip], dtype=float))
        elif kind == "folded":
            # gripper pulled in close to the robot's base, low
            ang = rng.uniform(-0.5, 0.5)
            xy = ex.base_xy + rng.uniform(0.27, 0.33) * np.array([np.cos(ang), np.sin(ang)])
            z = ex.support_z + rng.uniform(0.12, 0.25)
            z_top = ex._safe_z(sc.eef_pos()[:2], xy, 0.0)
            _drive(env, ex._move(np.array([*sc.eef_pos()[:2], z_top]), None, OPEN, tol=0.02, max_steps=60))
            _drive(env, ex._move(np.array([xy[0], xy[1], max(z, z_top)]), HOME_YAW + rng.uniform(-1.2, 1.2), grip,
                                 tol=0.02, max_steps=150, stall=20))
        elif kind == "twisted":
            # wrist turned as far as it goes about the vertical, gripper still pointing down
            z = ex._safe_z(sc.eef_pos()[:2], sc.eef_pos()[:2], 0.0)
            _drive(env, ex._move(np.array([*sc.eef_pos()[:2], z]), None, OPEN, tol=0.02, max_steps=60))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            for _ in range(80):
                env.step(np.array([0, 0, 0, 0, 0, sign, grip], dtype=float))
        q = sc.d.qpos[sc.arm_qpos_idx]
        info["joints"] = [round(float(v), 2) for v in q]
    else:
        raise NotImplementedError(scenario)

    info.update(_describe(ex, up_body, start_xy))
    if sc.body_pos(ex.obj)[2] < ex.support_z - 0.05:
        info["invalid"] = "object left the table during setup"
    elif info["already_success"] and not any(info["touching"]):
        info["invalid"] = "task already satisfied after setup"
    return info


def _unit(rng, horizontal=False):
    v = rng.normal(size=3)
    if horizontal:
        v[2] = 0.0
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ faults
def apply_fault(env, fault):
    """Edit the MuJoCo model. Faults: none, friction_X (gripper friction times X),
    mass_X (target object mass times X), or several joined with '+'."""
    if fault in (None, "", "none"):
        return {}
    ex = ScriptedExpert(env)
    sc = ex.sc
    m, d = sc.m, sc.d
    info = {}
    for part in fault.split("+"):
        kind, value = part.rsplit("_", 1)
        x = float(value)
        if kind == "friction":
            # MuJoCo uses the larger of the two touching geoms' friction. Giving the gripper's
            # geoms priority makes its own, reduced friction the one that counts.
            geoms = [g for f in sc.finger_geoms for g in f]
            for g in geoms:
                m.geom_priority[g] = 1
                m.geom_friction[g, 0] *= x
            info["pad_friction"] = round(float(m.geom_friction[sc.pad_geoms[0], 0]), 3)
        elif kind == "mass":
            root = sc.root[ex.obj]
            bodies = [b for b in range(m.nbody) if sc._descends(b, root)]
            for b in bodies:
                m.body_mass[b] *= x
                m.body_inertia[b] *= x
            # mj_setConst recomputes the constants that depend on mass. It uses the data as
            # scratch space and leaves it in the model's default pose, so save and restore.
            qpos, qvel, ctrl, t = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.time
            mujoco.mj_setConst(m, d)
            d.qpos[:], d.qvel[:], d.ctrl[:], d.time = qpos, qvel, ctrl, t
            info["object_mass_g"] = round(1000 * float(sum(m.body_mass[b] for b in bodies)), 1)
        else:
            raise NotImplementedError(part)
    mujoco.mj_forward(m, d)
    return info
