"""Disturbed start states for take-over tests, ported from spikes/teleop-expert/disturb.py.

`apply_scenario(env, scenario, rng, control_mode)` is called after `env` (fleetlab's
`LiberoEnv`) has been reset to one of LIBERO's initial states. It changes the
scene into the kind of state a failing policy leaves behind and returns a small
description of what it did. The expert under test is created afterwards and
knows nothing about how the state came about.

Two ways are used to create a state:
  * teleport: the object's pose is written into the simulator and left to settle
    (displaced, on_side, inverted);
  * act it out: a throw-away expert moves the arm through the normal action
    interface, e.g. picks the object up and lets go of it in mid-air (dropped,
    held, closed_empty, awkward).

Changes against the original: every action goes through the throw-away
expert's conversion, so the arm is driven in the episode's control mode (here
always absolute); the model faults (friction, mass) are left out; the stepping
goes through fleetlab's adapter. The scenario logic and its random draws are
unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from teleop.actions import ControlMode
from teleop.expert import CLOSE, HOME_YAW, OPEN, ScriptedExpert, top_down

KINDS = ("displaced", "dropped", "on_side", "inverted", "closed_empty", "held", "awkward")


class _Driver:
    """Steps the environment with unit commands converted for the control mode."""

    def __init__(self, env: Any, ex: ScriptedExpert) -> None:
        self.env, self.ex = env, ex
        self.steps = 0

    def step(self, u: Any) -> None:
        self.env.step(self.ex.to_env_action(np.asarray(u, dtype=float)))
        self.steps += 1

    def steps_of(self, n: int, u: Any) -> None:
        for _ in range(n):
            self.step(u)

    def drive(self, gen: Any) -> Any:
        """Run a generator of unit commands to its end and return its return value."""
        try:
            while True:
                self.step(next(gen))
        except StopIteration as stop:
            return stop.value


def _stay(grip: float) -> list[float]:
    return [0, 0, 0, 0, 0, 0, grip]


def _set_pose(ex: ScriptedExpert, name: str, pos: Any, R: Any) -> None:
    """Write an object's pose (world frame) into the simulator, at rest."""
    q = Rotation.from_matrix(R).as_quat()  # x, y, z, w
    e = ex.e
    joint = e.objects_dict[name].joints[-1]
    e.sim.data.set_joint_qpos(joint, np.concatenate([pos, [q[3], q[0], q[1], q[2]]]))
    e.sim.data.set_joint_qvel(joint, np.zeros(6))
    e.sim.forward()


def _up_axis(ex: ScriptedExpert) -> Any:
    """The object's 'up' direction in its own frame, taken from how it stands right now."""
    return ex.sc.body_mat(ex.obj).T @ np.array([0.0, 0.0, 1.0])


def _lean_deg(ex: ScriptedExpert, up_body: Any) -> float:
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
        if table is not None and not (
            table[0] + 0.08 < xy[0] < table[1] - 0.08 and table[2] + 0.08 < xy[1] < table[3] - 0.08
        ):
            continue
        if len(Q) and np.linalg.norm(Q[:, :2] - xy, axis=1).min() < radius + 0.04:
            continue
        if away_from is not None and np.linalg.norm(xy - away_from) < 0.10:
            continue
        return xy
    return None


def describe(ex: ScriptedExpert, up_body: Any, start_xy: Any) -> dict[str, Any]:
    sc = ex.sc
    pos = sc.body_pos(ex.obj)
    return dict(
        object_moved_m=round(float(np.linalg.norm(pos[:2] - start_xy)), 3),
        object_height_m=round(float(pos[2] - ex.support_z), 3),
        object_lean_deg=round(_lean_deg(ex, up_body), 1),
        gripper_gap_mm=round(1000 * sc.finger_gap(), 1),
        gripper_lean_deg=round(float(np.degrees(ex._tilt())), 1),
        touching=[bool(t) for t in sc.fingers_touching(ex.obj)],
        already_success=ex._success(),
    )


def apply_scenario(env: Any, scenario: str, rng: Any, control_mode: ControlMode = "absolute") -> dict[str, Any]:
    """Turn the freshly reset scene into a disturbed state of the given kind."""
    ex = ScriptedExpert(env._env.env, control_mode=control_mode)  # throw-away: motion primitives and view
    sc = ex.sc
    drv = _Driver(env, ex)
    up_body = _up_axis(ex)
    start_xy = sc.body_pos(ex.obj)[:2].copy()
    P = sc.points(ex.obj)
    centre = sc.body_pos(ex.obj)
    radius = float(np.linalg.norm(P[:, :2] - centre[:2], axis=1).max())
    longest = float(np.linalg.norm(P - centre, axis=1).max())
    info: dict[str, Any] = dict(scenario=scenario)

    if scenario in ("displaced", "on_side", "inverted"):
        spot = _free_spot(ex, rng, radius if scenario == "displaced" else longest, away_from=start_xy)
        if spot is None:
            return dict(info, invalid="no free spot", setup_steps=drv.steps)
        R = sc.body_mat(ex.obj)
        yaw = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix()
        if scenario == "displaced":
            pos = np.array([spot[0], spot[1], ex.support_z + (centre[2] - P[:, 2].min()) + 0.01])
            R_new = yaw @ R
        else:
            tip = Rotation.from_euler("x", np.pi / 2 if scenario == "on_side" else np.pi).as_matrix()
            pos = np.array([spot[0], spot[1], ex.support_z + longest + 0.01])
            R_new = yaw @ tip @ R
        _set_pose(ex, ex.obj, pos, R_new)
        drv.steps_of(60, _stay(OPEN))

    elif scenario in ("dropped", "held"):
        g = ex._plan()
        if g is None or not drv.drive(ex._pick(g)):
            return dict(info, invalid="setup pick failed", setup_steps=drv.steps)
        ori = sc.eef_mat()
        spot = _free_spot(ex, rng, 0.0, r_range=(0.42, 0.66))
        if spot is None:
            spot = sc.eef_pos()[:2]
        hang = float(sc.eef_pos()[2] - sc.points(ex.obj)[:, 2].min())
        z_top = ex._safe_z(sc.eef_pos()[:2], spot, hang, exclude=(ex.obj,))
        drv.drive(ex._move(np.array([*sc.eef_pos()[:2], z_top]), ori, CLOSE, tol=0.02, max_steps=60))
        drv.drive(ex._move(np.array([spot[0], spot[1], z_top]), ori, CLOSE, tol=0.015, max_steps=150))
        if scenario == "held":
            # stop somewhere between carrying height and just above the support
            z = rng.uniform(ex.support_z + hang + 0.03, z_top)
            drv.drive(ex._move(np.array([spot[0], spot[1], z]), ori, CLOSE, tol=0.015, max_steps=80))
            drv.steps_of(5, _stay(CLOSE))
            if not ex._held():
                return dict(info, invalid="object lost during setup", setup_steps=drv.steps)
        else:
            drv.steps_of(14, _stay(OPEN))  # let go in mid-air
            drv.steps_of(50, _stay(OPEN))  # and let it land
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
        drv.drive(ex._move(np.array([*sc.eef_pos()[:2], z_top]), None, OPEN, tol=0.02, max_steps=60))
        drv.drive(ex._move(np.array([xy[0], xy[1], z_top]), yaw, OPEN, tol=0.015, max_steps=150))
        drv.drive(ex._move(np.array([xy[0], xy[1], z_low]), yaw, OPEN, tol=0.01, max_steps=80, stall=15))
        drv.steps_of(18, _stay(CLOSE))
        if all(sc.fingers_touching(ex.obj)):
            return dict(info, invalid="gripper closed on the object", setup_steps=drv.steps)

    elif scenario == "awkward":
        kind = ("tilted", "stretched", "pressing", "folded", "twisted")[int(rng.integers(5))]
        info["kind"] = kind
        grip = OPEN if rng.random() < 0.5 else CLOSE
        if kind == "tilted":
            # wrist rolled and pitched 25-50 degrees, somewhere above the scene
            spot = _free_spot(ex, rng, 0.0, r_range=(0.40, 0.65))
            spot = sc.eef_pos()[:2] if spot is None else spot
            z = ex._safe_z(sc.eef_pos()[:2], spot, 0.0) + rng.uniform(0.0, 0.08)
            drv.drive(ex._move(np.array([spot[0], spot[1], z]), None, OPEN, tol=0.02, max_steps=120))
            rot = Rotation.from_rotvec(rng.uniform(0.45, 0.9) * _unit(rng, horizontal=True)).as_matrix()
            target = rot @ top_down(HOME_YAW + rng.uniform(-1.2, 1.2))
            drv.drive(ex._move(sc.eef_pos(), target, grip, tol=0.05, rtol=0.1, max_steps=60, stall=15))
        elif kind == "stretched":
            # arm driven out toward the edge of its reach until the elbow is nearly straight
            ang = rng.uniform(-0.7, 0.7)
            out = np.array([np.cos(ang), np.sin(ang), 0.0])
            z = ex._safe_z(sc.eef_pos()[:2], sc.eef_pos()[:2], 0.0) + 0.10
            drv.drive(ex._move(np.array([*sc.eef_pos()[:2], z]), None, OPEN, tol=0.02, max_steps=60))
            for _ in range(70):
                a = np.zeros(7)
                a[:3] = out + np.array([0, 0, 0.3])
                a[6] = grip
                drv.step(np.clip(a, -1, 1))
        elif kind == "pressing":
            # gripper pushed down onto the support next to the object and still pushing
            ang = rng.uniform(-np.pi, np.pi)
            xy = sc.body_pos(ex.obj)[:2] + (radius + rng.uniform(0.06, 0.10)) * np.array([np.cos(ang), np.sin(ang)])
            spot = _free_spot(ex, rng, 0.03, r_range=(0.40, 0.68))
            Q = sc.obstacle_points(exclude=())
            if len(Q) and np.linalg.norm(Q[:, :2] - xy, axis=1).min() < 0.08 and spot is not None:
                xy = spot
            z_top = ex._safe_z(sc.eef_pos()[:2], xy, 0.0)
            drv.drive(ex._move(np.array([*sc.eef_pos()[:2], z_top]), None, OPEN, tol=0.02, max_steps=60))
            drv.drive(
                ex._move(np.array([xy[0], xy[1], z_top]), HOME_YAW + rng.uniform(-1, 1), grip, tol=0.02, max_steps=150)
            )
            drv.steps_of(60, [0, 0, -0.6, 0, 0, 0, grip])
        elif kind == "folded":
            # gripper pulled in close to the robot's base, low
            ang = rng.uniform(-0.5, 0.5)
            xy = ex.base_xy + rng.uniform(0.27, 0.33) * np.array([np.cos(ang), np.sin(ang)])
            z = ex.support_z + rng.uniform(0.12, 0.25)
            z_top = ex._safe_z(sc.eef_pos()[:2], xy, 0.0)
            drv.drive(ex._move(np.array([*sc.eef_pos()[:2], z_top]), None, OPEN, tol=0.02, max_steps=60))
            drv.drive(
                ex._move(
                    np.array([xy[0], xy[1], max(z, z_top)]),
                    HOME_YAW + rng.uniform(-1.2, 1.2),
                    grip,
                    tol=0.02,
                    max_steps=150,
                    stall=20,
                )
            )
        elif kind == "twisted":
            # wrist turned as far as it goes about the vertical, gripper still pointing down
            z = ex._safe_z(sc.eef_pos()[:2], sc.eef_pos()[:2], 0.0)
            drv.drive(ex._move(np.array([*sc.eef_pos()[:2], z]), None, OPEN, tol=0.02, max_steps=60))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            drv.steps_of(80, [0, 0, 0, 0, 0, sign, grip])
        q = sc.d.qpos[sc.arm_qpos_idx]
        info["joints"] = [round(float(v), 2) for v in q]
    else:
        raise NotImplementedError(scenario)

    info.update(describe(ex, up_body, start_xy))
    info["setup_steps"] = drv.steps
    if sc.body_pos(ex.obj)[2] < ex.support_z - 0.05:
        info["invalid"] = "object left the table during setup"
    elif info["already_success"] and not any(info["touching"]):
        info["invalid"] = "task already satisfied after setup"
    return info


def _unit(rng: Any, horizontal: bool = False) -> Any:
    v = rng.normal(size=3)
    if horizontal:
        v[2] = 0.0
    return v / np.linalg.norm(v)
