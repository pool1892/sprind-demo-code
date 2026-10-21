"""Scripted pick-and-place expert that stands in for a teleoperator.

It reads the true state of the simulation (object pose and shape, target pose,
contacts) and commands the arm through the same 7-number action a policy sends
to the LIBERO environment: three position deltas, three rotation deltas, one
gripper command, each in [-1, 1], at 20 Hz. It never writes to the simulator.

The expert has no memory of what happened before it was switched on. At every
(re)start it looks at the scene and decides where to continue:
  * task already satisfied and gripper clear  -> stop
  * object between the fingers                -> carry it to the target
  * otherwise                                 -> open, rise, plan a grasp, pick
That is what makes it usable for taking over in mid-episode.

Usage:
    expert = ScriptedExpert(env)          # env: libero OffScreenRenderEnv, after reset
    while not expert.done:
        obs, reward, done, info = env.step(expert.act())
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from grasp import HALF_OPEN, PALM_ABOVE, TIP_BELOW, plan_grasps, site_floor
from scene import Scene

HOME_YAW = np.pi / 2      # closing direction along world y, as in LIBERO's start pose
POS_STEP = 0.05           # metres of position goal per unit action (controller output_max)
ROT_STEP = 0.5            # radians of rotation goal per unit action
KP = 1.5
KR = 1.0
OPEN, CLOSE, HOLD = -1.0, 1.0, 0.0


def _top_down(yaw, tilt=0.0):
    """Gripper orientation (columns: closing direction, pad width direction, fingertip direction).

    tilt = 0 points the fingers straight down with the closing line at `yaw` about world z.
    A non-zero tilt leans the gripper about the pad width direction, so that the pads become
    parallel to a wall that leans by `tilt` toward the closing direction as it goes up.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    ct, st = np.cos(tilt), np.sin(tilt)
    x = np.array([ct * c, ct * s, -st])
    z = np.array([-st * c, -st * s, -ct])
    return np.stack([x, np.cross(z, x), z], axis=1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ScriptedExpert:
    def __init__(self, env, max_steps=600, max_attempts=5, verbose=False, rng=None, planner_kwargs=None):
        self.env = env
        self.e = env.env
        self.sc = Scene(env)
        pred, obj, target = self.e.parsed_problem["goal_state"][0]
        self.pred, self.obj, self.target = pred.lower(), obj, target
        if self.pred == "in":
            # target is a containment region (a site) that belongs to a container object
            self.container = self.e.object_sites_dict[target].parent_name if hasattr(
                self.e.object_sites_dict[target], "parent_name") else target.rsplit("_contain_region", 1)[0]
        elif self.pred == "on":
            self.container = target
        else:
            raise NotImplementedError(f"goal predicate {pred}")
        if self.container not in self.sc.geoms:
            raise NotImplementedError(f"target {target} is not a movable object or fixture")
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.verbose = verbose
        self.rng = rng or np.random.default_rng(0)
        self.planner_kwargs = planner_kwargs or {}
        self.support_z = self.sc.support_height()
        self.base_xy = np.array(self.e.robots[0].base_pos[:2])
        self.t = 0
        self.attempt = 0
        self.done = False
        self.status = "running"
        self.events = []          # (step, text)
        self.first_success_step = None
        self.tried = []
        self._gen = self._script()

    # ------------------------------------------------------------------ API
    def act(self):
        if self.first_success_step is None and self._success():
            self.first_success_step = self.t
        if not self.done:
            if self.t >= self.max_steps:
                self._finish("fail:step_budget")
            else:
                try:
                    a = next(self._gen)
                    self.t += 1
                    return np.clip(a, -1.0, 1.0)
                except StopIteration:
                    if not self.done:
                        self._finish("fail:script_ended")
        self.t += 1
        return np.array([0, 0, 0, 0, 0, 0, OPEN], dtype=float)

    # -------------------------------------------------------------- helpers
    def _log(self, text):
        self.events.append((self.t, text))
        if self.verbose:
            print(f"[{self.t:4d}] {text}", flush=True)

    def _finish(self, status):
        self.done = True
        self.status = status
        self._log(status)

    def _success(self):
        return bool(self.env.check_success())

    def _yaw(self):
        R = self.sc.eef_mat()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _tilt(self):
        return float(np.arccos(np.clip(-self.sc.eef_mat()[2, 2], -1, 1)))

    def _pick_yaw(self, yaw, xy=None):
        """The closing line has no direction, so `yaw` and `yaw + 180 deg` are the same grasp.
        Choose the one that leaves the wrist joint (joint 7, limits +-2.9 rad) furthest from its
        limits. With the fingers pointing down, turning the gripper by an angle turns joint 7 by
        the opposite angle, and swinging the arm around the base (joint 1) adds to it."""
        sc = self.sc
        q = sc.d.qpos[sc.arm_qpos_idx]
        cur = self._yaw()
        dj1 = 0.0
        if xy is not None:
            dj1 = _wrap(np.arctan2(xy[1] - self.base_xy[1], xy[0] - self.base_xy[0]) - q[0])
        best = None
        for cand in (yaw, yaw + np.pi):
            turn = _wrap(cand - cur)
            j7 = q[6] - turn + dj1
            # the smaller turn, unless it would leave the wrist joint close to a limit
            cost = abs(turn) + 10.0 * max(0.0, abs(j7) - 2.2)
            if best is None or cost < best[0]:
                best = (cost, cand)
        return float(_wrap(best[1]))

    def _held(self):
        return all(self.sc.fingers_touching(self.obj))

    def _obj_points(self):
        return self.sc.points(self.obj)

    def _rotvec_to(self, ori):
        """Rotation still to do, as an axis-angle vector. `ori` is a yaw angle (fingers straight
        down) or a full 3x3 orientation."""
        target = _top_down(ori) if np.isscalar(ori) else ori
        return Rotation.from_matrix(target @ self.sc.eef_mat().T).as_rotvec()

    def _action(self, pos, ori, grip, vmax=1.0, rmax=1.0):
        a = np.zeros(7)
        # Near the edge of its reach the arm must be moved slowly. At full speed the elbow snaps
        # straight, and a straight elbow no longer answers to small position commands.
        q4 = self.sc.d.qpos[self.sc.arm_qpos_idx][3]
        if q4 > -0.9:
            vmax = min(vmax, 0.2 if q4 > -0.5 else 0.4)
        a[:3] = np.clip(KP * (pos - self.sc.eef_pos()) / POS_STEP, -vmax, vmax)
        if ori is not None:
            a[3:6] = np.clip(KR * self._rotvec_to(ori) / ROT_STEP, -rmax, rmax)
        a[6] = grip() if callable(grip) else grip
        return a

    def _preshape(self, half_open):
        """Gripper command that brings the fingers to a partial opening and keeps them there.
        The gripper command is a rate: +1 closes a bit more, -1 opens a bit more, 0 holds."""
        if half_open >= HALF_OPEN - 0.001:
            return OPEN
        target = 2 * half_open

        def cmd():
            gap = self.sc.finger_gap()
            return CLOSE if gap > target + 0.005 else OPEN if gap < target - 0.005 else HOLD
        return cmd

    def _rot_err(self, ori):
        return 0.0 if ori is None else float(np.linalg.norm(self._rotvec_to(ori)))

    def _move(self, pos, yaw, grip, tol=0.006, rtol=0.06, vmax=1.0, max_steps=140, until=None, stall=30, rmax=1.0):
        """Drive the grip site to `pos` (array or callable) with the closing line at `yaw`."""
        best, best_t, freed = np.inf, 0, 0
        for i in range(max_steps):
            p = pos() if callable(pos) else pos
            err = float(np.linalg.norm(p - self.sc.eef_pos()))
            rerr = self._rot_err(yaw)
            if until is not None and until():
                return True
            if err < tol and rerr < rtol:
                return True
            if err + 0.05 * rerr < best - 0.002:
                best, best_t = err + 0.05 * rerr, i
            elif i - best_t > stall:
                if freed < 1 and self._elbow_straight():
                    freed += 1
                    yield from self._free_elbow(grip)
                    best, best_t = np.inf, i
                    continue
                q = self.sc.d.qpos[self.sc.arm_qpos_idx]
                lim = self.sc.m.jnt_range[self.e.robots[0]._ref_joint_indexes]
                near = [f"j{i+1}" for i in range(len(q)) if min(q[i] - lim[i, 0], lim[i, 1] - q[i]) < 0.06]
                self._log(f"move stalled: pos err {err*1000:.0f} mm, rot err {np.degrees(rerr):.0f} deg; "
                          f"robot touches {self.sc.robot_contact_pairs() or 'nothing'}; joints at limit {near or 'none'}")
                return False
            yield self._action(p, yaw, grip, vmax, rmax)
        self._log("move timed out")
        return False

    def _elbow_straight(self):
        return self.sc.d.qpos[self.sc.arm_qpos_idx][3] > -0.30

    def _free_elbow(self, grip):
        """Get out of a straight elbow. With the elbow joint (joint 4) at its limit the arm is at
        the edge of its reach and small position commands do nothing: the controller would have
        to bend the elbow a lot to move the gripper a little. Measured (probes/straight_elbow.py):
        pulling the gripper toward the robot while pitching the fingertips toward the robot
        bends the elbow within a second; the usual 'keep the gripper pointing down' does not."""
        sc = self.sc
        self._log("elbow is straight: folding the arm back toward the robot")
        for _ in range(30):
            if sc.d.qpos[sc.arm_qpos_idx][3] < -1.0:
                break
            out = sc.eef_pos()[:2] - self.base_xy
            out /= max(np.linalg.norm(out), 1e-9)
            a = np.zeros(7)
            a[:2] = -out
            a[3:6] = np.array([-out[1], out[0], 0.0])       # axis z x outward
            a[6] = grip() if callable(grip) else grip
            yield a

    def _wound_up(self):
        """True if the forearm roll joint (joint 5) is turned far from its middle. That happens
        when the gripper was turned about the vertical beyond what the last joint can do: the
        gripper's direction still looks fine, but the arm is twisted and soon stops moving."""
        return abs(self.sc.d.qpos[self.sc.arm_qpos_idx][4]) > 2.0

    def _unwind(self, grip):
        """Turn the gripper about the vertical in whichever direction untwists joint 5."""
        sc = self.sc
        self._log("arm is twisted (forearm roll joint far from its middle): unwinding")
        hold = sc.eef_pos()
        sign = 1.0
        for i in range(90):
            q5 = abs(sc.d.qpos[sc.arm_qpos_idx][4])
            if q5 < 0.5:
                break
            a = np.zeros(7)
            a[:3] = np.clip(KP * (hold - sc.eef_pos()) / POS_STEP, -0.5, 0.5)
            a[5] = sign
            a[6] = grip() if callable(grip) else grip
            yield a
            if i % 4 == 3 and abs(sc.d.qpos[sc.arm_qpos_idx][4]) > q5 - 1e-3 and i < 12:
                sign = -sign      # that made it worse: turn the other way

    def _safe_z(self, a_xy, b_xy, hang, exclude=()):
        """Grip-site height at which a straight move from a to b clears everything."""
        Q = self.sc.obstacle_points(exclude=exclude)
        top = self.support_z
        if len(Q):
            ab = b_xy - a_xy
            L2 = float(ab @ ab)
            tt = np.clip(((Q[:, :2] - a_xy) @ ab) / L2, 0, 1) if L2 > 1e-9 else np.zeros(len(Q))
            dist = np.linalg.norm(Q[:, :2] - (a_xy + tt[:, None] * ab), axis=1)
            near = dist < 0.13
            if near.any():
                top = max(top, float(Q[near, 2].max()))
        z = top + 0.03 + max(TIP_BELOW, hang)
        return float(np.clip(z, self.support_z + 0.12, self.support_z + 0.48))

    def _wait_rest(self, grip, max_steps=40):
        pos, yaw = self.sc.eef_pos(), self._yaw()
        for _ in range(max_steps):
            lin, ang = self.sc.body_vel(self.obj)
            if np.linalg.norm(lin) < 0.01 and np.linalg.norm(ang) < 0.15:
                return True
            yield self._action(pos, None, grip)
        return False

    # --------------------------------------------------------------- phases
    def _script(self):
        while True:
            held = self._held()
            if self._success() and not held:
                yield from self._retreat()
                yield from self._wait_rest(OPEN)
                if self._success():
                    self._finish("success")
                    return
            if self.sc.body_pos(self.obj)[2] < self.support_z - 0.05:
                self._finish("fail:object_off_table")
                return
            self.attempt += 1
            if self.attempt > self.max_attempts:
                self._finish("fail:attempts_exhausted")
                return
            if self._elbow_straight():
                yield from self._free_elbow(CLOSE if held else OPEN)
            if self._wound_up():
                yield from self._unwind(CLOSE if held else OPEN)
            if not held:
                yield from self._clear()
                yield from self._wait_rest(OPEN)
                if self._success():
                    continue
                grasp = self._plan()
                if grasp is None:
                    self._finish("fail:no_grasp_found")
                    return
                ok = yield from self._pick(grasp)
                if not ok:
                    continue
            else:
                self._log("object is between the fingers: continuing from a held object")
                pos = self.sc.eef_pos()
                for _ in range(6):   # make sure the grip is firm
                    yield self._action(pos, None, CLOSE)
            ok = yield from self._place()
            if not ok:
                continue
            yield from self._wait_rest(OPEN)
            if self._success():
                self._finish("success")
                return
            self._log("released but the task is not satisfied")

    def _clear(self):
        """Open the gripper, rise clear of the scene and point the gripper straight down."""
        sc = self.sc
        pos = sc.eef_pos()
        if sc.finger_gap() < 0.06:
            self._log("opening the gripper")
            for _ in range(14):
                yield self._action(pos, None, OPEN)
                if sc.finger_gap() > 0.07:
                    break
        if any(sc.fingers_touching(self.obj)) and sc.finger_gap() > 0.06:
            # open, yet the object touches a finger: it may hang on it
            yield from self._shake_off()
        pos = sc.eef_pos()
        z = self._safe_z(pos[:2], pos[:2], 0.0)
        tilted = self._tilt() > 0.15
        if pos[2] < z - 0.01 or tilted:
            up = np.array([pos[0], pos[1], max(z, pos[2])])
            if tilted:
                self._log(f"gripper tilted {np.degrees(self._tilt()):.0f} deg: rising, then levelling")
                yield from self._move(up, None, OPEN, tol=0.02, max_steps=60)
                yield from self._move(up, self._pick_yaw(self._yaw()), OPEN, tol=0.02, rtol=0.08, max_steps=120)
            else:
                yield from self._move(up, self._pick_yaw(self._yaw()), OPEN, tol=0.02, rtol=0.2, max_steps=60)

    def _plan(self):
        sc = self.sc
        P = self._obj_points()
        Q = sc.obstacle_points(exclude=(self.obj,), near=sc.body_pos(self.obj), radius=0.4)
        com = sc.d.xipos[sc.root[self.obj]].copy()
        grasps = plan_grasps(P, Q, self.support_z, com, HOME_YAW, self.base_xy, rng=self.rng,
                             ref_xy=sc.body_pos(self.obj)[:2], place_r=float(np.linalg.norm(self._goal_xy() - self.base_xy)),
                             **self.planner_kwargs)
        if not grasps:
            self._log("planner found no top-down grasp")
            return None
        # On a retry, prefer a grasp that differs from the ones already tried.
        for g in grasps:
            if all(np.linalg.norm(g.pos[:2] - t.pos[:2]) > 0.015 or abs(g.yaw - t.yaw) > 0.3 for t in self.tried):
                break
        else:
            g = grasps[0]
        self.tried.append(g)
        self._log(f"grasp attempt {len(self.tried)} ({len(grasps)} candidates): pos {np.round(g.pos,3)} yaw {np.degrees(g.yaw):.0f} deg "
                  f"width {g.width*1000:.0f} mm overlap {g.overlap*1000:.0f} mm lean {np.degrees(g.tilt):.0f} deg"
                  + ("" if g.half_open >= HALF_OPEN - 0.001 else f", gripper pre-closed to {2000*g.half_open:.0f} mm"))
        return g

    def _pick(self, g):
        sc = self.sc
        pos = sc.eef_pos()
        grip = self._preshape(g.half_open)   # fully open, or narrower where space is tight
        z = self._safe_z(pos[:2], g.pos[:2], 0.0)
        z = max(z, g.pos[2] + 0.04)
        if pos[2] < z - 0.015:
            yield from self._move(np.array([pos[0], pos[1], z]), None, OPEN, tol=0.015, max_steps=60)
        yaw = self._pick_yaw(g.yaw, g.pos[:2])
        tilt = g.tilt if abs(_wrap(yaw - g.yaw)) < 1.0 else -g.tilt   # the lean flips with the gripper
        # Turn the gripper where the arm is comfortable, then travel without turning. Close to
        # its base the arm is folded up, and turning the gripper there twists the forearm
        # instead of the last joint and drags the gripper off course (measured).
        pos = sc.eef_pos()
        r_now = np.linalg.norm(pos[:2] - self.base_xy)
        r_goal = np.linalg.norm(g.pos[:2] - self.base_xy)
        if abs(_wrap(yaw - self._yaw())) > 0.35 and min(r_now, r_goal) < 0.50:
            if 0.50 <= r_now <= 0.68:
                stage = pos[:2]
            elif r_goal >= 0.50:
                stage = g.pos[:2]            # the goal itself is a comfortable place to turn
            else:
                stage = self.base_xy + 0.56 * (pos[:2] - self.base_xy) / max(r_now, 1e-6)
            yield from self._move(np.array([stage[0], stage[1], z]), None, grip, tol=0.02, max_steps=80)
            yield from self._move(np.array([stage[0], stage[1], z]), yaw, grip, tol=0.03, rtol=0.05, max_steps=80)
        ok = yield from self._move(np.array([g.pos[0], g.pos[1], z]), yaw, grip, tol=0.008, rtol=0.05)
        if not ok:
            self._log("could not reach the point above the grasp")
            return False
        pre = g.pos + np.array([0, 0, 0.03])
        yield from self._move(pre, yaw, grip, tol=0.006, rtol=0.04, max_steps=100)
        ok = yield from self._move(g.pos, yaw, grip, tol=0.003, rtol=0.04, vmax=0.5, max_steps=60, stall=15)
        if not ok and np.linalg.norm(sc.eef_pos() - g.pos) > 0.012:
            self._log("could not reach the grasp pose")
            return False
        ori = yaw
        if abs(tilt) > 0.03:
            # Thin leaning wall (a bowl's rim): lean the gripper so the pads lie flat against it.
            ori = _top_down(yaw, tilt)
            yield from self._move(g.pos, ori, grip, tol=0.003, rtol=0.03, vmax=0.5, rmax=0.25, max_steps=50, stall=15)
        # Close until the fingers stop moving.
        hold = sc.eef_pos()
        last = sc.finger_gap()
        still = 0
        for _ in range(25):
            yield self._action(hold, ori, CLOSE)
            gap = sc.finger_gap()
            still = still + 1 if abs(gap - last) < 2e-4 else 0
            last = gap
            if still >= 3:
                break
        if not self._held() or sc.finger_gap() < 0.002:
            self._log(f"grasp missed (gap {sc.finger_gap()*1000:.1f} mm, touching {sc.fingers_touching(self.obj)})")
            return False
        z0 = sc.body_pos(self.obj)[2]
        up = hold + np.array([0, 0, 0.06])
        yield from self._move(up, ori, CLOSE, tol=0.01, vmax=0.6, max_steps=60)
        if not self._held() or sc.body_pos(self.obj)[2] - z0 < 0.03:
            self._log("object slipped out while lifting")
            return False
        self._log("object lifted")
        return True

    def _goal_xy(self):
        if self.pred == "in":
            return self.sc.site_pos(self.target)[0][:2]
        return self.sc.body_pos(self.container)[:2]

    def _place(self):
        sc = self.sc
        yaw = self._yaw()
        ori = sc.eef_mat()      # carry the object in the orientation it is held in
        hang = float(sc.eef_pos()[2] - self._obj_points()[:, 2].min())

        def above_goal(z):
            # Put the object's reference point, not the gripper, over the target centre.
            def f():
                off = sc.eef_pos()[:2] - sc.body_pos(self.obj)[:2]
                g = self._goal_xy() + off
                return np.array([g[0], g[1], z])
            return f

        lost = lambda: not any(sc.fingers_touching(self.obj))  # noqa: E731
        pos = sc.eef_pos()
        z = self._safe_z(pos[:2], above_goal(0)()[:2], hang, exclude=(self.obj,))
        if pos[2] < z - 0.015:
            yield from self._move(np.array([pos[0], pos[1], z]), ori, CLOSE, tol=0.015, max_steps=80, until=lost)
        if lost():
            self._log("object dropped while rising")
            return False
        # Swinging the arm around its base turns the wrist joint if the gripper keeps its
        # direction in the room. Turn the gripper along with the arm instead.
        g0 = above_goal(z)()
        swing = _wrap(np.arctan2(g0[1] - self.base_xy[1], g0[0] - self.base_xy[0])
                      - np.arctan2(pos[1] - self.base_xy[1], pos[0] - self.base_xy[0]))
        turn = swing
        if self.pred == "in":
            # A long object carried lying down (a bottle on its side) does not fit through the
            # opening straight on. Turn it to lie along the opening's diagonal.
            alt = self._turn_for_diagonal(swing)
            if alt is not None:
                turn = alt
        ori = Rotation.from_euler("z", turn).as_matrix() @ ori
        yaw = _wrap(yaw + turn)
        ok = yield from self._move(above_goal(z), ori, CLOSE, tol=0.006, rtol=0.08, max_steps=200, until=lost)
        if lost():
            self._log("object dropped in transit")
            return False
        if not ok:
            self._log("could not reach the point above the target")
            return False
        # Lowest height for releasing: nothing of the open gripper may touch the container.
        Qc = sc.points(self.container, 0.006)
        gxy = above_goal(0)()[:2]
        z_grip = site_floor(gxy, yaw, Qc)
        if self.pred == "in":
            spos, smat, ssize = sc.site_pos(self.target)
            inner = Qc[(np.linalg.norm(Qc[:, :2] - spos[:2], axis=1) < 0.03) & (Qc[:, 2] < spos[2])]
            floor = float(inner[:, 2].max()) if len(inner) else float(spos[2] - ssize[2])
        else:
            cpos = sc.body_pos(self.container)
            inner = Qc[np.linalg.norm(Qc[:, :2] - cpos[:2], axis=1) < 0.03]
            floor = float(inner[:, 2].max()) if len(inner) else cpos[2]
        # Aim a few millimetres into the surface: the descent stops at the first contact, so the
        # object is set down, not dropped, wherever the gripper can go low enough.
        z_obj = floor + hang - 0.004
        z_rel = max(z_grip, z_obj)
        touching = lambda: sc.contacts_between(sc.geoms[self.obj], sc.geoms[self.container]) > 0  # noqa: E731
        stop = lambda: lost() or touching()  # noqa: E731
        yield from self._move(above_goal(z_rel + 0.03), ori, CLOSE, tol=0.008, max_steps=120, until=stop)
        yield from self._move(above_goal(z_rel), ori, CLOSE, tol=0.004, vmax=0.3, max_steps=80, until=stop, stall=15)
        if lost():
            self._log("object dropped while lowering")
            return False
        self._log(f"releasing at height {sc.eef_pos()[2]:.3f} (object bottom {self._obj_points()[:,2].min() - floor:+.3f} m "
                  f"above the target surface, {'resting on it' if touching() else 'in the air'})")
        hold = sc.eef_pos()
        for _ in range(14):
            yield self._action(hold, ori, OPEN)
            if sc.finger_gap() > 0.07:
                break
        yield from self._unstick(ori)
        yield from self._retreat()
        if any(sc.fingers_touching(self.obj)):
            self._log("object came up with the open gripper")
            yield from self._shake_off()
        return True

    def _turn_for_diagonal(self, swing):
        """Angle to turn the gripper about the vertical so that the held object's long side lies
        along a diagonal of the container opening. None if the object fits anyway."""
        sc = self.sc
        P = self._obj_points()
        xy = P[:, :2] - P[:, :2].mean(0)
        w, vec = np.linalg.eigh(xy.T @ xy)
        long_axis = vec[:, 1]
        length = float(np.ptp(xy @ long_axis))
        spos, smat, ssize = sc.site_pos(self.target)
        if length < 2 * min(ssize[0], ssize[1]) - 0.02:
            return None
        obj_ang = np.arctan2(long_axis[1], long_axis[0])
        q7 = sc.d.qpos[sc.arm_qpos_idx][6]
        best = None
        for sx in (1.0, -1.0):
            d = smat[:, 0] * ssize[0] + sx * smat[:, 1] * ssize[1]
            for flip in (0.0, np.pi):
                turn = _wrap(np.arctan2(d[1], d[0]) + flip - obj_ang)
                cost = abs(q7 - turn + swing - 0.4)       # where the wrist joint would end up
                if best is None or cost < best[0]:
                    best = (cost, turn)
        self._log(f"object is {length*100:.0f} cm long: turning it {np.degrees(best[1]):.0f} deg to lie along the opening's diagonal")
        return float(best[1])

    def _unstick(self, ori):
        """After opening, an object can stay hooked on one finger (seen with the milk carton,
        whose roof ridge catches on a pad). Step away from that finger, slightly downward."""
        sc = self.sc
        for _ in range(2):
            f1, f2 = sc.fingers_touching(self.obj)
            if not (f1 or f2):
                return
            pos = sc.eef_pos()
            side = np.zeros(3)
            if f1 != f2:
                side = sc.pad_pos()[0 if f1 else 1] - pos
                side[2] = 0.0
                side /= max(np.linalg.norm(side), 1e-9)
            self._log("object still touches a finger after opening: stepping away from it")
            yield from self._move(pos + 0.012 * side + np.array([0, 0, -0.006]), ori, OPEN, tol=0.003,
                                  max_steps=15, stall=8)

    def _shake_off(self):
        """Last resort for an object hanging on an open finger: quick sideways moves."""
        sc = self.sc
        pos = sc.eef_pos()
        R = sc.eef_mat()
        axis = np.array([R[0, 0], R[1, 0], 0.0])
        axis /= max(np.linalg.norm(axis), 1e-9)
        for k in range(4):
            if not any(sc.fingers_touching(self.obj)):
                return
            yield from self._move(pos + (0.04 if k % 2 == 0 else -0.04) * axis, None, OPEN, tol=0.01, max_steps=12)

    def _retreat(self):
        sc = self.sc
        pos = sc.eef_pos()
        yaw = self._pick_yaw(self._yaw()) if self._tilt() < 0.3 else None
        up = pos + np.array([0, 0, 0.08])
        up[2] = min(up[2], self.support_z + 0.48)
        yield from self._move(up, yaw, OPEN, tol=0.015, max_steps=40)
