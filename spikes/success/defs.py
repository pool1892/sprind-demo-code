"""Candidate definitions of success for the pick-and-place suites, as functions of the simulator state.

Everything here reads MuJoCo's model and data and nothing else: no LIBERO object, no
controller, no history beyond what a caller passes in. `GoalIndex` is built once per
scene (it only looks up ids); `features(m, d, ix)` turns one simulator state into a
handful of numbers; the definitions are boolean expressions of those numbers, and the
ones that need time ("for K steps") are written as streaks over a sequence of feature
rows.

LIBERO's own check, replicated (`libero_check`), for the two goal predicates of
`libero_spatial` and `libero_object`:

* `(On bowl plate)`: the plate's body origin is not above the bowl's
  (`z_plate <= z_bowl`), any collision geom of the bowl touches any collision geom of the
  plate (an entry in MuJoCo's contact list, whatever its distance), and the two body
  origins are less than 3 cm apart horizontally. Gripper and velocities do not enter.
* `(In item basket_1_contain_region)`: the item's body origin lies strictly inside the
  box of the site `basket_1_contain_region` (half sizes 6.1 x 6.1 x 6.9 cm, rotated by
  the site's orientation and then taken axis-aligned), with the lower face lowered by
  1 cm. LIBERO's contact test for a site always returns true, so the item does not have
  to touch the basket at all. Gripper and velocities do not enter.

Source: third_party/libero/libero/libero/envs/predicates/base_predicates.py (On, In),
envs/object_states/base_object_states.py (check_ontop, check_contain), and
envs/objects/site_object.py (in_box). `check_replica.py` compares this replica with
`env.check_success()` at every step of every episode run in this spike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Thresholds, chosen before the runs. The expert of spikes/teleop-expert waits for the same
# 1 cm/s and 0.15 rad/s before it calls an object settled.
REST_LIN = 0.01  # m/s
REST_ANG = 0.15  # rad/s
OPEN_GAP = 0.07  # m between the pads; fully open is 0.078, the soup can is held at 0.062


@dataclass(frozen=True)
class GoalIndex:
    """Ids of what the goal of one LIBERO pick-and-place task refers to."""

    kind: str  # "on" or "in"
    obj_body: int
    obj_geoms: frozenset[int]  # LIBERO's contact geoms of the object
    obj_dof: int  # first velocity index of the object's free joint
    target_body: int  # plate, or basket
    target_geoms: frozenset[int]
    site: int  # contain region (kind "in"), else -1
    site_size: tuple[float, float, float]
    robot_geoms: frozenset[int]  # every collision geom of arm and gripper
    finger_geoms: tuple[frozenset[int], frozenset[int]]  # finger 1 and finger 2, pads included
    pad_geoms: tuple[int, int]
    pad_half_width: float
    obj_weight: float  # N
    obj_up_local: tuple[float, float, float]  # the object's axis that pointed up when the index was built


def _ids(m: Any, names: list[str]) -> frozenset[int]:
    import mujoco

    out = set()
    for n in names:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        if g >= 0:
            out.add(g)
    return frozenset(out)


def build_index(problem_env: Any) -> GoalIndex:
    """From LIBERO's problem environment (`OffScreenRenderEnv.env`), once per scene."""
    import mujoco

    e = problem_env
    m = e.sim.model._model
    goal = e.parsed_problem["goal_state"]
    assert len(goal) == 1, goal
    pred, obj_name, target_name = goal[0]
    pred = pred.lower()
    obj = e.objects_dict[obj_name]
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, obj.joints[-1])
    if pred == "on":
        target = e.objects_dict.get(target_name) or e.fixtures_dict[target_name]
        site, size = -1, (0.0, 0.0, 0.0)
    elif pred == "in":
        parent = e.object_sites_dict[target_name].parent_name
        target = e.objects_dict.get(parent) or e.fixtures_dict[parent]
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, target_name)
        size = tuple(float(x) for x in e.object_sites_dict[target_name].size)
    else:
        raise NotImplementedError(pred)
    target_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target.root_body)
    robot = [
        g
        for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("robot0", "gripper0"))
        and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    pads = tuple(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")
    )
    fingers = tuple(
        frozenset(g for g in robot if f"finger{i}" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
        for i in (1, 2)
    )
    weight = float(m.body_subtreemass[obj_body]) * float(np.linalg.norm(m.opt.gravity))
    # Built right after a reset, so the object stands as LIBERO placed it: remember which of its own
    # axes points up then. Its tilt later is the angle of that axis against the vertical.
    r0 = e.sim.data._data.xmat[obj_body].reshape(3, 3)
    up_local = tuple(float(x) for x in r0.T @ np.array([0.0, 0.0, 1.0]))
    return GoalIndex(
        kind=pred,
        obj_body=obj_body,
        obj_geoms=_ids(m, list(obj.contact_geoms)),
        obj_dof=int(m.jnt_dofadr[jid]),
        target_body=target_body,
        target_geoms=_ids(m, list(target.contact_geoms)),
        site=site,
        site_size=size,  # type: ignore[arg-type]
        robot_geoms=frozenset(robot),
        finger_geoms=fingers,  # type: ignore[arg-type]
        pad_geoms=pads,  # type: ignore[arg-type]
        pad_half_width=float(m.geom_size[pads[0]][1]),
        obj_weight=weight,
        obj_up_local=up_local,  # type: ignore[arg-type]
    )


def _touch(d: Any, a: frozenset[int], b: frozenset[int], penetrating_only: bool = False) -> bool:
    for i in range(d.ncon):
        c = d.contact[i]
        if penetrating_only and c.dist > 0.0:
            continue
        if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
            return True
    return False


def _force(m: Any, d: Any, a: frozenset[int], b: frozenset[int]) -> float:
    """Sum of the normal forces of all contacts between geoms in a and geoms in b (N)."""
    import mujoco

    total, f6 = 0.0, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(float(f6[0]))
    return total


def in_region(m: Any, d: Any, ix: GoalIndex) -> bool:
    """The geometric half of LIBERO's check, without the contact test."""
    p = d.xpos[ix.obj_body]
    if ix.kind == "in":
        c = d.site_xpos[ix.site]
        half = np.abs(d.site_xmat[ix.site].reshape(3, 3) @ np.asarray(ix.site_size))
        lb, ub = c - half, c + half
        lb[2] -= 0.01
        return bool(np.all(p > lb) and np.all(p < ub))
    q = d.xpos[ix.target_body]
    return bool(q[2] <= p[2] and np.linalg.norm(p[:2] - q[:2]) < 0.03)


def libero_check(m: Any, d: Any, ix: GoalIndex) -> bool:
    """LIBERO's success check for one On or In goal, replicated from the simulator state."""
    if ix.kind == "in":
        return in_region(m, d, ix)  # the site's contact test is always true
    return in_region(m, d, ix) and _touch(d, ix.obj_geoms, ix.target_geoms)


def features(m: Any, d: Any, ix: GoalIndex) -> dict[str, float]:
    """Everything the definitions below need, from one simulator state."""
    v = d.qvel[ix.obj_dof : ix.obj_dof + 6]  # free joint: linear (world), angular (body)
    p, q = d.xpos[ix.obj_body], d.xpos[ix.target_body]
    pads = d.geom_xpos[list(ix.pad_geoms)]
    gap = float(np.linalg.norm(pads[0] - pads[1]) - 2 * ix.pad_half_width)
    fingers = ix.finger_geoms[0] | ix.finger_geoms[1]
    return {
        "libero": float(libero_check(m, d, ix)),
        "in_region": float(in_region(m, d, ix)),
        "robot_touch": float(_touch(d, ix.robot_geoms, ix.obj_geoms)),
        "f1_touch": float(_touch(d, ix.finger_geoms[0], ix.obj_geoms)),
        "f2_touch": float(_touch(d, ix.finger_geoms[1], ix.obj_geoms)),
        "other_robot_touch": float(_touch(d, ix.robot_geoms - fingers, ix.obj_geoms)),
        "robot_force": _force(m, d, ix.robot_geoms, ix.obj_geoms),
        "weight": ix.obj_weight,
        "target_touch": float(_touch(d, ix.obj_geoms, ix.target_geoms)),
        "gap": gap,
        "lin_speed": float(np.linalg.norm(v[:3])),
        "ang_speed": float(np.linalg.norm(v[3:])),
        "dz": float(p[2] - q[2]),  # object origin above target origin
        "tilt": float(np.degrees(np.arccos(np.clip((d.xmat[ix.obj_body].reshape(3, 3) @ np.asarray(ix.obj_up_local))[2], -1.0, 1.0)))),
        "dxy": float(np.linalg.norm(p[:2] - q[:2])),
        "obj_z": float(p[2]),
    }


# --------------------------------------------------------------------------------------
# The definitions, on a table of feature rows (one per control step, in order).
# Each returns a boolean array: does the definition hold at this step.
# --------------------------------------------------------------------------------------


def streak(ok: np.ndarray, k: int) -> np.ndarray:
    """True at step t if `ok` held at every step of t-k+1 .. t."""
    ok = np.asarray(ok, bool)
    out = np.zeros_like(ok)
    run = 0
    for t, x in enumerate(ok):
        run = run + 1 if x else 0
        out[t] = run >= k
    return out


def at_rest(f: dict[str, np.ndarray]) -> np.ndarray:
    return (f["lin_speed"] < REST_LIN) & (f["ang_speed"] < REST_ANG)


def def_a(f: dict[str, np.ndarray]) -> np.ndarray:
    """(a) LIBERO's check as it is."""
    return f["libero"] > 0.5


def def_b(f: dict[str, np.ndarray]) -> np.ndarray:
    """(b) LIBERO's check, and nothing of the robot touches the object."""
    return def_a(f) & (f["robot_touch"] < 0.5)


def not_grasped(f: dict[str, np.ndarray]) -> np.ndarray:
    """Not both fingers on the object: a single finger can touch it, but cannot hold it."""
    return ~((f["f1_touch"] > 0.5) & (f["f2_touch"] > 0.5))


def def_b_ungrasped(f: dict[str, np.ndarray]) -> np.ndarray:
    """(b'') LIBERO's check, and the object not between both fingers."""
    return def_a(f) & not_grasped(f)


def def_b_open(f: dict[str, np.ndarray]) -> np.ndarray:
    """(b') LIBERO's check, and the gripper open (pads further apart than OPEN_GAP)."""
    return def_a(f) & (f["gap"] > OPEN_GAP)


def def_c(f: dict[str, np.ndarray], k: int) -> np.ndarray:
    """(c) LIBERO's check, released and at rest, at every one of the last K steps."""
    return streak(def_b(f) & at_rest(f), k)


def def_d(f: dict[str, np.ndarray]) -> np.ndarray:
    """(d) Placed: LIBERO's check, released, at rest. One step.

    Adds to (b) that the object does not move. An object still falling, bouncing or
    sliding fails; one that has come to rest in the goal with the robot clear passes.
    Needs no waiting, because the velocity is part of the state.

    A first version also asked that the object touch the target. That is redundant for
    the bowl (LIBERO's On already asks for contact with the plate), it held in every
    recorded state where the rest held (expert and policy runs), and for the basket it
    would turn an item resting on another object inside the basket into a failure. It
    was dropped after the review; the scores did not change (README, section 4).
    """
    return def_b(f) & at_rest(f)


def def_d_k(f: dict[str, np.ndarray], k: int) -> np.ndarray:
    """(d, K) Placed at every one of the last K steps."""
    return streak(def_d(f), k)


def def_e(f: dict[str, np.ndarray]) -> np.ndarray:
    """(e) Carried by the target: like (d), but the robot may still touch the object.

    LIBERO's check, not between both fingers, touching the target, at rest, and the
    robot's contacts push on it with less than half its weight. Meant for a gripper that
    has opened but still rests against the object, e.g. a finger inside the bowl.
    """
    return (
        def_b_ungrasped(f)
        & (f["target_touch"] > 0.5)
        & at_rest(f)
        & (f["robot_force"] < 0.5 * f["weight"])
    )


DEFINITIONS = {
    "a_libero": def_a,
    "b_released": def_b,
    "b_ungrasped": def_b_ungrasped,
    "b_open": def_b_open,
    "c_rest_k10": lambda f: def_c(f, 10),
    "c_rest_k20": lambda f: def_c(f, 20),
    "d_placed": def_d,
    "d_placed_k5": lambda f: def_d_k(f, 5),
    "e_carried": def_e,
}
