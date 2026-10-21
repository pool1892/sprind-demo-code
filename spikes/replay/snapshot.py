"""Save and restore the complete state of a LIBERO episode in mid-flight.

MuJoCo's positions and velocities are only part of what decides the next
step. A snapshot here has these groups, each of which can be left out on
restore to measure whether it matters:

  mujoco      time, positions, velocities, controls, mocap bodies, applied forces
  warmstart   the constraint solver's starting point (last step's accelerations)
  model       the model fields LIBERO changes at reset (fixture positions) and at run time
  arm         robosuite's arm controller (its goal pose lives here, not in MuJoCo)
  gripper     the gripper controller and the gripper's integrated command
  composite   the composite controller that splits the action
  robot       the robot object's recent-value buffers
  observables per-sensor sampling clocks and last sampled values, including the camera images
  env         robosuite's step counter, clock and done flag
  objects     LIBERO's object-state trackers
  obs         the observation the policy would be given at this instant
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

import common as C

GROUPS = (
    "mujoco",
    "warmstart",
    "model",
    "arm",
    "gripper",
    "composite",
    "robot",
    "observables",
    "env",
    "objects",
    "obs",
)
PLAIN = (int, float, bool, str, type(None), np.integer, np.floating, np.bool_)
BUFFERS = ("DeltaBuffer", "RingBuffer")
SPEC_WARM = int(mujoco.mjtState.mjSTATE_WARMSTART)
SPEC_NO_WARM = int(C.SPEC_INTEGRATION) & ~SPEC_WARM
MODEL_SKIP = {"names", "paths", "text_data"}
MODEL_SMALL = 20000  # larger model arrays are meshes and textures; they are hashed, not stored


def _snap_value(v, depth=0):
    if isinstance(v, np.ndarray):
        return ("nd", v.copy()) if v.dtype != object else None
    if isinstance(v, PLAIN):
        return ("pl", v)
    if type(v).__name__ in BUFFERS:
        return ("ob", _snap_obj(v))
    if depth < 3 and isinstance(v, (list, tuple)):
        items = [_snap_value(x, depth + 1) for x in v]
        return None if any(i is None for i in items) else ("li" if isinstance(v, list) else "tu", items)
    if depth < 3 and isinstance(v, dict):
        items = {k: _snap_value(x, depth + 1) for k, x in v.items()}
        return None if any(i is None for i in items.values()) else ("di", type(v), items)
    return None


def _snap_obj(o) -> dict:
    return {k: s for k, v in vars(o).items() if (s := _snap_value(v)) is not None}


def _restore_into(cur, s, inplace=True):
    """Return the value to store for snapshot entry s. With inplace, arrays are overwritten
    where they are, so that arrays shared between attributes (robosuite's controllers do
    this) stay shared. Sensor values are replaced instead, because observations already
    handed out refer to the same arrays."""
    kind = s[0]
    if kind == "ob":
        _restore_obj(cur, s[1], inplace)
        return cur
    if kind == "nd":
        if (
            inplace
            and isinstance(cur, np.ndarray)
            and cur.shape == s[1].shape
            and cur.dtype == s[1].dtype
            and cur.flags.writeable
        ):
            cur[...] = s[1]
            return cur
        return s[1].copy()
    if kind == "pl":
        return s[1]
    if kind in ("li", "tu"):
        old = list(cur) if isinstance(cur, (list, tuple)) and len(cur) == len(s[1]) else [None] * len(s[1])
        items = [_restore_into(c, x, inplace) for c, x in zip(old, s[1], strict=True)]
        return items if kind == "li" else tuple(items)
    if kind == "di":
        target = cur if isinstance(cur, dict) else {}
        for key in [key for key in target if key not in s[2]]:
            del target[key]
        for key, x in s[2].items():
            target[key] = _restore_into(target.get(key), x, inplace)
        return target
    raise ValueError(kind)


def _restore_obj(o, snap: dict, inplace=True):
    for k, s in snap.items():
        setattr(o, k, _restore_into(vars(o).get(k), s, inplace))


def _targets(env) -> dict:
    """group -> {name: object} for the Python-side state."""
    e = C.rs_env_of(env)
    robot = e.robots[0]
    parts = robot.composite_controller.part_controllers
    t = {
        "arm": {n: c for n, c in parts.items() if "gripper" not in n},
        "gripper": {n: c for n, c in parts.items() if "gripper" in n},
        "composite": {"composite": robot.composite_controller},
        "robot": {"robot": robot},
        "observables": {n: o for n, o in e._observables.items()},
        "env": {"env": e},
        "objects": {n: o for n, o in getattr(e, "object_states_dict", {}).items()},
    }
    for arm, g in robot.gripper.items():
        t["gripper"][f"model:{arm}"] = g
    return t


def model_arrays(env) -> dict:
    m = C.sim_of(env).model._model
    out = {}
    for name in dir(m):
        if name.startswith("_") or name in MODEL_SKIP:
            continue
        try:
            v = getattr(m, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.dtype != object and v.size:
            out[name] = v.copy()
    return out


def take(env, obs: dict | None = None) -> dict:
    arrays = model_arrays(env)
    snap = {
        "mujoco": C.get_mj_state(env, SPEC_NO_WARM),
        "warmstart": C.get_mj_state(env, SPEC_WARM),
        "model": {k: v for k, v in arrays.items() if v.size <= MODEL_SMALL},
        "model_big_hash": {k: C.sha(v) for k, v in arrays.items() if v.size > MODEL_SMALL},
        "obs": copy.deepcopy(obs),
    }
    for group, objs in _targets(env).items():
        snap[group] = {n: _snap_obj(o) for n, o in objs.items()}
    # the observation cache holds the camera images; it belongs to the observables
    snap["observables"]["__cache__"] = {
        k: np.array(v, copy=True) for k, v in C.rs_env_of(env)._obs_cache.items()
    }
    snap["env"]["env"].pop("_obs_cache", None)
    return snap


def restore(env, snap: dict, skip: tuple[str, ...] = (), forward: bool = False) -> dict | None:
    """Put a snapshot back. Returns the saved observation (or None if 'obs' is skipped)."""
    e = C.rs_env_of(env)
    sim = C.sim_of(env)
    if "model" not in skip:
        m = sim.model._model
        for name, v in snap["model"].items():
            cur = getattr(m, name)
            if cur.shape == v.shape and not np.array_equal(cur, v):
                cur[...] = v
    targets = _targets(env)
    for group, objs in targets.items():
        if group in skip:
            continue
        for n, o in objs.items():
            if n in snap[group]:
                _restore_obj(o, snap[group][n], inplace=group != "observables")
    if "observables" not in skip:
        e._obs_cache = {k: np.array(v, copy=True) for k, v in snap["observables"]["__cache__"].items()}
    if "mujoco" not in skip:
        C.set_mj_state(env, snap["mujoco"], SPEC_NO_WARM)
    if "warmstart" not in skip:
        C.set_mj_state(env, snap["warmstart"], SPEC_WARM)
    if forward:
        sim.forward()
    return None if "obs" in skip else copy.deepcopy(snap["obs"])


def _leaf_diff(a, b, path, out):
    if a is None or b is None or a[0] != b[0]:
        out.append(path)
    elif a[0] == "nd":
        if a[1].shape != b[1].shape or a[1].tobytes() != b[1].tobytes():
            out.append(path)
    elif a[0] == "pl":
        if a[1] != b[1] and not (a[1] != a[1] and b[1] != b[1]):
            out.append(path)
    elif a[0] == "ob":
        for k in a[1].keys() | b[1].keys():
            _leaf_diff(a[1].get(k), b[1].get(k), f"{path}.{k}", out)
    elif a[0] in ("li", "tu"):
        if len(a[1]) != len(b[1]):
            out.append(path)
        else:
            for i, (x, y) in enumerate(zip(a[1], b[1], strict=True)):
                _leaf_diff(x, y, f"{path}[{i}]", out)
    elif a[0] == "di":
        for k in a[2].keys() | b[2].keys():
            _leaf_diff(a[2].get(k), b[2].get(k), f"{path}[{k}]", out)


def stale(env, snap: dict, group: str) -> list[str]:
    """Names of the values in a group that currently differ from the snapshot."""
    out: list[str] = []
    if group == "warmstart":
        return [] if np.array_equal(C.get_mj_state(env, SPEC_WARM), snap["warmstart"]) else ["qacc_warmstart"]
    if group == "model":
        cur = model_arrays(env)
        return [n for n, v in snap["model"].items() if not np.array_equal(cur[n], v)]
    for n, o in _targets(env)[group].items():
        now = _snap_obj(o)
        if group == "env":
            now.pop("_obs_cache", None)
        for k in now.keys() | snap[group].get(n, {}).keys():
            _leaf_diff(now.get(k), snap[group].get(n, {}).get(k), f"{n}.{k}", out)
    return sorted(out)


def libero_restore(env, snap: dict) -> dict:
    """What LIBERO itself offers: positions and velocities, then recompute and re-observe."""
    spec = C.SPEC_QPOSQVEL
    full = snap["mujoco"]
    m = C.sim_of(env).model._model
    flat = full[: mujoco.mj_stateSize(m, spec)]  # time, qpos, qvel come first in the state vector
    raw = env._env.regenerate_obs_from_state(flat)
    return env._format_raw_obs(raw)
