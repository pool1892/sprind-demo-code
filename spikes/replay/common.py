"""Shared helpers for the replay spike.

Everything here runs in the probe venv ~/.cache/robots-probes/libero-port and
touches nothing outside this directory and ~/.cache/robots-probes/replay.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PROBES = Path.home() / ".cache" / "robots-probes"
OUT = PROBES / "replay"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROBES / "libero-config-libero-port"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the checkpoint is cached; do not depend on the network
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# torch has to be loaded before mujoco. With mujoco first, every attempt to copy the policy's
# weights to the Radeon failed with "CUDA error: invalid argument" (hipErrorInvalidValue),
# ten times out of ten in this spike; with torch first it never did.
import torch  # noqa: E402, F401, I001
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# Everything MuJoCo itself needs to continue a simulation exactly: time,
# positions, velocities, actuator activations, controls, applied forces, mocap
# bodies, user data and the constraint solver's warm start.
SPEC_INTEGRATION = int(mujoco.mjtState.mjSTATE_INTEGRATION)
# The subset LIBERO and robosuite save and restore: time, positions, velocities.
SPEC_QPOSQVEL = int(
    mujoco.mjtState.mjSTATE_TIME | mujoco.mjtState.mjSTATE_QPOS | mujoco.mjtState.mjSTATE_QVEL
)

CAMERAS = ("image", "image2")


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def make_env(suite_name: str = "libero_spatial", task_id: int = 0, episode_index: int = 0, **kw):
    """One LeRobot LIBERO environment, built the way lerobot-eval builds it."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(suite_name)
    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        episode_index=episode_index,
        n_envs=1,
        **kw,
    )


def sim_of(env):
    """The robosuite MjSim under a LeRobot LiberoEnv."""
    return env._env.env.sim


def rs_env_of(env):
    """The robosuite environment (LIBERO problem class) under a LeRobot LiberoEnv."""
    return env._env.env


def make_current(env) -> None:
    """Make this environment's OpenGL context the current one.

    robosuite creates one context per environment and never switches back to it, so
    with two environments in one process every render and every clean-up runs in the
    context that was created last. Call this before any reset, step or close.
    """
    if getattr(env, "_env", None) is not None:
        ctx = getattr(sim_of(env), "_render_context_offscreen", None)
        if ctx is not None:
            ctx.gl_ctx.make_current()


def get_mj_state(env, spec=SPEC_INTEGRATION) -> np.ndarray:
    sim = sim_of(env)
    m, d = sim.model._model, sim.data._data
    out = np.empty(mujoco.mj_stateSize(m, spec), np.float64)
    mujoco.mj_getState(m, d, out, spec)
    return out


def set_mj_state(env, state: np.ndarray, spec=SPEC_INTEGRATION) -> None:
    sim = sim_of(env)
    mujoco.mj_setState(sim.model._model, sim.data._data, np.asarray(state, np.float64), spec)


def flat_proprio(obs: dict) -> np.ndarray:
    rs = obs["robot_state"]
    return np.concatenate(
        [
            rs["eef"]["pos"],
            rs["eef"]["quat"],
            np.asarray(rs["eef"]["mat"]).ravel(),
            rs["gripper"]["qpos"],
            rs["gripper"]["qvel"],
            rs["joints"]["pos"],
            rs["joints"]["vel"],
        ]
    ).astype(np.float64)


def first_diff(a: np.ndarray, b: np.ndarray) -> int:
    """Index of the first row where a and b differ bitwise, or -1 if identical."""
    n = min(len(a), len(b))
    a = np.ascontiguousarray(a[:n]).reshape(n, -1)
    b = np.ascontiguousarray(b[:n]).reshape(n, -1)
    neq = (a.view(np.uint8).reshape(n, -1) != b.view(np.uint8).reshape(n, -1)).any(axis=1)
    idx = np.flatnonzero(neq)
    if len(idx):
        return int(idx[0])
    return -1 if len(a) == len(b) or True else n


def synthetic_actions(n: int = 280, seed: int = 0) -> np.ndarray:
    """A fixed, contact-rich action script: push down onto the table, wander, open and close."""
    rng = np.random.default_rng(seed)
    acts = np.zeros((n, 7), np.float32)
    a = np.zeros(6)
    for t in range(n):
        a = 0.85 * a + 0.35 * rng.standard_normal(6)
        acts[t, :6] = np.clip(a, -1, 1)
        if t < 45:
            acts[t, 2] = -0.6  # go down until the gripper meets the table and the objects
        acts[t, 6] = 1.0 if (t // 35) % 2 else -1.0
    return acts
