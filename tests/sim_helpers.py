"""Shared by the simulator-level tests (`make check-sim`)."""

from __future__ import annotations

import hashlib
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from fleetlab.sim.libero_env import LiberoEnv, Observation

# Small images keep the tests quick. The scene and the renderer are the same.
IMAGE_SIZE = 128
SEED = 1000


def digest(array: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def simulator_state(env: LiberoEnv) -> Any:
    """Everything MuJoCo needs to continue exactly, the contact solver's warm start included."""
    import mujoco

    sim = env._env.sim
    everything = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model._model, everything))
    mujoco.mj_getState(sim.model._model, sim.data._data, state, everything)
    return state


def images(observation: Observation) -> dict[str, str]:
    return {name: digest(image) for name, image in observation["pixels"].items()}


def fingerprint(env: LiberoEnv, observation: Observation) -> dict[str, str]:
    """What an episode starts from: simulator state, where the furniture stands, both images."""
    model = env._env.sim.model._model
    return {
        "state": digest(simulator_state(env)),
        "fixtures": digest(np.concatenate([model.body_pos.ravel(), model.body_quat.ravel()])),
        **images(observation),
    }


def in_fresh_process(function: Callable[..., Any], *args: Any) -> Any:
    """Run a module-level function in a new interpreter and return its result."""
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        return pool.submit(function, *args).result()
