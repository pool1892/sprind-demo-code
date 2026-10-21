"""Shared helpers: build a LIBERO environment the way LeRobot's wrapper does.

The environment is `libero.libero.envs.OffScreenRenderEnv`, stepped at 20 Hz
with 7-number actions in [-1, 1]: three position deltas, three rotation deltas
(axis-angle) and one gripper command (-1 open, +1 close). That is exactly the
interface `lerobot/envs/libero.py` gives a policy (control_mode="relative").
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "LIBERO_CONFIG_PATH",
    str(Path.home() / ".cache/robots-probes/libero-config-libero-port"),
)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

OUT = Path.home() / ".cache/robots-probes/teleop"
NUM_STEPS_WAIT = 10  # same settle period as LeRobot's wrapper
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)

# torch and LIBERO's benchmark module are imported only where needed: together they cost
# about 0.6 GB per process, and the batch workers do not need them.


def suite_tasks(suite_name: str):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return suite, suite.tasks


def load_init_states(suite, task_id: int) -> np.ndarray:
    import torch
    from libero.libero import get_libero_path

    task = suite.tasks[task_id]
    path = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
    return np.asarray(torch.load(path, weights_only=False))


def bddl_path(suite, task_id: int) -> str:
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def make_env_bddl(bddl: str, camera_hw: int = 256, renderer: bool = True, **kwargs):
    """Build the environment from a task file. `renderer=False` skips the offscreen renderer
    (same physics, about 0.15 GB less memory, no pictures)."""
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

    kwargs.setdefault("use_camera_obs", False)
    # robosuite ends an episode after `horizon` steps (default 1000) and then refuses to step.
    # A take-over test first plays a disturbance and then the expert, so give it room.
    kwargs.setdefault("horizon", 5000)
    if renderer:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_hw, camera_widths=camera_hw,
                                 control_freq=20, hard_reset=True, **kwargs)
    else:
        env = ControlEnv(bddl_file_name=bddl, control_freq=20, hard_reset=True, has_renderer=False,
                         has_offscreen_renderer=False, **kwargs)
    env.reset()
    return env


def make_env(suite_name: str, task_id: int, camera_hw: int = 256, **kwargs):
    suite, tasks = suite_tasks(suite_name)
    task = suite.get_task(task_id)
    env = make_env_bddl(bddl_path(suite, task_id), camera_hw, **kwargs)
    return env, task, load_init_states(suite, task_id)


def set_delta(env, use_delta: bool = True) -> None:
    for robot in env.robots:
        robot.part_controllers["right"].input_type = "delta" if use_delta else "absolute"


def reset_to_init(env, init_state, seed: int = 0, settle: int = NUM_STEPS_WAIT, pin_fixtures: bool = True):
    """Same sequence as LiberoEnv.reset(): seed, reset, set state, settle, delta mode.

    `pin_fixtures` adds what LeRobot's wrapper does not do, because without it a
    reset is not repeatable in this stack. Two separate causes, both measured:

    1. LIBERO places the fixtures (cabinet, stove) with numpy's *global* random
       generator. On the robosuite 1.5 port `env.seed()` no longer reaches it
       (robosuite 1.5 stores `self.seed = None` on the environment, which hides
       LIBERO's `seed()` method, so the port's shim seeds `env.rng` instead, and
       LIBERO's samplers do not use `env.rng`).
    2. LIBERO keeps a list (`object_property_initializers`) that grows by one
       entry at every hard reset and is never cleared. Each entry draws from the
       global generator before the fixtures are placed, so even a seeded reset
       number n differs from reset number 1.

    The effect is that the fixtures land up to about 2 cm apart from one reset
    to the next, although seed and initial state are the same. Nothing in
    site-packages is edited here; the two lines below work around it.
    """
    env.seed(seed)
    if pin_fixtures:
        env.env.object_property_initializers.clear()
        np.random.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(settle):
        obs, _, _, _ = env.step(DUMMY)
    set_delta(env, True)
    return obs


def frame(env, camera: str = "agentview", hw: int = 256) -> np.ndarray:
    """Render one RGB frame, upright."""
    img = env.sim.render(camera_name=camera, height=hw, width=hw)
    return np.ascontiguousarray(img[::-1])
