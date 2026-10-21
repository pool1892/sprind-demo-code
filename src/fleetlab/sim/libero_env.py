"""Provisional LIBERO environment on robosuite 1.5. Enough for the smoke test.

This module builds LIBERO's off-screen environment directly and returns
observations in the layout that LeRobot's own LIBERO wrapper
(`lerobot/envs/libero.py`) produces, so LeRobot's observation preprocessing
applies unchanged. The reset sequence follows the wrapper as well. The wrapper
itself reads the arm controller the way robosuite 1.4 exposed it;
`fleetlab.sim.lerobot_compat` lets it run on 1.5 for the reference evaluation.

The defaults are those of `lerobot.envs.configs.LiberoEnv`, the configuration
`lerobot-eval --env.type=libero` runs with: 360-pixel images, 20 Hz, relative
control, hard resets, robot state beside the images. The wrapper class on its
own defaults to 256-pixel images. The values repeated below are compared with
LeRobot's in `tests/test_libero_env.py`.

The real environment interface replaces this module.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from fleetlab.runtime import configure_process

# LIBERO's camera observation -> the image name LeRobot gives it.
CAMERAS = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}

# Episode limits LeRobot evaluates with, per suite.
MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

CONTROL_HZ = 20
# Six zeros for the arm and -1 to keep the gripper open.
NO_OP_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
# Steps without motion after a reset, so that objects settle.
SETTLE_STEPS = 10

Observation = dict[str, Any]


# robosuite 1.5 moved the arm controller from `robot.controller` to one entry
# per body part. The three functions below are everything that depends on it.


def arm_controller(robot: Any) -> Any:
    return robot.part_controllers["right"]


def set_relative_control(robot: Any, relative: bool) -> None:
    """Interpret arm actions as changes of the end-effector pose, or as target poses.

    Target poses are in world coordinates, as on robosuite 1.4 and in the data
    the policies were trained on. robosuite 1.5 reads commands in the frame of
    the robot's base unless told otherwise, and LIBERO's base is never at the
    world origin (in the table scenes 0.66 m behind and 0.91 m above it): left
    at that, "hold this pose" sends the arm 0.9 m away within a second.

    Relative commands stay in the base frame, robosuite's own setting, so
    relative control is bit for bit what it was. A change of pose means the
    same there as in world coordinates only while the base is not turned
    against the world. That holds exactly in all 40 tasks of the four suites,
    and it is checked on every call: a scene that turns the robot fails here,
    instead of steering a little wrong in every episode.
    """
    controller = arm_controller(robot)
    if relative and not np.allclose(controller.origin_ori, np.eye(3), atol=1e-9):
        raise RuntimeError(
            "the robot's base is turned against the world, so relative commands would be read "
            "along the base's axes and not along the world's, as the policies expect:\n"
            f"{controller.origin_ori}"
        )
    controller.input_type = "delta" if relative else "absolute"
    controller.input_ref_frame = "base" if relative else "world"


def end_effector_rotation(robot: Any) -> npt.NDArray[np.float64]:
    # A copy: the controller keeps updating its own array.
    return np.array(arm_controller(robot).ref_ori_mat, dtype=np.float64)


def format_observation(raw: dict[str, Any], robot: Any) -> Observation:
    """Arrange a raw robosuite observation the way LeRobot's LIBERO wrapper does."""
    return {
        "pixels": {name: raw[camera] for camera, name in CAMERAS.items()},
        "robot_state": {
            "eef": {
                "pos": raw["robot0_eef_pos"],
                "quat": raw["robot0_eef_quat"],
                "mat": end_effector_rotation(robot),
            },
            "gripper": {
                "qpos": raw["robot0_gripper_qpos"],
                "qvel": raw["robot0_gripper_qvel"],
            },
            "joints": {
                "pos": raw["robot0_joint_pos"],
                "vel": raw["robot0_joint_vel"],
            },
        },
    }


def as_batch_of_one(observation: Observation) -> Observation:
    """Add a leading axis of length one to every array.

    LeRobot's preprocessing expects what a vector environment returns.
    """
    return {
        key: as_batch_of_one(value) if isinstance(value, dict) else np.asarray(value)[None]
        for key, value in observation.items()
    }


def camera_frame(observation: Observation) -> npt.NDArray[np.uint8]:
    """Both camera images side by side and upright, for a video.

    The simulator delivers images rotated by 180 degrees.
    """
    upright = [image[::-1, ::-1] for image in observation["pixels"].values()]
    return np.ascontiguousarray(np.concatenate(upright, axis=1))


class LiberoEnv:
    """One LIBERO task with two cameras, rendered off-screen."""

    def __init__(
        self,
        suite: str,
        task_id: int,
        image_size: int = 360,
        control_mode: Literal["relative", "absolute"] = "relative",
        hard_reset: bool = True,
    ) -> None:
        """`hard_reset=False` keeps model and renderer across resets, which is faster.

        Both kinds of reset start an episode from the identical state and images.
        """
        if suite not in MAX_STEPS:
            raise ValueError(f"unknown LIBERO suite {suite!r}; known: {', '.join(MAX_STEPS)}")
        if control_mode not in ("relative", "absolute"):
            raise ValueError(f"unknown control mode {control_mode!r}")
        configure_process()  # LIBERO prompts on import unless its config file exists
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        tasks = benchmark.get_benchmark_dict()[suite]()
        self.task_description: str = tasks.get_task(task_id).language
        self.max_steps = MAX_STEPS[suite]
        self._relative = control_mode == "relative"
        self._init_states = tasks.get_task_init_states(task_id)
        self._env = OffScreenRenderEnv(
            bddl_file_name=tasks.get_task_bddl_file_path(task_id),
            camera_heights=image_size,
            camera_widths=image_size,
            control_freq=CONTROL_HZ,
            hard_reset=hard_reset,
        )

    def _make_current(self) -> None:
        """Make this environment's OpenGL context the current one.

        robosuite does that only when it creates a context. With two
        environments in one process, every later render and clean-up would run
        in the context created last, and a scene would be drawn with the other
        scene's meshes and textures: wrong images, and no error.
        """
        self._env.sim._render_context_offscreen.gl_ctx.make_current()

    def reset(self, init_state: int, seed: int) -> Observation:
        """Start an episode from one of the task's recorded initial states.

        The recorded state fixes the robot and the movable objects. The seed
        fixes the rest of the scene, the exact place of the furniture. Together
        they give the same simulator state and the same images every time,
        whatever this process did before.
        """
        self._make_current()
        self._env.seed(seed)  # directly before the reset, which draws the furniture's place
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[init_state % len(self._init_states)])
        # Every reset builds new controllers, in relative mode: the no-op holds still.
        for _ in range(SETTLE_STEPS):
            raw, _, _, _ = self._env.step(NO_OP_ACTION)
        for robot in self._env.robots:
            set_relative_control(robot, self._relative)
        return format_observation(raw, self._env.robots[0])

    def step(self, action: npt.NDArray[np.float32]) -> tuple[Observation, bool, bool]:
        """Apply one action. Returns the observation, whether the episode ended, and success."""
        self._make_current()
        raw, _, done, _ = self._env.step(action)
        success = bool(self._env.check_success())
        return format_observation(raw, self._env.robots[0]), bool(done) or success, success

    def close(self) -> None:
        self._make_current()  # the renderer is freed in whichever context is current
        self._env.close()
