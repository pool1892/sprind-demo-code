"""The parts of the LIBERO adapter that need no simulator.

`make check-sim` (the `test_sim_*.py` files) and `make smoke` cover the rest.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fleetlab.runtime import configure_process
from fleetlab.sim import libero_env


def fake_robot() -> Any:
    """A robot the way robosuite 1.5 exposes its arm controller."""
    controller = SimpleNamespace(
        input_type="delta", input_ref_frame="base", ref_ori_mat=np.eye(3), origin_ori=np.eye(3)
    )
    return SimpleNamespace(part_controllers={"right": controller})


def fake_raw_observation() -> dict[str, Any]:
    return {
        "agentview_image": np.full((4, 6, 3), 1, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((4, 6, 3), 2, dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_gripper_qvel": np.zeros(2),
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
    }


def test_control_mode_is_set_on_the_arm_controller() -> None:
    robot = fake_robot()
    controller = robot.part_controllers["right"]
    # Target poses are read in world coordinates; in the base frame the arm flies off.
    libero_env.set_relative_control(robot, False)
    assert (controller.input_type, controller.input_ref_frame) == ("absolute", "world")
    # Back to exactly what robosuite builds, so that relative control is not changed.
    libero_env.set_relative_control(robot, True)
    assert (controller.input_type, controller.input_ref_frame) == ("delta", "base")


def test_relative_control_is_refused_when_the_base_is_turned() -> None:
    # Relative commands are read along the base's axes. They mean what the policies
    # expect, changes along the world's axes, only while the two sets of axes agree.
    robot = fake_robot()
    controller = robot.part_controllers["right"]
    controller.origin_ori = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(RuntimeError, match="turned against the world"):
        libero_env.set_relative_control(robot, True)
    assert (controller.input_type, controller.input_ref_frame) == ("delta", "base")  # as it was

    libero_env.set_relative_control(robot, False)  # target poses do not depend on the base
    assert (controller.input_type, controller.input_ref_frame) == ("absolute", "world")


def test_unknown_control_mode_is_rejected_before_anything_is_built() -> None:
    with pytest.raises(ValueError, match="unknown control mode"):
        libero_env.LiberoEnv("libero_spatial", 0, control_mode="sideways")  # type: ignore[arg-type]


def test_observation_has_the_layout_lerobot_expects() -> None:
    observation = libero_env.format_observation(fake_raw_observation(), fake_robot())

    assert list(observation["pixels"]) == ["image", "image2"]
    assert observation["pixels"]["image"][0, 0, 0] == 1  # the scene camera comes first
    assert observation["pixels"]["image2"][0, 0, 0] == 2
    state = observation["robot_state"]
    assert state["eef"]["pos"].shape == (3,)
    assert state["eef"]["quat"].shape == (4,)
    assert state["eef"]["mat"].shape == (3, 3)
    assert state["gripper"]["qpos"].shape == (2,)
    assert state["joints"]["pos"].shape == (7,)


def test_batch_of_one_adds_a_leading_axis_everywhere() -> None:
    observation = libero_env.format_observation(fake_raw_observation(), fake_robot())
    batch = libero_env.as_batch_of_one(observation)

    assert batch["pixels"]["image"].shape == (1, 4, 6, 3)
    assert batch["robot_state"]["eef"]["quat"].shape == (1, 4)
    assert batch["robot_state"]["eef"]["mat"].shape == (1, 3, 3)
    assert batch["robot_state"]["joints"]["vel"].shape == (1, 7)


def test_end_effector_rotation_is_a_copy() -> None:
    robot = fake_robot()
    rotation = libero_env.end_effector_rotation(robot)
    robot.part_controllers["right"].ref_ori_mat[0, 0] = 5.0
    assert rotation[0, 0] == 1.0


def test_camera_frame_puts_both_images_upright_side_by_side() -> None:
    raw = fake_raw_observation()
    raw["agentview_image"][0, 0] = 9  # top left as delivered: bottom right once upright
    frame = libero_env.camera_frame(libero_env.format_observation(raw, fake_robot()))

    assert frame.shape == (4, 12, 3)
    assert frame.dtype == np.uint8
    assert frame[3, 5, 0] == 9
    assert (frame[:, 6:] == 2).all()


def test_unknown_suite_is_rejected_before_anything_is_built() -> None:
    with pytest.raises(ValueError, match="unknown LIBERO suite"):
        libero_env.LiberoEnv("libero_nowhere", 0)


def test_copied_values_are_still_lerobots() -> None:
    """The adapter repeats values from LeRobot's LIBERO wrapper and its evaluation defaults.

    When a LeRobot upgrade changes one of them, this fails instead of the two drifting apart.
    """
    configure_process()  # LeRobot's wrapper imports LIBERO, which prompts without its config file
    from lerobot.envs import libero as lerobot_libero
    from lerobot.envs.configs import LiberoEnv as LeRobotLiberoConfig
    from libero.libero import benchmark

    # What `lerobot-eval --env.type=libero` runs with, and the wrapper it would
    # build from that. The wrapper creates its simulator on the first reset, so
    # none is built here.
    config = LeRobotLiberoConfig()
    wrapper = lerobot_libero.LiberoEnv(
        task_suite=benchmark.get_benchmark_dict()["libero_spatial"](),
        task_id=0,
        task_suite_name="libero_spatial",
        camera_name=config.camera_name,
        control_mode=config.control_mode,
        **config.gym_kwargs,
    )

    assert lerobot_libero.TASK_SUITE_MAX_STEPS == libero_env.MAX_STEPS
    assert lerobot_libero.get_libero_dummy_action() == list(libero_env.NO_OP_ACTION)
    assert wrapper.camera_name_mapping == libero_env.CAMERAS
    assert wrapper.camera_name == list(libero_env.CAMERAS)  # same order: scene camera first
    assert config.fps == wrapper.control_freq == libero_env.CONTROL_HZ
    assert wrapper.num_steps_wait == libero_env.SETTLE_STEPS

    defaults = inspect.signature(libero_env.LiberoEnv).parameters
    image_size = defaults["image_size"].default
    assert config.observation_height == config.observation_width == image_size
    assert wrapper.observation_height == wrapper.observation_width == image_size
    assert config.control_mode == wrapper.control_mode == defaults["control_mode"].default
    assert config.hard_reset == wrapper.hard_reset == defaults["hard_reset"].default

    # Built into the adapter: it always returns the robot state beside the images.
    assert wrapper.obs_type == "pixels_agent_pos"
