"""Absolute control goes where it is told, and relative control is untouched.

The same holds for LeRobot's own LIBERO wrapper, which runs on robosuite 1.5
through `fleetlab.sim.lerobot_compat`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim_helpers import SEED, digest, simulator_state

from fleetlab.sim import libero_env
from fleetlab.sim.libero_env import LiberoEnv, Observation

pytestmark = pytest.mark.sim

IMAGE_SIZE = 64  # the images play no part here
MILLIMETRE = 0.001


def pose(observation: Observation) -> tuple[Any, Any]:
    end_effector = observation["robot_state"]["eef"]
    return np.array(end_effector["pos"]), np.array(end_effector["mat"])


def absolute_action(position: Any, rotation: Any) -> Any:
    """Target position, target orientation as a rotation vector, gripper open."""
    return np.concatenate([position, Rotation.from_matrix(rotation).as_rotvec(), [-1.0]])


def hold_then_move(step: Any, start: Observation) -> tuple[float, float, float]:
    """Command the pose the arm has for 20 steps, then one 5 cm up and 5 cm aside for 40.

    Returns how far the end effector is from the held pose, in metres and in degrees of
    orientation, and how far it ends up from the second target, in metres.
    """
    position, rotation = pose(start)
    for _ in range(20):
        observation = step(absolute_action(position, rotation))
    held = float(np.linalg.norm(pose(observation)[0] - position))
    turned = Rotation.from_matrix(pose(observation)[1] @ rotation.T).magnitude()
    target = position + np.array([0.0, 0.05, 0.05])
    for _ in range(40):
        observation = step(absolute_action(target, rotation))
    reached = float(np.linalg.norm(pose(observation)[0] - target))
    return held, float(np.degrees(turned)), reached


def assert_held_and_reached(held: float, turned: float, reached: float) -> None:
    # Measured: 0.05 mm, 0.018 degrees and 0.5 mm. With target poses read in the
    # robot-base frame it was 898 mm and 62 degrees after the hold.
    assert held < MILLIMETRE
    assert turned < 0.1
    assert reached < MILLIMETRE


def test_absolute_control_holds_and_reaches_the_commanded_pose() -> None:
    env = LiberoEnv("libero_spatial", 0, image_size=IMAGE_SIZE, control_mode="absolute")
    start = env.reset(init_state=0, seed=SEED)

    result = hold_then_move(lambda action: env.step(action)[0], start)
    env.close()

    assert_held_and_reached(*result)


@pytest.mark.usefixtures("lerobot_on_robosuite_15")
def test_lerobots_wrapper_runs_and_agrees_with_the_adapter() -> None:
    from lerobot.envs.libero import LiberoEnv as LeRobotLiberoEnv
    from lerobot.envs.libero import _get_suite

    theirs = LeRobotLiberoEnv(
        task_suite=_get_suite("libero_spatial"),
        task_id=0,
        task_suite_name="libero_spatial",
        obs_type="pixels_agent_pos",
        observation_width=IMAGE_SIZE,
        observation_height=IMAGE_SIZE,
        control_mode="absolute",
    )
    their_start, _ = theirs.reset(seed=SEED)  # a new wrapper begins with start state 0
    result = hold_then_move(lambda action: theirs.step(action)[0], their_start)
    theirs.close()  # before ours is built: nothing on LeRobot's side switches rendering contexts
    assert_held_and_reached(*result)

    ours = LiberoEnv("libero_spatial", 0, image_size=IMAGE_SIZE, control_mode="absolute")
    our_start = ours.reset(init_state=0, seed=SEED)
    ours.close()
    assert digest(pose(our_start)[0]) == digest(pose(their_start)[0])
    assert digest(pose(our_start)[1]) == digest(pose(their_start)[1])
    for name in ("image", "image2"):
        assert digest(our_start["pixels"][name]) == digest(their_start["pixels"][name])


def test_relative_control_is_what_robosuite_built(monkeypatch: pytest.MonkeyPatch) -> None:
    env = LiberoEnv("libero_spatial", 0, image_size=IMAGE_SIZE, hard_reset=False)
    random = np.random.default_rng(0)
    actions = random.uniform(-0.5, 0.5, size=(40, 7)).astype(np.float32)

    def states_along_the_episode(after_reset: Any = None) -> list[str]:
        env.reset(init_state=0, seed=SEED)
        if after_reset is not None:
            after_reset(env._env.robots[0])
        states = []
        for action in actions:
            env.step(action)
            states.append(digest(simulator_state(env)))
        return states

    ours = states_along_the_episode()

    def to_absolute_and_back(robot: Any) -> None:
        libero_env.set_relative_control(robot, False)
        libero_env.set_relative_control(robot, True)

    there_and_back = states_along_the_episode(to_absolute_and_back)

    # The controller exactly as robosuite builds it at a reset: the adapter's switch is off.
    monkeypatch.setattr(libero_env, "set_relative_control", lambda robot, relative: None)
    untouched = states_along_the_episode()
    controller = libero_env.arm_controller(env._env.robots[0])
    env.close()

    assert (controller.input_type, controller.input_ref_frame) == ("delta", "base")
    assert ours == untouched  # bit for bit, all 40 steps
    assert there_and_back == untouched
    # Relative commands are read along the base's axes. These are the world's, exactly;
    # `set_relative_control` refuses a robot for which they are not.
    assert np.array_equal(controller.origin_ori, np.eye(3))
