"""What the reference launcher adds around LeRobot's evaluation, checked on stand-ins.

The evaluation itself needs a model: `make smoke` and the earlier reference runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from fleetlab import reference_eval
from fleetlab.sim.lerobot_compat import Robosuite14Controller


def fake_robot() -> Any:
    """A robot the way robosuite 1.5 exposes its arm controller."""
    controller = SimpleNamespace(
        input_type="delta", input_ref_frame="base", ref_ori_mat=np.eye(3), origin_ori=np.eye(3)
    )
    return SimpleNamespace(part_controllers={"right": controller})


def test_the_old_controller_names_reach_the_arm_controller() -> None:
    robot = fake_robot()
    arm = robot.part_controllers["right"]
    controller = Robosuite14Controller(robot)

    assert controller.use_delta is True
    controller.use_delta = False  # what LeRobot's wrapper does for control_mode="absolute"
    assert (arm.input_type, arm.input_ref_frame) == ("absolute", "world")
    assert controller.use_delta is False
    controller.use_delta = True
    assert (arm.input_type, arm.input_ref_frame) == ("delta", "base")

    # robosuite 1.5 keeps updating its array in place; 1.4 handed out a new one each time.
    rotation = controller.ee_ori_mat
    arm.ref_ori_mat[0, 0] = 5.0
    assert rotation[0, 0] == 1.0


class FakeVectorEnv:
    """Five LIBERO environments as LeRobot's wrapper counts start states: 50 stored, stride 5."""

    def __init__(self) -> None:
        self.counters = [0, 1, 2, 3, 4]

    def call(self, name: str) -> tuple[Any, ...]:
        stored: list[Any] = [np.zeros((50, 3))] * 5
        return tuple({"init_state_id": self.counters, "_init_states": stored}[name])


def test_start_states_of_the_counted_episodes_are_recorded() -> None:
    def rollout(env: FakeVectorEnv, **kwargs: Any) -> None:
        # A round resets every environment once. The first one finishes early here and
        # is reset once more, which is how start states get skipped with `--no-freeze`.
        env.counters = [counter + 5 for counter in env.counters]
        env.counters[0] += 5

    def eval_one(env: FakeVectorEnv, **kwargs: Any) -> dict[str, Any]:
        for _ in range(11):  # 50 episodes and one round more, to pass the last stored state
            lerobot_eval.rollout(env=env, policy=None)
        return {"successes": [True] * 52}  # LeRobot counts n_episodes, not whole rounds

    lerobot_eval = SimpleNamespace(rollout=rollout, eval_one=eval_one)
    reference_eval.record_start_states(lerobot_eval)

    metrics = lerobot_eval.eval_one(FakeVectorEnv(), n_episodes=52)

    assert metrics["init_states"][:10] == [0, 1, 2, 3, 4, 10, 6, 7, 8, 9]
    assert metrics["init_states"][50:] == [0, 1]  # the counters wrap around the 50 stored states
    assert len(metrics["init_states"]) == len(metrics["successes"])


def test_without_stored_start_states_there_is_nothing_to_record() -> None:
    # `--env.init_states=false`: LIBERO then places the objects itself.
    env = SimpleNamespace(call=lambda name: (0, 1) if name == "init_state_id" else (None, None))

    def eval_one(env: Any, **kwargs: Any) -> dict[str, Any]:
        lerobot_eval.rollout(env=env)
        return {"successes": [False, False]}

    lerobot_eval = SimpleNamespace(rollout=lambda env: None, eval_one=eval_one)
    reference_eval.record_start_states(lerobot_eval)

    assert lerobot_eval.eval_one(env)["init_states"] == [None, None]


def test_lerobot_evals_flags_pass_through_and_simulation_stays_in_this_process() -> None:
    flags = ["--policy.path=lerobot/xvla-libero", "--seed=1000"]

    # LeRobot's default, worker processes, is out of the compatibility layer's reach.
    # Finished environments are frozen, as LeRobot freezes them in its worker processes.
    in_process = [*flags, "--eval.use_async_envs=false"]
    assert reference_eval.lerobot_eval_arguments(flags) == (in_process, True)
    # The launcher's own flag does not go on to lerobot-eval.
    assert reference_eval.lerobot_eval_arguments([*flags, "--no-freeze"]) == (in_process, False)
    # Whoever asks for worker processes gets them.
    asked = [*flags, "--eval.use_async_envs=true"]
    assert reference_eval.lerobot_eval_arguments(["--no-freeze", *asked]) == (asked, False)
