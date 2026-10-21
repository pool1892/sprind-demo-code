"""A scripted in-worker policy for LIBERO: the stand-in for the teleoperator.

The scripted teleoperator (branch spike/teleop-absolute) reads the
live simulator through the environment handle and acts one step at a time. It
was not on main when the engine was built, so the in-worker path is tested
with this policy instead: it reads the end effector's pose through the
handle, opens the gripper, and moves the hand 10 cm up and 5 cm toward the
robot in absolute control, keeping its orientation. Its command depends only
on the live state, so a continuation from a snapshot commands what the
uninterrupted episode commanded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
from fleetlab.contracts.policy import Action7, ActionChunk, ControlMode

LIFT_SPEC = PolicySpec(
    name="scripted-lift",
    revision="1",
    precision="fp32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=1,
    sampling_steps=None,
    image_views=1,
)
TOP_Z = 1.25  # metres above the world origin; LIBERO's table is at about 0.9


class LiftPolicy:
    """In-worker: open the gripper and lift, read from the live simulator every step."""

    spec = LIFT_SPEC
    chunk_length = 1
    needs_noise = False
    noise_shape: tuple[int, ...] | None = None
    noise_dtype: str | None = None
    row_independent = True
    in_worker = True
    commands_in_world_frame = True
    image_views = 1

    def __init__(self) -> None:
        self.done = False  # it never finishes: the step limit ends it
        self.status = "lifting"

    def load(self, device: str, precision: str) -> None:
        pass

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        pass

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[ActionChunk]:
        raise NotImplementedError("an in-worker policy")

    def act_one(self, observation: Any, env: Any) -> Action7:
        from robosuite.utils import transform_utils

        handle = env.handle()
        robot = handle.env.robots[0]
        site = robot.eef_site_id["right"]  # read from the simulator, as the teleoperator does
        pos = np.array(handle.data.site_xpos[site], np.float64)
        rotation = np.array(handle.data.site_xmat[site], np.float64).reshape(3, 3)
        axis_angle = transform_utils.quat2axisangle(transform_utils.mat2quat(rotation))
        target = pos.copy()
        target[2] = min(TOP_Z, pos[2] + 0.01)
        target[0] = pos[0] - 0.002
        action = np.zeros(7, np.float32)
        action[:3] = target
        action[3:6] = axis_angle
        action[6] = -1.0
        return action

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        raise NotImplementedError("an in-worker policy")


SWAY_SPEC = PolicySpec(
    name="scripted-sway",
    revision="1",
    precision="fp32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=30,
    sampling_steps=None,
    image_views=1,
)


class SwayPolicy:
    """A server policy without a model: each chunk sways the hand around where it is.

    Lets the worker layout run LIBERO end to end (requests, padded calls,
    queues, hooks, the log) without loading X-VLA. Deterministic, and a row's
    chunk depends on its own observation only.
    """

    spec = SWAY_SPEC
    chunk_length = 30
    needs_noise = False
    noise_shape: tuple[int, ...] | None = None
    noise_dtype: str | None = None
    row_independent = True
    in_worker = False
    image_views = 1

    def load(self, device: str, precision: str) -> None:
        self.spec = SWAY_SPEC.model_copy(update={"precision": precision})

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[ActionChunk]:
        from robosuite.utils import transform_utils

        out = []
        for observation in batch:
            eef = observation["robot_state"]["eef"]
            pos = np.asarray(eef["pos"], np.float64)
            axis_angle = transform_utils.quat2axisangle(np.asarray(eef["quat"], np.float64))
            actions = np.zeros((self.chunk_length, 7), np.float32)
            for k in range(self.chunk_length):
                actions[k, :3] = pos + 0.03 * np.array([np.sin(k / 5), np.cos(k / 5) - 1, 0.0])
                actions[k, 3:6] = axis_angle
                actions[k, 6] = 1.0 if k >= 15 else -1.0
            out.append(ActionChunk(chunk_index=0, actions=actions))
        return out

    def act_one(self, observation: Any, env: Any) -> Action7:
        raise NotImplementedError("a server policy")

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        return [np.asarray(a, np.float32) for a in chunk.actions]
