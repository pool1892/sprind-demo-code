"""The two faults of the walking skeleton and the places where they hook into an episode.

A fault is described by a small dictionary, for example
{"kind": "scene_camera_pan", "degrees": 8.0}. Fault descriptions live only in
the manifests and in the truth directory; nothing here is written to the log
the agent sees.

Two hook points were needed:

  apply_cell(env) / undo_cell(env)   change the simulated cell. Called on the live
                                     MuJoCo model after the simulator exists and
                                     before the reset renders its first image, and
                                     undone after the episode, because with soft
                                     resets a model change outlives the episode.
  on_action(action)                  sits between the policy's action queue and
                                     env.step: receives the commanded action and
                                     returns the one that is executed.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

import xs_common as C

class Fault:
    kind = "none"

    def apply_cell(self, env) -> None:
        pass

    def undo_cell(self, env) -> None:
        pass

    def new_episode(self) -> None:
        pass

    def on_action(self, action: np.ndarray) -> np.ndarray:
        return action


class SceneCameraPan(Fault):
    """The fixed scene camera is turned about the vertical axis, as if its mount had been knocked."""

    kind = "scene_camera_pan"

    def __init__(self, degrees: float):
        self.degrees = float(degrees)
        self._saved = None

    def apply_cell(self, env) -> None:
        m = C.sim_of(env).model._model
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, C.SCENE_CAMERA)
        assert cam >= 0 and m.cam_bodyid[cam] == 0, "the scene camera is expected to hang in the world frame"
        assert self._saved is None, "fault applied twice without undo"
        self._saved = (cam, m.cam_quat[cam].copy())
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), np.deg2rad(self.degrees))
        new = np.zeros(4)
        mujoco.mju_mulQuat(new, turn, m.cam_quat[cam])
        m.cam_quat[cam] = new

    def undo_cell(self, env) -> None:
        if self._saved is not None:
            cam, quat = self._saved
            C.sim_of(env).model._model.cam_quat[cam] = quat
            self._saved = None


class ActionDelay(Fault):
    """Every commanded action is executed `steps` control steps late.

    Until the line has filled, the first command of the episode is executed.
    """

    kind = "action_delay"

    def __init__(self, steps: int):
        self.steps = int(steps)
        self.line: deque = deque()

    def new_episode(self) -> None:
        self.line = deque()

    def on_action(self, action: np.ndarray) -> np.ndarray:
        self.line.append(action)
        if len(self.line) > self.steps:
            return self.line.popleft()
        return self.line[0]


def make(spec: dict) -> Fault:
    kind = spec["kind"]
    if kind == "scene_camera_pan":
        return SceneCameraPan(spec["degrees"])
    if kind == "action_delay":
        return ActionDelay(spec["steps"])
    raise ValueError(f"unknown fault kind {kind!r}")


def active(specs: list[dict]) -> list[Fault]:
    """Fault objects for an episode's fault set; a severity of zero is no fault."""
    out = []
    for s in specs or []:
        if s.get("degrees", 1) == 0 or s.get("steps", 1) == 0:
            continue
        out.append(make(s))
    return out
