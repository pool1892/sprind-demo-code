"""The 7-number action the expert sends, in either of LIBERO's two control modes.

The expert plans every control step as a *unit command* `u`, seven numbers:

    u[0:3]  change of the grip site's position, world frame, in units of
            POS_STEP (0.05 m), each component in [-1, 1]
    u[3:6]  change of the gripper's orientation as a rotation vector, world
            frame, applied on the left (R_next = exp(ROT_STEP * u[3:6]) @ R),
            in units of ROT_STEP (0.5 rad), each component in [-1, 1]
    u[6]    gripper: -1 open, +1 close, 0 hold the present opening

and `to_env_action` turns it into what the environment is sent.

**Absolute control (X-VLA's mode, the default).** The action is a target pose
in world coordinates, the convention X-VLA's actions are converted to by
LeRobot's `XVLARotation6DToAxisAngleProcessorStep`:

    a[0:3]  target position of the grip site, world frame, metres
    a[3:6]  target orientation of the gripper as a rotation vector
            (axis times angle, angle in [0, pi]), world frame
    a[6]    gripper command

The target is the present pose moved by the unit command:
`p + POS_STEP * u[0:3]` and `exp(ROT_STEP * u[3:6]) @ R`. That is the goal
robosuite's operational-space controller computes itself from a relative
command (`input_type="delta"`, goal relative to the achieved pose), so in both
modes the arm is driven toward the same goal; the two differ only in rounding
and in which reading of the present pose the goal is built on (the expert reads
MuJoCo's data as the step left it, the controller recomputes it first; measured
in the README). The fleetlab adapter reads absolute targets in world
coordinates (`set_relative_control(robot, False)`).

**Relative control (the fallback).** The action is the unit command itself,
as LeRobot's relative mode and the expert of spikes/teleop-expert send it.

**Gripper.** robosuite's Panda gripper uses only the sign of `a[6]`: every
step it moves its finger target by 0.2 of the full range towards closed (+),
towards open (-), or not at all (0). X-VLA's converted actions are always -1
or +1. The expert also sends 0, to hold a partial opening while it approaches
a grasp in a tight spot (the drawer task); everything else is -1 or +1.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

ControlMode = Literal["absolute", "relative"]
CONTROL_MODES: tuple[ControlMode, ...] = ("absolute", "relative")

# Metres and radians per unit of a relative command: the output limits of the
# operational-space controller LIBERO configures (robosuite's osc_pose.json,
# input limits +-1). `check_controller` verifies them against the live controller.
POS_STEP = 0.05
ROT_STEP = 0.5

OPEN, CLOSE, HOLD = -1.0, 1.0, 0.0

Vector = npt.NDArray[np.float64]


def to_env_action(u: Any, mode: ControlMode, eef_pos: Vector, eef_mat: Vector) -> Vector:
    """The environment action for unit command `u`, given the grip site's present pose.

    `u` is clipped to [-1, 1] first, as robosuite clips a relative command.
    """
    u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    if mode == "relative":
        return u
    if mode != "absolute":
        raise ValueError(f"unknown control mode {mode!r}")
    target_pos = eef_pos + POS_STEP * u[:3]
    target_mat = Rotation.from_rotvec(ROT_STEP * u[3:6]).as_matrix() @ eef_mat
    return np.concatenate([target_pos, Rotation.from_matrix(target_mat).as_rotvec(), u[6:7]])


def hold_action(mode: ControlMode, eef_pos: Vector, eef_mat: Vector, grip: float = OPEN) -> Vector:
    """Stay where you are, with the given gripper command."""
    return to_env_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip], mode, eef_pos, eef_mat)


def controller_mode(arm_controller: Any) -> ControlMode:
    """The control mode the arm controller is in, as fleetlab's adapter sets it.

    Raises if the controller reads targets in a frame the expert does not
    produce: absolute targets must be read in world coordinates.
    """
    kind, frame = arm_controller.input_type, arm_controller.input_ref_frame
    if kind == "delta":
        return "relative"
    if kind == "absolute" and frame == "world":
        return "absolute"
    raise RuntimeError(f"arm controller reads {kind} targets in the {frame} frame; expected world")


def check_controller(arm_controller: Any) -> None:
    """Fail if the controller scales relative commands differently from POS_STEP and ROT_STEP."""
    out_max = np.asarray(arm_controller.output_max, dtype=np.float64)
    in_max = np.asarray(arm_controller.input_max, dtype=np.float64)
    expected = np.array([POS_STEP] * 3 + [ROT_STEP] * 3)
    if out_max.shape != (6,) or not np.allclose(out_max / in_max, expected):
        raise RuntimeError(
            f"controller output limits {out_max} / input limits {in_max}, expected {expected}"
        )
