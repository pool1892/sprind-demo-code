"""Lets LeRobot's own LIBERO wrapper run on robosuite 1.5, without changing its files.

`lerobot/envs/libero.py` is written against robosuite 1.4, where a robot has
one `controller`. It uses two things on it: `use_delta`, to choose relative or
absolute control, and `ee_ori_mat`, the end effector's orientation. robosuite
1.5 has one controller per body part and other names for both.

`install()` gives robosuite's robot class a `controller` that answers under
the old names. Nothing else is touched: the wrapper, `lerobot-eval` and
robosuite's own code run as they are.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.sim.libero_env import arm_controller, end_effector_rotation, set_relative_control


class Robosuite14Controller:
    """A robot's arm controller under the names robosuite 1.4 gave it."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    @property
    def use_delta(self) -> bool:
        return bool(arm_controller(self._robot).input_type == "delta")

    @use_delta.setter
    def use_delta(self, relative: bool) -> None:
        set_relative_control(self._robot, relative)

    @property
    def ee_ori_mat(self) -> npt.NDArray[np.float64]:
        return end_effector_rotation(self._robot)


def install() -> None:
    """Call before LeRobot builds an environment. Calling it again changes nothing."""
    from robosuite.robots.fixed_base_robot import FixedBaseRobot  # the class of LIBERO's arms

    FixedBaseRobot.controller = property(Robosuite14Controller)
