"""Bounded, observed-state recovery. No object poses, simulator handles or resets.

Absolute targets stay in the policy's command coordinates, including any
belief-to-world transform applied by action faults. Relative control uses
observed end-effector position and LIBERO's 5 cm/unit translation scale.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp

from fleetlab.contracts.live import Decision
from fleetlab.contracts.policy import Action7, ControlMode

POSITION_TOLERANCE = 0.03
MAX_RETURN_DISTANCE = 0.35
RETRY_STEPS = 70


@dataclass
class Pregrasp:
    """First tracked grasp approach, never a later close during transport.

    A close above the lowest observed approach is not a grasp-site anchor.
    Only command/proprioception is used. There is no initial-pose fallback.
    The model judges from current images whether the object remains at the grasp site.
    """

    command: Action7 | None = None
    position: npt.NDArray[np.float64] | None = None
    last_open: Action7 | None = None
    last_gripper: float = 0.0
    seen_close: bool = False
    lowest_z: float = float("inf")

    def observe(self, command: Action7, position: npt.NDArray[np.float64]) -> None:
        self.lowest_z = min(self.lowest_z, float(position[2]))
        gripper = float(command[6])
        if gripper < 0:
            self.last_open = command.copy()
        elif self.last_gripper < 0 and not self.seen_close:
            self.seen_close = True
            if (
                self.last_open is not None
                and np.linalg.norm(self.last_open[:3] - position) <= POSITION_TOLERANCE
                and max(float(self.last_open[2]), float(position[2]))
                <= self.lowest_z + POSITION_TOLERANCE
            ):
                self.command = self.last_open.copy()
                self.position = position.copy()
        self.last_gripper = gripper

    def available(
        self, mode: ControlMode, last: Action7 | None, position: npt.NDArray[np.float64]
    ) -> bool:
        return bool(
            mode is ControlMode.absolute
            and self.command is not None
            and self.position is not None
            and max(float(self.command[2]), float(self.position[2]))
            <= min(self.lowest_z, float(position[2])) + POSITION_TOLERANCE
            and last is not None
            and np.linalg.norm(last[:3] - position) <= POSITION_TOLERANCE
            and np.linalg.norm(self.command[:3] - position) <= MAX_RETURN_DISTANCE
        )


@dataclass
class Recovery:
    # Targets derive from canonical policy commands and observed proprioception.
    commands_in_world_frame: ClassVar[bool] = False

    mode: ControlMode
    targets: deque[Action7]

    def __bool__(self) -> bool:
        return bool(self.targets)

    def action(self, eef_pos: npt.NDArray[np.float64]) -> Action7:
        target = self.targets.popleft().copy()
        if self.mode is ControlMode.relative:
            target[:3] = np.clip((target[:3] - eef_pos) / 0.05, -0.3, 0.3)
            target[3:6] = 0
        return target


def primitive(
    decision: Decision,
    mode: ControlMode,
    last: Action7,
    eef_pos: npt.NDArray[np.float64],
    pregrasp: Action7 | None,
) -> Recovery:
    """Open for 10 ticks, lift 8 cm over 16; retry returns via a high waypoint.

    The retained command belongs to the first tracked grasp approach.
    Relative retry is disabled pending validated pose return. All ticks pass
    through the same action faults and simulator as ordinary policy commands.
    """
    if decision not in (Decision.retry, Decision.recover):
        raise ValueError("not a recovery primitive")
    if decision is Decision.retry and (mode is ControlMode.relative or pregrasp is None):
        raise ValueError("retry needs a validated absolute pre-grasp pose")
    pose = np.asarray(last, np.float32).copy()
    if mode is ControlMode.relative:
        pose[:3] = eef_pos
        pose[3:6] = 0
    pose[6] = -1
    targets = [pose.copy() for _ in range(10)]

    def segment(end: Action7, ticks: int) -> None:
        nonlocal pose
        start = pose.copy()
        rotation = Slerp([0, 1], Rotation.from_rotvec(np.stack((start[3:6], end[3:6]))))
        for k in range(1, ticks + 1):
            target = np.asarray(start + (end - start) * (k / ticks), np.float32)
            target[3:6] = rotation(k / ticks).as_rotvec()
            targets.append(target)
        pose = end.copy()

    raised = pose.copy()
    raised[2] += 0.08
    segment(raised, 16)
    if decision is Decision.retry and pregrasp is not None:
        goal = np.asarray(pregrasp, np.float32).copy()
        goal[6] = -1
        high = goal.copy()
        high[2] = max(float(raised[2]), float(goal[2]) + 0.08)
        segment(high, 20)
        segment(goal, 16)
    targets += [pose.copy() for _ in range(8)]
    return Recovery(mode, deque(targets))
