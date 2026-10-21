"""The scripted teleoperator: a privileged pick-and-place expert for LIBERO.

    from teleop import ScriptedExpert
    expert = ScriptedExpert(robosuite_env, control_mode="absolute")   # fleetlab's LiberoEnv: env._env.env

`teleop.actions` documents the action convention (absolute target poses in
world coordinates as X-VLA sends them, or relative commands as the fallback).
`teleop.expert` is the controller, `teleop.grasp` its grasp planner and
`teleop.scene` its read-only view of the simulator. Nothing in this package
writes to the simulator, keeps global state or imports anything outside numpy,
scipy and mujoco, so it can move into `fleetlab` as it is.
"""

import fleetlab  # noqa: F401  (torch before mujoco; teleop.scene imports mujoco)

from .actions import CONTROL_MODES, ControlMode, to_env_action
from .expert import ScriptedExpert

__all__ = ["CONTROL_MODES", "ControlMode", "ScriptedExpert", "to_env_action"]
