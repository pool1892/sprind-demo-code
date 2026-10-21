"""The scripted teleoperator: a privileged pick-and-place expert for LIBERO.

Moved from spikes/teleop-absolute/teleop as it was measured there (999 of
1,000 from the start states in absolute control, 98.4% on six kinds of
disturbed state, 88.6% when taking over from X-VLA's recorded failures),
reformatted only. `fleetlab.policies.teleoperator.Teleoperator` puts it behind
the contracts' in-worker policy.

    from fleetlab.policies.teleop import ScriptedExpert
    expert = ScriptedExpert(robosuite_env, control_mode="absolute")

`teleop.actions` documents the action convention (absolute target poses in
world coordinates as X-VLA sends them, or relative commands), `teleop.expert`
is the controller, `teleop.grasp` its grasp planner and `teleop.scene` its
read-only view of the simulator. Nothing here writes to the simulator or
keeps global state.
"""

from fleetlab.policies.teleop.actions import CONTROL_MODES, ControlMode, to_env_action
from fleetlab.policies.teleop.expert import ScriptedExpert

__all__ = ["CONTROL_MODES", "ControlMode", "ScriptedExpert", "to_env_action"]
