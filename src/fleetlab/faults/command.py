"""The command convention in force at the action hooks, including delegated hooks.

This context is scoped to one synchronous command application. It is runtime
provenance, supplied by the engine and command producer, never fault truth.
Direct hook callers default to absolute policy coordinates and can use
``action_context`` for another convention. No context is stored in snapshots.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from fleetlab.contracts.faults import FaultState, FaultType
from fleetlab.contracts.policy import Action7, ControlMode


@dataclass(frozen=True)
class CommandContext:
    control_mode: ControlMode = ControlMode.absolute
    world_frame: bool = False


DEFAULT_COMMAND = CommandContext()

CURRENT_COMMAND: ContextVar[CommandContext] = ContextVar(
    "fleetlab_command", default=DEFAULT_COMMAND
)


@contextmanager
def action_context(control_mode: ControlMode, *, world_frame: bool = False) -> Iterator[None]:
    token = CURRENT_COMMAND.set(CommandContext(control_mode, world_frame))
    try:
        yield
    finally:
        CURRENT_COMMAND.reset(token)


def in_policy_frame(
    action: Action7, pairs: Sequence[tuple[FaultType, FaultState]], control_mode: ControlMode
) -> Action7:
    """Apply explicitly declared inverse command-frame transforms before buffering.

    An observation error is not necessarily an origin change. Only mechanisms
    that translate command frames supply ``to_policy_frame``. Inverting those
    transforms in reverse order puts expert and saved policy targets in the
    same convention, without inspecting fault truth or changing delay buffers.
    """
    for ft, state in reversed(pairs):
        convert = getattr(ft, "to_policy_frame", None)
        if convert is not None:
            action = np.asarray(convert(action, state, control_mode), action.dtype)
    return action
