"""An overloaded controller that falls behind: a physical fault of the robot's computer,
as an action hook with a growing command buffer.

The computer that runs the policy and the control stack misses control
cycles now and then (an aging board throttled by heat, a background process
competing for the processor). On a missed cycle the drives repeat the last
command, and the new command waits in the controller's buffer behind the
ones not yet executed. The buffer never catches up, so every miss adds a
step of delay: the episode starts on time and falls further behind as it
goes, by `level` steps per step on average (the level is the chance of a
missed cycle). This is the "service controller" fix that the experimental design
names for a command delay: the same symptom as a longer
`control.action_delay_steps`, but growing within each episode and with no
entry in the change log. The misses are drawn per step from the episode's
own generator for this fault, so an episode replays bit for bit and a
snapshot carries the buffer and the generator. `service` or `replace` of the
controller removes it. The health checks do not see it
(`controller.cpu_load` varies benignly only).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    DosePoint,
    NoParams,
    component_truth,
    default_points,
    episode_rng,
)

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7


@dataclass
class OverloadState(FaultState):
    rng: np.random.Generator | None = None
    line: deque[Any] = field(default_factory=deque)
    last: Any = None  # the command the drives executed last
    missed: int = 0


@register("controller_overload")
class ControllerOverload(FaultTypeBase):
    """Missed control cycles repeat the last command and back up the buffer (action hook).

    Level: the chance that a control cycle is missed, 0 to 0.5.
    """

    kind = "controller_overload"
    fault_class = FaultClass.physical
    Params = NoParams
    dose_levels = (0.0, 0.02, 0.05, 0.1, 0.2)
    truth = component_truth({Verb.service, Verb.replace_})
    calibration: tuple[DosePoint, ...] = default_points(
        (0.02, 0.05, 0.1, 0.2), Component.controller.value, unit=" cycles missed"
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.controller:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        if spec.level > 0.5:
            raise ValueError(f"fault {spec.fault_id}: at most half the cycles are missed")
        return OverloadState(
            spec=spec,
            episode_id=identity.episode_id,
            rng=episode_rng(identity, spec, "controller-cycles"),
        )

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, OverloadState) and state.rng is not None
        state.line.append(action)
        missed = bool(state.rng.random() < state.spec.level)
        if missed and state.last is not None:
            state.missed += 1
            out: Action7 = state.last
            return out
        out = state.line.popleft()
        state.last = out
        return out
