"""A fault type that acts through other fault types: what a software release carries,
or what a generic change of a key or a condition amounts to.

`CarrierBase.inner_specs` turns the fault into the faults whose hooks do the
work; every hook runs those faults' hooks in their order, and the cell hook's
undo undoes them in reverse. The inner faults share the outer fault's id and
scope and are never listed anywhere: the identity, the log and the oracle see
only the outer fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.faults import get
from fleetlab.faults.command import in_policy_frame

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Observation, SuccessVerdict, Undo
    from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
    from fleetlab.contracts.policy import Action7, ControlMode


@dataclass
class CarrierState(FaultState):
    inner: list[FaultState] = field(default_factory=list)


class CarrierBase(FaultTypeBase):
    """Forwards every hook to the faults `inner_specs` returns. Subclasses define that."""

    def inner_specs(self, spec: FaultSpec, identity: EpisodeIdentity) -> list[FaultSpec]:
        raise NotImplementedError

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        inner = [get(s.kind).new_episode(s, identity) for s in self.inner_specs(spec, identity)]
        return CarrierState(spec=spec, episode_id=identity.episode_id, inner=inner)

    @staticmethod
    def _inner(state: FaultState) -> list[FaultState]:
        assert isinstance(state, CarrierState)
        return state.inner

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        undos: list[Undo] = []
        try:
            for s in self._inner(state):
                undo = get(s.spec.kind).cell_hook(handle, s)
                if undo is not None:
                    undos.append(undo)
        except BaseException:
            for undo in reversed(undos):
                undo()
            raise
        if not undos:
            return None

        def undo_all() -> None:
            for undo in reversed(undos):
                undo()

        return undo_all

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        for s in self._inner(state):
            observation = get(s.spec.kind).observation_hook(observation, s)
        return observation

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        for s in self._inner(state):
            action = get(s.spec.kind).action_hook(action, s)
        return action

    def to_policy_frame(
        self, action: Action7, state: FaultState, control_mode: ControlMode
    ) -> Action7:
        """Invert carried origin transforms in the reverse of their hook order."""
        pairs = [(get(s.spec.kind), s) for s in self._inner(state)]
        return in_policy_frame(action, pairs, control_mode)

    def policy_input_hook(self, policy_spec: PolicySpec, state: FaultState) -> PolicySpec:
        for s in self._inner(state):
            policy_spec = get(s.spec.kind).policy_input_hook(policy_spec, s)
        return policy_spec

    def data_hook(self, verdict: SuccessVerdict, state: FaultState) -> SuccessVerdict:
        for s in self._inner(state):
            verdict = get(s.spec.kind).data_hook(verdict, s)
        return verdict
