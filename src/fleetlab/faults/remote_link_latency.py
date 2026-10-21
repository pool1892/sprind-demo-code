"""A site whose robots are driven over a remote link: a lasting site condition that no
action removes, as an action hook with its own delay line (docs/design/loop.md, section 7).

The site's customer runs the policy servers in its own data centre, so every
command crosses a wide-area link and reaches the robot `level` control steps
after the policy issued it. That is the same physics as a long command buffer
(`action_latency`), but it is not a setting of the fleet: the delay line here
is its own, outside every configuration key, so `set` or `revert` of
`control.action_delay_steps` (which runs a separate delay line in series),
any other catalogue action and any setting leave it in force. Its truth
mapping names the controller component of each robot at the site (a
component fault is one fault per robot) with no fixing verb: servicing or
replacing the robot's own computer does not move the customer's servers.

What an operator knows goes into the site's public description
(`SITE_NOTE`, through `records.site_notes`): that the robots are driven over a
remote link, and that this stays. Neither its size nor its effect is written
anywhere visible, and it writes no change and moves no health reading.

What the fleet can do is adapt: `control.latency_compensation_steps` skips the
stale head of each chunk after an attempt's first (the engine's cut,
`engine.episode.call_policy`), and `policy.actions_executed_per_chunk` sets how
often the policy replans. Both act on this delay line's input like on any
command stream.

Until the line has filled, the episode's first command is executed, as for
`action_latency`; with the same level the two give the same executed actions
bit for bit (the delay-only cells of `fleetlab.adaptation` repeat the
calibration's `action_latency` episodes chain for chain). A real link would
deliver nothing in those first steps; since the first chunk is never
compensated, the warm-up is the same with or without the setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.contracts.faults import Direction, FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass
from fleetlab.faults import register
from fleetlab.faults._common import DosePoint, NoParams, component_truth, default_points
from fleetlab.faults.action_latency import LatencyState

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7

# What the site's public description says, for every site with the condition; it names
# neither the delay nor its effect.
SITE_NOTE = (
    "The robots at this site are driven by policy servers in the customer's own data "
    "centre, over a wide-area network link. This is the customer's requirement and stays."
)


@register("remote_link_latency")
class RemoteLinkLatency(FaultTypeBase):
    """Every command reaches the robot `level` control steps late (action hook).

    No verb fixes it (an empty truth mapping): the condition lasts. Level: the
    delay in whole control steps of 50 ms.
    """

    kind = "remote_link_latency"
    fault_class = FaultClass.site
    Params = NoParams
    dose_levels = (0.0, 8.0, 14.0, 17.0, 20.0)
    truth = component_truth(set(), frozenset({Direction.changed}))
    calibration: tuple[DosePoint, ...] = default_points(
        (8.0, 14.0, 17.0, 20.0), Component.controller.value, unit=" steps"
    )
    site_note = SITE_NOTE
    # The same delay line as `action_latency`, with the same point labels: its curve.
    curve_kind = "action_latency"

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        if spec.target.component is not Component.controller:
            raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {spec.target}")
        steps = round(spec.level)
        if steps != spec.level:
            raise ValueError(f"fault {spec.fault_id}: a delay is a whole number of steps")
        return LatencyState(spec=spec, episode_id=identity.episode_id, steps=steps)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, LatencyState)
        state.line.append(action)
        if len(state.line) > state.steps:
            out: Action7 = state.line.popleft()
            return out
        first: Action7 = state.line[0]
        return first


#
