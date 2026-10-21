"""Supervisors for the engine's tests: one that never intervenes and a timeout rule.

The live layer's supervisor is stage 5. The engine needs the
hook and two stand-ins: `noop` (the risk estimator never fires) and `timeout`
(fires once the episode has run `after` steps and then decides `decision`,
taking `decision_ms` to do so, during which the robot holds still).

Both are deterministic, so a supervised episode replays exactly even without
its decision trace; a real supervisor's latency is not, which is why the
trace exists ("scripted" jobs).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fleetlab.contracts.live import (
    Decision,
    StepTelemetry,
    SupervisorContext,
    SupervisorDecision,
)
from fleetlab.engine.episode import SupervisorParts


class NeverFires:
    def update(self, telemetry: StepTelemetry) -> float:
        return 0.0


@dataclass
class AfterSteps:
    """Risk 1 from step `after` on, 0 before: the timeout rule's estimator."""

    after: int

    def update(self, telemetry: StepTelemetry) -> float:
        return 1.0 if telemetry.step >= self.after else 0.0


class ContinueAlways:
    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        return SupervisorDecision(Decision.continue_, "no-op supervisor", 0.0, 0.0)


@dataclass
class Fixed:
    """Always the same decision, with a fixed decision time."""

    decision: Decision
    decision_ms: float

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        return SupervisorDecision(
            self.decision, f"timeout rule: {self.decision.value}", self.decision_ms, 0.0
        )


# The supervisors here read no pixels; any other (stage 5's) counts as reading every frame,
# so that a frame differing in a chain-sample replay of a job it supervised is never
# taken for an unconsumed render transient (docs/design/contracts.md section 4).
PIXEL_FREE = frozenset({"noop", "timeout"})


def reads_pixels(name: str) -> bool:
    """Whether the supervisor named `name` may look at the camera images."""
    return name not in PIXEL_FREE


def make(name: str, params: Mapping[str, Any]) -> SupervisorParts:
    """The supervisor named in a job's `SupervisorSpec`."""
    if name == "noop":
        return SupervisorParts(NeverFires(), ContinueAlways())
    if name == "timeout":
        decision = Decision(params.get("decision", "hand_over"))
        return SupervisorParts(
            AfterSteps(int(params.get("after", 200))),
            Fixed(decision, float(params.get("decision_ms", 500))),
            should_call=lambda context: decision in context.options,
        )
    if name == "risk":
        from fleetlab.live.supervisor import make_risk_rule

        return make_risk_rule(params)
    if name == "vlm":
        from fleetlab.live.vlm import make_vlm

        return make_vlm(params)
    raise KeyError(f"no supervisor named {name!r}; known: noop, timeout, risk, vlm")
