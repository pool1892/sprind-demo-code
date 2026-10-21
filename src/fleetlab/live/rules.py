"""Claim-4 rule adapters, recorded calls and common FIFO operator admission."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fleetlab.contracts.identity import started_at_for
from fleetlab.contracts.live import (
    Decision,
    DecisionRecord,
    OperatorQueue,
    StepTelemetry,
    Supervisor,
    SupervisorCallFailure,
    SupervisorContext,
    SupervisorDecision,
)
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SnapshotRef
from fleetlab.engine.episode import SupervisorParts
from fleetlab.policies.teleop.human import Admission, HumanSettings, admit, arrival_tick


@dataclass(frozen=True)
class CostRule:
    """Failure cost / reserved operator cost, both in equivalent seconds.

    Risk is the estimator's failure probability within its horizon. The
    avoidable fraction is an explicit assumption, not an oracle outcome.
    Queue idle is priced separately; it is never charged as operator use.
    """

    ratio: float
    service_s: float = 90.0
    idle_cost: float = 0.0
    avoidable_fraction: float = 1.0
    wait_s: float = 0.0

    def __post_init__(self) -> None:
        values = (self.ratio, self.service_s, self.idle_cost, self.avoidable_fraction, self.wait_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("cost parameters must be finite")
        if self.ratio <= 0 or self.service_s <= 0 or self.idle_cost < 0 or self.wait_s < 0:
            raise ValueError("costs and waits must be nonnegative; ratio and service positive")
        if not 0 <= self.avoidable_fraction <= 1:
            raise ValueError("avoidable_fraction must be in [0, 1]")

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        if not math.isfinite(context.risk) or not 0 <= context.risk <= 1:
            raise ValueError("risk must be a probability")
        if Decision.retry in context.options:
            raise ValueError("cost gate requires autonomous options exhausted or inapplicable")
        expected = context.risk * self.avoidable_fraction * self.ratio * self.service_s
        cost = self.service_s + self.idle_cost * self.wait_s
        decision = (
            Decision.hand_over
            if expected > cost and Decision.hand_over in context.options
            else Decision.continue_
        )
        return SupervisorDecision(
            decision,
            f"expected avoidable failure cost {expected:g}s; operator/idle cost {cost:g}s; "
            f"queue {context.operator_queue.waiting}, free {context.operator_queue.free_operators}",
            0.0,
            0.0,
        )


@dataclass(frozen=True)
class QueuePolicy:
    """A queue limit shared by every rule; no estimator or cost ratio gates requests."""

    max_wait_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_wait_s is not None and (
            not math.isfinite(self.max_wait_s) or self.max_wait_s < 0
        ):
            raise ValueError("maximum queue wait must be finite and nonnegative")


def start_tick(job: Job, episodes_per_robot_day: int) -> int:
    hours, minutes, seconds = map(
        int, started_at_for(job.identity, episodes_per_robot_day).split(":")
    )
    return (job.identity.day * 86400 + hours * 3600 + minutes * 60 + seconds) * 20


@dataclass
class QueueView:
    """Queue view from the harness's completed requests, clipped to the call's fleet time.

    Proposal jobs are executed in fixed fleet-start order with one worker. No
    worker completion order, labels, future requests, or truth enter this view.
    """

    layout: RunLayout
    job: Job
    settings: HumanSettings
    per_day: int
    policy: QueuePolicy
    cost_ratio: float = 2.0

    def context(self, context: SupervisorContext) -> SupervisorContext:
        return replace(
            context,
            operator_queue=self(context),
            task_text=context.task_text
            + "\nFailure-cost / reserved operator-cost ratio: "
            + str(self.cost_ratio),
        )

    def __call__(self, context: SupervisorContext) -> OperatorQueue:
        tick = start_tick(self.job, self.per_day) + context.telemetry_window[-1].step + 1
        requests = [j for j in self.layout.continuations() if arrival_tick(j, self.per_day) <= tick]
        admitted, _, _ = gate_requests(requests, {}, self.settings, self.per_day, self.policy)
        slots = [Admission.model_validate(j.teleoperator_params) for j in admitted]
        waiting = sum(p.start_tick > tick for p in slots if p is not None)
        occupied = {
            p.operator
            for p in slots
            if p is not None and p.start_tick <= tick < p.start_tick + self.settings.service_steps
        }
        return OperatorQueue(waiting, self.settings.operators - len(occupied))


def gate_requests(
    jobs: Sequence[Job],
    risks: Mapping[str, float],
    settings: HumanSettings,
    episodes_per_robot_day: int,
    rule: QueuePolicy | None,
) -> tuple[list[Job], list[Job], list[dict[str, Any]]]:
    """Decide in fleet-time order, reserving only accepted requests.

    State comes from previously admitted reservations, independent of worker
    order. Admissions are checked against the existing human model's FIFO.
    """
    free = [0] * settings.operators
    reservations: list[tuple[int, int]] = []
    accepted: list[Job] = []
    denied: list[Job] = []
    events: list[dict[str, Any]] = []
    for job in sorted(
        jobs, key=lambda j: (arrival_tick(j, episodes_per_robot_day), j.identity.episode_id)
    ):
        tick = arrival_tick(job, episodes_per_robot_day)
        operator = min(range(settings.operators), key=lambda i: (free[i], i))
        start = max(tick, free[operator])
        queue = OperatorQueue(
            waiting=sum(a > tick for a, _ in reservations),
            free_operators=sum(t <= tick for t in free),
        )
        wait_s = (start - tick) / 20
        taken = rule is None or rule.max_wait_s is None or wait_s <= rule.max_wait_s
        decision = SupervisorDecision(
            Decision.hand_over if taken else Decision.continue_,
            "FIFO reservation" if taken else "queue wait exceeds maximum",
            0,
            0,
        )
        events.append(
            {
                "episode_id": job.identity.episode_id,
                "arrival_tick": tick,
                "waiting": queue.waiting,
                "free_operators": queue.free_operators,
                "proposed_queue_s": wait_s,
                "risk": risks.get(job.identity.episode_id),
                "decision": decision.decision.value,
                "rationale": decision.rationale,
            }
        )
        if taken:
            accepted.append(job)
            free[operator] = start + settings.service_steps
            reservations.append((start, free[operator]))
        else:
            denied.append(job)
    admitted = admit(accepted, settings, episodes_per_robot_day)
    for job in admitted:
        assert job.teleoperator_params is not None
        event = next(e for e in events if e["episode_id"] == job.identity.episode_id)
        if job.teleoperator_params["queue_steps"] != round(event["proposed_queue_s"] * 20):
            raise ValueError("economic queue and human admission disagree")
    return admitted, denied, events


@dataclass
class RecordingSupervisor:
    inner: Supervisor
    path: Path
    latency_route: str = "subscription"
    queue_provider: Any = None

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        if self.queue_provider is not None:
            context = self.queue_provider.context(context)
        step = context.telemetry_window[-1].step + 1
        event: dict[str, Any] = {
            "step": step,
            "risk": context.risk,
            "waiting": context.operator_queue.waiting,
            "free_operators": context.operator_queue.free_operators,
        }
        try:
            answer = self.inner.decide(context)
        except SupervisorCallFailure as failure:
            event.update(
                decision="hand_over",
                rationale=str(failure),
                decision_ms=failure.decision_ms,
                model_ms=failure.model_ms,
                call_status="failed" if failure.called else "skipped",
                error=str(failure),
            )
            self.record(event)
            if self.latency_route == "model":
                if failure.model_ms is None:
                    raise ValueError("model route requires an available model timer") from failure
                raise SupervisorCallFailure(
                    str(failure), failure.model_ms, failure.model_ms, called=failure.called
                ) from failure
            raise
        requested = answer.decision is Decision.hand_over
        if requested and self.queue_provider is not None:
            view = self.queue_provider
            physical_ms = answer.model_ms if self.latency_route == "model" else answer.decision_ms
            step_after_hold = step + max(1, math.ceil(physical_ms / 50))
            hypothetical = Job(
                identity=view.job.identity,
                mode=JobMode.continue_,
                from_snapshot=SnapshotRef(
                    episode_id=view.job.identity.episode_id,
                    step=step_after_hold,
                    path="queue-probe.pkl",
                ),
            )
            requests = [*view.layout.continuations(), hypothetical]
            _, denied, _ = gate_requests(requests, {}, view.settings, view.per_day, view.policy)
            if any(j.identity.episode_id == view.job.key for j in denied):
                answer = answer._replace(
                    decision=Decision.continue_,
                    rationale=answer.rationale + "; queue refused; rule retained",
                )
        event["requested"] = requested
        event["refused"] = requested and answer.decision is not Decision.hand_over
        event.update(
            decision=answer.decision.value,
            rationale=answer.rationale,
            decision_ms=answer.decision_ms,
            model_ms=answer.model_ms,
            call_status="ok",
        )
        self.record(event)
        return (
            answer._replace(decision_ms=answer.model_ms)
            if self.latency_route == "model"
            else answer
        )

    def record(self, event: dict[str, Any]) -> None:
        previous = json.loads(self.path.read_text()) if self.path.exists() else []
        previous.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(previous, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)


@dataclass
class PrefixRisk:
    steps: set[int]

    def update(self, telemetry: StepTelemetry) -> float:
        return float(telemetry.step + 1 in self.steps)


@dataclass
class PrefixSupervisor:
    """Replay a recorded prefix and request a snapshot before its last primitive."""

    trace: list[DecisionRecord]

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        step = context.telemetry_window[-1].step + 1
        record = next(r for r in self.trace if r.step == step)
        decision = Decision.hand_over if record is self.trace[-1] else record.decision
        return SupervisorDecision(
            decision,
            "recorded counterfactual prefix",
            record.decision_ms,
            record.model_ms if record.model_ms is not None else 0.0,
        )


@dataclass
class PointRisk:
    step: int
    risk: float

    def update(self, telemetry: StepTelemetry) -> float:
        return self.risk if telemetry.step + 1 == self.step else 0.0


class RequestOperator:
    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        return SupervisorDecision(Decision.hand_over, "recorded economic threshold", 0, 0)


@dataclass
class HoldSupervisor:
    trace: list[DecisionRecord]

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        step = context.telemetry_window[-1].step + 1
        record = next(r for r in self.trace if r.step == step)
        return SupervisorDecision(
            Decision.continue_,
            "hold-only control; no model call",
            record.decision_ms,
            record.model_ms or 0.0,
        )


def make(name: str, params: Mapping[str, Any]) -> SupervisorParts:
    """A factory that accepts any engine-compatible Supervisor factory.

    The nested factory delivers SupervisorParts; its supervisor implements
    contracts.live.Supervisor. No contract or other builder's adapter changes.
    """
    if name == "economic_point":
        risk = float(params["risk"])
        return SupervisorParts(PointRisk(int(params["step"]), risk), RequestOperator(), risk)
    if name == "holds":
        trace = [DecisionRecord.model_validate(r) for r in params["trace"] if r["hold_steps"] > 0]
        return SupervisorParts(PrefixRisk({r.step for r in trace}), HoldSupervisor(trace))
    if name == "prefix":
        trace = [DecisionRecord.model_validate(r) for r in params["trace"]]
        if not trace or len({r.step for r in trace}) != len(trace):
            raise ValueError("a prefix needs distinct recorded decision steps")
        return SupervisorParts(PrefixRisk({r.step for r in trace}), PrefixSupervisor(trace))
    if name != "claim4":
        raise KeyError(name)
    module, _, symbol = str(params["factory"]).partition(":")
    parts: SupervisorParts = getattr(importlib.import_module(module), symbol)(
        params["inner"]["name"], params["inner"]["params"]
    )
    path = Path(params["events"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]\n")  # a retried incomplete job replaces its abandoned decision calls
    view = None
    if params.get("queue") is not None:
        q = params["queue"]
        view = QueueView(
            RunLayout(Path(q["fleet"])),
            Job.model_validate(q["job"]),
            HumanSettings.model_validate(q["settings"]),
            q["per_day"],
            QueuePolicy(q["max_wait_s"]),
            q.get("cost_ratio", 2.0),
        )
    should_call = parts.should_call
    if view is not None and should_call is not None:
        queue_view, predicate = view, should_call

        def observe_queue(context: SupervisorContext) -> bool:
            return predicate(queue_view.context(context))

        should_call = observe_queue
    return replace(
        parts,
        supervisor=RecordingSupervisor(parts.supervisor, path, params["latency_route"], view),
        should_call=should_call,
    )


def reads_pixels(name: str) -> bool:
    # Be conservative for externally supplied supervisors.
    return True
