"""The round decision's public checks and the experiment budget's status (tool server side).

`decision_problems` checks a `RoundDecision` against what an operator can see: the
public fleet (scopes, components, value domains, recorded changes for `revert`, a
simulated effect for every change), the cap of changes per round, and the fleet's
ledger as the runtime publishes it for the round (`results/ledger.json`,
`PublicLedger`): only an active entry can be replaced or undone, component work cannot
be taken back, and nothing already in force is adopted again. The ranked alternatives
are optional, but each one supplied is checked like an adoption (complete, applicable,
simulated), because the plan fixed after one look may adopt it. It never predicts a
benefit.

`budget_status` is the one computation of what a session's requests have charged
against the budget in force; `request_experiment` refuses beyond it and
`experiment_budget` reports it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.actions import Action
from fleetlab.contracts.decisions import (
    MAX_ADOPTED_PER_ROUND,
    ExperimentBudgetStatus,
    LedgerEntryId,
    RoundDecision,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
from fleetlab.contracts.vocabulary import COMPONENT_VERBS

__all__ = [
    "DECISION_FILE",
    "LEDGER_FILE",
    "PublicLedger",
    "PublicLedgerEntry",
    "action_problems",
    "budget_status",
    "decision_problems",
    "read_public_ledger",
    "write_public_ledger",
]

# Written by the runtime into the analysis' results directory (read-only for the agent).
LEDGER_FILE = "ledger.json"
# The latest accepted decision in the outbox (every one is also in submissions.jsonl).
# Not "decision.json": a single-word hidden column of the supervisor's truth table is
# called decision, and the strict audit reads file names as labels.
DECISION_FILE = "round_decision.json"
LEDGER_FORMAT: Literal["fleetlab.public-ledger/2"] = "fleetlab.public-ledger/2"


class PublicLedgerEntry(ContractModel):
    """One adopted change as the fleet's operators know it: no word on whether it helped."""

    entry_id: LedgerEntryId
    round: int
    adopted_day: int
    action: Action
    status: Literal["active", "replaced", "undone"]
    replaced_by: LedgerEntryId | None = None


class PublicLedger(ContractModel):
    """The fleet's ledger of adopted changes for one round's sessions. The list is called
    `changes`, not entries: that word is a truth-side key, and the strict audit reads the
    keys of this runtime-written file as labels."""

    format: Literal["fleetlab.public-ledger/2"] = LEDGER_FORMAT
    round: int
    changes: list[PublicLedgerEntry]

    def active(self) -> set[str]:
        return {e.entry_id for e in self.changes if e.status == "active"}


def read_public_ledger(results: Path) -> PublicLedger | None:
    path = results / LEDGER_FILE
    if not path.is_file():
        return None
    return PublicLedger.model_validate_json(path.read_text())


def _robots_covered(action: Action, fleet: Fleet) -> list[str]:
    return [r.id for r in fleet.robots() if action.target.covers(r.id, r.site_id)]


def action_problems(action: Action, fleet: Fleet) -> list[str]:
    """The public experiment preflight of one action, as if tested on every robot it covers."""
    from fleetlab.toolserver.action_validation import request_problems

    robots = _robots_covered(action, fleet)
    if not robots:
        return [f"{action.verb.value} at {action.target.scope}: covers no robot of this fleet"]
    request = ExperimentRequest(
        arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[action])],
        robots=robots,
        episodes_per_arm=1,
        paired_starts=True,
        power_note="decision check",
    )
    return [
        f"{action.verb.value} at {action.target.scope}: {p.removeprefix('arm 1: ')}"
        for p in request_problems(request, fleet)
    ]


def decision_problems(
    decision: RoundDecision, fleet: Fleet, ledger: PublicLedger | None
) -> list[str]:
    """Why a decision cannot be adopted as submitted; empty when it can."""
    problems: list[str] = []
    for change in decision.adopt:
        problems += action_problems(change.action, fleet)
    entries = {e.entry_id: e for e in ledger.changes} if ledger is not None else {}
    for replacement in decision.replace:
        problems += action_problems(replacement.action, fleet)
    named = [r.entry_id for r in decision.replace] + [u.entry_id for u in decision.undo]
    for entry_id in named:
        entry = entries.get(entry_id)
        if entry is None:
            problems.append(f"no ledger entry {entry_id}; the task lists the fleet's entries")
        elif entry.status != "active":
            problems.append(f"ledger entry {entry_id} is {entry.status}, no longer in force")
        elif entry.action.verb in COMPONENT_VERBS:
            problems.append(
                f"ledger entry {entry_id} ({entry.action.verb.value}) is physical work on a "
                "component and cannot be taken back or replaced"
            )
    staying = {
        e.action.model_dump_json()
        for e in entries.values()
        if e.status == "active" and e.entry_id not in named
    }
    for action in [c.action for c in decision.adopt] + [r.action for r in decision.replace]:
        if action.model_dump_json() in staying:
            problems.append(
                f"{action.verb.value} at {action.target.scope} is already in force in the ledger"
            )
    if len(decision.adopt) + len(decision.replace) > MAX_ADOPTED_PER_ROUND:
        problems.append(
            f"a decision makes at most {MAX_ADOPTED_PER_ROUND} changes (adopt and replace "
            "together; an undo counts nothing)"
        )
    # The alternatives are optional, but every one supplied passes the same public checks
    # as an adoption: the one-look plan may adopt it, so it must be adoptable as written.
    for rank, candidate in enumerate(decision.ranked_candidates, start=1):
        action = candidate.action
        if not action.is_complete:
            problems.append(
                f"ranked candidate {rank} ({action.verb.value} at {action.target.scope}) "
                "needs its value"
            )
            continue
        problems += [f"ranked candidate {rank}: {p}" for p in action_problems(action, fleet)]
    return list(dict.fromkeys(problems))


def _arm_attempts(request: Mapping[str, Any]) -> int:
    return int(request["episodes_per_arm"]) * len(request["arms"])


def _probe_attempts(request: Mapping[str, Any]) -> int:
    return len(request["robots"]) * int(request["repeats"])


def budget_status(
    limits: Any,
    requests: Iterable[Mapping[str, Any]],
    refused: set[str],
    *,
    probes: Iterable[Mapping[str, Any]] = (),
    probe_outcomes: Mapping[str, Any] | None = None,
    probes_offered: bool = False,
) -> ExperimentBudgetStatus:
    """What the requests of the budget's period have charged: refused ones cost nothing,
    and requests of sessions before `limits.budget_first_session` do not count. Probe
    requests (`outbox/probes.jsonl`) are charged to the same attempts, one per probe
    episode, and reported apart from the experiments' task attempts: a queued probe its
    requested episodes, one with an outcome (`contracts.probes.ProbeOutcome`) its executed
    and reserved attempts; a probe refused before it ran costs nothing and uses no request.
    """
    outcomes = probe_outcomes or {}

    def counts(record: Mapping[str, Any]) -> bool:
        return int(record.get("session", 1)) >= int(limits.budget_first_session)

    def probe_charge(record: Mapping[str, Any]) -> int:
        outcome = outcomes.get(record["probe_id"])
        if outcome is None:
            return _probe_attempts(record["request"])
        return int(outcome.attempts_executed) + int(outcome.attempts_reserved)

    charged = [r for r in requests if r["experiment_id"] not in refused and counts(r)]
    charged_probes = [
        p
        for p in probes
        if counts(p) and getattr(outcomes.get(p["probe_id"]), "status", "queued") != "refused"
    ]
    tasks = sum(_arm_attempts(r["request"]) for r in charged)
    probe_attempts = sum(probe_charge(p) for p in charged_probes)
    used = tasks + probe_attempts
    return ExperimentBudgetStatus(
        attempts=limits.max_experiment_episodes,
        attempts_used=used,
        attempts_left=max(0, limits.max_experiment_episodes - used),
        max_per_arm=limits.max_episodes_per_arm,
        requests=limits.max_experiment_requests,
        requests_used=len(charged),
        counted_from_session=limits.budget_first_session,
        task_attempts_used=tasks,
        probe_attempts_used=probe_attempts,
        probes=getattr(limits, "max_probe_requests", 0) if probes_offered else 0,
        probes_used=len(charged_probes),
    )


def write_public_ledger(results: Path, ledger: PublicLedger) -> None:
    """The runtime's side: publish the ledger for the round's sessions."""
    (results / LEDGER_FILE).write_text(json.dumps(ledger.model_dump(mode="json"), indent=1))
