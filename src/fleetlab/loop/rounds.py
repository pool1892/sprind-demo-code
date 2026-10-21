"""Rounds of the improvement loop (docs/design/loop.md, section 8): the engagement driver.

An engagement is one analysis of one fleet over several rounds. Round `n`:

1. **Operation.** The fleet does its regular work on the round's operation days
   under the ledger as it stands (lane state's `operate`; round 1 of an existing
   world is the as-run observed window).
2. **Export.** Everything the fleet has recorded up to the round, with the adopted
   changes in the change and maintenance logs (lane state's `export`).
3. **Investigation.** Sessions of the analysis agent on that export, with the
   round's experiments run from the current state by the ledger-aware queue in a
   heavy job of its own, within the round's attempt budget (`AttemptBudget`,
   counted from the round's first session, so earlier rounds do not count).
4. **Decision.** The agent's `RoundDecision` (`submit_decision`), adopted into the
   next ledger version (lane state's `Ledger.adopt`). Round 1's ranked candidates
   are frozen before any second-round record exists: the one-look plan is cut
   from them at the end.

The agent keeps one analysis directory for the whole engagement, so its notes carry
over by construction and experiment ids stay unique. Into the next round's task the
driver carries the agent's earlier decisions, predictions and expectations and the
fact that its changes were adopted (the public ledger), and nothing else: no
evaluation, no oracle value, no word on whether a change removed a fault.

The states (original, after each round, the one-look plan) are evaluated at the end
on the reserved attempts, which no agent sees (lane state's `evaluate`).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fleetlab.agents.sessions import (
    AgentSpec,
    Analysis,
    Engine,
    EngineError,
    SessionRecord,
    reply_problems,
    run_session,
)
from fleetlab.contracts.actions import Action
from fleetlab.contracts.decisions import RoundDecision
from fleetlab.loop.learned import with_learning
from fleetlab.loop.probes import PROBE_TOOLS, probe_tool_texts, with_probes
from fleetlab.toolserver.decisions import PublicLedger, write_public_ledger

__all__ = [
    "ROUND_TOOLS",
    "Budget",
    "Engagement",
    "LoopState",
    "RoundOutcome",
    "decision_of",
    "decision_reminder",
    "freeze_learned",
    "one_look_actions",
    "round_continuation",
    "round_task",
    "run_engagement",
    "run_round_sessions",
]

# The tools of a round's sessions: the analysis tools, the budget's status, the
# diagnosis (the agent's current beliefs) and the decision; no ranked fix set.
ROUND_TOOLS = (
    "query",
    "get_frames",
    "describe",
    "list_skills",
    "read_skill",
    "request_experiment",
    "experiment_results",
    "experiment_budget",
    "earlier_decisions",
    "submit_diagnosis",
    "submit_decision",
    "write_learned_skill",
    "memory_read",
    "memory_write",
)


class LoopState(Protocol):
    """What the driver needs of lane state (`fleetlab.loop.layout`, `state`, `operate`,
    `export`, `evaluate`): one object per world and engagement."""

    def rounds(self) -> int: ...
    def operation_days(self, round_number: int) -> tuple[int, int]: ...
    def experiment_days(self, round_number: int) -> tuple[int, int]: ...
    def empty_ledger(self) -> Any: ...
    def operate(self, round_number: int, ledger: Any) -> list[Path]: ...
    def export(self, round_number: int, ledger: Any, analysis_id: str) -> Path: ...
    def engine(self, round_number: int, ledger: Any, analysis_id: str) -> Engine: ...
    def adopt(self, ledger: Any, round_number: int, decision: RoundDecision) -> Any: ...
    def public_ledger(self, ledger: Any, round_number: int) -> PublicLedger: ...
    def changes(self, ledger: Any) -> int: ...
    def one_look(self, actions: Sequence[Action], final: Any) -> Any: ...
    def evaluate(self, states: Mapping[str, Any]) -> Path: ...
    def audit_runs(self) -> tuple[Path, ...]: ...
    def experiment_root(self, analysis_id: str) -> Path: ...


def _action_text(action: Mapping[str, Any]) -> str:
    target = action["target"]
    thing = target.get("component") or target.get("config_key") or target.get("condition_key")
    text = f"{action['verb']} {thing} at {target['scope']}" if thing else action["verb"]
    if action.get("value") is not None:
        text += f" to {json.dumps(action['value'])}"
    return text


def _ledger_lines(ledger: PublicLedger) -> list[str]:
    """The ledger as the fleet's operators record it: ids, days, status and actions. The
    agent's own reasons are agent prose and stay in its audited outbox (`earlier_decisions`):
    the task text is pipeline text and is audited strictly."""
    lines = []
    for entry in ledger.changes:
        status = {
            "active": "in force",
            "replaced": f"replaced by {entry.replaced_by}",
            "undone": "undone",
        }[entry.status]
        lines.append(
            f"- {entry.entry_id} (round {entry.round}, from day {entry.adopted_day}, {status}): "
            f"{_action_text(entry.action.model_dump(mode='json'))}"
        )
    return lines


def round_task(
    *,
    round_number: int,
    rounds: int,
    operation: Sequence[tuple[int, tuple[int, int]]],
    experiment_days: tuple[int, int],
    budget_text: str,
    ledger: PublicLedger,
    sessions: Sequence[tuple[int, tuple[int, int]]] = (),
) -> str:
    """The task of a round's first session. `operation` holds (round, (first, last day)) of
    every round's regular work so far, `sessions` (round, (first, last session)) of every
    earlier round. Nothing the agent wrote is pasted here: its earlier decisions, reasons,
    predictions and expectations are its own audited state, read with earlier_decisions."""
    days = "; ".join(f"round {n}: days {a} to {b}" for n, (a, b) in operation)
    parts = [
        f"""You are the engineering agent responsible for this fleet's task success. This is
round {round_number} of an ongoing engagement with the fleet, of at most {rounds} rounds.

How a round works. You investigate the fleet's records, test with experiments on the
robots what the records cannot settle, and then decide what to change. The changes you
adopt stay in force: the fleet then does its regular work with them, and in the next
round you see those records and look again. In a round you may adopt up to three
changes, replace or undo changes you made earlier, or change nothing. Stopping is a valid
choice. When nothing further is worth changing, say so with stop.

The records. The log holds the fleet's regular work ({days}) and the episodes of every
experiment you ran (their experiment_id is set; keep them out of the fleet's observed
rates). Day numbers in the tables use this numbering. Read the system description in parts
with describe (dictionary, fleet, vocabulary, costs); the description shows the fleet's
configuration as it now stands, and the change and maintenance logs include the changes
you adopted. Query the catalogue table for the actions and their costs. Read the skills
with read_skill and follow them: triage, episode-cards, failure-modes, scope,
sensitivity-from-logs, sensitivity-by-experiment, rank-actions, write-diagnosis. Return to
an earlier step when new evidence contradicts a hypothesis. Save evidence, open questions
and plans with memory_write: your notes are the only thing that carries from session to
session and from round to round.

Experiments. request_experiment runs an A/B test on the robots as they are now, with your
adopted changes in force, on days {experiment_days[0]} to {experiment_days[1]} after
their regular work; the results arrive in the next session. This round's experiment
budget is {budget_text}. experiment_budget reports what is left. A refused request costs
nothing. Every request needs a power calculation. Before you request experiments, write
down your provisional competing explanations, each with a rough confidence, in your notes
(memory_write) or as an interim diagnosis (submit_diagnosis), and revise them when the
results arrive.

The decision. Submit your current diagnosis with submit_diagnosis (the causes you believe
in, each with target, direction, confidence and evidence; mode summaries; warnings and
retraining indicators if any). Then submit your decision with submit_decision:
- adopt: at most three catalogue actions, each with its rationale;
- replace and undo: earlier changes, by their ledger entry id, with your reason;
- predicted_set_gain: low, mid and high of the gain you expect from this decision's
  changes taken together, in percentage points of whole-fleet success on the fleet's
  regular work, over the fleet as it now stands; repairs overlap and interact, so do not
  add single gains;
- expectations: what the next round's records should show if the changes work, concretely
  (which robots, sites or tasks, and roughly what rates);
- stop: true when nothing further is worth changing after this decision;
- ranked_candidates: optionally, the further catalogue actions you considered and did
  not adopt, best first, each with its expected gain alone and an interval; leave it
  empty when there are none worth naming.
Compute numbers with query. Finish with status done once submit_decision has accepted
your decision. Complete the work autonomously; report uncertainty rather than asking."""
    ]
    if ledger.changes:
        parts.append(
            "The fleet's ledger: the changes you adopted, as the fleet's operators record them.\n"
            + "\n".join(_ledger_lines(ledger))
        )
    if sessions:
        spans = "; ".join(f"round {n}: sessions {a} to {b}" for n, (a, b) in sessions)
        parts.append(
            f"Your earlier rounds ran these sessions: {spans}. earlier_decisions returns every "
            "decision you submitted, with its session: what you adopted, replaced and undid, "
            "your reasons, your predicted gain and your expectations. The last one of a round "
            "is the decision that was adopted."
        )
    if round_number > 1:
        parts.append(
            "Nobody has told you whether your changes were right: the records are your "
            "evidence. Compare what happened with what you predicted, look for what remains "
            "and for what your earlier changes may have hidden, and decide again."
        )
    return "\n\n".join(parts)


def round_continuation(round_number: int, session: int, task: str, probes: bool = False) -> str:
    """After a session that requested experiments (or, with the probe switch on, probes)."""
    if probes:  # lane probe: the opt-in probe path
        return (
            f"This is session {session} of round {round_number}. Your earlier session "
            "requested experiments or probes; they have run. Read your notes with memory_read, "
            "the results of experiments with experiment_results and of probes with "
            "probe_results; the new episodes are in the log and the probe records in the "
            "probe tables. Then continue with the task."
            f"\n\nThe task:\n{task}"
        )
    return (
        f"This is session {session} of round {round_number}. Your earlier session requested "
        "experiments; they have run. Read your notes with memory_read and the results with "
        "experiment_results; the new episodes are in the log. Then continue with the task."
        f"\n\nThe task:\n{task}"
    )


def decision_reminder(round_number: int, session: int, task: str) -> str:
    """After a session that ended without an accepted decision and without a request."""
    return (
        f"This is session {session} of round {round_number}. Your last session ended without "
        "an accepted decision (submit_decision). Read your notes with memory_read, finish what "
        "the task still needs and submit the decision; a decision to change nothing, or to "
        f"stop, is a decision too.\n\nThe task:\n{task}"
    )


def decision_of(analysis: Analysis, sessions: Sequence[int]) -> RoundDecision | None:
    """The round's decision: the last one accepted in one of its sessions."""
    found = [
        s
        for s in analysis.submissions()
        if s.get("kind") == "decision" and int(s.get("session", 0)) in set(sessions)
    ]
    return RoundDecision.model_validate(found[-1]["answer"]) if found else None


def one_look_actions(decision: RoundDecision) -> list[Action]:
    """Round 1's list the plan fixed after one look is cut from: the adopted actions in
    their order, then the ranked alternatives in theirs, each action once. Lane state's
    `one_look_ledger` cuts it at the final state's number of changes, so the plan holds
    round 1's adoptions even when the agent named no alternative."""
    out: list[Action] = []
    seen: set[str] = set()
    for action in [c.action for c in decision.adopt] + [
        r.action for r in decision.ranked_candidates
    ]:
        key = action.model_dump_json()
        if key not in seen:
            seen.add(key)
            out.append(action)
    return out


@dataclass
class RoundOutcome:
    """What a round's investigation left: its sessions, the last export and the decision."""

    sessions: list[SessionRecord] = field(default_factory=list)
    export: Path | None = None
    decision: RoundDecision | None = None
    stopped: str | None = None  # why the sessions ended without a decision


def run_round_sessions(
    analysis: Analysis,
    export: Path,
    agent: AgentSpec,
    system: str,
    task: str,
    env: Any,
    engine: Engine | None,
    *,
    round_number: int,
    limits: Mapping[str, int],
    attempts: int,
    max_sessions: int,
    audit: Any,
    run_root: Path,
    tools: tuple[str, ...] = ROUND_TOOLS,
) -> RoundOutcome:
    """Sessions of one round until a decision is accepted, an auth or rate limit stops
    them, or `max_sessions`. A session that requested experiments hands its queue to
    `engine`; its reply is checked (`reply_problems`) before the results are recorded
    and the next session starts on the reply's export."""
    outcome = RoundOutcome(export=export)
    numbers: list[int] = []
    prompt = task
    for _ in range(max_sessions):
        session = run_session(
            analysis,
            outcome.export or export,
            agent,
            system,
            prompt,
            env,
            tools=tools,
            max_experiment_episodes=attempts,
            limits=limits,
            run_root=run_root,
            audit=audit,
        )
        outcome.sessions.append(session)
        numbers.append(session.number)
        if session.error in ("auth", "rate_limit") or session.lost_twice:
            outcome.stopped = session.error or "lost twice"
            break
        if session.state == "waiting":
            written = analysis.write_queue(session.number, outcome.export or export)
            if written is None:
                outcome.stopped = "waiting without a pending request"
                break
            queue_file, queue = written
            if engine is None:
                outcome.stopped = "no engine for the queued experiments"
                break
            reply = engine.run_queue(queue_file, queue)
            if problems := reply_problems(reply, queue, audit.experiment_days):
                raise EngineError("; ".join(problems))
            analysis.record_results(reply.results)
            analysis.record_probe_outcomes(reply.probes)  # probes the agent composed (lane probe)
            outcome.export = reply.export
            prompt = round_continuation(
                round_number, analysis.sessions_done() + 1, task, "request_probe" in tools
            )
            continue
        outcome.decision = decision_of(analysis, numbers)
        if outcome.decision is not None:
            break
        prompt = decision_reminder(round_number, analysis.sessions_done() + 1, task)
    else:
        outcome.decision = decision_of(analysis, numbers)
        if outcome.decision is None:
            outcome.stopped = f"no decision in {max_sessions} sessions"
    return outcome


def freeze(path: Path, data: Any) -> str:
    """Write a record that must not change later and return its SHA-256."""
    text = json.dumps(data, indent=1, sort_keys=True, default=str)
    path.write_text(text + "\n")
    return hashlib.sha256((text + "\n").encode()).hexdigest()


def publish_ledger(analysis: Analysis, ledger: PublicLedger) -> None:
    """The round's public ledger in the analysis' results (read-only for the agent)."""
    write_public_ledger(analysis.results, ledger)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Budget(Protocol):
    """A round's attempt budget (`fleetlab.loop.driver.AttemptBudget`)."""

    @property
    def attempts(self) -> int: ...

    def limits(self, first_session: int = 1) -> dict[str, int]: ...
    def text(self) -> str: ...


@dataclass
class Engagement:
    """How one engagement runs: who decides, within what, and around which runtime."""

    agent: AgentSpec
    budget: Budget
    max_sessions: int  # per round
    system: str
    env: Any  # the agents' ToolEnv
    fleet: Any  # the full fleet model: audit vocabulary only, never mounted
    label: str = "engagement"
    # Before a round's first session: the CLI gate where stale, the source audit of the
    # round's export and prompts, and the login check (None for a scripted agent).
    gate: Callable[[Path, Mapping[str, str], Sequence[Path]], None] | None = None
    # Wraps the state's engine for a round, e.g. the heavy queue with its queue log.
    wrap_engine: Callable[[Engine, int], Engine] | None = None
    tools: tuple[str, ...] = ROUND_TOOLS
    # A fresh analysis to run in, made by the caller (the reuse comparison makes each arm's,
    # with its own id and, in one arm, a learned library: fleetlab.loop.reuse); None makes one.
    analysis: Analysis | None = None
    # An aborted engagement (a round without a decision) is not evaluated unless asked.
    evaluate_aborted: bool = False
    # Learning (docs/design/loop.md, section 9; fleetlab.loop.learned): the round task
    # carries lane learn's paragraph and the agent may write learned skills; after the
    # rounds the host checks and freezes them into a library. `learned` mounts an earlier
    # frozen library into this engagement's analysis (a reuse comparison's later arm).
    learning: bool = True
    learned: Path | None = None  # a frozen library's root, as `freeze_learned` records it
    fleet_label: str = "an earlier fleet"  # how a learned skill's provenance names the fleet
    # Probes the agent composes (docs/design/loop.md, section 10; fleetlab.loop.probes),
    # opt-in: the probe tools, the task's probe paragraph and its tool texts in the gate.
    probes: bool = False


def freeze_learned(
    analysis: Analysis,
    state: LoopState,
    how: Engagement,
    out: Path,
    spans: Sequence[tuple[int, tuple[int, int]]] = (),
) -> dict[str, Any]:
    """After the rounds: lane learn's checks and freeze of the skills the agent wrote in
    this engagement (`fleetlab.loop.learned.freeze`), against the truth side of this
    fleet. A refused skill is dropped and listed with the host's reasons (never mounted).
    `spans` (round, (first, last session)) gives each draft's provenance the round its
    session belonged to (lane learn's `Source.session_rounds`). The record's `library` is
    the library's root: a later engagement takes that path (`Engagement.learned`)."""
    from fleetlab.loop.learned import Source, freeze, stored_drafts

    drafts = stored_drafts(analysis.root)
    out_record: dict[str, Any] = {"drafts": sorted(drafts), "library": None, "digest": None}
    if not drafts:
        return out_record
    rounds_of = {s: n for n, (first, last) in spans for s in range(first, last + 1)}
    out_record["rounds"] = {name: rounds_of.get(d.session) for name, d in drafts.items()}
    report = freeze(
        [
            Source(
                analysis.root,
                how.fleet_label,
                None,
                state.audit_runs(),
                how.fleet,
                session_rounds=rounds_of,
            )
        ],
        out / "learned",
        drop_refused=True,
    )
    out_record.update(kept=report.kept, refused=report.refused)
    if report.library is not None:
        out_record.update(library=str(report.library.root), digest=report.library.digest)
    return out_record


def run_engagement(state: LoopState, world_id: str, out: Path, how: Engagement) -> dict[str, Any]:
    """All rounds of one engagement, then the evaluation of the states; the record is
    `out/engagement.json`, rewritten after every round."""
    out.mkdir(parents=True, exist_ok=False)
    record_file = out / "engagement.json"
    record: dict[str, Any] = {
        "kind": "engagement",
        "label": how.label,
        "world_id": world_id,
        "agent": {k: getattr(how.agent, k) for k in ("backend", "model", "effort", "service_tier")},
        "budget": {"attempts": how.budget.attempts, "text": how.budget.text()},
        "max_sessions_per_round": how.max_sessions,
        "rounds_planned": state.rounds(),
        "started_at": now(),
        "rounds": [],
    }

    def save() -> None:
        record_file.write_text(json.dumps(record, indent=1, default=str) + "\n")

    save()
    if how.analysis is not None:
        # The reuse harness's fresh analysis, which carries its own verified library.
        if how.learned is not None:
            raise ValueError("a given analysis carries its own library; leave `learned` unset")
        analysis = how.analysis
        if analysis.world_id != world_id:
            raise ValueError(f"the given analysis belongs to {analysis.world_id}, not {world_id}")
        if analysis.sessions_done() or analysis.requests() or analysis.submissions():
            raise ValueError("an engagement starts on a fresh analysis")
        if analysis.learned is not None:
            record["learned_mounted"] = analysis.meta.get("learned")
    else:
        mounted = None
        if how.learned is not None:
            # The handoff from an earlier engagement is the library's root as its record names
            # it; only the verified skills tree is mounted, the host's library.json stays out.
            from fleetlab.loop.learned import LIBRARY_RECORD, FrozenLibrary

            if not (Path(how.learned) / LIBRARY_RECORD).is_file():
                raise ValueError(
                    f"{how.learned}: not a frozen library's root (the path freeze_learned records, "
                    "with library.json beside skills/)"
                )
            library = FrozenLibrary.load(how.learned)
            mounted = library.skills
            record["learned_mounted"] = {
                "library": str(library.root),
                "digest": library.digest,
                "skills": library.names,
            }
        analysis = Analysis.create(world_id, how.label, base=out, learned=mounted)
        if mounted is not None:
            record["learned_mounted"]["analysis"] = analysis.meta.get("learned")
    record["analysis"] = str(analysis.root)
    ledger = state.empty_ledger()
    states: dict[str, Any] = {"original": ledger}
    spans: list[tuple[int, tuple[int, int]]] = []  # each earlier round's sessions
    first_list: list[Action] = []  # round 1's one-look list, once round 1 decided
    try:
        record.update(_rounds(state, analysis, how, out, record, save, states, spans, first_list))
    except BaseException as error:
        record.update(outcome="failed", error=f"{type(error).__name__}: {error}", ended_at=now())
        save()
        raise
    decided = [n for n in range(1, state.rounds() + 1) if f"round-{n}" in states]
    ledger = states[f"round-{decided[-1]}"] if decided else states["original"]
    ranked_first = first_list if decided else None
    if ranked_first is not None:
        plan = state.one_look(ranked_first, ledger)
        states["one-look"] = plan
        # The requested cutoff beside what the plan could hold and holds: an honest shorter
        # plan when round 1 named fewer adoptable actions than the final state's changes.
        record["one_look"] = {
            "requested_cutoff": state.changes(ledger),
            "listed": len(ranked_first),
            "adoptable": sum(a.is_complete and a.has_replay for a in ranked_first),
            "applied": state.changes(plan),
        }
    record["states"] = sorted(states)
    if how.learning:
        record["learned"] = freeze_learned(analysis, state, how, out, spans)
    save()
    if record["outcome"] == "aborted" and not how.evaluate_aborted:
        record["evaluation"] = {"skipped": "the engagement was aborted"}
        record["ended_at"] = now()
        save()
        return record
    # Only now, after the last decision, is anything evaluated; nothing of it reaches an
    # agent (the engagement is over and its analysis is never started again).
    clock = time.perf_counter()
    evaluation = state.evaluate(states)
    record["evaluation"] = {"path": str(evaluation), "seconds": time.perf_counter() - clock}
    summary = evaluation / "summary.json"
    if summary.is_file():
        record["evaluation"]["summary"] = json.loads(summary.read_text())
    record["ended_at"] = now()
    save()
    return record


def _rounds(
    state: LoopState,
    analysis: Analysis,
    how: Engagement,
    out: Path,
    record: dict[str, Any],
    save: Callable[[], None],
    states: dict[str, Any],
    spans: list[tuple[int, tuple[int, int]]],
    first_list: list[Action],
) -> dict[str, Any]:
    """Every round; returns the engagement's outcome: `completed` (every round decided),
    `stopped` (the agent decided to stop) or `aborted` (a round ended without a decision:
    an auth or rate limit, a lost call, sessions exhausted), with why."""
    from fleetlab.agents.audit import AuditContext
    from fleetlab.loop.costs import experiment_costs, probe_costs, session_costs

    ledger = states["original"]
    for number in range(1, state.rounds() + 1):
        entry: dict[str, Any] = {"round": number, "started_at": now()}
        record["rounds"].append(entry)
        clock = time.perf_counter()
        runs = state.operate(number, ledger)
        entry["operation"] = {
            "days": state.operation_days(number),
            "runs": [str(r) for r in runs],
            "seconds": time.perf_counter() - clock,
        }
        clock = time.perf_counter()
        export = state.export(number, ledger, analysis.analysis_id)
        entry["export"] = {"path": str(export), "seconds": time.perf_counter() - clock}
        public = state.public_ledger(ledger, number)
        publish_ledger(analysis, public)
        first = analysis.sessions_done() + 1
        task = round_task(
            round_number=number,
            rounds=state.rounds(),
            operation=[(k, state.operation_days(k)) for k in range(1, number + 1)],
            experiment_days=state.experiment_days(number),
            budget_text=how.budget.text(),
            ledger=public,
            sessions=spans,
        )
        if how.learning:
            task = with_learning(task)
        texts = {"system": how.system}
        if how.probes:  # lane probe: the opt-in probe path
            task = with_probes(task, learning=how.learning)
            texts.update(probe_tool_texts(how.budget.limits(first)))
        entry["task"] = task
        save()
        if how.gate is not None:
            how.gate(export, {**texts, "task": task}, state.audit_runs())
        engine = state.engine(number, ledger, analysis.analysis_id)
        if how.wrap_engine is not None:
            engine = how.wrap_engine(engine, number)
        span = (state.experiment_days(1)[0], state.experiment_days(number)[1])
        audit = AuditContext(
            state.audit_runs(), how.fleet, state.experiment_root(analysis.analysis_id), span
        )
        try:
            outcome = run_round_sessions(
                analysis,
                export,
                how.agent,
                how.system,
                task,
                how.env,
                engine,
                round_number=number,
                limits=how.budget.limits(first),
                attempts=how.budget.attempts,
                max_sessions=how.max_sessions,
                audit=audit,
                run_root=out / "agent-runs",
                tools=(*how.tools, *PROBE_TOOLS) if how.probes else how.tools,
            )
        finally:
            sessions = [s for s in analysis.session_records() if s.number >= first]
            entry["costs"] = {
                **session_costs(sessions, how.agent.model),
                **experiment_costs(analysis, first),
                **probe_costs(analysis, first),
                "queues": list(getattr(engine, "records", [])),
            }
            entry["sessions"] = [
                {"number": s.number, "state": s.state, "error": s.error, "starts": len(s.attempts)}
                for s in sessions
            ]
            save()
        decision = outcome.decision
        entry["decision"] = decision.model_dump(mode="json") if decision is not None else None
        entry["stopped"] = outcome.stopped
        if sessions:
            spans.append((number, (sessions[0].number, sessions[-1].number)))
        if decision is None:
            why = f"round {number}: {outcome.stopped or 'no decision'}"
            return {"outcome": "aborted", "ended_because": why}
        if number == 1:
            # Frozen before any second-round record exists.
            listed = one_look_actions(decision)
            first_list.extend(listed)
            entry["one_look_list_sha256"] = freeze(
                out / "one-look-list.json", [a.model_dump(mode="json") for a in listed]
            )
        ledger = state.adopt(ledger, number, decision)
        states[f"round-{number}"] = ledger
        entry["ledger"] = state.public_ledger(ledger, number + 1).model_dump(mode="json")
        entry["ended_at"] = now()
        save()
        if decision.stop:
            return {"outcome": "stopped", "ended_because": f"round {number}: the agent stopped"}
    return {"outcome": "completed", "ended_because": "all rounds ran"}
