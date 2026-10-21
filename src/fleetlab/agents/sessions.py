"""Analyses in sessions, memory per world, and the experiment queue handed to the engine.

An experiment is a heavy job of minutes, and a tool call inside `claude -p`
cannot wait for it (docs/design/contracts.md, section 12). An analysis is
therefore a sequence of sessions, each one CLI call:

1. A session runs with every tool. `request_experiment` queues a request in
   the analysis' outbox and returns its experiment id at once.
2. The session ends. Its state is decided by the runtime from what the tool
   server recorded, not from what the agent says: "waiting" when it queued
   experiments, "done" when it submitted a diagnosis or a fix set and queued
   nothing, "ended" when it did neither, "failed" when the call failed.
3. While "waiting", the runtime writes the pending requests to a queue file
   (`ExperimentQueue`) and hands it to the engine (`Engine.run_queue`, the
   rollout engine). The engine runs every request in the
   experiment window, both arms always simulated, and answers with an
   `EngineReply`: one `ExperimentResult` per request and the export that now
   holds the new episodes.
4. The next session starts on that export, with the results visible through
   `experiment_results` and the notes through `memory_read`. Memory is the
   only state that crosses sessions.

Memory is per analysis of a world: `Analysis.create` makes a new directory
and refuses an existing one, so two worlds, or two methods on one world,
never share notes. That is the reset between worlds, by construction.

    analysis/
      analysis.json            world id, label, created
      memory/notes.md          mounted read-write at /memory
      outbox/                  mounted read-write at /outbox (toolserver.files)
      results/experiments.jsonl  mounted read-only at /results; written here only
      queues/session-NN.json   what was handed to the engine after session NN
      sessions/session-NN.json the session's record (state, run directory, usage)
      learned/                 a frozen library of learned skills, copied in at creation and
                               mounted read-only at /learned (docs/design/loop.md, section 9)

A learned library is given to an analysis only when it is created, so an
analysis' skills never change between its sessions, and it is the only thing
an analysis can receive from an earlier one.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from fleetlab.agents.audit import AuditContext, tree_digest
from fleetlab.agents.cli import Backend, agent_root
from fleetlab.agents.run import (
    AgentResult,
    Auth,
    ToolAccess,
    attempt_record,
    lost_runtime_answer,
    parse_claude,
    parse_codex,
    run_agent,
)
from fleetlab.agents.toolenv import ToolEnv
from fleetlab.contracts.ids import WorldId
from fleetlab.contracts.probes import ProbeOutcome, QueuedProbe
from fleetlab.contracts.tools import Diagnosis, ExperimentRequest, ExperimentResult, RankedFixSet
from fleetlab.toolserver.files import (
    DIAGNOSIS_FILE,
    EXPERIMENTS_FILE,
    FIX_SET_FILE,
    NOTES_FILE,
    PROBES_FILE,
    RESULTS_FILE,
    SUBMISSIONS_FILE,
    append_line,
    read_lines,
)
from fleetlab.toolserver.jobs import JOB_REQUESTS_FILE, SAVED_JOBS_FILE
from fleetlab.toolserver.public_results import (
    probe_outcome_problems,
    probe_outcomes_problems,
    request_results_problems,
    result_problems,
    results_problems,
)

__all__ = [
    "BUDGET_LIMITS",
    "QUEUE_FORMAT",
    "AgentSpec",
    "Analysis",
    "AnalysisRecord",
    "Engine",
    "EngineError",
    "EngineReply",
    "ExperimentQueue",
    "QueuedExperiment",
    "SessionRecord",
    "SessionReport",
    "SessionState",
    "arm_problems",
    "continuation_prompt",
    "run_analysis",
    "run_session",
]

SessionState = Literal["waiting", "done", "ended", "failed"]
QUEUE_FORMAT: Literal["fleetlab.experiment-queue/1"] = "fleetlab.experiment-queue/1"
# Budget fields of the tool server's `Limits` a caller may set per session, besides
# `max_experiment_episodes` (`ToolAccess` carries them to the server's command line).
BUDGET_LIMITS = ("max_experiment_requests", "max_episodes_per_arm", "budget_first_session")


class SessionReport(BaseModel):
    """A session's final answer: the agent's own account (the runtime decides the state)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["waiting", "done"]
    summary: str


class QueuedExperiment(BaseModel):
    """One request as the engine receives it."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    session: int = Field(ge=1)
    request: ExperimentRequest


class ExperimentQueue(BaseModel):
    """What the runtime hands to the engine after a session ended "waiting".

    Rules the engine keeps (contracts, section 12): every request runs in the
    experiment window of `world_id`; both arms are always simulated, also
    when an action targets something healthy, or the agent would learn health
    from identical results; the episodes go into the visible log like any
    other, with neutral ids, and the engine's reply names them per arm; every
    arm episode's identity carries the experiment's id
    (`EpisodeIdentity.experiment_id`, visible as `episodes.experiment_id`), so
    that a later session can keep the arms out of the observed rates. The
    runtime checks the last rule in the reply's export (`arm_problems`).
    """

    format: Literal["fleetlab.experiment-queue/1"] = QUEUE_FORMAT
    world_id: str
    analysis_id: str
    session: int
    created_at: str
    export: str  # the export the session saw; the reply's export extends it
    rules: dict[str, Any] = Field(
        default_factory=lambda: {
            "window": "experiment",
            "both_arms_always_simulated": True,
            "arm_episodes_carry_experiment_id": True,
        }
    )
    experiments: list[QueuedExperiment]
    # Probes the agent composed (contracts.probes); left out of the dump when there are
    # none, so that every queue written before probes keeps its bytes and hash.
    probes: list[QueuedProbe] = Field(default_factory=list, exclude_if=lambda value: not value)


class EngineReply(BaseModel):
    """The engine's answer to a queue: one result per experiment, one outcome per probe,
    and the export to use next."""

    results: list[ExperimentResult]
    export: Path
    probes: list[ProbeOutcome] = Field(default_factory=list, exclude_if=lambda value: not value)


class Engine(Protocol):
    """Runs a queue of experiments (the rollout engine implements it)."""

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply: ...


@dataclass(frozen=True)
class AgentSpec:
    """Which agent runs a session, and how."""

    backend: Backend
    model: str
    effort: str | None = None
    service_tier: str = "default"
    thinking_tokens: int | None = None
    auth: Auth = "subscription"
    timeout: float = 1800.0


@dataclass
class SessionRecord:
    number: int
    state: SessionState
    export: str
    run_dir: str
    ok: bool
    error: str | None
    seconds: float
    usage: dict[str, Any]
    queued: list[str]  # experiment ids requested in this session
    submitted: list[str]  # receipt ids of accepted submissions in this session
    report: dict[str, Any] | None  # the SessionReport, as the agent gave it
    tool_calls: int
    attempts: list[dict[str, Any]] = field(default_factory=list)  # per-start receipts
    rerun: bool = False
    lost_twice: bool = False  # entry 11: no answer after the single replacement start
    window: dict[str, Any] | None = None  # the subscription window the call reported (claude)
    queue_file: str | None = None
    in_progress: bool = False  # durable checkpoint; dispatch already consumes a start


@dataclass
class AnalysisRecord:
    analysis_id: str
    world_id: str
    sessions: list[SessionRecord] = field(default_factory=list)

    @property
    def state(self) -> SessionState | Literal["not started"]:
        return self.sessions[-1].state if self.sessions else "not started"


@dataclass(frozen=True)
class Analysis:
    """One analysis of one world: its own memory, outbox, results and sessions."""

    root: Path

    @classmethod
    def create(
        cls,
        world_id: str,
        label: str = "",
        base: Path | None = None,
        analysis_id: str | None = None,
        learned: Path | None = None,
    ) -> Analysis:
        """A new analysis directory; refuses one that exists, so memory never carries over.

        `learned`: a frozen library of learned skills (`fleetlab.loop.learned`),
        copied in and mounted read-only in every session; its digest is recorded.
        """
        WorldIdCheck.model_validate({"world_id": world_id})
        analysis_id = (
            analysis_id or f"{world_id}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        root = (base or agent_root() / "analyses") / analysis_id
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()  # FileExistsError for a reused id: a new world gets a new directory
        for part in ("memory", "outbox", "results", "queues", "sessions"):
            (root / part).mkdir()
        meta: dict[str, Any] = {"analysis_id": analysis_id, "world_id": world_id, "label": label}
        if learned is not None:
            source = tree_digest(learned)
            shutil.copytree(learned, root / "learned")
            if tree_digest(root / "learned") != source:
                raise ValueError("the learned library changed while it was copied")
            meta["learned"] = {"source": str(learned), "digest": source}
        (root / "analysis.json").write_text(json.dumps({**meta, "created": time.time()}, indent=1))
        return cls(root)

    @property
    def meta(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.root / "analysis.json").read_text())
        return data

    @property
    def analysis_id(self) -> str:
        return str(self.meta["analysis_id"])

    @property
    def world_id(self) -> str:
        return str(self.meta["world_id"])

    @property
    def memory(self) -> Path:
        return self.root / "memory"

    @property
    def outbox(self) -> Path:
        return self.root / "outbox"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def learned(self) -> Path | None:
        """The analysis' learned library, or None when it was created without one."""
        path = self.root / "learned"
        return path if "learned" in self.meta else None

    def notes(self) -> str:
        path = self.memory / NOTES_FILE
        return path.read_text() if path.is_file() else ""

    def requests(self) -> list[QueuedExperiment]:
        return [
            QueuedExperiment.model_validate(r) for r in read_lines(self.outbox / EXPERIMENTS_FILE)
        ]

    def probe_requests(self) -> list[QueuedProbe]:
        """The accepted probe requests (`request_probe`), oldest first."""
        return [QueuedProbe.model_validate(r) for r in read_lines(self.outbox / PROBES_FILE)]

    def probe_outcomes(self) -> dict[str, ProbeOutcome]:
        return {
            r["probe_id"]: ProbeOutcome.model_validate(r)
            for r in read_lines(self.results / PROBES_FILE)
        }

    def pending_probes(self) -> list[QueuedProbe]:
        """Probe requests without an outcome yet."""
        done = self.probe_outcomes()
        return [q for q in self.probe_requests() if q.probe_id not in done]

    def record_probe_outcomes(self, outcomes: Sequence[ProbeOutcome]) -> None:
        """Store the engine's probe outcomes, which the next session sees (`probe_results`),
        after the checks the tool server repeats (`public_results.probe_outcome_problems`)."""
        known = {q.probe_id: q.request for q in self.probe_requests()}
        done = self.probe_outcomes()
        for outcome in outcomes:
            if outcome.probe_id not in known:
                raise ValueError(f"no probe {outcome.probe_id} in this analysis")
            if outcome.probe_id in done:
                raise ValueError(f"{outcome.probe_id} has an outcome already")
            if problems := probe_outcome_problems(outcome, known[outcome.probe_id]):
                raise ValueError("; ".join(problems))
        if problems := probe_outcomes_problems([*done.values(), *outcomes]):
            raise ValueError("; ".join(problems))
        # Validate the entire reply before appending any agent-visible record.
        for outcome in outcomes:
            append_line(self.results / PROBES_FILE, outcome.model_dump(mode="json"))

    def results_by_id(self) -> dict[str, ExperimentResult]:
        return {
            r["experiment_id"]: ExperimentResult.model_validate(r)
            for r in read_lines(self.results / RESULTS_FILE)
        }

    def pending(self) -> list[QueuedExperiment]:
        """Requests without a result yet."""
        done = self.results_by_id()
        return [q for q in self.requests() if q.experiment_id not in done]

    def submissions(self) -> list[dict[str, Any]]:
        return list(read_lines(self.outbox / SUBMISSIONS_FILE))

    def diagnosis(self) -> Diagnosis | None:
        path = self.outbox / DIAGNOSIS_FILE
        return Diagnosis.model_validate_json(path.read_text()) if path.is_file() else None

    def fix_set(self) -> RankedFixSet | None:
        path = self.outbox / FIX_SET_FILE
        return RankedFixSet.model_validate_json(path.read_text()) if path.is_file() else None

    def sessions_done(self) -> int:
        return sum(
            not json.loads(path.read_text()).get("in_progress", False)
            for path in (self.root / "sessions").glob("session-*.json")
        )

    def session_records(self) -> list[SessionRecord]:
        return [
            SessionRecord(**json.loads(path.read_text()))
            for path in sorted((self.root / "sessions").glob("session-*.json"))
        ]

    def tool_access(self, env: ToolEnv, export: Path, session: int, **kwargs: Any) -> ToolAccess:
        kwargs.setdefault("learned", self.learned)
        return ToolAccess(
            env=env,
            export=export,
            memory=self.memory,
            outbox=self.outbox,
            results=self.results,
            session=session,
            **kwargs,
        )

    def write_queue(self, session: int, export: Path) -> tuple[Path, ExperimentQueue] | None:
        """The pending requests as a queue file for the engine; None when nothing is pending."""
        pending = self.pending()
        probes = self.pending_probes()
        if not pending and not probes:
            return None
        queue = ExperimentQueue(
            world_id=self.world_id,
            analysis_id=self.analysis_id,
            session=session,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            export=str(export),
            experiments=pending,
            probes=probes,
        )
        path = self.root / "queues" / f"session-{session:02d}.json"
        path.write_text(queue.model_dump_json(indent=1))
        return path, queue

    def record_results(self, results: Sequence[ExperimentResult]) -> None:
        """Store the engine's results, which the next session sees through experiment_results."""
        known = {q.experiment_id: q.request for q in self.requests()}
        done = self.results_by_id()
        for result in results:
            if result.experiment_id not in known:
                raise ValueError(f"no request {result.experiment_id} in this analysis")
            if result.experiment_id in done:
                raise ValueError(f"{result.experiment_id} has a result already")
            if problems := result_problems(result, known[result.experiment_id]):
                raise ValueError("; ".join(problems))
        if problems := results_problems(results):
            raise ValueError("; ".join(problems))
        # Validate the entire reply before appending any agent-visible record.
        for result in results:
            append_line(self.results / RESULTS_FILE, result.model_dump(mode="json"))


class WorldIdCheck(BaseModel):
    world_id: WorldId


class EngineError(RuntimeError):
    """The engine's reply breaks the queue's rules."""


def arm_problems(
    reply: EngineReply,
    experiment_days: tuple[int, int] | None = None,
) -> list[str]:
    """Arm episodes of the reply that its export lacks, or that do not carry their
    experiment's id in `episodes.experiment_id`."""
    from fleetlab.log import LogReader

    problems = results_problems(reply.results)
    wanted = {
        episode: result.experiment_id
        for result in reply.results
        if result.status == "done"
        for arm in result.arms
        for episode in arm.episode_ids
    }
    if not wanted and experiment_days is None:
        return problems
    with LogReader(reply.export) as reader:
        rows = reader.query("SELECT episode_id, experiment_id, day FROM episodes").to_pylist()
    if experiment_days is not None and any(
        row["experiment_id"] is not None
        and not experiment_days[0] <= row["day"] <= experiment_days[1]
        for row in rows
    ):
        problems.append("experiment episodes lie outside the authorized experiment window")
    found = {row["episode_id"]: row["experiment_id"] for row in rows}
    for episode, experiment in sorted(wanted.items()):
        if episode not in found:
            problems.append(f"{episode} of {experiment} is not in the reply's export")
        elif found[episode] != experiment:
            problems.append(
                f"{episode} carries experiment_id {found[episode]!r}, not {experiment!r}"
            )
    return problems


def reply_problems(
    reply: EngineReply,
    queue: ExperimentQueue,
    experiment_days: tuple[int, int] | None = None,
) -> list[str]:
    """A queue has exactly one completed/refused result for every request."""
    problems = arm_problems(reply, experiment_days)
    expected = {q.experiment_id: q.request for q in queue.experiments}
    problems.extend(request_results_problems(reply.results, expected))
    problems.extend(probe_reply_problems(reply, queue))
    return problems


def probe_reply_problems(reply: EngineReply, queue: ExperimentQueue) -> list[str]:
    """One outcome per queued probe, each bound to its request and to the probe rows of
    the reply's export (the checks the tool server repeats on every later read)."""
    from fleetlab.log import LogReader

    problems = probe_outcomes_problems(reply.probes)
    requests = {q.probe_id: q.request for q in queue.probes}
    if sorted(o.probe_id for o in reply.probes) != sorted(requests):
        problems.append("reply probe outcomes differ from the queue's probes")
    if not reply.probes:
        return problems
    with LogReader(reply.export) as reader:
        rows = {
            r["episode_id"]: (r["probe_id"], r["robot_id"], r["repeat"])
            for r in reader.query(
                "SELECT episode_id, probe_id, robot_id, repeat FROM probe_episodes"
            ).to_pylist()
        }
        with_steps = set(
            reader.query("SELECT DISTINCT episode_id FROM probe_steps")
            .column("episode_id")
            .to_pylist()
        )
    for outcome in reply.probes:
        request = requests.get(outcome.probe_id)
        if request is not None:
            problems.extend(probe_outcome_problems(outcome, request, rows, with_steps))
    return problems


def continuation_prompt(session: int, task: str) -> str:
    """The prompt of a session after the first: the task again, and where the news is."""
    return (
        f"This is session {session} of this analysis. Your earlier sessions requested experiments; "
        "they have run. Read your notes with memory_read and the results with experiment_results; "
        "the new episodes are in the log. Then continue with the task.\n\n"
        f"The task:\n{task}"
    )


SESSION_RULES = (
    "\n\nHow sessions work: request_experiment queues an experiment that runs only after this "
    "session ends. If you request experiments, write what you need to remember with memory_write, "
    'then finish with status "waiting"; the next session sees the results. When you have '
    'submitted your answer, finish with status "done".'
)


def run_session(
    analysis: Analysis,
    export: Path,
    agent: AgentSpec,
    system: str,
    prompt: str,
    env: ToolEnv,
    *,
    tools: tuple[str, ...] | None = None,
    max_experiment_episodes: int | None = None,
    run_root: Path | None = None,
    audit: AuditContext | None = None,
    on_start: Callable[[Path], None] | None = None,
    limits: Mapping[str, int] | None = None,
) -> SessionRecord:
    """One ex-post session, with at most one replacement for an untouched lost call.

    `limits` sets further budget fields of the tool server (`BUDGET_LIMITS`), for
    example a round's own budget (fleetlab.loop.driver); None keeps the defaults.
    """
    number = analysis.sessions_done() + 1
    checkpoint = analysis.root / "session-state" / f"session-{number:02d}"
    before = checkpoint / "before"
    conditions: dict[str, Any] = {
        "export": str(export),
        "agent": asdict(agent),
        "system": system,
        "prompt": prompt,
        "tools": list(tools) if tools is not None else None,
        "max_experiment_episodes": max_experiment_episodes,
    }
    if limits:
        if unknown := sorted(set(limits) - set(BUDGET_LIMITS)):
            raise ValueError(f"unknown tool-server limits: {unknown}")
        conditions["limits"] = dict(limits)
    session_file = analysis.root / "sessions" / f"session-{number:02d}.json"
    saved = SessionRecord(**json.loads(session_file.read_text())) if session_file.exists() else None
    if not before.exists():
        if saved is not None:
            raise ValueError("unfinished session has no original state snapshot")
        before.mkdir(parents=True)
        for part in ("memory", "outbox"):
            shutil.copytree(analysis.root / part, before / part, symlinks=True)
        (checkpoint / "conditions.json").write_text(json.dumps(conditions))
    elif (saved is not None and not saved.in_progress) or json.loads(
        (checkpoint / "conditions.json").read_text()
    ) != conditions:
        raise ValueError("unfinished session conditions differ from its first start")
    before_requests = {r["experiment_id"] for r in read_lines(before / "outbox" / EXPERIMENTS_FILE)}
    before_requests |= {r["probe_id"] for r in read_lines(before / "outbox" / PROBES_FILE)}
    before_submissions = {r["receipt_id"] for r in read_lines(before / "outbox" / SUBMISSIONS_FILE)}
    # A job session's accepted work (fleetlab.toolserver.jobs): queued job programs and saves
    # pending release. Like experiments, they are charged and kept, never undone by a
    # replacement call.
    before_jobs = {r["job_id"] for r in read_lines(before / "outbox" / JOB_REQUESTS_FILE)}
    before_saves = len(list(read_lines(before / "outbox" / SAVED_JOBS_FILE)))
    access = analysis.tool_access(
        env,
        export,
        number,
        tools=tools,
        max_experiment_episodes=max_experiment_episodes,
        audit=audit,
        **dict(limits or {}),
    )
    attempts = saved.attempts if saved else []

    def record_result(res: AgentResult, *, dispatched: bool = False) -> SessionRecord:
        queued = [
            q.experiment_id for q in analysis.requests() if q.experiment_id not in before_requests
        ]
        queued += [
            q.probe_id for q in analysis.probe_requests() if q.probe_id not in before_requests
        ]
        submitted = [
            s["receipt_id"]
            for s in analysis.submissions()
            if s["receipt_id"] not in before_submissions
        ]
        queued += [
            r["job_id"]
            for r in read_lines(analysis.outbox / JOB_REQUESTS_FILE)
            if r["job_id"] not in before_jobs
        ]
        saves = len(list(read_lines(analysis.outbox / SAVED_JOBS_FILE))) > before_saves
        lost = lost_runtime_answer(res, with_tools=True) and not (queued or submitted or saves)
        continuing = dispatched or (lost and len(attempts) < 2)
        state: SessionState = (
            "waiting" if queued else "done" if submitted else "failed" if not res.ok else "ended"
        )
        record = SessionRecord(
            number=number,
            state=state,
            export=str(export),
            run_dir=res.run_dir,
            ok=res.ok,
            error=res.error,
            seconds=sum(a["seconds"] for a in attempts),
            usage=res.usage,
            queued=queued,
            submitted=submitted,
            report=res.result if isinstance(res.result, dict) else None,
            tool_calls=len(res.tool_log),
            attempts=attempts,
            rerun=len(attempts) == 2,
            lost_twice=lost and len(attempts) == 2 and not dispatched,
            window=res.meta.get("window"),
            in_progress=continuing,
        )
        _save(analysis, record)
        return record

    def started(call_dir: Path) -> None:
        # Dispatch consumes a start, including when the caller dies before the
        # result is written. A failed preflight consumes none.
        pending = AgentResult(
            False,
            None,
            {},
            0,
            error="timeout",
            error_detail="caller interrupted after dispatch; result unavailable",
            run_dir=str(call_dir),
            meta={"pending": True},
        )
        attempts.append(attempt_record(pending, with_tools=True))
        record_result(pending, dispatched=True)
        if on_start is not None:
            on_start(call_dir)

    if attempts:
        res = _recover_attempt(attempts[-1], agent.backend)
        attempts[-1] = attempt_record(res, with_tools=True)
        record = record_result(res)
        if not record.in_progress:
            return record

    while len(attempts) < 2:
        if attempts:
            discarded = checkpoint / "lost-start-1"
            if not discarded.exists():
                discarded.mkdir()
                for part in ("memory", "outbox"):
                    shutil.copytree(analysis.root / part, discarded / part, symlinks=True)
            # Only these two writable trees carry between CLI boxes. Each call
            # gets a fresh CLI cwd/home; its old box remains in its run directory.
            for part in ("memory", "outbox"):
                shutil.rmtree(analysis.root / part)
                shutil.copytree(before / part, analysis.root / part, symlinks=True)
        count = len(attempts)
        res = run_agent(
            agent.backend,
            agent.model,
            system + SESSION_RULES,
            prompt,
            tools=access,
            schema=SessionReport,
            effort=agent.effort,
            service_tier=agent.service_tier,
            thinking_tokens=agent.thinking_tokens,
            auth=agent.auth,
            timeout=agent.timeout,
            run_root=run_root,
            label=f"{analysis.analysis_id}/session-{number:02d}",
            on_start=started,
        )
        receipt = attempt_record(res, with_tools=True)
        if len(attempts) == count:  # stand-ins without a process-start callback
            attempts.append(receipt)
        else:
            attempts[-1] = receipt
        record = record_result(res)
        if not record.in_progress:
            return record
    raise AssertionError("a scored session must have at most two starts")


def _recover_attempt(receipt: dict[str, Any], backend: Backend) -> AgentResult:
    """Recover an ended child, or the evidence left by an interrupted dispatch."""
    run_dir = Path(receipt["run_dir"])
    result_file = run_dir / "result.json"
    if result_file.is_file():
        data = json.loads(result_file.read_text())
        return AgentResult(**{f.name: data[f.name] for f in fields(AgentResult) if f.name in data})
    res = AgentResult(
        bool(receipt["ok"]),
        None,
        receipt["usage"],
        receipt["seconds"],
        error=receipt["error"],
        error_detail=receipt["detail"],
        run_dir=receipt["run_dir"],
        meta={**receipt["runtime_meta"]},
    )
    if res.meta.pop("pending", False):
        # A host interruption may occur after the final message or submission.
        # Preserve that evidence without inventing token/time measurements.
        stream = run_dir / "stdout.jsonl"
        lines = stream.read_text().splitlines() if stream.is_file() else []
        if backend == "codex":
            parse_codex(lines, res, run_dir / "box" / "last_message.txt")
        else:
            parse_claude(lines, res)
            if res.error == "cli_error" and res.error_detail == "no result event":
                res.error, res.error_detail = receipt["error"], receipt["detail"]
        log = run_dir / "box" / "toolbox" / "tool_calls.jsonl"
        res.tool_log = list(read_lines(log))
        res.meta["interrupted_without_result"] = True
    return res


def _save(analysis: Analysis, record: SessionRecord) -> None:
    path = analysis.root / "sessions" / f"session-{record.number:02d}.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(asdict(record), indent=1, default=str))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_analysis(
    analysis: Analysis,
    export: Path,
    agent: AgentSpec,
    system: str,
    task: str,
    env: ToolEnv,
    engine: Engine | None,
    *,
    max_sessions: int = 4,
    prompt_for: Callable[[int, str], str] | None = None,
    tools: tuple[str, ...] | None = None,
    max_experiment_episodes: int | None = None,
    run_root: Path | None = None,
    check_arms: bool = True,
    audit: AuditContext | None = None,
    limits: Mapping[str, int] | None = None,
) -> AnalysisRecord:
    """Sessions until one ends without queued experiments, or `max_sessions`.

    After a "waiting" session the queue goes to the engine and the next session
    starts on the engine's export. Without an engine the analysis stops at the
    first "waiting" session, with the queue file written for whoever runs it.
    With `check_arms` an engine reply whose arm episodes are missing from its
    export or lack the experiment's id raises `EngineError` (`arm_problems`).
    """
    previous = analysis.session_records()
    if previous and previous[-1].lost_twice:
        return AnalysisRecord(analysis.analysis_id, analysis.world_id, previous)
    record = AnalysisRecord(analysis.analysis_id, analysis.world_id)
    prompt_for = prompt_for or (lambda n, t: t if n == 1 else continuation_prompt(n, t))
    current = export
    for _ in range(max_sessions):
        number = analysis.sessions_done() + 1
        session = run_session(
            analysis,
            current,
            agent,
            system,
            prompt_for(number, task),
            env,
            tools=tools,
            max_experiment_episodes=max_experiment_episodes,
            run_root=run_root,
            audit=audit,
            limits=limits,
        )
        record.sessions.append(session)
        if session.error in ("auth", "rate_limit") or session.state != "waiting":
            break
        written = analysis.write_queue(session.number, current)
        if written is None:
            break
        queue_file, queue = written
        session.queue_file = str(queue_file)
        _save(analysis, session)
        if engine is None:
            break
        reply = engine.run_queue(queue_file, queue)
        if check_arms:
            problems = reply_problems(reply, queue, audit.experiment_days if audit else None)
            if problems:
                raise EngineError("; ".join(problems))
        analysis.record_results(reply.results)
        analysis.record_probe_outcomes(reply.probes)
        current = reply.export
    return record
