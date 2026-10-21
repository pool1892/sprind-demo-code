"""A batch of agent calls or analyses: the gate first, bounded concurrency, a stop on login errors.

    batch = Batch(env, backends=["claude", "codex"], concurrency=4)   # GateError when unchecked
    batch = Batch.after_gate(env, ["claude"], export, run_dirs, fleet=fleet,
                             texts={"system": SYSTEM, "task": TASK})  # runs the checks first
    results = batch.map(lambda item: run_agent(...), items)
    records = batch.analyses(jobs, engine)                             # sessions, queue, engine

A batch refuses to start unless every backend it uses has a passing gate
record for its pinned version, its binary and the tool environment
(`checks.require_verified`). Concurrency is capped at eight, the
most the agent-runtime spike measured without errors on both subscriptions.
An `auth` error stops the batch: no further calls start, because a login
problem needs a human; a `rate_limit` error stops it too,
because retrying into a subscription limit only burns the window. The
subscription is shared with every interactive session on the account, so a
batch also stops once a call reports the five-hour window fuller than
`stop_at_window` (claude reports it on every call; codex does not): on
2026-10-07 the window went from 88 % to its limit in twenty minutes while
other sessions worked, and the next calls were refused until it reset.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from fleetlab.agents.audit import AuditContext
from fleetlab.agents.checks import GateError, auth_check, require_verified, run_gate, source_audit
from fleetlab.agents.cli import Backend
from fleetlab.agents.run import AgentResult
from fleetlab.agents.sessions import AgentSpec, Analysis, AnalysisRecord, Engine, run_analysis
from fleetlab.agents.toolenv import ToolEnv
from fleetlab.contracts.model import Fleet

__all__ = ["MAX_CONCURRENCY", "STOP_ERRORS", "AnalysisJob", "Batch", "StopBatch"]

MAX_CONCURRENCY = 8
STOP_ERRORS = frozenset({"auth", "rate_limit"})

T = TypeVar("T")


class StopBatch(RuntimeError):
    """The batch stopped early; the message says why."""


@dataclass(frozen=True)
class AnalysisJob:
    """One agent's analysis of one world."""

    world_id: str
    export: Path
    agent: AgentSpec
    system: str
    task: str
    label: str = ""
    max_sessions: int = 4
    tools: tuple[str, ...] | None = None
    max_experiment_episodes: int | None = None
    audit: AuditContext | None = None


@dataclass
class Batch:
    env: ToolEnv
    backends: Sequence[Backend]
    concurrency: int = 4
    run_root: Path | None = None
    analyses_root: Path | None = None
    stop_at_window: float = 0.95  # utilization of a subscription window; 1.0: run into the limit
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    stopped: str | None = None
    audit: AuditContext | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"concurrency is 1 to {MAX_CONCURRENCY}, not {self.concurrency}")
        for backend in self.backends:
            self.records[backend] = require_verified(backend, self.env)
        self._lock = threading.Lock()

    @classmethod
    def after_gate(
        cls,
        env: ToolEnv,
        backends: Sequence[Backend],
        export: Path,
        run_dirs: Sequence[Path],
        *,
        fleet: Fleet | None = None,
        texts: Mapping[str, str] | None = None,
        rerun: bool = True,
        **kwargs: Any,
    ) -> Batch:
        """A batch after the free checks (before every batch): the full gate for
        every backend (about 25 s each; `rerun=False` runs it only where the record is
        missing or stale), then, for this batch, the source audit of its own export,
        prompts (`texts`) and mounted code, and the login check. GateError if any fails.
        """
        stale = []
        for backend in backends:
            try:
                require_verified(backend, env)
            except GateError:
                stale.append(backend)
        if rerun:
            stale = list(backends)
        if stale:
            run_gate(stale, env, export, run_dirs, fleet=fleet, texts=texts)
        audit = source_audit(export, run_dirs, env, fleet=fleet, texts=texts)
        if not audit.passed:
            raise GateError("source audit: " + "; ".join(audit.failures()))
        for backend in backends:
            login = auth_check(backend)
            if not login.passed:
                raise GateError(f"{backend}: not logged in; needs a human")
        kwargs.setdefault("audit", AuditContext(tuple(run_dirs), fleet))
        return cls(env, backends, **kwargs)

    def _check(self, result: Any) -> None:
        results = result.sessions if isinstance(result, AnalysisRecord) else [result]
        for item in results:
            error = getattr(item, "error", None)
            where = getattr(item, "run_dir", "?")
            reason = None
            if error in STOP_ERRORS:
                reason = f"{error} in {where}"
            meta = getattr(item, "meta", None) or {}
            window = meta.get("window") or getattr(item, "window", None) or {}
            fullest = max((u for u in (window.get("utilization") or {}).values() if u), default=0.0)
            if fullest >= self.stop_at_window:
                reason = reason or f"subscription window at {fullest:.0%} in {where}"
            if reason is not None:
                with self._lock:
                    if self.stopped is None:
                        self.stopped = reason

    def map(self, call: Callable[[T], Any], items: Sequence[T]) -> list[Any]:
        """`call(item)` for every item, `concurrency` at once; None for items never started."""

        def guarded(item: T) -> Any:
            if self.stopped is not None:
                return None
            result = call(item)
            self._check(result)
            return result

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(guarded, items))

    def analyses(
        self, jobs: Sequence[AnalysisJob], engine: Engine | None
    ) -> list[AnalysisRecord | None]:
        """Each job as a sequence of sessions, with a new analysis directory (fresh memory) each."""

        def one(job: AnalysisJob) -> AnalysisRecord:
            analysis = Analysis.create(job.world_id, job.label, base=self.analyses_root)
            return run_analysis(
                analysis,
                job.export,
                job.agent,
                job.system,
                job.task,
                self.env,
                engine,
                max_sessions=job.max_sessions,
                tools=job.tools,
                max_experiment_episodes=job.max_experiment_episodes,
                run_root=self.run_root,
                audit=job.audit or self.audit,
            )

        return self.map(one, jobs)

    @staticmethod
    def usage(results: Sequence[AgentResult | None]) -> dict[str, float]:
        """Tokens summed over the results of `map`, and claude's `cost_usd_list_price`: what the
        same calls would cost on the API-key route at list price (the estimate to make first)."""
        totals: dict[str, float] = {}
        for result in results:
            if result is None:
                continue
            for key, value in result.usage.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
        return totals
