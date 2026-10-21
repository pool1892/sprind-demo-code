"""The agents' tools (docs/design/contracts.md, section 12), over one export and one analysis.

`ToolBox` implements every tool of `contracts.tools.TOOLS`. It runs inside the
inner sandbox, where the export is mounted read-only at `/work`, the skills at
`/skills`, a frozen library of learned skills (if the analysis has one) at
`/learned`, and the analysis' memory, outbox and results at `/memory`,
`/outbox` and `/results` (see `files`). Every call is appended to the call log
with its arguments and the full text it returned (images by their hash), so
that every number in an answer can be traced to a tool result.

Blinding rests here as much as on the sandbox: `query` runs on a
`LogReader` that copies the visible tables into memory and then switches
DuckDB's file access off and locks its configuration (`materialize=True`
with the lock; agent-runtime spike: an unlocked DuckDB read the canary within
three calls; log review: views with the lock can still read and write under
the export). One reader serves the whole session, and `query` accepts one
SELECT per call and nothing else: not even EXPLAIN, because DuckDB classifies
`EXPLAIN ANALYZE INSERT ...` as EXPLAIN and runs the statement inside, which
would let an agent change the session's tables and fabricate evidence (review
finding S1). So a query's result depends on the export alone and the scorer
can re-run it.

The experiment budget is the contracts' `EXPERIMENT_BUDGET` (at most 8
requests per analysis; `ExperimentRequest` itself refuses more than 60
episodes per arm) and a cap on the episodes of all requests together
(`Limits.max_experiment_episodes`, 960 unless the runtime raises it). A
runtime may set every limit (`Limits`), including fewer episodes per arm, and
may start the count at a later session (`budget_first_session`), which is how
the round driver (`fleetlab.loop.driver`) gives each round of one engagement
its own attempt budget.

Learned skills (docs/design/loop.md, section 9): `write_learned_skill` checks a
draft (`contracts.learned.draft_problems`; each SQL one SELECT that runs once
on this fleet with the draft's check values, which are not stored) and appends
it to the outbox; the host checks it against the truth side and freezes it.
`list_skills` and `read_skill` serve the base skills and, after them, the
mounted learned library, whose manifests carry `origin`, `applies_when` and
`provenance`.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from PIL import Image
from pydantic import BaseModel, ValidationError

from fleetlab.contracts.day_layout import ARM_SLOTS
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.jobs import JobBudget
from fleetlab.contracts.learned import (
    MAX_LEARNED_SKILLS,
    LearnedSkillDraft,
    LearnedSkillReceipt,
    draft_problems,
    render_learned_skill,
    sql_parameters,
    typed_value,
)
from fleetlab.contracts.probes import (
    PROBE_BOUNDS,
    PROBE_MESSAGES,
    ProbeOutcome,
    ProbeReceipt,
    ProbeRequest,
    ProbeResult,
)
from fleetlab.contracts.skills import SKILL_FILE, load_skill
from fleetlab.contracts.tools import (
    DESCRIBE_SIZE_CAP,
    EXPERIMENT_BUDGET,
    JOB_TOOLS,
    OPT_IN_TOOLS,
    TOOLS,
    DescribeTooLarge,
    Diagnosis,
    ExperimentBudgetStatus,
    ExperimentRequest,
    ExperimentResult,
    RankedFixSet,
    Receipt,
    RoundDecision,
    ToolSpec,
    describe_part,
    json_schema_strict,
)
from fleetlab.contracts.vocabulary import Verb
from fleetlab.log import LogReader
from fleetlab.toolserver.action_validation import public_fleet, request_problems
from fleetlab.toolserver.decisions import (
    DECISION_FILE,
    budget_status,
    decision_problems,
    read_public_ledger,
)
from fleetlab.toolserver.files import (
    DIAGNOSIS_FILE,
    EXPERIMENTS_FILE,
    FIX_SET_FILE,
    LEARNED_FILE,
    NOTES_FILE,
    PROBES_FILE,
    RESULTS_FILE,
    SUBMISSIONS_FILE,
    append_line,
    read_lines,
    replace_text,
)
from fleetlab.toolserver.jobs import JOB_FRAMES_NOTE, JobTools, job_description
from fleetlab.toolserver.mcp import Tool, ToolResult, image_part, text_part
from fleetlab.toolserver.public_results import load_probe_outcomes, load_results
from fleetlab.toolserver.public_system import public_description, public_dictionary

__all__ = ["Limits", "Paths", "ToolBox", "default_tool_names", "tool_description", "tool_names"]

SERVER_NAME = "robots"
SERVER_VERSION = "1"


def tool_names() -> tuple[str, ...]:
    """Every tool of the contracts, in their order."""
    return tuple(spec.name for spec in TOOLS)


def default_tool_names() -> tuple[str, ...]:
    """The tools of a session whose runtime names none: every tool but the opt-in ones."""
    return tuple(name for name in tool_names() if name not in OPT_IN_TOOLS)


@dataclass(frozen=True)
class Paths:
    """Where the tool server finds what it serves (paths inside the tool sandbox)."""

    export: Path  # the export: log/ and data_dictionary.md, read-only
    memory: Path  # this analysis' memory directory, read-write
    outbox: Path  # experiment requests and submissions, read-write
    results: Path  # experiment results written by the runtime, read-only
    skills: Path | None = None  # the skill library, read-only
    log: Path | None = None  # the call log of this session
    learned: Path | None = None  # a frozen library of learned skills, read-only


@dataclass(frozen=True)
class Limits:
    """What one call may ask for."""

    max_rows: int = 200  # rows a query returns
    max_frames: int = 8  # frames per get_frames call
    query_seconds: float = 60.0  # a query is interrupted after this
    duckdb_threads: int = 4
    duckdb_memory: str = "4GB"
    max_notes_bytes: int = 64 * 1024
    max_experiment_requests: int = EXPERIMENT_BUDGET["requests"]  # per analysis
    # Episodes over all requests of an analysis: by default the budget's requests times
    # two arms times its episodes per arm (960); the runtime may raise it.
    max_experiment_episodes: int = (
        EXPERIMENT_BUDGET["requests"] * 2 * EXPERIMENT_BUDGET["episodes_per_arm"]
    )
    # At most this many episodes in one arm; the request model refuses more than the
    # contracts' bound in any case, so a runtime can only lower it.
    max_episodes_per_arm: int = EXPERIMENT_BUDGET["episodes_per_arm"]
    # Requests of sessions before this one do not count against the limits above: an
    # engagement run in rounds gives each round its own budget (fleetlab.loop.driver).
    budget_first_session: int = 1
    # Probe requests per budget period, where `request_probe` is offered; their episodes
    # are charged to `max_experiment_episodes` with the experiments' (contracts.probes).
    max_probe_requests: int = 4


class ToolError(Exception):
    """A problem the agent caused and can fix: reported as a tool error, not a crash."""


class ToolBox:
    """The tools of one session: one export, one analysis directory, one call log."""

    def __init__(
        self,
        paths: Paths,
        enabled: Sequence[str] | None = None,
        *,
        session: int = 1,
        limits: Limits | None = None,
    ) -> None:
        self.paths = paths
        self.session = session
        self.limits = limits or Limits()
        known = tool_names()
        # Without a choice every tool but the opt-in ones (the probes) is offered.
        chosen = tuple(enabled) if enabled is not None else default_tool_names()
        unknown = sorted(set(chosen) - set(known))
        if unknown:
            raise ValueError(f"unknown tools: {unknown}")
        self.enabled = tuple(name for name in known if name in chosen)
        self._specs = {spec.name: spec for spec in TOOLS}
        self._handlers: dict[str, Callable[[Any], ToolResult]] = {
            "query": self._query,
            "get_frames": self._get_frames,
            "describe": self._describe,
            "list_skills": self._list_skills,
            "read_skill": self._read_skill,
            "request_experiment": self._request_experiment,
            "experiment_results": self._experiment_results,
            "experiment_budget": self._experiment_budget,
            "earlier_decisions": self._earlier_decisions,
            "submit_diagnosis": self._submit_diagnosis,
            "submit_fix_set": self._submit_fix_set,
            "submit_decision": self._submit_decision,
            "memory_read": self._memory_read,
            "memory_write": self._memory_write,
            "write_learned_skill": self._write_learned_skill,
            "request_probe": self._request_probe,
            "probe_results": self._probe_results,
        }
        # The job tools (fleetlab.toolserver.jobs; docs/design/jobs.md, section 3).
        self._jobs = JobTools(self)
        self._handlers.update(self._jobs.handlers())
        self._seq = 0
        self._log_lock = threading.Lock()
        self._world: tuple[set[str], set[str]] | None = None
        self._reader: LogReader | None = None

    # The MCP side.

    def mcp_tools(self) -> list[Tool]:
        jobs = any(name in JOB_TOOLS for name in self.enabled)
        return [
            Tool(
                name=name,
                description=tool_description(
                    self._specs[name],
                    self.limits,
                    job_budget=self._jobs.budget() if name in JOB_TOOLS else None,
                )
                + (JOB_FRAMES_NOTE if jobs and name == "get_frames" else ""),
                input_schema=json_schema_strict(self._specs[name].arguments),
                handler=functools.partial(self.call, name),
            )
            for name in self.enabled
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one tool: validate the arguments, run it, write the call log."""
        started = time.perf_counter()
        if self.paths.log is not None:
            append_line(
                self.paths.log.with_name("tool_requests.jsonl"),
                {"at": time.time(), "tool": name, "arguments": arguments},
            )
        error: str | None = None
        if name not in self.enabled:
            result = ToolResult.text(f"tool {name!r} is not available", is_error=True)
            error = "not enabled"
        else:
            try:
                parsed = self._specs[name].arguments.model_validate(arguments)
                result = self._handlers[name](parsed)
            except ValidationError as exc:
                error = _validation_text(exc)
                result = ToolResult.text(f"invalid arguments: {error}", is_error=True)
            except ToolError as exc:
                error = str(exc)
                result = ToolResult.text(error, is_error=True)
            except Exception as exc:  # DuckDB errors, missing files: the agent sees the message
                error = f"{type(exc).__name__}: {exc}"
                result = ToolResult.text(error, is_error=True)
        self._log(name, arguments, result, error, time.perf_counter() - started)
        return result

    def _log(
        self,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        error: str | None,
        seconds: float,
    ) -> None:
        if self.paths.log is None:
            return
        texts = [part["text"] for part in result.content if part.get("type") == "text"]
        images = [
            {"sha256": hashlib.sha256(base64.b64decode(part["data"])).hexdigest()[:16]}
            for part in result.content
            if part.get("type") == "image"
        ]
        with self._log_lock:
            self._seq += 1
            append_line(
                self.paths.log,
                {
                    "seq": self._seq,
                    "session": self.session,
                    "at": round(time.time(), 3),
                    "tool": name,
                    "arguments": arguments,
                    "ok": not result.is_error,
                    "error": error,
                    "result": texts,
                    "images": images,
                    "seconds": round(seconds, 4),
                },
            )

    # The export.

    def reader(self) -> LogReader:
        """The session's reader: the visible tables in memory, file access off and locked."""
        if self._reader is None:
            reader = LogReader(self.paths.export, materialize=True)
            reader.con.execute(f"SET threads = {int(self.limits.duckdb_threads)}")
            reader.con.execute(f"SET memory_limit = '{self.limits.duckdb_memory}'")
            reader.lock(materialized=True)
            self._reader = reader
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _world_ids(self) -> tuple[set[str], set[str]]:
        """The robot and site ids of the fleet, from the system description."""
        if self._world is None:
            fleet = self.reader().description(include_logs=False)["fleet"]
            sites = {site["id"] for site in fleet.get("sites", [])}
            robots = {r["id"] for site in fleet.get("sites", []) for r in site.get("robots", [])}
            self._world = (robots, sites)
        return self._world

    def _episode_ids(self) -> set[str]:
        return set(self.reader().episode_ids())

    def _query(self, args: Any) -> ToolResult:
        limit = self.limits.max_rows
        con = self.reader().con
        statements = con.extract_statements(args.sql)
        if len(statements) != 1:
            raise ToolError(f"one SQL statement per call, not {len(statements)}")
        kind = statements[0].type
        if kind not in _READ_STATEMENTS:
            raise ToolError(f"only SELECT is allowed, not {kind.name}")
        timer = threading.Timer(self.limits.query_seconds, con.interrupt)
        timer.start()
        try:
            cursor = con.execute(args.sql)
            columns = [c[0] for c in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit + 1) if cursor.description else []
        finally:
            timer.cancel()
        out = {
            "columns": columns,
            "rows": [list(row) for row in rows[:limit]],
            "truncated": len(rows) > limit,
        }
        return ToolResult.text(json.dumps(out, default=str, ensure_ascii=False))

    def _get_frames(self, args: Any) -> ToolResult:
        if len(args.steps) > self.limits.max_frames:
            raise ToolError(f"at most {self.limits.max_frames} frames per call")
        if self._jobs.is_job_episode(args.episode_id):  # a job episode's scene video
            return self._jobs.frames(args.episode_id, args.steps, args.camera)
        try:
            frames = self.reader().frames(args.episode_id, args.steps, args.camera)
        except KeyError as exc:
            raise ToolError(str(exc).strip("'\"")) from None
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        parts: list[dict[str, Any]] = []
        for frame in frames:
            parts.append(
                text_part(
                    f"{args.episode_id}, {args.camera.value} camera, step {frame.step}: "
                    f"the logged frame at step {frame.frame_step} (video frame {frame.frame_index})"
                )
            )
            buffer = io.BytesIO()
            Image.fromarray(frame.image).save(buffer, format="PNG")
            parts.append(image_part(base64.b64encode(buffer.getvalue()).decode()))
        return ToolResult(parts)

    def _describe(self, args: Any) -> ToolResult:
        """One part of the dictionary and the description, as compact JSON.

        At most `DESCRIBE_SIZE_CAP` (30,000) characters per answer, refused above with a
        pointer to the smaller parts and the tables. Claude Code 2.1.293 accepts an MCP
        result of up to 25,000 tokens (`MAX_MCP_OUTPUT_TOKENS`); it passes a result of at
        most 50,000 characters without counting (characters / 4 at most half the limit),
        counts tokens through the API above that and refuses above 25,000 tokens, saving
        the result to a file the sandboxed agent cannot read. The miniature fleet's whole
        description, 92,874 characters, was refused. The cap keeps every answer well
        inside the uncounted band, also for text that tokenizes densely.
        """
        reader = self.reader()
        try:
            text = describe_part(
                args.part,
                public_description(reader.description()),
                public_dictionary(reader.data_dictionary()),
                args.include_logs,
            )
        except DescribeTooLarge as error:
            raise ToolError(str(error)) from None
        return ToolResult.text(text)

    # Skills.

    def _skill_dirs(self, root: Path | None = None) -> list[Path]:
        root = self.paths.skills if root is None else root
        if root is None or not root.is_dir():
            return []
        return sorted(p for p in root.iterdir() if (p / SKILL_FILE).is_file())

    def _learned_dirs(self) -> list[Path]:
        """The mounted learned skills; a name the base library has stays the base skill's."""
        if self.paths.learned is None:
            return []
        base = {d.name for d in self._skill_dirs()}
        return [d for d in self._skill_dirs(self.paths.learned) if d.name not in base]

    def _list_skills(self, args: Any) -> ToolResult:
        manifests = [
            load_skill(d).manifest.model_dump(mode="json", exclude_none=True)
            for d in [*self._skill_dirs(), *self._learned_dirs()]
        ]
        return ToolResult.text(json.dumps(manifests, ensure_ascii=False))

    def _read_skill(self, args: Any) -> ToolResult:
        by_name = {d.name: d for d in [*self._learned_dirs(), *self._skill_dirs()]}
        if args.name not in by_name:
            raise ToolError(f"no skill named {args.name!r}; list_skills names them")
        skill = load_skill(by_name[args.name])
        text = (skill.root / SKILL_FILE).read_text(encoding="utf-8")
        if skill.helpers:
            names = ", ".join(p.relative_to(skill.root).as_posix() for p in skill.helpers)
            text += f"\n\n(Helper files beside this skill, not callable as tools: {names}.)\n"
        return ToolResult.text(text)

    # Experiments.

    def _requests(self) -> list[dict[str, Any]]:
        return list(read_lines(self.paths.outbox / EXPERIMENTS_FILE))

    def _scope_problems(self, scopes: Sequence[str]) -> list[str]:
        robots, sites = self._world_ids()
        known = robots | sites | {"fleet"}
        return [
            f"no robot or site {scope!r} in this fleet" for scope in scopes if scope not in known
        ]

    def _request_experiment(self, request: ExperimentRequest) -> ToolResult:
        problems = request_problems(request, public_fleet(self.reader().description()))
        problems += self._scope_problems(
            [a.target.scope for arm in request.arms for a in arm.actions]
        )
        problems += [
            f"{a.verb.value} has no simulated effect and cannot be tested by experiment"
            for arm in request.arms
            for a in arm.actions
            if a.verb in (Verb.route_to_teleop, Verb.request_training_data)
        ]
        if request.episodes_per_arm > self.limits.max_episodes_per_arm:
            problems.append(
                f"the experiment budget allows at most {self.limits.max_episodes_per_arm} "
                f"episodes per arm ({self._budget_text()})"
            )
        requests = self._requests()
        status = self._budget_status(requests)
        if status.requests_used >= self.limits.max_experiment_requests:
            problems.append(
                f"the experiment budget is {self.limits.max_experiment_requests} requests per "
                f"analysis; {status.requests_used} are requested already ({self._budget_text()})"
            )
        cap = self.limits.max_experiment_episodes
        used = status.attempts_used
        if used + _episodes(request) > cap:
            problems.append(
                f"the experiment budget is {cap} episodes per analysis; {used} are requested "
                f"already and this request needs {_episodes(request)} ({self._budget_text()})"
            )
        if problems:
            return _receipt(Receipt(accepted=False, receipt_id="", problems=problems))
        experiment_id = neutral_id(IdKind.experiment, len(requests) + 1)
        append_line(
            self.paths.outbox / EXPERIMENTS_FILE,
            {
                "experiment_id": experiment_id,
                "session": self.session,
                "request": request.model_dump(mode="json"),
            },
        )
        return _receipt(Receipt(accepted=True, receipt_id=experiment_id, problems=[]))

    def _budget_text(self) -> str:
        return budget_text(self.limits)

    def _budget_status(self, requests: list[dict[str, Any]]) -> ExperimentBudgetStatus:
        """What the budget's period has charged: refused requests and requests of sessions
        before `budget_first_session` cost nothing (`toolserver.decisions.budget_status`).
        Probe episodes are charged to the same attempts and reported apart."""
        refused = {
            result.experiment_id
            for result in self._validated_results()
            if result.status == "refused"
        }
        # A probe refused before it ran is free; one whose run broke off stays charged,
        # executed and reserved (contracts.probes.ProbeOutcome).
        return budget_status(
            self.limits,
            requests,
            refused,
            probes=self._probe_requests(),
            probe_outcomes={o.probe_id: o for o in self._probe_outcomes()},
            probes_offered="request_probe" in self.enabled,
        )

    def _experiment_budget(self, args: Any) -> ToolResult:
        status = self._budget_status(self._requests())
        return ToolResult.text(status.model_dump_json())

    def _validated_results(self) -> list[ExperimentResult]:
        try:
            rows = (
                self.reader()
                .query("SELECT episode_id, experiment_id, robot_id FROM episodes")
                .to_pylist()
            )
            return load_results(
                self.paths.results / RESULTS_FILE,
                self.paths.outbox / EXPERIMENTS_FILE,
                self.session,
                {r["episode_id"]: (r["experiment_id"], r["robot_id"]) for r in rows},
            )
        except ValueError:
            raise ToolError("experiment results failed public validation") from None

    def _experiment_results(self, args: Any) -> ToolResult:
        done = {result.experiment_id: result for result in self._validated_results()}
        out = []
        for record in self._requests():
            experiment_id = record["experiment_id"]
            result = done.get(experiment_id) or ExperimentResult(
                experiment_id=experiment_id,
                status="queued",
                arms=[],
                message="runs after this session ends; results in the next session",
            )
            out.append(result.model_dump(mode="json"))
        return ToolResult.text(json.dumps(out, ensure_ascii=False))

    # Probes (contracts.probes; docs/design/loop.md, section 10).

    def _probe_requests(self) -> list[dict[str, Any]]:
        return list(read_lines(self.paths.outbox / PROBES_FILE))

    def _probe_outcomes(self) -> list[ProbeOutcome]:
        """The runtime's outcomes of earlier sessions' probes, each bound to its request
        and to the export's probe rows (`public_results.load_probe_outcomes`); any problem
        refuses them as a whole, as for experiment results."""
        try:
            reader = self.reader()
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
            return load_probe_outcomes(
                self.paths.results / PROBES_FILE,
                self.paths.outbox / PROBES_FILE,
                self.session,
                rows,
                with_steps,
            )
        except ValueError:
            raise ToolError("probe results failed public validation") from None

    def _request_probe(self, request: ProbeRequest) -> ToolResult:
        robots, _ = self._world_ids()
        problems = [f"no robot {r!r} in this fleet" for r in request.robots if r not in robots]
        records = self._probe_requests()
        status = self._budget_status(self._requests())
        if status.probes_used >= self.limits.max_probe_requests:
            problems.append(
                f"at most {self.limits.max_probe_requests} probe requests per budget period; "
                f"{status.probes_used} are requested already"
            )
        cost = request.attempts
        if status.attempts_used + cost > self.limits.max_experiment_episodes:
            problems.append(
                f"this probe needs {cost} attempts (robots x repeats) and "
                f"{status.attempts_left} of the experiment budget's "
                f"{self.limits.max_experiment_episodes} are left ({self._budget_text()})"
            )
        steps = request.program.steps
        if problems:
            receipt = ProbeReceipt(
                accepted=False,
                probe_id="",
                problems=problems,
                attempts=0,
                steps_per_episode=steps,
                probe_attempts_used=status.probe_attempts_used,
                task_attempts_used=status.task_attempts_used,
                attempts_left=status.attempts_left,
            )
            return ToolResult.text(receipt.model_dump_json())
        probe_id = neutral_id(IdKind.probe, len(records) + 1)
        append_line(
            self.paths.outbox / PROBES_FILE,
            {"probe_id": probe_id, "session": self.session, "request": request.model_dump()},
        )
        receipt = ProbeReceipt(
            accepted=True,
            probe_id=probe_id,
            problems=[],
            attempts=cost,
            steps_per_episode=steps,
            probe_attempts_used=status.probe_attempts_used + cost,
            task_attempts_used=status.task_attempts_used,
            attempts_left=max(0, status.attempts_left - cost),
        )
        return ToolResult.text(receipt.model_dump_json())

    def _probe_results(self, args: Any) -> ToolResult:
        done = {o.probe_id: o for o in self._probe_outcomes()}
        out = []
        for record in self._probe_requests():
            request = ProbeRequest.model_validate(record["request"])
            outcome = done.get(record["probe_id"])
            result = ProbeResult(
                probe_id=record["probe_id"],
                status=outcome.status if outcome is not None else "queued",
                name=request.name,
                purpose=request.purpose,
                expectation=request.expectation,
                episode_ids=list(outcome.episode_ids) if outcome is not None else [],
                message=(
                    PROBE_MESSAGES[outcome.message]
                    if outcome is not None
                    else "runs after this session ends; results in the next session"
                ),
                attempts_charged=outcome.charged if outcome is not None else request.attempts,
                attempts_executed=outcome.attempts_executed if outcome is not None else 0,
                attempts_reserved=outcome.attempts_reserved if outcome is not None else 0,
            )
            out.append(result.model_dump(mode="json"))
        return ToolResult.text(json.dumps(out, ensure_ascii=False))

    # Submissions.

    def _submit(self, kind: str, answer: BaseModel, latest: str, problems: list[str]) -> ToolResult:
        if problems:
            return _receipt(Receipt(accepted=False, receipt_id="", problems=problems))
        count = sum(
            1 for r in read_lines(self.paths.outbox / SUBMISSIONS_FILE) if r["kind"] == kind
        )
        receipt_id = f"{kind}-{count + 1}"
        data = answer.model_dump(mode="json")
        append_line(
            self.paths.outbox / SUBMISSIONS_FILE,
            {"kind": kind, "receipt_id": receipt_id, "session": self.session, "answer": data},
        )
        replace_text(self.paths.outbox / latest, json.dumps(data, indent=1, ensure_ascii=False))
        return _receipt(Receipt(accepted=True, receipt_id=receipt_id, problems=[]))

    def _submit_diagnosis(self, diagnosis: Diagnosis) -> ToolResult:
        problems = self._scope_problems(
            [c.target.scope for c in diagnosis.causes] + [w.scope for w in diagnosis.warnings]
        )
        cited = {e.episode_id for c in diagnosis.causes for e in c.evidence if e.episode_id}
        cited |= {e for r in diagnosis.retrain for e in r.episode_ids}
        if cited:
            known = self._episode_ids()
            problems += [f"no episode {e!r} in the log" for e in sorted(cited - known)]
        return self._submit("diagnosis", diagnosis, DIAGNOSIS_FILE, problems)

    def _submit_fix_set(self, fix_set: RankedFixSet) -> ToolResult:
        # The model checks the rest (at least ten actions, values filled, none twice);
        # the list carries no budget: the scorer cuts it at every budget.
        problems = self._scope_problems([a.action.target.scope for a in fix_set.actions])
        return self._submit("fix_set", fix_set, FIX_SET_FILE, problems)

    # Learned skills.

    def _learned_drafts(self) -> dict[str, dict[str, Any]]:
        """The latest accepted draft of each learned skill of this analysis, by name."""
        latest: dict[str, dict[str, Any]] = {}
        for record in read_lines(self.paths.outbox / LEARNED_FILE):
            latest[record["draft"]["name"]] = record
        return latest

    def _learned_sql_problems(self, draft: LearnedSkillDraft) -> list[str]:
        """Each evidence SQL is one SELECT and runs once on this fleet with the check values."""
        problems: list[str] = []
        values = {c.name: typed_value(c.value) for c in draft.check_with}
        con = self.reader().con
        for i, item in enumerate(draft.evidence):
            where = f"evidence[{i}].sql"
            try:
                statements = con.extract_statements(item.sql)
            except duckdb.Error as error:
                problems.append(f"{where}: does not parse: {error}")
                continue
            if len(statements) != 1 or statements[0].type not in _READ_STATEMENTS:
                problems.append(f"{where}: one SELECT, as query takes it")
                continue
            names = sql_parameters(item.sql)
            missing = [n for n in names if n not in values]
            if missing:
                problems.append(
                    f"{where}: give a value of this fleet for {missing} in check_with, so that "
                    "the query runs once here (check values are not stored)"
                )
                continue
            timer = threading.Timer(self.limits.query_seconds, con.interrupt)
            timer.start()
            try:
                cursor = con.execute(item.sql, {n: values[n] for n in names} if names else None)
                if cursor.description:
                    cursor.fetchmany(self.limits.max_rows + 1)
            except duckdb.Error as error:
                problems.append(f"{where}: failed on this fleet: {error}")
            finally:
                timer.cancel()
        return problems

    def _learned_probe_problems(self, draft: LearnedSkillDraft) -> list[str]:
        """A carried probe can be requested on this fleet as the skill says: each robot
        parameter has a robot of this fleet in check_with (used once, not stored), and the
        request those robots make is within the probe's bounds."""
        if draft.probe is None:
            return []
        values = {c.name: c.value for c in draft.check_with}
        missing = [n for n in draft.probe.robot_parameters if n not in values]
        if missing:
            return [
                f"probe: give a robot of this fleet for {missing} in check_with, so that its "
                "request is checked here (check values are not stored)"
            ]
        named = [values[n] for n in draft.probe.robot_parameters]
        robots, _ = self._world_ids()
        problems = [f"probe: no robot {r!r} in this fleet" for r in named if r not in robots]
        if not problems:
            try:
                draft.probe.request(named)
            except ValueError as error:
                problems.append(f"probe: on {named} the request is refused: {error}")
        return problems

    def _write_learned_skill(self, draft: LearnedSkillDraft) -> ToolResult:
        problems = draft_problems(draft)
        if draft.name in {d.name for d in self._skill_dirs()}:
            problems.append(f"name: {draft.name!r} is a base skill; choose another name")
        drafts = self._learned_drafts()
        if draft.name not in drafts and len(drafts) >= MAX_LEARNED_SKILLS:
            problems.append(
                f"at most {MAX_LEARNED_SKILLS} learned skills per analysis ({sorted(drafts)} are "
                "written); update one of them by writing it again under its name"
            )
        if not problems:
            problems = self._learned_sql_problems(draft)
        if not problems:
            problems = self._learned_probe_problems(draft)
        if problems:
            receipt = LearnedSkillReceipt(
                accepted=False, receipt_id="", problems=problems, skill_md=""
            )
            return ToolResult.text(receipt.model_dump_json())
        kept = draft.model_copy(update={"check_with": []})
        count = sum(1 for _ in read_lines(self.paths.outbox / LEARNED_FILE))
        receipt_id = f"learned-{count + 1}"
        append_line(
            self.paths.outbox / LEARNED_FILE,
            {"receipt_id": receipt_id, "session": self.session, "draft": kept.model_dump()},
        )
        receipt = LearnedSkillReceipt(
            accepted=True, receipt_id=receipt_id, problems=[], skill_md=render_learned_skill(kept)
        )
        return ToolResult.text(receipt.model_dump_json())

    def _earlier_decisions(self, args: Any) -> ToolResult:
        """The agent's own accepted decisions, from its outbox (agent-authored state that
        the launch audit reads as agent prose), never from the task text."""
        out = [
            {
                "session": r.get("session"),
                "receipt_id": r.get("receipt_id"),
                "decision": r["answer"],
            }
            for r in read_lines(self.paths.outbox / SUBMISSIONS_FILE)
            if r.get("kind") == "decision"
        ]
        return ToolResult.text(json.dumps(out, ensure_ascii=False))

    def _submit_decision(self, decision: RoundDecision) -> ToolResult:
        # The round's decision, checked against what an operator can see: the public
        # fleet and the ledger the runtime published for the round (results/ledger.json).
        actions = [c.action for c in decision.adopt] + [r.action for r in decision.replace]
        actions += [r.action for r in decision.ranked_candidates]
        problems = self._scope_problems([a.target.scope for a in actions])
        if not problems:
            problems = decision_problems(
                decision,
                public_fleet(self.reader().description()),
                read_public_ledger(self.paths.results),
            )
        return self._submit("decision", decision, DECISION_FILE, problems)

    # Memory.

    def _memory_read(self, args: Any) -> ToolResult:
        path = self.paths.memory / NOTES_FILE
        notes = path.read_text(encoding="utf-8") if path.is_file() else ""
        return ToolResult.text(json.dumps({"notes": notes}, ensure_ascii=False))

    def _memory_write(self, args: Any) -> ToolResult:
        size = len(args.notes.encode())
        if size > self.limits.max_notes_bytes:
            return _receipt(
                Receipt(
                    accepted=False,
                    receipt_id="",
                    problems=[f"notes are {size} bytes; at most {self.limits.max_notes_bytes}"],
                )
            )
        replace_text(self.paths.memory / NOTES_FILE, args.notes)
        return _receipt(Receipt(accepted=True, receipt_id=f"notes-{self.session}", problems=[]))


_READ_STATEMENTS = frozenset({duckdb.StatementType.SELECT})


def budget_text(limits: Limits) -> str:
    """The experiment budget in the words the agent reads."""
    text = (
        f"at most {limits.max_experiment_requests} requests and "
        f"{limits.max_experiment_episodes} episodes per analysis, at most "
        f"{limits.max_episodes_per_arm} episodes per arm; episodes count "
        "episodes_per_arm times arms"
    )
    if limits.budget_first_session > 1:
        text += (
            f"; counted from session {limits.budget_first_session}: requests of earlier "
            "sessions do not count"
        )
    return text


def _episodes(request: ExperimentRequest) -> int:
    """What a request costs against the experiment budget: episodes over all arms."""
    return request.episodes_per_arm * len(request.arms)


def _receipt(receipt: Receipt) -> ToolResult:
    return ToolResult.text(receipt.model_dump_json(), is_error=False)


def _validation_text(exc: ValidationError) -> str:
    parts = []
    for item in exc.errors()[:10]:
        where = ".".join(str(p) for p in item.get("loc", ())) or "(arguments)"
        parts.append(f"{where}: {item.get('msg')}")
    return "; ".join(parts)


_EXTRA = {
    "query": (
        " One SELECT per call. Tables and columns: describe with part dictionary; the "
        "action catalogue is the table catalogue. At most {rows} rows."
    ),
    "describe": (
        " Each answer is at most {cap:,} characters; a larger part (or all, on a large "
        "fleet) is refused with a pointer to the smaller parts and to the tables "
        "change_log, maintenance and catalogue."
    ),
    "get_frames": " At most {frames} steps per call; the nearest logged frame is returned.",
    "write_learned_skill": (
        " SQL takes parameters as $name, prose as {{name}}; declare each under parameters. "
        "check_with gives a value of this fleet for each SQL parameter, used once to run the "
        "query and not stored. Name no robot, site, episode or experiment of this fleet. "
        "Returns a receipt with the SKILL.md as it is kept; the host checks it and adds its "
        "provenance after the investigation. A skill can carry the probe that showed its "
        "pattern: probe is request_probe's request with robot_parameters (declared "
        "parameters, in order) in place of robots; its evidence SQL reads the probe with "
        "$probe, and check_with gives a robot of this fleet for each robot parameter. "
        "Without one, probe is null."
    ),
    "request_experiment": (
        " Returns a receipt whose receipt_id is the experiment id. End the session after "
        "requesting: the experiments run between sessions. Budget: {budget}. A robot-day "
        "holds at most {arm_slots} experiment-arm episodes: a request's episodes go to its "
        "robots in turn, then to the experiment days, each arm on every one, and the "
        "earlier experiments and probes of this analysis on that robot and day count too; "
        "a request past that is refused with exceeds_robot_day_capacity."
    ),
    "request_probe": (
        " Name it, say what question it answers and what you expect to see. Targets are "
        "end-effector offsets from the position the robot reports at the probe's start, on "
        "x, y or z of the frame it reports in (z up); the orientation is held; the gripper "
        "starts open. step and ramp: amplitude_m is the offset reached; sine: the amplitude "
        "around the current offset, with period_steps; hold, open_gripper and close_gripper "
        "take null axis and amplitude_m. Bounds: offsets at most {offset} m, at most "
        "{speed} m/s in a ramp or sine, periods of at least {period} steps, segments of at "
        "most {segment} steps, at most {total} steps (20 per second) and {segments} segments "
        "a probe, at most {robots} robots and {repeats} repeats a request, at most {probes} "
        "probe requests a budget period. Each probe episode costs one attempt of the "
        "experiment budget ({budget}). The receipt's probe_id names the probe in the tables "
        "probe_episodes, probe_segments and probe_steps. End the session after requesting."
    ),
}


def tool_description(spec: ToolSpec, limits: Limits, job_budget: JobBudget | None = None) -> str:
    """What the agent reads about a tool; a job tool states `job_budget` (the request's
    budget; the plan's ceilings when None)."""
    if spec.name in JOB_TOOLS:
        return job_description(spec, job_budget, limits.max_rows)
    b = PROBE_BOUNDS
    extra = _EXTRA.get(spec.name, "").format(
        rows=limits.max_rows,
        frames=limits.max_frames,
        budget=budget_text(limits),
        cap=DESCRIBE_SIZE_CAP,
        offset=b.max_offset_m,
        speed=b.max_speed_m_s,
        period=b.min_period_steps,
        segment=b.max_segment_steps,
        total=b.max_probe_steps,
        segments=b.max_segments,
        robots=b.max_robots,
        repeats=b.max_repeats,
        probes=limits.max_probe_requests,
        arm_slots=ARM_SLOTS,
    )
    return f"{spec.description[0].upper()}{spec.description[1:]}. Returns {spec.returns}.{extra}"
