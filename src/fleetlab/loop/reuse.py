"""The reuse comparison: one later case, with and without what the agent learned.

docs/design/loop.md, section 9. The same later case is investigated twice by
the same model with the same base skills, tools and budget: once fresh
("without") and once with a frozen library of learned skills mounted beside
the base skills ("with"). Each arm is a separate analysis created fresh for
the comparison (`Analysis.create`): no notes, outbox, results or sessions of
any other analysis reach it, and the library is the only thing transferred.
The harness enforces that and records, per arm, the digests of the export, the
base skills, the tool environment, the library mounted (or none), the model,
the budget and the prompts, and refuses to run when the arms differ in
anything but the library.

It then compares what each arm cost and what it achieved:

- sessions, tool calls (by tool), reads of learned skills;
- experiments requested, done and refused, attempts requested and simulated;
- model wall time and tokens, from the session records;
- the outcome, from a caller's `score` (the truth side: for example success on
  the reserved panel after the arm's decision is adopted) or, without one,
  the arm's public answer;
- negative transfer: the arm with the library scoring worse, or costing more
  for the same outcome.

The seam for the round driver is `runner`: a callable that runs one prepared
arm (`ArmSetup`) and may return costs of its own (attempts simulated by its
engine, refused requests). The default runs `fleetlab.agents.sessions.run_analysis`.

Costs are summed over every model start of every session, failed and replaced
starts included (`fleetlab.loop.costs.session_costs`, the loop's one total-cost
helper): a session's record keeps only its last start's usage.

Each comparison has its own id (`reuse.json` in `out`), and its arms' analysis
ids are `reuse-<id>-<arm>`, so two comparisons on one later world never share
an engine's per-analysis plans and exports. `compare(..., resume=True)` on the
same `out` reopens the same analyses, checks that their inputs are unchanged,
skips finished arms and continues an interrupted one: an in-progress session is
recovered by `run_session`, and a queue that was handed to the engine is
handed to it again from its saved file, so the engine resumes its own plan.

    comparison = compare(case, agent, env, library, out, budget=Budget(max_sessions=4),
                         engine_for=lambda arm: engine, score=my_scorer)
    comparison.write(out)                      # comparison.json and comparison.md
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fleetlab.agents.audit import AuditContext, tree_digest
from fleetlab.agents.run import call_log
from fleetlab.agents.sessions import (
    AgentSpec,
    Analysis,
    Engine,
    EngineError,
    EngineReply,
    ExperimentQueue,
    reply_problems,
    run_analysis,
)
from fleetlab.agents.toolenv import ToolEnv
from fleetlab.contracts.tools import ExperimentRequest
from fleetlab.loop.costs import session_costs
from fleetlab.loop.learned import FrozenLibrary, stored_drafts
from fleetlab.toolserver.files import read_lines

__all__ = [
    "ARMS",
    "ArmResult",
    "ArmSetup",
    "Budget",
    "LaterCase",
    "ReuseComparison",
    "ReuseError",
    "arm_costs",
    "compare",
    "public_outcome",
    "run_arm_analysis",
]

Arm = Literal["without", "with"]
ARMS: tuple[Arm, Arm] = ("without", "with")
STATE_FILE = "reuse.json"  # the comparison's id and its arms' analysis ids
STATE_FORMAT = "fleetlab.reuse-comparison/1"
# Inputs that must be identical in both arms; the library is the one that may differ.
SHARED_INPUTS = (
    "world_id",
    "world",
    "export",
    "base_skills",
    "tool_env",
    "agent",
    "budget",
    "system_sha256",
    "task_sha256",
    "audit",
    "runner",
)


class ReuseError(RuntimeError):
    """The arms are not comparable: they differ in something besides the library."""


@dataclass(frozen=True)
class LaterCase:
    """The later investigation both arms run: the same export, prompts and audit inputs.

    An analysis arm (`run_arm_analysis`) starts on `export`. An engagement arm
    (`fleetlab.loop.reuse_run.EngagementRunner`) starts on the stored later fleet at
    `world_root`, whose world hash is recorded instead; the round driver makes each
    round's export and task, and the harness checks that round 1's export is the same
    in both arms.
    """

    world_id: str
    export: Path | None
    system: str
    task: str  # carries the learning paragraph (`with_learning`) in both arms alike
    audit: AuditContext | None = None  # truth-side inputs of the launch audit (real runs)
    label: str = "later case"
    world_root: Path | None = None


@dataclass(frozen=True)
class Budget:
    """What each arm may spend; the same for both."""

    max_sessions: int = 4
    max_experiment_episodes: int | None = None
    tools: tuple[str, ...] | None = None  # None: every tool of the contracts


@dataclass(frozen=True)
class ArmSetup:
    """One prepared arm: its analysis (fresh, or reopened on resume) and what it runs with."""

    arm: Arm
    analysis: Analysis
    case: LaterCase
    agent: AgentSpec
    env: ToolEnv
    budget: Budget
    engine: Engine | None
    run_root: Path


ArmRunner = Callable[[ArmSetup], Mapping[str, Any] | None]
# An arm's outcome from its analysis and what its runner returned (the truth side's view).
Scorer = Callable[[Analysis, Mapping[str, Any]], Mapping[str, Any]]


def world_hash(root: Path) -> str:
    """The stored world's hash (`<files>/hash`), its identity without reading its runs."""
    from fleetlab.oracle.world import world_dir_of

    return (world_dir_of(root) / "hash").read_text().strip()


@dataclass
class _SavingEngine:
    """An engine whose replies are kept beside their queue files, for a resume."""

    inner: Engine

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        reply = self.inner.run_queue(queue_file, queue)
        saved = queue_file.with_suffix(".reply.json")
        if not saved.is_file():
            saved.write_text(reply.model_dump_json(indent=1) + "\n")
        return reply


def _finish_queue(setup: ArmSetup, queue_file: Path, engine: Engine) -> Path:
    """Hand an interrupted arm's saved queue to its engine again; the export to continue on."""
    analysis = setup.analysis
    queue = ExperimentQueue.model_validate_json(queue_file.read_text())
    saved = queue_file.with_suffix(".reply.json")
    if analysis.pending():
        reply = engine.run_queue(queue_file, queue)  # the engine finds its own saved plan
        audit = setup.case.audit
        problems = reply_problems(reply, queue, audit.experiment_days if audit else None)
        if problems:
            raise EngineError("; ".join(problems))
        done = analysis.results_by_id()
        analysis.record_results([r for r in reply.results if r.experiment_id not in done])
    elif saved.is_file():
        reply = EngineReply.model_validate_json(saved.read_text())
    else:
        raise ReuseError(f"{analysis.analysis_id}: results recorded but no saved reply to resume")
    return reply.export


def run_arm_analysis(setup: ArmSetup) -> dict[str, Any]:
    """The default runner: the analysis' sessions through `run_analysis`, resumable.

    A new arm starts on the case's export. A reopened arm continues where it
    stopped: an in-progress session on its own export, a waiting session's
    saved queue through the engine again; a finished arm is left as it is.
    The budget's sessions count every session of the arm, earlier ones too.
    """
    analysis = setup.analysis
    engine = _SavingEngine(setup.engine) if setup.engine is not None else None
    if setup.case.export is None:
        raise ReuseError("an analysis arm starts on an export; the case has none")
    export = setup.case.export
    records = analysis.session_records()
    if records:
        last = records[-1]
        if last.in_progress:
            export = Path(last.export)
        elif last.state == "waiting" and last.queue_file and engine is not None:
            export = _finish_queue(setup, Path(last.queue_file), engine)
        else:
            return {"final_state": last.state, "resumed": True}
    remaining = setup.budget.max_sessions - analysis.sessions_done()
    if remaining > 0:
        run_analysis(
            analysis,
            export,
            setup.agent,
            setup.case.system,
            setup.case.task,
            setup.env,
            engine,
            max_sessions=remaining,
            tools=setup.budget.tools,
            max_experiment_episodes=setup.budget.max_experiment_episodes,
            run_root=setup.run_root,
            audit=setup.case.audit,
        )
    final = analysis.session_records()
    return {"final_state": final[-1].state if final else "not started", "resumed": bool(records)}


@dataclass
class ArmResult:
    arm: Arm
    analysis: str
    inputs: dict[str, Any]
    costs: dict[str, Any]
    outcome: dict[str, Any]
    runner: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReuseComparison:
    case: str
    arms: dict[str, ArmResult]
    differences: dict[str, Any]
    negative_transfer: list[str]
    # Problems with the comparison itself (for example round 1's export differing between
    # engagement arms): a non-empty list means the arms were not comparable.
    validity: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "arms": {name: asdict(arm) for name, arm in self.arms.items()},
            "differences": self.differences,
            "negative_transfer": self.negative_transfer,
            "validity": self.validity,
        }

    def write(self, out: Path) -> None:
        (out / "comparison.json").write_text(json.dumps(self.as_json(), indent=1) + "\n")
        (out / "comparison.md").write_text(comparison_markdown(self))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _inputs(
    analysis: Analysis,
    case: LaterCase,
    agent: AgentSpec,
    env: ToolEnv,
    budget: Budget,
    library: FrozenLibrary | None,
    runner_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = case.audit
    inputs = {
        "world_id": case.world_id,
        "world": world_hash(case.world_root) if case.world_root is not None else None,
        "export": tree_digest(case.export) if case.export is not None else None,
        "base_skills": tree_digest(env.skills) if env.skills is not None else None,
        "tool_env": env.digest(),
        "agent": asdict(agent),
        "budget": asdict(budget),
        "system_sha256": _sha(case.system),
        "task_sha256": _sha(case.task),
        "audit": None
        if audit is None
        else {
            "run_dirs": [str(p) for p in audit.run_dirs],
            "fleet": audit.fleet.id if audit.fleet is not None else None,
            "experiment_days": audit.experiment_days,
        },
        "library": None if library is None else library.digest,
        "mounted_library": (
            tree_digest(analysis.learned) if analysis.learned is not None else None
        ),
        "runner": dict(runner_inputs) if runner_inputs is not None else None,
    }
    # The form inputs.json stores (tuples become lists), so that a resume compares like with like.
    serialized: dict[str, Any] = json.loads(json.dumps(inputs))
    return serialized


def _fresh_problems(analysis: Analysis) -> list[str]:
    """Anything an arm would inherit: notes, outbox records, results, sessions."""
    problems = []
    for part in ("memory", "outbox", "results", "queues", "sessions"):
        if any((analysis.root / part).iterdir()):
            problems.append(f"{analysis.analysis_id}: {part}/ is not empty at the start")
    return problems


def analysis_ids(comparison_id: str) -> dict[Arm, str]:
    """The arms' analysis ids of one comparison: unique per comparison, stable on resume."""
    return {arm: f"reuse-{comparison_id}-{arm}" for arm in ARMS}


def prepare_arms(
    case: LaterCase,
    agent: AgentSpec,
    env: ToolEnv,
    library: FrozenLibrary,
    out: Path,
    budget: Budget,
    ids: Mapping[Arm, str] | None = None,
    *,
    reopen: bool = False,
    runner_inputs: Mapping[str, Any] | None = None,
) -> dict[Arm, tuple[Analysis, dict[str, Any]]]:
    """Two analyses whose inputs differ only in the library; raises ReuseError otherwise.

    New analyses are created fresh (and checked to be empty); with `reopen`
    the existing ones under `out/analyses` are opened again for a resume.
    """
    library.verify()
    ids = ids or analysis_ids(uuid.uuid4().hex[:12])
    arms: dict[Arm, tuple[Analysis, dict[str, Any]]] = {}
    for arm in ARMS:
        if reopen:
            analysis = Analysis(out / "analyses" / ids[arm])
            if analysis.world_id != case.world_id:
                raise ReuseError(f"{ids[arm]} belongs to another world")
        else:
            analysis = Analysis.create(
                case.world_id,
                label=f"reuse {arm} the learned library",
                base=out / "analyses",
                analysis_id=ids[arm],
                learned=library.skills if arm == "with" else None,
            )
            problems = _fresh_problems(analysis)
            if problems:
                raise ReuseError("; ".join(problems))
        arms[arm] = (
            analysis,
            _inputs(
                analysis,
                case,
                agent,
                env,
                budget,
                library if arm == "with" else None,
                runner_inputs,
            ),
        )
    without, with_ = arms["without"][1], arms["with"][1]
    differ = [key for key in SHARED_INPUTS if without[key] != with_[key]]
    if differ:
        raise ReuseError(f"the arms differ in {differ}")
    if without["library"] is not None or without["mounted_library"] is not None:
        raise ReuseError("the arm without the library has one")
    if with_["mounted_library"] != library.digest:
        raise ReuseError("the mounted library is not the frozen one")
    return arms


def _session_usage(analysis: Analysis, model: str | None) -> dict[str, Any]:
    """Sessions, starts, tool calls, model time, tokens and price over every start.

    The totals, their completeness (`usage_complete`, `usage_note`: a start without a
    usage record makes the usage figures lower bounds) and the prices are
    `fleetlab.loop.costs.session_costs`, the loop's one cost helper.
    """
    sessions = analysis.session_records()
    totals = session_costs(sessions, model)
    return {
        "sessions": totals["sessions"],
        "session_states": [s.state for s in sessions],
        "starts": totals["model_starts"],
        "failed_starts": totals["failed_starts"],
        "starts_without_usage": totals["starts_without_usage"],
        "usage_complete": totals["usage_complete"],
        "usage_note": totals["usage_note"],
        "tool_calls": totals["tool_calls"],
        "model_seconds": round(totals["model_seconds"], 3),
        "tokens": totals["tokens"],
        "cost_usd_list_price": totals["cost_usd_list_price"],
        "cost_usd_list_price_status": totals["cost_usd_list_price_status"],
        "api_equivalent_usd": totals["api_equivalent_usd"],
        "per_session": totals["per_session"],
    }


def _tool_breakdown(analysis: Analysis) -> dict[str, Any]:
    """Calls by tool and reads of learned skills, from the tool server's call logs."""
    learned = set()
    if analysis.learned is not None:
        learned = {p.name for p in analysis.learned.iterdir() if p.is_dir()}
    by_tool: Counter[str] = Counter()
    learned_reads: Counter[str] = Counter()
    logged = 0
    for session in analysis.session_records():
        for attempt in session.attempts or [{"run_dir": session.run_dir}]:
            run_dir = attempt.get("run_dir")
            if not run_dir:
                continue
            for call in read_lines(call_log(Path(run_dir))):
                logged += 1
                by_tool[str(call.get("tool"))] += 1
                name = (call.get("arguments") or {}).get("name")
                if call.get("tool") == "read_skill" and name in learned:
                    learned_reads[str(name)] += 1
    return {
        "logged_calls": logged,
        "by_tool": dict(sorted(by_tool.items())),
        "learned_skill_reads": dict(learned_reads),
    }


def _experiments(analysis: Analysis) -> dict[str, Any]:
    results = analysis.results_by_id()
    requests = analysis.requests()
    refused = {i for i, r in results.items() if r.status == "refused"}
    done = [r for r in results.values() if r.status == "done"]

    def attempts(request: ExperimentRequest) -> int:
        return request.episodes_per_arm * len(request.arms)

    return {
        "requested": len(requests),
        "done": len(done),
        "refused": len(refused),
        "pending": len(requests) - len(results),
        "attempts_requested": sum(
            attempts(q.request) for q in requests if q.experiment_id not in refused
        ),
        "attempts_simulated": sum(len(a.episode_ids) for r in done for a in r.arms),
    }


def arm_costs(analysis: Analysis, model: str | None = None) -> dict[str, Any]:
    """What one arm spent, from its analysis directory alone; `model` prices its tokens."""
    return {
        **_session_usage(analysis, model),
        **_tool_breakdown(analysis),
        "experiments": _experiments(analysis),
        "learned_skills_written": sorted(stored_drafts(analysis.root)),
    }


def public_outcome(analysis: Analysis) -> dict[str, Any]:
    """The arm's answer as submitted, when no truth-side scorer is given."""
    diagnosis = analysis.diagnosis()
    fix_set = analysis.fix_set()
    return {
        "diagnosis_targets": (
            [c.target.model_dump(mode="json") for c in diagnosis.causes] if diagnosis else None
        ),
        "top_actions": (
            [r.action.model_dump(mode="json") for r in fix_set.actions[:3]] if fix_set else None
        ),
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _differences(arms: Mapping[str, ArmResult]) -> dict[str, Any]:
    without, with_ = arms["without"], arms["with"]
    pairs: dict[str, Callable[[ArmResult], Any]] = {
        "sessions": lambda a: a.costs["sessions"],
        "tool_calls": lambda a: a.costs["tool_calls"],
        "experiments_requested": lambda a: a.costs["experiments"]["requested"],
        "attempts_simulated": lambda a: a.costs["experiments"]["attempts_simulated"],
        "model_seconds": lambda a: a.costs["model_seconds"],
        "input_tokens": lambda a: a.costs["tokens"].get("input_tokens", 0),
        "output_tokens": lambda a: a.costs["tokens"].get("output_tokens", 0),
        "cost_usd_list_price": lambda a: a.costs["cost_usd_list_price"],
        "score": lambda a: a.outcome.get("score"),
    }
    out: dict[str, Any] = {}
    for key, get in pairs.items():
        a, b = _number(get(without)), _number(get(with_))
        partial = [arm.arm for arm in (without, with_) if not _complete(arm, key)]
        row: dict[str, Any] = {"without": a, "with": b}
        if partial:
            # A lower bound minus a lower bound is no difference at all.
            row["with_minus_without"] = None
            row["status"] = "unavailable"
            row["reason"] = f"incomplete in {partial}: " + "; ".join(
                str(arm.costs.get("usage_note") or f"{key} not reported")
                for arm in (without, with_)
                if arm.arm in partial
            )
        else:
            row["with_minus_without"] = None if a is None or b is None else b - a
            row["status"] = "complete" if a is not None and b is not None else "unavailable"
        out[key] = row
    return out


# The figures a start without a usage record leaves incomplete.
USAGE_FIGURES = frozenset({"model_seconds", "input_tokens", "output_tokens", "cost_usd_list_price"})


def _complete(arm: ArmResult, key: str) -> bool:
    if key == "cost_usd_list_price":
        return arm.costs.get("cost_usd_list_price_status") in ("complete", "unavailable") and bool(
            arm.costs.get("usage_complete", True)
        )
    return key not in USAGE_FIGURES or bool(arm.costs.get("usage_complete", True))


def _negative_transfer(differences: Mapping[str, Any], arms: Mapping[str, ArmResult]) -> list[str]:
    notes = []
    score = differences["score"]["with_minus_without"]
    if score is not None and score < 0:
        notes.append(f"the arm with the library scored {score:+.3f} below the arm without")
    same_or_worse = score is None or score <= 0
    for key in ("attempts_simulated", "tool_calls", "model_seconds"):
        delta = differences[key]["with_minus_without"]
        if delta is not None and delta > 0 and same_or_worse:
            notes.append(f"the arm with the library spent more {key} ({delta:+g})")
    if not arms["with"].costs["learned_skill_reads"] and arms["with"].costs["logged_calls"]:
        notes.append("the arm with the library never read a learned skill")
    return notes


def compare(
    case: LaterCase,
    agent: AgentSpec,
    env: ToolEnv,
    library: FrozenLibrary,
    out: Path,
    *,
    budget: Budget | None = None,
    engine_for: Callable[[Arm], Engine | None] = lambda arm: None,
    runner: ArmRunner = run_arm_analysis,
    score: Scorer | None = None,
    order: Sequence[Arm] = ARMS,
    resume: bool = False,
) -> ReuseComparison:
    """Run the later case in both arms and compare cost and outcome (see the module docstring).

    `out` must not exist, unless `resume`: then it holds an earlier call's
    comparison, whose arms are reopened with their recorded analysis ids and
    whose inputs must be unchanged. `engine_for` gives each arm its own engine
    (each arm's experiments run separately, from the same export). `score` reads
    an arm's analysis and its runner's record with the truth side and returns its
    outcome, with a number under `score` when the outcome is comparable; higher is
    better. A runner with an `inputs()` method adds what it runs with to both arms'
    recorded inputs, which must then be equal too.
    """
    budget = budget or Budget()
    runner_inputs = runner.inputs() if hasattr(runner, "inputs") else None
    if sorted(order) != sorted(ARMS):
        raise ValueError(f"order names each arm once: {ARMS}")
    if resume:
        state = json.loads((out / STATE_FILE).read_text())
        if state.get("format") != STATE_FORMAT or state.get("library") != library.digest:
            raise ReuseError(f"{out} is not a comparison with this library")
        recorded = json.loads((out / "inputs.json").read_text())
        arms = prepare_arms(
            case,
            agent,
            env,
            library,
            out,
            budget,
            state["analysis_ids"],
            reopen=True,
            runner_inputs=runner_inputs,
        )
        changed = [arm for arm, (_, inputs) in arms.items() if recorded[arm] != inputs]
        if changed:
            raise ReuseError(f"the inputs of {changed} changed since the comparison started")
    else:
        out.mkdir(parents=True, exist_ok=False)
        comparison_id = uuid.uuid4().hex[:12]
        state = {
            "format": STATE_FORMAT,
            "comparison_id": comparison_id,
            "analysis_ids": analysis_ids(comparison_id),
            "library": library.digest,
            "case": case.label,
            "created": time.time(),
        }
        (out / STATE_FILE).write_text(json.dumps(state, indent=1) + "\n")
        arms = prepare_arms(
            case,
            agent,
            env,
            library,
            out,
            budget,
            state["analysis_ids"],
            runner_inputs=runner_inputs,
        )
        (out / "inputs.json").write_text(
            json.dumps({arm: inputs for arm, (_, inputs) in arms.items()}, indent=1) + "\n"
        )
    (out / "arms").mkdir(exist_ok=True)
    results: dict[Arm, ArmResult] = {}
    for arm in order:
        analysis, inputs = arms[arm]
        finished = out / "arms" / f"{arm}.json"
        if finished.is_file():
            extra = json.loads(finished.read_text())
        else:
            setup = ArmSetup(
                arm, analysis, case, agent, env, budget, engine_for(arm), out / "agent-runs" / arm
            )
            extra = dict(runner(setup) or {})
            finished.write_text(json.dumps(extra, indent=1, default=str) + "\n")
        if case.export is not None and tree_digest(case.export) != inputs["export"]:
            raise ReuseError(f"the export changed while the {arm} arm ran")
        if case.world_root is not None and world_hash(case.world_root) != inputs["world"]:
            raise ReuseError(f"the stored later fleet changed while the {arm} arm ran")
        if analysis.learned is not None and tree_digest(analysis.learned) != inputs["library"]:
            raise ReuseError("the mounted library changed while the arm ran")
        outcome = dict(score(analysis, extra)) if score is not None else public_outcome(analysis)
        costs = arm_costs(analysis, agent.model)
        results[arm] = ArmResult(arm, str(analysis.root), inputs, costs, outcome, extra)
    ordered: dict[str, ArmResult] = {arm: results[arm] for arm in ARMS}
    differences = _differences(ordered)
    # A runner reports what makes its arm unusable (an aborted engagement, a skipped
    # evaluation); any such problem makes the whole comparison invalid.
    validity = [str(p) for arm in ARMS for p in ordered[arm].runner.get("validity") or []]
    first = [ordered[arm].runner.get("round_exports", {}).get("1") for arm in ARMS]
    if any(first) and first[0] != first[1]:
        validity.append(f"round 1's export differs between the arms: {first}")
    comparison = ReuseComparison(
        case.label, ordered, differences, _negative_transfer(differences, ordered), validity
    )
    comparison.write(out)
    return comparison


def comparison_markdown(comparison: ReuseComparison) -> str:
    """A short table of the two arms for a reader of the run directory."""
    arms = comparison.arms
    lines = [
        f"# Reuse comparison: {comparison.case}",
        "",
        "Both arms: the same export, model, base skills, tools, budget and prompts "
        f"(export {str(arms['without'].inputs['export'])[:12]}, world "
        f"{str(arms['without'].inputs.get('world'))[:12]}, base skills "
        f"{str(arms['without'].inputs['base_skills'])[:12]}). The arm with the library had "
        f"library {str(arms['with'].inputs['library'])[:12]} mounted.",
        "",
        "| | without | with | with minus without |",
        "|---|---|---|---|",
    ]
    for key, row in comparison.differences.items():
        if row.get("reason"):
            lines.append(
                f"| {key} | {row['without']} | {row['with']} | unavailable ({row['reason']}) |"
            )
        else:
            lines.append(
                f"| {key} | {row['without']} | {row['with']} | {row['with_minus_without']} |"
            )
    reads = arms["with"].costs["learned_skill_reads"]
    lines += ["", f"Learned skills read in the arm with the library: {reads or 'none'}."]
    if comparison.validity:
        lines += ["", "The arms were not comparable:", ""]
        lines += [f"- {note}" for note in comparison.validity]
    if comparison.negative_transfer:
        lines += ["", "Possible negative transfer:", ""]
        lines += [f"- {note}" for note in comparison.negative_transfer]
    return "\n".join(lines) + "\n"
