"""The reuse run: a library learned on the lead fleet, reused on the later fleet, with the
round driver (docs/design/reuse-run.md; docs/design/learned.md; loop.md, section 9).

The round driver learns by default: the round task carries the learning paragraph,
`write_learned_skill` is a round tool, and after the rounds it checks and freezes the
agent's skills (`fleetlab.loop.rounds.freeze_learned`; engagement.json `learned`). The
lead fleet's engagement is the source, and its frozen library is what this module reuses.

1. `compare`: the later fleet in two engagements of the round driver (`run_rounds`),
   each in a fresh analysis made by the reuse harness (`fleetlab.loop.reuse.compare`):
   identical in everything but the library, which only the "with" arm has mounted.
   Round 1's export must be the same bytes in both, and both engagements must have ended
   with a decision in every round and an evaluation, or the comparison is invalid.
2. `score`: the metrics registered in docs/design/reuse-run.md, from each arm's own
   submissions, decisions, call logs and reserved-panel attempts, against the later
   fleet's truth (`ConditionTruth`), host-side only. The identification metric is a
   proxy (the diagnosis targets the affected controller at the condition's site); the
   matching diagnosis is shown verbatim beside it, with the session's call log.

Nothing here is mounted into a sandbox or shown to an agent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.agents.audit import tree_digest
from fleetlab.agents.sessions import AgentSpec, Analysis
from fleetlab.contracts.actions import Action
from fleetlab.contracts.tools import Diagnosis
from fleetlab.loop.costs import session_costs
from fleetlab.loop.learned import FrozenLibrary
from fleetlab.loop.reuse import ArmSetup, Budget, LaterCase, ReuseError, compare

__all__ = [
    "ConditionTruth",
    "EngagementRunner",
    "ReuseScorer",
    "adoption",
    "between_arms",
    "engagement_problems",
    "identification",
    "main",
    "score_comparison",
]

ADAPT_KEY = "control.latency_compensation_steps"
DELAY_KEY = "control.action_delay_steps"


# ---- the truth key ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionTruth:
    """What the metrics match against: where the later fleet's lasting condition is.

    Host-side, derived from the later fleet's truth (`from_world`) and fixed before any
    run (docs/design/reuse-run.md, "Metrics"). `site` and `robots` are where the
    condition is; `lookalikes` are the sites (with their robots) that carry the same
    public note without the effect.
    """

    site: str
    robots: tuple[str, ...]
    lookalikes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    components: tuple[str, ...] = ("controller",)
    config_keys: tuple[str, ...] = (DELAY_KEY, ADAPT_KEY)
    # A cause naming no component or key counts when its failure mode says so.
    words: str = r"\b(late|latency|lag|lags|lagging|delay|delays|delayed)\b"
    min_confidence: float = 0.5
    adapt_verbs: tuple[str, ...] = ("set",)
    adapt_keys: tuple[str, ...] = (ADAPT_KEY,)
    effective_points: float = 10.0  # the condition robots' panel success, round k over original
    effective_p: float = 0.05  # McNemar's exact two-sided p on their paired panel attempts

    @classmethod
    def from_world(cls, world: Any) -> ConditionTruth:
        """The lasting remote-link condition of a loop fleet (`fleetlab.loop.world.LoopWorld`):
        its faults that are not decoys; decoy remote links mark the lookalike sites."""
        site_of = {r.id: r.site_id for s in world.model.sites for r in s.robots}
        decoys = set(world.spec.patterns.decoys)
        remote = [f for f in world.spec.fault_schedule if f.kind == "remote_link_latency"]

        def sites(faults: Sequence[Any]) -> set[str]:
            return {site_of.get(f.target.scope, f.target.scope) for f in faults}

        active = [f for f in remote if f.fault_id not in decoys]
        where = sites(active)
        if len(where) != 1:
            raise ValueError(f"expected one site with the lasting condition, found {where}")
        site = where.pop()

        def robots_at(where: str) -> tuple[str, ...]:
            return tuple(sorted(r for r, s in site_of.items() if s == where))

        lookalikes = sorted(sites([f for f in remote if f.fault_id in decoys]) - {site})
        return cls(
            site=site,
            robots=robots_at(site),
            lookalikes=tuple((other, robots_at(other)) for other in lookalikes),
        )

    @property
    def lookalike_sites(self) -> tuple[str, ...]:
        return tuple(site for site, _ in self.lookalikes)

    def _names(self, cause: Any) -> bool:
        target = cause.target
        if target.component is not None:
            return str(target.component) in self.components
        if target.config_key is not None:
            return str(target.config_key) in self.config_keys
        if target.condition_key is not None:
            return False
        return re.search(self.words, cause.failure_mode, re.IGNORECASE) is not None

    def targeting(self, diagnosis: Diagnosis, site: str, robots: Sequence[str]) -> list[Any]:
        """The causes of a diagnosis that target the controller or the delay keys at `site`
        (its scope, or at least half of its `robots`); empty when they do not. A proxy: it
        shows where the diagnosis points, not that its reasoning was right."""
        named = [
            c for c in diagnosis.causes if c.confidence >= self.min_confidence and self._names(c)
        ]
        at_site = [c for c in named if c.target.scope == site]
        on_robots = [c for c in named if c.target.scope in set(robots)]
        enough = bool(robots) and len({c.target.scope for c in on_robots}) >= math.ceil(
            len(robots) / 2
        )
        return at_site + (on_robots if enough else [])

    def sets_adaptation(self, action: Any) -> bool:
        """An action that sets the adaptation key to a non-zero value (anywhere)."""
        target = action.target
        return (
            str(action.verb) in self.adapt_verbs
            and target.config_key is not None
            and str(target.config_key) in self.adapt_keys
            and (str(action.verb) != "set" or bool(action.value))
        )

    def where(self, scope: str) -> str:
        """Where an action's scope lies: "condition", "lookalike", "fleet" or "elsewhere"."""
        if scope == self.site or scope in self.robots:
            return "condition"
        if any(scope == site or scope in robots for site, robots in self.lookalikes):
            return "lookalike"
        return "fleet" if scope == "fleet" else "elsewhere"

    def adapts(self, action: Any) -> bool:
        """Whether an action sets the adaptation at the condition's site or robots."""
        return self.sets_adaptation(action) and self.where(action.target.scope) == "condition"


# ---- the metrics -----------------------------------------------------------------------------


def _session_rounds(engagement: Mapping[str, Any]) -> dict[int, int]:
    return {
        int(s["number"]): int(entry["round"])
        for entry in engagement.get("rounds", [])
        for s in entry.get("sessions", [])
    }


def _attempts_before(analysis: Analysis, session: int) -> int:
    """Experiment attempts delivered before `session` started: requests of earlier sessions."""
    results = analysis.results_by_id()
    return sum(
        len(a.episode_ids)
        for q in analysis.requests()
        if q.session < session
        and (r := results.get(q.experiment_id)) is not None
        and r.status == "done"
        for a in r.arms
    )


def _through(analysis: Analysis, session: int, model: str | None) -> dict[str, Any]:
    """Model time and tokens of the sessions up to and including `session`, every start."""
    costs = session_costs([s for s in analysis.session_records() if s.number <= session], model)
    complete = costs["starts_without_usage"] == 0
    return {
        "model_seconds": costs["model_seconds"],
        "tokens": costs["tokens"],
        "usage_complete": complete,
        "note": None
        if complete
        else f"{costs['starts_without_usage']} starts without usage: lower bounds",
    }


def _session_trace(analysis: Analysis, session: int) -> list[str]:
    """The call logs of a session's starts: the agent's trace, shown beside a metric."""
    from fleetlab.agents.run import call_log

    for record in analysis.session_records():
        if record.number == session:
            return [str(call_log(Path(a["run_dir"]))) for a in record.attempts if a.get("run_dir")]
    return []


def identification(
    analysis: Analysis, truth: ConditionTruth, engagement: Mapping[str, Any], model: str | None
) -> dict[str, Any]:
    """The first accepted diagnosis whose causes target the affected controller (or the
    delay keys) at the condition's site, from the agent's own submissions: a proxy for
    identification, shown with the diagnosis verbatim and the session's call log; what
    was spent until then. Diagnoses that target a lookalike site the same way are listed
    on their own, whether or not the condition was ever targeted."""
    rounds = _session_rounds(engagement)
    diagnoses = [s for s in analysis.submissions() if s.get("kind") == "diagnosis"]
    lookalike = []
    first: dict[str, Any] | None = None
    for item in diagnoses:
        diagnosis = Diagnosis.model_validate(item["answer"])
        session = int(item["session"])
        for site, robots in truth.lookalikes:
            if causes := truth.targeting(diagnosis, site, robots):
                lookalike.append(
                    {
                        "session": session,
                        "round": rounds.get(session),
                        "site": site,
                        "causes": [c.model_dump(mode="json") for c in causes],
                    }
                )
        if first is None and (causes := truth.targeting(diagnosis, truth.site, truth.robots)):
            first = {
                "session": session,
                "round": rounds.get(session),
                "receipt_id": item.get("receipt_id"),
                "attempts_before": _attempts_before(analysis, session),
                **_through(analysis, session, model),
                "matching_causes": [c.model_dump(mode="json") for c in causes],
                "diagnosis": item["answer"],  # verbatim, as the agent submitted it
                "trace": _session_trace(analysis, session),
            }
    return {
        "proxy": "the first accepted diagnosis whose cause targets the controller or a delay "
        "key at the condition's site (or half its robots); it shows where the diagnosis "
        "points, not that its reasoning was right: read the diagnosis and the trace",
        "targeted": first is not None,
        "first": first,
        "diagnoses": len(diagnoses),
        "lookalike_targeted": lookalike,
    }


def _panel(eval_dir: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = eval_dir / "attempts.parquet"
    return pq.read_table(path).to_pylist() if path.is_file() else []


def _site_effect(
    rows: Sequence[Mapping[str, Any]], state: str, robots: Sequence[str]
) -> dict[str, Any] | None:
    """The condition robots' panel success under `state` against `original`, paired."""
    from fleetlab.loop.evaluate import mcnemar_exact, wilson

    def outcomes(name: str) -> dict[str, bool]:
        return {
            str(r["episode_id"]): bool(r["success"])
            for r in rows
            if r["state"] == name and r["robot_id"] in robots
        }

    first, second = outcomes("original"), outcomes(state)
    shared = sorted(set(first) & set(second))
    if not shared:
        return None
    n = len(shared)
    before, after = sum(first[e] for e in shared), sum(second[e] for e in shared)
    rescued = sum(1 for e in shared if not first[e] and second[e])
    broken = sum(1 for e in shared if first[e] and not second[e])
    return {
        "attempts": n,
        "original": {"successes": before, "rate": before / n, "interval": wilson(before, n)},
        state: {"successes": after, "rate": after / n, "interval": wilson(after, n)},
        "points": 100.0 * (after - before) / n,
        "rescued": rescued,
        "broken": broken,
        "p_mcnemar": mcnemar_exact(rescued, broken),
    }


def adoption(
    truth: ConditionTruth, engagement: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Every round whose decision adopted, or replaced into, the adaptation at the
    condition's site, with the paired panel effect of that round's whole state at the
    condition's robots; the first effective one and the experiment attempts through it.

    "Effective" means the round's whole state improved the condition's robots on the
    reserved panel (at least `effective_points` and McNemar p below `effective_p`); it does
    not attribute the gain to the setting, which only the agent's own experiments can.
    Settings of the adaptation key at a lookalike site, fleet-wide or elsewhere are
    recorded on their own; they count as measured harm only if the paired outcomes at
    the lookalike's robots show a loss."""
    spent = 0
    attempts: list[dict[str, Any]] = []
    elsewhere: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for entry in engagement.get("rounds", []):
        spent += int((entry.get("costs") or {}).get("arm_attempts") or 0)
        decision = entry.get("decision") or {}
        actions = [Action.model_validate(c["action"]) for c in decision.get("adopt", [])]
        actions += [Action.model_validate(c["action"]) for c in decision.get("replace", [])]
        state = f"round-{entry['round']}"
        for action in actions:
            if not truth.sets_adaptation(action) or truth.where(action.target.scope) == "condition":
                continue
            place = truth.where(action.target.scope)
            harm: list[dict[str, Any]] = []
            for site, robots in truth.lookalikes:
                if place == "fleet" or action.target.scope in (site, *robots):
                    effect = _site_effect(rows, state, robots)
                    harm.append(
                        {
                            "site": site,
                            "effect": effect,
                            "measured_harm": effect is not None
                            and effect["points"] < 0
                            and effect["p_mcnemar"] < truth.effective_p,
                        }
                    )
            elsewhere.append(
                {
                    "round": entry["round"],
                    "where": place,
                    "action": action.model_dump(mode="json"),
                    "at_lookalikes": harm,
                }
            )
        matching = [a for a in actions if truth.adapts(a)]
        if not matching:
            continue
        effect = _site_effect(rows, state, truth.robots)
        effective = (
            effect is not None
            and effect["points"] >= truth.effective_points
            and effect["p_mcnemar"] < truth.effective_p
        )
        record = {
            "round": entry["round"],
            "actions": [a.model_dump(mode="json") for a in matching],
            "attempts_through_round": spent,
            "effect_at_condition": effect,
            "effective": effective,
        }
        attempts.append(record)
        if effective and first is None:
            first = record
    return {
        "effective_means": "the round's whole state improved the condition's robots on the "
        "reserved panel; not attributable to the setting alone",
        "adopted": bool(attempts),
        "attempts": attempts,
        "first_effective": first,
        # The headline: experiment attempts through the first effective round (None when no
        # adoption was effective; then every adoption is listed above).
        "attempts_until_effective": first["attempts_through_round"] if first else None,
        "attempts_all_rounds": spent,
        "adaptation_elsewhere": elsewhere,
    }


def final_outcome(engagement: Mapping[str, Any], truth: ConditionTruth) -> dict[str, Any]:
    """The reserved panel under the engagement's last state, fleet-wide and at the condition."""
    evaluation = engagement.get("evaluation") or {}
    summary = evaluation.get("summary") or {}
    states = {s["name"]: s for s in summary.get("states", [])}
    rounds = sorted(
        (int(n.split("-")[1]) for n in states if re.fullmatch(r"round-\d+", n)), reverse=True
    )
    final = f"round-{rounds[0]}" if rounds else "original"
    rows = _panel(Path(evaluation["path"])) if evaluation.get("path") else []
    return {
        "final_state": final,
        "final": states.get(final),
        "original": states.get("original"),
        "at_condition": _site_effect(rows, final, truth.robots) if final != "original" else None,
    }


@dataclass
class ReuseScorer:
    """`fleetlab.loop.reuse.compare`'s scorer for engagement arms: the registered metrics."""

    truth: ConditionTruth
    model: str | None = None

    def __call__(self, analysis: Analysis, extra: Mapping[str, Any]) -> dict[str, Any]:
        engagement = json.loads(Path(extra["engagement"]).read_text())
        evaluation = engagement.get("evaluation") or {}
        rows = _panel(Path(evaluation["path"])) if evaluation.get("path") else []
        outcome = final_outcome(engagement, self.truth)
        final = outcome.get("final") or {}
        return {
            "identification": identification(analysis, self.truth, engagement, self.model),
            "adoption": adoption(self.truth, engagement, rows),
            "procedural_use": procedural_use(analysis),
            "outcome": outcome,
            "engagement_outcome": engagement.get("outcome"),
            "ended_because": engagement.get("ended_because"),
            "score": final.get("rate"),
        }


def procedural_use(analysis: Analysis) -> dict[str, Any]:
    """What the arm did with the mounted library, from its call logs: the learned skills
    it read, the learned SQL blocks it ran (they carry `-- helper: evidence-N`), and the
    values it set; reuse of a procedure is shown from these traces, not inferred."""
    from fleetlab.agents.run import call_log
    from fleetlab.toolserver.files import read_lines

    learned = (
        {p.name for p in analysis.learned.iterdir() if p.is_dir()}
        if analysis.learned is not None
        else set()
    )
    reads: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for record in analysis.session_records():
        for start in record.attempts:
            if not start.get("run_dir"):
                continue
            for call in read_lines(call_log(Path(start["run_dir"]))):
                arguments = call.get("arguments") or {}
                if call.get("tool") == "read_skill" and arguments.get("name") in learned:
                    reads.append({"session": record.number, "skill": arguments["name"]})
                sql = str(arguments.get("sql") or "")
                if call.get("tool") == "query" and (
                    m := re.search(r"-- helper: (evidence-\d+)", sql)
                ):
                    queries.append({"session": record.number, "block": m[1], "sql": sql})
    return {"mounted": sorted(learned), "skill_reads": reads, "learned_queries_run": queries}


# ---- the runner ------------------------------------------------------------------------------


def engagement_problems(record: Mapping[str, Any], arm: str) -> list[str]:
    """Why an arm's engagement cannot count: it did not end with a decision in every round
    it ran (aborted), or its states were not evaluated on the reserved panel."""
    problems = []
    if record.get("outcome") not in ("completed", "stopped"):
        problems.append(
            f"the {arm} engagement did not complete: {record.get('outcome')} "
            f"({record.get('ended_because')})"
        )
    evaluation = record.get("evaluation") or {}
    if not evaluation.get("summary"):
        problems.append(
            f"the {arm} engagement has no evaluation: {evaluation.get('skipped', 'no summary')}"
        )
    return problems


@dataclass
class EngagementRunner:
    """Runs an arm of the reuse comparison as an engagement of the round driver.

    Every arm gets the same world, layout, attempt budget, sessions per round, panel
    and engine settings (`inputs`, recorded and compared by the harness), learning on,
    and the arm's own fresh analysis from the harness (the library mounted in one).
    An interrupted engagement arm is not resumed: the driver has no resume, so a
    rerun needs a new comparison.
    """

    world_root: Any  # a stored world's root, or a loaded LoopWorld (the toy cell)
    budget: Any  # fleetlab.loop.driver.AttemptBudget
    max_sessions: int
    out: Path  # each arm's engagement goes to out/<arm>
    workers: int = 4
    queue_log: Path | None = None
    heavy: bool = True
    setup: Any = None
    layout: Any = None
    panel_per_robot_day: int | None = None
    panel_video_indices: tuple[int, ...] = ()  # panel attempts that record video (M3: (0,))
    scripted: Callable[[str], Any] | None = None  # a scripted agent per arm (CPU tests)
    records: dict[str, Any] = field(default_factory=dict)

    def inputs(self) -> dict[str, Any]:
        layout = self.layout.model_dump(mode="json") if self.layout is not None else "for_world"
        return {
            "kind": "engagement",
            "budget": asdict(self.budget),
            "max_sessions_per_round": self.max_sessions,
            "panel_per_robot_day": self.panel_per_robot_day,
            "panel_video_indices": list(self.panel_video_indices),
            "layout": layout,
            "workers": self.workers,
            "heavy": self.heavy,
            "learning": True,
            "scripted": self.scripted is not None,
        }

    def __call__(self, setup: ArmSetup) -> dict[str, Any]:
        from fleetlab.loop.driver import run_rounds

        target = self.out / setup.arm
        if target.exists():
            raise ReuseError(
                f"{target} exists: an interrupted engagement arm cannot be resumed; "
                "start a new comparison"
            )
        record = run_rounds(
            self.world_root,
            target,
            agent=setup.agent,
            budget=self.budget,
            max_sessions=self.max_sessions,
            workers=self.workers,
            queue_log=self.queue_log,
            label=f"reuse {setup.arm} the learned library",
            scripted=self.scripted(setup.arm) if self.scripted is not None else None,
            heavy=self.heavy,
            setup=self.setup,
            layout=self.layout,
            env=setup.env,
            panel_per_robot_day=self.panel_per_robot_day,
            panel_video_indices=self.panel_video_indices,
            analysis=setup.analysis,  # its library, if any, is mounted already
        )
        self.records[setup.arm] = record
        exports = {
            str(entry["round"]): tree_digest(Path(entry["export"]["path"]))
            for entry in record.get("rounds", [])
            if entry.get("export")
        }
        problems = engagement_problems(record, setup.arm)
        return {
            "engagement": str(target / "engagement" / "engagement.json"),
            "loop": record.get("loop"),
            "outcome": record.get("outcome"),
            "ended_because": record.get("ended_because"),
            "states": record.get("states"),
            "round_exports": exports,
            "simulated": record.get("simulated"),
            "validity": problems,
        }


# ---- the commands ----------------------------------------------------------------------------


def _world(root: Path, registry: str | None) -> Any:
    from fleetlab.loop.world import LoopWorld

    return LoopWorld.load(root, registry) if registry else LoopWorld.load(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.reuse_run")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="the later fleet with and without the library")
    c.add_argument("--world-root", type=Path, required=True)
    c.add_argument("--library", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--backend", default="codex", choices=("codex", "claude"))
    c.add_argument("--model", default="gpt-6-astra")
    c.add_argument("--effort", default="xhigh")
    c.add_argument("--service-tier", default="fast")
    c.add_argument("--attempts", type=int, default=480, help="experiment attempts per round")
    c.add_argument("--per-arm", type=int)
    c.add_argument("--max-sessions", type=int, default=4, help="per round")
    c.add_argument("--panel-per-robot-day", type=int, default=6)
    c.add_argument(
        "--panel-video-index",
        type=int,
        action="append",
        default=[],
        help="panel attempts with this index within their robot and day record video",
    )
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--queue-log", type=Path)
    c.add_argument("--order", default="without,with")
    s = sub.add_parser("score", help="the registered metrics of a finished comparison")
    s.add_argument("--comparison", type=Path, required=True)
    s.add_argument("--world-root", type=Path, required=True)
    for p in (c, s):
        p.add_argument("--registry", help="the world's fault registry, if not the default")
    args = parser.parse_args(argv)
    if args.command == "compare":
        from fleetlab.agents.toolenv import build_tool_env
        from fleetlab.integration.chain import SYSTEM
        from fleetlab.loop.driver import AttemptBudget
        from fleetlab.loop.rounds import ROUND_TOOLS

        root = args.world_root.expanduser().resolve()
        world = _world(root, args.registry)
        out = args.out.expanduser().resolve()
        library = FrozenLibrary.load(args.library.expanduser().resolve())
        runner = EngagementRunner(
            world if args.registry else root,
            AttemptBudget(args.attempts, args.per_arm),
            args.max_sessions,
            out / "engagements",
            workers=args.workers,
            queue_log=args.queue_log,
            panel_per_robot_day=args.panel_per_robot_day,
            panel_video_indices=tuple(args.panel_video_index),
        )
        case = LaterCase(
            world.world_id,
            None,
            SYSTEM,
            "the round driver's round task with the learning paragraph (fleetlab.loop.rounds)",
            label=f"reuse on {world.world_id}",
            world_root=root,
        )
        agent = AgentSpec(
            args.backend, args.model, effort=args.effort, service_tier=args.service_tier
        )
        result = compare(
            case,
            agent,
            build_tool_env(role="analysis"),
            library,
            out,
            budget=Budget(max_sessions=args.max_sessions, tools=ROUND_TOOLS),
            runner=runner,
            order=tuple(args.order.split(",")),
        )
        print(json.dumps({"comparison": str(out), "validity": result.validity}, indent=1))
        # A comparison with a validity problem is never reported as complete.
        return 0 if not result.validity else 1
    # score
    world = _world(args.world_root.expanduser().resolve(), args.registry)
    comparison = args.comparison.expanduser().resolve()
    truth = ConditionTruth.from_world(world)
    scored = score_comparison(comparison, truth)
    print(json.dumps(scored, indent=1, default=str))
    return 0


def score_comparison(comparison: Path, truth: ConditionTruth) -> dict[str, Any]:
    """The registered metrics of both arms of a finished comparison, written beside it."""
    data = json.loads((comparison / "comparison.json").read_text())
    model = data["arms"]["without"]["inputs"]["agent"]["model"]
    scorer = ReuseScorer(truth, model)
    out: dict[str, Any] = {"truth": asdict(truth), "arms": {}}
    for arm, result in data["arms"].items():
        extra = json.loads((comparison / "arms" / f"{arm}.json").read_text())
        out["arms"][arm] = {
            "metrics": scorer(Analysis(Path(result["analysis"])), extra),
            "costs": result["costs"],
        }
    out["between_arms"] = between_arms(comparison)
    out["validity"] = data.get("validity", [])
    (comparison / "metrics.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


def between_arms(comparison: Path) -> dict[str, Any]:
    """The two arms' final states on the same reserved attempts: both engagements mint the
    panel from the same world and layout, so the attempts pair by episode id (checked
    through the panel's digest); rescued and broken count with minus without."""
    from fleetlab.loop.evaluate import mcnemar_exact

    finals = {}
    for arm in ("without", "with"):
        extra = json.loads((comparison / "arms" / f"{arm}.json").read_text())
        engagement = json.loads(Path(extra["engagement"]).read_text())
        evaluation = engagement.get("evaluation") or {}
        summary = evaluation.get("summary") or {}
        state = final_outcome(engagement, ConditionTruth(site="", robots=()))["final_state"]
        rows = _panel(Path(evaluation["path"])) if evaluation.get("path") else []
        finals[arm] = {
            "state": state,
            "panel_sha256": summary.get("panel_sha256"),
            "outcomes": {
                str(r["episode_id"]): bool(r["success"]) for r in rows if r["state"] == state
            },
        }
    a, b = finals["without"], finals["with"]
    if not a["outcomes"] or a["panel_sha256"] != b["panel_sha256"]:
        return {"paired": False, "reason": "the arms' panels differ or are missing"}
    shared = sorted(set(a["outcomes"]) & set(b["outcomes"]))
    rescued = sum(1 for e in shared if not a["outcomes"][e] and b["outcomes"][e])
    broken = sum(1 for e in shared if a["outcomes"][e] and not b["outcomes"][e])
    return {
        "paired": True,
        "states": {"without": a["state"], "with": b["state"]},
        "attempts": len(shared),
        "points_with_minus_without": 100.0 * (rescued - broken) / len(shared) if shared else None,
        "rescued": rescued,
        "broken": broken,
        "p_mcnemar": mcnemar_exact(rescued, broken),
    }


if __name__ == "__main__":
    sys.exit(main())
