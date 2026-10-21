"""A dry run of the analysis skills on the skills' toy fleet, ending in write-diagnosis.

No model is involved: the decisions an agent would make are replaced by
simple rules on the helpers' outputs, so that the test checks the plumbing,
not the judgment. What it checks is what write-diagnosis promises: a
`Diagnosis` and a ranked list (here in the older `RankedFixSet` form, at least
ten actions) that validate against the strict schemas the CLIs receive,
actions from the world's catalogue, the round's adopted changes and their set
forecast from rank-actions' blocks, and every gain, interval and day traceable
to the result of a query that was run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from skills_support import Exports, helpers
from test_contracts_tools import conforms

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.tools import (
    Cause,
    Diagnosis,
    EvidenceRef,
    Interval,
    RankedAction,
    RankedFixSet,
    RetrainIndicator,
    Warning,
    json_schema_strict,
)
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.log import LogReader

triage = helpers("triage")
cards = helpers("episode-cards")
modes = helpers("failure-modes")
scope = helpers("scope")
from_logs = helpers("sensitivity-from-logs")
rank = helpers("rank-actions")

# Class knowledge the skills state in words: which component a signature points to,
# and which verbs to try on it with the share of the gap each would win back.
SIGNATURE_COMPONENT = {
    "lost the object": Component.gripper,
    "closed on nothing": Component.scene_camera,
}
VERBS = {
    Component.gripper: ((Verb.replace_, 1.0), (Verb.service, 0.5), (Verb.clean, 0.0)),
    Component.scene_camera: ((Verb.recalibrate, 1.0), (Verb.clean, 0.0), (Verb.replace_, 1.0)),
}


@dataclass
class Session:
    """The tool calls of a run: every query text and every result row."""

    reader: Any
    calls: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)

    def run(
        self, module: Any, block: str, params: dict[str, Any] | None = None, rows: Any = None
    ) -> tuple[str, list[dict[str, Any]]]:
        sql = module.fill(module.blocks()[block], params, rows)
        result: list[dict[str, Any]] = self.reader.query(sql).to_pylist()
        self.calls.append((sql, result))
        return sql, result

    def numbers(self) -> set[float]:
        def walk(value: Any) -> Iterator[float]:
            if isinstance(value, bool):
                return
            if isinstance(value, int | float):
                yield float(value)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)

        return {n for _, rows in self.calls for n in walk(rows)}


@dataclass(frozen=True)
class Plan:
    """A candidate action and what the diagnosis says about its cause."""

    label: str
    action: Action
    from_day: int
    attributable: float
    cause: Target | None
    direction: Direction | None
    confidence: float
    mode: str
    confounded: bool = False  # low 0, high from the row where it alone is the cause

    @property
    def alone(self) -> str:
        return f"{self.label} (alone)"


def spread(text: str) -> list[tuple[str, int, int]]:
    """`r.. 22/40, r.. 0/40` as (level, k, n), in the block's order."""
    return [(m[1], int(m[2]), int(m[3])) for m in re.finditer(r"(\S+) (\d+)/(\d+)", text)]


def plan_actions(session: Session) -> tuple[list[Plan], dict[str, Any]]:
    _, outcomes = session.run(triage, "outcomes")
    _, signatures = session.run(modes, "signatures")
    _, changes = session.run(scope, "changes")
    _, trend = session.run(scope, "trend")
    worse = [
        r["level"] for r in outcomes if r["dimension"] == "robot" and r["stands_out"] == "worse"
    ]
    plans: list[Plan] = []
    for robot in worse:
        signature, _ = max(
            (
                (s["signature"], k / n)
                for s in signatures
                for lvl, k, n in spread(s["by_robot"])
                if lvl == robot
            ),
            key=lambda pair: pair[1],
        )
        component = SIGNATURE_COMPONENT[signature]
        for i, (verb, share) in enumerate(VERBS[component]):
            plans.append(
                Plan(
                    f"{verb.value} {component.value} at {robot}",
                    Action(verb=verb, target=Target(scope=robot, component=component)),
                    0,
                    share,
                    Target(scope=robot, component=component) if i == 0 else None,
                    Direction.degraded,
                    0.8,
                    signature,
                )
            )
    # The skill's rule: an entry lines up with a loss when its whole interval is below zero.
    harmful = [c for c in changes if c["kind"] == "change" and c["did_high"] < 0]
    for change in harmful:
        key = ConfigKey(change["what"])
        together = 1 + len([x for x in change["same_day_overlapping"].split("; ") if x])
        old = json.loads(change["detail"].split(" -> ")[0])
        target = Target(scope=change["scope"], config_key=key)
        direction = (
            Direction.changed if key is ConfigKey.software_version else Direction.misconfigured
        )
        plans.append(
            Plan(
                f"revert {key.value} at {change['scope']}",
                Action(verb=Verb.revert, target=target),
                change["day"],
                1 / together,
                target,
                direction,
                0.5 if together > 1 else 0.7,
                "collision",
                together > 1,
            )
        )
        plans.append(
            Plan(
                f"set {key.value} at {change['scope']}",
                Action(verb=Verb.set, target=target, value=old),
                change["day"],
                1 / together,
                None,
                None,
                0.0,
                "collision",
                together > 1,
            )
        )
    covered = {r for p in plans for r in _robots_in(session.reader, p.action.target.scope)}
    drifting = [
        r
        for r in trend
        if r["dimension"] == "robot"
        and r["within_site_slope"] < 0
        and r["within_site_r2"] >= 0.4
        and r["level"] not in covered
    ]
    for row in drifting:
        robot = row["level"]
        target = Target(scope=robot, component=Component.scene_camera)
        for i, (verb, share) in enumerate(VERBS[Component.scene_camera]):
            plans.append(
                Plan(
                    f"{verb.value} scene_camera at {robot}",
                    Action(verb=verb, target=target),
                    0,
                    share,
                    target if i == 0 else None,
                    Direction.drifting,
                    0.6,
                    "closed on nothing",
                )
            )
    return plans, {"trend": trend, "signatures": signatures, "drifting": drifting}


def test_a_dry_run_gives_answers_that_validate(skill_exports: Exports) -> None:
    with LogReader(skill_exports.fleet_export, materialize=True, lock=True) as reader:
        session = Session(reader)
        plans, found = plan_actions(session)
        assert len(plans) >= 10, "the rules give enough candidates on this fleet"
        candidates = [
            from_logs.Candidate(p.label, p.action.target.scope, p.from_day, p.attributable)
            for p in plans
        ]
        # A confounded member also runs as the only cause: its high end comes from there.
        candidates += [
            from_logs.Candidate(p.alone, p.action.target.scope, p.from_day, 1.0)
            for p in plans
            if p.confounded
        ]
        gains_sql, gains = session.run(
            from_logs, "gains", {"project_to_day": 7}, [tuple(c) for c in candidates]
        )
        by_label = {g["label"]: g for g in gains}

        def estimate(plan: Plan) -> tuple[float, float, float]:
            """Gain, low and high, each a value the block returned (sensitivity-from-logs, 5)."""
            row = by_label[plan.label]
            change = plan.from_day > 0 and row["gain_points_change"] is not None
            keys = ("gain_points_change", "gain_change_low", "gain_change_high")
            if not change:
                keys = ("gain_points", "gain_low", "gain_high")
            gain, low, high = (row[k] for k in keys)
            if plan.confounded:  # low 0, high as if it alone were the cause
                low, high = 0.0, by_label[plan.alone][keys[2]]
            projected = row["projected_gain_points"]
            if plan.direction is Direction.drifting and projected is not None:
                high = max(high, projected)
            return gain, low, high

        ordered = sorted(plans, key=lambda p: -estimate(p)[0])
        ranked = [
            RankedAction(
                action=p.action,
                expected_gain=estimate(p)[0],
                uncertainty=Interval(low=estimate(p)[1], high=estimate(p)[2]),
                rationale=f"{p.label}: gains block, from day {p.from_day}",
            )
            for p in ordered
        ]
        _, cut = session.run(rank, "cut", None, rank.ranked_rows(ranked))
        assert [r["expected_gain"] for r in cut] == [r.expected_gain for r in ranked]
        fix_set = RankedFixSet(actions=ranked)
        # The forecast of the adopted set, one part per scope, from the gains block's rows.
        _, forecast = session.run(
            rank,
            "set_gain",
            None,
            [
                (
                    p.action.target.scope,
                    "adopt",
                    p.label,
                    by_label[p.label]["share"],
                    by_label[p.label]["rate"],
                    *estimate(p),
                )
                for p in ordered[:3]
            ],
        )

        # The evidence for each cause: the gains query, and an episode of its mode.
        failures = [
            r["episode_id"]
            for r in session.run(triage, "sample", {"failures_per_robot": 4})[1]
            if r["outcome"] == "failure"
        ]
        _, profiles = session.run(cards, "profile", None, [(e,) for e in failures])
        causes = []
        for plan in plans:
            if plan.cause is None or plan.direction is None:
                continue
            example = next(
                (p for p in profiles if p["robot_id"] == plan.cause.scope), None
            ) or next(p for p in profiles if p["robot_id"] in _robots_in(reader, plan.cause.scope))
            causes.append(
                Cause(
                    target=plan.cause,
                    direction=plan.direction,
                    confidence=plan.confidence,
                    evidence=[
                        EvidenceRef(episode_id=None, step=None, table="episodes", query=gains_sql),
                        EvidenceRef(
                            episode_id=example["episode_id"],
                            step=example["collapse_step"] or example["close_step"],
                            table=None,
                            query=None,
                        ),
                    ],
                    failure_mode=plan.mode,
                )
            )
        warnings = [
            Warning(
                scope=r["level"],
                metric="success rate",
                expected_day=r["crossing_day"],
                confidence=0.6,
            )
            for r in found["drifting"]
            if r["crossing_day"] is not None and r["crossing_day"] >= 0
        ]
        spread_mode = next(s for s in found["signatures"] if s["signature"] == "stalled")
        stalled = [p["episode_id"] for p in profiles if "stall" in p["events"]][:5]
        diagnosis = Diagnosis(
            causes=causes,
            mode_summaries=[
                f"{s['signature']}: {s['failures']} failures; {s['by_robot']}"
                for s in found["signatures"]
            ],
            warnings=warnings,
            retrain=[
                RetrainIndicator(
                    failure_mode=spread_mode["signature"],
                    episode_ids=stalled,
                    why_no_action_fixes_it="it occurs on every robot at a steady rate",
                    data_wanted="approaches that stall before the grasp",
                )
            ],
        )
        catalogue = reader.query("SELECT * FROM catalogue").to_pylist()  # a table
        robots = set(reader.query("SELECT DISTINCT robot_id FROM episodes")["robot_id"].to_pylist())
        sites = set(reader.query("SELECT DISTINCT site_id FROM episodes")["site_id"].to_pylist())
        episodes = set(reader.episode_ids())

    # Both answers validate against the strict schemas the CLIs get, and round-trip.
    for answer in (diagnosis, fix_set):
        dumped = json.loads(answer.model_dump_json())
        assert conforms(dumped, json_schema_strict(type(answer)))
        assert type(answer).model_validate(dumped) == answer
    assert len(fix_set.actions) >= 10
    # The cut block marks the round's adopted changes: the first three distinct actions.
    adopted = [r.action for r, row in zip(ranked, cut, strict=True) if row["adopted"]]
    assert adopted == [r.action for r in ranked[:3]]
    # The set forecast is one row of set_gain, not a sum of single gains, and traceable.
    whole = forecast[-1]
    assert whole["part"] == "set" and whole["low"] <= whole["gain"] <= whole["high"]
    assert whole["gain"] <= sum(r.expected_gain for r in ranked[:3]) + 0.01
    assert {whole["low"], whole["gain"], whole["high"]} <= session.numbers()
    named = {(c.target.scope, c.target.thing) for c in diagnosis.causes}
    assert {("r02", Component.gripper), ("r03", Component.scene_camera)} <= named
    assert {c.target.thing for c in diagnosis.causes if c.target.scope == "s2"} == {
        ConfigKey.control_action_delay_steps,
        ConfigKey.software_version,
    }
    assert all(c.confidence == 0.5 for c in diagnosis.causes if c.target.scope == "s2")
    # The harmless calibration change is not named: its interval spans zero.
    assert not any(
        c.target.config_key is ConfigKey.camera_scene_calibration for c in diagnosis.causes
    )

    # Every action is in the world's catalogue table (verb and target; values are the agent's).
    fields = ("verb", "scope", "component", "config_key", "condition_key")
    entries = {tuple(row[f] for f in fields) for row in catalogue}
    for item in fix_set.actions:
        target = item.action.target.model_dump(mode="json")
        key = (item.action.verb.value, *(target[f] for f in fields[1:]))
        assert key in entries, item.action

    # Every number traceable to a query result; every query cited was run.
    numbers = session.numbers()
    for item in fix_set.actions:
        for number in (item.expected_gain, item.uncertainty.low, item.uncertainty.high):
            # zero is the one number the skill sets itself: a confounded member's low end
            assert number in numbers or number == 0.0, (item.rationale, number)
    for warning in diagnosis.warnings:
        assert float(warning.expected_day) in numbers
    run_queries = {sql for sql, _ in session.calls}
    cited = [e for c in diagnosis.causes for e in c.evidence]
    assert all(e.query in run_queries for e in cited if e.query is not None)

    # What the tool server's submit checks: scopes and episodes exist.
    assert {c.target.scope for c in diagnosis.causes} <= robots | sites | {"fleet"}
    assert {w.scope for w in diagnosis.warnings} <= robots | sites
    assert {e.episode_id for e in cited if e.episode_id} <= episodes
    assert set(diagnosis.retrain[0].episode_ids) <= episodes


def _robots_in(reader: Any, scope_id: str) -> Sequence[str]:
    rows = reader.query(
        "SELECT DISTINCT robot_id FROM episodes WHERE ? IN ('fleet', robot_id, site_id)", [scope_id]
    )
    robots: list[str] = rows["robot_id"].to_pylist()
    return robots


def test_a_ranked_list_needs_ten_distinct_complete_actions() -> None:
    # What write-diagnosis tells the agent; the tool server refuses less.
    def action(i: int) -> RankedAction:
        return RankedAction(
            action=Action(
                verb=Verb.recalibrate, target=Target(scope=f"r{i:02d}", component=Component.arm)
            ),
            expected_gain=1.0,
            uncertainty=Interval(low=0.0, high=2.0),
            rationale="",
        )

    assert len(RankedFixSet(actions=[action(i) for i in range(10)]).actions) == 10
    with pytest.raises(ValueError):
        RankedFixSet(actions=[action(i) for i in range(9)])
    with pytest.raises(ValueError, match="once"):
        RankedFixSet(actions=[action(i) for i in range(9)] + [action(0)])
