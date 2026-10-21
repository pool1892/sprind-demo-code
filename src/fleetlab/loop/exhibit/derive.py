"""Tables derived from a run's records, each with the files it comes from and how.

Pure functions of a `RunView`: no clock, no randomness, no model. Every number in a
table is a count, a rate, an interval or a test computed here from recorded rows, or a
value copied from a record (a confidence, a forecast, a cost); `Item.derivation` says
which. The agent's forecasts are labelled as forecasts wherever they appear.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fleetlab.dashboard.data import Rate
from fleetlab.loop.evaluate import mcnemar_exact
from fleetlab.loop.exhibit.records import (
    Arm,
    Experiment,
    Origin,
    RoundView,
    RunView,
    Sources,
)

__all__ = [
    "ArmResult",
    "Item",
    "Table",
    "action_text",
    "arm_results",
    "cost_table",
    "decision_tables",
    "experiment_table",
    "fleet_by_robot",
    "fleet_by_site_day",
    "next_round_check",
    "panel_pairs",
    "panel_robots",
    "panel_states",
    "reuse_table",
    "target_text",
    "truth_table",
    "weak_spots",
    "working_table",
]

ALPHA = 0.05  # the level at which an experiment's arm is called different from control


@dataclass
class Item:
    """One thing on a page that holds numbers, with its sources (provenance)."""

    item: str
    title: str
    files: list[str]
    derivation: str
    origin: Origin = "real"


@dataclass
class Table(Item):
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""


def _table(
    sources: Sources,
    item: str,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    files: Iterable[str],
    derivation: str,
) -> Table:
    keys = sorted(set(files))
    return Table(
        item=item,
        title=title,
        files=keys,
        derivation=derivation,
        origin=sources.origin(keys),
        headers=list(headers),
        rows=[list(r) for r in rows],
        caption=caption,
    )


# ---- text of numbers and actions ----------------------------------------------------------


def days_text(first: int, last: int) -> str:
    return f"day {first}" if first == last else f"days {first} to {last}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not recorded"
    return f"{100 * value:.{digits}f}%"


def rate_text(rate: Rate) -> str:
    if rate.n == 0:
        return "no attempts"
    low, high = rate.interval
    return f"{rate.successes}/{rate.n} = {100 * rate.rate:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


def points(value: float | None) -> str:
    """A forecast in points as the agent recorded it, to two decimals (no rounding of a
    copied value to one decimal, which moved -1.95 to -1.9)."""
    return "not recorded" if value is None else f"{value:+.2f}"


def target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    thing = target.get("component") or target.get("config_key") or target.get("condition_key")
    return (str(target.get("scope")), str(thing or ""))


def target_text(target: Mapping[str, Any]) -> str:
    scope, thing = target_key(target)
    return f"{thing} at {scope}" if thing else str(scope)


def action_text(action: Mapping[str, Any]) -> str:
    """`revert control.action_delay_steps at s2`, `set ... at r05 to 0`."""
    text = f"{action.get('verb')} {target_text(action.get('target') or {})}"
    if action.get("value") is not None:
        text += f" to {action['value']}"
    return text


def arm_text(arm: Arm) -> str:
    return (
        "control (no change)" if not arm.actions else "; ".join(action_text(a) for a in arm.actions)
    )


def _same_action(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (
        a.get("verb") == b.get("verb")
        and target_key(a.get("target") or {}) == target_key(b.get("target") or {})
        and a.get("value") == b.get("value")
    )


def newcombe(a: Rate, b: Rate) -> tuple[float, float, float]:
    """b minus a, with Newcombe's hybrid score (MOVER-Wilson) 95% interval for independent
    proportions; conservative for paired arms whose outcomes go together."""
    if a.n == 0 or b.n == 0:
        return (float("nan"), float("nan"), float("nan"))
    la, ua = a.interval
    lb, ub = b.interval
    d = b.rate - a.rate
    low = d - math.sqrt((b.rate - lb) ** 2 + (ua - a.rate) ** 2)
    high = d + math.sqrt((ub - b.rate) ** 2 + (a.rate - la) ** 2)
    return (d, low, high)


# ---- the fleet before ---------------------------------------------------------------------


def _ok(row: Mapping[str, Any]) -> bool:
    return row.get("outcome") == "success"


def fleet_by_robot(view: RunView, rnd: RoundView) -> Table:
    days = sorted({int(r["day"]) for r in rnd.ops_rows})
    by_robot: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rnd.ops_rows:
        by_robot[(str(row["site_id"]), str(row["robot_id"]))].append(row)
    rows = []
    for (site, robot), mine in sorted(by_robot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cells = [robot, site, rate_text(Rate.of(_ok(r) for r in mine))]
        for day in days:
            on = [r for r in mine if int(r["day"]) == day]
            cells.append(f"{sum(_ok(r) for r in on)}/{len(on)}" if on else "none")
        rows.append(cells)
    total = Rate.of(_ok(r) for r in rnd.ops_rows)
    rows.append(
        ["all", "", rate_text(total)]
        + [
            f"{sum(_ok(r) for r in rnd.ops_rows if int(r['day']) == d)}/"
            f"{sum(1 for r in rnd.ops_rows if int(r['day']) == d)}"
            for d in days
        ]
    )
    first, last = rnd.operation_days
    return _table(
        view.sources,
        f"r{rnd.number}-fleet-robots",
        "Success by robot and by day",
        ["robot", "site", "success, 95% interval", *[f"day {d}" for d in days]],
        rows,
        f"The fleet's regular work on {days_text(first, last)}, before the round's decision: "
        "successes out of attempts per robot, the rate with its 95% Wilson interval, and "
        "successes out of attempts on each day. Experiment attempts are not included.",
        rnd.ops_files,
        "episodes rows with no experiment_id and day within the round's operation days; "
        "success = outcome 'success'; Wilson score interval, z = 1.96",
    )


def fleet_by_site_day(view: RunView, rnd: RoundView) -> Table:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rnd.ops_rows:
        groups[(str(row["site_id"]), int(row["day"]))].append(row)
    rows = [
        [site, str(day), rate_text(Rate.of(_ok(r) for r in mine))]
        for (site, day), mine in sorted(groups.items())
    ]
    return _table(
        view.sources,
        f"r{rnd.number}-fleet-sites",
        "Success by site and day",
        ["site", "day", "success, 95% interval"],
        rows,
        "The same regular work by site and day, with 95% Wilson intervals.",
        rnd.ops_files,
        "episodes rows as in the table by robot, grouped by site_id and day",
    )


def weak_spots(view: RunView, rnd: RoundView) -> tuple[list[str], Item]:
    """Robots, and sites on a day, whose whole 95% interval lies below the fleet's rate
    over the same days: a fixed rule, stated with each line."""
    total = Rate.of(_ok(r) for r in rnd.ops_rows)
    lines = []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    site_days: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rnd.ops_rows:
        groups[str(row["robot_id"])].append(row)
        site_days[(str(row["site_id"]), int(row["day"]))].append(row)
    if total.n:
        for robot, mine in sorted(groups.items()):
            rate = Rate.of(_ok(r) for r in mine)
            if rate.n and rate.interval[1] < total.rate:
                lines.append(f"robot {robot}: {rate_text(rate)}")
        for (site, day), mine in sorted(site_days.items()):
            rate = Rate.of(_ok(r) for r in mine)
            if rate.n and rate.interval[1] < total.rate:
                lines.append(f"site {site} on day {day}: {rate_text(rate)}")
    item = Item(
        f"r{rnd.number}-fleet-weak",
        "Where the fleet is weak",
        list(rnd.ops_files),
        "robots and site-days whose 95% Wilson upper bound lies below the fleet's success "
        f"rate over the same days ({rate_text(total)})",
        view.sources.origin(rnd.ops_files),
    )
    return lines, item


# ---- experiments --------------------------------------------------------------------------


@dataclass
class ArmResult:
    experiment: Experiment
    arm: Arm
    control: Arm
    rate: Rate
    control_rate: Rate
    difference: tuple[float, float, float]
    paired: bool  # paired starts requested and every pair has the same task
    rescued: int | None
    broken: int | None
    p: float | None
    verdict: str  # better, worse, no difference detected, no outcome recorded
    missing: int  # episodes no export held


def _control(experiment: Experiment) -> Arm | None:
    for arm in experiment.arms:
        if not arm.actions:
            return arm
    return experiment.arms[0] if experiment.arms else None


def arm_results(experiment: Experiment) -> list[ArmResult]:
    control = _control(experiment)
    if control is None:
        return []
    out = []
    base = Rate.of(o for o in control.outcomes if o is not None)
    for arm in experiment.arms:
        if arm is control:
            continue
        known = [o for o in arm.outcomes if o is not None]
        rate = Rate.of(known)
        missing = sum(o is None for o in arm.outcomes) + sum(o is None for o in control.outcomes)
        paired = (
            experiment.paired
            and len(arm.rows) == len(control.rows)
            and all(
                a is not None and c is not None and a.get("task_text") == c.get("task_text")
                for a, c in zip(arm.rows, control.rows, strict=True)
            )
        )
        rescued = broken = None
        p = None
        if paired:
            pairs = list(zip(control.outcomes, arm.outcomes, strict=True))
            rescued = sum(1 for c, a in pairs if c is False and a is True)
            broken = sum(1 for c, a in pairs if c is True and a is False)
            p = mcnemar_exact(rescued, broken)
        difference = newcombe(base, rate)
        if not known or not base.n:
            verdict = "no outcome recorded"
        elif paired and p is not None:
            if p < ALPHA:
                verdict = (
                    "better than control"
                    if (rescued or 0) > (broken or 0)
                    else "worse than control"
                )
            else:
                verdict = "no difference detected"
        elif not math.isnan(difference[1]) and difference[1] > 0:
            verdict = "better than control"
        elif not math.isnan(difference[2]) and difference[2] < 0:
            verdict = "worse than control"
        else:
            verdict = "no difference detected"
        out.append(
            ArmResult(
                experiment,
                arm,
                control,
                rate,
                base,
                difference,
                paired,
                rescued,
                broken,
                p,
                verdict,
                missing,
            )
        )
    return out


def _p_text(p: float | None) -> str:
    if p is None:
        return "not paired"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def result_text(result: ArmResult) -> str:
    d, low, high = result.difference
    text = (
        f"{result.rate.successes}/{result.rate.n} against control "
        f"{result.control_rate.successes}/{result.control_rate.n}"
    )
    if result.paired:
        text += (
            f"; difference {100 * d:+.0f} points; {result.rescued} rescued, "
            f"{result.broken} broken, McNemar {_p_text(result.p)}"
        )
    elif not math.isnan(d):
        text += f"; difference {100 * d:+.0f} points [{100 * low:+.0f}, {100 * high:+.0f}]"
    if result.missing:
        text += f"; {result.missing} attempts without a recorded outcome"
    return text


def _followed(rnd: RoundView, action: Mapping[str, Any]) -> str:
    decision = rnd.decision
    if decision is not None:
        for change in decision.get("adopt", []):
            if _same_action(change["action"], action):
                entry = _entry_of(rnd, action)
                return f"adopted{f' ({entry})' if entry else ''}"
        for change in decision.get("replace", []):
            if _same_action(change["action"], action):
                return f"adopted, replacing {change['entry_id']}"
        for rank, candidate in enumerate(decision.get("ranked_candidates", []), 1):
            if _same_action(candidate["action"], action):
                return f"not adopted; named as alternative {rank}"
        return "not adopted"
    if rnd.fix_set is not None:
        for rank, ranked in enumerate(rnd.fix_set.get("actions", []), 1):
            if _same_action(ranked["action"], action):
                return (
                    f"ranked {rank} of {len(rnd.fix_set['actions'])} (no adoption in this format)"
                )
        return "not in the ranked list"
    return "no decision recorded"


def ledger_entries(ledger: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """A ledger's entries: lane state's ledger and the public ledger's first format keep
    them under `entries`, the public ledger's second (`fleetlab.public-ledger/2`) under
    `changes`."""
    if not ledger:
        return []
    found = ledger.get("entries")
    if found is None:
        found = ledger.get("changes")
    return [dict(e) for e in found or [] if isinstance(e, Mapping)]


def _entry_of(rnd: RoundView, action: Mapping[str, Any]) -> str | None:
    for entry in ledger_entries(rnd.ledger_after):
        if int(entry.get("round", 0)) == rnd.number and _same_action(entry["action"], action):
            return str(entry["entry_id"])
    return None


def experiment_table(view: RunView, rnd: RoundView) -> Table:
    rows = []
    files: list[str] = []
    for experiment in rnd.experiments:
        files += experiment.files
        results = arm_results(experiment)
        attempts = sum(len(a.episode_ids) for a in experiment.arms)
        head = [
            experiment.experiment_id,
            str(experiment.session),
            ", ".join(experiment.robots),
            f"{attempts} ({' + '.join(str(len(a.episode_ids)) for a in experiment.arms)})",
        ]
        if experiment.status != "done" or not results:
            rows.append(
                [
                    *head,
                    arm_text(experiment.arms[0]) if experiment.arms else "",
                    f"{experiment.status}: {experiment.message}".strip(": "),
                    "",
                    "",
                ]
            )
            continue
        for result in results:
            rows.append(
                [
                    *head,
                    arm_text(result.arm),
                    result_text(result),
                    result.verdict,
                    "; ".join(_followed(rnd, a) for a in result.arm.actions),
                ]
            )
    return _table(
        view.sources,
        f"r{rnd.number}-experiments",
        "The experiment ledger",
        [
            "experiment",
            "session",
            "robots",
            "attempts (per arm)",
            "arm",
            "result",
            "reading",
            "what followed",
        ],
        rows,
        "Every experiment the agent requested in this round, each arm against the control arm "
        "on the same robots: successes out of attempts, the difference in points of success "
        "and, for paired arms, the attempts the change rescued (control failed, arm "
        "succeeded on the same task and start) and broke (the reverse) with McNemar's exact "
        "p; unpaired arms get a 95% interval for the difference instead. 'Better' or "
        f"'worse' needs p < {ALPHA} (paired) or an interval that excludes zero; otherwise no "
        "difference was detected at this size. 'What followed' says what the round's "
        "decision did with the arm's change; it does not say the experiment caused it.",
        files,
        "arms and episode ids from results/experiments.jsonl, outcomes from the episodes rows "
        "of the exports that hold them; pairs by position within paired arms, checked for "
        "equal task text; Newcombe hybrid score interval for the difference; McNemar exact "
        "two-sided p on discordant pairs",
    )


# ---- the working table --------------------------------------------------------------------


@dataclass
class _Suspicion:
    key: tuple[str, str]
    text: str
    first: tuple[int, str]
    claims: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    tests: list[ArmResult] = field(default_factory=list)


def working_table(view: RunView, rnd: RoundView) -> Table:
    """Each suspicion of the round: what the agent claimed in its diagnoses (in order), the
    tests that acted on it, and its confidence before and after the tests."""
    found: dict[tuple[str, str], _Suspicion] = {}
    for session, diagnosis in rnd.diagnoses:
        for cause in diagnosis.get("causes", []):
            key = target_key(cause["target"])
            s = found.setdefault(
                key, _Suspicion(key, target_text(cause["target"]), (session, "diagnosis"))
            )
            s.claims.append((session, cause))
    for experiment in rnd.experiments:
        for result in arm_results(experiment):
            for action in result.arm.actions:
                key = target_key(action["target"])
                s = found.setdefault(
                    key,
                    _Suspicion(
                        key,
                        target_text(action["target"]),
                        (experiment.session, f"experiment {experiment.experiment_id}"),
                    ),
                )
                s.tests.append(result)
    for s in found.values():
        firsts = [(n, "diagnosis") for n, _ in s.claims] + [
            (t.experiment.session, f"experiment {t.experiment.experiment_id}") for t in s.tests
        ]
        s.first = min(firsts)
    last_diagnosis_session = max((s for s, _ in rnd.diagnoses), default=None)
    final_keys = {
        target_key(c["target"])
        for s, d in rnd.diagnoses
        if s == last_diagnosis_session
        for c in d.get("causes", [])
    }
    rows = []
    for s in sorted(found.values(), key=lambda x: (x.first[0], x.text)):
        latest = s.claims[-1][1] if s.claims else None
        test_session = min((t.experiment.session for t in s.tests), default=None)
        before = [c for n, c in s.claims if test_session is not None and n <= test_session]
        after = [c for n, c in s.claims if test_session is not None and n > test_session]
        if not s.tests:
            conf_before, conf_after = "no test", "no test"
        else:
            conf_before = f"{before[-1]['confidence']:.2f}" if before else "not stated"
            conf_after = f"{after[-1]['confidence']:.2f}" if after else "not stated"
        if s.key in final_keys:
            status = "claimed in the final diagnosis"
        elif s.claims:
            status = "withdrawn: not in the final diagnosis"
        else:
            status = "tested, never claimed"
        cited = ""
        if latest is not None:
            evidence = latest.get("evidence", [])
            queries = sum(1 for e in evidence if e.get("query"))
            episodes = [str(e["episode_id"]) for e in evidence if e.get("episode_id")]
            cited = f"{queries} {'query' if queries == 1 else 'queries'}" + (
                f", episodes {', '.join(episodes)}" if episodes else ""
            )
        tests = "; ".join(
            f"{t.experiment.experiment_id}: {arm_text(t.arm)}, {t.verdict} "
            f"({t.rate.successes}/{t.rate.n} vs {t.control_rate.successes}/{t.control_rate.n})"
            for t in s.tests
        )
        what = s.text
        if latest is not None:
            what += f", {latest.get('direction')}: {latest.get('failure_mode')}"
        rows.append(
            [
                what,
                f"session {s.first[0]} ({s.first[1]})",
                cited or "none cited",
                tests or "none",
                conf_before,
                conf_after,
                status,
            ]
        )
    files = list(rnd.decision_files) + [f for e in rnd.experiments for f in e.files]
    return _table(
        view.sources,
        f"r{rnd.number}-working",
        "The agent's working table",
        [
            "suspicion (target, direction: failure mode)",
            "first stated",
            "evidence the agent cited",
            "tests and what they showed",
            "confidence before testing",
            "confidence after testing",
            "status at the end of the round",
        ],
        rows,
        "Each suspicion the agent stated in a diagnosis or tested in an experiment this round. "
        "Confidences are the agent's own, copied from its diagnoses: 'before' is the last one "
        "submitted no later than the session that requested the first test, 'after' the last "
        "one submitted afterwards. 'Not stated' means no diagnosis named the target at that "
        "time. A suspicion missing from the final diagnosis is shown as withdrawn.",
        files,
        "causes of every diagnosis submission (outbox/submissions.jsonl, kind diagnosis) by "
        "target scope and component or key, in session order; tests are experiment arms whose "
        "actions name the target, read as in the experiment ledger",
    )


# ---- the decision -------------------------------------------------------------------------


def decision_tables(view: RunView, rnd: RoundView) -> list[Table]:
    out: list[Table] = []
    decision = rnd.decision
    if decision is not None:
        rows = []
        for change in decision.get("adopt", []):
            rows.append(
                [
                    "adopt",
                    _entry_of(rnd, change["action"]) or "not in the ledger",
                    action_text(change["action"]),
                    str(change.get("rationale", "")),
                ]
            )
        for change in decision.get("replace", []):
            rows.append(
                [
                    f"replace {change['entry_id']}",
                    _entry_of(rnd, change["action"]) or "",
                    action_text(change["action"]),
                    str(change.get("rationale", "")),
                ]
            )
        for change in decision.get("undo", []):
            rows.append([f"undo {change['entry_id']}", "", "", str(change.get("rationale", ""))])
        gain = decision.get("predicted_set_gain") or {}
        out.append(
            _table(
                view.sources,
                f"r{rnd.number}-decision",
                "The changes decided",
                ["decision", "ledger entry", "action", "the agent's reason, verbatim"],
                rows or [["no change", "", "", ""]],
                "What the agent adopted, replaced or undid at the end of the round, with the "
                "ledger entry each change became. The agent's forecast for the whole set: "
                f"{points(gain.get('mid'))} points of fleet success (range "
                f"{points(gain.get('low'))} to {points(gain.get('high'))}) over the fleet as "
                "it then stood. It is a forecast, not a measurement."
                + (" The agent also decided to stop." if decision.get("stop") else ""),
                rnd.decision_files,
                "the round's RoundDecision as accepted (engagement record, round entry "
                "'decision'); ledger entries from the round's public ledger",
            )
        )
        ranked = decision.get("ranked_candidates", [])
        if ranked:
            out.append(
                _table(
                    view.sources,
                    f"r{rnd.number}-alternatives",
                    "Alternatives the agent named",
                    [
                        "rank",
                        "action",
                        "forecast gain alone (points)",
                        "forecast range",
                        "the agent's reason, verbatim",
                    ],
                    [
                        [
                            str(i),
                            action_text(c["action"]),
                            f"{c['expected_gain']:+.1f}",
                            f"{c['uncertainty']['low']:+.1f} to {c['uncertainty']['high']:+.1f}",
                            str(c.get("rationale", "")),
                        ]
                        for i, c in enumerate(ranked, 1)
                    ],
                    "Further changes the agent considered and did not adopt, best first, with "
                    "its forecast of each one's gain on its own. Forecasts, not measurements; "
                    "single gains do not add up.",
                    rnd.decision_files,
                    "ranked_candidates of the round's RoundDecision",
                )
            )
    elif rnd.fix_set is not None:
        labels = {
            int(x["rank"]): str(x["label"])
            for x in view.annotations.get("labels", [])
            if x.get("section") == "ranked" and x.get("round", 1) == rnd.number
        }
        rows = []
        for i, ranked in enumerate(rnd.fix_set.get("actions", []), 1):
            rows.append(
                [
                    str(i),
                    action_text(ranked["action"]),
                    f"{ranked['expected_gain']:+.2f}",
                    f"{ranked['uncertainty']['low']:+.2f} to {ranked['uncertainty']['high']:+.2f}",
                    labels.get(i, ""),
                    str(ranked.get("rationale", "")),
                ]
            )
        out.append(
            _table(
                view.sources,
                f"r{rnd.number}-ranked",
                "The ranked list the agent submitted",
                [
                    "rank",
                    "action",
                    "forecast gain alone (points)",
                    "forecast range",
                    "reviewer's label",
                    "the agent's reason, verbatim",
                ],
                rows,
                "This investigation predates round decisions: nothing was adopted, and the "
                "format required at least ten ranked actions. Each gain is the agent's "
                "forecast of that action alone in points of whole-fleet success, not a "
                "measurement; single gains do not add up."
                + (" Labels are a reviewer's, not the agent's." if labels else ""),
                rnd.decision_files + view.annotation_files,
                "the last fix_set submission (outbox/submissions.jsonl); labels from the "
                "annotations file",
            )
        )
    return out


def diagnosis_summaries(rnd: RoundView) -> list[str]:
    if not rnd.diagnoses:
        return []
    return [str(s) for s in rnd.diagnoses[-1][1].get("mode_summaries", [])]


def next_round_check(view: RunView, rnd: RoundView) -> Table | None:
    """The next round's regular work against this round's, on two labelled baselines: all
    of this round's regular work (pooled), and its last day of regular work before the
    decision. The agent's forecast (RoundDecision: `predicted_set_gain`, a gain over the
    fleet's current state) is shown beside the last day for context only: the agent may
    have used other reference periods for parts of it, which a run's annotations can
    explain. Different days and attempts: shown, not the evidence."""
    later = [r for r in view.rounds if r.number == rnd.number + 1]
    if not later or rnd.decision is None or not rnd.ops_rows:
        return None
    nxt = later[0]
    first, last = rnd.operation_days
    after = Rate.of(_ok(r) for r in nxt.ops_rows)
    gain = rnd.decision.get("predicted_set_gain") or {}
    forecast = (
        f"{points(gain.get('mid'))} ({points(gain.get('low'))} to {points(gain.get('high'))})"
    )

    def change(before: Rate, after_: Rate) -> str:
        d, low, high = newcombe(before, after_)
        return "" if math.isnan(d) else f"{100 * d:+.2f} [{100 * low:+.1f}, {100 * high:+.1f}]"

    pooled = Rate.of(_ok(r) for r in rnd.ops_rows)
    last_day = Rate.of(_ok(r) for r in rnd.ops_rows if int(r["day"]) == last)
    pooled_label = f"round {rnd.number}'s regular work, {days_text(first, last)} pooled"
    last_label = f"last day of regular work before the decision (round {rnd.number}, day {last})"
    rows = [
        [
            "all robots",
            pooled_label,
            rate_text(pooled),
            rate_text(after),
            change(pooled, after),
            "shown beside the last day, for context",
        ],
        [
            "all robots",
            last_label,
            rate_text(last_day),
            rate_text(after),
            change(last_day, after),
            forecast,
        ],
    ]
    robots = sorted({str(r["robot_id"]) for r in rnd.ops_rows + nxt.ops_rows})
    for robot in robots:
        b = Rate.of(_ok(r) for r in rnd.ops_rows if r["robot_id"] == robot)
        a = Rate.of(_ok(r) for r in nxt.ops_rows if r["robot_id"] == robot)
        rows.append([robot, pooled_label, rate_text(b), rate_text(a), change(b, a), ""])
    return _table(
        view.sources,
        f"r{rnd.number}-next",
        "The next round's regular work",
        [
            "robots",
            "baseline",
            "baseline success",
            f"round {nxt.number} work",
            "change in points, 95% interval",
            "agent's forecast gain over the current fleet state",
        ],
        rows,
        f"Regular work in round {nxt.number}, with this round's changes in force, against "
        f"two baselines from round {rnd.number}: all of its regular work pooled, and its "
        "last day of regular work before the decision. The agent's forecast is a gain over "
        "the fleet's state when it decided; it is shown beside that last day for context, "
        "and the agent may have used other reference periods for parts of it. The days, "
        "tasks and attempts differ from the baselines': this is what the agent saw next, not "
        "the reserved evidence; the reserved attempts on the summary page are.",
        rnd.ops_files + nxt.ops_files + rnd.decision_files,
        "episodes rows of each round's export as in the fleet table, pooled over the "
        "round's operation days or from its last operation day; forecast from the round's "
        "RoundDecision; Newcombe interval for the difference of independent proportions",
    )


# ---- the reserved panel -------------------------------------------------------------------

STATE_ORDER = ("original", "round-1", "round-2", "one-look")
STATE_LABEL = {
    "original": "as at the start (original)",
    "round-1": "after round 1",
    "round-2": "after round 2",
    "one-look": "plan fixed after one look",
}


def state_label(name: str, labels: Mapping[str, str] | None = None) -> str:
    """A state's name for a reader; a reference method's from the annotations'
    `state_labels` when given (what it is and what it was told), else its name."""
    if name in STATE_LABEL:
        return STATE_LABEL[name]
    if name.startswith("round-"):
        return f"after round {name.split('-', 1)[1]}"
    return f"reference: {(labels or {}).get(name, name)}"


def _labels(view: RunView) -> dict[str, str]:
    return {str(k): str(v) for k, v in (view.annotations.get("state_labels") or {}).items()}


def ordered_states(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    states = list(summary.get("states", []))
    rank = {n: i for i, n in enumerate(STATE_ORDER)}
    return sorted(states, key=lambda s: (rank.get(str(s["name"]), 99), str(s["name"])))


def panel_states(view: RunView) -> Table | None:
    ev = view.evaluation
    if ev is None:
        return None
    rows = []
    for state in ordered_states(ev.summary):
        low, high = state["interval"]
        rows.append(
            [
                state_label(str(state["name"]), _labels(view)),
                str(state["changes"]),
                f"{state['successes']}/{state['attempts']}",
                f"{100 * state['rate']:.1f}% [{100 * low:.1f}, {100 * high:.1f}]",
            ]
        )
    days = ev.summary.get("days", [])
    return _table(
        view.sources,
        "panel-states",
        "The fleet on the reserved attempts",
        ["state", "changes in force", "successes", "success, 95% interval"],
        rows,
        f"The same {ev.summary.get('attempts')} reserved attempts "
        f"({'day' if len(days) == 1 else 'days'} {', '.join(str(d) for d in days)}), never "
        "seen by the agent, simulated under each "
        "state of the fleet. Success with its 95% Wilson interval.",
        ev.files,
        "evaluation summary.json, states: successes, attempts, rate and Wilson interval as "
        "recorded by the evaluation",
    )


def one_look_counts(view: RunView) -> tuple[dict[str, str], list[str], str] | None:
    """The plan fixed after one look as four separate counts, no reason inferred: the
    requested cutoff (the final state's changes in force), the actions listed after round
    1, how many of them were adoptable, and how many the plan applied. The engagement's
    `one_look` record gives them where it exists; otherwise each comes from the frozen
    list, the plan's ledger and the evaluation summary, or is "not recorded"."""
    ev = view.evaluation
    record = view.one_look_counts
    if record is None and (ev is None or ev.one_look_list is None):
        return None
    files: list[str] = []
    if record is not None:
        counts = {
            k: "not recorded" if record.get(k) is None else str(record[k])
            for k in ("requested_cutoff", "listed", "adoptable", "applied")
        }
        files += [f for f in view.record_files if f.endswith("engagement.json")]
        derivation = "the engagement record's one_look entry, as recorded"
    else:
        assert ev is not None and ev.one_look_list is not None
        states = {str(s["name"]): s for s in ev.summary.get("states", [])}
        finals = sorted(
            (n for n in states if n.startswith("round-")), key=lambda n: int(n.split("-")[1])
        )
        applied = (
            sum(1 for e in ledger_entries(ev.one_look_ledger) if e.get("status") == "active")
            if ev.one_look_ledger
            else states["one-look"]["changes"]
            if "one-look" in states
            else None
        )
        counts = {
            "requested_cutoff": str(states[finals[-1]]["changes"]) if finals else "not recorded",
            "listed": str(len(ev.one_look_list)),
            "adoptable": "not recorded",
            "applied": "not recorded" if applied is None else str(applied),
        }
        derivation = (
            "listed: entries of one-look-list.json; applied: active entries of "
            "one-look.ledger.json; requested cutoff: changes of the last round state in "
            "summary.json; adoptable: not in these records"
        )
    if ev is not None:
        files += ev.one_look_files + ev.files
    return counts, sorted(set(files)), derivation


def one_look_text(view: RunView) -> tuple[str, Item] | None:
    found = one_look_counts(view)
    if found is None:
        return None
    counts, files, derivation = found
    text = (
        "The plan fixed after one look takes round 1's adopted changes, then its named "
        "alternatives, as frozen before round 2, up to the final state's number of changes. "
        f"Requested cutoff: {counts['requested_cutoff']}. Actions listed: {counts['listed']}. "
        f"Adoptable among them: {counts['adoptable']}. Applied in the plan: "
        f"{counts['applied']}."
    )
    return text, Item(
        "panel-one-look", "The one-look plan", files, derivation, view.sources.origin(files)
    )


def panel_pairs(view: RunView) -> Table | None:
    ev = view.evaluation
    if ev is None:
        return None
    order = {str(s["name"]): i for i, s in enumerate(ordered_states(ev.summary))}
    pairs = sorted(
        ev.summary.get("pairs", []),
        key=lambda p: (order.get(str(p["first"]), 99), order.get(str(p["second"]), 99)),
    )
    rows = [
        [
            state_label(str(p["first"]), _labels(view)),
            state_label(str(p["second"]), _labels(view)),
            str(p["rescued"]),
            str(p["broken"]),
            f"{p['net']:+d}",
            _p_text(float(p["p_mcnemar"])),
        ]
        for p in pairs
    ]
    return _table(
        view.sources,
        "panel-pairs",
        "Attempts rescued and broken between states",
        ["from", "to", "rescued", "broken", "net", "McNemar exact"],
        rows,
        "For each pair of states on the same reserved attempts: attempts that failed under the "
        "first and succeeded under the second (rescued), the reverse (broken), the net "
        "change, and McNemar's exact two-sided p on those discordant attempts.",
        ev.files,
        "evaluation summary.json, pairs, as recorded by the evaluation",
    )


def panel_sites(view: RunView) -> Table | None:
    """The reserved attempts per site and state: where a state changed nothing shows as
    plainly as where it did."""
    ev = view.evaluation
    if ev is None or not ev.attempts:
        return None
    names = [str(s["name"]) for s in ordered_states(ev.summary)]
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in ev.attempts:
        groups[str(row["site_id"])][str(row["state"])].append(bool(row["success"]))
    rows = [
        [site, *[_k_of_n(by_state.get(n, [])) for n in names]]
        for site, by_state in sorted(groups.items())
    ]
    return _table(
        view.sources,
        "panel-sites",
        "The reserved attempts by site",
        ["site", *[state_label(n, _labels(view)) for n in names]],
        rows,
        "Successes out of the reserved attempts at each site under each state of the fleet, "
        "on the same attempts in every column. Where a site's count is the same in two "
        "states, only its success count is unchanged: equal totals can hide rescues and "
        "breaks that cancel, or a changed configuration.",
        ev.files,
        "evaluation attempts.parquet: one row per state and attempt, grouped by site",
    )


def panel_robots(view: RunView) -> Table | None:
    ev = view.evaluation
    if ev is None or not ev.attempts:
        return None
    names = [str(s["name"]) for s in ordered_states(ev.summary)]
    groups: dict[tuple[str, str], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in ev.attempts:
        groups[(str(row["robot_id"]), str(row["site_id"]))][str(row["state"])].append(
            bool(row["success"])
        )
    rows = []
    for (robot, site), by_state in sorted(groups.items()):
        rows.append([robot, site, *[_k_of_n(by_state.get(n, [])) for n in names]])
    return _table(
        view.sources,
        "panel-robots",
        "The reserved attempts by robot",
        ["robot", "site", *[state_label(n, _labels(view)) for n in names]],
        rows,
        "Successes out of attempts on the reserved panel, per robot and state: where the "
        "change between states happened.",
        ev.files,
        "evaluation attempts.parquet: one row per state and attempt, grouped by robot",
    )


def _k_of_n(values: Sequence[bool]) -> str:
    return f"{sum(values)}/{len(values)}" if values else "none"


# ---- costs --------------------------------------------------------------------------------


def _usage_missing(costs: Mapping[str, Any]) -> tuple[int, int]:
    """(model starts, starts without a usage record). A record that says its usage is
    incomplete without counting the starts counts as one missing, so it stays a bound."""
    starts = int(costs.get("model_starts") or costs.get("starts") or 0)
    missing = costs.get("starts_without_usage")
    count = int(missing) if missing is not None else 0
    if not count and costs.get("usage_complete") is False:
        count = 1
    return starts, count


def _tokens_text(costs: Mapping[str, Any], key: str, part: str | None = None) -> str:
    """A token count, `part` of it in brackets; unknown when no start reported usage, a
    lower bound when some did not."""
    tokens = costs.get("tokens") or {}
    starts, missing = _usage_missing(costs)
    if key not in tokens or (starts and missing >= starts):
        return "not recorded (no model start reported usage)"
    value = int(tokens.get(key) or 0)
    text = f"{value:,}"
    if part is not None and part in tokens:
        text += f" ({int(tokens.get(part) or 0):,})"
    if missing:
        text = f"at least {text}; {missing} of {starts} starts reported no usage"
    return text


def _seconds_text(costs: Mapping[str, Any]) -> str:
    if "model_seconds" not in costs:
        return "not recorded"
    value = float(costs["model_seconds"])
    starts, missing = _usage_missing(costs)
    if starts and missing >= starts:
        return f"not recorded (no model start reported usage; {value:.2f} s of wall time)"
    text = f"{value / 60:.1f} min"
    return f"at least {text}" if missing else text


def usd_text(api: Mapping[str, Any] | None) -> str:
    """An API-equivalent price record as text: a bound stays a bound, a lower bound (also
    one nested in another, as the reuse comparison may wrap the driver's) says "at
    least", and nothing missing becomes zero."""
    if not api:
        return "not recorded"
    status = str(api.get("status", "not recorded"))
    if status == "bounded_api_equivalent" and "low_usd" in api and "high_usd" in api:
        return (
            f"{float(api['low_usd']):.2f} to {float(api['high_usd']):.2f} USD "
            "(API-equivalent bound, not a bill)"
        )
    if status == "lower_bound":
        inner: Mapping[str, Any] | None = api.get("partial")
        while inner is not None and inner.get("status") == "lower_bound":
            inner = inner.get("partial")
        if inner is not None and "low_usd" in inner:
            return (
                f"at least {float(inner['low_usd']):.2f} USD (API-equivalent, a lower bound: "
                f"{api.get('reason', 'some usage not recorded')})"
            )
        return f"not recorded; only a lower bound exists without a figure ({api.get('reason', '')})"
    if status == "unavailable":
        return f"not available ({api.get('reason', 'no price')})"
    return status


def list_price_text(costs: Mapping[str, Any]) -> str | None:
    """The CLI's reported list price, if the record has the field."""
    if "cost_usd_list_price" not in costs:
        return None
    value = costs.get("cost_usd_list_price")
    status = str(costs.get("cost_usd_list_price_status", ""))
    if value is None:
        return "not recorded"
    if status == "lower_bound" or costs.get("usage_complete") is False:
        return f"at least {float(value):.2f} USD"
    return f"{float(value):.2f} USD"


@dataclass(frozen=True)
class Simulation:
    """A round's simulation figures from its cost record, in either recorded shape: the
    driver's queue records with a `simulated` block, or the earlier top-level counts.
    None means the records do not say; `partial_timing` that some queue gave no time."""

    completed: int | None  # experiment attempts simulated to the end
    failed: int | None  # simulations that failed (confirmed failures and partial jobs)
    unconfirmed: int | None  # assignments whose fate the queue could not confirm
    replays: int | None  # the engine's chain-sample replays
    failed_queues: int | None  # queue runs that failed as a whole (counted apart)
    seconds: float | None
    partial_timing: bool


def simulation(costs: Mapping[str, Any]) -> Simulation:
    """Completed attempts are the queues' recorded run jobs (`simulated.run_jobs`, or the
    legacy `simulated_run_jobs`), which also count the finished jobs of a queue that
    failed before it delivered a result; only without queue records do they come from the
    delivered results (`arm_attempts`, or the experiments' attempts)."""
    experiments = costs.get("experiments") or []
    delivered = costs.get("arm_attempts")
    if delivered is None and experiments:
        delivered = sum(int(e.get("attempts") or 0) for e in experiments)
    completed: int | None = None if delivered is None else int(delivered)
    queues = list(costs.get("queues") or [])
    requested = costs.get("requests")
    if (
        not queues
        and isinstance(costs.get("experiments"), list)
        and not experiments
        and (requested is None or int(requested) == 0)
    ):
        # The round requested no experiment: nothing was queued, so nothing was simulated.
        return Simulation(0, 0, 0, 0, 0, None, False)
    if not queues:
        seconds = costs.get("simulation_seconds")
        return Simulation(
            completed=None if completed is None else int(completed),
            failed=_int_or_none(costs.get("simulated_failed_confirmed")),
            unconfirmed=_int_or_none(costs.get("simulated_unconfirmed")),
            replays=_int_or_none(costs.get("simulated_replays")),
            failed_queues=_int_or_none(costs.get("failed_queues")),
            seconds=None if seconds is None else float(seconds),
            partial_timing=False,
        )
    failed: int | None = 0
    unconfirmed: int | None = 0
    replays: int | None = 0
    run_jobs: int | None = 0
    for q in queues:
        block = q.get("simulated")
        if isinstance(block, Mapping):
            run_jobs = _add(run_jobs, int(block.get("run_jobs") or 0))
            failed = _add(failed, int(block.get("failed_confirmed") or 0))
            failed = _add(failed, int(block.get("partial_jobs") or 0))
            unconfirmed = _add(unconfirmed, int(block.get("unconfirmed") or 0))
            replays = _add(replays, int(block.get("replays") or 0))
        elif "simulated_jobs" in q:
            run_jobs = _add(run_jobs, int(q.get("simulated_run_jobs") or 0))
            failed = _add(failed, int(q.get("partial_jobs") or 0))
            planned, delivered = q.get("planned_attempts"), q.get("delivered_attempts")
            unconfirmed = (
                None
                if planned is None or delivered is None or unconfirmed is None
                else unconfirmed
                + max(0, int(planned) - int(delivered) - int(q.get("partial_jobs") or 0))
            )
            replays = _add(
                replays, int(q.get("simulated_jobs") or 0) - int(q.get("simulated_run_jobs") or 0)
            )
        else:
            failed = unconfirmed = replays = run_jobs = None
    if run_jobs is not None:
        completed = run_jobs
    timed = [
        float(t["simulation_seconds"])
        for q in queues
        if isinstance(t := q.get("timing"), Mapping) and t.get("simulation_seconds") is not None
    ]
    return Simulation(
        completed=completed,
        failed=failed,
        unconfirmed=unconfirmed,
        replays=replays,
        failed_queues=sum(1 for q in queues if q.get("state") == "failed"),
        seconds=sum(timed) if timed else None,
        partial_timing=bool(timed) and len(timed) < len(queues),
    )


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _add(total: int | None, value: int) -> int | None:
    return None if total is None else total + value


@dataclass(frozen=True)
class ProbeSimulation:
    """A round's probes (lane probe), apart from the experiments' task attempts: the
    probe outcomes the cost record keeps (`loop.costs.probe_costs`) and the probe runs'
    simulations of the queue records (`simulated_probes`). None means not recorded;
    `recorded` is False for a record from before probes existed."""

    recorded: bool
    requests: int | None
    refused: int | None
    executed: int | None  # probe episodes the engine finished (else the outcomes' count)
    replays: int | None
    failed: int | None  # confirmed failed and partly written probe jobs
    unconfirmed: int | None  # assignments the engine could not confirm either way
    reserved: int | None  # charged by a failed probe, neither confirmed nor refuted
    charged: int | None
    # Requests with no settled outcome (still `queued`, e.g. the queue aborted before its
    # reply), and what they charged; their execution is unknown, never assumed.
    unresolved: int | None = None
    unresolved_charged: int | None = None
    failed_queues: int = 0  # queue runs with probes that failed


def probe_simulation(costs: Mapping[str, Any]) -> ProbeSimulation:
    queues = [q for q in costs.get("queues") or [] if isinstance(q, Mapping)]
    listed = costs.get("probes")
    recorded = (
        "probe_requests" in costs
        or isinstance(listed, list)
        or any("simulated_probes" in q or "probes" in q for q in queues)
    )
    if not recorded:
        return ProbeSimulation(False, None, None, None, None, None, None, None, None)
    settled = ("done", "refused", "failed")
    unresolved: int | None
    unresolved_charged: int | None
    if isinstance(listed, list):
        open_ = [p for p in listed if isinstance(p, Mapping) and p.get("status") not in settled]
        unresolved = len(open_)
        unresolved_charged = (
            sum(int(p["attempts_charged"]) for p in open_)
            if all(p.get("attempts_charged") is not None for p in open_)
            else None
        )
    else:
        unresolved, unresolved_charged = 0, 0
        for q in queues:
            ids = list(q.get("probes") or [])
            done_ids = {o.get("probe_id") for o in q.get("probe_outcomes") or []}
            missing = [i for i in ids if i not in done_ids]
            unresolved += len(missing)
            if missing and not done_ids and q.get("planned_probe_attempts") is not None:
                unresolved_charged = _add(unresolved_charged, int(q["planned_probe_attempts"]))
            elif missing:
                unresolved_charged = None
    requests = _int_or_none(costs.get("probe_requests"))
    if requests is None and isinstance(listed, list):
        requests = len(listed)
    if requests is None and queues:
        requests = sum(len(q.get("probes") or []) for q in queues)
    blocks: list[Mapping[str, Any]] = []
    for q in queues:
        block = q.get("simulated_probes")
        if not isinstance(block, Mapping):
            block = (q.get("simulated_all") or {}).get("probes")
        if isinstance(block, Mapping):
            blocks.append(block)
    outcomes = [o for q in queues for o in q.get("probe_outcomes") or [] if isinstance(o, Mapping)]

    def outcome_sum(cost_key: str, outcome_key: str) -> int | None:
        if cost_key in costs:
            return int(costs[cost_key] or 0)
        if outcomes:
            return sum(int(o.get(outcome_key) or 0) for o in outcomes)
        return None

    if blocks:
        executed: int | None = sum(int(b.get("run_jobs") or 0) for b in blocks)
        replays: int | None = sum(int(b.get("replays") or 0) for b in blocks)
        failed: int | None = sum(
            int(b.get("failed_confirmed") or 0) + int(b.get("partial_jobs") or 0) for b in blocks
        )
        unconfirmed: int | None = sum(int(b.get("unconfirmed") or 0) for b in blocks)
    else:
        executed = outcome_sum("probe_attempts_executed", "attempts_executed")
        replays = failed = unconfirmed = None
    charged = _int_or_none(costs.get("probe_attempts"))
    if charged is None and queues and any("charged_probe_attempts" in q for q in queues):
        charged = sum(int(q.get("charged_probe_attempts") or 0) for q in queues)
    return ProbeSimulation(
        recorded=True,
        requests=requests,
        refused=_int_or_none(costs.get("probe_requests_refused")),
        executed=executed,
        replays=replays,
        failed=failed,
        unconfirmed=unconfirmed,
        reserved=outcome_sum("probe_attempts_reserved", "attempts_reserved"),
        charged=charged,
        unresolved=unresolved,
        unresolved_charged=unresolved_charged,
        failed_queues=sum(
            1 for q in queues if q.get("state") == "failed" and (q.get("probes") or [])
        ),
    )


def probe_attempts(rounds: Sequence[RoundView]) -> str:
    """Every probe attempt over `rounds`, apart from task attempts: executed plus failed
    simulations; "at least" where a failed probe keeps attempts reserved, assignments are
    unconfirmed or failures are not recorded; "not recorded" where a round's record has
    no probe accounting or no executed count (`probe_attempts_of` per round)."""
    total = 0
    lower = False
    for rnd in rounds:
        text = probe_attempts_of(rnd.costs)
        if text == "not recorded":
            return text
        lower = lower or text.startswith("at least")
        total += int(text.removeprefix("at least "))
    return f"at least {total}" if lower else str(total)


def robot_attempts(rounds: Sequence[RoundView]) -> str:
    """Task and probe attempts together, each part by its own rule; where probes are not
    recorded (a record from before probes), the task attempts alone, said so."""
    tasks = experiment_attempts(rounds)
    probes = probe_attempts(rounds)
    if tasks.startswith("not recorded"):
        return "not recorded"
    task_n = int(tasks.removeprefix("at least "))
    if probes.startswith("not recorded"):
        return f"{tasks} task executions; probe executions not recorded"
    probe_n = int(probes.removeprefix("at least "))
    lower = tasks.startswith("at least") or probes.startswith("at least")
    total = f"{'at least ' if lower else ''}{task_n + probe_n}"
    return f"{total} ({tasks} task, {probes} probe)"


def experiment_attempts(rounds: Sequence[RoundView]) -> str:
    """Every experiment attempt simulated, failed ones included, over `rounds`: completed
    plus failed; "at least" where some assignments are unconfirmed or a round does not
    say how many failed; "not recorded" where a round does not say how many completed."""
    total = 0
    lower = False
    for rnd in rounds:
        sim = simulation(rnd.costs)
        if sim.completed is None:
            return "not recorded"
        total += sim.completed + (sim.failed or 0)
        lower = lower or sim.failed is None or sim.unconfirmed is None or sim.unconfirmed > 0
    return f"at least {total}" if lower else str(total)


def model_minutes(rounds: Sequence[RoundView]) -> str:
    """Model wall time over `rounds`, by the detail table's rule (`_seconds_text`)."""
    total = 0.0
    lower = False
    for rnd in rounds:
        c = rnd.costs
        if "model_seconds" not in c:
            return "not recorded"
        starts, missing = _usage_missing(c)
        if starts and missing >= starts:
            return "not recorded (no model start reported usage)"
        lower = lower or missing > 0
        total += float(c["model_seconds"])
    text = f"{total / 60:.1f}"
    return f"at least {text}" if lower else text


def _count(value: int | None) -> str:
    return "not recorded" if value is None else str(value)


def _probe_rows(costs: Mapping[str, Any]) -> dict[str, str]:
    probe = probe_simulation(costs)
    if not probe.recorded:
        return {"probe attempts": "not recorded (the record has no probe accounting)"}
    return {
        "probe requests (refused)": f"{_count(probe.requests)} ({_count(probe.refused)})",
        "probe attempts executed": _count(probe.executed),
        "probe executions (failed ones included)": probe_attempts_of(costs),
        "failed probe simulations": _count(probe.failed),
        "probe attempts reserved (a failed probe's rest, unknown)": _count(probe.reserved),
        "probe simulations not confirmed": _count(probe.unconfirmed),
        "probe chain-check replays": _count(probe.replays),
        "probe attempts charged": _count(probe.charged),
        "probe requests unresolved (outcome unknown)": unresolved_text(probe),
    }


def unresolved_text(probe: ProbeSimulation) -> str:
    """The probe requests with no settled outcome, e.g. "1 request unresolved: 6 charged,
    outcome unknown"; their execution is never assumed."""
    if probe.unresolved is None:
        return "not recorded"
    if probe.unresolved == 0:
        return "0"
    charged = (
        "charged not recorded"
        if probe.unresolved_charged is None
        else f"{probe.unresolved_charged} charged"
    )
    noun = "request" if probe.unresolved == 1 else "requests"
    text = f"{probe.unresolved} {noun} unresolved: {charged}, outcome unknown"
    if probe.failed_queues:
        text += " (the queue run failed before its reply)"
    return text


def probe_attempts_of(costs: Mapping[str, Any]) -> str:
    """`probe_attempts` of one cost record."""
    probe = probe_simulation(costs)
    if not probe.recorded or probe.executed is None:
        return "not recorded"
    total = probe.executed + (probe.failed or 0)
    lower = (
        probe.failed is None
        or probe.unconfirmed is None
        or bool(probe.unconfirmed)
        or probe.reserved is None
        or bool(probe.reserved)
        or probe.unresolved is None
        or bool(probe.unresolved)
        or bool(probe.failed_queues)
    )
    return f"at least {total}" if lower else str(total)


def _cost_column(view: RunView, rnd: RoundView) -> dict[str, str]:
    c = rnd.costs
    sim = simulation(c)
    experiments = c.get("experiments") or []
    stored = rnd.number == 1
    ops = len(rnd.ops_rows)
    seconds = (
        "not recorded"
        if sim.seconds is None
        else ("at least " if sim.partial_timing else "") + f"{sim.seconds / 60:.1f} min"
    )
    return {
        "regular work (attempts)": f"{ops}"
        + (" (the world's stored run)" if stored else " (simulated under the ledger)"),
        "experiment requests (refused)": (
            f"{c.get('requests', c.get('requests_charged', len(experiments)))} "
            f"({c.get('requests_refused', 'not recorded')})"
        ),
        "task attempts completed": _count(sim.completed),
        "task executions (failed ones included)": experiment_attempts([rnd]),
        "failed task simulations": _count(sim.failed),
        "task simulations not confirmed": _count(sim.unconfirmed),
        "task chain-check replays": _count(sim.replays),
        **_probe_rows(c),
        "failed queue runs": _count(sim.failed_queues),
        "simulation time": seconds,
        "sessions (model starts, failed)": (
            f"{c.get('sessions', 'not recorded')} ({c.get('model_starts', 'not recorded')}, "
            f"{c.get('failed_starts', 'not recorded')})"
        ),
        "tool calls": "not recorded" if "tool_calls" not in c else str(c["tool_calls"]),
        "model time": _seconds_text(c),
        "input tokens (of which cached)": _tokens_text(c, "input_tokens", "cached_input_tokens"),
        "output tokens (of which reasoning)": _tokens_text(c, "output_tokens", "reasoning_tokens"),
        "price equivalent": usd_text(c.get("api_equivalent_usd")),
        "list price reported by the CLI": list_price_text(c) or "not recorded",
    }


def cost_table(
    view: RunView, rounds: Sequence[RoundView] | None = None, *, evaluation: bool = True
) -> Table:
    chosen = list(view.rounds if rounds is None else rounds)
    columns = [(f"round {r.number}", _cost_column(view, r)) for r in chosen]
    ev = view.evaluation if evaluation else None
    if ev is not None:
        states = ev.summary.get("states", [])
        total = sum(int(s.get("attempts") or 0) for s in states)
        columns.append(
            (
                "reserved evaluation",
                {
                    "regular work (attempts)": f"{total} ({len(states)} states x "
                    f"{ev.summary.get('attempts')} attempts)",
                    "evaluation wall time": "not recorded"
                    if ev.seconds is None
                    else f"{ev.seconds / 60:.1f} min",
                },
            )
        )
    measures: list[str] = []
    for _, col in columns:
        for key in col:
            if key not in measures:
                measures.append(key)
    rows = [[m, *[col.get(m, "") for _, col in columns]] for m in measures]
    files = [f for r in chosen for f in r.cost_files] + [f for r in chosen for f in r.ops_files]
    if ev is not None:
        files += ev.files
    return _table(
        view.sources,
        "costs" if rounds is None else "costs-" + "-".join(f"r{r.number}" for r in chosen),
        "What the run cost",
        ["", *[name for name, _ in columns]],
        rows,
        "Attempts by purpose, simulations, agent sessions, tool calls, model time and tokens, "
        "per round and for the reserved evaluation. Experiment task attempts and the "
        "agent's probe attempts are counted apart, each with its executed, failed, "
        "unconfirmed and replayed simulations; a failed probe's reserved rest is neither "
        "executed nor refuted. 'Not recorded' means the records hold no value; 'at least' "
        "marks a lower bound (model starts without usage, unconfirmed or reserved work). "
        "Nothing missing is shown as zero. Regular work in round 1 of a stored world is the "
        "world's own run, not simulated again by the loop.",
        files,
        "round cost records (session_costs over every model start, experiment_costs, "
        "probe_costs, queue records with their simulated and simulated_probes blocks); "
        "regular work = episodes rows of the round; evaluation attempts from "
        "summary.json",
    )


# ---- reuse --------------------------------------------------------------------------------

_REUSE_ROWS = (
    ("sessions", "sessions"),
    ("tool_calls", "tool calls"),
    ("experiments_requested", "experiments requested"),
    ("attempts_simulated", "experiment attempts simulated"),
    ("model_seconds", "model time (s)"),
    ("input_tokens", "input tokens"),
    ("output_tokens", "output tokens"),
    ("cost_usd_list_price", "list price (USD)"),
    ("score", "outcome score"),
)


def _number(value: Any) -> str:
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.4g}"


_USAGE_KEYS = frozenset({"model_seconds", "input_tokens", "output_tokens", "cost_usd_list_price"})


def reuse_table(view: RunView) -> Table | None:
    data = view.reuse
    if data is None:
        return None
    diffs = data.get("differences", {})
    arms = data.get("arms", {})

    def complete(arm: str) -> bool:
        costs = (arms.get(arm) or {}).get("costs") or {}
        return costs.get("usage_complete", True) is not False

    def value(arm: str, key: str, number: Any) -> str:
        if number is None:
            return "not recorded"
        text = _number(number)
        return f"at least {text}" if key in _USAGE_KEYS and not complete(arm) else text

    rows = []
    for key, label in _REUSE_ROWS:
        row = diffs.get(key)
        if row is None:
            continue
        if row.get("status") == "unavailable":
            delta = f"no difference: {row.get('reason', 'incomplete')}"
        else:
            delta = (
                "not recorded"
                if row.get("with_minus_without") is None
                else ("+" if row["with_minus_without"] > 0 else "")
                + _number(row["with_minus_without"])
            )
        rows.append(
            [
                label,
                value("without", key, row.get("without")),
                value("with", key, row.get("with")),
                delta,
            ]
        )
    rows.append(
        [
            "price equivalent",
            usd_text(((arms.get("without") or {}).get("costs") or {}).get("api_equivalent_usd")),
            usd_text(((arms.get("with") or {}).get("costs") or {}).get("api_equivalent_usd")),
            "",
        ]
    )
    reads = ((arms.get("with") or {}).get("costs") or {}).get("learned_skill_reads") or {}
    rows.append(
        ["reads of a learned skill", "not applicable", str(sum(int(v) for v in reads.values())), ""]
    )
    return _table(
        view.sources,
        "reuse",
        "The same later fleet with and without the agent's procedure",
        ["", "without the procedure", "with the procedure", "with minus without"],
        rows,
        f"Case '{data.get('case')}': the same model, base skills, export and budget, as two "
        "fresh analyses; the learned procedure is the only difference. 'At least' marks a "
        "lower bound where an arm's model starts did not all report usage; then no "
        "difference is given. One run per arm is one case, not a rate.",
        view.reuse_files,
        "comparison.json of fleetlab.loop.reuse: differences, each arm's usage completeness "
        "and price record, and the with-arm's learned skill reads, as recorded",
    )


_PURPOSES = (
    ("operation", "the fleet's regular work after round 1"),
    ("experiments", "experiments (task attempts)"),
    ("probes", "probes"),
    ("evaluation", "the reserved evaluation, every state"),
)


def simulated_table(view: RunView) -> Table | None:
    """Every simulation the loop ran, by purpose, from the driver's own count of its run
    directories (`driver.json`, `simulated`): new attempts, replays, failed and
    unconfirmed simulations. Round 1's regular work is the world's stored run, never
    simulated again, so it is not in the operation row."""
    totals = view.simulated_totals
    if not totals:
        return None
    rows = []
    sums = dict.fromkeys(("run_jobs", "replays", "failed", "unconfirmed", "executions"), 0)
    for key, label in _PURPOSES:
        part = totals.get(key)
        if not isinstance(part, Mapping):
            if key == "probes":
                continue
            rows.append([label, *["not recorded"] * 5])
            continue
        failed = int(part.get("failed_confirmed") or 0) + int(part.get("partial_jobs") or 0)
        values = {
            "run_jobs": int(part.get("run_jobs") or 0),
            "replays": int(part.get("replays") or 0),
            "failed": failed,
            "unconfirmed": int(part.get("unconfirmed") or 0),
        }
        values["executions"] = values["run_jobs"] + values["replays"] + failed
        for k, v in values.items():
            sums[k] += v
        rows.append([label, *[str(values[k]) for k in sums]])
    rows.append(["all", *[str(v) for v in sums.values()]])
    return _table(
        view.sources,
        "simulated",
        "Every simulation the loop ran",
        [
            "purpose",
            "new attempts",
            "replays",
            "failed or partial",
            "not confirmed",
            "executions (new, replays and failed)",
        ],
        rows,
        "Robot attempts the physics simulator ran for this run, by purpose: new attempts, and "
        "replays (a sample of attempts run a second time to check that the simulation "
        "reproduces them exactly). Failed or partial simulations and assignments the "
        "simulator could not confirm are counted apart; executions add new attempts, replays "
        "and failed simulations. The first round's regular work is the world's stored run, "
        "observed before the loop began and not simulated again, so it is not counted here.",
        view.simulated_files,
        "driver.json 'simulated': run_jobs, replays, failed_confirmed plus partial_jobs, "
        "unconfirmed, per purpose, as the driver counted them from the run directories",
    )


def _sum_price(costs: Sequence[Mapping[str, Any]]) -> str:
    """The API-equivalent bound summed over cost records, when every one is a complete
    bound; otherwise each record's own text is the honest answer, so say so."""
    bounds = [c.get("api_equivalent_usd") or {} for c in costs]
    if bounds and all(b.get("status") == "bounded_api_equivalent" for b in bounds):
        low = sum(float(b["low_usd"]) for b in bounds)
        high = sum(float(b["high_usd"]) for b in bounds)
        return f"{low:.2f} to {high:.2f} USD (API-equivalent bound, not a bill)"
    return "see the cost table (not a complete bound in every round)"


def _final_state(summary: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    states = {str(s["name"]): s for s in summary.get("states", [])}
    rounds = sorted((n for n in states if n.startswith("round-")), key=lambda n: int(n[6:]))
    final = states.get(rounds[-1]) if rounds else None
    pair: dict[str, Any] = next(
        (
            dict(p)
            for p in summary.get("pairs", [])
            if final is not None and p["first"] == "original" and p["second"] == final["name"]
        ),
        {},
    )
    return final, pair


def _result_cells(summary: Mapping[str, Any]) -> list[str]:
    states = {str(s["name"]): s for s in summary.get("states", [])}
    final, pair = _final_state(summary)
    original = states.get("original")
    if final is None or original is None:
        return ["not recorded", "not recorded", "not recorded"]
    return [
        str(final["changes"]),
        f"{original['successes']} to {final['successes']} of {final['attempts']}",
        f"{pair.get('rescued', 'not recorded')} rescued, {pair.get('broken', 'not recorded')} "
        "broken",
    ]


def _sum_count(each: Sequence[Mapping[str, Any]], key: str) -> str:
    """A count summed over cost records; "not recorded" when any record lacks it."""
    if any(c.get(key) is None for c in each):
        return "not recorded"
    return str(sum(int(c[key]) for c in each))


def _attempts_text(sims: Sequence[Simulation]) -> str:
    """Experiment attempts over some simulation figures by `experiment_attempts`' rule:
    completed plus failed, "at least" when failures are not recorded or some assignments
    are unconfirmed or of unknown fate, "not recorded" without a completed count."""
    total = 0
    lower = False
    for sim in sims:
        if sim.completed is None:
            return "not recorded"
        total += sim.completed + (sim.failed or 0)
        lower = lower or sim.failed is None or sim.unconfirmed is None or sim.unconfirmed > 0
    return f"at least {total}" if lower else str(total)


def _cost_cells(costs: Mapping[str, Any], rounds: Sequence[RoundView] | None = None) -> list[str]:
    """Experiment attempts (failed included) and replays, sessions, model starts, model
    time, tool calls and price, by the cost tables' rules: a missing count is "not
    recorded", never zero, and unconfirmed work keeps a total "at least"."""
    each: list[Mapping[str, Any]]
    if rounds is not None:
        sims = [simulation(r.costs) for r in rounds]
        each = [r.costs for r in rounds]
        minutes = model_minutes(rounds)
    else:
        sims = [simulation(costs)]
        each = [costs]
        minutes = _seconds_text(costs)
    replays = (
        "not recorded"
        if any(s.replays is None for s in sims)
        else str(sum(s.replays or 0 for s in sims))
    )
    if rounds is not None and not minutes.startswith("not"):
        minutes = f"{minutes} min"
    sessions = _sum_count(each, "sessions")
    scripted = costs.get("scripted_sessions") if rounds is None else None
    if scripted is not None:
        count = len(scripted) if isinstance(scripted, list) else scripted
        sessions += f" ({count} scripted)"
    if rounds is not None:
        price = _sum_price(each)
    else:
        api = costs.get("api_equivalent_usd") or {}
        price = usd_text(api)
        if api.get("basis"):
            price += f"; {api['basis']}"
    return [
        f"{_attempts_text(sims)} (+{replays} replays)",
        sessions,
        _sum_count(each, "model_starts"),
        minutes,
        _sum_count(each, "tool_calls"),
        price,
    ]


def panel_match(view: RunView, summary: Mapping[str, Any]) -> str | None:
    """None when a comparator's evaluation is on the agent's panel: both fingerprints
    recorded, non-empty and equal. Otherwise the reason it is not comparable."""
    ev = view.evaluation
    ours = (ev.summary.get("panel_sha256") if ev is not None else None) or ""
    theirs = summary.get("panel_sha256") or ""
    if not ours:
        return "not comparable: the agent's evaluation records no panel fingerprint"
    if not theirs:
        return "not comparable: this method's evaluation records no panel fingerprint"
    if ours != theirs:
        return "not comparable: measured on a different panel"
    return None


def comparator_table(view: RunView) -> Table | None:
    """The agent's final state beside the other methods measured on the same reserved
    attempts, each with what it spent; a method whose evaluation is not there yet is a
    marked placeholder, and one measured on another panel is said to be not comparable."""
    if not view.comparators or view.evaluation is None:
        return None
    ev = view.evaluation
    rows = [
        [
            str((view.annotations.get("names") or {}).get("agent") or "the analysis agent"),
            "the fleet's records and the experiment catalogue; told nothing about where to look",
            *_result_cells(ev.summary),
            *_cost_cells({}, view.rounds),
        ]
    ]
    files = list(ev.files) + [f for r in view.rounds for f in r.cost_files]
    for comparator in view.comparators:
        files += comparator.files
        if comparator.kind == "pending" or comparator.evaluation is None:
            rows.append(
                [
                    comparator.label,
                    "",
                    "**[PENDING]** " + (comparator.status or "not yet"),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
            continue
        summary = comparator.evaluation
        reason = panel_match(view, summary)
        result = [reason, "", ""] if reason else _result_cells(summary)
        record = comparator.record or {}
        did = ", ".join(action_text(a) for a in record.get("adopted") or []) or "no change"
        rows.append(
            [comparator.label, f"adopted: {did}", *result, *_cost_cells(comparator.costs or {})]
        )
    return _table(
        view.sources,
        "comparators",
        "Other methods on the same reserved attempts",
        [
            "method",
            "what it was given or did",
            "changes in force",
            "reserved successes",
            "against the start",
            "experiment executions",
            "sessions",
            "model starts",
            "model time",
            "tool calls",
            "price equivalent",
        ],
        rows,
        "The agent's final state beside other methods measured on the same reserved attempts "
        "(the same panel, checked by its fingerprint): successes as at the start and after "
        "the method's changes, the attempts rescued and broken against the start, and what "
        "each spent. Experiment executions are the attempts run plus any execution that failed "
        "and was run again; replays are apart. "
        "A method's costs come from its own receipt. A placeholder marks a method whose "
        "evaluation has not finished.",
        files,
        "the engagement's evaluation summary and round cost records (summed by the cost "
        "tables' rules); each comparator's own record: its evaluation summary (same "
        "panel_sha256) and its costs receipt",
    )


def comparator_breakdown(view: RunView) -> Table | None:
    """Where the agent's final state and each comparator differ on the reserved attempts:
    successes per site and per robot, as at the start and after each method (the agent's
    attempts and each comparator's own, on the same panel)."""
    ev = view.evaluation
    if ev is None or not ev.attempts or not (ev.summary.get("panel_sha256") or ""):
        return None
    usable = [
        c
        for c in view.comparators
        if c.attempts and c.evaluation is not None and panel_match(view, c.evaluation) is None
    ]
    if not usable:
        return None
    agent_final, _ = _final_state(ev.summary)
    if agent_final is None:
        return None

    def tally(rows: Sequence[Mapping[str, Any]], state: str) -> dict[tuple[str, str], list[bool]]:
        out: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for row in rows:
            if str(row["state"]) == state:
                out[(str(row["site_id"]), str(row["robot_id"]))].append(bool(row["success"]))
        return out

    start = tally(ev.attempts, "original")
    agent = tally(ev.attempts, str(agent_final["name"]))
    others = []
    for c in usable:
        final, _ = _final_state(c.evaluation or {})
        others.append(tally(c.attempts, str(final["name"])) if final else {})
    keys = sorted(start)
    sites = sorted({site for site, _ in keys})
    rows = []

    def cells(group: Sequence[tuple[str, str]]) -> list[str]:
        def k(table: Mapping[tuple[str, str], list[bool]]) -> str:
            values = [v for key in group for v in table.get(key, [])]
            return f"{sum(values)}/{len(values)}" if values else "none"

        return [k(start), k(agent), *[k(o) for o in others]]

    for site in sites:
        mine = [key for key in keys if key[0] == site]
        rows.append([f"site {site}", "all", *cells(mine)])
        rows += [[f"site {site}", robot, *cells([(s, robot)])] for s, robot in mine]
    rows.append(["all sites", "all", *cells(keys)])
    files = list(ev.files) + [f for c in usable for f in c.files]
    return _table(
        view.sources,
        "comparator-robots",
        "Where the methods differ, by site and robot",
        [
            "site",
            "robot",
            "as at the start",
            "the agent's final state",
            *[f"{c.label}" for c in usable],
        ],
        rows,
        "Successes out of the reserved attempts per site and per robot: as at the start, "
        "after the agent's changes, and after each other method's. The same attempts in "
        "every column.",
        files,
        "evaluation attempts.parquet of the engagement (original and final state) and of "
        "each comparator (its final state), grouped by site and robot",
    )


def incidents_table(view: RunView) -> Table | None:
    """The engine's incidents in the rounds' queue runs (lost assignments it ran again),
    from each queue run's own record (`run.json`, extra.incidents): what happened to the
    worker, the jobs, their state, whether the run recovered and how the replay check of
    the job came out. No cause is attributed."""
    rows = []
    files: list[str] = []
    for rnd in view.rounds:
        for queue in rnd.costs.get("queues") or []:
            block = queue.get("simulated") or {}
            if not int(block.get("incidents") or 0) or not block.get("run_dir"):
                continue
            run_file = view.sources.path(block["run_dir"]) / "run.json"
            if not run_file.is_file():
                rows.append(
                    [
                        str(rnd.number),
                        str(queue.get("session")),
                        "not recorded",
                        f"{block['incidents']} incident(s) counted; run record missing",
                        "",
                        "",
                        "",
                    ]
                )
                continue
            record = view.sources.json(run_file, "queue run record (incidents)", rnd.origin)
            files.append(str(run_file.resolve()))
            for incident in (record.get("extra") or {}).get("incidents") or []:
                jobs = [str(j) for j in incident.get("jobs") or []]
                status = incident.get("assignment_status") or {}
                checks = incident.get("replay_checks") or {}
                rows.append(
                    [
                        str(rnd.number),
                        str(queue.get("session")),
                        ", ".join(jobs),
                        f"worker stopped ({incident.get('why', 'not recorded')})",
                        ", ".join(f"{j}: {status.get(j, 'not recorded')}" for j in jobs),
                        "yes, run again" if incident.get("recovered") else "no",
                        ", ".join(
                            f"{j}: {(checks.get(j) or {}).get('verdict', 'not checked')}"
                            for j in jobs
                        ),
                    ]
                )
    if not rows:
        return None
    return _table(
        view.sources,
        "incidents",
        "Incidents in the simulations",
        [
            "round",
            "queue of session",
            "job",
            "what happened",
            "state of the job",
            "recovered",
            "replay check",
        ],
        rows,
        "Simulation jobs the engine lost and ran again, from each queue run's own record: "
        "what happened to the worker, the job's state when it was lost, whether the run "
        "recovered, and whether a replay of the job reproduced it exactly. The record "
        "attributes no cause, and neither does this page.",
        files,
        "run.json extra.incidents of each queue run whose simulated block counts incidents",
    )


def reuse_notes(view: RunView) -> list[str]:
    return [str(n) for n in (view.reuse or {}).get("negative_transfer", [])]


# ---- the planted truth --------------------------------------------------------------------


def lasting(kind: str) -> bool:
    """A fault kind no catalogue action removes: its registered truth mapping has no
    fixing verb (a lasting condition, such as a remote link)."""
    from fleetlab.faults import REGISTRY

    try:
        mapping = REGISTRY.get(kind).truth
    except (KeyError, ValueError):
        return False
    return not mapping.fixing_verbs


def _lasting_cells(
    view: RunView, fault: Mapping[str, Any], site_of: Mapping[str, str]
) -> list[str]:
    """Per round, for a lasting fault: that no removing action exists, and the agent's
    setting changes (set or revert of a configuration key) at the fault's robot, its site
    or the whole fleet, as the ledger after that round's decision holds them: adopted in
    the round, adopted earlier and still active, or undone or replaced in the round (an
    entry active in the ledger before and not after). Listed as actions at this site,
    without saying they acted on the fault."""
    scope = str(fault["target"].get("scope"))
    site = site_of.get(scope, scope)
    covering = {scope, site, "fleet"}

    def at_site(ledger: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        return [
            e
            for e in ledger_entries(ledger)
            if e["action"].get("verb") in ("set", "revert")
            and e["action"]["target"].get("config_key")
            and str(e["action"]["target"].get("scope")) in covering
        ]

    def listed(entries: Sequence[Mapping[str, Any]]) -> str:
        return ", ".join(
            f"{action_text(e['action'])} ({e['entry_id']}, round {e['round']})" for e in entries
        )

    cells = []
    before: list[dict[str, Any]] = []
    for rnd in view.rounds:
        after = at_site(rnd.ledger_after)
        active_before = {e["entry_id"] for e in before if e.get("status") == "active"}
        adopted = [
            e for e in after if int(e.get("round", 0)) == rnd.number and e.get("status") == "active"
        ]
        kept = [
            e for e in after if int(e.get("round", 0)) < rnd.number and e.get("status") == "active"
        ]
        retired = [
            e for e in after if e.get("status") != "active" and e["entry_id"] in active_before
        ]
        parts = []
        if adopted:
            parts.append("actions at this site: " + listed(adopted))
        if kept:
            parts.append(
                "actions at this site from an earlier round, still in force: " + listed(kept)
            )
        for status in ("undone", "replaced"):
            gone = [e for e in retired if e.get("status") == status]
            if gone:
                parts.append(f"{status} at this site in this round: " + listed(gone))
        text = "lasting: no removing action exists; " + (
            "; ".join(parts) if parts else "no setting in force at this site"
        )
        cells.append(text)
        before = after
    return cells


def truth_table(view: RunView) -> Table | None:
    truth = view.truth
    if truth is None:
        return None
    gains = {
        str(g["item"]): g
        for g in (truth.oracle_gains or [])
        if g.get("item_kind") == "fault_effect"
    }
    rows = []
    site_of = {str(r["robot_id"]): str(r["site_id"]) for rnd in view.rounds for r in rnd.ops_rows}
    for fault in truth.faults:
        key = target_key(fault["target"])
        cells = []
        if lasting(str(fault.get("kind"))):
            cells = _lasting_cells(view, fault, site_of)
        for rnd in view.rounds if not cells else []:
            causes = rnd.diagnoses[-1][1].get("causes", []) if rnd.diagnoses else []
            keys = [target_key(c["target"]) for c in causes]
            named = (
                "named"
                if key in keys
                else (
                    "named at another scope" if any(k[1] == key[1] for k in keys) else "not named"
                )
            )
            acted: list[dict[str, Any]] = []
            if rnd.decision is not None:
                acted = [
                    c["action"]
                    for c in [*rnd.decision.get("adopt", []), *rnd.decision.get("replace", [])]
                    if target_key(c["action"]["target"]) == key
                ]
                change = ", ".join(action_text(a) for a in acted) or "no change on it"
            elif rnd.fix_set is not None:
                ranks = [
                    str(i)
                    for i, r in enumerate(rnd.fix_set.get("actions", []), 1)
                    if target_key(r["action"]["target"]) == key
                ]
                change = f"ranked {', '.join(ranks)} (no adoption)" if ranks else "not ranked"
            else:
                change = "no decision"
            cells.append(f"{named}; {change}")
        schedule = fault.get("schedule") or {}
        gain = gains.get(str(fault.get("fault_id")))
        rows.append(
            [
                str(fault.get("fault_id")),
                str(fault.get("kind")),
                target_text(fault["target"]),
                f"day {schedule.get('onset_day')}",
                *cells,
                "not measured"
                if gain is None
                else (
                    f"{gain['net_points']:+.1f} ({gain['rescued']} rescued, "
                    f"{gain['broken']} broken)"
                ),
            ]
        )
    files = truth.files + [f for r in view.rounds for f in r.decision_files]
    return _table(
        view.sources,
        "truth",
        "What was actually wrong (planted truth)",
        [
            "fault",
            "kind",
            "where",
            "from",
            *[f"round {r.number}: final diagnosis; decision" for r in view.rounds],
            "removing it alone (oracle)",
        ],
        rows,
        "The causes we planted in the simulated fleet, shown only now, after the run; the agent "
        "never saw them."
        + (
            f" The fleet is registered as {truth.kind} material, which is why its truth may be "
            "shown."
            if truth.kind
            else ""
        )
        + " For each round, whether its final diagnosis named the fault's target "
        "and what its decision did there."
        + (
            " The last column is the world's oracle: the gain in points from removing that "
            "fault alone on the world's stored scoring day "
            f"({truth.oracle_scoring_episodes} attempts), a different set of attempts from "
            "the reserved panel."
            if truth.oracle_gains is not None
            else " This world keeps no oracle, so the effect of each fault is not measured."
        ),
        files,
        "fault_schedule of world_truth.json; each round's last diagnosis and its decision or "
        "ranked list; oracle.json gains of kind fault_effect",
    )


def by_state_rates(view: RunView) -> list[tuple[str, Rate]]:
    ev = view.evaluation
    if ev is None:
        return []
    return [
        (str(s["name"]), Rate(n=int(s["attempts"]), successes=int(s["successes"])))
        for s in ordered_states(ev.summary)
    ]


def robots_by_state(view: RunView) -> dict[str, dict[str, Rate]]:
    ev = view.evaluation
    out: dict[str, dict[str, Rate]] = {}
    if ev is None:
        return out
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in ev.attempts:
        groups[str(row["state"])][str(row["robot_id"])].append(bool(row["success"]))
    for state, robots in groups.items():
        out[state] = {r: Rate.of(v) for r, v in sorted(robots.items())}
    return out
