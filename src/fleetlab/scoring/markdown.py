"""The report of one world as Markdown: a headline table, then one section per claim."""

from __future__ import annotations

from collections.abc import Sequence

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.oracle import Gain
from fleetlab.scoring.comparisons import exact_problem
from fleetlab.scoring.models import FixSetValue, Metric, ScoreReport

# Replays listed one by one in the report; the JSON holds them all.
MAX_REPLAYS_LISTED = 40


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return [*out, ""]


def fmt(metric: Metric, digits: int = 2, signed: bool = False) -> str:
    """A metric's value, or a dash with the reason it is missing."""
    if metric.value is None:
        return f"- ({metric.missing})"
    return f"{metric.value:+.{digits}f}" if signed else f"{metric.value:.{digits}f}"


def _number(value: float | None, digits: int = 2, signed: bool = False, why: str = "") -> str:
    if value is None:
        return f"- ({why})" if why else "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def target_text(target: Target) -> str:
    thing = target.thing
    return target.scope if thing is None else f"{target.scope} {thing.value}"


def action_text(action: Action) -> str:
    """For example `recalibrate r01 scene_camera`, `set s2 control.action_delay_steps = 0`."""
    text = f"{action.verb.value} {target_text(action.target)}"
    if action.value is not None:
        text += f" = {action.value}"
    return text


def _gain(gain: Gain | None) -> str:
    return "-" if gain is None else f"{gain.rescued} / {gain.broken}"


def _fix_set_points(value: FixSetValue) -> str:
    return _number(value.points, 2, signed=True, why=value.missing or "")


def _fix_set_share(value: FixSetValue) -> str:
    """The share, or a bare dash when the points beside it already say why it is missing."""
    return "-" if value.gain is None else fmt(value.share)


def _yes(flag: bool | None) -> str:
    return "-" if flag is None else ("yes" if flag else "no")


def report_markdown(report: ScoreReport) -> str:
    """The report of one world as Markdown tables: the headline, then one section per claim,
    traceability, the replays the scoring waits for, and notes."""
    lines: list[str] = [f"# Score of world {report.world_id}", ""]
    method = str(report.metadata.get("method", "agent"))
    lines += [
        f"Method: **{method}**; condition: **{report.metadata.get('condition', 'not supplied')}**; "
        f"repeat: **{report.metadata.get('repeat', 'not supplied')}**.",
        "",
    ]
    lines += [
        f"{report.scoring_episodes} scoring episodes; random fixes: {report.draws} draws, "
        f"seed {report.seed}. Points are points of fleet success over the scoring window; a "
        "share is the net gain over the achievable gain at the same budget.",
        "",
    ]

    if report.claim1.truth_disagreements:
        lines += [
            "**PROVISIONAL recall and precision: truth lists disagree for "
            + ", ".join(report.claim1.truth_disagreements)
            + ".**",
            "",
        ]
    head = report.headline
    lines += [f"## Headline: share of the achievable gain at k = {head.k}", ""]
    lines += _table(
        ["Method", "Share of achievable", "Net (points)"],
        [
            (method if r.method == "agent" else r.method, fmt(r.share), fmt(r.points, signed=True))
            for r in head.rows
        ],
    )
    lines += [
        "Chance (oracle exact 95th percentile share, upper bound if ranged): "
        f"{fmt(head.chance_p95_share)}. "
        f"Best baseline: {head.best_baseline or '-'}. "
        f"Ratio: {'-' if head.ratio is None else f'{head.ratio:.2f}'}.",
        "",
        f"**{head.verdict}**",
        "",
    ]
    if head.opponents:
        lines += _table(
            ["Comparison opponent", "Net (points)", "Share of achievable"],
            [
                (
                    name + (f" ({value['name']})" if value.get("name") else ""),
                    _number(value.get("points"), signed=True, why=value.get("missing", "")),
                    _number(value.get("share")),
                )
                for name, value in head.opponents.items()
            ],
        )

    c1 = report.claim1
    lines += ["## Claim 1: diagnosis", ""]
    if c1.missing:
        suffix = "; no causes found" if c1.missing == "no diagnosis submitted" else ""
        lines += [f"{c1.missing}{suffix}.", ""]
    if report.claim1_curve is not None:
        curve = report.claim1_curve
        if curve.truth_disagreements:
            lines += [
                "Curve audit: generator lists disagree for "
                + ", ".join(curve.truth_disagreements)
                + "; its recall and precision are provisional and descriptive only.",
                "",
            ]
        lines += _table(
            ["Membership", "Causes", "Recall", "Precision", "Older precision", "Decoy rejection"],
            [
                (
                    "Measured (decides marks)",
                    c1.detectable,
                    fmt(c1.recall),
                    fmt(c1.precision),
                    fmt(c1.precision_without_vague),
                    fmt(c1.decoy_rejection),
                ),
                (
                    "Curves (descriptive only)",
                    curve.detectable,
                    fmt(curve.recall),
                    fmt(curve.precision),
                    fmt(curve.precision_without_vague),
                    fmt(curve.decoy_rejection),
                ),
            ],
        )
        measured_units = report.measured_causes.get("causes", [])
        lines += [
            "Measured causes: "
            + (", ".join(unit["unit"] for unit in measured_units) or "none/unavailable")
            + ".",
            "Curve causes: "
            + (
                ", ".join(report.measured_causes.get("curve_causes", []))
                or ", ".join(unit["unit"] for unit in curve.units if unit["detectable"])
                or ", ".join(f.fault_id for f in curve.faults if f.detectable)
                or "none"
            )
            + ".",
            "",
        ]
    lines += _table(
        ["Metric", "Value"],
        [
            ("detectable faults (n)", c1.detectable),
            (f"named causes (confidence >= 0.5 or top {c1.named_cut})", c1.named),
            ("recall", fmt(c1.recall)),
            ("precision", fmt(c1.precision)),
            (
                "older precision excluding targetless causes at scopes with effects",
                fmt(c1.precision_without_vague),
            ),
            ("decoy rejection", fmt(c1.decoy_rejection)),
            ("hits", c1.hits),
            ("false positives", c1.false_positives),
            ("wrong direction", c1.wrong_direction),
            ("small (neither)", c1.small),
            ("targetless causes (false positives)", c1.targetless),
            ("targetless at scopes with effects (older precision excludes these)", c1.vague),
        ],
    )
    lines += ["### Causes", ""]
    lines += _table(
        ["#", "Target", "Direction", "Confidence", "Named", "Verdict", "Matches"],
        [
            (
                c.index,
                target_text(c.target),
                c.direction.value,
                f"{c.confidence:.2f}",
                _yes(c.named),
                c.verdict,
                ", ".join(c.faults) or "-",
            )
            for c in c1.causes
        ],
    )
    lines += ["### Faults", ""]
    lines += _table(
        ["Fault", "Kind", "Role", "Target", "Detectable", "Small", "Found", "Named by"],
        [
            (
                f.fault_id,
                f.kind,
                f.role,
                target_text(f.target),
                _yes(f.detectable),
                _yes(f.small),
                _yes(f.found),
                ", ".join(str(i) for i in f.named_by) or "-",
            )
            for f in c1.faults
        ],
    )
    lines += ["### Confounders", ""]
    if c1.confounders:
        lines += _table(
            ["Inert", "Inert named", "Effective", "Effective named", "Resolved"],
            [
                (
                    c.inert,
                    _yes(c.inert_named),
                    c.effective or "-",
                    _yes(c.effective_found),
                    _yes(c.resolved),
                )
                for c in c1.confounders
            ],
        )
    else:
        lines += ["No inert entry in this world.", ""]
    lines += [f"Note: {c1.confounder_note}.", ""]

    c2 = report.claim2
    lines += ["## Claim 2: expected against true gains", ""]
    if c2.missing:
        lines += [f"Not scored: {c2.missing}.", ""]
    else:
        rows: list[tuple[str, str]] = [
            ("actions valued", f"{c2.valued} of {len(c2.actions)}"),
            ("Spearman rank correlation", fmt(c2.spearman)),
            ("mean absolute error (points)", fmt(c2.mae_points)),
            ("slope of true on expected", fmt(c2.slope)),
            ("true gain within the interval (share)", fmt(c2.coverage)),
        ]
        rows += [(f"share of achievable, top {k}", fmt(m)) for k, m in c2.top_k_share.items()]
        lines += _table(["Metric", "Value"], rows)
        lines += _table(
            ["Rank", "Action", "Expected", "Interval", "True (points)", "Rescued / broken", "In"],
            [
                (
                    a.rank,
                    action_text(a.action),
                    f"{a.expected_gain:+.2f}",
                    f"{a.low:+.2f} to {a.high:+.2f}",
                    _number(a.true_points, 2, signed=True, why=a.missing or ""),
                    _gain(a.true),
                    _yes(a.covered),
                )
                for a in c2.actions
            ],
        )

    c3 = report.claim3
    lines += ["## Claim 3: fix sets at the scoring budgets", ""]
    lines += _table(
        [
            "Budget",
            "Achievable (points)",
            "Agent (points)",
            "Agent rescued / broken",
            "Agent share",
            "No-data (points)",
            "No-data share",
        ],
        [
            (
                s.label,
                _number(s.achievable.points, 2, signed=True, why=s.achievable.missing or ""),
                _fix_set_points(s.agent),
                _gain(s.agent.gain),
                _fix_set_share(s.agent),
                _fix_set_points(s.no_data),
                _fix_set_share(s.no_data),
            )
            for s in c3.budgets
        ],
    )
    lines += ["### Supplied baselines", ""]
    lines += _table(
        ["Budget", "Method", "Rescued / broken", "Net points", "Share", "Hours"],
        [
            (b.label, name, _gain(v.gain), _fix_set_points(v), _fix_set_share(v), v.hours)
            for b in c3.budgets
            for name, v in b.baselines.items()
        ],
    )
    lines += ["### Random fixes", ""]
    lines += _table(
        [
            "Budget",
            "Valued / skipped (replay, error)",
            "Mean (points)",
            "5th pct",
            "95th pct",
            "Draws > 0",
            "Mean share",
            "95th pct share",
        ],
        [
            (
                s.label,
                f"{s.random.valued} / {s.random.skipped_fresh_replay}, {s.random.skipped_error}"
                if s.random.missing is None or s.random.draws
                else f"- ({s.random.missing})",
                fmt(s.random.mean_points, signed=True),
                fmt(s.random.p5_points, signed=True),
                fmt(s.random.p95_points, signed=True),
                fmt(s.random.above_zero),
                fmt(s.random.mean_share),
                fmt(s.random.p95_share),
            )
            for s in c3.budgets
        ],
    )
    lines += ["### Oracle exact chance (count budgets)", ""]
    lines += _table(
        ["Budget", "Mean points", "5th percentile", "95th percentile", "Bounds", "Missing"],
        [
            (
                b.label,
                (b.exact_chance or {}).get("mean_points")
                if not exact_problem(b.exact_chance)
                else None,
                (b.exact_chance or {}).get("p5_points")
                if not exact_problem(b.exact_chance)
                else None,
                (b.exact_chance or {}).get("p95_points")
                if not exact_problem(b.exact_chance)
                else None,
                (b.exact_chance or {}).get("bounds") if not exact_problem(b.exact_chance) else None,
                b.chance_missing or exact_problem(b.exact_chance) or "",
            )
            for b in c3.budgets
            if b.budget.kind == "count"
        ],
    )
    for b in c3.budgets:
        if b.budget.kind == "cost":
            lines += [
                f"{b.label}: oracle draw bounds {b.chance_bounds}; sampling percentile range: "
                f"{b.sampling_percentile_range or 'unavailable in oracle report'}. "
                "No pass mark at this budget.",
                "",
            ]
    lines += ["### The fix sets", ""]
    lines += _table(
        ["Budget", "Agent's actions", "Hours", "Achievable with"],
        [
            (
                s.label,
                "; ".join(action_text(a) for a in s.agent.actions) or "-",
                f"{s.agent.hours:g}",
                "best k-set of resolved fixes"
                if s.achievable.fixes is None
                else "; ".join(action_text(a) for a in s.achievable.fixes) or "nothing",
            )
            for s in c3.budgets
        ],
    )

    lines += ["## Frozen pass marks (2026-10-08)", ""]
    lines += _table(
        ["Metric", "Value", "Pass mark", "Result"],
        [
            (
                p["metric"],
                p["value"] if p["value"] is not None else "unavailable",
                f"{p['comparison']} {p['threshold']}",
                p["status"]
                + (" (provisional)" if p.get("provisional") else "")
                + (f" ({p['missing']})" if p.get("missing") else ""),
            )
            for p in report.pass_marks
        ],
    )
    lines += ["## Claim 5: foresight", "", report.claim5.get("meaning", ""), ""]
    lines += _table(
        [
            "Drift",
            "Scope",
            "Drift day",
            "Old wording day",
            "State",
            "Warning day",
            "Lead (days)",
            "Before crossing",
            "Missing",
        ],
        [
            (
                w["fault_id"],
                w["scope"],
                w["crossing_day"],
                w.get("old_wording_crossing_day"),
                w.get("state"),
                w["issued_day"],
                w["lead_days"],
                w["before_crossing"],
                w["missing"] or "",
            )
            for w in report.claim5.get("warnings", [])
        ],
    )
    lines += ["### Warning measured against logged rates (section 4)", ""]
    lines += _table(
        ["Drift", "Crossing day", "State", "Noticed", "Lead days", "Forecast errors", "Missing"],
        [
            (
                w["fault_id"],
                w.get("logged_rate_measure", {}).get("crossing_day"),
                w.get("logged_rate_measure", {}).get("state"),
                w.get("logged_rate_measure", {}).get("noticed"),
                w.get("logged_rate_measure", {}).get("lead_days"),
                w.get("logged_rate_measure", {}).get("forecast_errors"),
                w.get("logged_rate_measure", {}).get("missing"),
            )
            for w in report.claim5.get("warnings", [])
        ],
    )
    lines += [
        f"False alarms: {report.claim5.get('false_alarm_count', 0)}. "
        f"{report.claim5.get('false_alarms', [])}",
        "",
    ]
    lines += _table(
        ["Drift", "Forecast errors (expected day minus drift day)"],
        [(w["fault_id"], w.get("forecast_errors", [])) for w in report.claim5.get("warnings", [])],
    )
    lines += _table(
        ["Training mode", "Examples", "Healthy failures", "Precision", "Missing"],
        [
            (
                r["failure_mode"],
                r["examples"],
                r["healthy_failures"],
                r["precision"],
                r["missing"] or "",
            )
            for r in report.claim5.get("retrain", [])
        ],
    )
    if report.claim5.get("missing_examples"):
        lines += [
            f"Missing healthy examples: {report.claim5['missing_examples']}. "
            f"{report.claim5.get('replay_request_command')}",
            "",
        ]
    lines += ["## Additive-pair prose remarks (not scored)", ""]
    lines += _table(
        ["Cause index", "Pair", "Remark"],
        [(r["index"], r["pair"], r["matched"]) for r in report.remarks],
    )
    lines += ["## Claim 6: runtime costs", "", report.claim6.get("wall_time_basis", ""), ""]
    lines += [
        f"Attempts analysed: {report.claim6.get('attempts_analysed', 'unavailable')}; "
        f"basis: {report.claim6.get('denominator_basis', 'unavailable')}. "
        f"Accounting: {report.claim6.get('missing') or 'ledger-backed'}.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Service tier",
            "Calls / failed",
            "Input / cached / output tokens per 1,000 attempts",
            "Wall seconds per 1,000 attempts",
            "API-price USD per 1,000 attempts",
        ],
        [
            (
                t["model"],
                t["service_tier"],
                f"{t['calls']} / {t['failed_calls']}",
                " / ".join(
                    str(t["per_thousand_attempts"].get(k))
                    for k in ("input_tokens", "cached_input_tokens", "output_tokens")
                ),
                t["per_thousand_attempts"]["wall_seconds"],
                t["per_thousand_attempts"]["api_price_equivalent_usd"],
            )
            for t in report.claim6.get("tiers", [])
        ],
    )
    lines += ["## Known and unseen types", ""]
    lines += _table(
        ["Types", "Detectable", "Found", "Recall", "Precision", "Decoy rejection", "Confounders"],
        [
            (
                name,
                row["detectable"],
                row["found"],
                row["recall"],
                row["precision"],
                row.get("decoy_rejection"),
                row.get("confounder_resolution"),
            )
            for name, row in report.splits.get("diagnosis", {}).items()
            if name in ("known", "unseen")
        ],
    )
    lines += _table(
        ["Action types", "Ranked", "Valued", "Spearman", "MAE points", "Slope"],
        [
            (name, row["ranked"], row["valued"], row["spearman"], row["mae_points"], row["slope"])
            for name, row in report.splits.get("ranking", {}).items()
        ],
    )
    lines += [
        "Decision cohorts are disjoint. Shares use each cohort's own achievable gain; "
        "world points use the original world's scoring-attempt denominator.",
        "",
    ]
    lines += _table(
        ["Cohort", "Attempts", "Budget", "Method", "Rescued", "Broken", "World points", "Share"],
        [
            (
                cohort_name,
                cohort["episodes"],
                budget["budget"],
                method,
                None if row["gain"] is None else row["gain"]["rescued"],
                None if row["gain"] is None else row["gain"]["broken"],
                row["world_points"],
                row["share"]["value"],
            )
            for cohort_name, cohort in report.splits.get("decision_cohorts", {}).items()
            for budget in cohort["budgets"]
            for method, row in budget["methods"].items()
        ],
    )
    from fleetlab.scoring.feedback import feedback_markdown

    lines += feedback_markdown(report.errors)
    tr = report.traceability
    lines += ["## Traceability of the answer's numbers", ""]
    if tr.missing:
        lines += [f"Not checked: {tr.missing}.", ""]
    else:
        lines += _table(
            [
                "Traced",
                "Echo only",
                "Untraced",
                "Small whole numbers (not checked)",
                "Calls searched",
                "Not checked (confidences)",
            ],
            [
                (
                    tr.traced,
                    tr.echo,
                    tr.untraced,
                    tr.small,
                    f"{tr.calls_searched} of {tr.calls}",
                    len(tr.not_checked),
                )
            ],
        )
        untraced = [n for n in tr.numbers if n.status == "untraced"]
        if untraced:
            lines += ["### Untraced", ""]
            lines += _table(["Where", "As written"], [(n.where, n.text) for n in untraced])
        echoes = [n for n in tr.numbers if n.status == "echo"]
        if echoes:
            lines += ["### Echo only (the call's own arguments carry the number)", ""]
            lines += _table(
                ["Where", "As written", "Call"], [(n.where, n.text, n.source) for n in echoes]
            )

    needs = report.needs_fresh_replay
    lines += ["## Replays the scoring waits for", ""]
    if not needs:
        lines += [
            "No missing replay is listed for ranked actions, supplied fix sets or the "
            "removal ceiling. Chance availability is reported separately.",
            "",
        ]
    else:
        episodes = len({n.episode_id for n in needs})
        lines += [
            f"{len(needs)} replays of {episodes} episodes (needs_fresh_replay in the JSON).",
            "",
        ]
        lines += _table(
            ["Episode", "Faults in force", "Needed for"],
            [
                (n.episode_id, ", ".join(n.fault_ids) or "none", "; ".join(n.needed_for))
                for n in needs[:MAX_REPLAYS_LISTED]
            ],
        )
        if len(needs) > MAX_REPLAYS_LISTED:
            lines += [f"... and {len(needs) - MAX_REPLAYS_LISTED} more.", ""]

    if report.notes:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in report.notes]
        lines += [""]
    return "\n".join(lines)
