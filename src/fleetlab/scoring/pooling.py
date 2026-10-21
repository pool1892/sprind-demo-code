"""World-unit summaries: no episode-weighted pooling and no mixing of repeats."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from fleetlab.scoring.comparisons import exact_problem, required_baselines, required_values
from fleetlab.scoring.markdown import _table
from fleetlab.scoring.models import ScoreReport
from fleetlab.scoring.pooled_chance import pooled_floor
from fleetlab.scoring.roster import read_roster
from fleetlab.scoring.thresholds import PASS_MARKS, pass_marks


def summary(values: list[float | None], worlds: int) -> dict[str, Any]:
    held = [v for v in values if v is not None]
    return {
        "mean": sum(held) / len(held) if held else None,
        "minimum": min(held) if held else None,
        "maximum": max(held) if held else None,
        "worlds_valued": len(held),
        "worlds_expected": worlds,
        "complete": len(held) == worlds,
    }


def pool(reports: list[ScoreReport], *, roster: Path) -> dict[str, Any]:
    if not reports:
        raise ValueError("no score reports to pool")
    expected_worlds, roster_record = read_roster(roster)
    worlds = sorted(expected_worlds)
    unexpected = sorted({r.world_id for r in reports} - set(worlds))
    if unexpected:
        raise ValueError(f"reports outside expected worlds: {unexpected}")
    identities: set[tuple[str, str, str, int]] = set()
    hashes: dict[str, Any] = {}
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metrics: dict[tuple[str, str, int, str, str], dict[str, float | None]] = defaultdict(dict)
    baseline_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    analysis_paths: set[str] = set()
    passes = []
    error_rows = []
    truth_caveats = []
    provisional_metrics: dict[tuple[str, str, int, str], dict[str, list[str]]] = defaultdict(dict)
    for report in reports:
        if report.metadata.get("run_labels_verified") is not True or any(
            "unverified" in note for note in report.notes
        ):
            raise ValueError(
                f"unverified run labels refused for {report.world_id}; rescore the run"
            )
        report.pass_marks = pass_marks(report)
        method = str(report.metadata.get("method", "agent"))
        condition = str(report.metadata.get("condition", "logs alone"))
        if any(at.condition != condition for at in report.claim3.budgets):
            raise ValueError("report comparison condition differs from its run condition")
        repeat = int(report.metadata.get("repeat", 1))
        if report.claim1.truth_disagreements:
            truth_caveats.append(
                {
                    "world": report.world_id,
                    "method": method,
                    "condition": condition,
                    "repeat": repeat,
                    "faults": report.claim1.truth_disagreements,
                }
            )
        for name, diagnosis_metric in (
            ("recall", report.claim1.recall),
            ("precision", report.claim1.precision),
            ("older precision", report.claim1.precision_without_vague),
        ):
            if diagnosis_metric.provisional:
                provisional_metrics[(method, condition, repeat, name)][report.world_id] = (
                    report.claim1.truth_disagreements
                )
        if report.claim1.recall.provisional or report.claim1.precision.provisional:
            for name in ("known recall", "unseen recall", "known precision", "unseen precision"):
                provisional_metrics[(method, condition, repeat, name)][report.world_id] = (
                    report.claim1.truth_disagreements
                )
        key = (report.world_id, method, condition, repeat)
        if key in identities:
            raise ValueError(
                f"duplicate analysis cell: {key}; use distinct method names for model tiers"
            )
        identities.add(key)
        analysis_path = report.metadata.get("analysis_root")
        if analysis_path:
            if analysis_path in analysis_paths:
                raise ValueError(
                    "one runtime analysis cannot represent multiple repeats or conditions"
                )
            analysis_paths.add(analysis_path)
        world_hash = report.metadata.get("world_hash")
        if expected_worlds[report.world_id] != world_hash:
            raise ValueError(f"report world hash differs from roster for {report.world_id}")
        if report.world_id in hashes and hashes[report.world_id] != world_hash:
            raise ValueError(f"different world hashes for {report.world_id}")
        hashes[report.world_id] = world_hash
        for name, value in (
            ("recall", report.claim1.recall.value),
            ("precision", report.claim1.precision.value),
            ("older precision", report.claim1.precision_without_vague.value),
            ("decoy rejection", report.claim1.decoy_rejection.value),
            ("Spearman", report.claim2.spearman.value),
            ("MAE points", report.claim2.mae_points.value),
            ("slope", report.claim2.slope.value),
            ("prediction interval coverage", report.claim2.coverage.value),
        ):
            metrics[(method, condition, repeat, "all", name)][report.world_id] = value
        if report.claim1_curve is not None:
            for name in ("recall", "precision", "precision_without_vague", "decoy_rejection"):
                metrics[(method, condition, repeat, "all", "curve " + name + " (descriptive)")][
                    report.world_id
                ] = getattr(report.claim1_curve, name).value
        resolved = [c.resolved for c in report.claim1.confounders]
        metrics[(method, condition, repeat, "all", "confounder resolution")][report.world_id] = (
            sum(v is True for v in resolved) / len(resolved)
            if resolved and all(v is not None for v in resolved)
            else None
        )
        for split_name in ("known", "unseen"):
            split = report.splits.get("diagnosis", {}).get(split_name, {})
            for metric in ("recall", "precision", "decoy_rejection", "confounder_resolution"):
                metrics[(method, condition, repeat, "all", f"{split_name} {metric}")][
                    report.world_id
                ] = split.get(metric)
        for split_name, split in report.splits.get("ranking", {}).items():
            for metric in ("spearman", "mae_points", "slope"):
                metrics[(method, condition, repeat, "all", f"{split_name} {metric}")][
                    report.world_id
                ] = split.get(metric)
        for cohort_name, cohort in report.splits.get("decision_cohorts", {}).items():
            cohort_budgets = cohort["budgets"] or [
                {
                    "budget": b.label,
                    "methods": {
                        name: {"share": {"value": None}, "world_points": None, "gain": None}
                        for name in ("agent", *b.baselines)
                    },
                }
                for b in report.claim3.budgets
            ]
            for budget in cohort_budgets:
                for name, result in budget["methods"].items():
                    owner, cell, run = (
                        (method, condition, repeat) if name == "agent" else (name, "baseline", 0)
                    )
                    for metric, value in (
                        ("share", result["share"]["value"]),
                        ("world_points", result["world_points"]),
                        ("rescued", None if result["gain"] is None else result["gain"]["rescued"]),
                        ("broken", None if result["gain"] is None else result["gain"]["broken"]),
                    ):
                        metric_key = (owner, cell, run, budget["budget"], f"{cohort_name} {metric}")
                        held = metrics[metric_key]
                        if report.world_id in held and held[report.world_id] != value:
                            raise ValueError(f"conflicting cohort results: {metric_key}")
                        held[report.world_id] = value
        warnings = report.claim5.get("warnings", [])
        leads = [w["lead_days"] for w in warnings if w["lead_days"] is not None]
        metrics[(method, condition, repeat, "all", "mean warning lead days")][report.world_id] = (
            sum(leads) / len(leads)
            if leads and all(w.get("state") != "unavailable" for w in warnings)
            else None
        )
        logged = [w.get("logged_rate_measure", {}) for w in warnings]
        logged_leads = [w.get("lead_days") for w in logged if w.get("lead_days") is not None]
        metrics[(method, condition, repeat, "all", "logged rates mean warning lead days")][
            report.world_id
        ] = (
            sum(logged_leads) / len(logged_leads)
            if logged_leads and all(w.get("state") != "unavailable" for w in logged)
            else None
        )
        logged_timely = [w.get("before_crossing") for w in logged]
        metrics[(method, condition, repeat, "all", "logged rates warnings before crossing share")][
            report.world_id
        ] = (
            sum(logged_timely) / len(logged_timely)
            if logged_timely and all(v is not None for v in logged_timely)
            else None
        )
        timely = [w["before_crossing"] for w in warnings]
        metrics[(method, condition, repeat, "all", "warnings before crossing share")][
            report.world_id
        ] = sum(timely) / len(timely) if timely and all(v is not None for v in timely) else None
        metrics[(method, condition, repeat, "all", "warning false alarms")][report.world_id] = (
            report.claim5.get("false_alarm_count")
        )
        retrain = report.claim5.get("retrain", [])
        precision = [r["precision"] for r in retrain]
        metrics[(method, condition, repeat, "all", "mean indicator precision")][report.world_id] = (
            sum(precision) / len(precision)
            if precision and all(v is not None for v in precision)
            else None
        )
        for tier in report.claim6.get("tiers", []):
            model = f"{method}: {tier['backend']}/{tier['model']}/{tier['service_tier']}"
            for metric, value in tier["per_thousand_attempts"].items():
                metrics[(model, condition, repeat, "per 1000 attempts", metric)][
                    report.world_id
                ] = value
        for mark in report.pass_marks:
            passes.append(
                {
                    "world": report.world_id,
                    "method": method,
                    "condition": condition,
                    "repeat": repeat,
                    **mark,
                }
            )
        error_rows.append(
            {
                "world": report.world_id,
                "method": method,
                "condition": condition,
                "repeat": repeat,
                **report.errors,
            }
        )
        for budget in report.claim3.budgets:
            agent = budget.agent
            by_world[report.world_id].append(
                {
                    "method": method,
                    "condition": condition,
                    "repeat": repeat,
                    "budget": budget.label,
                    "rescued": None if agent.gain is None else agent.gain.rescued,
                    "broken": None if agent.gain is None else agent.gain.broken,
                    "net_points": agent.points,
                    "share": agent.share.value,
                    "achievable_points": budget.achievable.points,
                    "missing": agent.missing,
                }
            )
            for metric, value in (
                ("net_points", agent.points),
                ("share", agent.share.value),
                ("rescued", None if agent.gain is None else agent.gain.rescued),
                ("broken", None if agent.gain is None else agent.gain.broken),
            ):
                metrics[(method, condition, repeat, budget.label, metric)][report.world_id] = value
            baseline_values = dict(budget.baselines)
            if budget.no_data.points is not None:
                baseline_values["no-data ranker"] = budget.no_data
            for name, baseline_value in baseline_values.items():
                record = {
                    "method": name,
                    "condition": "baseline",
                    "repeat": 0,
                    "budget": budget.label,
                    "rescued": None if baseline_value.gain is None else baseline_value.gain.rescued,
                    "broken": None if baseline_value.gain is None else baseline_value.gain.broken,
                    "net_points": baseline_value.points,
                    "share": baseline_value.share.value,
                    "achievable_points": budget.achievable.points,
                    "missing": baseline_value.missing,
                }
                bkey = (report.world_id, name, budget.label)
                if bkey in baseline_records and baseline_records[bkey] != record:
                    raise ValueError(
                        f"conflicting baseline records: {bkey}; name distinct repeats explicitly"
                    )
                if bkey not in baseline_records:
                    by_world[report.world_id].append(record)
                baseline_records[bkey] = record
                for metric, measured in (
                    ("net_points", baseline_value.points),
                    ("share", baseline_value.share.value),
                    (
                        "rescued",
                        None if baseline_value.gain is None else baseline_value.gain.rescued,
                    ),
                    ("broken", None if baseline_value.gain is None else baseline_value.gain.broken),
                ):
                    metrics[(name, "baseline", 0, budget.label, metric)][report.world_id] = measured
            chance_key = (report.world_id, "random fixes", budget.label)
            chance_record = {
                "method": "random fixes",
                "condition": "baseline",
                "repeat": 0,
                "budget": budget.label,
                "rescued": None,
                "broken": None,
                "net_points": budget.random.mean_points.value,
                "share": budget.random.mean_share.value,
                "chance_p5": budget.random.p5_points.value,
                "chance_p95": budget.random.p95_points.value,
                "draws": budget.random.draws,
                "valued": budget.random.valued,
                "achievable_points": budget.achievable.points,
                "missing": budget.random.missing,
                "exact": budget.exact_chance,
                "exact_missing": exact_problem(budget.exact_chance),
                "bounds": budget.chance_bounds,
                "sampling_percentile_range": budget.sampling_percentile_range,
                "chance_command": budget.chance_command,
            }
            if chance_key in baseline_records and baseline_records[chance_key] != chance_record:
                raise ValueError(f"different chance records for {chance_key}")
            if chance_key not in baseline_records:
                by_world[report.world_id].append(chance_record)
            baseline_records[chance_key] = chance_record
            for name, value in (
                ("chance_mean", budget.random.mean_points.value),
                ("chance_p5", budget.random.p5_points.value),
                ("chance_p95", budget.random.p95_points.value),
                ("chance_share", budget.random.mean_share.value),
                ("chance_p95_share", budget.random.p95_share.value),
            ):
                metrics[("random fixes", "baseline", 0, budget.label, name)][report.world_id] = (
                    value
                )
    pooled = [
        {
            "method": method,
            "condition": condition,
            "repeat": repeat,
            "budget": budget,
            "metric": metric,
            **summary([values.get(w) for w in worlds], len(worlds)),
            "missing_worlds": [w for w in worlds if values.get(w) is None],
            "provisional": bool(provisional_metrics.get((method, condition, repeat, metric))),
            "truth_disagreements": provisional_metrics.get((method, condition, repeat, metric), {}),
        }
        for (method, condition, repeat, budget, metric), values in sorted(metrics.items())
    ]
    # Both repeats and both required claim-2 conditions are explicit, including absent cells.
    required = []
    methods = sorted({str(r.metadata.get("method", "agent")) for r in reports})
    for world in worlds:
        for method in methods:
            for condition in ("logs alone", "with experiments"):
                for repeat in (1, 2):
                    required.append(
                        {
                            "world": world,
                            "method": method,
                            "condition": condition,
                            "repeat": repeat,
                            "available": (world, method, condition, repeat) in identities,
                        }
                    )
    ratios = []
    pooled_marks = []
    grouped: dict[tuple[str, str, int], dict[str, ScoreReport]] = defaultdict(dict)
    for report in reports:
        grouped[
            (
                str(report.metadata.get("method", "agent")),
                str(report.metadata.get("condition", "logs alone")),
                int(report.metadata.get("repeat", 1)),
            )
        ][report.world_id] = report
    for (method, condition, repeat), cells in sorted(grouped.items()):
        for metric, threshold in PASS_MARKS.items():
            marks = {
                world: next((p for p in r.pass_marks if p["metric"] == metric), {})
                for world, r in cells.items()
            }
            values = {world: mark.get("value") for world, mark in marks.items()}
            absent = [
                w
                for w in worlds
                if values.get(w) is None or marks.get(w, {}).get("status") == "unavailable"
            ]
            mark_summary = summary([values.get(w) for w in worlds], len(worlds))
            status = "unavailable"
            if not absent:
                value = mark_summary["mean"]
                status = (
                    ("met" if all(marks[w]["status"] == "met" for w in worlds) else "missed")
                    if metric == "claim3.above_chance"
                    else ("met" if value >= threshold else "missed")
                )
            pooled_marks.append(
                {
                    "method": method,
                    "condition": condition,
                    "repeat": repeat,
                    "metric": metric,
                    "threshold": threshold,
                    "comparison": ">" if metric == "claim3.above_chance" else ">=",
                    "value": mark_summary["mean"],
                    "status": status,
                    "worlds_met": sum(m.get("status") == "met" for m in marks.values()),
                    "worlds_expected": len(worlds),
                    "missing_worlds": absent,
                    "provisional": any(mark.get("provisional", False) for mark in marks.values()),
                    "truth_disagreements": {
                        w: mark["truth_disagreements"]
                        for w, mark in marks.items()
                        if mark.get("provisional")
                    },
                }
            )
        budgets = {
            w: next(b for b in r.claim3.budgets if b.label == "k = 3") for w, r in cells.items()
        }
        absent_baselines = {
            w: required_values(budgets[w])[1] for w in budgets if required_values(budgets[w])[1]
        }
        missing_worlds = sorted(set(worlds) - set(cells))
        floor = pooled_floor(budgets)
        best_world = []
        by_baseline: dict[str, list[float]] = defaultdict(list)
        agent_values = [b.agent.share.value for b in budgets.values()]
        ratio = None
        single_ratio = None
        single_name = None
        best_mean = None
        single_mean = None
        reason = (
            "expected worlds absent: " + ", ".join(missing_worlds)
            if missing_worlds
            else "incomplete agent, required baseline or exact chance comparison"
        )
        if not missing_worlds and not absent_baselines:
            for at in budgets.values():
                required_values_at, _ = required_values(at)
                shares = {name: value.share.value for name, value in required_values_at.items()}
                if any(v is None for v in shares.values()):
                    break
                best_world.append(max(float(v) for v in shares.values() if v is not None))
                for name in required_baselines(condition):
                    by_baseline[name].append(float(shares[name] or 0))
            if len(best_world) == len(budgets):
                best_mean = sum(best_world) / len(worlds)
                means = {name: sum(v) / len(worlds) for name, v in by_baseline.items()}
                single_name = max(means, key=lambda name: means[name])
                single_mean = means[single_name]
                if floor["p95_high"] is not None and all(v is not None for v in agent_values):
                    agent_mean = sum(float(v) for v in agent_values if v is not None) / len(worlds)
                    if best_mean <= floor["p95_high"]:
                        reason = "the baselines recover no more than pooled chance"
                    elif best_mean > 0:
                        ratio = agent_mean / best_mean
                        reason = "ratio to the best baseline chosen world by world"
                    if single_mean > max(0, floor["p95_high"]):
                        single_ratio = agent_mean / single_mean
        if missing_worlds:
            floor = {"p95_low": None, "p95_high": None, "missing_worlds": missing_worlds}
        ratios.append(
            {
                "method": method,
                "condition": condition,
                "repeat": repeat,
                "best_baseline": "best required baseline in each world",
                "baseline_mean_share": best_mean,
                "ratio": ratio,
                "verdict": reason,
                "missing_baselines": absent_baselines,
                "missing_worlds": missing_worlds,
                "pooled_chance": floor,
                "best_on_average": {
                    "baseline": single_name,
                    "mean_share": single_mean,
                    "ratio": single_ratio,
                },
            }
        )
    return {
        "worlds": worlds,
        "roster": roster_record,
        "world_hashes": hashes,
        "truth_caveats": truth_caveats,
        "per_world": {world: by_world[world] for world in worlds},
        "world_metrics": {
            world: [
                {
                    "method": method,
                    "condition": condition,
                    "repeat": repeat,
                    "budget": budget,
                    "metric": metric,
                    "value": values.get(world),
                }
                for (method, condition, repeat, budget, metric), values in sorted(metrics.items())
            ]
            for world in worlds
        },
        "pooled": pooled,
        "required_cells": required,
        "headline_ratios": ratios,
        "pass_marks": passes,
        "pooled_pass_marks": pooled_marks,
        "errors": error_rows,
        "analysis_reports": [r.model_dump(mode="json") for r in reports],
        "basis": (
            "arithmetic mean, minimum and maximum of world metrics; "
            "repeats stay separate; repeat 1 is primary"
        ),
    }


def pool_markdown(data: dict[str, Any]) -> str:
    lines = ["# Development-world comparison", "", data["basis"], ""]
    lines += [
        f"World roster: {data['roster']['path']}",
        f"Roster SHA-256: {data['roster']['sha256']}",
        "",
    ]
    if data.get("truth_caveats"):
        lines += [
            "**Provisional diagnosis metrics: cause units disagree with per-fault truth.**",
            "",
            *[
                f"{c['world']}, {c['method']}, {c['condition']}, repeat {c['repeat']}: "
                f"faults {', '.join(c['faults'])}."
                for c in data["truth_caveats"]
            ],
            "",
        ]
    for world, rows in data["per_world"].items():
        lines += [f"## {world}", ""]
        lines += _table(
            [
                "Method",
                "Condition",
                "Repeat",
                "Budget",
                "Rescued",
                "Broken",
                "Net points",
                "Share",
                "Achievable points",
                "200 draws p5-p95; valued / draws (descriptive at count budgets)",
                "Exact count-budget p95 (points)",
                "Cost-budget p95 confidence interval",
            ],
            [
                (
                    r["method"],
                    r["condition"],
                    r["repeat"],
                    r["budget"],
                    r["rescued"],
                    r["broken"],
                    r["net_points"] if r["net_points"] is not None else r["missing"],
                    r["share"],
                    r["achievable_points"],
                    (
                        f"{r.get('chance_p5')}-{r.get('chance_p95')}; "
                        f"{r.get('valued')} / {r.get('draws')}"
                        if r["method"] == "random fixes"
                        else ""
                    ),
                    (
                        str(
                            (r["exact"].get("bounds") or {}).get("p95_points")
                            or r["exact"].get("p95_points")
                        )
                        if r.get("exact") and not r.get("exact_missing")
                        else "not available; " + str(r.get("chance_command"))
                        if r["method"] == "random fixes" and r["budget"].startswith("k =")
                        else ""
                    ),
                    r.get("sampling_percentile_range") or "",
                )
                for r in rows
            ],
        )
        relevant = [r for r in data["analysis_reports"] if r["world_id"] == world]
        for method in sorted({r["metadata"].get("method", "agent") for r in relevant}):
            columns = [
                (condition, repeat)
                for repeat in (1, 2)
                for condition in ("logs alone", "with experiments")
            ]
            comparison = []
            for label, claim, metric in (
                ("recall", "claim1", "recall"),
                ("precision", "claim1", "precision"),
                ("older precision", "claim1", "precision_without_vague"),
                ("decoy rejection", "claim1", "decoy_rejection"),
                ("Spearman", "claim2", "spearman"),
                ("MAE points", "claim2", "mae_points"),
                ("slope", "claim2", "slope"),
                ("prediction interval coverage", "claim2", "coverage"),
            ):
                values = []
                for condition, repeat in columns:
                    record = next(
                        (
                            r
                            for r in relevant
                            if r["metadata"].get("method", "agent") == method
                            and r["metadata"].get("condition", "logs alone") == condition
                            and r["metadata"].get("repeat", 1) == repeat
                        ),
                        None,
                    )
                    values.append(
                        "unavailable" if record is None else record[claim][metric]["value"]
                    )
                comparison.append((label, *values))
            lines += [f"### {method}: conditions and repeats", ""]
            lines += _table(["Metric", *(f"{c}, repeat {r}" for c, r in columns)], comparison)
        lines += ["### All world metrics, including type splits and runtime costs", ""]
        lines += _table(
            ["Method", "Condition", "Repeat", "Budget", "Metric", "Value"],
            [
                (r["method"], r["condition"], r["repeat"], r["budget"], r["metric"], r["value"])
                for r in data["world_metrics"][world]
            ],
        )
    lines += ["## Pooled over worlds", ""]
    lines += _table(
        [
            "Method",
            "Condition",
            "Repeat",
            "Budget",
            "Metric",
            "Mean",
            "Minimum",
            "Maximum",
            "Worlds valued / expected",
            "Provisional truth",
        ],
        [
            (
                r["method"],
                r["condition"],
                r["repeat"],
                r["budget"],
                r["metric"],
                r["mean"],
                r["minimum"],
                r["maximum"],
                f"{r['worlds_valued']} / {r['worlds_expected']}",
                r.get("truth_disagreements", {}),
            )
            for r in data["pooled"]
        ],
    )
    lines += ["## Headline ratios", ""]
    lines += _table(
        ["Method", "Condition", "Repeat", "Best baseline", "Ratio", "Verdict"],
        [
            (r["method"], r["condition"], r["repeat"], r["best_baseline"], r["ratio"], r["verdict"])
            for r in data["headline_ratios"]
        ],
    )
    lines += ["## Frozen pass marks, pooled", ""]
    lines += _table(
        [
            "Method",
            "Condition",
            "Repeat",
            "Metric",
            "Mean",
            "Mark",
            "Result",
            "Worlds met / expected",
            "Missing worlds",
        ],
        [
            (
                r["method"],
                r["condition"],
                r["repeat"],
                r["metric"],
                r["value"],
                f"{r['comparison']} {r['threshold']}",
                r["status"] + (" (provisional)" if r.get("provisional") else ""),
                f"{r['worlds_met']} / {r['worlds_expected']}",
                r["missing_worlds"],
            )
            for r in data["pooled_pass_marks"]
        ],
    )
    for r in data["headline_ratios"]:
        lines += [
            f"{r['method']}, {r['condition']}, repeat {r['repeat']}: "
            f"pooled floor {r['pooled_chance']}; best on average {r['best_on_average']}; "
            f"missing baselines {r['missing_baselines']}.",
            "",
        ]
    lines += ["## Frozen pass marks, per world", ""]
    lines += _table(
        ["World", "Method", "Condition", "Repeat", "Metric", "Value", "Pass mark", "Result"],
        [
            (
                r["world"],
                r["method"],
                r["condition"],
                r["repeat"],
                r["metric"],
                r["value"],
                f"{r['comparison']} {r['threshold']}",
                r["status"] + (" (provisional)" if r.get("provisional") else ""),
            )
            for r in data["pass_marks"]
        ],
    )
    from fleetlab.scoring.feedback import feedback_markdown

    for record in data["errors"]:
        lines += [
            f"### {record['world']}: {record['method']}, {record['condition']}, "
            f"repeat {record['repeat']}",
            "",
        ]
        lines += feedback_markdown(record)
    lines += [
        "",
        "Full per-analysis costs and splits are retained in the JSON.",
        "",
    ]
    return "\n".join(lines)
