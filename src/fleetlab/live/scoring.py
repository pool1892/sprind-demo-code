"""Score stored claim-4 records, including paired equal-time comparisons.

Linear segments mean randomizing between two *whole-window* operating
policies. They do not predict a partly rerun queue, and never extrapolate.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.policies.teleop.measure import dump


def summarize(record: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = record["episodes"] if rows is None else rows
    failures = [r for r in rows if not r["as_run"]]
    successes = [r for r in rows if r["as_run"]]
    snapshots = [r for r in rows if r["snapshot_without"] is not None]

    def count(items: list[dict[str, Any]], condition: Any) -> int:
        return sum(bool(condition(r)) for r in items)

    def weighted(condition: Any) -> float:
        return sum(float(r["weight"]) * float(condition(r)) for r in rows)

    def share(n: int, d: int) -> float | None:
        return n / d if d else None

    rescued = count(failures, lambda r: r["success"])
    no_person = count(failures, lambda r: r["success"] and r["operator_s"] == 0)
    with_person = rescued - no_person
    lost = count(successes, lambda r: not r["success"])
    false_alarms = count(
        successes,
        lambda r: r["request"] or any(d["decision"] != "continue" for d in r["decisions"]),
    )
    snapshot_fail = [r for r in snapshots if not r["snapshot_without"]]
    snapshot_good = [r for r in snapshots if r["snapshot_without"]]
    control = {
        r["episode_id"]: r["success"] for r in record.get("hold_control", {}).get("episodes", [])
    }
    control_net = (
        sum(r["weight"] * (int(control[r["episode_id"]]) - int(r["as_run"])) for r in rows)
        if control
        else 0.0
    )
    total = sum(r["weight"] for r in rows)
    trajectories = [r for r in rows if r["handover"]]
    handover_fail = [
        r
        for r in trajectories
        if r.get("handover_snapshot_without", r["snapshot_without"]) is False
    ]
    handover_good = [
        r for r in trajectories if r.get("handover_snapshot_without", r["snapshot_without"]) is True
    ]
    return {
        "rule": record["rule"],
        "world": record["population"].get("world", "fixture"),
        "latency_route": record.get("latency_route", "subscription"),
        "measured_code": record.get("config", {}).get("code"),
        "after": record["after"],
        "cost_ratio": record["cost_ratio"],
        "operators": record["settings"]["operators"],
        "cohort_episodes": len(rows),
        "would_fail_as_run": len(failures),
        "sampled_as_run_successes": len(successes),
        "rescued_as_run": rescued,
        "rescued_without_person": no_person,
        "rescued_with_person": with_person,
        "rescued_without_person_share": share(no_person, len(failures)),
        "rescued_with_person_share": share(with_person, len(failures)),
        "rescued_share": share(rescued, len(failures)),
        "successes_lost": lost,
        "successes_lost_share": share(lost, len(successes)),
        "healthy_assessments": count(successes, lambda r: bool(r["decisions"])),
        "false_alarms": false_alarms,
        "false_alarm_share": share(false_alarms, len(successes)),
        "false_requests": count(successes, lambda r: r["request"]),
        "false_handovers": count(successes, lambda r: r["handover"]),
        "snapshot_coverage": len(snapshots),
        "intervention_counters": sum(len(r.get("interventions", [])) for r in rows),
        "would_fail_snapshot": len(snapshot_fail),
        "rescued_snapshot": count(snapshot_fail, lambda r: r["success"]),
        "rescued_snapshot_without_person": count(
            snapshot_fail, lambda r: r["success"] and r["operator_s"] == 0
        ),
        "rescued_snapshot_with_person": count(
            snapshot_fail, lambda r: r["success"] and r["operator_s"] > 0
        ),
        "rescued_snapshot_share": share(
            count(snapshot_fail, lambda r: r["success"]), len(snapshot_fail)
        ),
        "successes_lost_snapshot": count(snapshot_good, lambda r: not r["success"]),
        "handover_would_fail_snapshot": len(handover_fail),
        "handover_rescued_snapshot": count(handover_fail, lambda r: r["success"]),
        "handover_successes_lost_snapshot": count(handover_good, lambda r: not r["success"]),
        "cohort_successes": count(rows, lambda r: r["success"]),
        "estimated_window_episodes": total,
        "estimated_window_successes": weighted(lambda r: r["success"]),
        "estimated_window_success": weighted(lambda r: r["success"]) / total if total else None,
        "estimated_window_net_gain": rescued - weighted(lambda r: r["as_run"] and not r["success"]),
        "hold_control_net": control_net,
        "net_above_hold_control": rescued
        - weighted(lambda r: r["as_run"] and not r["success"])
        - control_net,
        "actual_model_calls": sum(r.get("actual_calls", 0) for r in rows),
        "robot_active_steps": sum(r.get("robot_active_steps", 0) for r in rows),
        "retry_active_steps": sum(r.get("retry_active_steps", 0) for r in rows),
        "robot_time_s": sum(r.get("robot_time_s", 0) for r in rows),
        "estimated_window_broken": weighted(lambda r: r["as_run"] and not r["success"]),
        "handovers": len(trajectories),
        "requests": count(rows, lambda r: r["request"]),
        "operator_used_s": sum(r["operator_s"] for r in rows),
        "estimated_window_operator_s": weighted(lambda r: r["operator_s"]),
        "operator_reserved_s": sum(r["reserved_s"] for r in rows),
        "estimated_window_reserved_s": weighted(lambda r: r["reserved_s"]),
        "queue_idle_s": sum(r["queue_s"] for r in rows),
        "estimated_window_queue_s": weighted(lambda r: r["queue_s"]),
        "subscription_decision_s": sum(r["subscription_s"] for r in rows),
        "model_response_s": sum(r["model_s"] for r in rows)
        if all(r["model_s"] is not None for r in rows)
        else None,
        "physical_decision_hold_s": sum(r["decision_hold_s"] for r in rows),
        "estimated_window_subscription_s": weighted(lambda r: r["subscription_s"]),
        "estimated_window_model_s": weighted(lambda r: r["model_s"])
        if all(r["model_s"] is not None for r in rows)
        else None,
        "recovery_trajectories": {
            "count": len(trajectories),
            "placed": count(trajectories, lambda r: r["success"]),
            "placed_while_waiting": count(
                trajectories, lambda r: r["success"] and r["human"]["active_steps"] == 0
            ),
            "fumbles_drawn": count(trajectories, lambda r: r["human"]["fumble_draw"]),
            "fumbles_applied": count(trajectories, lambda r: r["human"]["fumble_applied"]),
            "by_terminal_status": dict(Counter(r["status"] for r in trajectories)),
            "by_task": dict(Counter(r.get("task", "unrecorded") for r in trajectories)),
            "by_fault_type": dict(
                Counter(kind for r in trajectories for kind in r.get("fault_types", ["unrecorded"]))
            ),
        },
    }


def frontier(points: list[dict[str, Any]], budget: float) -> dict[str, Any] | None:
    """Best attainable baseline expectation at exactly this measured time.

    Include the no-intervention policy at zero cost and zero rescues. Choose
    among *all* segments, so a dominated or nonmonotone setting cannot bias
    the comparison. No right-side extrapolation is permitted.
    """
    candidates = [
        {"estimated_window_operator_s": 0.0, "estimated_window_net_gain": 0.0, "label": "none"},
        *points,
    ]
    result: list[dict[str, Any]] = []
    for point in candidates:
        if abs(point["estimated_window_operator_s"] - budget) < 1e-8:
            result.append(
                {
                    "net": float(point["estimated_window_net_gain"]),
                    "mix": [point["label"]],
                    "fraction": 1.0,
                }
            )
    for left, right in itertools.combinations(candidates, 2):
        x, y = left["estimated_window_operator_s"], right["estimated_window_operator_s"]
        if x > y:
            left, right, x, y = right, left, y, x
        if y > x and x <= budget <= y:
            f = (budget - x) / (y - x)
            result.append(
                {
                    "net": (1 - f) * left["estimated_window_net_gain"]
                    + f * right["estimated_window_net_gain"],
                    "mix": [left["label"], right["label"]],
                    "fraction": f,
                }
            )
    return max(result, key=lambda r: r["net"]) if result else None


def resample_indices(rows: list[dict[str, Any]], resamples: int, seed: int) -> Iterator[list[int]]:
    """The same paired draws for every rule on an ordered cohort."""
    rng = np.random.default_rng(seed)
    strata = [[i for i, r in enumerate(rows) if r["as_run"] == label] for label in (False, True)]
    for _ in range(resamples):
        yield [
            int(i)
            for indices in strata
            if indices
            for i in rng.choice(indices, len(indices), replace=True)
        ]


def net_intervals(record: dict[str, Any], resamples: int, seed: int = 652) -> dict[str, Any]:
    """Intervals for each measured net, including an attached hold-only control."""
    rows = record["episodes"]
    values = np.asarray(
        [
            int(r["success"]) if not r["as_run"] else -r["weight"] * int(not r["success"])
            for r in rows
        ]
    )
    control = {
        r["episode_id"]: r["success"] for r in record.get("hold_control", {}).get("episodes", [])
    }
    holds = (
        np.asarray([r["weight"] * (int(control[r["episode_id"]]) - int(r["as_run"])) for r in rows])
        if control
        else None
    )
    nets, hold_nets = [], []
    for draw in resample_indices(rows, resamples, seed):
        nets.append(float(values[draw].sum()))
        if holds is not None:
            hold_nets.append(float(holds[draw].sum()))
    return {
        "estimated_window_net_interval95": np.quantile(nets, [0.025, 0.975]).tolist()
        if nets
        else None,
        "hold_control_net_interval95": np.quantile(hold_nets, [0.025, 0.975]).tolist()
        if hold_nets
        else None,
    }


def compare_records(
    candidate: dict[str, Any],
    baselines: list[dict[str, Any]],
    *,
    resamples: int = 1000,
    seed: int = 652,
) -> dict[str, Any]:
    """Paired attempt interval with the best simple rule reselected.

    Descriptive on one world; the cohort does not establish world transfer.
    Success-sample weights are fixed, so the interval is conditional on the
    sampled successes and does not cover another false-alarm sample.
    """
    if not baselines:
        return {
            "status": "no_simple_rules",
            "reason": "No paired simple-rule records were supplied.",
        }
    code = candidate.get("config", {}).get("code")
    if not code or not code.get("commit") or code.get("dirty") is not False:
        raise ValueError("equal-time comparison requires clean committed measured code")
    for base in baselines:
        if base.get("config", {}).get("code") != code:
            raise ValueError("equal-time comparison requires the same measured code revision")
        if (
            base["population"] != candidate["population"]
            or base["settings"] != candidate["settings"]
        ):
            raise ValueError(
                "equal-time comparison requires paired cohort and identical human settings"
            )
        if base["rule"] not in {"none", "timeout", "random", "decision"}:
            raise ValueError("comparison baseline is not a simple rule")
    records = [candidate, *baselines]
    id_lists = [[r["episode_id"] for r in rec["episodes"]] for rec in records]
    if any(ids != id_lists[0] for ids in id_lists):
        raise ValueError("comparison episode pairing differs")

    def calculate(
        sampled: list[list[dict[str, Any]]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        scores = [
            summarize(rec, sampled[i] if sampled is not None else None)
            for i, rec in enumerate(records)
        ]
        points = [
            {**s, "label": f"{b['rule']}:after={b['after']}:ratio={b['cost_ratio']}"}
            for b, s in zip(baselines, scores[1:], strict=True)
        ]
        return scores[0], frontier(points, scores[0]["estimated_window_operator_s"])

    score, best = calculate()
    if best is None:
        return {
            "status": "outside_measured_baseline_time",
            "budget_s": score["estimated_window_operator_s"],
            "reason": "Supervisor operator time exceeds the largest measured simple-rule point; "
            "no extrapolation is available.",
        }
    estimate = score["estimated_window_net_gain"] / best["net"] - 1 if best["net"] > 0 else None
    robots = sorted({r["robot"] for r in candidate["episodes"]})
    gains, differences = [], []
    candidate_nets, baseline_nets, control_nets = [], [], []
    for draw in resample_indices(candidate["episodes"], resamples, seed):
        sampled = [[rec["episodes"][i] for i in draw] for rec in records]
        c, b = calculate(sampled)
        control_nets.append(c["hold_control_net"])
        if b is not None:
            candidate_nets.append(c["estimated_window_net_gain"])
            baseline_nets.append(b["net"])
            differences.append(c["estimated_window_net_gain"] - b["net"])
            if b["net"] > 0:
                gains.append(c["estimated_window_net_gain"] / b["net"] - 1)
    interval = np.quantile(gains, [0.025, 0.975]).tolist() if gains else None
    return {
        "status": "measured" if estimate is not None else "nonpositive_baseline_net",
        "reason": None
        if estimate is not None
        else "The best simple-rule net is zero or negative; 20% more is undefined. "
        "Both nets are reported.",
        "latency_route": candidate.get("latency_route", "subscription"),
        "measured_code": code,
        "operator_axis": "estimated whole-window consumed operator seconds",
        "budget_s": score["estimated_window_operator_s"],
        "candidate_rescues": score["rescued_as_run"],
        "candidate_broken": score["estimated_window_broken"],
        "candidate_net": score["estimated_window_net_gain"],
        "metric": "window rescued minus broken; hold control shown separately",
        "best_simple_rule": best,
        "relative_gain": estimate,
        "relative_gain_interval95": interval,
        "net_difference": score["estimated_window_net_gain"] - best["net"],
        "candidate_net_interval95": np.quantile(candidate_nets, [0.025, 0.975]).tolist()
        if candidate_nets
        else None,
        "baseline_net_interval95": np.quantile(baseline_nets, [0.025, 0.975]).tolist()
        if baseline_nets
        else None,
        "hold_control_net_interval95": np.quantile(control_nets, [0.025, 0.975]).tolist()
        if control_nets
        else None,
        "candidate_control_zero": score["hold_control_net"],
        "candidate_net_above_control": score["net_above_hold_control"],
        "difference_interval95": np.quantile(differences, [0.025, 0.975]).tolist()
        if differences
        else None,
        "point_meets_20_percent": (estimate >= 0.2 or math.isclose(estimate, 0.2, abs_tol=1e-12))
        if estimate is not None
        else None,
        "lower_interval_meets_20_percent": interval[0] >= 0.2 if interval else None,
        "interval": {
            "method": "paired attempt bootstrap stratified by as-run outcome; "
            "best baseline reselected",
            "robots": len(robots),
            "resamples": resamples,
            "valid_relative": len(gains),
            "valid_difference": len(differences),
            "seed": seed,
            "scope": "conditional on world/success sample; unsupported time resamples omitted",
        },
    }


def score_files(paths: list[Path], out: Path, resamples: int = 1000) -> dict[str, Any]:
    records = [json.loads(p.read_text()) for p in paths]
    result: dict[str, Any] = {
        "version": 1,
        "records": [str(p.resolve()) for p in paths],
        "rows": [{**summarize(r), **net_intervals(r, resamples)} for r in records],
        "comparisons": [],
    }
    for rec in records:
        if rec["rule"] == "supervisor":
            bases = [
                b
                for b in records
                if b["rule"] in {"none", "timeout", "random", "decision"}
                and b["population"] == rec["population"]
                and b["settings"] == rec["settings"]
            ]
            result["comparisons"].append(
                {
                    "rule": rec["rule"],
                    "ratio": rec["cost_ratio"],
                    "operators": rec["settings"]["operators"],
                    **compare_records(rec, bases, resamples=resamples),
                }
            )
    dump(out / "table.json", result)
    (out / "table.md").write_text(markdown(result))
    return result


def markdown(result: dict[str, Any]) -> str:
    def row(cells: list[Any]) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    lines = [
        "# Claim 4: stored live passes",
        "",
        "Rescues use all as-run failures. Losses use sampled successes. Window success and "
        "operator seconds expand successes by inverse inclusion probability. Human settings "
        "are assumptions; world and furniture-shift recoveries are provisional. Model time "
        "is a production estimate. Physical holds use the selected latency route.",
        "",
        row(
            [
                "World",
                "Rule",
                "Latency route",
                "Operators",
                "Ratio / timeout step",
                "Rescued without / with person",
                "Would fail",
                "Lost / sampled successes",
                "Window net [95% interval] / hold-only net [95% interval]",
                "False alarms / sampled successes",
                "Estimated window success",
                "Cohort / window operator s",
                "Queue s",
                "Subscription / model / hold s",
            ]
        ),
        row(["---"] * 14),
    ]
    for r in result["rows"]:
        lines.append(
            row(
                [
                    r["world"],
                    r["rule"],
                    r["latency_route"],
                    r["operators"],
                    f"{r['cost_ratio']:g} / {r['after']}",
                    f"{r['rescued_without_person']} / {r['rescued_with_person']}",
                    r["would_fail_as_run"],
                    f"{r['successes_lost']} / {r['sampled_as_run_successes']}",
                    f"{r['estimated_window_net_gain']:.2f} "
                    f"{r.get('estimated_window_net_interval95')} / "
                    f"{r.get('hold_control_net', 'unmeasured')} "
                    f"{r.get('hold_control_net_interval95')}",
                    f"{r['false_alarms']} / {r['sampled_as_run_successes']}",
                    f"{r['estimated_window_success']:.3%}",
                    f"{r['operator_used_s']:.2f} / {r['estimated_window_operator_s']:.2f}",
                    f"{r['queue_idle_s']:.2f}",
                    f"{r['subscription_decision_s']:.2f} / {r['model_response_s']} / "
                    f"{r['physical_decision_hold_s']:.2f}",
                ]
            )
        )
    lines += [
        "",
        "## Counterfactual at the first intervention",
        "",
        "Counters continue policy after the decision hold, before the intervention. New records "
        "use the first applied intervention; earlier records cover hand-over requests only. "
        "Per-action counters are retained. Both readings are reported. Operator state is "
        "sampled when the trajectory ends: `running` can accompany a successful placement "
        "detected by the simulator before the operator script finishes. Recovery `placed` "
        "counts in JSON report the actual outcome.",
        "",
        row(
            [
                "Rule",
                "Operators",
                "Snapshot coverage",
                "Would fail",
                "Rescued",
                "Successes lost",
                "Recovery trajectories",
                "Operator state at end",
            ]
        ),
        row(["---"] * 8),
    ]
    for r in result["rows"]:
        recovery = r["recovery_trajectories"]
        lines.append(
            row(
                [
                    r["rule"],
                    r["operators"],
                    r["snapshot_coverage"],
                    r["would_fail_snapshot"],
                    r["rescued_snapshot"],
                    r["successes_lost_snapshot"],
                    recovery["count"],
                    json.dumps(recovery["by_terminal_status"], sort_keys=True),
                ]
            )
        )
    lines += [
        "",
        "## Equal consumed operator time",
        "",
        "Segments mix whole-window policies in expectation, within the measured range. "
        "Comparisons require the same clean measured code revision. "
        "Intervals resample paired attempts within as-run outcome strata "
        "and reselect the best baseline. They "
        "condition on this world and success sample; world transfer is unmeasured.",
        "",
    ]
    for comparison in result["comparisons"]:
        if comparison.get("reason"):
            lines.append(f"- {comparison['reason']}")
        lines += [f"- {json.dumps(comparison, sort_keys=True)}"]
    return "\n".join(lines) + "\n"
