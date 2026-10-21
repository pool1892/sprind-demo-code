"""Frozen pass marks, accepted 2026-10-08."""

from types import MappingProxyType
from typing import Any

from fleetlab.scoring.comparisons import chance_net, chance_points, required_values
from fleetlab.scoring.models import ScoreReport

PASS_MARKS = MappingProxyType(
    {
        "claim1.recall": 0.8,
        "claim1.decoy_rejection": 1.0,
        "claim2.spearman": 0.7,
        "claim3.share_k3": 0.5,
        "claim3.margin_points": 3.0,
        "claim3.above_chance": 0.0,
        "claim4.relative_rescue_improvement": 0.2,
    }
)


def pass_marks(report: ScoreReport) -> list[dict[str, Any]]:
    at = next(b for b in report.claim3.budgets if b.label == "k = 3")
    baseline_values, absent = required_values(at)
    best = max((v.gain.net for v in baseline_values.values() if v.gain is not None), default=None)
    agent = at.agent.gain.net if at.agent.gain is not None else None
    nets_complete = not absent and all(v.gain is not None for v in baseline_values.values())
    margin_net = agent - best if nets_complete and agent is not None and best is not None else None
    n = report.scoring_episodes
    margin = 100 * margin_net / n if margin_net is not None and n > 0 else None
    chance_metric = chance_points(at)
    chance = chance_metric.value
    tail = chance_net(at)
    above_net = agent - tail if agent is not None and tail is not None else None
    above = float(100 * above_net / n) if above_net is not None and n > 0 else None
    values = {
        "claim1.recall": report.claim1.recall.value,
        "claim1.decoy_rejection": report.claim1.decoy_rejection.value,
        "claim2.spearman": report.claim2.spearman.value,
        "claim3.share_k3": at.agent.share.value,
        "claim3.margin_points": margin,
        "claim3.above_chance": above,
        "claim4.relative_rescue_improvement": None,
    }
    out = []
    for name, threshold in PASS_MARKS.items():
        value = values[name]
        strict = name == "claim3.above_chance"
        status = (
            "unavailable"
            if value is None
            else ("met" if (value > threshold if strict else value >= threshold) else "missed")
        )
        reason = None
        if name == "claim2.spearman" and not report.claim2.complete:
            status = "unavailable"
            reason = (
                report.claim2.missing
                or "rank correlation is partial: ranked actions lack replay outcomes"
            )
        if name == "claim3.margin_points" and margin_net is not None and n > 0:
            status = "met" if 100 * margin_net >= 3 * n else "missed"
        if name == "claim3.above_chance" and above_net is not None:
            status = "met" if above_net > 0 else "missed"
        if name == "claim3.margin_points" and absent:
            reason = "required baselines missing: " + ", ".join(absent)
        if name == "claim3.above_chance" and chance is None:
            reason = chance_metric.missing
        if value is None:
            reason = reason or "required metric or comparison unavailable"
        if name.startswith("claim3.") and at.valuation_missing:
            value = None
            status = "unavailable"
            reason = at.valuation_missing
        out.append(
            {
                "metric": name,
                "threshold": threshold,
                "comparison": ">" if strict else ">=",
                "value": value,
                "status": status,
                "missing": reason,
                "provisional": name == "claim1.recall" and report.claim1.recall.provisional,
                "truth_disagreements": report.claim1.truth_disagreements
                if name == "claim1.recall"
                else [],
            }
        )
    return out
