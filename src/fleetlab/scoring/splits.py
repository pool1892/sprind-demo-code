"""Known/unseen breakdown without assigning an interaction's gain to one member."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fleetlab.contracts.actions import resolve_all
from fleetlab.scoring.decisions import score_decisions
from fleetlab.scoring.measured import score_claim1
from fleetlab.scoring.models import ScoringInputs
from fleetlab.scoring.ranking import score_ranking, slope, spearman


def score_splits(inputs: ScoringInputs, unseen: set[str]) -> dict[str, Any]:
    diagnosis, _ = score_claim1(inputs)
    types = {f.fault_id: f.kind for f in inputs.fault_table.faults}
    known_ids = {f for f, kind in types.items() if kind not in unseen}
    unseen_ids = set(types) - known_ids
    out: dict[str, Any] = {"unseen_type_names": sorted(unseen), "diagnosis": {}}
    units = diagnosis.units or [
        {
            "unit": f.fault_id,
            "fault_ids": [f.fault_id],
            "detectable": f.detectable,
            "found": f.found,
        }
        for f in diagnosis.faults
    ]
    for label, ids in (("known", known_ids), ("unseen", unseen_ids)):
        subset = [u for u in units if u["detectable"] and set(u["fault_ids"]) <= ids]
        named = [c for c in diagnosis.causes if c.named and c.faults and set(c.faults) <= ids]
        hits = sum(c.verdict == "hit" for c in named)
        fp = sum(c.verdict in ("decoy", "inert", "no fault") for c in named)
        decoys = [f for f in diagnosis.faults if f.fault_id in ids and f.role == "decoy"]
        confounders = [c for c in diagnosis.confounders if c.effective in ids and c.inert in ids]
        out["diagnosis"][label] = {
            "missing": diagnosis.missing if diagnosis.recall.value is None else None,
            "detectable": len(subset),
            "found": sum(u["found"] for u in subset),
            "recall": sum(u["found"] for u in subset) / len(subset) if subset else None,
            "precision": hits / (hits + fp) if hits + fp else None,
            "hits": hits,
            "false_positives": fp,
            "small": sum(c.verdict == "small" for c in named),
            "decoy_rejection": (
                0 if inputs.diagnosis is None else sum(not f.named_by for f in decoys) / len(decoys)
            )
            if decoys
            else None,
            "confounder_resolution": sum(c.resolved is True for c in confounders) / len(confounders)
            if confounders and all(c.resolved is not None for c in confounders)
            else None,
        }
    out["diagnosis"]["mixed_units"] = [
        u for u in units if set(u["fault_ids"]) & known_ids and set(u["fault_ids"]) & unseen_ids
    ]
    out["diagnosis"]["unmatched_false_positives"] = [
        c.index for c in diagnosis.causes if c.named and c.verdict in ("no fault", "vague")
    ]
    if inputs.replay_missing:
        out["valuation_missing"] = inputs.replay_missing
        out["ranking"] = {}
        out["decision_cohorts"] = {}
        return out
    actions, _ = score_ranking(inputs.fix_set, inputs.oracle, inputs.truth, inputs.current)
    classified: dict[str, list[int]] = {
        "known": [],
        "unseen": [],
        "mixed": [],
        "no_active_fault": [],
    }
    for row in actions.actions:
        touched: set[str] = set()
        try:
            for identity in inputs.oracle.identities:
                resolution = resolve_all(
                    [row.action],
                    identity,
                    identity.faults,
                    inputs.truth,
                    None if inputs.current is None else inputs.current(identity),
                )
                touched.update(resolution.removed)
        except (ValueError, KeyError):
            continue
        known, novel = bool(touched & known_ids), bool(touched & unseen_ids)
        label = (
            "mixed"
            if known and novel
            else "known"
            if known
            else "unseen"
            if novel
            else "no_active_fault"
        )
        classified[label].append(row.rank)
    out["action_ranks"] = classified
    out["ranking"] = {}
    for label, ranks in classified.items():
        ranked_subset = [
            a for a in actions.actions if a.rank in ranks and a.true_points is not None
        ]
        expected = [a.expected_gain for a in ranked_subset]
        actual = [float(a.true_points or 0) for a in ranked_subset]
        out["ranking"][label] = {
            "ranked": len(ranks),
            "valued": len(ranked_subset),
            "spearman": spearman(expected, actual),
            "mae_points": sum(abs(e - t) for e, t in zip(expected, actual, strict=True))
            / len(ranked_subset)
            if ranked_subset
            else None,
            "slope": slope(expected, actual),
        }
    cohorts: dict[str, list[Any]] = {
        "known_only": [],
        "unseen_including_mixed": [],
        "no_effective_fault": [],
    }
    for identity in inputs.oracle.identities:
        active = identity.fault_ids - inputs.oracle.excluded
        label = (
            "unseen_including_mixed"
            if active & unseen_ids
            else "known_only"
            if active
            else "no_effective_fault"
        )
        cohorts[label].append(identity)
    out["decision_cohorts"] = {}
    for label, identities in cohorts.items():
        if not identities:
            out["decision_cohorts"][label] = {"episodes": 0, "budgets": []}
            continue
        scored, _, _ = score_decisions(
            replace(inputs, oracle=inputs.oracle.restricted(identities), catalogue=[], draws=0)
        )
        rows = []
        for budget in scored.budgets:
            methods = {"agent": budget.agent, **budget.baselines}
            if inputs.no_data_ranked is not None:
                methods["no-data ranker"] = budget.no_data
            rows.append(
                {
                    "budget": budget.label,
                    "achievable_net": None
                    if budget.achievable.gain is None
                    else budget.achievable.gain.net,
                    "methods": {
                        name: {
                            "gain": None if value.gain is None else value.gain.model_dump(),
                            "share": value.share.model_dump(),
                            "world_points": None
                            if value.gain is None
                            else 100 * value.gain.net / len(inputs.oracle.identities),
                        }
                        for name, value in methods.items()
                    },
                }
            )
        out["decision_cohorts"][label] = {"episodes": len(identities), "budgets": rows}
    return out
