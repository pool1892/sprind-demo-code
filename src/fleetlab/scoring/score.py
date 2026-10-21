"""The scorer's entry points: `score` and `replays_needed_for`."""

from __future__ import annotations

import re
from collections.abc import Iterable

from fleetlab.contracts.actions import SCORING_BUDGETS, Action, FixSet, cut
from fleetlab.contracts.oracle import ReplayNeed, fault_list_hash, replays_needed
from fleetlab.scoring.decisions import NeedsLog, budget_label, headline, score_decisions
from fleetlab.scoring.errors import error_account
from fleetlab.scoring.foresight import score_foresight
from fleetlab.scoring.measured import score_claim1
from fleetlab.scoring.models import (
    TOP_K,
    Metric,
    RankingScore,
    ReplayNeedRecord,
    ScoreReport,
    ScoringInputs,
)
from fleetlab.scoring.ranking import score_ranking, single
from fleetlab.scoring.thresholds import pass_marks
from fleetlab.scoring.trace import score_traceability
from fleetlab.scoring.valuation import removal_options


def _asserts_interaction(text: str) -> bool:
    # Cause has one target, so joint assertions only exist in its prose.
    lowered = text.lower()
    return bool(re.search(r"\binteraction\b", lowered)) and not bool(
        re.search(
            r"\b(?:no|not|without|ruled out)(?: an?| any| evidence of)? interaction\b",
            lowered,
        )
    )


def _records(needs: NeedsLog) -> list[ReplayNeedRecord]:
    """The replays, each once, with every item that waits for it."""
    by_key: dict[tuple[str, str], ReplayNeedRecord] = {}
    for label, items in needs:
        for need in items:
            key = (need.episode_id, fault_list_hash(need.faults))
            record = by_key.get(key)
            if record is None:
                record = ReplayNeedRecord(
                    episode_id=need.episode_id,
                    fault_ids=sorted(f.fault_id for f in need.faults),
                    faults=list(need.faults),
                    needed_for=[],
                )
                by_key[key] = record
            if label not in record.needed_for:
                record.needed_for.append(label)
    return sorted(by_key.values(), key=lambda r: (r.episode_id, r.fault_ids))


def score(inputs: ScoringInputs) -> ScoreReport:
    """Score claims 1, 2, 3 and 5, the headline, frozen marks and traceability.

    Never raises for a gap in the oracle: what cannot be valued is reported with
    its reason, and the replays it waits for are listed in `needs_fresh_replay`.
    """
    claim1, curve = score_claim1(inputs)
    claim2, ranking_needs = score_ranking(
        inputs.fix_set, inputs.oracle, inputs.truth, inputs.current
    )
    if inputs.replay_missing:
        absent = Metric.absent(inputs.replay_missing)
        claim2 = RankingScore(
            missing=inputs.replay_missing,
            spearman=absent,
            mae_points=absent,
            slope=absent,
            coverage=absent,
        )
        ranking_needs = []
    claim3, decision_needs, notes = score_decisions(inputs)
    if inputs.fix_set is not None or inputs.decision_actions is not None:
        for k in TOP_K:
            at = next(
                (s for s in claim3.budgets if s.budget.kind == "count" and s.budget.amount == k),
                None,
            )
            claim2.top_k_share[str(k)] = (
                at.agent.share if at is not None else Metric.absent(f"k = {k} is not a budget")
            )
        no_replay = [a.rank for a in claim2.actions if not a.action.has_replay]
        if no_replay:
            notes.append(
                f"ranked actions {no_replay} have no replay (claim 5) and are not valued under "
                "claim 2; under claim 3 they take a place in the cut and change nothing"
            )
    for fault in inputs.fault_table.faults:
        if fault.detectable and fault.role != "effective":
            notes.append(
                f"curve audit: fault {fault.fault_id} is marked detectable but its planted "
                f"role is {fault.role}; curve membership is descriptive only"
            )
    foresight = score_foresight(inputs)
    healthy_needs = [
        (f"claim 5: retraining indicator {n}", [ReplayNeed(e, ()) for e in row["missing_examples"]])
        for n, row in enumerate(foresight["retrain"])
    ]
    report = ScoreReport(
        world_id=inputs.fault_table.world_id,
        scoring_episodes=len(inputs.oracle.identities),
        seed=inputs.seed,
        draws=inputs.draws,
        claim1=claim1,
        claim1_curve=curve,
        measured_causes={}
        if inputs.measured_causes is None
        else inputs.measured_causes.model_dump(mode="json"),
        claim2=claim2,
        claim3=claim3,
        headline=headline(claim3),
        traceability=score_traceability(inputs.diagnosis, inputs.fix_set, inputs.tool_log),
        needs_fresh_replay=_records([*ranking_needs, *decision_needs, *healthy_needs]),
        notes=notes,
        claim5=foresight,
    )
    if claim1.truth_disagreements:
        report.notes.insert(
            0,
            "PROVISIONAL recall and precision: cause units disagree with "
            "per-fault detectable flags: " + ", ".join(claim1.truth_disagreements),
        )
    report.pass_marks = pass_marks(report)
    report.errors = error_account(report)
    if inputs.diagnosis is not None:
        for pair in inputs.additive_pairs:
            members = [f for f in inputs.fault_table.faults if f.fault_id in pair]
            claims = [
                c
                for c in report.claim1.causes
                if c.named
                and set(c.faults) & set(pair)
                and _asserts_interaction(inputs.diagnosis.causes[c.index].failure_mode)
            ]
            for claim in claims:
                report.remarks.append(
                    {
                        **claim.model_dump(mode="json"),
                        "matched": "an additive pair reported as an interaction",
                        "inferred_from_prose": True,
                        "pair": list(pair),
                        "targets": [f.target.model_dump(mode="json") for f in members],
                    }
                )
    return report


def _needed(fix_set: FixSet, inputs: ScoringInputs) -> list[ReplayNeed]:
    try:
        return list(replays_needed(fix_set, inputs.oracle, inputs.truth, inputs.current))
    except (ValueError, KeyError):
        return []  # cannot be resolved at all; `score` reports why


def _cuts(ranked: list[Action] | None, inputs: ScoringInputs) -> Iterable[tuple[str, FixSet]]:
    if ranked is None:
        return
    for budget in SCORING_BUDGETS:
        try:
            yield budget_label(budget), cut(ranked, budget, inputs.costs)
        except (ValueError, KeyError):
            continue


def replays_needed_for(inputs: ScoringInputs) -> list[ReplayNeed]:
    """Every replay the full scoring needs that the oracle table lacks, each once.

    The agent's ranked actions one by one, its fix sets at every budget, the
    no-data baseline's fix sets, and the achievable gains. Random draws that
    would need a fresh replay are skipped by `score` and counted there, not
    requested here.
    """
    needs: NeedsLog = []
    if inputs.fix_set is not None:
        ranked = [r.action for r in inputs.fix_set.actions]
        for rank, action in enumerate(ranked, start=1):
            if action.has_replay:
                needs.append((f"claim 2: ranked action {rank}", _needed(single(action), inputs)))
        for label, fix_set in _cuts(ranked, inputs):
            needs.append((f"claim 3: agent at {label}", _needed(fix_set, inputs)))
    for label, fix_set in _cuts(inputs.no_data_ranked, inputs):
        needs.append((f"claim 3: no-data ranker at {label}", _needed(fix_set, inputs)))
    for n, row in enumerate(score_foresight(inputs)["retrain"]):
        needs.append(
            (
                f"claim 5: retraining indicator {n}",
                [ReplayNeed(e, ()) for e in row["missing_examples"]],
            )
        )
    needs.append(("achievable gain", removal_options(inputs.oracle)[1]))
    return [ReplayNeed(r.episode_id, tuple(r.faults)) for r in _records(needs)]
