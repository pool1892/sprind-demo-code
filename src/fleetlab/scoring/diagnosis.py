"""Claim 1: the agent's diagnosis against the hidden fault table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fleetlab.contracts.faults import Target
from fleetlab.contracts.tools import Cause, Diagnosis
from fleetlab.oracle.world import Cause as OracleCause
from fleetlab.scoring.models import (
    CauseVerdict,
    ConfounderResolution,
    DiagnosisScore,
    FaultFinding,
    Metric,
    Verdict,
)
from fleetlab.worlds.truth import FaultTable, FaultTruth

# A cause with at least this confidence is named, whatever its rank.
NAMED_CONFIDENCE = 0.5
# Causes within the top n + NAMED_EXTRA by confidence are named (n: detectable faults).
NAMED_EXTRA = 2

# The verdicts that count as false positives in the precision.
FALSE_POSITIVES: frozenset[Verdict] = frozenset({"decoy", "inert", "no fault"})


def target_matches(cause: Target, fault: Target) -> bool:
    """Same scope and the same component, configuration key or condition key."""
    return (
        cause.kind is not None
        and cause.scope == fault.scope
        and cause.kind is fault.kind
        and cause.thing == fault.thing
    )


def is_detectable(fault: FaultTruth) -> bool:
    """A detectable cause: an effective fault the table marks detectable."""
    return fault.role == "effective" and fault.detectable


def is_small(fault: FaultTruth) -> bool:
    """An effective fault that misses the detectability rule (scored as neither)."""
    return fault.role == "effective" and not fault.detectable


def named_flags(causes: Sequence[Cause], n: int) -> list[bool]:
    """Which causes are named: confidence at least 0.5, or within the top n + 2 by
    confidence, ties broken by the order of the list."""
    order = sorted(range(len(causes)), key=lambda i: (-causes[i].confidence, i))
    top = set(order[: n + NAMED_EXTRA])
    return [c.confidence >= NAMED_CONFIDENCE or i in top for i, c in enumerate(causes)]


def cause_verdict(cause: Cause, faults: Sequence[FaultTruth]) -> tuple[Verdict, list[str]]:
    """The verdict on one cause and the faults its target matches.

    With several matching faults the first applicable verdict wins, in the order
    hit, wrong direction, small, decoy, inert.
    """
    matched = [f for f in faults if target_matches(cause.target, f.target)]
    ids = [f.fault_id for f in matched]
    if not matched:
        if cause.target.kind is None and any(
            f.target.scope == cause.target.scope
            and f.role == "effective"
            and any(abs(points) > 0 for points in f.expected_loss_points.values())
            for f in faults
        ):
            return "vague", ids
        return "no fault", ids
    detectable = [f for f in matched if is_detectable(f)]
    if any(cause.direction in f.directions for f in detectable):
        return "hit", ids
    if detectable:
        return "wrong direction", ids
    if any(is_small(f) for f in matched):
        return "small", ids
    if any(f.role == "decoy" for f in matched):
        return "decoy", ids
    return "inert", ids


def score_diagnosis(
    diagnosis: Diagnosis | None,
    table: FaultTable,
    confounders: Sequence[tuple[str, str]] | None = None,
    cause_units: Sequence[OracleCause] = (),
) -> DiagnosisScore:
    """Recall, precision, decoy rejection and confounder resolution of a diagnosis."""
    faults = table.faults
    n = (
        sum(c.detectable for c in cause_units)
        if cause_units
        else sum(is_detectable(f) for f in faults)
    )
    missing = diagnosis is None
    if missing:
        diagnosis = Diagnosis(causes=[], mode_summaries=[], warnings=[], retrain=[])
    assert diagnosis is not None
    causes = diagnosis.causes
    # Deduplicate before the confidence cut. Highest confidence wins, with list
    # order resolving ties (including conflicting duplicate directions).
    unique: dict[str, int] = {}
    for i, cause in enumerate(causes):
        key = cause.target.model_dump_json()
        if key not in unique or cause.confidence > causes[unique[key]].confidence:
            unique[key] = i
    indices = sorted(unique.values())
    selected = named_flags([causes[i] for i in indices], n)
    named = [False] * len(causes)
    for i, flag in zip(indices, selected, strict=True):
        named[i] = flag
    verdicts: list[CauseVerdict] = []
    for i, (cause, is_named) in enumerate(zip(causes, named, strict=True)):
        verdict, ids = cause_verdict(cause, faults)
        verdicts.append(
            CauseVerdict(
                index=i,
                target=cause.target,
                direction=cause.direction,
                confidence=cause.confidence,
                named=is_named,
                verdict=verdict,
                faults=ids,
            )
        )
    named_verdicts = [v for v in verdicts if v.named]

    findings: list[FaultFinding] = []
    for fault in faults:
        by = [v.index for v in named_verdicts if fault.fault_id in v.faults]
        accepted = [i for i in by if causes[i].direction in fault.directions]
        findings.append(
            FaultFinding(
                fault_id=fault.fault_id,
                kind=fault.kind,
                role=fault.role,
                target=fault.target,
                detectable=is_detectable(fault),
                small=is_small(fault),
                directions=fault.directions,
                found=is_detectable(fault) and bool(accepted),
                named_by=by,
                wrong_direction_by=[i for i in by if i not in accepted],
            )
        )

    found = sum(f.found for f in findings)
    units: list[dict[str, Any]] = []
    if cause_units:
        by_id = {f.fault_id: f for f in findings}
        for unit in cause_units:
            members = [by_id[f] for f in unit.fault_ids if f in by_id]
            unit_found = all(
                any(
                    v.named and f.fault_id in v.faults and v.direction in f.directions
                    for v in verdicts
                )
                for f in members
            ) and len(members) == len(unit.fault_ids)
            units.append(
                {
                    "unit": unit.unit,
                    "fault_ids": list(unit.fault_ids),
                    "detectable": unit.detectable,
                    "qualified_by": list(unit.qualified_by),
                    "found": unit.detectable and unit_found,
                }
            )
        found = sum(u["found"] for u in units)
        detectable_ids = {
            f for u in units if u["detectable"] and u["found"] for f in u["fault_ids"]
        }
        eligible_ids = {f for u in units if u["detectable"] for f in u["fault_ids"]}
        for named_verdict in named_verdicts:
            if named_verdict.verdict in ("small", "hit", "wrong direction"):
                matched = [f for f in faults if f.fault_id in named_verdict.faults]
                named_verdict.verdict = "small"
                if any(f.fault_id in eligible_ids for f in matched) and not any(
                    f.fault_id in eligible_ids and named_verdict.direction in f.directions
                    for f in matched
                ):
                    named_verdict.verdict = "wrong direction"
                if any(
                    f.fault_id in detectable_ids and named_verdict.direction in f.directions
                    for f in matched
                ):
                    named_verdict.verdict = "hit"
    recall = Metric.of(found / n) if n else Metric.absent("the world has no detectable fault")
    hits = sum(v.verdict == "hit" for v in named_verdicts)
    vague = sum(v.verdict == "vague" for v in named_verdicts)
    ordinary_fp = sum(v.verdict in FALSE_POSITIVES for v in named_verdicts)
    false_positives = ordinary_fp + vague
    precision = (
        Metric.of(hits / (hits + false_positives))
        if hits + false_positives
        else Metric.absent("no named cause is a hit or a false positive")
    )
    decoys = [f for f in findings if f.role == "decoy"]
    decoy_rejection = (
        Metric.of(0 if missing else sum(not f.named_by for f in decoys) / len(decoys))
        if decoys
        else Metric.absent("the world has no decoy")
    )

    by_id = {f.fault_id: f for f in findings}
    resolutions: list[ConfounderResolution] = []
    if confounders is not None:
        note = (
            "pairs from the world's spec; the effective fault counts when a named cause matches it"
        )
        for effective, inert in confounders:
            inert_named = bool(by_id[inert].named_by) if inert in by_id else False
            eff = by_id.get(effective)
            effective_found = None if eff is None else bool(eff.named_by)
            resolutions.append(
                ConfounderResolution(
                    inert=inert,
                    inert_named=inert_named,
                    effective=effective,
                    effective_found=effective_found,
                    resolved=None
                    if effective_found is None
                    else effective_found and not inert_named,
                )
            )
    else:
        note = (
            "the fault table does not carry the confounder pairs, so only whether each inert "
            "entry was named is reported, not whether its effective partner was"
        )
        for f in findings:
            if f.role == "inert":
                resolutions.append(
                    ConfounderResolution(
                        inert=f.fault_id,
                        inert_named=bool(f.named_by),
                        effective=None,
                        effective_found=None,
                        resolved=None,
                    )
                )

    disagreement: set[str] = set()
    if cause_units:
        individual = {f.fault_id for f in faults if is_detectable(f)}
        single = {u.fault_ids[0] for u in cause_units if u.detectable and len(u.fault_ids) == 1}
        disagreement.update(individual ^ single)
        for unit in cause_units:
            if unit.detectable and len(unit.fault_ids) > 1:
                disagreement.update(individual & set(unit.fault_ids))
            disagreement.update(set(unit.fault_ids) - {f.fault_id for f in faults})
    legacy = (
        Metric.of(hits / (hits + ordinary_fp))
        if hits + ordinary_fp
        else Metric.absent("no precision denominator")
    )
    for metric in (recall, precision, legacy):
        metric.provisional = bool(disagreement)
    return DiagnosisScore(
        missing="no diagnosis submitted" if missing else None,
        truth_disagreements=sorted(disagreement),
        precision_with_vague=precision,
        precision_without_vague=legacy,
        targetless=sum(v.target.kind is None for v in named_verdicts),
        detectable=n,
        named_cut=n + NAMED_EXTRA,
        named=len(named_verdicts),
        recall=recall,
        precision=precision,
        decoy_rejection=decoy_rejection,
        hits=hits,
        false_positives=false_positives,
        wrong_direction=sum(v.verdict == "wrong direction" for v in named_verdicts),
        small=sum(v.verdict == "small" for v in named_verdicts),
        vague=sum(v.verdict == "vague" for v in named_verdicts),
        causes=verdicts,
        faults=findings,
        confounders=resolutions,
        confounder_note=note,
        units=units,
    )
