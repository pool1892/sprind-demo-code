"""Claim 1's measured membership; curve diagnoses are descriptive only."""

from pathlib import Path
from typing import Any

from fleetlab.oracle.causes import CLASS_RULE, MeasuredCauses
from fleetlab.oracle.world import Cause as OracleCause
from fleetlab.scoring.diagnosis import score_diagnosis
from fleetlab.scoring.models import DiagnosisScore, Metric, ScoringInputs
from fleetlab.worlds.truth import DETECT_EPISODES, DETECT_POINTS, FaultTable


def read_measured(path: Path, world_id: str, world_hash: str | None) -> MeasuredCauses | None:
    """Read the oracle's public schema, with the world's real identity/hash."""
    if not path.is_file():
        return None
    record = MeasuredCauses.model_validate_json(path.read_text())
    if record.world_id != world_id or record.world_hash != world_hash:
        raise ValueError("measured causes belong to another world")
    if (
        record.rule != CLASS_RULE
        or record.threshold_points != DETECT_POINTS
        or record.min_observed_carriers != DETECT_EPISODES
    ):
        raise ValueError("measured cause rule differs from the frozen design")
    return record


def measured_table(
    record: MeasuredCauses, table: FaultTable
) -> tuple[FaultTable, tuple[OracleCause, ...]]:
    if record.world_id != table.world_id:
        raise ValueError("measured causes belong to another world")
    if record.cause_count != len(record.causes):
        raise ValueError("measured cause_count differs from its cause list")
    items = {item.item: item for item in record.faults}
    if set(items) != {fault.fault_id for fault in table.faults}:
        raise ValueError("measured faults differ from the world's fault table")
    raw_units: list[dict[str, Any]] = record.model_dump()["causes"]
    for unit in raw_units:
        if (
            not isinstance(unit.get("fault_ids"), list)
            or not unit["fault_ids"]
            or not all(isinstance(f, str) for f in unit["fault_ids"])
        ):
            raise ValueError("measured cause has invalid fault_ids")
    singles = {unit["fault_ids"][0] for unit in raw_units if len(unit["fault_ids"]) == 1}
    if singles != {item.item for item in record.faults if item.measured_class == "cause"}:
        raise ValueError("measured singleton list differs from its classifications")
    faults = []
    for fault in table.faults:
        item = items[fault.fault_id]
        if item.measured_class is None:
            raise ValueError("complete measured list has an unclassified fault")
        role = "effective" if item.measured_class in ("cause", "small") else item.measured_class
        faults.append(
            fault.model_copy(
                update={
                    "role": role,
                    "detectable": fault.fault_id in singles,
                    "expected_loss_points": {
                        name: window.points or 0 for name, window in item.windows.items()
                    },
                }
            )
        )
    units = tuple(
        OracleCause(
            unit=str(unit["unit"]),
            fault_ids=tuple(str(f) for f in unit["fault_ids"]),
            roles=("effective",),
            detectable=True,
            qualified_by=tuple(str(w) for w in unit["qualified_by"]),
            expected_points=None,
            counts=True,
        )
        for unit in raw_units
    )
    if any(set(unit.fault_ids) - set(items) for unit in units):
        raise ValueError("measured cause names a fault outside the world")
    return table.model_copy(update={"faults": faults}), units


def score_claim1(inputs: ScoringInputs) -> tuple[DiagnosisScore, DiagnosisScore]:
    curve = score_diagnosis(
        inputs.diagnosis, inputs.fault_table, inputs.confounders, inputs.cause_units
    )
    record = inputs.measured_causes
    if record is None or not record.complete or record.problems:
        reason = inputs.measured_missing
        if record is not None:
            reason = (
                "oracle measured cause list incomplete: "
                + "; ".join(record.problems)
                + "; "
                + reason
            )
        absent = Metric.absent(reason)
        return DiagnosisScore(
            missing=reason,
            recall=absent,
            precision=absent,
            precision_with_vague=absent,
            precision_without_vague=absent,
            decoy_rejection=absent,
        ), curve
    table, units = measured_table(record, inputs.fault_table)
    return score_diagnosis(inputs.diagnosis, table, inputs.confounders, units), curve
