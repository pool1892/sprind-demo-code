"""Claim 5, using public oracle outcomes; no missing example is dropped."""

from collections import defaultdict
from typing import Any

from fleetlab.scoring.models import ScoringInputs


def score_foresight(inputs: ScoringInputs) -> dict[str, Any]:
    diagnosis = inputs.diagnosis
    warnings: list[dict[str, Any]] = []
    for fault in inputs.drift_faults:
        scope = fault.target.scope
        rows: dict[int, list[tuple[bool | None, bool]]] = defaultdict(list)
        if inputs.as_run is not None:
            for identity in inputs.as_run.identities:
                rows[identity.day].append(
                    (
                        inputs.as_run.success(identity.episode_id, identity.faults),
                        fault.target.covers(identity.robot_id, identity.site_id),
                    )
                )
        crossing = None
        missing = None
        rates = []
        for day, entries in sorted(rows.items()):
            local = [outcome for outcome, covered in entries if covered]
            if any(outcome is None for outcome, _ in entries) or not local:
                missing = "daily as-run outcomes incomplete"
                break
            fleet_rate = sum(not outcome for outcome, _ in entries) / len(entries)
            local_rate = sum(not outcome for outcome in local) / len(local)
            rates.append(
                {"day": day, "scope_failure_rate": local_rate, "fleet_failure_rate": fleet_rate}
            )
            local_failed = sum(not outcome for outcome in local)
            fleet_failed = sum(not outcome for outcome, _ in entries)
            above = 100 * (local_failed * len(entries) - fleet_failed * len(local))
            if day >= fault.schedule.onset_day and above > 5 * len(local) * len(entries):
                crossing = day
                break
        if not rows:
            missing = "no daily as-run outcomes supplied"
        matched = (
            []
            if diagnosis is None
            else [i for i, w in enumerate(diagnosis.warnings) if w.scope == scope]
        )
        old_crossing = crossing
        crossing = inputs.drift_crossing_days.get(fault.fault_id)
        issued = inputs.issued_day
        truth_missing = fault.fault_id not in inputs.drift_crossing_days
        if truth_missing:
            state = "unavailable"
            reason = "fault_table.json lacks drift_crossing_day for " + fault.fault_id
        elif crossing is None:
            state, reason = "never crossed", None
        elif issued is None:
            state, reason = "unavailable", "analysis issue day missing"
        elif crossing <= issued:
            state, reason = "already crossed", None
        elif matched:
            state, reason = "later crossing, warned", None
        else:
            state, reason = "later crossing, not warned", None
        eligible = state in ("later crossing, warned", "later crossing, not warned")
        forecast_errors = (
            []
            if diagnosis is None or crossing is None
            else [
                {
                    "warning_index": i,
                    "expected_day": diagnosis.warnings[i].expected_day,
                    "error_days": diagnosis.warnings[i].expected_day - crossing,
                }
                for i in matched
            ]
        )
        logged_state = (
            "unavailable"
            if missing or issued is None
            else "never crossed"
            if old_crossing is None
            else "already crossed"
            if old_crossing <= issued
            else "later crossing, warned"
            if matched
            else "later crossing, not warned"
        )
        logged_future = logged_state.startswith("later crossing")
        logged_measure = {
            "crossing_day": old_crossing,
            "issued_day": issued,
            "state": logged_state,
            "noticed": bool(matched) if diagnosis is not None else None,
            "lead_days": old_crossing - issued
            if logged_future and matched and old_crossing is not None and issued is not None
            else None,
            "before_crossing": bool(matched) if logged_future and diagnosis is not None else None,
            "forecast_errors": []
            if diagnosis is None or old_crossing is None
            else [
                {
                    "warning_index": i,
                    "expected_day": diagnosis.warnings[i].expected_day,
                    "error_days": diagnosis.warnings[i].expected_day - old_crossing,
                }
                for i in matched
            ],
            "missing": missing or ("analysis issue day missing" if issued is None else None),
        }
        warnings.append(
            {
                "fault_id": fault.fault_id,
                "kind": fault.kind,
                "scope": scope,
                "crossing_day": crossing,
                "old_wording_crossing_day": old_crossing,
                "logged_rate_measure": logged_measure,
                "old_wording_missing": missing,
                "state": state,
                "noticed": bool(matched) if diagnosis is not None else None,
                "forecast_errors": forecast_errors,
                "issued_day": issued,
                "warning_indices": matched,
                "rates": rates,
                "lead_days": crossing - issued
                if eligible and crossing is not None and issued is not None and matched
                else None,
                "before_crossing": bool(matched) if eligible and diagnosis is not None else None,
                "missing": reason,
            }
        )
    retrain: list[dict[str, Any]] = []
    healthy = inputs.as_run if inputs.as_run is not None else inputs.oracle
    by_id = {i.episode_id: i for i in healthy.identities}
    for indicator in [] if diagnosis is None else diagnosis.retrain:
        ids = list(dict.fromkeys(indicator.episode_ids))
        outcomes = {}
        for episode_id in ids:
            outcomes[episode_id] = healthy.success(episode_id, ()) if episode_id in by_id else None
        absent = [i for i, outcome in outcomes.items() if outcome is None]
        hits = sum(outcome is False for outcome in outcomes.values())
        retrain.append(
            {
                "failure_mode": indicator.failure_mode,
                "examples": len(ids),
                "healthy_failures": hits,
                "outcomes": outcomes,
                "precision": hits / len(ids) if ids and not absent else None,
                "missing_examples": absent,
                "missing": "missing all-faults-removed outcomes"
                if absent
                else ("no examples" if not ids else None),
            }
        )
    missing_examples = sorted({i for r in retrain for i in r["missing_examples"]})
    if missing_examples:
        for row in retrain:
            row["precision"] = None
            row["missing"] = "analysis has missing all-faults-removed examples"
    matched_warnings = {i for w in warnings for i in w["warning_indices"]}
    false_alarms = (
        []
        if diagnosis is None
        else [
            {"warning_index": i, **warning.model_dump(mode="json")}
            for i, warning in enumerate(diagnosis.warnings)
            if i not in matched_warnings
        ]
    )
    return {
        "warnings": warnings,
        "false_alarms": false_alarms,
        "false_alarm_count": len(false_alarms),
        "missing_examples": missing_examples,
        "replay_request_command": "uv run --locked python -m fleetlab.oracle fresh WORLD_ROOT "
        "--name NEW_NAME --request report.json",
        "retrain": retrain,
        "meaning": "the policy itself fails this attempt on a healthy robot",
        "missing": "no diagnosis submitted" if diagnosis is None else None,
    }
