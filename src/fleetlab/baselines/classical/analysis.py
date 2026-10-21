"""Logs-only diagnosis and action ranking. No oracle or generator imports."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from fleetlab.baselines.classical.data import Row, VisibleData, load_export
from fleetlab.baselines.classical.features import Design, token
from fleetlab.baselines.classical.onset import match_gain, release_contrast, shifted
from fleetlab.baselines.classical.protocol import (
    EXPERIMENT_LENGTH,
    SCORING_LENGTH,
    Z90,
    population_bound,
    tie_key,
)
from fleetlab.baselines.classical.regression import predict, regress
from fleetlab.baselines.classical.selection import JointPrediction
from fleetlab.baselines.classical.signals import (
    event_signatures,
    outcome_check,
    stratified_contrast,
)
from fleetlab.contracts.actions import SCORING_BUDGETS, FixSet, cost_of
from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.ids import scope_covers
from fleetlab.contracts.tools import (
    Cause,
    Diagnosis,
    EvidenceRef,
    FleetWarning,
    Interval,
    RankedAction,
    RankedFixSet,
)


@dataclass(frozen=True)
class Answer:
    diagnosis: Diagnosis
    fix_set: RankedFixSet
    fix_sets: list[FixSet]
    evidence: dict[str, Any]

    def write(self, output: Path) -> None:
        """New output directory only: existing result records are never replaced."""
        import json

        serialized_evidence = json.dumps(self.evidence, indent=2, allow_nan=False) + "\n"
        output.mkdir(parents=True, exist_ok=False)
        outbox = output / "outbox"
        outbox.mkdir()
        (outbox / "diagnosis.json").write_text(self.diagnosis.model_dump_json(indent=2) + "\n")
        (outbox / "fix_set.json").write_text(self.fix_set.model_dump_json(indent=2) + "\n")
        cuts = output / "fix_sets"
        cuts.mkdir()
        for fix_set in self.fix_sets:
            label = f"{fix_set.budget.kind}-{fix_set.budget.amount:g}"
            (cuts / f"{label}.json").write_text(fix_set.model_dump_json(indent=2) + "\n")
        (output / "evidence.json").write_text(serialized_evidence)


def change_comparisons(data: VisibleData) -> list[dict[str, Any]]:
    """Task-standardized before/after contrasts at every recorded change date.

    Descriptive difference-in-differences, not a second independent estimate:
    equal weight over task strata present in every required comparison cell.
    A fleet change has no untreated comparator and is explicitly unidentified.
    """
    comparisons = []
    for change in data.changes:
        day, scope = change["day"], change["scope"]
        cells: dict[tuple[bool, bool, str], list[float]] = {}
        for row in data.episodes:
            affected = scope_covers(scope, row["robot_id"], row["site_id"])
            key = affected, row["day"] >= day, row["task_text"]
            cells.setdefault(key, []).append(float(row["outcome"] == "success"))
        tasks = sorted({r["task_text"] for r in data.episodes})
        common = [
            t
            for t in tasks
            if all((a, p, t) in cells for a in (False, True) for p in (False, True))
        ]
        difference = None
        if common:
            difference = float(
                np.mean(
                    [
                        np.mean(cells[True, True, t])
                        - np.mean(cells[True, False, t])
                        - np.mean(cells[False, True, t])
                        + np.mean(cells[False, False, t])
                        for t in common
                    ]
                )
                * 100
            )
        comparisons.append(
            {
                "scope": scope,
                "key": change["key"],
                "day": day,
                "old": change["old"],
                "new": change["new"],
                "common_tasks": len(common),
                "adjusted_change_points": difference,
                "missing": None if common else "no common before/after treated/control task strata",
            }
        )
    return comparisons


def change_points(data: VisibleData) -> list[dict[str, Any]]:
    """Scan every observed day boundary using task-stratified score tests.

    Cochran-Mantel-Haenszel score for a binary before/after contrast, aggregated
    across tasks; Bonferroni correction over every robot/site and five cuts.
    These are descriptive alarms, never a causal assignment to a nearby edit.
    """
    scopes = sorted(
        ["fleet", *[s.id for s in data.fleet.sites], *[r.id for r in data.fleet.robots()]]
    )
    output = []
    for scope in scopes:
        rows = [r for r in data.episodes if scope_covers(scope, r["robot_id"], r["site_id"])]
        contrasts = []
        for day in range(1, 6):
            cells: dict[str, list[float]] = {}
            for row in rows:
                counts = cells.setdefault(row["task_text"], [0.0, 0.0, 0.0, 0.0])
                index = 2 if row["day"] >= day else 0
                counts[index] += 1
                counts[index + 1] += row["outcome"] == "success"
            score, variance = 0.0, 0.0
            for pre, pre_success, post, post_success in cells.values():
                n = pre + post
                successes = pre_success + post_success
                if n > 1 and pre and post:
                    score += post_success - post * successes / n
                    variance += pre * post * successes * (n - successes) / (n * n * (n - 1))
            z = score / math.sqrt(variance) if variance > 0 else 0.0
            contrasts.append((day, z))
        day, z = max(contrasts, key=lambda item: abs(item[1]))
        adjusted = min(1.0, math.erfc(abs(z) / math.sqrt(2)) * len(scopes) * 5)
        output.append(
            {
                "scope": scope,
                "day": day,
                "score_z": z,
                "familywise_p": adjusted,
                "detected": adjusted <= 0.05,
                "direction": "decline" if z < 0 else "improvement",
                "all_cuts": [
                    {
                        "day": d,
                        "score_z": score,
                        "familywise_p": min(
                            1.0, math.erfc(abs(score) / math.sqrt(2)) * len(scopes) * 5
                        ),
                    }
                    for d, score in contrasts
                ],
                "coincident_changes": [
                    c["key"] for c in data.changes if c["scope"] == scope and c["day"] == day
                ],
            }
        )
    return output


def event_profiles(data: VisibleData) -> list[dict[str, Any]]:
    "Diagnostic descriptions only; events are downstream of failures, not confounders."
    kinds: dict[str, set[str]] = {}
    for event in data.events:
        kinds.setdefault(event["episode_id"], set()).add(event["kind"])
    result = []
    for robot in sorted(r.id for r in data.fleet.robots()):
        rows = [r for r in data.episodes if r["robot_id"] == robot]
        counts = Counter(k for r in rows for k in kinds.get(r["episode_id"], set()))
        result.append(
            {
                "robot_id": robot,
                "episodes": len(rows),
                "successes": sum(r["outcome"] == "success" for r in rows),
                "episodes_with_event": dict(sorted(counts.items())),
            }
        )
    return result


def _reference_rows(data: VisibleData) -> tuple[list[Row], np.ndarray[Any, Any]]:
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in data.episodes:
        groups.setdefault((row["robot_id"], row["task_text"]), []).append(row)
    rows = [r[0] for r in groups.values()]
    weights = np.asarray([len(r) for r in groups.values()], dtype=float)
    return rows, weights / weights.sum()


def fleet_diagnostic(data: VisibleData, onset: int) -> dict[str, Any]:
    """Robot/task-standardized before/after contrasts and Cochran heterogeneity.

    Pool the within-phase task rates across robots for Bernoulli sampling
    variances: a sparse all-success robot cell must not get excessive weight.
    Jeffreys smoothing avoids a zero pooled variance. A fleet release has no
    untreated time comparator; even a homogeneous contrast is conditional,
    not causally identified.
    """
    effects, variances = [], []
    pooled_cells: dict[tuple[str, bool], list[float]] = {}
    for row in data.episodes:
        pooled_cells.setdefault((row["task_text"], row["day"] >= onset), []).append(
            float(row["outcome"] == "success")
        )
    rates = {key: (sum(v) + 0.5) / (len(v) + 1) for key, v in pooled_cells.items()}
    for robot in data.fleet.robots():
        cells: dict[tuple[str, bool], list[float]] = {}
        for row in data.episodes:
            if row["robot_id"] == robot.id:
                cells.setdefault((row["task_text"], row["day"] >= onset), []).append(
                    float(row["outcome"] == "success")
                )
        tasks = [t for t in {t for t, _ in cells} if (t, False) in cells and (t, True) in cells]
        if not tasks:
            continue
        differences, noise = [], []
        for task in sorted(tasks):
            pre, post = cells[task, False], cells[task, True]
            differences.append(float(np.mean(post) - np.mean(pre)))
            noise.append(
                sum(
                    rates[task, phase] * (1 - rates[task, phase]) / len(v)
                    for phase, v in ((False, pre), (True, post))
                )
            )
        effects.append(float(np.mean(differences)))
        variances.append(sum(noise) / len(tasks) ** 2)
    if not effects:
        return dict(contrast_points=None, heterogeneity_p=1.0, robots=0, causal_identified=False)
    precision = 1 / np.asarray(variances)
    pooled = float(precision @ effects / precision.sum())
    q = float(precision @ (np.asarray(effects) - pooled) ** 2)
    return dict(
        contrast_points=100 * pooled,
        heterogeneity_q=q,
        heterogeneity_p=float(chi2.sf(q, len(effects) - 1)) if len(effects) > 1 else 1.0,
        robots=len(effects),
        causal_identified=False,
    )


def _onset_evidence(data: VisibleData, history: list[Row], scans: list[Row]) -> tuple[bool, str]:
    """Visible onset coherence, with no assumption about generator scope choices."""
    if not history or history[-1]["day"] == 0:
        return True, "persistent association; requires exchangeable robot/site effects"
    last = history[-1]
    scope, onset = last["scope"], last["day"]
    if scope == "fleet":
        return True, (
            "fleet release: local pooled site/task contrast standardized within robot/task; "
            "no site-significance veto; common time shocks remain alternatives"
        )
    scope_scan = next(r for r in scans if r["scope"] == scope)
    later = [
        r["day"]
        for r in scope_scan["all_cuts"]
        if r["day"] > onset and r["score_z"] < 0 and r["familywise_p"] <= 0.05
    ]
    if later:
        stop = min(later)
        rows = [
            r
            for r in data.episodes
            if r["day"] < stop and scope_covers(scope, r["robot_id"], r["site_id"])
        ]
        z, _ = stratified_contrast(
            rows,
            [r["day"] >= onset for r in rows],
            [float(r["outcome"] == "success") for r in rows],
        )
        if z > -1.0:
            return (
                False,
                "no decline before a distinct later break; earlier edit's contrast withheld",
            )
    return True, "each logged onset checked independently; a second break does not veto it"


def _bh(probabilities: list[float], rate: float = 0.10) -> set[int]:
    """Benjamini-Hochberg screening, explicitly conditional on the fitted model."""
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    passed = [k for k, i in enumerate(order, 1) if probabilities[i] <= rate * k / len(order)]
    return set(order[: max(passed)]) if passed else set()


def named_actions(records: list[dict[str, Any]]) -> set[int]:
    """The same current-evidence naming screen before and after experiments."""
    eligible = [
        i
        for i, r in enumerate(records)
        if (
            r.get("estimate_source")
            or (r["changed_features"] and not r["latent_candidate"] and not r["aliases"])
        )
        and r["gain_points"] >= 0.5
    ]
    accepted = {
        eligible[i] for i in _bh([1 - records[i]["diagnosis_confidence"] for i in eligible])
    }
    accepted.update(
        i for i, r in enumerate(records) if r.get("signature") and r["diagnosis_confidence"] >= 0.95
    )
    return accepted


def warning_crossing(
    projected: dict[int, np.ndarray[Any, Any]],
    frozen: dict[int, np.ndarray[Any, Any]],
    weight: np.ndarray[Any, Any],
    covered: np.ndarray[Any, Any],
    issued_day: int,
) -> int | None:
    """A future fleet-relative threshold crossing attributable to this trend.

    An already impaired robot supplies no positive warning lead time. Freezing
    this reading alone must prevent the crossing; common shocks do not qualify.
    """
    robot_weight = weight * covered / max(float(weight @ covered), 1e-12)
    contrast = weight - robot_weight  # robot failure rate minus fleet failure rate
    if float(contrast @ projected[issued_day]) >= 0.05:
        return None
    return next(
        (
            day
            for day in sorted(projected)
            if day > issued_day
            and float(contrast @ projected[day]) >= 0.05
            and float(contrast @ frozen[day]) < 0.05
        ),
        None,
    )


def analyse(data: VisibleData, *, bootstraps: int = 200, seed: int = 0) -> Answer:
    if bootstraps < 20:
        raise ValueError("at least 20 bootstrap draws are required")
    design = Design(data)
    y, label_checks = outcome_check(data)
    clusters = [f"{r['robot_id']}:{r['day']}" for r in data.episodes]
    folds = [int.from_bytes(hashlib.sha256(c.encode()).digest()[:4], "big") % 5 for c in clusters]
    fitted = regress(
        design.x,
        y,
        design.penalty,
        clusters,
        folds,
        bootstraps=bootstraps,
        seed=seed,
        nuisance=design.nuisance,
        sites=[r["site_id"] for r in data.episodes],
    )
    reference, weight = _reference_rows(data)
    for check in label_checks:
        indices = [
            i
            for i, r in enumerate(data.episodes)
            if r["robot_id"] == check["scope"] and r["day"] >= check["onset"]
        ]
        inferred = float(y[indices].sum())
        logged = sum(data.episodes[i]["outcome"] == "success" for i in indices)
        rate = min(0.99, max(0.0, (inferred - logged) / max(inferred, 1e-9)))
        check["estimated_missed_success_fraction"] = rate
    # Diagnosis may identify a recording fault, but repairing labels cannot
    # improve physical task success. Do not convert missed labels into gains.
    future = design.matrix(reference, future=True)
    flat = design.matrix(reference, future=True, flat=True)
    betas = np.column_stack([fitted.beta, fitted.draws.T])
    original = predict(future, betas, design.nuisance, fitted.link)
    original_flat = predict(flat, betas, design.nuisance, fitted.link)
    scans = change_points(data)
    symptoms = event_signatures(data)
    exposure = design.x[:, design.nuisance :]
    alias_pairs = []
    for left in range(exposure.shape[1]):
        for right in range(left):
            left_name = design.names[left + design.nuisance]
            right_name = design.names[right + design.nuisance]
            # Two levels of one categorical variable encode one contrast;
            # complementary columns are not competing causal assignments.
            if left_name.split("=", 1)[0] == right_name.split("=", 1)[0] and left_name.startswith(
                ("config:", "condition:")
            ):
                continue
            a, b = exposure[:, left], exposure[:, right]
            if (
                np.std(a) > 1e-8
                and np.std(b) > 1e-8
                and abs(float(np.corrcoef(a, b)[0, 1])) >= 0.98
            ):
                alias_pairs.append((left + design.nuisance, right + design.nuisance))
    candidates = design.candidates()
    predictions: list[RankedAction] = []
    records: list[dict[str, Any]] = []
    repairs = []
    offsets = []
    for action in candidates:
        repaired = design.matrix(reference, future=True, action=action)
        repaired_flat = design.matrix(reference, future=True, flat=True, action=action)
        history = [
            c
            for c in data.changes
            if c["scope"] == action.target.scope and c["key"] == action.target.config_key
        ]
        unseen = bool(history) and token(
            history[-1]["old"]
        ) not in design.observed_config_values.get(str(action.target.config_key), set())
        coherent, onset_rule = _onset_evidence(data, history, scans)
        fleet_contrast = (
            fleet_diagnostic(data, history[-1]["day"])
            if (history and action.target.scope == "fleet")
            else None
        )
        if unseen or not coherent:
            repaired, repaired_flat = future.copy(), flat.copy()
        changed = set(np.flatnonzero(np.any(np.abs(repaired - future) > 1e-10, axis=0)))
        aliased = [p for p in alias_pairs if changed.intersection(p)]
        scope = action.target.scope
        covered = np.asarray([scope_covers(scope, r["robot_id"], r["site_id"]) for r in reference])
        population = population_bound(weight, covered)
        # Unexplained random intercepts are actionable hypotheses, never discarded
        # simply because all sites have the same recorded condition values.
        latent = False
        signature = None
        nuisance_name = f"site:{scope}" if action.target.condition_key else f"robot:{scope}"
        if not history and nuisance_name in design.names:
            column = design.names.index(nuisance_name)
            latent_eligible = action.target.condition_key is not None or not changed
            if latent_eligible and fitted.beta[column] < 0:
                condition_keys = {
                    a.target.condition_key
                    for a in candidates
                    if a.target.scope == scope and a.target.condition_key is not None
                }
                alternatives = (
                    len(condition_keys)
                    if action.target.condition_key
                    else len(
                        {
                            a.target.component
                            for a in candidates
                            if a.target.scope == scope and a.target.component is not None
                        }
                    )
                )
                allocation = 1.0 / max(1, alternatives)
                supported = [e for e in symptoms if e.scope == scope and e.p <= 0.05]
                drops = [e for e in supported if e.kind == "drop" and e.robots >= 2]
                if drops and action.target.condition_key and "object_batch" in condition_keys:
                    allocation = (
                        (1.0 if alternatives == 1 else 0.75)
                        if str(action.target.condition_key) == "object_batch"
                        else 0.25 / max(1, alternatives - 1)
                    )
                    if str(action.target.condition_key) == "object_batch":
                        signature = drops[0]
                # Remove a fraction of an unexplained negative intercept. This
                # is an equal-cause prior unless the shared drop signature updates it.
                repaired[:, column] *= 1 - allocation
                repaired_flat[:, column] *= 1 - allocation
                changed.add(column)
                latent = True
        # A coherent shared drop signature may be hidden by fitted robot effects;
        # retain it even if the site's intercept is slightly positive.
        drops = [
            e
            for e in symptoms
            if e.scope == scope and e.kind == "drop" and e.robots >= 2 and e.p <= 0.05
        ]
        if drops and str(action.target.condition_key) == "object_batch":
            signature = drops[0]
        gains = weight @ (predict(repaired, betas, design.nuisance, fitted.link) - original) * 100
        flat_gains = (
            weight
            @ (predict(repaired_flat, betas, design.nuisance, fitted.link) - original_flat)
            * 100
        )
        onset_contrast = None
        offset = np.zeros_like(original)
        if history and scope == "fleet" and changed and not unseen:
            affected = np.any(np.abs(repaired - future) > 1e-10, axis=1)
            gains, onset_contrast = release_contrast(
                data,
                history[-1]["day"],
                reference,
                weight,
                affected,
                draws=bootstraps,
                seed=seed,
            )
            offset = match_gain(original, weight, affected, gains)
            # This release uses its local contrast; other repairs keep their
            # feature-union model. All cuts use exactly the same transported gain.
            repaired, repaired_flat = future.copy(), flat.copy()
            gains = weight @ (shifted(original, offset) - original) * 100
            flat_offset = match_gain(original_flat, weight, affected, gains)
            flat_gains = weight @ (shifted(original_flat, flat_offset) - original_flat) * 100
            onset_rule = onset_contrast["rule"] + "; " + onset_contrast["limitation"]
        point = float(gains[0])
        low = min(
            float(np.quantile(gains[1:], 0.05)), float(np.quantile(flat_gains[1:], 0.05)), point
        )
        high = max(
            float(np.quantile(gains[1:], 0.95)), float(np.quantile(flat_gains[1:], 0.95)), point
        )
        # Predictive envelope for the fixed design's two scoring days, relative
        # to six observed days. Unknown counterfactual pairing uses its maximum
        # Bernoulli difference variance; this is conservative, not an oracle SE.
        p0 = original[:, 0]
        p1 = shifted(predict(repaired, fitted.beta, design.nuisance, fitted.link), offset[:, 0])
        variance = np.maximum(0, np.minimum(p0 + p1, 2 - p0 - p1) - (p1 - p0) ** 2)
        affected = np.any(np.abs(repaired - future) > 1e-10, axis=1) | (
            np.abs(offset[:, 0]) > 1e-10
        )
        if onset_contrast:
            affected |= covered
        variance *= affected
        observed_days = len({r["day"] for r in data.episodes})
        scoring_n = len(y) * SCORING_LENGTH / observed_days
        prediction_radius = Z90 * math.sqrt(float(weight @ variance) / scoring_n) * 100
        low -= prediction_radius
        high += prediction_radius
        low, high = max(low, -population), min(high, population)
        confidence = float(np.mean(gains[1:] > 1e-6))
        informative = bool(changed) and not aliased and not latent and not unseen
        if onset_contrast and not onset_contrast["complete"]:
            informative = False
            low, high, confidence = -population, population, 0.0
        if latent:
            # Causal allocation among alternatives is unidentified. Keep the
            # candidate with its explicit prior mean and a zero-inclusive range.
            low, high = min(low, 0), max(high, population)
            confidence = 0.0 if signature is None else 1 - signature.p
        if aliased:
            low, high, confidence = min(low, 0), max(high, population), min(confidence, 0.5)
        if not changed or unseen:
            low, high, confidence = -population, population, 0.0
        reason = (
            f"Task/site/robot/day-adjusted {fitted.link}; "
            "selected daily level/step/trend health histories, "
            "trends extrapolated to day 8.5, with a flat-health sensitivity envelope. "
            "90% hierarchical multiplier interval plus a conservative finite "
            "two-day scoring-panel envelope. " + onset_rule + "."
        )
        if onset_contrast:
            onset_contrast["forecast_clipped"] = abs(point - onset_contrast["gain_points"]) > 1e-6
            reason = (
                onset_rule + ". 90% pooled-Bernoulli normal contrast interval plus "
                "a conservative finite scoring-panel envelope; persistent local effect "
                "transported as a final-success log-odds shift in the joint forecast."
            )
        if latent:
            reason += (
                " Unexplained deficit: equal allocation among scope-compatible causes, "
                "updated by shared drop evidence; causal allocation unidentifiable."
            )
        if signature:
            reason += " Excess drops across multiple robots support an object-batch hypothesis."
        if aliased:
            reason += " Nearly identical actionable histories; separate effects unidentified."
        if unseen:
            reason += " Restoration value never observed; no estimated contrast."
        if not changed:
            reason += " No measured repair contrast; ignorance range, not evidence of no fault."
        if not coherent:
            reason += (
                " Recorded edit fails visible onset coherence; withheld in both ranking "
                "and diagnosis."
            )
        predictions.append(
            RankedAction(
                action=action,
                expected_gain=point,
                uncertainty=Interval(low=low, high=high),
                rationale=reason,
            )
        )
        repairs.append(repaired)
        offsets.append(offset[:, 0])
        records.append(
            dict(
                action=action.model_dump(mode="json"),
                gain_points=point,
                low=low,
                high=high,
                diagnosis_confidence=confidence,
                positive_draw_fraction=float(np.mean(gains[1:] > 1e-6)),
                prediction_radius=prediction_radius,
                estimation_variance=float(np.var(gains[1:], ddof=1)),
                diagnosis_rule=onset_rule,
                unobserved_restoration=unseen,
                informative_interval=informative,
                observational_pooling=informative,
                fleet_contrast=fleet_contrast,
                onset_contrast=onset_contrast,
                latent_candidate=latent,
                signature=asdict(signature) if signature else None,
                changed_features=[design.names[i] for i in sorted(changed)],
                aliases=[[design.names[i] for i in pair] for pair in aliased],
            )
        )
    if len(predictions) < 10:
        raise ValueError("catalogue supplies fewer than ten distinct repair candidates")
    joint = JointPrediction(
        future, fitted.beta, design.nuisance, fitted.link, repairs, weight, np.asarray(offsets)
    )
    order, selection = joint.order(
        [cost_of(a, data.costs) or 0 for a in candidates], [tie_key(a) for a in candidates]
    )
    ranked = RankedFixSet(actions=[predictions[i] for i in order])
    accepted = named_actions(records)
    causes = []
    for i in sorted(accepted):
        row, rec = predictions[i], records[i]
        action = row.action
        history = [
            c
            for c in data.changes
            if c["scope"] == action.target.scope and c["key"] == action.target.config_key
        ]
        onset = history[-1]["day"] if history else 0
        examples = [
            r["episode_id"]
            for r in data.episodes
            if r["day"] >= onset
            and scope_covers(action.target.scope, r["robot_id"], r["site_id"])
            and r["outcome"] == "failure"
        ]
        if rec["signature"]:
            examples = rec["signature"]["episodes"]
        causes.append(
            Cause(
                target=action.target,
                direction=Direction.changed
                if action.target.config_key or action.target.condition_key
                else Direction.degraded,
                confidence=rec["diagnosis_confidence"],
                evidence=[
                    EvidenceRef(episode_id=e, step=None, table=None, query=None)
                    for e in examples[:3]
                ],
                failure_mode=row.rationale,
            )
        )
    for check in label_checks:
        # The duration signature locates a robot's recording path, not a hardware
        # component. Do not infer a target from development-world truth labels.
        recording_target = Target(scope=check["scope"])
        causes = [c for c in causes if c.target != recording_target]
        causes.append(
            Cause(
                target=recording_target,
                direction=Direction.changed,
                confidence=1 - check["familywise_p"],
                evidence=[
                    EvidenceRef(episode_id=e, step=None, table=None, query=None)
                    for e in check["episode_ids"][:3]
                ],
                failure_mode=check["interpretation"]
                + "; robot scope only; no component identified by these logs",
            )
        )
    warnings = []
    warning_forecasts = []
    issued_day = max(r["day"] for r in data.episodes)
    forecast_days = range(issued_day, issued_day + EXPERIMENT_LENGTH + SCORING_LENGTH + 1)
    daily = {
        day: predict(
            design.matrix(reference, forecast_day=float(day)),
            fitted.beta,
            design.nuisance,
            fitted.link,
        )
        for day in forecast_days
    }
    for h in design.health:
        column = design.names.index(f"health:{h.component}.{h.metric}:{h.direction:g}")
        effect_confidence = float(np.mean(fitted.draws[:, column] < 0))
        for robot, trajectory in h.robots.items():
            if trajectory.slope * h.direction <= 0 or effect_confidence < 0.9:
                continue
            covered = np.asarray([r["robot_id"] == robot for r in reference], dtype=float)
            frozen = {}
            for day in forecast_days:
                matrix = design.matrix(reference, forecast_day=float(day))
                matrix[covered.astype(bool), column] = h.exposure(robot, issued_day)
                frozen[day] = predict(matrix, fitted.beta, design.nuisance, fitted.link)
            crossing = warning_crossing(daily, frozen, weight, covered, issued_day)
            warning_forecasts.append(
                dict(
                    scope=robot,
                    metric=h.metric,
                    issued_day=issued_day,
                    crossing_day=crossing,
                    lead_days=crossing - issued_day if crossing else None,
                )
            )
            if crossing is not None:
                z = abs(trajectory.slope) / max(trajectory.slope_se, 1e-9)
                warnings.append(
                    FleetWarning(
                        scope=robot,
                        metric="failure_rate",
                        expected_day=crossing,
                        confidence=min(effect_confidence, float(1 - math.erfc(z / math.sqrt(2)))),
                    )
                )
    diagnosis = Diagnosis(
        causes=sorted(causes, key=lambda c: -c.confidence),
        mode_summaries=[
            (
                "Conditional associations assume exchangeable residual robot/site "
                "effects; persistent exposures are not identified against arbitrary unit "
                "confounders."
            ),
            (
                "Unexplained deficits remain repair candidates. Event kinds and durations"
                " are diagnostic evidence, not independent causal outcomes."
            ),
            (
                "Warnings require a future fleet-relative five-point crossing prevented by "
                "freezing the selected health trend. Already-elevated robots have no lead time. "
                "Handed-over attempts count as unsuccessful autonomous attempts; suspect "
                "short-failure labels use an explicit peer-based sensitivity model."
            ),
        ],
        warnings=warnings,
        retrain=[],
    )
    evidence = dict(
        method="classical-binomial-v2",
        selected_link=fitted.link,
        world_id=data.fleet.id,
        observed_episodes=len(data.episodes),
        observed_days=len({r["day"] for r in data.episodes}),
        experiment_robots=[dict(robot_id=r.id, site_id=r.site_id) for r in data.fleet.robots()],
        successes=sum(r["outcome"] == "success" for r in data.episodes),
        fitted_success_total=float(y.sum()),
        handed_over=sum(r["outcome"] == "handed_over" for r in data.episodes),
        seed=seed,
        bootstrap_draws=bootstraps,
        cluster_unit="positive site/robot/robot-day multipliers",
        input_sha256=data.hashes,
        features=design.names,
        coefficients=fitted.beta.tolist(),
        penalty_multiplier=fitted.penalty_multiplier,
        validation_log_loss=fitted.validation,
        unsupported_features=[design.names[i] for i in np.flatnonzero(design.unsupported)],
        exposure_aliases=[[design.names[i] for i in pair] for pair in alias_pairs],
        health_models=[asdict(h) for h in design.health],
        warning_forecasts=warning_forecasts,
        label_checks=label_checks,
        event_signatures=[asdict(e) for e in symptoms],
        change_points=scans,
        change_comparisons=change_comparisons(data),
        event_profiles=event_profiles(data),
        actions=records,
        selection=selection,
        # Sufficient visible-only model state for experiment updating;
        # not a pickle, executable code, or hidden result.
        forecast=dict(
            reference=[
                {k: r[k] for k in ("robot_id", "site_id", "task_text", "day")} for r in reference
            ],
            weight=weight.tolist(),
            matrix=future.tolist(),
            nuisance=design.nuisance,
            repaired=[r.tolist() for r in repairs],
            offsets=[r.tolist() for r in offsets],
        ),
        limits=[
            (
                "Conditional intervals omit arbitrary causal confounding; a conservative "
                "finite scoring-panel noise envelope is included. Informative and "
                "ignorance ranges are reported separately."
            ),
            (
                "No videos, hidden truth, oracle, generator or runs read. Experiment "
                "results, when supplied, are separate randomized evidence."
            ),
            (
                "Camera cleaning and recalibration are distinct repair hypotheses; "
                "other components use service. Unknown metrics use both "
                "tails, usage counters excluded."
            ),
        ],
    )
    return Answer(
        diagnosis, ranked, [ranked.as_fix_set(b, data.costs) for b in SCORING_BUDGETS], evidence
    )


def run(
    export: Path, *, catalogue: Path | None = None, bootstraps: int = 200, seed: int = 0
) -> Answer:
    return analyse(load_export(export, catalogue), bootstraps=bootstraps, seed=seed)
