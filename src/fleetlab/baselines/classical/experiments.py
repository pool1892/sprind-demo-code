"""Fixed, visible-only A/B planning and updating through the agents' queue.

No simulator or truth imports. An external engine answers the queue. Reserve a
logged fleet change and all conditions at the strongest site alarm/deficit,
then fill from joint rank subject to the public per-robot/day arm capacity.
Eight matched-start requests, 60 attempts per arm; correlated arm outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.special import ndtr

from fleetlab.agents.sessions import Analysis
from fleetlab.baselines.classical.analysis import Answer, named_actions
from fleetlab.baselines.classical.capacity import select_feasible
from fleetlab.baselines.classical.data import load_export, visible_file
from fleetlab.baselines.classical.onset import shifted
from fleetlab.baselines.classical.protocol import (
    SCORING_LENGTH,
    Z90,
    experiment_window,
    population_bound,
    tie_key,
)
from fleetlab.baselines.classical.regression import Array, predict
from fleetlab.baselines.classical.selection import JointPrediction
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, CostKey, cost_of
from fleetlab.contracts.faults import Direction
from fleetlab.contracts.ids import scope_covers
from fleetlab.contracts.tools import (
    Cause,
    Diagnosis,
    EvidenceRef,
    ExperimentArm,
    ExperimentRequest,
    Interval,
    RankedAction,
    RankedFixSet,
    Receipt,
)
from fleetlab.toolserver.public_results import EXECUTION_FAILED, request_results_problems
from fleetlab.toolserver.server import Paths, ToolBox

RULE = (
    "Reserve the highest-estimated logged fleet change and all conditions at the "
    "strongest significant site decline (else largest unexplained deficit); "
    "then fill from joint rank and remaining singleton gains, skipping duplicates "
    "and requests exceeding the "
    "public per-robot/day arm capacity. Eight requests, sixty per arm; "
    "sorted affected robots, matched starts with correlated arm outcomes."
)


def read_answer(root: Path) -> Answer:
    diagnosis = Diagnosis.model_validate_json((root / "outbox/diagnosis.json").read_text())
    ranked = RankedFixSet.model_validate_json((root / "outbox/fix_set.json").read_text())
    return Answer(diagnosis, ranked, [], json.loads((root / "evidence.json").read_text()))


def choose(answer: Answer, *, episodes_per_arm: int = 60, requests: int = 8) -> list[Action]:
    return selection(answer, episodes_per_arm=episodes_per_arm, requests=requests)[0]


def selection(
    answer: Answer, *, episodes_per_arm: int = 60, requests: int = 8
) -> tuple[list[Action], dict[str, Any]]:
    records = answer.evidence["actions"]
    fleet = [
        r
        for r in records
        if r["action"]["target"]["scope"] == "fleet" and r["action"]["target"]["config_key"]
    ]
    selected = []
    if fleet:
        selected.append(
            Action.model_validate(
                min(
                    fleet,
                    key=lambda r: (-r["gain_points"], tie_key(Action.model_validate(r["action"]))),
                )["action"]
            )
        )
    sites: dict[str, float] = {}
    for r in records:
        if r["action"]["target"]["condition_key"] and r.get("latent_candidate"):
            scope = r["action"]["target"]["scope"]
            sites[scope] = sites.get(scope, 0.0) + max(0, r["gain_points"])
    condition_sites = {
        r["action"]["target"]["scope"] for r in records if r["action"]["target"]["condition_key"]
    }
    alarms = [
        r
        for r in answer.evidence["change_points"]
        if r["scope"] in condition_sites and r["detected"] and r["direction"] == "decline"
    ]
    if alarms or (sites and max(sites.values()) > 0):
        site = (
            min(alarms, key=lambda r: (r["score_z"], r["scope"]))["scope"]
            if alarms
            else min(sites, key=lambda s: (-sites[s], s))
        )
        conditions = sorted(
            [
                r
                for r in records
                if r["action"]["target"]["scope"] == site and r["action"]["target"]["condition_key"]
            ],
            key=lambda r: (-r["gain_points"], tie_key(Action.model_validate(r["action"]))),
        )
        selected.extend(Action.model_validate(r["action"]) for r in conditions)
    selected.extend(r.action for r in answer.fix_set.actions)
    selected.extend(
        Action.model_validate(r["action"])
        for r in sorted(
            records, key=lambda r: (-r["gain_points"], tie_key(Action.model_validate(r["action"])))
        )
    )
    return select_feasible(
        selected,
        answer.evidence.get("experiment_robots", answer.evidence["forecast"]["reference"]),
        episodes_per_arm=episodes_per_arm,
        request_limit=requests,
    )


def plan(
    export: Path, answer: Answer, root: Path, *, requests: int = 8, episodes_per_arm: int = 60
) -> Path:
    if not 1 <= requests <= 8 or not 1 <= episodes_per_arm <= 60:
        raise ValueError("design budget: at most eight requests, sixty attempts per arm")
    data = load_export(export)
    if answer.evidence["world_id"] != data.fleet.id:
        raise ValueError("answer and export belong to different worlds")
    analysis = Analysis.create(
        data.fleet.id, label="classical fixed A/B rule", base=root.parent, analysis_id=root.name
    )
    actions, capacity = selection(answer, episodes_per_arm=episodes_per_arm, requests=requests)
    box = ToolBox(
        Paths(
            export=export.resolve(),
            memory=analysis.memory,
            outbox=analysis.outbox,
            results=analysis.results,
            log=analysis.root / "tool_calls.jsonl",
        ),
        session=1,
    )
    try:
        for action in actions:
            robots = sorted(
                r.id
                for r in data.fleet.robots()
                if scope_covers(action.target.scope, r.id, r.site_id)
            )
            n = episodes_per_arm
            request = ExperimentRequest(
                arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[action])],
                robots=robots,
                episodes_per_arm=n,
                paired_starts=True,
                power_note=(
                    f"n={n} pairs; two-sided alpha .05, 80% power. "
                    "Conservative unpaired Bernoulli MDE=(1.96+.84)*sqrt(.5/n)="
                    f"{2.80 * np.sqrt(0.5 / n):.3f} affected-robot probability, "
                    "multiplied by affected fleet share for fleet points. "
                    "Arm scene/noise seeds differ; small effects remain unresolved."
                ),
            )
            result = box.call("request_experiment", request.model_dump(mode="json"))
            receipt = Receipt.model_validate_json(result.content[0]["text"])
            if not receipt.accepted:
                raise ValueError(f"experiment refused: {receipt.problems}")
    finally:
        box.close()
    (analysis.memory / "rule.json").write_text(
        json.dumps(
            dict(
                rule=RULE,
                capacity=capacity,
                requests=requests,
                episodes_per_arm=episodes_per_arm,
                source_world=data.fleet.id,
                experiment_window=experiment_window(max(r["day"] for r in data.episodes)),
            ),
            indent=2,
        )
        + "\n"
    )
    queued = analysis.write_queue(1, export.resolve())
    if queued is None:
        raise ValueError("no eligible experiments")
    return queued[0]


def arm_estimate(
    control: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    robot_weights: dict[str, float],
    seed: int = 0,
) -> dict[str, Any]:
    """Joint outcome posterior for matched starts with potentially correlated noise.

    Different engine seeds do not remove dependence from shared robot/task/scene.
    Dirichlet half-counts on the four outcome pairs allow either sign of covariance
    and retain uncertainty even with no observed discordant pairs.
    """
    if not control or len(control) != len(treatment):
        raise ValueError("matched arms must have the same nonzero size")
    cells: dict[str, tuple[list[float], list[float]]] = {}
    for before, after in zip(control, treatment, strict=True):
        if (before["robot_id"], before["task_text"]) != (after["robot_id"], after["task_text"]):
            raise ValueError("paired arm robot/task mismatch")
        if any(r["outcome"] not in {"success", "failure", "handed_over"} for r in (before, after)):
            raise ValueError("unknown experimental outcome")
        arms = cells.setdefault(before["robot_id"], ([], []))
        for values, row in zip(arms, (before, after), strict=True):
            values.append(float(row["outcome"] == "success"))
    if set(cells) != set(robot_weights):
        raise ValueError("experiment must populate every requested robot")
    rng = np.random.default_rng(seed)
    draws = np.zeros(10000)
    point = 0.0
    counts = {}
    for robot, (before_values, after_values) in sorted(cells.items()):
        joint_counts = np.bincount(
            [int(a) * 2 + int(b) for a, b in zip(before_values, after_values, strict=True)],
            minlength=4,
        )
        counts[robot] = joint_counts.tolist()  # failure/failure, rescued, broken, success/success
        posterior = rng.dirichlet(joint_counts + 0.5, size=len(draws))
        scale = robot_weights[robot] * 100
        draws += scale * (posterior[:, 1] - posterior[:, 2])
        point += scale * (float(np.mean(after_values)) - float(np.mean(before_values)))
    return dict(
        gain_points=point,
        low=min(point, float(np.quantile(draws, 0.05))),
        high=max(point, float(np.quantile(draws, 0.95))),
        variance=float(np.var(draws, ddof=1)),
        confidence=float(np.mean(draws > 0)),
        pairs=len(control),
        counts=counts,
        interval=("90% paired-outcome Dirichlet half-count interval, robot-standardized"),
    )


# Compatibility for callers of the first frozen version.
paired_estimate = arm_estimate


def combine_estimates(observed: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Independent normal precision pooling of mean effects, in fleet points.

    Unidentified observational assignments have zero precision. Prediction-panel
    noise does not count as uncertainty in the observational mean effect.
    """
    variance = float(
        measured.get("variance", ((measured["high"] - measured["low"]) / (2 * Z90)) ** 2)
    )
    prior_variance = float(
        observed.get("estimation_variance", ((observed["high"] - observed["low"]) / (2 * Z90)) ** 2)
    )
    unidentified = (
        not observed.get("observational_pooling", observed.get("informative_interval"))
        or prior_variance <= 1e-8
    )
    difference = measured["gain_points"] - observed["gain_points"]
    conflict = (
        not unidentified
        and variance > 1e-12
        and abs(difference) > 1.96 * np.sqrt(prior_variance + variance)
    )
    if conflict:
        # Observational bias is not sampling precision. A commensurability
        # check adds enough bias variance to cover the between-source discrepancy.
        prior_variance = max(prior_variance, difference**2 - variance)
    if variance < 1e-12 or unidentified:
        fraction = 1.0
    else:
        fraction = prior_variance / (prior_variance + variance)
    precision_point = (1 - fraction) * observed["gain_points"] + fraction * measured["gain_points"]
    point = precision_point
    pooled_variance = fraction * variance
    radius = Z90 * np.sqrt(max(0, pooled_variance))
    low, high = point - radius, point + radius
    # A pretest safeguard for candidates with no identified observational
    # estimate: an unresolved experiment may lower, but cannot promote, its
    # hypothesis-prior point. Preserve the unscreened precision interval.
    screened = (
        unidentified
        and variance > 1e-12
        and measured["low"] <= 0 <= measured["high"]
        and precision_point > observed["gain_points"]
    )
    if screened:
        point = observed["gain_points"]
        low, high = min(low, point), max(high, point)
    return dict(
        gain_points=float(point),
        precision_point=float(precision_point),
        low=float(low),
        high=float(high),
        variance=pooled_variance,
        observational_variance_used=prior_variance,
        observational_unidentified=unidentified,
        source_conflict=bool(conflict),
        unresolved_promotion_screened=bool(screened),
        measurement_weight=float(fraction),
        confidence=float(ndtr(point / np.sqrt(pooled_variance)))
        if pooled_variance > 0
        else float(point > 0),
    )


def calibrate(
    x: Array, beta: Array, nuisance: int, weight: Array, covered: Array, index: int, gain: float
) -> tuple[Array, bool]:
    base = JointPrediction(x, beta, nuisance, "logistic", [x], weight)
    repaired = x.copy()

    def contrast(offset: float) -> float:
        repaired[:, index + 1] = covered * offset
        return float((base._probability(repaired) - base.base) @ weight * 100)

    lower, upper = contrast(-30), contrast(30)
    target = float(np.clip(gain, lower + 1e-8, upper - 1e-8))
    offset = float(brentq(lambda t: contrast(t) - target, -30, 30))
    repaired[:, index + 1] = covered * offset
    return repaired, abs(target - gain) > 1e-6


def update(export: Path, original: Answer, analysis_root: Path, *, seed: int = 0) -> Answer:
    analysis = Analysis(analysis_root)
    data = load_export(export)
    evidence = deepcopy(original.evidence)
    if data.fleet.id != analysis.world_id or data.fleet.id != evidence["world_id"]:
        raise ValueError("experiment world differs from analysis")
    window = experiment_window(max(r["day"] for r in data.episodes))
    rule = json.loads((analysis.memory / "rule.json").read_text())
    if tuple(rule["experiment_window"]) != window:
        raise ValueError("analysis experiment window differs from the observed-window boundary")
    rows = pq.read_table(visible_file(export, "log/episodes.parquet")).to_pylist()
    by_id = {r["episode_id"]: r for r in rows}
    results = analysis.results_by_id()
    requests = analysis.requests()
    problems = request_results_problems(
        list(results.values()), {request.experiment_id: request.request for request in requests}
    )
    if any(result.message == EXECUTION_FAILED for result in results.values()):
        problems.append("experiment execution failed")
    if problems:
        raise ValueError("resume experiments before updating an answer: " + "; ".join(problems))
    reference = evidence["forecast"]["reference"]
    weight = np.asarray(evidence["forecast"]["weight"], dtype=float)
    estimates = []
    if len(requests) > 8:
        raise ValueError("analysis exceeds eight-request budget")
    for request in requests:
        result = results[request.experiment_id]
        if result.status != "done":
            continue
        if (
            len(request.request.arms) != 2
            or request.request.arms[0].actions
            or len(request.request.arms[1].actions) != 1
        ):
            raise ValueError("not a classical singleton-versus-control experiment")
        action = request.request.arms[1].actions[0]
        arms = {a.arm_index: [by_id[e] for e in a.episode_ids] for a in result.arms}
        if set(arms) != {0, 1} or not request.request.paired_starts:
            raise ValueError("paired arms 0 and 1 required")
        if any(len(arm) != request.request.episodes_per_arm for arm in arms.values()):
            raise ValueError("completed arm size differs from request")
        if any(not window[0] <= r["day"] <= window[1] for arm in arms.values() for r in arm):
            raise ValueError("arm episode outside the experiment window")
        if any(
            r.get("experiment_id") != request.experiment_id for arm in arms.values() for r in arm
        ):
            raise ValueError("arm episode carries a different experiment ID")
        weights = {
            robot: float(
                sum(w for r, w in zip(reference, weight, strict=True) if r["robot_id"] == robot)
            )
            for robot in request.request.robots
        }
        measured = arm_estimate(arms[0], arms[1], weights, seed)
        estimates.append(
            dict(
                action=action.model_dump(mode="json"),
                experiment_id=request.experiment_id,
                episode_ids=[r["episode_id"] for r in arms[1]],
                **measured,
            )
        )
    answer = revise(original, estimates, data.costs, requests=len(analysis.requests()))
    answer.evidence["experiment_input_sha256"] = data.hashes
    answer.evidence["request_sha256"] = hashlib.sha256(
        (analysis.outbox / "experiments.jsonl").read_bytes()
    ).hexdigest()
    answer.evidence["result_sha256"] = hashlib.sha256(
        (analysis.results / "experiments.jsonl").read_bytes()
    ).hexdigest()
    return answer


def revise(
    original: Answer,
    estimates: list[dict[str, Any]],
    costs: dict[CostKey, float | None],
    *,
    requests: int = 8,
    condition: str = "experiments",
) -> Answer:
    """Re-rank from explicitly supplied estimates. Offline oracle probes must
    say condition='oracle-proxy'; that is never a measured A/B condition.
    """
    if condition not in {"experiments", "oracle-proxy"}:
        raise ValueError("unknown estimate source")
    evidence = deepcopy(original.evidence)
    records = evidence["actions"]
    candidates = [Action.model_validate(r["action"]) for r in records]
    forecast = evidence["forecast"]
    x = np.asarray(forecast["matrix"], dtype=float)
    beta = np.asarray(evidence["coefficients"], dtype=float)
    nuisance = int(forecast["nuisance"])
    weight = np.asarray(forecast["weight"], dtype=float)
    reference = forecast["reference"]
    # Transport all singleton probabilities into a bounded logistic joint
    # forecast. Each action has its own offset column, so overlapping positive
    # repairs add on log odds instead of overwriting one another.
    base_probability = np.clip(
        predict(x, beta, nuisance, evidence["selected_link"]), 1e-8, 1 - 1e-8
    )
    log_odds = np.log(base_probability / (1 - base_probability))
    old_repairs = [np.asarray(r, dtype=float) for r in forecast["repaired"]]
    x = np.zeros((len(reference), len(candidates) + 1))
    x[:, 0] = log_odds
    beta = np.ones(x.shape[1])
    nuisance = x.shape[1]
    repairs = []
    for i, old in enumerate(old_repairs):
        probability = np.clip(
            predict(
                old,
                np.asarray(evidence["coefficients"]),
                int(forecast["nuisance"]),
                evidence["selected_link"],
            ),
            1e-8,
            1 - 1e-8,
        )
        if "offsets" in forecast:
            probability = shifted(probability, np.asarray(forecast["offsets"][i]))
        repair = x.copy()
        repair[:, i + 1] = np.log(probability / (1 - probability)) - log_odds
        repairs.append(repair)
    causes = list(original.diagnosis.causes)
    trials = []
    combined = [
        combine_estimates(records[candidates.index(Action.model_validate(m["action"]))], m)
        for m in estimates
    ]
    for measured, pooled in zip(estimates, combined, strict=True):
        action = Action.model_validate(measured["action"])
        index = candidates.index(action)
        covered = np.asarray(
            [scope_covers(action.target.scope, r["robot_id"], r["site_id"]) for r in reference],
            dtype=float,
        )
        repairs[index], clipped = calibrate(
            x, beta, nuisance, weight, covered, index, pooled["gain_points"]
        )
        p0 = base_probability
        p1 = predict(repairs[index], beta, nuisance, "logistic")
        variance = np.maximum(0, np.minimum(p0 + p1, 2 - p0 - p1) - (p1 - p0) ** 2)
        variance *= covered
        scoring_n = (
            evidence["observed_episodes"] * SCORING_LENGTH / evidence.get("observed_days", 6)
        )
        radius = Z90 * np.sqrt(float(weight @ variance) / scoring_n) * 100
        population = population_bound(weight, covered)
        records[index].update(
            gain_points=pooled["gain_points"],
            low=max(-population, pooled["low"] - radius),
            high=min(population, pooled["high"] + radius),
            prediction_radius=float(radius),
            estimation_variance=pooled["variance"],
            diagnosis_confidence=pooled["confidence"],
            informative_interval=True,
            experimental=condition == "experiments",
            estimate_source=condition,
            latent_candidate=False,
            aliases=[],
            signature=None,
        )
        trials.append(
            dict(
                **measured,
                combined=pooled,
                forecast_clipped=clipped,
            )
        )
    # Apply exactly the logs-only screen to the current (pooled) evidence.
    # Alternative repair verbs share a target: retain its strongest supported
    # confidence, never the stale confidence from before measurement.
    accepted = named_actions(records)
    retained = []
    for cause in causes:
        alternatives = [i for i, a in enumerate(candidates) if a.target == cause.target]
        supported = [i for i in alternatives if i in accepted]
        if not alternatives:  # robot-only recording signature, not a repair claim
            retained.append(cause)
        elif supported:
            best = max(supported, key=lambda i: records[i]["diagnosis_confidence"])
            retained.append(
                cause.model_copy(
                    update={
                        "confidence": records[best]["diagnosis_confidence"],
                        "failure_mode": cause.failure_mode
                        + "; confidence uses current pooled repair evidence",
                    }
                )
            )
    causes = retained
    for i in sorted(accepted):
        action = candidates[i]
        if any(c.target == action.target for c in causes):
            continue
        measured = next((m for m in estimates if m["action"] == records[i]["action"]), {})
        causes.append(
            Cause(
                target=action.target,
                direction=Direction.changed
                if action.target.config_key or action.target.condition_key
                else Direction.degraded,
                confidence=records[i]["diagnosis_confidence"],
                evidence=[
                    EvidenceRef(episode_id=e, step=None, table=None, query=None)
                    for e in measured.get("episode_ids", [])[:3]
                ],
                failure_mode=(
                    f"{condition}: current repair evidence passes the common naming screen; "
                    "transport to scoring assumed"
                ),
            )
        )
    joint = JointPrediction(x, beta, nuisance, "logistic", repairs, weight)
    order, selection = joint.order(
        [cost_of(a, costs) or 0 for a in candidates], [tie_key(a) for a in candidates]
    )
    predictions = [
        RankedAction(
            action=candidates[i],
            expected_gain=records[i]["gain_points"],
            uncertainty=Interval(low=records[i]["low"], high=records[i]["high"]),
            rationale=(
                (
                    f"{condition} estimate; see evidence for sampling uncertainty; "
                    "joint ranking calibrates the outcome model, assuming effects transport "
                    "from experiment to scoring window."
                )
                if records[i].get("estimate_source")
                else "Unexperimented logs-only estimate: " + original.evidence["method"]
            ),
        )
        for i in order
    ]
    ranked = RankedFixSet(actions=predictions)
    evidence.update(
        condition=condition,
        experiment_rule=RULE,
        experiments=trials if condition == "experiments" else [],
        oracle_estimates=trials if condition == "oracle-proxy" else [],
        requests=requests if condition == "experiments" else 0,
        completed_requests=len(trials) if condition == "experiments" else 0,
        oracle_lookups=len(trials) if condition == "oracle-proxy" else 0,
        selection=selection,
    )
    diagnosis = original.diagnosis.model_copy(
        update={"causes": sorted(causes, key=lambda c: -c.confidence)}
    )
    return Answer(
        diagnosis, ranked, [ranked.as_fix_set(b, costs) for b in SCORING_BUDGETS], evidence
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["plan", "update"])
    parser.add_argument("export", type=Path)
    parser.add_argument("answer", type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--episodes-per-arm", type=int, default=60)
    args = parser.parse_args()
    answer = read_answer(args.answer)
    if args.operation == "plan":
        print(
            plan(
                args.export,
                answer,
                args.analysis,
                requests=args.requests,
                episodes_per_arm=args.episodes_per_arm,
            )
        )
    else:
        if args.out is None:
            parser.error("update requires --out NEW_DIRECTORY")
        update(args.export, answer, args.analysis).write(args.out)


if __name__ == "__main__":
    main()
