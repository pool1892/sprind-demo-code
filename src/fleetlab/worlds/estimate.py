"""Expected success from measured task contexts, with historical ratio models retained.

New worlds use absolute context cells, robot-window floors, shared-cell removal
resampling and intervals retaining development-world discrepancies. Missing or
thin joint and counterfactual cells are refusals, never products of single ratios.

The historical model (an assumption, stated in the README): success is the
healthy rate of its suite times the ratio of each fault in force on it
(`fleetlab.worlds.curves.Ratio`), faults acting independently, except the two members
of a marked interaction or additive pair, which use their joint point when both act.
Decoys count as ratio 1 only on a suite with confirmed flat calibration; an inert
calibration record has ratio 1 by construction. Other neutral candidates and small
causes retain their measured ratio, capped at 1 (no gain credited to noise).

`removal_effect` is what the detectability rule asks of a detectable cause: the fleet
success, in points of the window's episodes, that removing the faults alone gives back,
with every other fault left in force. For a data fault it is computed on the logged
outcome, for every other fault on the true one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.worlds.curves import (
    CellKey,
    Curves,
    MeasuredCell,
    Ratio,
    calibrated_label,
    context_weights,
)

if TYPE_CHECKING:
    from fleetlab.worlds.base import Slot


class RobotDay(TypedDict):
    robot_id: str
    day: int
    true: float
    logged: float


class Episode(NamedTuple):
    """What the estimate needs of an episode: its suite and the faults in force."""

    suite: str
    faults: tuple[FaultSpec, ...]
    task_index: int | None = None

    @property
    def fault_ids(self) -> frozenset[str]:
        return frozenset(f.fault_id for f in self.faults)

    @classmethod
    def of(cls, identity: EpisodeIdentity) -> Episode:
        return cls(identity.suite, tuple(identity.faults), identity.task_index)


@dataclass(frozen=True)
class Context:
    """What the estimate needs beyond the faults: roles, pairs, the curves."""

    curves: Curves
    neutral: frozenset[str]  # decoys and inert entries: ratio 1
    small: frozenset[str]  # effective faults at a small-cause point: ratio capped at 1
    pairs: tuple[tuple[str, str], ...] = ()  # measured interaction or additive pairs by fault id
    task_aware: bool = False
    measured: dict[CellKey, MeasuredCell] | None = None
    # Engineering reserves beyond the design thresholds; not confidence intervals.
    success_margin: float = 0.03
    cause_margin_points: float = 0.5
    # Which retained error a forecast's acceptance interval carries (`RETAINED_ERRORS`).
    retained_error: str = "generation-3"
    # A composer shares one table over its whole search; its cells never change. A
    # context built by hand has none, so that a test may alter its cells between calls.
    weights_cache: dict[tuple[Any, ...], Any] | None = field(default=None, compare=False)
    _cache: dict[tuple[str, float, str, int | None], Ratio] = field(
        default_factory=dict, compare=False
    )

    def weights(
        self,
        faults: Sequence[FaultSpec],
        suite: str,
        task_index: int | None,
        signature: tuple[Any, ...] | None = None,
    ) -> dict[CellKey, float]:
        """The measured cells a context rests on; KeyError when one is missing or thin.

        `signature` is the faults' `mechanism_signature`s when the caller has them.
        """
        if self.measured is None:
            raise ValueError("measured contexts required")
        if self.weights_cache is None:
            return context_weights(faults, suite, task_index, self.measured)
        if signature is None:
            signature = tuple(mechanism_signature(f) for f in faults)
        key = (suite, task_index, signature)
        found = self.weights_cache.get(key)
        if found is None:
            try:
                found = context_weights(faults, suite, task_index, self.measured)
            except KeyError as error:
                found = error
            self.weights_cache[key] = found
        if isinstance(found, KeyError):
            raise found
        return cast(dict[CellKey, float], found)

    def calibration(self, suite: str, task_index: int | None) -> Curves:
        return self.curves.for_task(suite, task_index) if self.task_aware else self.curves

    def ratio(self, fault: FaultSpec, suite: str, task_index: int | None = None) -> Ratio:
        key = (fault.fault_id, fault.level, suite, task_index)
        found = self._cache.get(key)
        if found is None:
            found = self.calibration(suite, task_index).ratio(fault, suite)
            if self.task_aware:
                from dataclasses import replace

                found = replace(found, source=f"{found.source}; task_index={task_index}")
            self._cache[key] = found
        return found


def mechanism_signature(fault: FaultSpec) -> tuple[Any, ...]:
    """What decides a fault's measured context: its type, dose, parameters and target thing.

    Neither its identity nor its scope or onset: two faults alike in these are one
    mechanism, and the cells are keyed by mechanisms.
    """
    thing = fault.target.thing
    return (
        fault.kind,
        fault.level,
        repr(fault.params),
        repr(fault.nominal),
        None if thing is None else thing.value,
    )


def member_key(fault: FaultSpec) -> tuple[str, str]:
    label = calibrated_label(fault)
    if label is None:
        raise KeyError(f"interaction member {fault.fault_id} is not at a calibrated point")
    return (fault.kind, label)


def probability(
    faults: Sequence[FaultSpec],
    suite: str,
    ctx: Context,
    logged: bool = False,
    task_index: int | None = None,
) -> float:
    """The chance an episode with these faults in force succeeds (logged: as the fleet logs)."""
    if ctx.measured is not None:
        weights = ctx.weights(faults, suite, task_index)
        return sum(weight * ctx.measured[key].rate(logged) for key, weight in weights.items())
    calibration = ctx.calibration(suite, task_index)
    p = calibration.healthy[suite]
    by_id = {f.fault_id: f for f in faults}
    joined: set[str] = set()
    for a, b in ctx.pairs:
        if a in by_id and b in by_id and not ({a, b} & joined):
            joint, _ = calibration.joint(member_key(by_id[a]), member_key(by_id[b]), suite)
            p *= min(1.0, joint.ratio) if ctx.task_aware else joint.ratio
            joined |= {a, b}
    for fault in faults:
        if fault.fault_id in joined:
            continue
        if fault.fault_id in ctx.neutral:
            if fault.kind == "calibration_update":
                continue  # no mechanism, on any task suite
            label = calibrated_label(fault)
            point = ctx.curves.points.get((fault.kind, suite, label or ""))
            if point is not None and point.role == "decoy" and point.n >= 1000:
                continue  # confirmed flat on this suite only
        r = ctx.ratio(fault, suite, task_index)
        value = r.logged if logged else r.true
        if ctx.task_aware or fault.fault_id in ctx.small or fault.fault_id in ctx.neutral:
            value = min(value, 1.0)
        p *= value
    return min(1.0, max(0.0, p))


def mean_success(episodes: Sequence[Episode], ctx: Context, logged: bool = False) -> float:
    """The expected success of a set of episodes, a share."""
    if not episodes:
        return float("nan")
    return sum(probability(e.faults, e.suite, ctx, logged, e.task_index) for e in episodes) / len(
        episodes
    )


def removal_effect(
    removed: Iterable[str], episodes: Sequence[Episode], ctx: Context, logged: bool
) -> float:
    """Points of the episodes' fleet success that removing these faults gives back."""
    if not episodes:
        return 0.0
    gone = set(removed)
    total = 0.0
    for e in episodes:
        if not gone & e.fault_ids:
            continue
        kept = [f for f in e.faults if f.fault_id not in gone]
        total += probability(kept, e.suite, ctx, logged, e.task_index) - probability(
            e.faults, e.suite, ctx, logged, e.task_index
        )
    return 100.0 * total / len(episodes)


def carriers(episodes: Iterable[Episode], fault_ids: Iterable[str]) -> int:
    """How many episodes carry at least one of these faults."""
    wanted = set(fault_ids)
    return sum(1 for e in episodes if wanted & e.fault_ids)


def suite_mix(episodes: Iterable[Episode], fault_ids: Iterable[str]) -> dict[str, int]:
    wanted = set(fault_ids)
    out: dict[str, int] = {}
    for e in episodes:
        if wanted & e.fault_ids:
            out[e.suite] = out.get(e.suite, 0) + 1
    return out


def is_data(fault: FaultSpec) -> bool:
    """Whether a fault is of the data class: it changes the logged outcome, not the true one."""
    import fleetlab.faults as faults
    from fleetlab.contracts.vocabulary import FaultClass

    return faults.get(fault.kind).fault_class is FaultClass.data


def robot_day_success(
    schedule: Sequence[FaultSpec], slots: Sequence[Slot], ctx: Context
) -> list[RobotDay]:
    """Each robot-day's expected success on its actual task mix, with all faults in force."""
    from fleetlab.worlds import base

    groups: dict[tuple[str, int], list[Episode]] = {}
    for slot in slots:
        assert isinstance(slot, base.Slot)
        faults = tuple(base.in_force(schedule, slot.robot_id, slot.site_id, slot.day))
        groups.setdefault((slot.robot_id, slot.day), []).append(
            Episode(slot.suite, faults, slot.task.task_index)
        )
    return [
        RobotDay(
            robot_id=robot,
            day=day,
            true=mean_success(eps, ctx),
            logged=mean_success(eps, ctx, logged=True),
        )
        for (robot, day), eps in sorted(groups.items())
    ]


def drift_crossing_day(fault: FaultSpec, slots: Sequence[Slot], ctx: Context) -> int | None:
    """First day with >=5 points of calibrated harm from this drift alone on its scope."""
    from fleetlab.worlds import base

    if fault.schedule.drift is None:
        return None
    by_day: dict[int, list[Episode]] = {}
    for slot in slots:
        assert isinstance(slot, base.Slot)
        if fault.target.covers(slot.robot_id, slot.site_id):
            active = tuple(base.in_force([fault], slot.robot_id, slot.site_id, slot.day))
            by_day.setdefault(slot.day, []).append(
                Episode(slot.suite, active, slot.task.task_index)
            )
    for day, eps in sorted(by_day.items()):
        if removal_effect([fault.fault_id], eps, ctx, logged=is_data(fault)) >= 5.0:
            return day
    return None


def prediction(
    episodes: Sequence[Episode], ctx: Context, *, draws: int = 512, window: str = "observed"
) -> dict[str, Any]:
    """A deterministic, nominal 95% prediction interval, conditional on the suite model.

    Independent Jeffreys-binomial calibration posteriors form a sensitivity model,
    not a paired bootstrap: the curve record does not contain per-attempt outcomes.
    A draw is shared by all episodes using that point (not independent per episode).
    The predictive interval also draws the finite window's outcomes. Neither interval
    covers task heterogeneity, correlated robot failures or model misspecification.
    """
    from dataclasses import replace

    import numpy as np

    if ctx.measured is not None:
        return measured_prediction(episodes, ctx, draws=draws, window=window)
    if not episodes:
        return {"episodes": 0, "true": None, "logged": None}
    rng = np.random.default_rng(20261009)
    grouped: dict[tuple[str, int | None, tuple[tuple[str, float], ...]], tuple[Episode, int]] = {}
    for ep in episodes:
        key = (
            ep.suite,
            ep.task_index if ctx.task_aware else None,
            tuple((f.fault_id, f.level) for f in ep.faults),
        )
        grouped[key] = (ep, grouped.get(key, (ep, 0))[1] + 1)
    calibration: dict[str, list[float]] = {"true": [], "logged": []}
    predictive: dict[str, list[float]] = {"true": [], "logged": []}
    n = len(episodes)
    for _ in range(draws):
        points = {}
        for point_key, point in ctx.curves.points.items():
            healthy = (
                float(
                    rng.beta(point.healthy_successes + 0.5, point.n - point.healthy_successes + 0.5)
                )
                * point.n
            )
            successes = (
                float(rng.beta(point.successes + 0.5, point.n - point.successes + 0.5)) * point.n
            )
            logged = (
                successes
                if point.logged_successes == point.successes
                else float(
                    rng.beta(point.logged_successes + 0.5, point.n - point.logged_successes + 0.5)
                )
                * point.n
            )
            points[point_key] = replace(
                point, successes=successes, logged_successes=logged, healthy_successes=healthy
            )
        joints = tuple(
            replace(
                joint,
                successes=float(rng.beta(joint.successes + 0.5, joint.n - joint.successes + 0.5))
                * joint.n,
                healthy_successes=float(
                    rng.beta(joint.healthy_successes + 0.5, joint.n - joint.healthy_successes + 0.5)
                )
                * joint.n,
            )
            for joint in ctx.curves.joints
        )
        healthy_rates = dict(ctx.curves.healthy)
        for suite, (successes, trials) in ctx.curves.healthy_counts.items():
            healthy_rates[suite] = float(rng.beta(successes + 0.5, trials - successes + 0.5))
        sampled = replace(ctx.curves, points=points, joints=joints, healthy=healthy_rates)
        if ctx.task_aware:
            sampled = replace(
                sampled, tasks={key: _sample_tasks(c, rng) for key, c in ctx.curves.tasks.items()}
            )
        draw_ctx = replace(ctx, curves=sampled, _cache={})
        for outcome in ("true", "logged"):
            expected = realized = 0.0
            for ep, count in grouped.values():
                p = probability(
                    ep.faults,
                    ep.suite,
                    draw_ctx,
                    logged=outcome == "logged",
                    task_index=ep.task_index,
                )
                expected += count * p
                realized += int(rng.binomial(count, p))
            calibration[outcome].append(expected / n)
            predictive[outcome].append(realized / n)
    out: dict[str, Any] = {
        "episodes": n,
        "draws": draws,
        "coverage": 0.95,
        "method": "independent-Jeffreys-binomial calibration sensitivity "
        "plus independent-attempt sampling",
        "limits": "conditional on suite ratios; excludes paired covariance, task heterogeneity, "
        "within-robot dependence and model misspecification; not validated coverage",
    }
    if ctx.task_aware:
        out["limits"] = (
            "conditional on task-specific ratios; excludes paired covariance, "
            "within-robot dependence and model misspecification; not validated coverage"
        )
    for outcome in ("true", "logged"):
        out[outcome] = {
            "expected": round(mean_success(episodes, ctx, logged=outcome == "logged"), 4),
            "calibration_sd_points": round(float(np.std(calibration[outcome])) * 100, 3),
            "calibration_interval": [
                round(float(v), 4) for v in np.quantile(calibration[outcome], [0.025, 0.975])
            ],
            "prediction_interval": [
                round(float(v), 4) for v in np.quantile(predictive[outcome], [0.025, 0.975])
            ],
        }
    return out


def _sample_tasks(curves: Curves, rng: Any) -> Curves:
    """Share each calibration draw across every scheduled use of that task cell."""
    from dataclasses import replace

    def draw(success: float, n: int) -> float:
        return float(rng.beta(success + 0.5, n - success + 0.5)) * n

    points = {}
    for key, p in curves.points.items():
        successes = draw(p.successes, p.n)
        points[key] = replace(
            p,
            successes=successes,
            logged_successes=successes
            if p.logged_successes == p.successes
            else draw(p.logged_successes, p.n),
            healthy_successes=draw(p.healthy_successes, p.n),
        )
    joints = tuple(
        replace(
            j, successes=draw(j.successes, j.n), healthy_successes=draw(j.healthy_successes, j.n)
        )
        for j in curves.joints
    )
    healthy = {suite: draw(s, n) / n for suite, (s, n) in curves.healthy_counts.items()}
    return replace(curves, points=points, joints=joints, healthy=healthy)


class RobotWindow(TypedDict):
    robot_id: str
    window: str
    episodes: int
    true: float
    logged: float


def robot_window_success(
    schedule: Sequence[FaultSpec], slots: Sequence[Slot], ctx: Context, windows: Any
) -> list[RobotWindow]:
    """Decision 2026-10-10: average the scheduled attempts in EACH full window."""
    from fleetlab.worlds import base

    groups: dict[tuple[str, str], list[Episode]] = {}
    for slot in slots:
        for name in ("observed", "experiment", "scoring"):
            if slot.day in getattr(windows, name):
                faults = tuple(base.in_force(schedule, slot.robot_id, slot.site_id, slot.day))
                groups.setdefault((slot.robot_id, name), []).append(
                    Episode(slot.suite, faults, slot.task.task_index)
                )
    return [
        RobotWindow(
            robot_id=robot,
            window=window,
            episodes=len(eps),
            true=mean_success(eps, ctx),
            logged=mean_success(eps, ctx, True),
        )
        for (robot, window), eps in sorted(groups.items())
    ]


# The two calibration worlds cease to be validation worlds when their cells enter
# the model. Their model discrepancy is therefore retained, not refitted away.
MODEL_ERROR = (-0.007, 0.060)
MODEL_ERRORS = {
    "observed": MODEL_ERROR,
    "experiment": (-0.037, 0.066),
    "scoring": (-0.048, 0.054),
}
# The retained error a recipe names (`WorldRecipe.retained_error`). "generation-3" is what
# the two calibration worlds showed and stays the default. "none" is for worlds that
# measure the forecast's error anew, the fourth-generation development worlds: their
# acceptance interval is the forecast's own spread. What those worlds show is to become a
# further entry, for the test worlds.
RETAINED_ERRORS: dict[str, dict[str, tuple[float, float]]] = {
    "generation-3": MODEL_ERRORS,
    "none": dict.fromkeys(MODEL_ERRORS, (0.0, 0.0)),
    # Entry 20: the largest fresh error, pooled over w48/w49 and all three windows,
    # moved OUTWARD at both ends. No fitted correction to the point forecast.
    "generation-4": dict.fromkeys(MODEL_ERRORS, (-0.021, 0.021)),
}


def measurement_weights(
    episodes: Sequence[Episode], ctx: Context, removed: Iterable[str] = ()
) -> dict[CellKey, float]:
    """Shared cell coefficients of a window; also used for every one-fault removal."""
    if ctx.measured is None or not episodes:
        raise ValueError("nonempty window and measured contexts required")
    gone = set(removed)
    out: dict[CellKey, float] = {}
    for ep in episodes:
        weights = ctx.weights(
            [f for f in ep.faults if f.fault_id not in gone], ep.suite, ep.task_index
        )
        for key, value in weights.items():
            out[key] = out.get(key, 0) + value / len(episodes)
    return out


def _cell_samples(
    keys: Sequence[CellKey], ctx: Context, draws: int, rng: Any, logged: bool = False
) -> Any:
    import numpy as np

    assert ctx.measured is not None
    return np.array(
        [
            rng.beta(
                (s := (ctx.measured[k].logged_successes if logged else ctx.measured[k].successes))
                + 0.5,
                ctx.measured[k].n - s + 0.5,
                size=draws,
            )
            for k in keys
        ]
    )


def measured_prediction(
    episodes: Sequence[Episode],
    ctx: Context,
    *,
    draws: int = 2048,
    band: tuple[float, float] = (0.60, 0.75),
    window: str = "observed",
) -> dict[str, Any]:
    """Shared-cell sensitivity plus finite-run noise AND retained model discrepancy.

    The probability for the band is the fraction of resamples inside it for every
    error in the retained range (for the third generation's observed window, -0.7 to
    +6.0 points). No probability distribution for two errors is invented. Which
    retained error applies is the context's (`RETAINED_ERRORS`); with "none" the
    `prediction_interval` is the `conditional_prediction_interval`, rounded outward.
    """
    import math

    import numpy as np

    error = RETAINED_ERRORS[ctx.retained_error][window]

    weights = measurement_weights(episodes, ctx)
    keys = sorted(weights)
    coefficients = np.array([weights[k] for k in keys])
    rng = np.random.default_rng(20261010)
    out: dict[str, Any] = {
        "episodes": len(episodes),
        "window": window,
        "draws": draws,
        "coverage": 0.95,
        "method": "shared task-context Jeffreys resamples; independent attempt sampling; "
        "retained discrepancy envelope",
        "model_error_points": [round(100 * x, 3) for x in error],
        "limits": "not validated frequentist coverage; independent calibration cells; "
        "unmodelled robot dependence; w46/w47 now calibration, not validation",
    }
    if ctx.retained_error != "generation-3":
        out["retained_error"] = ctx.retained_error
    for logged, name in ((False, "true"), (True, "logged")):
        sampled = _cell_samples(keys, ctx, draws, rng, logged)
        means = coefficients @ sampled
        # Using a binomial with the window mean is conservative relative to the
        # independent Bernoulli sum with unequal task probabilities (larger variance).
        predictive = rng.binomial(len(episodes), means) / len(episodes)
        interval = np.quantile(predictive, [0.025, 0.975])
        widened = [
            max(0.0, float(interval[0]) + error[0]),
            min(1.0, float(interval[1]) + error[1]),
        ]
        out[name] = {
            "expected": round(mean_success(episodes, ctx, logged), 4),
            "calibration_sd_points": round(float(np.std(means)) * 100, 3),
            "calibration_interval": [
                round(float(x), 4) for x in np.quantile(means, [0.025, 0.975])
            ],
            "conditional_prediction_interval": [round(float(x), 4) for x in interval],
            "prediction_interval": [
                math.floor(widened[0] * 10000) / 10000,
                math.ceil(widened[1] * 10000) / 10000,
            ],
            "band_probability_for_all_recorded_errors": float(
                np.mean((predictive + error[0] >= band[0]) & (predictive + error[1] <= band[1]))
            ),
        }
    return out


def measured_causes(
    episodes: Sequence[Episode],
    ctx: Context,
    interactions: Sequence[tuple[str, str]] = (),
    *,
    draws: int = 2048,
    minimum: int = 6,
) -> dict[str, Any]:
    """Scoring-window forecast of measured removals; additive pairs get no fallback.

    Each cell draw is shared by the original world and every counterfactual. Two
    qualified interaction members are two causes; neither qualified permits the
    pair fallback. Overlapping pairs are evaluated separately, never merged.
    """
    import numpy as np

    ids = sorted(
        {
            f.fault_id
            for ep in episodes
            for f in ep.faults
            if f.kind != "calibration_update" and not is_data(f)
        }
    )
    baseline = measurement_weights(episodes, ctx)
    removals = {fid: measurement_weights(episodes, ctx, [fid]) for fid in ids}
    pairs = {pair: measurement_weights(episodes, ctx, pair) for pair in interactions}
    keys = sorted(set(baseline).union(*(set(w) for w in [*removals.values(), *pairs.values()])))
    rng = np.random.default_rng(20261011)
    sampled = _cell_samples(keys, ctx, draws, rng)
    assert ctx.measured is not None
    rates = np.array([ctx.measured[k].rate() for k in keys])

    def gains(weights: dict[CellKey, float]) -> tuple[Any, float]:
        coefficients = np.array([100 * (weights.get(k, 0) - baseline.get(k, 0)) for k in keys])
        return coefficients @ sampled, float(coefficients @ rates)

    gains_by_id = {fid: gains(weights) for fid, weights in removals.items()}
    qualified = {fid: samples >= 2.0 for fid, (samples, _) in gains_by_id.items()}
    counts = sum(qualified.values(), start=np.zeros(draws, dtype=int))
    point_count = sum(gain >= 2 for _, gain in gains_by_id.values())
    for pair, weights in pairs.items():
        samples, gain = gains(weights)
        a, b = pair
        counts += ~qualified[a] & ~qualified[b] & (samples >= 2)
        point_count += int(gains_by_id[a][1] < 2 and gains_by_id[b][1] < 2 and gain >= 2)
    return {
        "window": "scoring",
        "threshold_points": 2,
        "draws": draws,
        "count": point_count,
        "minimum": minimum,
        "probability_at_least_minimum": float(np.mean(counts >= minimum)),
        "count_interval": [float(x) for x in np.quantile(counts, [0.025, 0.975])],
        "limits": "calibration-cell sensitivity of expected oracle gains; "
        "does not include finite-scoring-window or replay-pair covariance uncertainty",
        "faults": {
            fid: {
                "gain_points": round(gain, 4),
                "interval_points": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
                "probability_at_least_two": float(np.mean(samples >= 2)),
            }
            for fid, (samples, gain) in gains_by_id.items()
        },
    }
