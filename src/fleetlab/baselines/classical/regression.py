"""Penalized binomial models, grouped validation and cluster uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def sigmoid(x: Array) -> Array:
    return np.asarray(np.exp(-np.logaddexp(0, -x)), dtype=float)


def fit(x: Array, y: Array, penalty: Array, weights: Array | None = None) -> Array:
    w = np.ones(len(y)) if weights is None else weights
    beta = np.zeros(x.shape[1])
    for _ in range(80):
        eta = x @ beta
        p = sigmoid(eta)
        grad = x.T @ (w * (p - y)) + penalty * beta
        hessian = x.T @ ((w * p * (1 - p))[:, None] * x) + np.diag(penalty)
        step = np.linalg.solve(hessian, grad)
        objective = float(np.sum(w * (np.logaddexp(0, eta) - y * eta)))
        objective += float(np.sum(penalty * beta**2) / 2)
        scale = 1.0
        for _ in range(30):
            candidate = beta - scale * step
            eta1 = x @ candidate
            after = float(np.sum(w * (np.logaddexp(0, eta1) - y * eta1)))
            after += float(np.sum(penalty * candidate**2) / 2)
            if after <= objective + 1e-10:
                break
            scale /= 2
        beta = candidate
        if float(np.max(np.abs(scale * step))) < 1e-7:
            return beta
    raise RuntimeError("logistic fit did not converge")


def predict(x: Array, beta: Array, nuisance: int, link: str) -> Array:
    if link == "logistic":
        return sigmoid(x @ beta)
    return np.asarray(
        sigmoid(x[:, :nuisance] @ beta[:nuisance]) * np.exp(x[:, nuisance:] @ beta[nuisance:]),
        dtype=float,
    )


def fit_survival(
    x: Array, y: Array, penalty: Array, nuisance: int, weights: Array | None = None
) -> Array:
    """Binomial likelihood, p(success) = logistic(baseline) * exp(-sum(hazards)).

    Nonnegative exposures have nonpositive log-survival coefficients. This is
    an explicit restoration prior, not evidence that actions can never hurt.
    It represents masking by multiple faults without a combinatorial search
    for named interactions. L-BFGS-B enforces the hazard constraints.
    """
    weights = np.ones(len(y)) if weights is None else weights

    scale = np.sqrt(np.maximum(np.sum(weights[:, None] * x * x, axis=0), 1.0))

    def objective(coordinates: Array) -> tuple[float, Array]:
        beta = coordinates / scale
        base = x[:, :nuisance] @ beta[:nuisance]
        logp = -np.logaddexp(0, -base) + x[:, nuisance:] @ beta[nuisance:]
        p = np.exp(logp)
        one_minus = np.maximum(-np.expm1(logp), 1e-15)
        loss = float(
            np.sum(weights * (-y * logp - (1 - y) * np.log(one_minus)))
            + np.sum(penalty * beta**2) / 2
        )
        factor = weights * (p - y) / one_minus
        gradient = (
            np.concatenate(
                [
                    x[:, :nuisance].T @ (factor * (1 - sigmoid(base))),
                    x[:, nuisance:].T @ factor,
                ]
            )
            + penalty * beta
        )
        return loss, gradient / scale

    result = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None)] * nuisance + [(None, 0.0)] * (x.shape[1] - nuisance),
        options={"maxiter": 2000, "maxcor": 50, "ftol": 1e-9, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"multiplicative fit failed: {result.message}")
    return np.asarray(result.x / scale, dtype=float)


@dataclass(frozen=True)
class Regression:
    link: str
    beta: Array
    draws: Array
    penalty_multiplier: float
    validation: dict[str, float]


def bootstrap_weights(
    clusters: list[str], rng: np.random.Generator, sites: list[str] | None = None
) -> Array:
    """Positive hierarchical multiplier bootstrap: no exposure cell disappears.

    Robot weights represent between-robot exposure uncertainty; robot-day
    weights represent transitions. Site weights add the shared site level.
    Unlike resampling clusters with replacement, even a single exposed day
    retains positive weight. This does not identify a fully confounded effect.
    """
    levels = [[c.split(":")[0] for c in clusters], clusters]
    if sites is not None:
        levels.insert(0, sites)
    weights = np.ones(len(clusters))
    for labels in levels:
        draws = {label: float(rng.exponential()) + 1e-8 for label in sorted(set(labels))}
        weights *= np.asarray([draws[label] for label in labels])
    return np.asarray(weights * len(weights) / weights.sum(), dtype=float)


def regress(
    x: Array,
    y: Array,
    penalty: Array,
    clusters: list[str],
    folds: list[int],
    *,
    bootstraps: int,
    seed: int,
    nuisance: int,
    sites: list[str] | None = None,
) -> Regression:
    """Select link and shrinkage multiplier using held-out robot-day groups.

    The intercept keeps its weak proper prior regardless of the multiplier.
    Positive multipliers vary site, robot and robot-day contributions;
    design columns and selected shrinkage remain fixed (conditional intervals).
    """
    validation: dict[str, float] = {}
    fold_array = np.asarray(folds)

    def estimate(
        x1: Array, y1: Array, penalties: Array, link: str, weights: Array | None = None
    ) -> Array:
        if link == "logistic":
            return fit(x1, y1, penalties, weights)
        return fit_survival(x1, y1, penalties, nuisance, weights)

    choices = [(link, a) for link in ("logistic", "multiplicative") for a in (0.25, 1.0, 4.0)]
    for link, multiplier in choices:
        penalties = penalty * multiplier
        penalties[0] = penalty[0]
        losses = []
        for fold in sorted(set(folds)):
            test = fold_array == fold
            if np.all(test):
                continue
            beta = estimate(x[~test], y[~test], penalties, link)
            p = np.clip(predict(x[test], beta, nuisance, link), 1e-12, 1 - 1e-12)
            losses.extend((-y[test] * np.log(p) - (1 - y[test]) * np.log1p(-p)).tolist())
        validation[f"{link}:{multiplier}"] = float(np.mean(losses)) if losses else 0.0
    link, multiplier = min(
        choices, key=lambda choice: (validation[f"{choice[0]}:{choice[1]}"], -choice[1], choice[0])
    )
    penalties = penalty * multiplier
    penalties[0] = penalty[0]
    beta = estimate(x, y, penalties, link)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstraps):
        weights = bootstrap_weights(clusters, rng, sites)
        samples.append(estimate(x, y, penalties, link, weights))
    return Regression(link, beta, np.asarray(samples), multiplier, validation)
