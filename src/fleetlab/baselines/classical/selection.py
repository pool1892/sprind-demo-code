"""Joint predicted gains, with exhaustive three-action search and nested cuts."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from fleetlab.baselines.classical.onset import shifted
from fleetlab.baselines.classical.regression import Array, sigmoid


class JointPrediction:
    """Compact forecasts: each repair changes baseline log-odds and/or log-survival.

    Actions with overlapping changes to the same fitted feature are combined by
    their feature-wise union, not counted twice. This is the same nonlinear model
    used for singleton estimates. Unidentified alternatives are explicit priors.
    """

    def __init__(
        self,
        x: Array,
        beta: Array,
        nuisance: int,
        link: str,
        repaired: list[Array],
        weight: Array,
        offsets: Array | None = None,
    ):
        x, beta, weight = (np.asarray(v, dtype=float) for v in (x, beta, weight))
        self.x, self.beta, self.nuisance, self.link = x, beta, nuisance, link
        self.deltas = np.asarray(repaired, dtype=float).reshape(len(repaired), *x.shape) - x
        if not np.isfinite(self.deltas).all():
            raise ValueError("repair deltas must be finite")
        self.offsets = np.zeros((len(repaired), len(x))) if offsets is None else offsets
        self.weight = weight
        self.base = self._probability(x)
        # Decompose each signed feature union into threshold increments. An
        # increment contributes iff any selected action reaches its threshold.
        # Group identical action memberships across rows/features, then evaluate
        # combinations using tiny coverage matrices rather than [batch,k,row,col].
        # This preserves max(positive)+min(negative), including unequal overlaps.
        groups: dict[bytes, tuple[np.ndarray[Any, Any], Array, Array]] = {}
        for row in range(x.shape[0]):
            for column in np.flatnonzero(beta):
                values = self.deltas[:, row, column]
                for sign in (-1, 1):
                    signed = sign * values
                    previous = 0.0
                    for level in np.unique(signed[signed > 0]):
                        membership = signed >= level
                        key = membership.tobytes()
                        if key not in groups:
                            groups[key] = (membership, np.zeros(len(x)), np.zeros(len(x)))
                        part = 1 if link == "logistic" or column < nuisance else 2
                        groups[key][part][row] += sign * (level - previous) * beta[column]
                        previous = float(level)
        self.coverage = (
            np.column_stack([g[0] for g in groups.values()])
            if groups
            else np.zeros((len(repaired), 0), dtype=bool)
        )
        self.baseline_atoms = np.asarray([g[1] for g in groups.values()]).reshape(
            len(groups), len(x)
        )
        self.hazard_atoms = np.asarray([g[2] for g in groups.values()]).reshape(len(groups), len(x))
        stop = len(beta) if link == "logistic" else nuisance
        self.base_eta = x[:, :stop] @ beta[:stop]
        self.base_hazard = np.zeros(len(x)) if link == "logistic" else x[:, stop:] @ beta[stop:]

    def _probability(self, x: Array) -> Array:
        if self.link == "logistic":
            return sigmoid(x @ self.beta)
        return np.asarray(
            sigmoid(x[..., : self.nuisance] @ self.beta[: self.nuisance])
            * np.exp(x[..., self.nuisance :] @ self.beta[self.nuisance :]),
            dtype=float,
        )

    def gain(self, indices: tuple[int, ...]) -> float:
        if not indices:
            return 0.0
        return float(self._gains(np.asarray([indices], dtype=int))[0])

    def _gains(self, indices: np.ndarray[Any, Any]) -> Array:
        covered = self.coverage[indices].any(axis=1)
        probabilities = sigmoid(self.base_eta + covered @ self.baseline_atoms)
        if self.link != "logistic":
            probabilities *= np.exp(self.base_hazard + covered @ self.hazard_atoms)
        if np.any(self.offsets):
            probabilities = shifted(probabilities, self.offsets[indices].sum(axis=1))
        return np.asarray((probabilities - self.base) @ self.weight * 100, dtype=float)

    def best_extension(
        self,
        chosen: tuple[int, ...],
        available: list[int],
        size: int,
        costs: list[float],
        keys: list[str],
    ) -> tuple[int, ...]:
        if not available:
            return ()
        # Exhaustive, not an oracle-guided candidate shortlist. At most three
        # additions per search; every complete candidate remains eligible.
        # Batch vectorization keeps small fleets inexpensive.
        best: tuple[int, ...] = ()
        best_value = -float("inf")
        best_tie: tuple[float, tuple[str, ...]] = (float("inf"), ())
        iterator = combinations(available, min(size, len(available)))
        while True:
            batch = []
            for _ in range(256):
                item = next(iterator, None)
                if item is None:
                    break
                batch.append(item)
            if not batch:
                break
            indices = np.asarray([(*chosen, *item) for item in batch], dtype=int)
            values = self._gains(indices)
            for item, value in zip(batch, values, strict=True):
                tie = (sum(costs[i] for i in item), tuple(sorted(keys[i] for i in item)))
                if value > best_value + 1e-10 or (
                    abs(value - best_value) <= 1e-10 and tie < best_tie
                ):
                    best, best_value, best_tie = item, float(value), tie
        return best

    def _next(
        self, chosen: tuple[int, ...], available: list[int], costs: list[float], keys: list[str]
    ) -> int:
        values = {i: self.gain((*chosen, i)) for i in available}
        maximum = max(values.values())
        return min(
            (i for i in available if maximum - values[i] <= 1e-10),
            key=lambda i: (costs[i], keys[i]),
        )

    def order(
        self, costs: list[float], keys: list[str], length: int = 10
    ) -> tuple[list[int], dict[str, Any]]:
        # A single list cannot in general contain the unconstrained best set at
        # every k (optimal sets need not nest). Give the design's headline k=3
        # priority, then the exact best three additions conditional on that cut.
        # Include zero and adverse candidates too: an exactly sized cut must
        # consider its fillers, and a negative singleton can mask another repair.
        active = list(range(len(keys)))
        chosen: tuple[int, ...] = ()
        order = []
        for size in (3, 3):
            best = self.best_extension(
                chosen, [i for i in active if i not in chosen], size, costs, keys
            )
            remaining = list(best)
            while remaining:
                item = self._next(chosen, remaining, costs, keys)
                chosen = (*chosen, item)
                order.append(item)
                remaining.remove(item)
        while len(order) < min(length, len(keys)):
            available = [i for i in range(len(keys)) if i not in chosen]
            item = self._next(chosen, available, costs, keys)
            chosen = (*chosen, item)
            order.append(item)
        return order, {
            "rule": (
                "exact best triple; exact best three additions preserving triple; "
                "then marginal gain"
            ),
            "eligible": len(active),
            "predicted_prefix_gain_points": [
                self.gain(tuple(order[:k])) for k in range(1, len(order) + 1)
            ],
        }
