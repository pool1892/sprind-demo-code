"""The gains that follow from the oracle table, with their uncertainty.

Every gain is a sum over scoring episodes of a per-episode change: +1 for an episode
turned from failure to success (rescued), -1 for one turned from success to failure
(broken), 0 otherwise, so it is computed here as a vector over the scoring episodes. The
point values are the contract's (`contracts.oracle`: `fault_effect`, `interaction`,
`achievable_gain`; the tests hold them equal); the vectors add the **bootstrap over
scoring episodes**: `boot` resamples of the N scoring episodes with replacement (one
multinomial weight matrix per table, seeded), and for each gain the 2.5th and 97.5th
percentiles of its net in points (100 times the net over N). A best-of gain (the
achievable gain) is maximised again in every resample, so its interval is that of the
statistic "the best k-set's gain", not of one fixed set.

Items:

- `fault_effect`: removing one fault wherever it is active. Decoys and inert entries
  too, through the confirmation replays; for every fault also by day when it drifts.
- `interaction`: joint removal minus the two single removals, over the episodes that
  carry both (the contract's `interaction`), for the world's marked pairs and for every
  other pair that occurs together.
- `achievable`: at every count budget k (1 to 6, the scoring budgets among them) the
  contract's `achievable_gain(k)`; at every cost budget the best set of resolved fixes
  whose cheapest actions fit the hours (the scorer's rule: `cheapest_fix`, then every
  subset that fits, smaller first, first highest net wins).
- `all_faults`: every fault of every episode removed (decoys and inert entries
  included), the per-episode verdict claim 5 reads: failures of the policy itself on
  a healthy robot, the measure behind "needs new training data".

A lookup the table lacks never becomes a guess: the item is reported missing with the
replays it needs (`MissingReplays`), which `fleetlab.oracle.fresh` can run.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from fleetlab.contracts.actions import COSTS, SCORING_BUDGETS, Budget, CostTable
from fleetlab.contracts.faults import FaultSpec, TruthTable
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import Gain, OracleTable, ReplayNeed
from fleetlab.oracle.valuation import HOURS_SLACK, cheapest_fix

BOOT = 1000
LOW, HIGH = 2.5, 97.5
ACHIEVABLE_KS = (1, 2, 3, 4, 5, 6)
CHUNK = 1024  # candidate sets valued per matrix product in the bootstrap

Vector = npt.NDArray[np.int8]


class Estimate(BaseModel):
    """One gain: its counts over the scoring window, its points and its bootstrap interval."""

    model_config = ConfigDict(extra="forbid")

    item: str
    item_kind: str  # fault_effect, fault_effect_day, interaction, achievable, all_faults
    rescued: int | None = None
    broken: int | None = None
    net: int | None = None
    points: float | None = None
    low: float | None = None  # bootstrap percentile of the net in points
    high: float | None = None
    episodes: int = 0  # scoring episodes the item touches
    missing: str | None = None
    detail: dict[str, Any] = {}

    @property
    def zero_within_noise(self) -> bool | None:
        """Whether the interval holds zero (None while missing)."""
        if self.low is None or self.high is None:
            return None
        return self.low <= 0.0 <= self.high


def _missing(needs: Sequence[ReplayNeed]) -> str:
    episodes = len({n.episode_id for n in needs})
    return f"MissingReplays: {episodes} episodes need a fresh replay ({len(needs)} replays)"


@dataclass
class Gains:
    """The oracle's gains over one table (module docstring)."""

    table: OracleTable
    boot: int = BOOT
    seed: int = 0
    weights: npt.NDArray[np.float32] = field(init=False)
    _before: list[bool | None] = field(init=False)
    _cache: dict[tuple[int, frozenset[str]], bool | None] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.table.identities)
        if n and self.boot > 0:
            rng = np.random.default_rng(self.seed)
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=self.boot)
            self.weights = counts.astype(np.float32)
        else:
            self.weights = np.zeros((0, n), dtype=np.float32)
        self._before = [self.table.success(i.episode_id, i.faults) for i in self.table.identities]
        self._cache = {}

    @property
    def n(self) -> int:
        return len(self.table.identities)

    # ---- lookups ----------------------------------------------------------------------------

    def remaining(self, index: int, remove: frozenset[str]) -> tuple[FaultSpec, ...]:
        identity = self.table.identities[index]
        return tuple(f for f in identity.faults if f.fault_id not in remove)

    def after(self, index: int, remove: frozenset[str]) -> bool | None:
        key = (index, remove)
        if key not in self._cache:
            identity = self.table.identities[index]
            self._cache[key] = self.table.success(
                identity.episode_id, self.remaining(index, remove)
            )
        return self._cache[key]

    def change(self, index: int, remove: frozenset[str], needs: list[ReplayNeed]) -> int:
        """+1 rescued, -1 broken, 0 unchanged for one episode; a missing lookup goes to `needs`."""
        identity = self.table.identities[index]
        hit = remove & identity.fault_ids
        if not hit:
            return 0
        before = self._before[index]
        after = self.after(index, frozenset(hit))
        if before is None:
            needs.append(ReplayNeed(identity.episode_id, tuple(identity.faults)))
        if after is None:
            needs.append(ReplayNeed(identity.episode_id, self.remaining(index, frozenset(hit))))
        if before is None or after is None:
            return 0
        return int(after) - int(before)

    def removal(
        self, remove: Iterable[str], episodes: Iterable[int] | None = None
    ) -> tuple[Vector, list[ReplayNeed]]:
        """The per-episode change of removing these faults (over `episodes`, else all)."""
        gone = frozenset(remove)
        out = np.zeros(self.n, dtype=np.int8)
        needs: list[ReplayNeed] = []
        for index in range(self.n) if episodes is None else episodes:
            out[index] = self.change(index, gone, needs)
        return out, needs

    # ---- estimates --------------------------------------------------------------------------

    def interval(self, vector: npt.NDArray[Any]) -> tuple[float | None, float | None]:
        if self.weights.shape[0] == 0:
            return None, None
        nets = self.weights @ vector.astype(np.float32)
        low, high = np.percentile(100.0 * nets / self.n, [LOW, HIGH])
        return round(float(low), 3), round(float(high), 3)

    def estimate(
        self,
        item: str,
        kind: str,
        vector: Vector,
        needs: Sequence[ReplayNeed],
        touched: int,
        detail: Mapping[str, Any] | None = None,
    ) -> Estimate:
        if needs:
            return Estimate(
                item=item,
                item_kind=kind,
                episodes=touched,
                missing=_missing(needs),
                detail={**(detail or {}), "needs": len(needs)},
            )
        rescued = int((vector > 0).sum())
        broken = int((vector < 0).sum())
        low, high = self.interval(vector)
        return Estimate(
            item=item,
            item_kind=kind,
            rescued=rescued,
            broken=broken,
            net=rescued - broken,
            points=round(100.0 * (rescued - broken) / self.n, 3) if self.n else None,
            low=low,
            high=high,
            episodes=touched,
            detail=dict(detail or {}),
        )

    def carrying(self, *fault_ids: str) -> list[int]:
        want = set(fault_ids)
        return [i for i, e in enumerate(self.table.identities) if want <= e.fault_ids]

    def fault_effect(self, fault_id: str) -> Estimate:
        vector, needs = self.removal({fault_id})
        return self.estimate(fault_id, "fault_effect", vector, needs, len(self.carrying(fault_id)))

    def fault_effect_by_day(self, fault_id: str) -> list[Estimate]:
        by_day: dict[int, list[int]] = defaultdict(list)
        for index in self.carrying(fault_id):
            by_day[self.table.identities[index].day].append(index)
        out = []
        for day, members in sorted(by_day.items()):
            vector, needs = self.removal({fault_id}, members)
            out.append(
                self.estimate(
                    f"{fault_id}@{day}",
                    "fault_effect_day",
                    vector,
                    needs,
                    len(members),
                    {"fault_id": fault_id, "day": day},
                )
            )
        return out

    def interaction(self, a: str, b: str, marked: bool) -> Estimate:
        members = self.carrying(a, b)
        joint, n1 = self.removal({a, b}, members)
        alone_a, n2 = self.removal({a}, members)
        alone_b, n3 = self.removal({b}, members)
        needs = [*n1, *n2, *n3]
        term = joint.astype(np.int16) - alone_a - alone_b
        item = f"{a}+{b}"
        detail = {"marked": marked, "joint_net": int(joint.sum())}
        if needs:
            return self.estimate(item, "interaction", joint, needs, len(members), detail)
        if not members:
            return Estimate(
                item=item, item_kind="interaction", missing="no episode carries both", detail=detail
            )
        low, high = self.interval(term)
        summed_harm = 2 * joint.astype(np.int16) - alone_a - alone_b
        harm_low, harm_high = self.interval(summed_harm)
        value = int(term.sum())
        return Estimate(
            item=item,
            item_kind="interaction",
            rescued=0,
            broken=0,
            net=value,
            points=round(100.0 * value / self.n, 3) if self.n else None,
            low=low,
            high=high,
            episodes=len(members),
            detail={
                **detail,
                "alone_net": [int(alone_a.sum()), int(alone_b.sum())],
                "standalone_harm_net": [int((joint - alone_b).sum()), int((joint - alone_a).sum())],
                "summed_harm_net": int(joint.sum()) + value,
                "summed_harm_low": harm_low,
                "summed_harm_high": harm_high,
                "excess_se_points": (
                    float(np.std(term, ddof=1) * np.sqrt(self.n) * 100 / self.n)
                    if self.n > 1
                    else None
                ),
            },
        )

    def all_faults(self) -> tuple[Estimate, list[dict[str, Any]]]:
        """Every fault removed: the gain, and per scoring episode its two outcomes."""
        rows: list[dict[str, Any]] = []
        vector = np.zeros(self.n, dtype=np.int8)
        needs: list[ReplayNeed] = []
        for index, identity in enumerate(self.table.identities):
            everything = frozenset(identity.fault_ids)
            healthy = self.after(index, everything) if everything else self._before[index]
            if everything:
                vector[index] = self.change(index, everything, needs)
            rows.append(
                {
                    "episode_id": identity.episode_id,
                    "robot_id": identity.robot_id,
                    "site_id": identity.site_id,
                    "day": identity.day,
                    "faults": sorted(identity.fault_ids),
                    "toggled": sorted(identity.fault_ids - self.table.excluded),
                    "as_run_success": self._before[index],
                    "all_removed_success": healthy,
                }
            )
        touched = sum(1 for i in self.table.identities if i.faults)
        return self.estimate("all faults", "all_faults", vector, needs, touched), rows

    # ---- the best set at a budget -------------------------------------------------------------

    def _groups(self) -> dict[frozenset[str], list[int]]:
        groups: dict[frozenset[str], list[int]] = defaultdict(list)
        for index, identity in enumerate(self.table.identities):
            toggled = frozenset(identity.fault_ids - self.table.excluded)
            if toggled:
                groups[toggled].append(index)
        return groups

    def subset_changes(
        self, candidates: Sequence[str]
    ) -> tuple[dict[frozenset[str], dict[frozenset[str], Vector]], list[ReplayNeed]]:
        """Per group of episodes with the same toggled faults, the change vector (over the
        group's members) of removing each non-empty subset of its candidates."""
        allowed = frozenset(candidates)
        options: dict[frozenset[str], dict[frozenset[str], Vector]] = {}
        needs: list[ReplayNeed] = []
        for toggled, members in self._groups().items():
            usable = sorted(toggled & allowed)
            per: dict[frozenset[str], Vector] = {}
            for size in range(1, len(usable) + 1):
                for chosen in itertools.combinations(usable, size):
                    remove = frozenset(chosen)
                    per[remove] = np.array(
                        [self.change(i, remove, needs) for i in members], dtype=np.int8
                    )
            options[toggled] = per
        return options, needs

    def _column(
        self,
        picked: frozenset[str],
        options: Mapping[frozenset[str], Mapping[frozenset[str], Vector]],
        groups: Mapping[frozenset[str], list[int]],
    ) -> Vector:
        column = np.zeros(self.n, dtype=np.int8)
        for toggled, per in options.items():
            hit = picked & toggled
            if hit:
                column[groups[toggled]] = per[frozenset(hit)]
        return column

    def best(
        self,
        item: str,
        sets: Iterable[tuple[str, ...]],
        candidates: Sequence[str],
        detail: Mapping[str, Any],
    ) -> Estimate:
        """The best removal set among `sets` (in their order, first highest net wins, never
        below the empty set's zero), with the bootstrap of the best in every resample."""
        options, needs = self.subset_changes(candidates)
        groups = self._groups()
        touched = sum(len(m) for m in groups.values())
        if needs:
            return self.estimate(
                item, "achievable", np.zeros(self.n, np.int8), needs, touched, detail
            )
        best_net = 0
        best_set: tuple[str, ...] = ()
        best_column = np.zeros(self.n, dtype=np.int8)
        boot_best = np.zeros(self.weights.shape[0], dtype=np.float32)
        considered = 0
        for chunk in _chunks(sets, CHUNK):
            columns = np.stack([self._column(frozenset(s), options, groups) for s in chunk], axis=1)
            considered += len(chunk)
            nets = columns.astype(np.int32).sum(axis=0)
            top = int(np.argmax(nets))
            if int(nets[top]) > best_net:
                best_net, best_set, best_column = int(nets[top]), chunk[top], columns[:, top]
            if self.weights.shape[0]:
                boot_best = np.maximum(
                    boot_best, (self.weights @ columns.astype(np.float32)).max(axis=1)
                )
        out = self.estimate(
            item,
            "achievable",
            best_column,
            [],
            touched,
            {**detail, "faults": list(best_set), "sets_considered": considered},
        )
        if self.weights.shape[0]:
            low, high = np.percentile(100.0 * boot_best / self.n, [LOW, HIGH])
            out.low, out.high = round(float(low), 3), round(float(high), 3)
        return out

    def candidates(self) -> list[str]:
        return [f for f in self.table.fault_ids() if f not in self.table.excluded]

    def achievable_count(self, k: int) -> Estimate:
        ids = self.candidates()
        sets = (
            chosen
            for size in range(1, min(k, len(ids)) + 1)
            for chosen in itertools.combinations(ids, size)
        )
        return self.best(f"k={k}", sets, ids, {"budget": "count", "amount": k})

    def achievable_cost(
        self, hours: float, specs: Mapping[str, FaultSpec], truth: TruthTable, costs: CostTable
    ) -> Estimate:
        fixes: dict[str, tuple[str, float]] = {}
        left_out: list[str] = []
        for fault_id in self.candidates():
            found = cheapest_fix(specs[fault_id], truth, costs)
            if found is None:
                left_out.append(fault_id)
            else:
                action, cost = found
                fixes[fault_id] = (action.model_dump_json(), cost)
        ids = sorted(fixes)

        def fitting() -> Iterator[tuple[str, ...]]:
            for size in range(1, len(ids) + 1):
                for chosen in itertools.combinations(ids, size):
                    if sum(fixes[c][1] for c in chosen) <= hours + HOURS_SLACK:
                        yield chosen

        out = self.best(
            f"{hours:g} h",
            fitting(),
            ids,
            {"budget": "cost", "amount": hours, "no_fix": left_out},
        )
        chosen = out.detail.get("faults", [])
        out.detail["fixes"] = [fixes[f][0] for f in chosen]
        out.detail["hours"] = round(sum(fixes[f][1] for f in chosen), 3)
        return out

    def achievable(
        self,
        specs: Mapping[str, FaultSpec],
        truth: TruthTable,
        costs: CostTable = COSTS,
        ks: Sequence[int] = ACHIEVABLE_KS,
        budgets: Sequence[Budget] = SCORING_BUDGETS,
    ) -> list[Estimate]:
        """The achievable gain at k = 1 to 6 and at every cost budget of the scoring budgets."""
        out = [self.achievable_count(k) for k in ks]
        counted = set(ks)
        for budget in budgets:
            if budget.kind == "count" and int(budget.amount) not in counted:
                out.append(self.achievable_count(int(budget.amount)))
            elif budget.kind == "cost":
                out.append(self.achievable_cost(budget.amount, specs, truth, costs))
        return out


def _chunks(items: Iterable[tuple[str, ...]], size: int) -> Iterator[list[tuple[str, ...]]]:
    chunk: list[tuple[str, ...]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def fault_specs(identities: Iterable[EpisodeIdentity]) -> dict[str, FaultSpec]:
    """Every fault of the episodes by id, as the first episode carrying it lists it."""
    out: dict[str, FaultSpec] = {}
    for identity in identities:
        for fault in identity.faults:
            out.setdefault(fault.fault_id, fault)
    return out


def contract_gain(estimate: Estimate) -> Gain | None:
    """The estimate's counts as the contract's `Gain` (None while missing)."""
    if estimate.rescued is None or estimate.broken is None:
        return None
    return Gain.of(estimate.rescued, estimate.broken)
