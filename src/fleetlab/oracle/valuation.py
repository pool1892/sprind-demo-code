"""Valuing fix sets through the oracle table: the contract's lookup with its gaps caught, a
cached lookup for many random draws, the best removal set under a cost budget, and the
filled catalogue the random draws come from.

Moved here from `fleetlab.scoring.valuation` (which re-exports it) so that the oracle
and the scorer value through one implementation. Draw selection and individual gains
are shared; distribution aggregation is the caller's responsibility.

The random draws (`random_draws`): one permutation of the complete catalogue entries per
draw from `numpy.random.default_rng(seed)`; its first k actions are the count budgets'
draw (k distinct actions, uniformly), `cut` of the whole permutation the cost budgets'.
A draw that needs a fresh replay, or whose actions cannot be resolved, retains its
place as a missing result; the report bounds the complete population.

The filled catalogue (`filled_catalogue`): the world's catalogue with a value in every
entry that takes one: `set` writes the fleet-wide value of its key, so that a site or
robot runs the fleet's standard setting (at fleet scope it changes nothing);
`change_condition` keeps the site's recorded condition; `request_training_data` names no
particular mode. A rule an operator could follow from the visible records alone (the
miniature fleet's choice).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np
from pydantic import ValidationError

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import (
    COSTS,
    Action,
    Budget,
    CostTable,
    FixSet,
    Resolution,
    catalogue,
    cost_of,
    cut,
    resolve,
)
from fleetlab.contracts.faults import FaultSpec, Target, TruthTable
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.model import Fleet, effective_config
from fleetlab.contracts.oracle import (
    CurrentValues,
    Gain,
    MissingReplays,
    OracleTable,
    ReplayNeed,
    value_fix_set,
)
from fleetlab.contracts.vocabulary import VERBS, ConditionKey, ConfigKey, Verb
from fleetlab.oracle.mechanism import mechanism_resolution, mechanism_table

__all__ = [
    "FRESH",
    "HOURS_SLACK",
    "DrawResult",
    "DrawValuer",
    "RemovalOptions",
    "Valued",
    "best_removal",
    "cheapest_fix",
    "complete_actions",
    "current_values",
    "filled_catalogue",
    "missing_text",
    "points",
    "random_draws",
    "removal_options",
    "value_or_missing",
]

# Sums of hours like 0.1 + 0.2 are not exact in binary; a budget is met within this.
HOURS_SLACK = 1e-9


def points(net: int, episodes: int) -> float:
    """A net gain in points of fleet success over the scoring window."""
    return 100.0 * net / episodes


def missing_text(needs: Sequence[ReplayNeed]) -> str:
    """The reason a valuation is missing, in the report's words."""
    episodes = len({n.episode_id for n in needs})
    return f"MissingReplays: {episodes} episodes need a fresh replay ({len(needs)} replays)"


class Valued(NamedTuple):
    """A gain, or why there is none, and the replays it waits for."""

    gain: Gain | None
    missing: str | None
    needs: list[ReplayNeed]


def value_or_missing(
    fix_set: FixSet, oracle: OracleTable, truth: TruthTable, current: CurrentValues | None
) -> Valued:
    """`value_fix_set`, with `MissingReplays` and resolution errors caught and reported.

    An action that adds a change is a lookup like any other once its fresh
    replay's row is in the table (contracts batch 9), so "run the
    listed replays, then score again" values it.
    """
    try:
        return Valued(value_fix_set(fix_set, mechanism_table(oracle), truth, current), None, [])
    except MissingReplays as exc:
        needs = list(exc.needs)
        return Valued(None, missing_text(needs), needs)
    except (ValueError, KeyError) as exc:
        return Valued(None, f"cannot be resolved: {exc}", [])


FRESH: Literal["fresh"] = "fresh"
DrawResult = Gain | Literal["fresh"] | str


class DrawValuer:
    """`value_or_missing` for many fix sets drawn from one catalogue, with each action
    resolved once per episode.

    `resolve_all` resolves every action of a fix set against the episode's own
    faults and unites the results, so the resolution of a fix set is the union of
    its actions' resolutions and can be cached per action. The result is
    `value_or_missing`'s: the same gain, "fresh" where it misses replays, and the
    error text where the resolution raises ValueError or KeyError.
    """

    def __init__(
        self, oracle: OracleTable, truth: TruthTable, current: CurrentValues | None
    ) -> None:
        self.oracle = mechanism_table(oracle)
        self.truth = truth
        self._values: list[Mapping[ConfigKey | ConditionKey, ConfigValue] | None] = [
            None if current is None else current(identity) for identity in oracle.identities
        ]
        self._cache: dict[str, dict[int, Resolution] | str] = {}

    def effects(self, action: Action) -> dict[int, Resolution] | str:
        """Per episode index, what the action changes; an error text if it cannot resolve."""
        key = action.model_dump_json()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        changes: dict[int, Resolution] = {}
        effects: dict[int, Resolution] | str = changes
        try:
            for index, identity in enumerate(self.oracle.identities):
                result = resolve(action, identity, identity.faults, self.truth, self._values[index])
                result = mechanism_resolution(result)
                if not result.is_no_change:
                    changes[index] = result
        except (ValueError, KeyError) as exc:
            effects = str(exc)
        self._cache[key] = effects
        return effects

    def value(self, actions: Iterable[Action]) -> DrawResult:
        removed: dict[int, set[str]] = defaultdict(set)
        added: dict[int, dict[str, FaultSpec]] = defaultdict(dict)
        for action in actions:
            effects = self.effects(action)
            if isinstance(effects, str):
                return effects
            for index, result in effects.items():
                removed[index] |= result.removed
                for change in result.added:
                    added[index].setdefault(change.fault_id, change)
        rescued = broken = 0
        for index, gone in removed.items():
            identity = self.oracle.identities[index]
            before = self.oracle.success(identity.episode_id, identity.faults)
            remaining = [f for f in identity.faults if f.fault_id not in gone]
            after = self.oracle.success(identity.episode_id, [*remaining, *added[index].values()])
            if before is None or after is None:
                return FRESH
            rescued += (not before) and after
            broken += before and not after
        return Gain.of(rescued, broken)


def complete_actions(actions: Sequence[Action]) -> list[Action]:
    """The actions that can stand in a fix set: every value filled."""
    return [a for a in actions if a.is_complete]


def random_draws(
    oracle: OracleTable,
    truth: TruthTable,
    current: CurrentValues | None,
    entries: Sequence[Action],
    budgets: Sequence[Budget],
    *,
    seed: int,
    draws: int,
    costs: CostTable = COSTS,
) -> list[list[DrawResult]]:
    """Per budget, the result of each random draw (module docstring)."""
    results: list[list[DrawResult]] = [[] for _ in budgets]
    valuer = DrawValuer(oracle, truth, current)
    for sets in random_fix_sets(entries, budgets, seed=seed, draws=draws, costs=costs):
        for b, fix_set in enumerate(sets):
            results[b].append(valuer.value(fix_set.actions))
    return results


def random_fix_sets(
    entries: Sequence[Action],
    budgets: Sequence[Budget],
    *,
    seed: int,
    draws: int,
    costs: CostTable = COSTS,
) -> Iterator[list[FixSet]]:
    """The exact seeded draws, exposed so their missing joint replays can be planned.
    One permutation per draw, cut at each budget in the supplied order."""
    pool = complete_actions(entries)
    if not pool:
        return
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        shuffled = [pool[int(i)] for i in rng.permutation(len(pool))]
        yield [cut(shuffled, budget, costs) for budget in budgets]


# The best removal set under a budget.

RemovalOptions = dict[frozenset[str], dict[frozenset[str], Gain]]


def removal_options(oracle: OracleTable) -> tuple[RemovalOptions, list[ReplayNeed]]:
    """Per group of episodes with the same toggled faults, the gain of removing each
    non-empty subset of them; and every lookup that is missing.

    The same grouping as `contracts.oracle.achievable_gain`, from the table's
    public lookups.
    """
    groups: dict[frozenset[str], list[EpisodeIdentity]] = defaultdict(list)
    for identity in oracle.identities:
        toggled = frozenset(identity.fault_ids - oracle.excluded)
        if toggled:
            groups[toggled].append(identity)
    options: RemovalOptions = {}
    needs: list[ReplayNeed] = []
    for toggled, members in groups.items():
        per_set: dict[frozenset[str], Gain] = {}
        for size in range(1, len(toggled) + 1):
            for chosen in itertools.combinations(sorted(toggled), size):
                remove = frozenset(chosen)
                rescued = broken = 0
                for identity in members:
                    before = oracle.success(identity.episode_id, identity.faults)
                    remaining = tuple(f for f in identity.faults if f.fault_id not in remove)
                    after = oracle.success(identity.episode_id, remaining)
                    if before is None:
                        needs.append(ReplayNeed(identity.episode_id, tuple(identity.faults)))
                    if after is None:
                        needs.append(ReplayNeed(identity.episode_id, remaining))
                    if before is not None and after is not None:
                        rescued += (not before) and after
                        broken += before and not after
                per_set[remove] = Gain.of(rescued, broken)
        options[toggled] = per_set
    return options, needs


def best_removal(
    options: RemovalOptions, costs: Mapping[str, float], hours: float
) -> tuple[Gain, tuple[str, ...]]:
    """The best net gain of removing a set of faults whose costs fit `hours`.

    `costs` holds each candidate fault's cost; every subset that fits is tried,
    smaller sets first, and the first set with the highest net wins (as in
    `achievable_gain`).
    """
    ids = sorted(costs)
    best = Gain()
    best_set: tuple[str, ...] = ()
    for size in range(1, len(ids) + 1):
        for chosen in itertools.combinations(ids, size):
            if sum(costs[c] for c in chosen) > hours + HOURS_SLACK:
                continue
            picked = frozenset(chosen)
            total = Gain()
            for toggled, per_set in options.items():
                hit = picked & toggled
                if hit:
                    total = total + per_set[hit]
            if total.net > best.net:
                best, best_set = total, chosen
    return best, best_set


_VERB_ORDER = {verb: i for i, verb in enumerate(Verb)}


def cheapest_fix(
    fault: FaultSpec, truth: TruthTable, costs: CostTable
) -> tuple[Action, float] | None:
    """The cheapest action that removes the fault wherever it is active, and its hours.

    The fault type's fixing verbs that apply to the fault's own target (the
    component, or the key at the fault's scope); `set` and `change_condition`
    put the fault's nominal value back. Ties go to the verb listed first in the
    vocabulary. None if no fixing verb applies (a failure mode no action fixes).
    """
    mapping = truth.get(fault.kind)
    if mapping is None:
        return None
    found: list[tuple[float, int, Action]] = []
    for verb in mapping.fixing_verbs:
        info = VERBS[verb]
        if info.exact_scope and not mapping.writes_change:
            continue
        value = fault.nominal if info.takes_value else None
        try:
            action = Action(verb=verb, target=fault.target, value=value)
            hours = cost_of(action, costs)
        except (ValidationError, KeyError):
            continue
        found.append((hours or 0.0, _VERB_ORDER[verb], action))
    if not found:
        return None
    hours, _, action = min(found, key=lambda t: (t[0], t[1]))
    return action, hours


# The catalogue the random draws come from, and the resolver's current values.


def _fleet_value(fleet: Fleet, target: Target) -> ConfigValue:
    """What `set` writes: the fleet-wide value of the key, so that the scope runs the
    fleet's standard setting (at fleet scope: no change)."""
    key = target.config_key
    assert key is not None
    return fleet.config.values[key]


def filled_catalogue(fleet: Fleet, task_classes: Sequence[str]) -> list[Action]:
    """The world's catalogue with a value in every entry that takes one (module docstring)."""
    out: list[Action] = []
    for action in catalogue(fleet, task_classes):
        if action.is_complete:
            out.append(action)
            continue
        value: ConfigValue
        if action.verb is Verb.set:
            value = _fleet_value(fleet, action.target)
        elif action.verb is Verb.change_condition:
            assert action.target.condition_key is not None
            recorded = fleet.site(action.target.scope).conditions
            if action.target.condition_key not in recorded:
                continue  # a condition the site does not record has no value to keep
            value = recorded[action.target.condition_key]
        elif action.verb is Verb.request_training_data:
            value = "any failure mode"
        else:
            continue
        out.append(Action(verb=action.verb, target=action.target, value=value))
    return out


def current_values(
    fleet: Fleet,
) -> Callable[[EpisodeIdentity], Mapping[ConfigKey | ConditionKey, ConfigValue]]:
    """`resolve`'s `current`: a robot's effective configuration and its site's conditions."""

    def current(identity: EpisodeIdentity) -> Mapping[ConfigKey | ConditionKey, ConfigValue]:
        values: dict[ConfigKey | ConditionKey, ConfigValue] = {}
        values.update(effective_config(fleet, identity.robot_id).items())
        values.update(fleet.site(identity.site_id).conditions.items())
        return values

    return current
