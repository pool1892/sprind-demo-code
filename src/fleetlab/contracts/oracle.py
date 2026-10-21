"""The oracle: per-episode outcomes under subsets of the active faults, and the gains
that follow (docs/design/contracts.md, section 9).

Each scoring episode is replayed under the full power set of its effective
active faults (decoys and inert confounder entries stay in force and are not
toggled) when it has at most four (`POWER_SET_CAP`): at most 15 replays, the
unmodified run providing the full set. Any fix set that only removes faults,
and the achievable gain at any budget k, are then exact lookups. Episodes above
the cap, and actions that add a change, need a fresh replay the first time;
once its row is in the table it is a lookup like any other.

These functions are pure: they operate on an in-memory `OracleTable` of
`OracleRow`s that the engine fills, and on a truth table of fault types. A
missing row is never guessed: the functions raise `MissingReplays`, which lists
what has to be simulated, and `replays_needed` says the same in advance. Rows
only come from runs whose unmodified sample reproduced its hash chains; a run
with a broken chain contributes none (the engine's job, `chain_ok` in run.json).

Signs: a `Gain` counts episodes that turn from failure to success (`rescued`)
and from success to failure (`broken`); `net` is their difference. The gain of
a fault is the gain of removing it, so a harmful fault has a positive net and a
fault that rescues an episode contributes to `broken` (gate 1 saw 2 of 40).

Choices where the note is silent: the effective fault list is hashed as the
canonical JSON of its `FaultSpec`s sorted by fault id; `achievable_gain`
takes as the resolved fix of a fault its removal wherever it is active, and
considers every set of at most k such fixes over the faults the power set
toggles.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

from pydantic import model_validator

from fleetlab.contracts._base import ConfigValue, ContractModel
from fleetlab.contracts.actions import FixSet, Resolution, resolve_all
from fleetlab.contracts.faults import FaultSpec, TruthTable
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.ids import EpisodeId
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey

POWER_SET_CAP = 4


def fault_list_hash(faults: Iterable[FaultSpec]) -> str:
    """The canonical hash of an effective fault list: the specs sorted by id, as JSON."""
    specs = sorted((f.model_dump(mode="json") for f in faults), key=lambda d: str(d["fault_id"]))
    return hash_json(specs)


class OracleRow(ContractModel):
    """One replay's outcome: an episode under an effective fault list (by its hash)."""

    episode_id: EpisodeId
    active_faults: str  # fault_list_hash of the effective fault list in the replay
    success: bool


class Gain(ContractModel):
    """Episodes rescued and broken over a set of episodes, and the net."""

    rescued: int = 0
    broken: int = 0
    net: int = 0

    @model_validator(mode="after")
    def _net(self) -> Gain:
        if self.net != self.rescued - self.broken:
            raise ValueError("net is rescued minus broken")
        return self

    @classmethod
    def of(cls, rescued: int, broken: int) -> Gain:
        return cls(rescued=rescued, broken=broken, net=rescued - broken)

    def __add__(self, other: Gain) -> Gain:
        return Gain.of(self.rescued + other.rescued, self.broken + other.broken)


class ReplayNeed(NamedTuple):
    """An episode that has to be simulated under this effective fault list."""

    episode_id: str
    faults: tuple[FaultSpec, ...]


class MissingReplays(LookupError):
    """The oracle table lacks rows a valuation needs. `needs` lists them."""

    def __init__(self, needs: Sequence[ReplayNeed]) -> None:
        self.needs = list(needs)
        shown = ", ".join(n.episode_id for n in self.needs[:5])
        more = "" if len(self.needs) <= 5 else f" and {len(self.needs) - 5} more"
        super().__init__(f"{len(self.needs)} replays missing: {shown}{more}")


def power_set_replays(
    identity: EpisodeIdentity, excluded: Iterable[str] = (), cap: int = POWER_SET_CAP
) -> list[tuple[FaultSpec, ...]] | None:
    """The effective fault lists to replay one episode under, or None above the cap.

    Every subset of the effective active faults (those not in `excluded`)
    except the full one, which the original run provides; excluded faults stay
    in force in every replay. 2**n - 1 lists for n effective faults.
    """
    leave_out = set(excluded)
    toggled = [f for f in identity.faults if f.fault_id not in leave_out]
    if len(toggled) > cap:
        return None
    fixed = [f for f in identity.faults if f.fault_id in leave_out]
    out: list[tuple[FaultSpec, ...]] = []
    for size in range(len(toggled)):
        for kept in itertools.combinations(toggled, size):
            keep = {f.fault_id for f in kept} | {f.fault_id for f in fixed}
            out.append(tuple(f for f in identity.faults if f.fault_id in keep))
    return out


class OracleTable:
    """The scoring episodes of a world with their replay outcomes.

    `identities` are the scoring episodes as run (each with its active
    faults); `excluded` the faults the power set does not toggle. Rows are
    keyed by episode and effective fault list; two rows with the same key and
    different outcomes are refused, because replay is exact.
    """

    def __init__(
        self,
        identities: Iterable[EpisodeIdentity],
        rows: Iterable[OracleRow] = (),
        excluded: Iterable[str] = (),
    ) -> None:
        self.identities: tuple[EpisodeIdentity, ...] = tuple(identities)
        ids = [i.episode_id for i in self.identities]
        if len(ids) != len(set(ids)):
            raise ValueError("an episode appears twice in the oracle table")
        self.excluded: frozenset[str] = frozenset(excluded)
        self._episode_ids = frozenset(ids)
        self._rows: dict[tuple[str, str], bool] = {}
        for row in rows:
            self.add(row)

    def add(self, row: OracleRow) -> None:
        """Add a replay's outcome; refuses an episode the table does not hold."""
        if row.episode_id not in self._episode_ids:
            raise ValueError(f"episode {row.episode_id} is not a scoring episode of this table")
        key = (row.episode_id, row.active_faults)
        if key in self._rows and self._rows[key] != row.success:
            raise ValueError(f"episode {row.episode_id}: two replays of one fault list disagree")
        self._rows[key] = row.success

    def success(self, episode_id: str, faults: Iterable[FaultSpec]) -> bool | None:
        """The outcome of an episode under an effective fault list, or None if not replayed."""
        return self._rows.get((episode_id, fault_list_hash(faults)))

    def resolution(
        self,
        fix_set: FixSet,
        identity: EpisodeIdentity,
        truth: TruthTable,
        values: Mapping[ConfigKey | ConditionKey, ConfigValue] | None,
    ) -> Resolution:
        """Resolve changes; engine-backed tables can omit guaranteed inert additions."""
        return resolve_all(fix_set.actions, identity, identity.faults, truth, values)

    def over_cap(self, cap: int = POWER_SET_CAP) -> list[str]:
        """Episodes with more effective faults than the power set covers."""
        return [
            i.episode_id
            for i in self.identities
            if sum(f.fault_id not in self.excluded for f in i.faults) > cap
        ]

    def restricted(self, identities: Iterable[EpisodeIdentity]) -> OracleTable:
        """A table of some of the episodes that shares this table's rows."""
        sub = OracleTable(identities, excluded=self.excluded)
        sub._rows = self._rows
        return sub

    def fault_ids(self) -> list[str]:
        """Every fault active in some scoring episode, sorted."""
        return sorted({f.fault_id for i in self.identities for f in i.faults})


def _outcomes(
    table: OracleTable, changes: Iterable[tuple[EpisodeIdentity, tuple[FaultSpec, ...]]]
) -> tuple[list[tuple[bool, bool]], list[ReplayNeed]]:
    """(before, after) per changed episode, and the replays that are missing."""
    pairs: list[tuple[bool, bool]] = []
    missing: list[ReplayNeed] = []
    for identity, remaining in changes:
        before = table.success(identity.episode_id, identity.faults)
        after = table.success(identity.episode_id, remaining)
        if before is None:
            missing.append(ReplayNeed(identity.episode_id, tuple(identity.faults)))
        if after is None:
            missing.append(ReplayNeed(identity.episode_id, remaining))
        if before is not None and after is not None:
            pairs.append((before, after))
    return pairs, missing


def _gain(pairs: Iterable[tuple[bool, bool]]) -> Gain:
    rescued = broken = 0
    for before, after in pairs:
        rescued += (not before) and after
        broken += before and not after
    return Gain.of(rescued, broken)


CurrentValues = Callable[[EpisodeIdentity], Mapping[ConfigKey | ConditionKey, ConfigValue]]


def _fix_set_changes(
    fix_set: FixSet, table: OracleTable, truth: TruthTable, current: CurrentValues | None
) -> list[tuple[EpisodeIdentity, tuple[FaultSpec, ...]]]:
    """Per changed episode, the effective fault list the fix set leaves.

    An added change is looked up like any other list: the hash of
    the effective list identifies the replay, and the added change's id comes
    from its action, so a fresh replay is needed only while its row is absent.
    """
    changes: list[tuple[EpisodeIdentity, tuple[FaultSpec, ...]]] = []
    for identity in table.identities:
        values = None if current is None else current(identity)
        result = table.resolution(fix_set, identity, truth, values)
        if result.is_no_change:
            continue  # nothing removed, nothing added: no simulation needed
        remaining = tuple(f for f in identity.faults if f.fault_id not in result.removed)
        changes.append((identity, remaining + result.added))
    return changes


def replays_needed(
    fix_set: FixSet, table: OracleTable, truth: TruthTable, current: CurrentValues | None = None
) -> list[ReplayNeed]:
    """The replays `value_fix_set` would miss: fault lists, with or without added changes,
    whose row the table does not hold yet."""
    _, missing = _outcomes(table, _fix_set_changes(fix_set, table, truth, current))
    return missing


def value_fix_set(
    fix_set: FixSet, table: OracleTable, truth: TruthTable, current: CurrentValues | None = None
) -> Gain:
    """The signed gain of a fix set over the table's episodes, all its actions in force.

    Resolves the actions per episode, then looks up the remaining fault list,
    added changes included. Raises `MissingReplays` for every list whose row
    the table does not hold yet; once the engine has added those rows, the same
    call values the set.
    `current` gives each episode's current values (its robot's effective
    configuration and its site's conditions); `resolve` needs them for a `set`
    or `change_condition` of a key no active fault touches.
    """
    pairs, missing = _outcomes(table, _fix_set_changes(fix_set, table, truth, current))
    if missing:
        raise MissingReplays(missing)
    return _gain(pairs)


def _removal_gain(table: OracleTable, remove: frozenset[str]) -> Gain:
    changes = [
        (identity, tuple(f for f in identity.faults if f.fault_id not in remove))
        for identity in table.identities
        if identity.fault_ids & remove
    ]
    pairs, missing = _outcomes(table, changes)
    if missing:
        raise MissingReplays(missing)
    return _gain(pairs)


def fault_effect(fault_id: str, table: OracleTable) -> Gain:
    """The signed gain of removing one fault wherever it is active."""
    return _removal_gain(table, frozenset({fault_id}))


def interaction(fault_a: str, fault_b: str, table: OracleTable) -> int:
    """Joint net gain of removing both faults minus the sum of removing each alone.

    Over the episodes where both are active. Negative when the two together do
    more harm than their separate effects add up to.
    """
    both = table.restricted(i for i in table.identities if {fault_a, fault_b} <= i.fault_ids)
    joint = _removal_gain(both, frozenset({fault_a, fault_b})).net
    alone = (
        _removal_gain(both, frozenset({fault_a})).net
        + _removal_gain(both, frozenset({fault_b})).net
    )
    return joint - alone


def achievable_gain(k: int, table: OracleTable) -> Gain:
    """The best gain any set of at most k resolved fixes reaches, by net.

    The resolved fix of a fault removes it wherever it is active; candidates
    are the faults the power set toggles. Raises `MissingReplays` if a needed
    lookup is missing (for instance an episode above the cap).
    """
    if k < 0:
        raise ValueError("k is a non-negative number of actions")
    candidates = [f for f in table.fault_ids() if f not in table.excluded]
    # Group episodes by their toggled faults; per group, the gain of each removal set.
    groups: dict[frozenset[str], list[EpisodeIdentity]] = defaultdict(list)
    for identity in table.identities:
        toggled = frozenset(identity.fault_ids - table.excluded)
        if toggled:
            groups[toggled].append(identity)
    per_group: dict[frozenset[str], dict[frozenset[str], Gain]] = {}
    for toggled, members in groups.items():
        sub = table.restricted(members)
        options: dict[frozenset[str], Gain] = {}
        for size in range(1, len(toggled) + 1):
            for removed in itertools.combinations(sorted(toggled), size):
                options[frozenset(removed)] = _removal_gain(sub, frozenset(removed))
        per_group[toggled] = options
    best = Gain()
    for size in range(1, min(k, len(candidates)) + 1):
        for chosen in itertools.combinations(candidates, size):
            picked = frozenset(chosen)
            total = Gain()
            for toggled, options in per_group.items():
                hit = picked & toggled
                if hit:
                    total = total + options[hit]
            if total.net > best.net:
                best = total
    return best
