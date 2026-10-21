"""The retained state of a fleet: the ledger of adopted changes and what it means for a
robot on a day (docs/design/loop.md, sections 2 and 4).

    ledger = Ledger.empty("w62")                         # the original state, version 0
    ledger = ledger.adopt(1, 8, adopt=[Adoption(action=a, rationale="...")])
    ledger.active(9)                                     # the actions in force on day 9
    state = materialize(world_faults, where, ledger.active_steps(9), truth, current)
    state = apply_arm(state, arm_actions, truth)         # an experiment arm, as a delta

**The ledger.** Every adopted action is an entry with a unique id (`L1`, `L2`, ...), the
round that adopted it, its adoption day, the agent's public rationale, and a status. A
replacement or an undo retires an entry on the day the deciding round's decision takes
effect (`retired_day`); `active(day)` answers from adoption and retirement days, so an
earlier version's history stays exact. `adopt` returns a new version and never changes
the old one; every version is stored as `ledger-r<N>.json` (`save`).

**One interpretation of a state.** Operation, experiments and the evaluation all ask
`materialize` what a robot's state is on a day; an experiment arm is a delta on top of
that state (`apply_arm`). This replaces the union of independently resolved actions
(`resolve_all`), under which a retained `set` of a healthy key became a no-op and an arm
could neither override nor revert a retained action (a review of c3985fd).

The rules, for one robot on one day:

1. **Revert undoes the most recent change at exactly its scope and key.** If an earlier
   step made it (a `set`, or a revert), the two cancel: the state is as if neither had
   been taken. Otherwise the revert undoes the world's own recorded change there. A
   second revert therefore undoes the first; two reverts after a retained `set` undo
   the set and then the world's change.
2. **Configuration is layered.** Every scope covering the robot (the fleet, its site,
   the robot) keeps a stack: the fleet's base value, the world's own change at that scope
   (a fault, else a benign own value), then the steps' sets in order; a revert removes
   the world's change at its scope. The narrowest scope with something on its stack
   decides. A wider setting that a narrower one shadows is kept, and comes back when the
   narrower one is removed (a review of 2e151ff).
   **The precedence of what runs** is the order of the faults in force, which is the
   order their hooks run in: the world's own behaviour first (its faults in the world's
   order, a release that carries a setting included, less what the steps removed or a
   deciding setting shadows), then the ledger's retained changes in adoption order, then
   an experiment arm's in its order. A deciding `set` adds a `config_change` whose
   nominal is the value in force before it (for a policy setting, what the engine's hooks
   before it give; else the base value), so an explicit setting always takes effect,
   also against a release that carries the same setting, and runs no hook only where it
   changes nothing (a review of cc1ff68; checked against the engine's own `effective_spec`
   over a table of combinations).
3. **Component verbs and condition changes** are resolved in order against the faults
   the earlier ones left (`contracts.actions.resolve`): a component verb removes the
   faults of that component whose type lists the verb; `change_condition` overrides the
   site's condition. A fault no verb fixes (a lasting condition) stays.
4. **The order of the world and the steps.** A step meets the world's changes made
   before it; a world change on the same scope and key after a step's day is refused
   rather than guessed (no stored world has one), and a component or condition step
   leaves alone a fault that starts after it.
5. **Changes a ledger entry adds** carry the entry's own fault id (`entry_fault_id`) and
   its adoption day as onset, so two entries never share an id and the truth side says
   which entry put a change there; an arm's carry `contracts.actions.added_fault_id`.

The robot's values (`current`, configuration and conditions) come out of the same
layers, so a later `set` of a key no fault touches compares with the value in force after
the ledger, not with the world's; the records (`believed_fleet`) apply the same rules to
the change log, and a test checks for every robot and key that they agree.

**Conventions shared by the driver and the evaluator** (a review, point 3):

- A round's decision counts `adopt + replace` against `MAX_CHANGES_PER_ROUND`; an undo
  counts nothing (`decision_changes`).
- The number of changes of a state is the number of entries in force at its end
  (`changes_in_force`); replaced and undone entries do not count.
- The one-look plan is round 1's ranked list, adoptable actions only, cut at the final
  state's number of changes with `contracts.actions.cut`, adopted on round 1's adoption
  day (`one_look_ledger`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from fleetlab.contracts._base import ConfigValue, ContractModel
from fleetlab.contracts.actions import (
    ADDED_CHANGE_KINDS,
    ADDED_ID_BASE,
    CONFIG_CHANGE_KIND,
    Action,
    Budget,
    added_fault_id,
    cut,
    resolve,
)
from fleetlab.contracts.faults import FaultSpec, Schedule, TruthTable
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.identity import PolicySpec
from fleetlab.contracts.ids import (
    FLEET_SCOPE,
    IdKind,
    ScopeKind,
    neutral_id,
    scope_covers,
    scope_kind,
)
from fleetlab.contracts.model import (
    Change,
    ChangeAuthor,
    ConfigVersion,
    Fleet,
    Maintenance,
)
from fleetlab.contracts.vocabulary import (
    COMPONENT_VERBS,
    ConditionKey,
    ConfigKey,
    Verb,
)

__all__ = [
    "MAX_CHANGES_PER_ROUND",
    "Adoption",
    "Applied",
    "EntryStatus",
    "Ledger",
    "LedgerEntry",
    "Replacement",
    "RobotDay",
    "Step",
    "Undo",
    "apply_arm",
    "believed_fleet",
    "changes_in_force",
    "decision_changes",
    "entry_fault_id",
    "ledger_records",
    "materialize",
    "one_look_ledger",
]

# The most changes one round's decision may make (adopt + replace); an undo is free.
MAX_CHANGES_PER_ROUND = 3

EntryStatus = Literal["active", "replaced", "undone"]
Key = ConfigKey | ConditionKey
Values = Mapping[Key, ConfigValue]

_COMPONENT_NOTES: dict[Verb, str] = {
    Verb.recalibrate: "recalibrated",
    Verb.clean: "cleaned",
    Verb.replace_: "replaced",
    Verb.service: "serviced",
}


# ---- the ledger ------------------------------------------------------------------------------


class Adoption(ContractModel):
    """An action a round adopts, with the agent's public rationale."""

    action: Action
    rationale: str = ""


class Replacement(ContractModel):
    """An entry in force replaced by another action."""

    entry_id: str
    action: Action
    rationale: str = ""


class Undo(ContractModel):
    """An entry in force taken back."""

    entry_id: str
    rationale: str = ""


class LedgerEntry(ContractModel):
    """One adopted action. `entry_id` is unique over the ledger's whole history."""

    entry_id: str = Field(pattern=r"^L[1-9][0-9]*$")
    round: int = Field(ge=1)
    adopted_day: int = Field(ge=0)
    action: Action
    rationale: str = ""
    status: EntryStatus = "active"
    retired_day: int | None = Field(default=None, ge=0)
    retired_round: int | None = Field(default=None, ge=1)
    retire_rationale: str = ""
    replaced_by: str | None = None
    replaces: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> LedgerEntry:
        retired = self.status != "active"
        if retired != (self.retired_day is not None) or retired != (self.retired_round is not None):
            raise ValueError(f"{self.entry_id}: a retirement day and round exactly when retired")
        if self.retired_day is not None and self.retired_day <= self.adopted_day:
            raise ValueError(f"{self.entry_id}: retired on or before its adoption day")
        if (self.status == "replaced") != (self.replaced_by is not None):
            raise ValueError(f"{self.entry_id}: replaced_by exactly when replaced")
        if not self.action.is_complete or not self.action.has_replay:
            raise ValueError(f"{self.entry_id}: an adopted action is complete and simulated")
        return self

    def in_force_on(self, day: int) -> bool:
        """Adopted on or before `day` and not retired by then."""
        return self.adopted_day <= day and (self.retired_day is None or day < self.retired_day)


@dataclass(frozen=True)
class Step:
    """One action of a fold: who took it, on which day it took effect, what it is."""

    source: str  # a ledger entry id, or "arm"
    day: int
    action: Action


class Ledger(ContractModel):
    """The fleet's adopted changes. Immutable; `adopt` makes the next version.

    Version 0 is the original state (no entry). Version N is the ledger after round N's
    decision; a round that decides nothing still makes its version.
    """

    world_id: str
    version: int = Field(default=0, ge=0)
    entries: list[LedgerEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> Ledger:
        ids = [e.entry_id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("ledger entry ids repeat")
        if any(e.round > self.version for e in self.entries):
            raise ValueError("an entry is from a round after the ledger's version")
        return self

    @classmethod
    def empty(cls, world_id: str) -> Ledger:
        """The original state."""
        return cls(world_id=world_id)

    def entry(self, entry_id: str) -> LedgerEntry:
        for e in self.entries:
            if e.entry_id == entry_id:
                return e
        raise KeyError(f"no ledger entry {entry_id!r}")

    def active_entries(self, day: int) -> list[LedgerEntry]:
        """The entries in force on `day`, in adoption order (the order of the fold)."""
        return [e for e in self.entries if e.in_force_on(day)]

    def active(self, day: int) -> list[Action]:
        """The actions in force on `day`, in adoption order."""
        return [e.action for e in self.active_entries(day)]

    def active_steps(self, day: int) -> list[Step]:
        """The fold's steps for `day`: each entry in force, from its adoption day."""
        return [Step(e.entry_id, e.adopted_day, e.action) for e in self.active_entries(day)]

    def in_force(self) -> list[LedgerEntry]:
        """The entries in force at the end of the ledger (status active)."""
        return [e for e in self.entries if e.status == "active"]

    @property
    def last_day(self) -> int:
        """The latest adoption or retirement day; -1 for the empty ledger."""
        days = [e.adopted_day for e in self.entries]
        days += [e.retired_day for e in self.entries if e.retired_day is not None]
        return max(days, default=-1)

    def adopt(
        self,
        round: int,
        day: int,
        adopt: Sequence[Adoption | Action] = (),
        replace: Sequence[Replacement] = (),
        undo: Sequence[Undo | str] = (),
    ) -> Ledger:
        """Round `round`'s decision, in effect from `day`: the next version.

        Raises ValueError for a round out of order, a day before the ledger's last,
        an unknown or retired entry, an entry named twice, the undo or replacement of a
        component verb (a recalibration or a new part is physical and cannot be taken
        back), an action that is incomplete or not simulated, or an action already in
        force. The cap of `MAX_CHANGES_PER_ROUND` is the driver's to enforce
        (`decision_changes`): the ledger records what was decided.
        """
        if round != self.version + 1:
            raise ValueError(f"round {round} follows version {self.version}")
        if day <= self.last_day:
            raise ValueError(f"day {day} is not after the ledger's last day {self.last_day}")
        adoptions = [a if isinstance(a, Adoption) else Adoption(action=a) for a in adopt]
        undos = [u if isinstance(u, Undo) else Undo(entry_id=u) for u in undo]
        named = [r.entry_id for r in replace] + [u.entry_id for u in undos]
        if len(named) != len(set(named)):
            raise ValueError("an entry is replaced or undone twice in one decision")
        entries = {e.entry_id: e for e in self.entries}
        for entry_id in named:
            if entry_id not in entries:
                raise ValueError(f"no ledger entry {entry_id!r}")
            old = entries[entry_id]
            if old.status != "active":
                raise ValueError(f"{entry_id} is {old.status}, not in force")
            if old.action.verb in COMPONENT_VERBS:
                raise ValueError(
                    f"{entry_id} ({old.action.verb.value}) is physical and cannot be taken back"
                )
        retiring = set(named)
        staying = [
            e.action.model_dump_json()
            for e in self.entries
            if e.status == "active" and e.entry_id not in retiring
        ]
        new_actions = [a.action for a in adoptions] + [r.action for r in replace]
        dumped = [a.model_dump_json() for a in new_actions]
        if len(dumped) != len(set(dumped)):
            raise ValueError("a decision names an action twice")
        for action, text in zip(new_actions, dumped, strict=True):
            if not action.is_complete or not action.has_replay:
                raise ValueError(
                    f"{action.verb.value}: an adopted action is complete and simulated"
                )
            if text in staying:
                raise ValueError(
                    f"{action.verb.value} on {action.target.scope} is already in force"
                )
        number = len(self.entries)
        out: list[LedgerEntry] = []
        replacing = {r.entry_id: r for r in replace}
        new_ids: dict[str, str] = {}
        for r in replace:
            number += 1
            new_ids[r.entry_id] = f"L{number}"
        for e in self.entries:
            if e.entry_id in replacing:
                out.append(
                    e.model_copy(
                        update={
                            "status": "replaced",
                            "retired_day": day,
                            "retired_round": round,
                            "retire_rationale": replacing[e.entry_id].rationale,
                            "replaced_by": new_ids[e.entry_id],
                        }
                    )
                )
            elif e.entry_id in {u.entry_id for u in undos}:
                reason = next(u.rationale for u in undos if u.entry_id == e.entry_id)
                out.append(
                    e.model_copy(
                        update={
                            "status": "undone",
                            "retired_day": day,
                            "retired_round": round,
                            "retire_rationale": reason,
                        }
                    )
                )
            else:
                out.append(e)
        for r in replace:
            out.append(
                LedgerEntry(
                    entry_id=new_ids[r.entry_id],
                    round=round,
                    adopted_day=day,
                    action=r.action,
                    rationale=r.rationale,
                    replaces=r.entry_id,
                )
            )
        for a in adoptions:
            number += 1
            out.append(
                LedgerEntry(
                    entry_id=f"L{number}",
                    round=round,
                    adopted_day=day,
                    action=a.action,
                    rationale=a.rationale,
                )
            )
        return Ledger(world_id=self.world_id, version=round, entries=out)

    def file_name(self) -> str:
        return f"ledger-r{self.version}.json"

    def save(self, directory: Path | str) -> Path:
        """Store this version as `ledger-r<version>.json`; an existing file must be identical."""
        path = Path(directory) / self.file_name()
        text = self.model_dump_json(indent=1) + "\n"
        if path.is_file():
            if path.read_text() != text:
                raise ValueError(f"{path} holds another ledger of version {self.version}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text)
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> Ledger:
        return cls.model_validate_json(Path(path).read_text())


# ---- conventions shared by the driver and the evaluator --------------------------------------


def decision_changes(adopt: Sequence[Any] = (), replace: Sequence[Any] = ()) -> int:
    """What a round's decision counts against `MAX_CHANGES_PER_ROUND`: adoptions and
    replacements; an undo counts nothing."""
    return len(adopt) + len(replace)


def changes_in_force(ledger: Ledger) -> int:
    """The number of changes of a state: entries in force at its end."""
    return len(ledger.in_force())


def one_look_ledger(world_id: str, ranked: Sequence[Action], k: int, day: int) -> Ledger:
    """The plan fixed after one look: round 1's ranked list, adoptable actions only
    (complete and simulated), cut at `k` changes (the final state's `changes_in_force`)
    and adopted on `day` (round 1's adoption day) as version 1."""
    adoptable = [a for a in ranked if a.is_complete and a.has_replay]
    chosen = cut(adoptable, Budget(kind="count", amount=k)).actions if k > 0 else []
    return Ledger.empty(world_id).adopt(1, day, adopt=chosen)


# ---- the fold: one robot on one day ----------------------------------------------------------


@dataclass(frozen=True)
class RobotDay:
    """Where a fold happens: a robot at its site on a day (what `resolve` reads)."""

    robot_id: str
    site_id: str
    day: int


@dataclass(frozen=True)
class Applied:
    """What one component or condition step did to the robot's state."""

    step: Step
    removed: tuple[str, ...]  # fault ids
    added: tuple[str, ...]


# The world's own configuration values per scope covering a robot ("fleet", its site, the
# robot): the model's own values, which a narrower scope shadows a wider one with.
Layers = Mapping[str, Mapping[ConfigKey, ConfigValue]]


@dataclass(frozen=True)
class RobotState:
    """A robot's effective state on a day: the faults in force and its values, after the
    steps. `steps`, `world_faults`, `world_current` and `layers` are kept so that a delta
    can be folded on top (`apply_arm`)."""

    where: RobotDay
    world_faults: tuple[FaultSpec, ...]
    world_current: Mapping[Key, ConfigValue]
    layers: Layers
    faults: tuple[FaultSpec, ...]
    current: Mapping[Key, ConfigValue]
    steps: tuple[Step, ...] = ()
    applied: tuple[Applied, ...] = field(default=())
    policy_of: Any = field(default=None, compare=False, repr=False)

    def fault_ids(self) -> list[str]:
        return [f.fault_id for f in self.faults]


def entry_fault_id(source: str, action: Action) -> str:
    """The fault id of a change a ledger entry adds: `f9` and eight digits from the entry
    id and the action, in the block of added changes (`ADDED_ID_BASE`)."""
    digest = int(hash_json({"entry": source, "action": action.model_dump(mode="json")}), 16)
    return neutral_id(IdKind.fault, ADDED_ID_BASE + digest % 100_000_000)


def _key(action: Action) -> Key | None:
    return action.target.config_key or action.target.condition_key


def effective_steps(steps: Sequence[Step]) -> list[Step]:
    """The steps left after revert cancellation (rule 1): a revert cancels the most recent
    uncancelled `set` or revert at exactly its scope and key; with none, it stays (it
    undoes the world's own change there) and a later revert cancels it."""
    stacks: dict[tuple[str, Key], list[int]] = {}
    cancelled: set[int] = set()
    for index, step in enumerate(steps):
        action = step.action
        if action.verb not in (Verb.set, Verb.revert):
            continue
        key = _key(action)
        assert key is not None
        stack = stacks.setdefault((action.target.scope, key), [])
        if action.verb is Verb.revert and stack:
            cancelled.update({stack.pop(), index})
        else:
            stack.append(index)
    return [s for i, s in enumerate(steps) if i not in cancelled]


def _fold_one(state: RobotState, step: Step, truth: TruthTable) -> RobotState:
    """A component verb or a condition change, resolved against the faults left so far."""
    where, action = state.where, step.action
    if not scope_covers(action.target.scope, where.robot_id, where.site_id):
        return _with(state, step, (), (), state.faults, state.current)
    eligible = [
        f for f in state.faults if f.kind in ADDED_CHANGE_KINDS or f.schedule.onset_day <= step.day
    ]
    later = [f for f in state.faults if f not in eligible]
    probe = cast(Any, RobotDay(where.robot_id, where.site_id, step.day))
    result = resolve(action, probe, eligible, truth, state.current)
    current = dict(state.current)
    key = _key(action)
    if action.verb is Verb.change_condition and key is not None:
        current[key] = action.value
    renamed = [
        change.model_copy(
            update={
                "fault_id": (
                    entry_fault_id(step.source, action) if step.source != "arm" else change.fault_id
                ),
                "schedule": Schedule(onset_day=step.day),
            }
        )
        for change in result.added
    ]
    if any(f.target.thing == c.target.thing for f in later for c in renamed):
        raise ValueError(
            f"{where.robot_id} day {where.day}: a change adopted on day {step.day} meets a "
            "fault on the same condition that starts later; the loop does not order those"
        )
    faults = [f for f in eligible if f.fault_id not in result.removed] + renamed + later
    return _with(
        state,
        step,
        tuple(sorted(result.removed)),
        tuple(c.fault_id for c in renamed),
        tuple(faults),
        current,
    )


def _json(value: ConfigValue) -> Any:
    return list(value) if isinstance(value, list) else value


def _with(
    state: RobotState,
    step: Step,
    removed: tuple[str, ...],
    added: tuple[str, ...],
    faults: tuple[FaultSpec, ...],
    current: Mapping[Key, ConfigValue],
) -> RobotState:
    return replace(
        state,
        faults=faults,
        current=current,
        applied=(*state.applied, Applied(step, removed, added)),
    )


@dataclass
class _Entry:
    """One layer's record of a configuration key: the world's base value (fleet), a world
    change (a fault, or a benign own value), or a step's `set`."""

    kind: Literal["base", "fault", "value", "set"]
    value: ConfigValue
    day: int
    fault: FaultSpec | None = None
    step: Step | None = None


def _settable(fault: FaultSpec, truth: TruthTable) -> bool:
    """A configuration fault that `set` and `revert` act on (the others last)."""
    mapping = truth[fault.kind]
    return fault.target.config_key is not None and Verb.set in mapping.fixing_verbs


def _config_key(
    key: ConfigKey,
    where: RobotDay,
    world: Sequence[FaultSpec],
    layers: Layers,
    steps: Sequence[Step],
) -> tuple[FaultSpec | None, Step | None, ConfigValue, ConfigValue]:
    """Who decides one configuration key for the robot: (the world's fault, or the step
    whose `set` decides, or neither; the value; the base value).

    Every scope covering the robot keeps a stack: the fleet's base value, the world's own
    change at that scope (a fault, else a benign own value), then the steps' sets in
    order; an uncancelled revert removes the world's change at its scope (rule 1). The
    narrowest scope with something on its stack decides, so a retained wider setting
    shadowed by a narrower one comes back when the narrower one is removed.
    """
    chain = [FLEET_SCOPE, where.site_id, where.robot_id]
    on_key = [f for f in world if f.target.config_key is key]
    fleet_fault = next((f for f in on_key if f.target.scope == FLEET_SCOPE), None)
    if fleet_fault is not None:
        base = fleet_fault.nominal
    elif key in layers.get(FLEET_SCOPE, {}):
        base = layers[FLEET_SCOPE][key]
    else:
        base = next((f.nominal for f in on_key), None)
    stacks: dict[str, list[_Entry]] = {scope: [] for scope in chain}
    stacks[FLEET_SCOPE].append(_Entry("base", base, -1))
    for scope in chain:
        here = [f for f in on_key if f.target.scope == scope]
        if len(here) > 1:
            raise ValueError(f"two faults change {key.value} at {scope}")
        own = layers.get(scope, {})
        if here:
            value = own.get(key, base)
            stacks[scope].append(_Entry("fault", value, here[0].schedule.onset_day, here[0]))
        elif scope != FLEET_SCOPE and key in own:
            stacks[scope].append(_Entry("value", own[key], -1))
    for step in steps:
        action = step.action
        scope = action.target.scope
        if action.target.config_key is not key or scope not in stacks:
            continue
        stack = stacks[scope]
        if any(e.kind == "fault" and e.day > step.day for e in stack):
            raise ValueError(
                f"{where.robot_id}: a change of {key.value} at {scope} on day {step.day} "
                "precedes a fault there; the loop does not order those"
            )
        if action.verb is Verb.set:
            stack.append(_Entry("set", action.value, step.day, step=step))
            continue
        for index in range(len(stack) - 1, -1, -1):
            if stack[index].kind in ("fault", "value"):
                del stack[index]  # the world's own change there, undone
                break
            if stack[index].kind == "base":
                break
    top = next(stacks[scope][-1] for scope in reversed(chain) if stacks[scope])
    return top.fault, top.step, top.value, base


# The engine's effective policy under a list of faults, their hooks run in that order
# (`fleetlab.engine.episode.effective_spec` with the registry's types).
PolicyOf = Callable[[Sequence[FaultSpec]], PolicySpec]


def materialize(
    world_faults: Sequence[FaultSpec],
    where: RobotDay,
    steps: Sequence[Step],
    truth: TruthTable,
    current: Values,
    layers: Layers,
    policy_of: PolicyOf | None = None,
) -> RobotState:
    """The robot's state on `where.day` under the steps: the one written state model.

    `world_faults` are every fault the world has in force on the robot that day, in the
    world's own order (a configuration fault that a narrower scope shadows included);
    `layers` the world model's own configuration values at the fleet, the robot's site
    and the robot; `current` the robot's values before any step; `steps` the ledger's
    steps in force (`Ledger.active_steps`), then an arm's.

    The faults in force, in the order their hooks run, which is the precedence of the
    effective policy: (1) the world's own behaviour, its releases included: the world's
    faults in their own order, less those the steps removed or a deciding setting
    shadows; (2) the ledger's retained changes, in adoption order; (3) an experiment
    arm's, in its order. An explicit `set` that decides a key adds a `config_change`
    whose nominal is the value in force before it: for a policy setting the value the
    hooks before it give (`policy_of`, which a release carrying that setting changes),
    else the base value. So a setting always takes effect, also against a release that
    carries the same setting, and runs no hook only where it changes nothing. Revert and
    undo are rule 1 and the ledger's retirement; components and conditions rule 3.
    Without steps the result is the world's own, the shadowing of
    `contracts.faults.active_faults` included.
    """
    from fleetlab.faults.policy_knob import POLICY_FIELDS

    effective = effective_steps(steps)
    position = {id(step): n for n, step in enumerate(effective)}
    layered = [f for f in world_faults if _settable(f, truth)]
    state = RobotState(
        where=where,
        world_faults=tuple(world_faults),
        world_current=dict(current),
        layers=layers,
        faults=tuple(f for f in world_faults if f not in layered),
        current=dict(current),
    )
    config_steps = []
    for step in effective:
        if step.action.target.config_key is not None:
            config_steps.append(step)
        else:
            state = _fold_one(state, step, truth)
    values = dict(state.current)
    deciding_faults: set[str] = set()
    settings: list[tuple[int, Step, ConfigValue]] = []
    keys = {f.target.config_key for f in layered} | {
        s.action.target.config_key for s in config_steps
    }
    for key in sorted(k for k in keys if k is not None):
        fault, decider, value, base = _config_key(key, where, layered, layers, config_steps)
        values[key] = value
        if fault is not None:
            deciding_faults.add(fault.fault_id)
        if decider is not None:
            settings.append((position[id(decider)], decider, base))
    kept = {f.fault_id for f in state.faults} | deciding_faults
    world_part = [f for f in world_faults if f.fault_id in kept]
    world_ids = {f.fault_id for f in world_faults}
    added_by = {a: n for n, applied in enumerate(state.applied) for a in applied.added}
    others_added = [
        (position[id(state.applied[added_by[f.fault_id]].step)], f)
        for f in state.faults
        if f.fault_id not in world_ids
    ]
    pending = sorted(
        [(n, "change", f, None) for n, f in others_added]
        + [(n, "setting", step, base) for n, step, base in settings],
        key=lambda item: item[0],
    )
    faults: list[FaultSpec] = list(world_part)
    for _, what, item, base in pending:
        if what == "change":
            assert isinstance(item, FaultSpec)
            faults.append(item)
            continue
        assert isinstance(item, Step)
        action = item.action
        setting = action.target.config_key
        assert setting is not None
        nominal: ConfigValue = base
        if setting in POLICY_FIELDS and policy_of is not None:
            nominal = getattr(policy_of(faults), POLICY_FIELDS[setting][0])
        fault_id = (
            entry_fault_id(item.source, action) if item.source != "arm" else added_fault_id(action)
        )
        faults.append(
            FaultSpec(
                fault_id=fault_id,
                kind=CONFIG_CHANGE_KIND,
                params={"value": _json(action.value)},
                level=1.0,
                target=action.target,
                schedule=Schedule(onset_day=item.day),
                nominal=nominal,
            )
        )
    return replace(
        state, faults=tuple(faults), current=values, steps=tuple(steps), policy_of=policy_of
    )


def apply_arm(state: RobotState, actions: Sequence[Action], truth: TruthTable) -> RobotState:
    """An experiment arm's actions as a delta on the state: the ledger's steps and then
    the arm's, on the state's day (precedence (3) of `materialize`). The control arm (no
    actions) is the state itself."""
    if not actions:
        return state
    arm = [Step("arm", state.where.day, a) for a in actions]
    return materialize(
        state.world_faults,
        state.where,
        [*state.steps, *arm],
        truth,
        state.world_current,
        state.layers,
        state.policy_of,
    )


# ---- the records: what the fleet believes after the ledger -----------------------------------


def _scope_layers(fleet: Fleet) -> dict[str, dict[ConfigKey, ConfigValue]]:
    layers: dict[str, dict[ConfigKey, ConfigValue]] = {FLEET_SCOPE: dict(fleet.config.values)}
    for site in fleet.sites:
        layers[site.id] = dict(site.config.values)
        for robot in site.robots:
            layers[robot.id] = dict(robot.config.values)
    return layers


def _effective_at(
    fleet: Fleet, layers: Mapping[str, Mapping[ConfigKey, ConfigValue]], scope: str, key: ConfigKey
) -> ConfigValue:
    """The value a scope sees for a key: its own, else the next wider scope's."""
    chain = [scope]
    kind = scope_kind(scope)
    if kind is ScopeKind.robot:
        chain.append(fleet.robot(scope).site_id)
    if kind is not ScopeKind.fleet:
        chain.append(FLEET_SCOPE)
    for s in chain:
        if key in layers.get(s, {}):
            return layers[s][key]
    return None


def _apply_config(
    layers: dict[str, dict[ConfigKey, ConfigValue]],
    history: Mapping[tuple[str, ConfigKey], list[Change]],
    steps: Sequence[Step],
) -> None:
    """The effective steps' configuration changes on the layers, in order (rule 1).

    `history` is the world's own change log by scope and key. A step meets the world's
    changes up to its own day; a world change at the same scope and key after it is a
    later change and wins (the step is skipped)."""
    popped: dict[tuple[str, ConfigKey], int] = {}
    for step in effective_steps(steps):
        action = step.action
        key = action.target.config_key
        if key is None:
            continue
        scope = action.target.scope
        recorded = history.get((scope, key), [])
        if any(c.day > step.day for c in recorded):
            continue
        if action.verb is Verb.set:
            layers.setdefault(scope, {})[key] = action.value
        elif action.verb is Verb.revert:
            depth = popped.get((scope, key), 0)
            if depth >= len(recorded):
                continue  # nothing recorded to undo here (preflight refuses such a revert)
            position = len(recorded) - 1 - depth
            popped[(scope, key)] = depth + 1
            if position == 0 and scope != FLEET_SCOPE:
                layers.setdefault(scope, {}).pop(key, None)  # the first change: inherit again
            else:
                layers.setdefault(scope, {})[key] = recorded[position].old


def _history(fleet: Fleet) -> dict[tuple[str, ConfigKey], list[Change]]:
    out: dict[tuple[str, ConfigKey], list[Change]] = {}
    for change in sorted(fleet.change_log, key=lambda c: c.day):
        out.setdefault((change.scope, change.key), []).append(change)
    return out


def ledger_records(
    fleet_on: Callable[[int], Fleet], ledger: Ledger, until_day: int
) -> tuple[list[Change], list[Maintenance], dict[str, dict[ConditionKey, ConfigValue]]]:
    """The fleet's own records of the ledger up to `until_day`: the change-log entries
    (author `ops`, on the day a change took effect: an adoption, or a retirement that
    takes one back), the maintenance entries of component verbs, and the conditions each
    site is believed to have after the ledger.

    `fleet_on(day)` is the world's model as it stands on a day without the ledger. A
    change-log entry is written for every scope whose own value of a key the ledger's
    steps change on that day, with the values the scope saw before and after.
    """
    days = sorted(
        {e.adopted_day for e in ledger.entries}
        | {e.retired_day for e in ledger.entries if e.retired_day is not None}
    )
    changes: list[Change] = []
    for day in [d for d in days if d <= until_day]:
        fleet = fleet_on(day)
        history = _history(fleet)
        before = _scope_layers(fleet)
        after = _scope_layers(fleet)
        _apply_config(before, history, ledger.active_steps(day - 1))
        _apply_config(after, history, ledger.active_steps(day))
        for scope in sorted(set(before) | set(after)):
            for key in sorted(set(before.get(scope, {})) | set(after.get(scope, {}))):
                own_before = before.get(scope, {}).get(key, _MISSING)
                own_after = after.get(scope, {}).get(key, _MISSING)
                if own_before == own_after:
                    continue
                changes.append(
                    Change(
                        day=day,
                        scope=scope,
                        key=key,
                        old=_effective_at(fleet, before, scope, key),
                        new=_effective_at(fleet, after, scope, key),
                        author=ChangeAuthor.ops,
                    )
                )
    upkeep = [
        Maintenance(
            day=e.adopted_day,
            robot_id=e.action.target.scope,
            component=e.action.target.component,
            verb=e.action.verb,
            note=_COMPONENT_NOTES.get(e.action.verb, ""),
        )
        for e in ledger.entries
        if e.action.verb in COMPONENT_VERBS
        and e.action.target.component is not None
        and e.adopted_day <= until_day
    ]
    final = fleet_on(until_day)
    conditions: dict[str, dict[ConditionKey, ConfigValue]] = {
        s.id: dict(s.conditions) for s in final.sites
    }
    for step in ledger.active_steps(until_day):
        condition = step.action.target.condition_key
        if step.action.verb is Verb.change_condition and condition is not None:
            for site in final.sites:
                if scope_covers(step.action.target.scope, site.robots[0].id, site.id):
                    conditions[site.id][condition] = step.action.value
    return changes, upkeep, conditions


_MISSING = object()


def believed_fleet(fleet_on: Callable[[int], Fleet], ledger: Ledger, until_day: int) -> Fleet:
    """The fleet's model on `until_day` after the ledger, as the fleet believes it: the
    configuration with the steps in force applied (rule 1), the change and maintenance
    logs with the ledger's own entries (`ledger_records`), the believed site conditions,
    and the last service day of components a ledger entry serviced or replaced.

    `fleet_on(day)` is the world's model on a day without the ledger (its records cut
    to that day). Nothing in the result says whether an entry removed a fault.
    """
    fleet = fleet_on(until_day)
    changes, upkeep, conditions = ledger_records(fleet_on, ledger, until_day)
    layers = _scope_layers(fleet)
    _apply_config(layers, _history(fleet), ledger.active_steps(until_day))
    counts: dict[str, list[int]] = {}
    for change in changes:
        counts.setdefault(change.scope, []).append(change.day)

    def version(old: ConfigVersion, scope: str) -> ConfigVersion:
        days = counts.get(scope, [])
        return ConfigVersion(
            version=old.version + len(days),
            values=layers.get(scope, {}),
            since_day=max([old.since_day, *days]),
        )

    serviced: dict[tuple[str, Any], int] = {}
    for m in upkeep:
        if m.verb in (Verb.service, Verb.replace_):
            serviced[(m.robot_id, m.component)] = m.day
    sites = []
    for site in fleet.sites:
        robots = []
        for robot in site.robots:
            components = {
                c: (
                    state.model_copy(update={"last_service_day": serviced[(robot.id, c)]})
                    if (robot.id, c) in serviced
                    else state
                )
                for c, state in robot.components.items()
            }
            robots.append(
                robot.model_copy(
                    update={"config": version(robot.config, robot.id), "components": components}
                )
            )
        sites.append(
            site.model_copy(
                update={
                    "robots": robots,
                    "config": version(site.config, site.id),
                    "conditions": conditions[site.id],
                }
            )
        )
    log = sorted([*fleet.change_log, *changes], key=lambda c: c.day)
    upkeep_log = sorted([*fleet.maintenance_log, *upkeep], key=lambda m: m.day)
    return Fleet(
        id=fleet.id,
        sites=sites,
        config=version(fleet.config, FLEET_SCOPE),
        change_log=log,
        maintenance_log=upkeep_log,
    )


def describe_ledger(ledger: Ledger) -> list[dict[str, Any]]:
    """A plain listing of the entries (public: the agent's own actions and words)."""
    return [json.loads(e.model_dump_json()) for e in ledger.entries]


def steps_of(entries: Iterable[LedgerEntry]) -> list[Step]:
    return [Step(e.entry_id, e.adopted_day, e.action) for e in entries]
