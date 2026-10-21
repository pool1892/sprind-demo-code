"""Actions, costs, budgets and fix sets, and how an action resolves per episode
(docs/design/contracts.md, section 8).

An action is a verb from the vocabulary applied to a target at a scope. The
catalogue of a world is every valid (verb, target) pair over its model. A fix
set is the actions a method proposes within a budget. `resolve` is the bridge
from an action to the oracle, evaluated per episode against the faults active
in it: which of them the action removes, and which changes it adds. It reads
each fault type's `TruthMapping` from a truth table (built from the registry in
`fleetlab.faults`), so the scorer and the oracle agree by construction.

The rules, for an episode of robot R at site S (verb metadata in
`vocabulary.VERBS`):

- Nothing happens unless the action's scope covers R and the verb has a
  replay (not `route_to_teleop`, not `request_training_data`: claim 5).
- A fault is touched only if it names the same component, configuration key
  or condition as the action and the verb is one of its type's fixing verbs.
- Component verbs are robot-scoped: `recalibrate scene_camera on r07` removes
  a camera fault active on r07's episodes.
- `revert key at X` removes the fault whose `Change` is at exactly X (its type
  writes a change), for every robot X covers.
- `set key = v at X` and `change_condition c = v at S` override: they remove a
  matching fault whose scope is X or wider, unless a narrower fault on the
  same key still decides R's value, in which case nothing changes for R. If v
  is not the value the fault replaced (`FaultSpec.nominal`), or no fault was
  there and v is not R's current value, the action adds a change, which no
  replayed subset covers: `added` always means a fresh replay.
- An action on a healthy target removes nothing and adds nothing.

Choices where the note is silent:

- An added change is a `FaultSpec` of the kind `CONFIG_CHANGE_KIND` or
  `CONDITION_CHANGE_KIND` with `params={"value": v}` at the action's target,
  in force from the episode's day; the fault types that simulate these two
  kinds are registered in `fleetlab.faults`. Its id is
  `f9........`, derived from the action, a block world generators do not use.
- `resolve` takes the robot's current values (`current`: its effective
  configuration and its site's conditions) for a `set` or `change_condition`
  of a key no fault touches: they say whether the value changes, and an added
  change records the value it replaced as its `nominal`. Without them such an
  action raises; with a fault on the key, the fault's `nominal` serves.
- The two verbs without a replay name no component or key; their value holds
  what they apply to. Both are fleet-scoped; the catalogue lists
  `route_to_teleop` once per task class it is given.
- A catalogue entry for a verb that takes a value has `value=None`, to be
  filled by whoever proposes it; a `FixSet` refuses an unfilled value.
- `Budget` is an object (`kind`, `amount`): a count of actions, or operator
  hours. Under a cost budget an action that is not costed
  (`request_training_data`) fits at no cost.

Costs (`COSTS`) are keyed by verb and what it names:
the component for a component verb, the target kind for a verb on a key, None
for the two fleet-wide verbs. They appear on every catalogue entry of the
system description and in the visible table `costs` (`costs_rows`). A method
hands in a ranked list; `cut` turns it into the fix set at a budget, walking
down the list and skipping what no longer fits, and the scorer applies it at
every budget of `SCORING_BUDGETS`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity

from pydantic import Field, JsonValue, model_validator

from fleetlab.contracts._base import ConfigValue, ContractModel
from fleetlab.contracts.faults import FaultSpec, Schedule, Target, TruthTable
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.ids import (
    FLEET_SCOPE,
    IdKind,
    neutral_id,
    scope_covers,
    scope_kind,
    scope_rank,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.vocabulary import (
    LATER_CONFIG_KEYS,
    NO_REPLAY_VERBS,
    VERBS,
    Component,
    ConditionKey,
    ConfigKey,
    TargetKind,
    Verb,
)

__all__ = [
    "ADDED_CHANGE_KINDS",
    "CONDITION_CHANGE_KIND",
    "CONFIG_CHANGE_KIND",
    "COSTS",
    "NO_REPLAY_VERBS",
    "SCORING_BUDGETS",
    "Action",
    "Budget",
    "Cost",
    "CostKey",
    "CostTable",
    "FixSet",
    "Resolution",
    "catalogue",
    "catalogue_rows",
    "cost_key",
    "cost_of",
    "costs_rows",
    "cut",
    "fits_budget",
    "resolve",
    "resolve_all",
    "total_cost",
]

# The kinds of the changes an action adds (registered in fleetlab.faults).
CONFIG_CHANGE_KIND = "config_change"
CONDITION_CHANGE_KIND = "condition_change"
# The only fault kind names this package spells. The tool server imports this
# module, so they reach the sandbox; the pre-export word search over the
# truth vocabulary lists them as its known exception (section 7).
ADDED_CHANGE_KINDS: tuple[str, ...] = (CONFIG_CHANGE_KIND, CONDITION_CHANGE_KIND)
# Added changes get fault ids in this block; generated worlds number their faults below it.
ADDED_ID_BASE = 900_000_000


class Action(ContractModel):
    """A verb applied to a target, with a value for the verbs that take one."""

    verb: Verb
    target: Target
    value: ConfigValue = None

    @model_validator(mode="after")
    def _fits_verb(self) -> Action:
        info = VERBS[self.verb]
        if self.target.kind != info.target_kind:
            wanted = "no component or key" if info.target_kind is None else info.target_kind.value
            raise ValueError(f"{self.verb.value} takes {wanted} as its target")
        component = self.target.component
        if component is not None and info.components and component not in info.components:
            allowed = [c.value for c in Component if c in info.components]
            raise ValueError(f"{self.verb.value} applies to {allowed}, not {component.value}")
        if self.target.scope_kind not in info.scope_kinds:
            allowed = sorted(k.value for k in info.scope_kinds)
            raise ValueError(
                f"{self.verb.value} applies at scope {allowed}, not {self.target.scope}"
            )
        if not info.takes_value and self.value is not None:
            raise ValueError(f"{self.verb.value} takes no value")
        if (
            self.verb in NO_REPLAY_VERBS
            and self.value is not None
            and not isinstance(self.value, str)
        ):
            raise ValueError(f"the value of {self.verb.value} is text")
        return self

    @property
    def has_replay(self) -> bool:
        """False for the two verbs the oracle cannot value by replay (claim 5)."""
        return VERBS[self.verb].has_replay

    @property
    def is_complete(self) -> bool:
        """Whether a verb that takes a value has one (catalogue entries do not)."""
        return not VERBS[self.verb].takes_value or self.value is not None


class Cost(ContractModel):
    """What actions cost in operator hours (experimental design section 3)."""

    hours: float = Field(ge=0.0)


# Costs are keyed by verb and what the verb names: the component for a component
# verb, the target kind for a verb on a key, None for the two fleet-wide verbs.
CostKey = tuple[Verb, Component | TargetKind | None]
# Hours per key; None for an action that is not costed (request_training_data).
CostTable = Mapping[CostKey, float | None]

_REPLACE_HOURS = {
    Component.scene_camera: 8.0,
    Component.wrist_camera: 8.0,
    Component.gripper: 8.0,
    Component.arm: 16.0,
    Component.controller: 16.0,
}

# The costs, in operator hours with parts folded in at four hours per part
# (stated as assumptions). route_to_teleop costs nothing up front (operator time is
# charged in claim 4); request_training_data is not costed (claim 5).
COSTS: dict[CostKey, float | None] = {
    **{(Verb.recalibrate, c): 0.5 for c in Component if c in VERBS[Verb.recalibrate].components},
    **{(Verb.clean, c): 0.25 for c in Component if c in VERBS[Verb.clean].components},
    **{(Verb.service, c): 3.0 for c in Component if c in VERBS[Verb.service].components},
    **{(Verb.replace_, c): _REPLACE_HOURS[c] for c in Component},
    (Verb.set, TargetKind.config_key): 0.1,
    (Verb.revert, TargetKind.config_key): 0.1,
    (Verb.change_condition, TargetKind.condition_key): 4.0,
    (Verb.route_to_teleop, None): 0.0,
    (Verb.request_training_data, None): None,
}


def cost_key(action: Action) -> CostKey:
    """The key of an action's cost: its verb and its component, its target kind, or None."""
    return (action.verb, action.target.component or action.target.kind)


def cost_of(action: Action, costs: CostTable = COSTS) -> float | None:
    """An action's cost in operator hours; None if it is not costed. KeyError if unknown."""
    key = cost_key(action)
    if key not in costs:
        raise KeyError(f"no cost for {action.verb.value} on {key[1]}")
    return costs[key]


def costs_rows(costs: CostTable = COSTS) -> list[dict[str, str | float | None]]:
    """The rows of the visible table `costs`: verb, what it names, hours (empty if not costed)."""
    return [
        {"verb": verb.value, "target": None if target is None else target.value, "hours": hours}
        for (verb, target), hours in costs.items()
    ]


def catalogue_rows(
    fleet: Fleet, task_classes: Sequence[str] = (), costs: CostTable = COSTS
) -> list[dict[str, str | float | None]]:
    """The rows of the visible table `catalogue`: one per valid action of `catalogue`.

    verb, scope, component, config_key, condition_key (the one the verb names,
    the others empty), value (a task class for `route_to_teleop`, empty where
    the proposer fills the value in) and hours (empty if not costed).
    """
    rows: list[dict[str, str | float | None]] = []
    for action in catalogue(fleet, task_classes):
        if action.value is not None and not isinstance(action.value, str):
            raise ValueError(f"a catalogue value is a task class, not {action.value!r}")
        target = action.target
        rows.append(
            {
                "verb": action.verb.value,
                "scope": target.scope,
                "component": None if target.component is None else target.component.value,
                "config_key": None if target.config_key is None else target.config_key.value,
                "condition_key": (
                    None if target.condition_key is None else target.condition_key.value
                ),
                "value": action.value,
                "hours": cost_of(action, costs),
            }
        )
    return rows


class Budget(ContractModel):
    """A budget: at most `amount` actions ("count"), or at most `amount` operator hours ("cost")."""

    kind: Literal["count", "cost"]
    amount: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _whole_count(self) -> Budget:
        if self.kind == "count" and not float(self.amount).is_integer():
            raise ValueError("a count budget is a whole number of actions")
        return self


# The budgets the scorer cuts every ranked list at (section 8): k = 3 and 6 actions,
# 4 and 8 operator hours.
SCORING_BUDGETS: tuple[Budget, ...] = (
    Budget(kind="count", amount=3),
    Budget(kind="count", amount=6),
    Budget(kind="cost", amount=4),
    Budget(kind="cost", amount=8),
)


def total_cost(actions: Iterable[Action], costs: CostTable = COSTS) -> Cost:
    """The summed hours of the actions; an action that is not costed counts as none."""
    return Cost(hours=sum(cost_of(action, costs) or 0.0 for action in actions))


class FixSet(ContractModel):
    """The actions a method proposes, within a budget. Values all filled, no action twice."""

    actions: list[Action]
    budget: Budget

    @model_validator(mode="after")
    def _valid(self) -> FixSet:
        incomplete = [a.verb.value for a in self.actions if not a.is_complete]
        if incomplete:
            raise ValueError(f"actions without a value: {incomplete}")
        dumped = [a.model_dump_json() for a in self.actions]
        if len(dumped) != len(set(dumped)):
            raise ValueError("a fix set lists each action once")
        if self.budget.kind == "count" and len(self.actions) > self.budget.amount:
            raise ValueError(
                f"{len(self.actions)} actions exceed the budget of {int(self.budget.amount)}"
            )
        return self


# Sums of hours like 0.1 + 0.2 are not exact in binary; a budget is met within this.
_HOURS_SLACK = 1e-9


def fits_budget(fix_set: FixSet, costs: CostTable = COSTS) -> bool:
    """Whether the fix set fits its budget (a count budget is checked on construction)."""
    if fix_set.budget.kind == "count":
        return len(fix_set.actions) <= fix_set.budget.amount
    return total_cost(fix_set.actions, costs).hours <= fix_set.budget.amount + _HOURS_SLACK


def cut(ranked: Sequence[Action], budget: Budget, costs: CostTable = COSTS) -> FixSet:
    """The fix set a ranked list gives at a budget: walk down the list in order and skip
    an action that no longer fits the remaining budget.

    Under a count budget the first k distinct actions; under a cost budget every
    action, in order, whose hours still fit (an action that is not costed fits
    at no cost). A repeated action is skipped. The scorer applies it at every
    budget of `SCORING_BUDGETS`. Raises ValueError for an action without its value.
    """
    chosen: list[Action] = []
    seen: set[str] = set()
    spent = 0.0
    for action in ranked:
        if not action.is_complete:
            raise ValueError(f"{action.verb.value} without a value cannot be in a fix set")
        key = action.model_dump_json()
        if key in seen:
            continue
        if budget.kind == "count":
            if len(chosen) >= budget.amount:
                break
        else:
            hours = cost_of(action, costs) or 0.0
            if spent + hours > budget.amount + _HOURS_SLACK:
                continue
            spent += hours
        chosen.append(action)
        seen.add(key)
    return FixSet(actions=chosen, budget=budget)


def catalogue(fleet: Fleet, task_classes: Sequence[str] = ()) -> list[Action]:
    """Every valid (verb, target) pair over the fleet's model, in a fixed order.

    Component verbs on every component of every robot; `set` and `revert` of
    every configuration key at the fleet, every site and every robot (a key of
    `LATER_CONFIG_KEYS` only if the fleet's configuration records its fleet
    default: a fleet composed before the key existed does not have the setting);
    `change_condition` of every condition at every site; `route_to_teleop` per
    task class (one entry with no value if none are given);
    `request_training_data` once. Entries of `set`, `change_condition` and
    `request_training_data` carry `value=None`.
    """
    scopes = [FLEET_SCOPE, *(s.id for s in fleet.sites), *(r.id for r in fleet.robots())]
    keys = [k for k in ConfigKey if k not in LATER_CONFIG_KEYS or k in fleet.config.values]
    out: list[Action] = []
    for verb, info in VERBS.items():
        for scope in [s for s in scopes if scope_kind(s) in info.scope_kinds]:
            if info.target_kind is TargetKind.component:
                robot = fleet.robot(scope)
                for component in Component:
                    if component in robot.components and component in info.components:
                        out.append(
                            Action(verb=verb, target=Target(scope=scope, component=component))
                        )
            elif info.target_kind is TargetKind.config_key:
                for key in keys:
                    out.append(Action(verb=verb, target=Target(scope=scope, config_key=key)))
            elif info.target_kind is TargetKind.condition_key:
                for condition in ConditionKey:
                    out.append(
                        Action(verb=verb, target=Target(scope=scope, condition_key=condition))
                    )
            elif verb is Verb.route_to_teleop and task_classes:
                for task in task_classes:
                    out.append(Action(verb=verb, target=Target(scope=scope), value=task))
            else:
                out.append(Action(verb=verb, target=Target(scope=scope)))
    return out


class Resolution(NamedTuple):
    """What an action does to one episode: faults it removes, changes it adds."""

    removed: frozenset[str]
    added: tuple[FaultSpec, ...]

    @property
    def is_no_change(self) -> bool:
        return not self.removed and not self.added

    @property
    def needs_fresh_replay(self) -> bool:
        """An added change is covered by no replayed subset."""
        return bool(self.added)


_NO_CHANGE = Resolution(frozenset(), ())


def added_fault_id(action: Action) -> str:
    """The fault id of the change an action adds: `f9` and eight digits from the action."""
    digest = int(hash_json(action.model_dump(mode="json")), 16)
    return neutral_id(IdKind.fault, ADDED_ID_BASE + digest % 100_000_000)


def resolve(
    action: Action,
    identity: EpisodeIdentity,
    faults_active: Sequence[FaultSpec],
    truth: TruthTable,
    current: Mapping[ConfigKey | ConditionKey, ConfigValue] | None = None,
) -> Resolution:
    """What `action` does to one episode: the ids of the faults it removes, the changes it adds.

    `faults_active` are the faults active in the episode (its identity's
    `faults`, or what earlier actions left of them); `truth` maps each fault
    kind to its `TruthMapping`; `current` gives the robot's current
    configuration values and site conditions, needed only for a `set` or
    `change_condition` of a key no active fault touches. Raises ValueError for
    a fault that cannot be active on the episode's robot, for an action whose
    value is missing, and for such a `set` without the key in `current`.
    """
    robot_id, site_id = identity.robot_id, identity.site_id
    outside = [f.fault_id for f in faults_active if not f.target.covers(robot_id, site_id)]
    if outside:
        raise ValueError(f"faults {outside} are not active for robot {robot_id} at {site_id}")
    if not action.is_complete:
        raise ValueError(f"{action.verb.value} without a value cannot be resolved")
    info = VERBS[action.verb]
    if not info.has_replay or not scope_covers(action.target.scope, robot_id, site_id):
        return _NO_CHANGE

    def fixes(fault: FaultSpec) -> bool:
        try:
            mapping = truth[fault.kind]
        except KeyError:
            raise KeyError(f"no truth mapping for fault kind {fault.kind!r}") from None
        return action.verb in mapping.fixing_verbs and (
            not info.exact_scope or mapping.writes_change
        )

    same = [
        f
        for f in faults_active
        if f.is_fault
        and f.target.kind is action.target.kind
        and f.target.thing == action.target.thing
    ]
    if info.exact_scope:  # revert: the change made at exactly this scope
        exact = [f for f in same if f.target.scope == action.target.scope and fixes(f)]
        return Resolution(frozenset(f.fault_id for f in exact), ())
    if not info.adds_change:  # component verbs, robot-scoped
        return Resolution(frozenset(f.fault_id for f in same if fixes(f)), ())

    # set and change_condition: a value at a scope, overriding wider scopes for this robot.
    rank = scope_rank(action.target.scope)
    if any(scope_rank(f.target.scope) > rank for f in same):
        return _NO_CHANGE  # a narrower fault still decides this robot's value
    overridden = sorted(same, key=lambda f: scope_rank(f.target.scope))
    removed_ids = frozenset(f.fault_id for f in overridden if fixes(f))
    if overridden:
        nominal = overridden[-1].nominal
        unchanged = nominal is not None and action.value == nominal
    else:
        key = action.target.config_key or action.target.condition_key
        if current is None or key is None or key not in current:
            raise ValueError(
                f"{action.verb.value} of {key} on a healthy robot needs its current value "
                "(`current`), to tell a change and to record the value it replaces"
            )
        nominal = current[key]
        unchanged = action.value == nominal
    if unchanged:
        return Resolution(removed_ids, ())
    kind = CONFIG_CHANGE_KIND if action.verb is Verb.set else CONDITION_CHANGE_KIND
    change = FaultSpec(
        fault_id=added_fault_id(action),
        kind=kind,
        params={"value": _json_value(action.value)},
        level=1.0,
        target=action.target,
        schedule=Schedule(onset_day=identity.day),
        nominal=nominal,
    )
    return Resolution(removed_ids, (change,))


def _json_value(value: ConfigValue) -> JsonValue:
    return list(value) if isinstance(value, list) else value


def resolve_all(
    actions: Iterable[Action],
    identity: EpisodeIdentity,
    faults_active: Sequence[FaultSpec],
    truth: TruthTable,
    current: Mapping[ConfigKey | ConditionKey, ConfigValue] | None = None,
) -> Resolution:
    """What a set of actions does to one episode together: removals and additions combined."""
    removed: set[str] = set()
    added: dict[str, FaultSpec] = {}
    for action in actions:
        result = resolve(action, identity, faults_active, truth, current)
        removed |= result.removed
        for change in result.added:
            added.setdefault(change.fault_id, change)
    return Resolution(frozenset(removed), tuple(added.values()))
