"""Faults as data, and the hooks a fault type implements (docs/design/contracts.md, section 7).

A fault is a declared change to a component, a configuration value or a
condition of the system model (`FaultSpec`), with a level, a target and a
schedule. A fault at level zero is no fault. The rollout loop knows the hook
points (`HOOK_ORDER`) and nothing about any fault type. Fault types are
registered under their `kind` in `fleetlab.faults`, outside this package: the
contracts name no fault kind but the two generic kinds of the changes an
action adds (`contracts.actions.ADDED_CHANGE_KINDS`), so nothing an agent can
read lists the fault types.

Choices this module makes where the note is silent:

- `FaultSpec` carries a `fault_id` (a neutral `f..` id). Jobs, oracle rows and
  the `faults_active` table refer to faults by id.
- `FaultSpec.nominal` records, for a fault on a configuration key or a
  condition, the value the fault replaced; such a fault always carries it.
  `set` and `change_condition` remove the fault without adding a change only
  when they put that value back (`contracts.actions.resolve`).
- `TruthMapping` names the kind of thing the fault acts on (component,
  configuration key or condition), not the thing itself: the fault's `Target`
  names it, so one fault type can hit either camera.
- `FaultState` is a small base class that a fault type extends with its own
  fields; `FaultTypeBase` gives every hook a do-nothing default, so a type
  implements only the hooks it needs.

The rules of a truth mapping (section 7), checked when it is built and when a
type is registered: a configuration-key fault that any verb fixes lists `set`
and `revert`; a condition fault lists `change_condition`; a component fault is
robot-scoped only (a bad batch at a site is one fault per robot, so k fixes are
k actions); only a configuration-key fault writes a `Change`, and
`writes_change` holds exactly for the configuration and software classes.
`active_faults` respects the configuration precedence: a wider-scoped
configuration fault does not reach a robot whose narrower scope sets the key.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, Field, JsonValue, model_validator

from fleetlab.contracts._base import ConfigValue, ContractModel
from fleetlab.contracts.ids import (
    SCOPE_RANK,
    FaultId,
    Scope,
    ScopeKind,
    scope_covers,
    scope_kind,
    scope_rank,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.vocabulary import (
    VERBS,
    Component,
    ConditionKey,
    ConfigKey,
    FaultClass,
    TargetKind,
    Verb,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Observation, SuccessVerdict, Undo
    from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
    from fleetlab.contracts.policy import Action7

KIND_PATTERN = r"^[a-z][a-z0-9_]*$"


class Direction(StrEnum):
    """How a diagnosed cause departs from normal (`Cause.direction`, section 12)."""

    degraded = "degraded"
    misconfigured = "misconfigured"
    changed = "changed"
    drifting = "drifting"


DIRECTION_DESCRIPTIONS: dict[Direction, str] = {
    Direction.degraded: "a component works worse than it should",
    Direction.misconfigured: "a configuration value is wrong",
    Direction.changed: "something changed and the robots did not adapt",
    Direction.drifting: "something gets gradually worse over days",
}


class Target(ContractModel):
    """Where a fault sits, what an action is applied to, or what a diagnosis blames.

    A scope ("fleet", a site id or a robot id) and at most one of a component,
    a configuration key or a condition key. A target with none of the three is
    the scope as a whole; `route_to_teleop` and `request_training_data` use
    that, and a diagnosis may when it cannot say more.
    """

    scope: Scope
    component: Component | None = None
    config_key: ConfigKey | None = None
    condition_key: ConditionKey | None = None

    @model_validator(mode="after")
    def _at_most_one_thing(self) -> Target:
        named = [x for x in (self.component, self.config_key, self.condition_key) if x is not None]
        if len(named) > 1:
            raise ValueError("a target names at most one component, configuration key or condition")
        return self

    @property
    def kind(self) -> TargetKind | None:
        """What the target names besides its scope, or None for a bare scope."""
        if self.component is not None:
            return TargetKind.component
        if self.config_key is not None:
            return TargetKind.config_key
        if self.condition_key is not None:
            return TargetKind.condition_key
        return None

    @property
    def thing(self) -> Component | ConfigKey | ConditionKey | None:
        """The component or key the target names, or None."""
        return self.component or self.config_key or self.condition_key

    @property
    def scope_kind(self) -> ScopeKind:
        return scope_kind(self.scope)

    def covers(self, robot_id: str, site_id: str) -> bool:
        """Whether the target's scope includes the given robot at the given site."""
        return scope_covers(self.scope, robot_id, site_id)


class Drift(ContractModel):
    """A fault whose level moves over the days: linearly, or in steps, at `rate`.

    How the level moves with the day is the world generator's to define (stage 3).
    """

    shape: Literal["linear", "step"]
    rate: float


class Schedule(ContractModel):
    """When a fault is in force: from `onset_day` for `duration_days` (None: to the end).

    An episode's identity carries the fault with the level of its day.
    """

    onset_day: int = Field(ge=0)
    duration_days: int | None = Field(default=None, ge=1)
    drift: Drift | None = None

    def active_on(self, day: int) -> bool:
        """Whether the fault is in force on `day`."""
        if day < self.onset_day:
            return False
        return self.duration_days is None or day < self.onset_day + self.duration_days


class FaultSpec(ContractModel):
    """One fault: its kind, parameters, level, target and schedule (truth side only)."""

    fault_id: FaultId
    kind: Annotated[str, Field(pattern=KIND_PATTERN)]
    params: dict[str, JsonValue] = Field(default_factory=dict)
    level: float = Field(ge=0.0)
    target: Target
    schedule: Schedule
    # For a fault on a configuration key or a condition: the value it replaced.
    nominal: ConfigValue = None

    @model_validator(mode="after")
    def _names_one_thing(self) -> FaultSpec:
        if self.target.kind is None:
            raise ValueError("a fault changes a component, a configuration value or a condition")
        if self.target.kind is TargetKind.component:
            if self.nominal is not None:
                raise ValueError("a fault on a component has no nominal value")
        elif self.nominal is None:
            raise ValueError(
                f"fault {self.fault_id} on a {self.target.kind.value} carries the nominal value "
                "it replaced"
            )
        return self

    @property
    def is_fault(self) -> bool:
        """A fault at level zero is no fault."""
        return self.level > 0.0


class TruthMapping(ContractModel):
    """How a fault type maps to the system description: what it acts on and what fixes it.

    `target_kind` is the kind of thing its faults change; the fault's `Target`
    names which one. `fixing_verbs` are the verbs that remove it (empty: a
    failure mode no action fixes, claim 5), `scopes` the scope kinds its faults
    sit at. `writes_change` says whether the fault appears in the change log as
    a `Change` of its key (configuration and software faults do; physical,
    sensor, site and data faults do not). `directions` are the diagnosis
    directions accepted for it. The scorer and the oracle both read this
    mapping, so they agree by construction.
    """

    target_kind: TargetKind
    fixing_verbs: frozenset[Verb]
    scopes: frozenset[ScopeKind]
    writes_change: bool
    directions: frozenset[Direction]

    @model_validator(mode="after")
    def _consistent(self) -> TruthMapping:
        wrong = sorted(
            v.value for v in self.fixing_verbs if VERBS[v].target_kind != self.target_kind
        )
        if wrong:
            raise ValueError(f"verbs {wrong} do not act on a {self.target_kind.value}")
        if not self.scopes:
            raise ValueError("a truth mapping names at least one scope kind")
        if not self.directions:
            raise ValueError("a truth mapping accepts at least one diagnosis direction")
        if self.writes_change and self.target_kind is not TargetKind.config_key:
            raise ValueError("only a fault on a configuration key writes a Change")
        if Verb.revert in self.fixing_verbs and not self.writes_change:
            raise ValueError("revert undoes a Change, so a fault it fixes writes one")
        if self.target_kind is TargetKind.config_key and self.fixing_verbs:
            missing = sorted(v.value for v in {Verb.set, Verb.revert} - self.fixing_verbs)
            if missing:
                raise ValueError(
                    f"a configuration fault is fixed by set and revert; missing {missing}"
                )
        if (
            self.target_kind is TargetKind.condition_key
            and Verb.change_condition not in self.fixing_verbs
        ):
            raise ValueError("a condition fault is fixed by change_condition")
        if self.target_kind is TargetKind.component and self.scopes != {ScopeKind.robot}:
            raise ValueError("a component fault is robot-scoped only: one fault per robot")
        return self

    def check(self, spec: FaultSpec) -> None:
        """Raise ValueError if `spec` does not fit this mapping."""
        if spec.target.kind is not self.target_kind:
            raise ValueError(
                f"fault {spec.fault_id} of kind {spec.kind} targets a "
                f"{spec.target.kind}, its type acts on a {self.target_kind.value}"
            )
        component = spec.target.component
        if component is not None and self.fixing_verbs:
            usable = [v for v in self.fixing_verbs if component in VERBS[v].components]
            if not usable:
                raise ValueError(
                    f"fault {spec.fault_id} on the {component.value}: none of its fixing verbs "
                    f"{sorted(v.value for v in self.fixing_verbs)} applies to that component"
                )
        if spec.target.scope_kind not in self.scopes:
            raise ValueError(
                f"fault {spec.fault_id} sits at a {spec.target.scope_kind.value} scope, "
                f"its type allows {sorted(s.value for s in self.scopes)}"
            )


# The truth mapping of every fault kind of a world, by kind. Built from the
# registry in `fleetlab.faults`; the contracts take it as an argument.
TruthTable = Mapping[str, TruthMapping]


@dataclass
class FaultState:
    """A fault's state within one episode. A fault type extends it with its own fields.

    `new_episode` makes a fresh one, so nothing carries over from one episode to
    the next (a delay line's queued commands are episode state).
    """

    spec: FaultSpec
    episode_id: str


@runtime_checkable
class FaultType(Protocol):
    """A kind of fault and its hooks. Implement it by extending `FaultTypeBase`.

    The data hook may change a verdict's `success` only when the verdict is
    `done`, and never `done` itself, so that a flipped verdict never ends an
    episode early or keeps it running (`check_data_hook`).
    """

    kind: str
    fault_class: FaultClass
    Params: type[BaseModel]  # the pydantic model of `FaultSpec.params`
    dose_levels: Sequence[float]  # the levels calibration runs
    truth: TruthMapping

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState: ...

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        """Change the live simulator model before the reset renders; return how to undo it."""
        ...

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        """Change what the policy and the log see."""
        ...

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        """Change the command between the action queue and the simulator."""
        ...

    def policy_input_hook(self, policy_spec: PolicySpec, state: FaultState) -> PolicySpec:
        """Change the configuration knobs the policy runs with (the effective spec)."""
        ...

    def data_hook(self, verdict: SuccessVerdict, state: FaultState) -> SuccessVerdict:
        """Change the success detector's verdict (the detector's own error).

        Only `success` of a `done` verdict may change; the reason may say why.
        """
        ...

    def snapshot(self, state: FaultState) -> bytes: ...

    def restore(self, blob: bytes) -> FaultState: ...


class FaultTypeBase:
    """Do-nothing defaults for every hook. A fault type overrides only the hooks it needs.

    Subclasses set the class attributes `kind`, `fault_class`, `Params`,
    `dose_levels` and `truth`. The default snapshot pickles the state:
    snapshots stay on the truth side and are read by the code that wrote them.
    """

    kind: str
    fault_class: FaultClass
    Params: type[BaseModel]
    dose_levels: Sequence[float]
    truth: TruthMapping

    def params(self, spec: FaultSpec) -> BaseModel:
        """The spec's parameters, validated by this type's `Params` model."""
        return self.Params.model_validate(spec.params)

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return FaultState(spec=spec, episode_id=identity.episode_id)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        return None

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        return observation

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        return action

    def policy_input_hook(self, policy_spec: PolicySpec, state: FaultState) -> PolicySpec:
        return policy_spec

    def data_hook(self, verdict: SuccessVerdict, state: FaultState) -> SuccessVerdict:
        return verdict

    def snapshot(self, state: FaultState) -> bytes:
        return pickle.dumps(state)

    def restore(self, blob: bytes) -> FaultState:
        state = pickle.loads(blob)
        if not isinstance(state, FaultState):
            raise TypeError(f"not a fault state: {type(state).__name__}")
        return state


def check_data_hook(before: SuccessVerdict, after: SuccessVerdict) -> None:
    """Raise ValueError unless a data hook kept to its contract.

    `done` is unchanged, and `success` changed only on a `done` verdict. The
    rollout calls this on every verdict a data hook returns.
    """
    if after.done != before.done:
        raise ValueError("a data hook does not change whether the episode is done")
    if after.success != before.success and not before.done:
        raise ValueError("a data hook changes success only on a done verdict")


# The fault classes whose faults appear in the change log (section 3).
CHANGE_WRITING_CLASSES = frozenset({FaultClass.configuration, FaultClass.software})


def check_fault_type(fault_type: object) -> FaultType:
    """Return `fault_type` if it implements `FaultType` consistently; raise otherwise.

    The registry in `fleetlab.faults` calls this for every type it registers.
    """
    if not isinstance(fault_type, FaultType):
        raise TypeError(f"{fault_type!r} does not implement FaultType")
    if not re.match(KIND_PATTERN, fault_type.kind):
        raise ValueError(f"fault kind {fault_type.kind!r} is not a lower-case identifier")
    if not isinstance(fault_type.fault_class, FaultClass):
        raise TypeError(f"fault kind {fault_type.kind}: fault_class must be a FaultClass")
    if not (isinstance(fault_type.Params, type) and issubclass(fault_type.Params, BaseModel)):
        raise TypeError(f"fault kind {fault_type.kind}: Params must be a pydantic model class")
    if not isinstance(fault_type.truth, TruthMapping):
        raise TypeError(f"fault kind {fault_type.kind}: truth must be a TruthMapping")
    writes = fault_type.fault_class in CHANGE_WRITING_CLASSES
    if fault_type.truth.writes_change != writes:
        raise ValueError(
            f"fault kind {fault_type.kind}: a {fault_type.fault_class.value} fault "
            f"{'writes' if writes else 'writes no'} Change (writes_change={writes})"
        )
    return fault_type


class LogSide(StrEnum):
    """On which side of a hook the episode log listens."""

    before = "before"
    after = "after"
    none = "none"


class HookPoint(NamedTuple):
    """One stage of an episode as the rollout runs it, and where the log listens."""

    name: str
    is_fault_hook: bool
    per_step: bool
    when: str
    log_side: LogSide
    log_note: str


# The fixed order of the stages. On every step: observation hook, then policy,
# then action hook, then simulator, then the success check with the data hook.
# The settle steps of the reset bypass all of them. The log listens where a
# fleet would.
HOOK_ORDER: tuple[HookPoint, ...] = (
    HookPoint(
        "new_episode",
        True,
        False,
        "before the reset: every active fault gets a fresh state for this episode",
        LogSide.none,
        "nothing is logged",
    ),
    HookPoint(
        "cell_hook",
        True,
        False,
        "during the reset, through the CellHandle, before anything is rendered; "
        "undone after the episode, with the texture upload repeated",
        LogSide.after,
        "the log sees the changed cell",
    ),
    HookPoint(
        "policy_input_hook",
        True,
        False,
        "before the first policy call: the configuration knobs the policy runs with",
        LogSide.none,
        "the effective spec is recorded on the truth side (effective_policy)",
    ),
    HookPoint(
        "observation_hook",
        True,
        True,
        "between the sensors and what the policy and the log receive",
        LogSide.after,
        "every observation signal, images and robot state, after the hook",
    ),
    HookPoint(
        "policy",
        False,
        True,
        "the policy computes a chunk, or the rollout's queue hands out the next action",
        LogSide.after,
        "the commanded action",
    ),
    HookPoint(
        "action_hook",
        True,
        True,
        "between the action queue and the simulator",
        LogSide.before,
        "the visible log records the commanded action; the executed action is truth (chain)",
    ),
    HookPoint(
        "simulator",
        False,
        True,
        "one control step of the simulator",
        LogSide.after,
        "the truth side records the state and image hashes",
    ),
    HookPoint(
        "data_hook",
        True,
        True,
        "after the success check of the step; it may change success only on a done verdict, "
        "never done itself; the episode ends on the verdict after the hook, which is what the "
        "fleet's runner sees",
        LogSide.after,
        "the visible outcome is the verdict after the hook; the truth side records the verdict "
        "before it, LIBERO's first-fire step and the fleet check's step (table verdicts)",
    ),
)

STEP_ORDER: tuple[str, ...] = tuple(point.name for point in HOOK_ORDER if point.per_step)
FAULT_HOOKS: tuple[str, ...] = tuple(point.name for point in HOOK_ORDER if point.is_fault_hook)


def active_faults(
    faults: Sequence[FaultSpec], fleet: Fleet, robot_id: str, day: int
) -> list[FaultSpec]:
    """The faults in force for one robot on one day, as an identity lists them.

    In scope, scheduled for the day, level above zero, and not shadowed: a
    configuration fault at a wider scope does not reach the robot when a
    narrower scope covering it (its site, or the robot itself) sets the same
    key in the fleet's model. A configuration fault writes its own `Change`,
    so the model holds its value at its own scope, and of two faults on one
    key only the narrower is in force.
    """
    robot = fleet.robot(robot_id)
    site = fleet.site(robot.site_id)
    layers = {
        SCOPE_RANK[ScopeKind.fleet]: fleet.config.values,
        SCOPE_RANK[ScopeKind.site]: site.config.values,
        SCOPE_RANK[ScopeKind.robot]: robot.config.values,
    }

    def shadowed(fault: FaultSpec) -> bool:
        key = fault.target.config_key
        if key is None:
            return False
        rank = scope_rank(fault.target.scope)
        return any(key in values for r, values in layers.items() if r > rank)

    return [
        f
        for f in faults
        if f.is_fault
        and f.schedule.active_on(day)
        and f.target.covers(robot.id, robot.site_id)
        and not shadowed(f)
    ]
