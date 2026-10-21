"""The one vocabulary of the test bed (docs/design/contracts.md, principle 5 and section 2).

Component names, configuration keys, condition keys, action verbs and the six
generic fault classes are defined here and nowhere else. The world generator,
the truth tables, the scorer, the tool schemas and the skills import them from
this module; the walking skeleton wrote them twice and they drifted.

Fault *kinds* are not here, by design: the fault registry lives in
`fleetlab.faults`, which is never mounted into an agent's sandbox, so the
unseen fault types cannot be read off this module.

Every term carries a one-line description, which is what the agents read in the
system description (`vocabulary_json`). Descriptions are written as a fleet
operator would read them and say nothing about how faults are made.

Each verb also carries `VerbInfo`: what it names besides its scope, at which
scopes it applies, whether it takes a value, and how `contracts.actions.resolve`
treats it, and for a component verb the components it applies to.
`route_to_teleop` is fleet-scoped with the class of work as its value, which
gives the catalogue one entry per task class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fleetlab.contracts.ids import ScopeKind
from fleetlab.contracts.version import CONTRACTS_VERSION


class Component(StrEnum):
    """The components of a robot, version 1."""

    scene_camera = "scene_camera"
    wrist_camera = "wrist_camera"
    gripper = "gripper"
    arm = "arm"
    controller = "controller"


class ConfigKey(StrEnum):
    """The configuration keys, version 1. Each is set at fleet, site or robot scope."""

    policy_sampling_steps = "policy.sampling_steps"
    policy_actions_executed_per_chunk = "policy.actions_executed_per_chunk"
    policy_image_views = "policy.image_views"
    policy_precision = "policy.precision"
    control_mode = "control.mode"
    control_action_delay_steps = "control.action_delay_steps"
    control_latency_compensation_steps = "control.latency_compensation_steps"
    camera_scene_calibration = "camera.scene.calibration"
    camera_wrist_calibration = "camera.wrist.calibration"
    software_version = "software.version"


# Configuration keys added after fleets had been stored, with their nominal values. A
# fleet has such a setting only if its spec says so (`WorldSpec.later_settings`): its
# configuration then records the nominal value at fleet scope (narrower scopes may override
# it, never hold it alone: `model.Fleet`), its description's vocabulary lists the key and
# the catalogue offers it (`contracts.actions.catalogue`). A fleet composed before the key
# existed has none of this, so it keeps its bytes, its catalogue and its description, and
# no action names a value its description lacks.
LATER_CONFIG_NOMINAL: dict[ConfigKey, int] = {ConfigKey.control_latency_compensation_steps: 0}
LATER_CONFIG_KEYS: frozenset[ConfigKey] = frozenset(LATER_CONFIG_NOMINAL)


class ConditionKey(StrEnum):
    """Site-level circumstances, version 1."""

    lighting = "lighting"
    placement = "placement"
    object_batch = "object_batch"
    clutter = "clutter"


class Verb(StrEnum):
    """What a fleet operator can do, version 1.

    The member for "replace" is `Verb.replace_`: a member named `replace` would
    shadow `str.replace`, and `Verb.replace` silently resolves to that string
    method, not to a verb. `Verb("replace")` gives the member.
    """

    recalibrate = "recalibrate"
    clean = "clean"
    replace_ = "replace"  # the trailing underscore keeps str.replace usable
    service = "service"
    set = "set"
    revert = "revert"
    change_condition = "change_condition"
    route_to_teleop = "route_to_teleop"
    request_training_data = "request_training_data"


class FaultClass(StrEnum):
    """The six generic classes of fault. A fault type belongs to exactly one."""

    physical = "physical"
    sensor = "sensor"
    configuration = "configuration"
    software = "software"
    site = "site"
    data = "data"


class TargetKind(StrEnum):
    """What a verb or a fault names besides its scope."""

    component = "component"
    config_key = "config_key"
    condition_key = "condition_key"


_DESCRIPTIONS: dict[StrEnum, str] = {
    Component.scene_camera: "the fixed camera that views the work cell",
    Component.wrist_camera: "the camera mounted on the robot's wrist",
    Component.gripper: "the two-finger gripper, its fingers and pads",
    Component.arm: "the seven-joint arm with its motors and joint sensors",
    Component.controller: "the computer that runs the policy and the control stack",
    ConfigKey.policy_sampling_steps: "denoising steps the policy runs per call (integer)",
    ConfigKey.policy_actions_executed_per_chunk: "how many actions of each returned chunk "
    "are executed before the policy is called again (integer)",
    ConfigKey.policy_image_views: "camera views fed to the policy (integer)",
    ConfigKey.policy_precision: "numeric precision the policy runs in (string)",
    ConfigKey.control_mode: "whether actions are target poses or changes of pose "
    "(absolute or relative)",
    ConfigKey.control_action_delay_steps: "control steps a command waits in the command "
    "buffer before it is executed (integer)",
    ConfigKey.control_latency_compensation_steps: "actions skipped at the start of each chunk "
    "after an attempt's first, for commands that reach the robot that many control steps late "
    "(integer, 0: none)",
    ConfigKey.camera_scene_calibration: "the pose the fleet believes its scene camera has",
    ConfigKey.camera_wrist_calibration: "the pose the fleet believes its wrist camera has",
    ConfigKey.software_version: "version of the robot software stack (string)",
    ConditionKey.lighting: "the site's lighting",
    ConditionKey.placement: "placement of table and furniture within installation tolerance",
    ConditionKey.object_batch: "the batch of objects in use: mass, friction, appearance",
    ConditionKey.clutter: "how much unrelated clutter stands in the work cell",
    Verb.recalibrate: "recalibrate a component of a robot",
    Verb.clean: "clean a component of a robot",
    Verb.replace_: "replace a component of a robot with a new one",
    Verb.service: "service a component of a robot",
    Verb.set: "set a configuration key to a value at a scope",
    Verb.revert: "revert the change of a configuration key made at a scope",
    Verb.change_condition: "change a condition at a site to a value",
    Verb.route_to_teleop: "route a class of work to teleoperation",
    Verb.request_training_data: "request training data for a failure mode",
    FaultClass.physical: "wear or damage of a mechanical part",
    FaultClass.sensor: "a sensor that reports or sees something other than it should",
    FaultClass.configuration: "a configuration value that is wrong for the work",
    FaultClass.software: "a software change that alters how the robot behaves",
    FaultClass.site: "a circumstance of the site the robot works in",
    FaultClass.data: "a record that says something other than what happened",
    TargetKind.component: "a component of a robot",
    TargetKind.config_key: "a configuration key",
    TargetKind.condition_key: "a site condition",
}


def describe(term: StrEnum) -> str:
    """The one-line description of a vocabulary term."""
    return _DESCRIPTIONS[term]


# The enums that make up the vocabulary, under the names the system description uses.
VOCABULARY: dict[str, type[StrEnum]] = {
    "components": Component,
    "config_keys": ConfigKey,
    "condition_keys": ConditionKey,
    "verbs": Verb,
    "fault_classes": FaultClass,
}


@dataclass(frozen=True)
class VerbInfo:
    """What a verb applies to and how `contracts.actions.resolve` treats it.

    `target_kind` is what the action's target names besides its scope; None for
    the two verbs that name no part of the model, whose `value` says what they
    apply to (a class of work, a failure mode). `exact_scope` marks `revert`,
    which removes only the fault whose change was made at exactly the action's
    scope; for the other verbs a scope at least as narrow as the fault's
    overrides it for the robots it covers. `adds_change` marks the verbs that
    take a value and add a change to the model when that value is not the
    nominal one. `has_replay` is False for the two verbs the oracle cannot value
    by replay; they are scored under claim 5. `components` are the components a
    component verb applies to: `recalibrate` the two cameras and the arm, `clean`
    the two cameras and the gripper, `replace` and `service` all five; empty for the other verbs.
    """

    target_kind: TargetKind | None
    scope_kinds: frozenset[ScopeKind]
    takes_value: bool
    value_meaning: str
    exact_scope: bool
    adds_change: bool
    has_replay: bool
    components: frozenset[Component] = frozenset()


_ROBOT = frozenset({ScopeKind.robot})
_SITE = frozenset({ScopeKind.site})
_FLEET = frozenset({ScopeKind.fleet})
_ANY = frozenset(ScopeKind)

COMPONENT_VERBS = frozenset({Verb.recalibrate, Verb.clean, Verb.replace_, Verb.service})

_CAMERAS = frozenset({Component.scene_camera, Component.wrist_camera})
_APPLIES_TO: dict[Verb, frozenset[Component]] = {
    Verb.recalibrate: _CAMERAS | {Component.arm},
    Verb.clean: _CAMERAS | {Component.gripper},
    Verb.replace_: frozenset(Component),
    Verb.service: frozenset(Component),
}

VERBS: dict[Verb, VerbInfo] = {
    **{
        verb: VerbInfo(
            target_kind=TargetKind.component,
            scope_kinds=_ROBOT,
            takes_value=False,
            value_meaning="",
            exact_scope=False,
            adds_change=False,
            has_replay=True,
            components=components,
        )
        for verb, components in _APPLIES_TO.items()
    },
    Verb.set: VerbInfo(
        target_kind=TargetKind.config_key,
        scope_kinds=_ANY,
        takes_value=True,
        value_meaning="the new value",
        exact_scope=False,
        adds_change=True,
        has_replay=True,
    ),
    Verb.revert: VerbInfo(
        target_kind=TargetKind.config_key,
        scope_kinds=_ANY,
        takes_value=False,
        value_meaning="",
        exact_scope=True,
        adds_change=False,
        has_replay=True,
    ),
    Verb.change_condition: VerbInfo(
        target_kind=TargetKind.condition_key,
        scope_kinds=_SITE,
        takes_value=True,
        value_meaning="the new condition",
        exact_scope=False,
        adds_change=True,
        has_replay=True,
    ),
    Verb.route_to_teleop: VerbInfo(
        target_kind=None,
        scope_kinds=_FLEET,
        takes_value=True,
        value_meaning="the class of work, as task text",
        exact_scope=False,
        adds_change=False,
        has_replay=False,
    ),
    Verb.request_training_data: VerbInfo(
        target_kind=None,
        scope_kinds=_FLEET,
        takes_value=True,
        value_meaning="the failure mode, in words",
        exact_scope=False,
        adds_change=False,
        has_replay=False,
    ),
}

NO_REPLAY_VERBS = frozenset(verb for verb, info in VERBS.items() if not info.has_replay)


def verbs_for(kind: TargetKind | None) -> frozenset[Verb]:
    """The verbs whose target names a thing of this kind."""
    return frozenset(verb for verb, info in VERBS.items() if info.target_kind == kind)


def vocabulary_json(hidden: frozenset[ConfigKey] = frozenset()) -> dict[str, Any]:
    """The whole vocabulary as JSON, for the agents' system description.

    Each term comes with its description; verbs also say what they apply to,
    at which scopes, and what their value means. How the oracle values a verb
    is not included. `hidden` leaves out keys of `LATER_CONFIG_KEYS` a fleet
    does not have (`model.describe_for_agents`).
    """
    if not hidden <= LATER_CONFIG_KEYS:
        raise ValueError("only a key added later can be left out of the vocabulary")
    out: dict[str, Any] = {"contracts_version": CONTRACTS_VERSION}
    for name, enum in VOCABULARY.items():
        if enum is Verb:
            out[name] = [
                {
                    "name": verb.value,
                    "description": describe(verb),
                    "target": None if info.target_kind is None else info.target_kind.value,
                    "scopes": [kind.value for kind in ScopeKind if kind in info.scope_kinds],
                    "components": [c.value for c in Component if c in info.components],
                    "value": info.value_meaning if info.takes_value else None,
                }
                for verb, info in VERBS.items()
            ]
        else:
            out[name] = [
                {"name": term.value, "description": describe(term)}
                for term in enum
                if term not in hidden
            ]
    return out


def is_vocabulary(value: Any) -> bool:
    """Whether `value` is the vocabulary as some fleet's description carries it: the whole
    of it, or without some of the keys added later."""
    keys = sorted(LATER_CONFIG_KEYS)
    for mask in range(2 ** len(keys)):
        hidden = frozenset(k for i, k in enumerate(keys) if mask >> i & 1)
        if value == vocabulary_json(hidden):
            return True
    return False
