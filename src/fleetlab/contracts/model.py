"""The system model: fleet, sites, robots, components, configuration and logs
(docs/design/contracts.md, section 3).

Pure data, serialisable to JSON, with neutral identifiers. Faults change this
model (section 7) and actions change it back or onward (section 8). The faults
themselves are not part of it: they live in the world's fault schedule on the
truth side.

The description and the change log hold the fleet's own records, as the fleet
believes them, and nothing marks an entry as a fault. A configuration fault is
a `Change` with its true key and value and an author drawn like a benign
change's; a software fault is a `Change` of `software.version`; physical,
sensor, site and data faults produce no `Change`.

Every field carries a visibility tag (`Annotated[type, VISIBLE]` or
`HIDDEN`), like the columns of the log. `describe_for_agents` builds the system
description from the fields tagged visible, never by removing hidden ones, and
a field without a tag makes it fail. `Robot.traits` is hidden.

Choices this module makes where the note is silent:

- `Change.key` is a configuration key: conditions are not in the change log.
- `Fleet.id` is the world's neutral id (`w..`): a world is one generated fleet.
- The description includes the vocabulary, the world's task classes, the
  costs and the number of catalogue entries (the static structure of section
  12); the catalogue itself is the visible table `catalogue`.
  `include_logs=False` leaves out the change and maintenance logs for the
  no-data baseline. Cutting the logs to the
  observed window is the caller's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, JsonValue, model_validator

from fleetlab.contracts._base import (
    HIDDEN,
    VISIBLE,
    ConfigValue,
    ContractModel,
    Visibility,
    visible_dump,
)
from fleetlab.contracts.ids import FLEET_SCOPE, RobotId, Scope, SiteId, WorldId, scope_covers
from fleetlab.contracts.version import CONTRACTS_VERSION
from fleetlab.contracts.vocabulary import (
    COMPONENT_VERBS,
    LATER_CONFIG_KEYS,
    Component,
    ConditionKey,
    ConfigKey,
    Verb,
    vocabulary_json,
)

__all__ = [
    "HIDDEN",
    "VISIBLE",
    "Change",
    "ChangeAuthor",
    "ComponentState",
    "ConfigVersion",
    "Fleet",
    "Maintenance",
    "Robot",
    "Site",
    "Visibility",
    "describe_for_agents",
    "effective_config",
]


class ChangeAuthor(StrEnum):
    """Who made a change-log entry."""

    ops = "ops"
    vendor = "vendor"
    auto = "auto"


class ComponentState(ContractModel):
    """A component of one robot: when it was installed and serviced, and its health readings."""

    component: Annotated[Component, VISIBLE]
    installed_day: Annotated[int, VISIBLE]
    last_service_day: Annotated[int | None, VISIBLE] = None
    health: Annotated[dict[str, float], VISIBLE] = Field(default_factory=dict)


class ConfigVersion(ContractModel):
    """The configuration values set at one scope, in their current version."""

    version: Annotated[int, VISIBLE] = Field(ge=0)
    values: Annotated[dict[ConfigKey, ConfigValue], VISIBLE] = Field(default_factory=dict)
    since_day: Annotated[int, VISIBLE] = 0


class Change(ContractModel):
    """One entry of the change log: a configuration value changed at a scope on a day."""

    day: Annotated[int, VISIBLE] = Field(ge=0)
    scope: Annotated[Scope, VISIBLE]
    key: Annotated[ConfigKey, VISIBLE]
    old: Annotated[ConfigValue, VISIBLE]
    new: Annotated[ConfigValue, VISIBLE]
    author: Annotated[ChangeAuthor, VISIBLE]


class Maintenance(ContractModel):
    """One entry of the maintenance log: a component verb carried out on a robot."""

    day: Annotated[int, VISIBLE] = Field(ge=0)
    robot_id: Annotated[RobotId, VISIBLE]
    component: Annotated[Component, VISIBLE]
    verb: Annotated[Verb, VISIBLE]
    note: Annotated[str, VISIBLE] = ""

    @model_validator(mode="after")
    def _component_verb(self) -> Maintenance:
        if self.verb not in COMPONENT_VERBS:
            raise ValueError(f"maintenance is a component verb, not {self.verb.value}")
        return self


def _empty_config() -> ConfigVersion:
    return ConfigVersion(version=0)


class Robot(ContractModel):
    """One robot: its components, its own configuration, and its traits (hidden).

    Traits are per-robot constants the generator draws (a camera mounting
    tolerance, a gripper pad age). The fault schedule uses them; they explain
    nothing by themselves and are never exported.
    """

    id: Annotated[RobotId, VISIBLE]
    site_id: Annotated[SiteId, VISIBLE]
    components: Annotated[dict[Component, ComponentState], VISIBLE]
    config: Annotated[ConfigVersion, VISIBLE] = Field(default_factory=_empty_config)
    traits: Annotated[dict[str, JsonValue], HIDDEN] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _components_named(self) -> Robot:
        for name, state in self.components.items():
            if state.component is not name:
                raise ValueError(f"robot {self.id}: component {name.value} holds {state.component}")
        return self


class Site(ContractModel):
    """One site: its robots, its conditions, and its configuration.

    `notes` is what an operator knows about how the site is installed, as
    sentences (for example that its robots are driven over a remote link);
    never the size or effect of anything. Left out of every serialized form,
    the description included, when empty, so older fleets keep their bytes.
    """

    id: Annotated[SiteId, VISIBLE]
    robots: Annotated[list[Robot], VISIBLE]
    conditions: Annotated[dict[ConditionKey, ConfigValue], VISIBLE] = Field(default_factory=dict)
    config: Annotated[ConfigVersion, VISIBLE] = Field(default_factory=_empty_config)
    notes: Annotated[list[str], VISIBLE] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def _robots_here(self) -> Site:
        for robot in self.robots:
            if robot.site_id != self.id:
                raise ValueError(
                    f"robot {robot.id} says site {robot.site_id} but stands at {self.id}"
                )
        return self


class Fleet(ContractModel):
    """The whole fleet: sites, fleet-wide configuration, change and maintenance logs."""

    id: Annotated[WorldId, VISIBLE]
    sites: Annotated[list[Site], VISIBLE]
    config: Annotated[ConfigVersion, VISIBLE] = Field(default_factory=_empty_config)
    change_log: Annotated[list[Change], VISIBLE] = Field(default_factory=list)
    maintenance_log: Annotated[list[Maintenance], VISIBLE] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique_and_known(self) -> Fleet:
        site_ids = [s.id for s in self.sites]
        robot_ids = [r.id for r in self.robots()]
        for kind, ids in (("site", site_ids), ("robot", robot_ids)):
            if len(ids) != len(set(ids)):
                raise ValueError(f"{kind} ids repeat: {sorted(ids)}")
        known = {FLEET_SCOPE, *site_ids, *robot_ids}
        unknown = sorted({c.scope for c in self.change_log} - known)
        unknown += sorted({m.robot_id for m in self.maintenance_log} - set(robot_ids))
        if unknown:
            raise ValueError(f"the logs name scopes that are not in the fleet: {unknown}")
        # A key added later exists as a fleet default, overridden at narrower scopes or not;
        # a fleet without the default does not have the setting (vocabulary.LATER_CONFIG_KEYS).
        narrower = {
            key
            for config in (*(s.config for s in self.sites), *(r.config for r in self.robots()))
            for key in config.values
        }
        alone = sorted(k.value for k in (narrower & LATER_CONFIG_KEYS) - set(self.config.values))
        if alone:
            raise ValueError(f"keys {alone} are set at a site or robot without the fleet default")
        return self

    def robots(self) -> list[Robot]:
        """Every robot, site by site in order."""
        return [robot for site in self.sites for robot in site.robots]

    def robot(self, robot_id: str) -> Robot:
        for robot in self.robots():
            if robot.id == robot_id:
                return robot
        raise KeyError(f"no robot {robot_id!r} in fleet {self.id}")

    def site(self, site_id: str) -> Site:
        for site in self.sites:
            if site.id == site_id:
                return site
        raise KeyError(f"no site {site_id!r} in fleet {self.id}")

    def robots_in(self, scope: str) -> list[Robot]:
        """The robots a scope covers."""
        return [r for r in self.robots() if scope_covers(scope, r.id, r.site_id)]


MODEL_CLASSES: tuple[type[ContractModel], ...] = (
    ComponentState,
    ConfigVersion,
    Change,
    Maintenance,
    Robot,
    Site,
    Fleet,
)


def effective_config(fleet: Fleet, robot_id: str) -> dict[ConfigKey, ConfigValue]:
    """A robot's effective configuration: robot over site over fleet."""
    robot = fleet.robot(robot_id)
    site = fleet.site(robot.site_id)
    return {**fleet.config.values, **site.config.values, **robot.config.values}


_LOGS = frozenset({"change_log", "maintenance_log"})


def describe_for_agents(
    fleet: Fleet, include_logs: bool = True, task_classes: Sequence[str] = ()
) -> dict[str, Any]:
    """The system description the agents receive (`description.json`): no traits, no fault.

    The vocabulary, the fleet's visible fields (structure, components,
    configuration, conditions and, with `include_logs`, the change and
    maintenance logs), the world's task classes, the action costs, and the
    number of catalogue entries. The catalogue itself is the visible table
    `catalogue` (`contracts.actions.catalogue_rows`), which the agent queries:
    whole, it made the description too large for a tool result.
    Built from the allow-list of visible fields; `include_logs=False` gives
    the no-data baseline's static structure. The agents read it in parts
    (`contracts.tools.describe_part`).
    """
    from fleetlab.contracts.actions import catalogue, costs_rows  # actions builds on this module

    return {
        "contracts_version": CONTRACTS_VERSION,
        "vocabulary": vocabulary_json(LATER_CONFIG_KEYS - set(fleet.config.values)),
        "fleet": visible_dump(fleet, drop=frozenset() if include_logs else _LOGS),
        "task_classes": list(task_classes),
        "costs": costs_rows(),
        "catalogue_entries": len(catalogue(fleet, task_classes)),
    }
