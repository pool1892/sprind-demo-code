"""The system model: every field tagged, the effective configuration merged robot over site
over fleet, and a system description without traits.

docs/design/contracts.md, section 3.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from fleetlab.contracts._base import (
    HIDDEN,
    VISIBLE,
    ConfigValue,
    field_visibility,
    untagged_fields,
)
from fleetlab.contracts.actions import catalogue_rows
from fleetlab.contracts.model import (
    MODEL_CLASSES,
    Change,
    ChangeAuthor,
    ComponentState,
    ConfigVersion,
    Fleet,
    Maintenance,
    Robot,
    Site,
    describe_for_agents,
    effective_config,
)
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb

SECRET = "pad-age-of-a-worn-gripper"


def robot(robot_id: str, site_id: str, delay: int | None = None) -> Robot:
    values: dict[ConfigKey, ConfigValue] = (
        {} if delay is None else {ConfigKey.control_action_delay_steps: delay}
    )
    return Robot(
        id=robot_id,
        site_id=site_id,
        components={c: ComponentState(component=c, installed_day=0) for c in Component},
        config=ConfigVersion(version=1 if values else 0, values=values),
        traits={"gripper_pad_age": SECRET},
    )


def fleet() -> Fleet:
    return Fleet(
        id="w01",
        config=ConfigVersion(
            version=1,
            values={
                ConfigKey.control_action_delay_steps: 0,
                ConfigKey.policy_precision: "float32",
            },
        ),
        sites=[
            Site(
                id="s1",
                robots=[robot("r01", "s1"), robot("r02", "s1", delay=2)],
                conditions={ConditionKey.lighting: "bright"},
            ),
            Site(
                id="s2",
                robots=[robot("r03", "s2")],
                config=ConfigVersion(version=2, values={ConfigKey.control_action_delay_steps: 3}),
            ),
        ],
        change_log=[
            Change(
                day=4,
                scope="s2",
                key=ConfigKey.control_action_delay_steps,
                old=0,
                new=3,
                author=ChangeAuthor.ops,
            )
        ],
        maintenance_log=[
            Maintenance(day=2, robot_id="r01", component=Component.gripper, verb=Verb.service)
        ],
    )


def test_every_model_field_carries_a_visibility_tag() -> None:
    for model in MODEL_CLASSES:
        assert untagged_fields(model) == [], model.__name__
    assert field_visibility(Robot, "traits") is HIDDEN
    assert field_visibility(Robot, "components") is VISIBLE


def test_effective_configuration_is_robot_over_site_over_fleet() -> None:
    f = fleet()
    delay = ConfigKey.control_action_delay_steps
    assert effective_config(f, "r01")[delay] == 0  # fleet
    assert effective_config(f, "r02")[delay] == 2  # robot over fleet
    assert effective_config(f, "r03")[delay] == 3  # site over fleet
    assert effective_config(f, "r03")[ConfigKey.policy_precision] == "float32"


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_the_system_description_has_no_traits() -> None:
    description = describe_for_agents(fleet())
    text = json.dumps(description)
    assert "traits" not in _keys(description)
    assert SECRET not in text
    sites = description["fleet"]["sites"]
    assert [r["id"] for r in sites[0]["robots"]] == ["r01", "r02"]
    assert description["fleet"]["change_log"][0]["new"] == 3
    assert {"vocabulary", "costs", "catalogue_entries", "contracts_version"} <= set(description)
    assert "catalogue" not in description  # it is the table catalogue


def test_the_description_without_logs_is_the_static_structure() -> None:
    tasks = ["pick up the black bowl", "put the bowl on the plate"]
    description = describe_for_agents(fleet(), include_logs=False, task_classes=tasks)
    assert "change_log" not in description["fleet"]
    assert "maintenance_log" not in description["fleet"]
    assert description["task_classes"] == tasks
    # The catalogue is the table catalogue; the description counts its rows.
    rows = catalogue_rows(fleet(), tasks)
    assert description["catalogue_entries"] == len(rows)
    routes = [r for r in rows if r["verb"] == "route_to_teleop"]
    assert [r["value"] for r in routes] == tasks  # one per task class
    # Every row carries its cost; request_training_data is not costed.
    hours = {(r["verb"], r["component"]): r["hours"] for r in rows}
    assert hours[("recalibrate", "scene_camera")] == 0.5
    assert hours[("replace", "controller")] == 16
    assert hours[("request_training_data", None)] is None
    assert {r["scope"] for r in rows if r["verb"] == "revert"} == {
        "fleet",
        "s1",
        "s2",
        "r01",
        "r02",
        "r03",
    }


def test_a_new_field_without_a_tag_cannot_reach_the_description() -> None:
    class Untagged(ComponentState):
        serial: str = "S-1"

    from fleetlab.contracts._base import visible_dump

    with pytest.raises(TypeError, match="no visibility tag"):
        visible_dump(Untagged(component=Component.arm, installed_day=0))


def test_the_model_refuses_inconsistent_structure() -> None:
    with pytest.raises(ValidationError, match="stands at"):
        Site(id="s1", robots=[robot("r01", "s2")])
    with pytest.raises(ValidationError, match="repeat"):
        Fleet(id="w01", sites=[Site(id="s1", robots=[robot("r01", "s1"), robot("r01", "s1")])])
    with pytest.raises(ValidationError, match="not in the fleet"):
        Fleet(
            id="w01",
            sites=[Site(id="s1", robots=[robot("r01", "s1")])],
            change_log=[
                Change(
                    day=0,
                    scope="s9",
                    key=ConfigKey.software_version,
                    old="1",
                    new="2",
                    author=ChangeAuthor.vendor,
                )
            ],
        )
    with pytest.raises(ValidationError, match="component verb"):
        Maintenance(day=0, robot_id="r01", component=Component.arm, verb=Verb.revert)
    with pytest.raises(ValidationError):
        Fleet(id="fleet-alpha", sites=[])


def test_the_model_round_trips_through_json() -> None:
    f = fleet()
    assert Fleet.model_validate_json(f.model_dump_json()) == f
    assert f.robots_in("s1") == f.sites[0].robots
    assert [r.id for r in f.robots_in("fleet")] == ["r01", "r02", "r03"]
