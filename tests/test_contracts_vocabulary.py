"""The vocabulary is single-sourced and free of fault kinds; identifiers are neutral.

docs/design/contracts.md, principles 5 and 6.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import fleetlab.contracts as contracts
from fleetlab.contracts import ids, tools
from fleetlab.contracts.actions import ADDED_CHANGE_KINDS, Action
from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.model import Change, ComponentState, Maintenance
from fleetlab.contracts.vocabulary import (
    VERBS,
    VOCABULARY,
    Component,
    ConditionKey,
    ConfigKey,
    FaultClass,
    Verb,
    describe,
    vocabulary_json,
)

CONTRACTS_DIR = Path(contracts.__file__).parent


def all_terms() -> list[StrEnum]:
    return [term for enum in VOCABULARY.values() for term in enum]


def test_the_vocabulary_holds_exactly_the_five_lists() -> None:
    assert set(VOCABULARY) == {
        "components",
        "config_keys",
        "condition_keys",
        "verbs",
        "fault_classes",
    }
    assert [c.value for c in Component] == [
        "scene_camera",
        "wrist_camera",
        "gripper",
        "arm",
        "controller",
    ]
    assert len(ConfigKey) == 10
    assert "policy.actions_executed_per_chunk" in {k.value for k in ConfigKey}
    assert {"camera.scene.calibration", "camera.wrist.calibration"} <= {k.value for k in ConfigKey}
    assert len(FaultClass) == 6
    assert len(Verb) == 9


def test_every_term_has_a_one_line_description() -> None:
    for term in all_terms():
        text = describe(term)
        assert text.strip() and "\n" not in text, term


def test_every_verb_has_its_metadata() -> None:
    assert set(VERBS) == set(Verb)
    no_replay = {v for v, info in VERBS.items() if not info.has_replay}
    assert no_replay == {Verb.route_to_teleop, Verb.request_training_data}
    assert [v for v, info in VERBS.items() if info.exact_scope] == [Verb.revert]
    # Which component each component verb applies to.
    cams = {Component.scene_camera, Component.wrist_camera}
    assert VERBS[Verb.recalibrate].components == cams | {Component.arm}
    assert VERBS[Verb.clean].components == cams | {Component.gripper}
    assert VERBS[Verb.replace_].components == VERBS[Verb.service].components == set(Component)
    assert not VERBS[Verb.set].components
    verbs = {v["name"]: v for v in vocabulary_json()["verbs"]}
    assert verbs["clean"]["components"] == ["scene_camera", "wrist_camera", "gripper"]


def test_vocabulary_json_lists_every_term_with_its_description() -> None:
    data = vocabulary_json()
    json.dumps(data)  # serialisable
    for name, enum in VOCABULARY.items():
        assert [item["name"] for item in data[name]] == [term.value for term in enum]
        assert all(item["description"] for item in data[name])
    assert "has_replay" not in json.dumps(data)  # how the oracle values a verb stays out


def test_contracts_never_import_the_fault_registry() -> None:
    # In a fresh process: importing the contracts must not load fleetlab.faults.
    code = "import sys, fleetlab.contracts; print('fleetlab.faults' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
    for path in CONTRACTS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("fleetlab.faults"), path.name
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("fleetlab.faults") for a in node.names), path.name


def test_the_vocabulary_names_no_fault_kind() -> None:
    import fleetlab.faults

    kinds = fleetlab.faults.load_all()
    if not kinds:
        pytest.skip("no fault type is registered yet")
    text = json.dumps(vocabulary_json())
    vocabulary_source = (CONTRACTS_DIR / "vocabulary.py").read_text()
    sources = {path.name: path.read_text() for path in CONTRACTS_DIR.glob("*.py")}
    for kind in kinds:
        assert kind not in text
        assert kind not in vocabulary_source
        if kind not in ADDED_CHANGE_KINDS:
            assert not [name for name, source in sources.items() if kind in source], kind


def test_the_added_change_kinds_are_the_one_known_exception() -> None:
    assert ADDED_CHANGE_KINDS == ("config_change", "condition_change")


def test_the_replace_verb_is_reached_by_its_value() -> None:
    assert Verb("replace") is Verb.replace_
    assert Verb.replace_.value == "replace"


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def test_component_key_and_verb_names_are_spelled_only_in_the_vocabulary() -> None:
    terms = {t.value for enum in (Component, ConfigKey, ConditionKey, Verb) for t in enum}
    for path in CONTRACTS_DIR.glob("*.py"):
        if path.name == "vocabulary.py":
            continue
        spelled = terms & _string_literals(path)
        assert not spelled, f"{path.name} spells {sorted(spelled)}; import them from vocabulary"


def _enum_of(model: type[BaseModel], *path: str) -> list[str]:
    node = tools.json_schema_strict(model)
    for step in path:
        node = node["items"] if step == "[]" else node["properties"][step]
    if "anyOf" in node:
        node = next(option for option in node["anyOf"] if option.get("type") != "null")
    return list(node["enum"])


def test_tool_schemas_take_their_terms_from_the_vocabulary() -> None:
    target = ("causes", "[]", "target")
    assert _enum_of(tools.Diagnosis, *target, "component") == [c.value for c in Component]
    assert _enum_of(tools.Diagnosis, *target, "config_key") == [k.value for k in ConfigKey]
    assert _enum_of(tools.Diagnosis, *target, "condition_key") == [k.value for k in ConditionKey]
    assert _enum_of(tools.Diagnosis, "causes", "[]", "direction") == [d.value for d in Direction]
    assert _enum_of(tools.RankedFixSet, "actions", "[]", "action", "verb") == [
        v.value for v in Verb
    ]


def test_model_and_actions_use_the_vocabulary_types() -> None:
    assert ComponentState.model_fields["component"].annotation is Component
    assert Change.model_fields["key"].annotation is ConfigKey
    assert Maintenance.model_fields["verb"].annotation is Verb
    assert Action.model_fields["verb"].annotation is Verb
    assert Target.model_fields["component"].annotation == Component | None
    assert Target.model_fields["config_key"].annotation == ConfigKey | None
    assert Target.model_fields["condition_key"].annotation == ConditionKey | None


@pytest.mark.parametrize(
    ("kind", "number", "expected"),
    [
        ("episode", 123, "e000123"),
        ("robot", 7, "r07"),
        ("site", 2, "s2"),
        ("world", 3, "w03"),
        ("fault", 1, "f01"),
        ("experiment", 5, "x0005"),
        ("robot", 123, "r123"),
    ],
)
def test_neutral_ids_have_one_canonical_form(kind: str, number: int, expected: str) -> None:
    assert ids.neutral_id(kind, number) == expected
    assert ids.is_neutral(kind, expected)
    assert ids.id_number(kind, expected) == number
    assert ids.which_kind(expected) == ids.IdKind(kind)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("robot", "r7"),  # too short
        ("robot", "r007"),  # not canonical
        ("robot", "robot_1"),
        ("episode", "d2-degrees4-t0-i1"),  # the skeleton's dose runs named episodes after levels
        ("episode", "e12"),
        ("site", "s02"),
        ("site", "site2"),
        ("robot", "s2"),
        ("episode", 123),
    ],
)
def test_the_validator_refuses_identifiers_that_are_not_neutral(kind: str, value: object) -> None:
    assert not ids.is_neutral(kind, value)
    with pytest.raises(ValueError, match="not a neutral"):
        ids.check_neutral(kind, value)


def test_pydantic_id_fields_refuse_non_neutral_values() -> None:
    class Probe(BaseModel):
        robot: ids.RobotId
        scope: ids.Scope

    assert Probe(robot="r07", scope="s2").robot == "r07"
    for bad in ({"robot": "r007", "scope": "s2"}, {"robot": "r07", "scope": "site2"}):
        with pytest.raises(ValidationError):
            Probe.model_validate(bad)


def test_scopes_cover_their_robots() -> None:
    assert ids.scope_kind("fleet") is ids.ScopeKind.fleet
    assert ids.scope_kind("s2") is ids.ScopeKind.site
    assert ids.scope_kind("r07") is ids.ScopeKind.robot
    assert ids.scope_covers("fleet", "r07", "s2")
    assert ids.scope_covers("s2", "r07", "s2")
    assert not ids.scope_covers("s1", "r07", "s2")
    assert ids.scope_covers("r07", "r07", "s2")
    assert not ids.scope_covers("r08", "r07", "s2")
    assert ids.scope_rank("fleet") < ids.scope_rank("s2") < ids.scope_rank("r07")
    with pytest.raises(ValueError):
        ids.scope_kind("everywhere")
