"""Agent tools and answer schemas: strict JSON schemas that OpenAI's strict mode accepts, and
answers that round-trip through them.

docs/design/contracts.md, section 12.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from fleetlab.contracts import tools
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, Budget, FixSet, fits_budget
from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.tools import (
    DESCRIBE_SIZE_CAP,
    EXPERIMENT_BUDGET,
    TOOLS,
    Cause,
    DescribePart,
    DescribeTooLarge,
    Diagnosis,
    EvidenceRef,
    ExperimentArm,
    ExperimentRequest,
    Interval,
    RankedAction,
    RankedFixSet,
    RetrainIndicator,
    Warning,
    describe_part,
    json_schema_strict,
    strict_schema_problems,
)
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb

ANSWER_MODELS: list[type[BaseModel]] = [
    Diagnosis,
    RankedFixSet,
    ExperimentRequest,
    tools.ExperimentResult,
    tools.Receipt,
]


def conforms(value: Any, schema: dict[str, Any]) -> bool:
    """Whether a JSON value satisfies a strict schema (the keywords strict mode uses)."""
    if "anyOf" in schema:
        return any(conforms(value, option) for option in schema["anyOf"])
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if kind is not None and not checks[kind](value):
        return False
    if kind == "string" and "pattern" in schema and not re.search(schema["pattern"], value):
        return False
    if kind == "object":
        if set(value) != set(schema["properties"]):
            return False
        return all(conforms(value[k], s) for k, s in schema["properties"].items())
    if kind == "array":
        return all(conforms(item, schema["items"]) for item in value)
    return True


def diagnosis() -> Diagnosis:
    return Diagnosis(
        causes=[
            Cause(
                target=Target(scope="r07", component=Component.scene_camera),
                direction=Direction.degraded,
                confidence=0.8,
                evidence=[
                    EvidenceRef(episode_id="e000123", step=40, table=None, query=None),
                    EvidenceRef(
                        episode_id=None,
                        step=None,
                        table="episodes",
                        query="select robot_id, avg(outcome = 'success') from episodes",
                    ),
                ],
                failure_mode="grasps land a few centimetres beside the object",
            )
        ],
        mode_summaries=["misplaced grasps on one robot from day 4"],
        warnings=[Warning(scope="s2", metric="success rate", expected_day=12, confidence=0.4)],
        retrain=[
            RetrainIndicator(
                failure_mode="drops tall objects",
                episode_ids=["e000200"],
                why_no_action_fixes_it="no setting changes the grasp height",
                data_wanted="grasps of tall objects",
            )
        ],
    )


def ranked_actions() -> list[RankedAction]:
    """Ten proposals, best first: a camera, a delay setting, then grippers robot by robot."""
    first = [
        RankedAction(
            action=Action(
                verb=Verb.recalibrate,
                target=Target(scope="r07", component=Component.scene_camera),
            ),
            expected_gain=4.5,
            uncertainty=Interval(low=2.0, high=7.0),
            rationale="camera",
        ),
        RankedAction(
            action=Action(
                verb=Verb.set,
                target=Target(scope="s2", config_key=ConfigKey.control_action_delay_steps),
                value=0,
            ),
            expected_gain=1.0,
            uncertainty=Interval(low=0.0, high=2.0),
            rationale="delay",
        ),
    ]
    grippers = [
        RankedAction(
            action=Action(
                verb=Verb.service, target=Target(scope=f"r{n:02d}", component=Component.gripper)
            ),
            expected_gain=0.5 - 0.05 * n,
            uncertainty=Interval(low=-0.5, high=1.0),
            rationale="worn pads",
        )
        for n in range(1, 9)
    ]
    return first + grippers


def ranked_fix_set() -> RankedFixSet:
    return RankedFixSet(actions=ranked_actions())


@pytest.mark.parametrize("model", ANSWER_MODELS + [t.arguments for t in TOOLS])
def test_every_schema_is_strict(model: type[BaseModel]) -> None:
    schema = json_schema_strict(model)
    assert strict_schema_problems(schema) == []
    assert "$ref" not in str(schema) and "$defs" not in schema


def test_the_checker_finds_what_strict_mode_refuses() -> None:
    raw = Diagnosis.model_json_schema()  # pydantic's own: references, optional fields
    problems = strict_schema_problems(raw)
    assert any("unsupported keywords" in p for p in problems)
    loose = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    assert any("additionalProperties" in p for p in strict_schema_problems(loose))
    assert any("required" in p for p in strict_schema_problems(loose))
    assert strict_schema_problems({"type": "array", "items": {"type": "string"}}) == [
        "the root is not an object"
    ]
    untyped = {
        "type": "object",
        "properties": {"a": {}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert any("untyped" in p for p in strict_schema_problems(untyped))


def test_models_strict_mode_cannot_express_are_refused() -> None:
    class Optional(BaseModel):
        count: int = 3

    class Mapping(BaseModel):
        values: dict[str, int]

    class Anything(BaseModel):
        value: Any

    for model in (Optional, Mapping, Anything):
        with pytest.raises(ValueError):
            json_schema_strict(model)


@pytest.mark.parametrize("answer", [diagnosis, ranked_fix_set])
def test_answers_round_trip_through_their_strict_schema(answer: Any) -> None:
    value = answer()
    schema = json_schema_strict(type(value))
    dumped = value.model_dump(mode="json")
    assert conforms(dumped, schema)
    assert type(value).model_validate(dumped) == value


def test_optional_fields_are_nullable_and_required() -> None:
    target = json_schema_strict(Diagnosis)["properties"]["causes"]["items"]["properties"]["target"]
    assert target["required"] == ["scope", "component", "config_key", "condition_key"]
    assert {"type": "null"} in target["properties"]["component"]["anyOf"]
    cause = json_schema_strict(Diagnosis)["properties"]["causes"]["items"]
    assert "scope" not in cause["properties"]  # the target carries it


def test_evidence_names_an_episode_or_a_query() -> None:
    with pytest.raises(ValidationError):
        EvidenceRef(episode_id="e000001", step=None, table="episodes", query="select 1")
    with pytest.raises(ValidationError):
        EvidenceRef(episode_id=None, step=None, table=None, query=None)
    with pytest.raises(ValidationError):
        EvidenceRef(episode_id=None, step=None, table="identities", query="select *")


def test_a_ranked_list_is_cut_at_every_budget_by_the_scorer() -> None:
    ranked = ranked_fix_set()
    actions = [r.action for r in ranked.actions]
    three = ranked.as_fix_set(Budget(kind="count", amount=3))
    assert isinstance(three, FixSet) and three.actions == actions[:3]
    # 4 hours: the recalibration (0.5) and the setting (0.1) fit, then one 3-hour service.
    assert ranked.as_fix_set(Budget(kind="cost", amount=4)).actions == actions[:3]
    for budget in SCORING_BUDGETS:
        assert fits_budget(ranked.as_fix_set(budget))
    assert "budget" not in RankedFixSet.model_fields


def test_a_ranked_list_has_ten_distinct_complete_actions() -> None:
    with pytest.raises(ValidationError, match="at least 10"):
        RankedFixSet(actions=ranked_actions()[:9])
    twice = ranked_actions()
    twice[-1] = twice[0]
    with pytest.raises(ValidationError, match="each action once"):
        RankedFixSet(actions=twice)
    unfilled = ranked_actions()
    unfilled[1] = unfilled[1].model_copy(
        update={"action": unfilled[1].action.model_copy(update={"value": None})}
    )
    with pytest.raises(ValidationError, match="without a value"):
        RankedFixSet(actions=unfilled)


def test_the_experiment_budget_is_the_design_s() -> None:
    # The historical design budget stays the tool server's default; the schema no longer
    # caps an arm, so the round driver's attempt budget can (loop.md, section 8).
    assert EXPERIMENT_BUDGET == {"requests": 8, "episodes_per_arm": 60}
    arms = [ExperimentArm(actions=[]), ExperimentArm(actions=[])]
    ExperimentRequest(
        arms=arms, robots=["r07"], episodes_per_arm=60, paired_starts=True, power_note="60 per arm"
    )
    ExperimentRequest(
        arms=arms, robots=["r07"], episodes_per_arm=200, paired_starts=True, power_note="big"
    )
    with pytest.raises(ValidationError):
        ExperimentRequest(
            arms=arms, robots=["r07"], episodes_per_arm=0, paired_starts=True, power_note="none"
        )
    schema = json_schema_strict(ExperimentRequest)
    assert "maximum" not in schema["properties"]["episodes_per_arm"]
    assert schema["properties"]["episodes_per_arm"]["minimum"] == 1


def test_answers_refuse_what_the_note_rules_out() -> None:
    with pytest.raises(ValidationError):
        Warning(scope="fleet", metric="success rate", expected_day=3, confidence=0.5)
    with pytest.raises(ValidationError):
        RetrainIndicator(
            failure_mode="x", episode_ids=["episode-7"], why_no_action_fixes_it="y", data_wanted="z"
        )
    with pytest.raises(ValidationError):
        ExperimentRequest(
            arms=[ExperimentArm(actions=[])],
            robots=["r07"],
            episodes_per_arm=10,
            paired_starts=True,
            power_note="one arm is no experiment",
        )
    with pytest.raises(ValidationError):
        Interval(low=2.0, high=1.0)
    with pytest.raises(ValidationError):
        Cause(
            target=Target(scope="r07"),
            direction=Direction.degraded,
            confidence=1.5,
            evidence=[],
            failure_mode="x",
        )


def test_the_tool_list_is_the_note_s() -> None:
    assert [t.name for t in TOOLS] == [
        "query",
        "get_frames",
        "describe",
        "list_skills",
        "read_skill",
        "request_experiment",
        "experiment_results",
        "submit_diagnosis",
        "submit_fix_set",
        "memory_read",
        "memory_write",
        "write_learned_skill",
        "experiment_budget",
        "earlier_decisions",
        "submit_decision",
        "request_probe",
        "probe_results",
        "operator_state",
        "submit_acceptance_proposal",
        "submit_job_program",
        "job_results",
        "save_job",
        "write_check",
        "run_check",
        "job_library",
        "submit_job_account",
    ]
    assert {"request_probe", "probe_results", *tools.JOB_TOOLS} == tools.OPT_IN_TOOLS
    schemas = tools.tool_schemas()
    assert all(strict_schema_problems(s["input_schema"]) == [] for s in schemas)
    describe = next(s for s in schemas if s["name"] == "describe")
    assert describe["input_schema"]["properties"]["include_logs"]["type"] == "boolean"
    assert describe["input_schema"]["properties"]["part"]["enum"] == [
        "dictionary",
        "fleet",
        "vocabulary",
        "costs",
        "all",
    ]


def description(robots: int = 2, changes: int = 1) -> dict[str, Any]:
    """A system description as `describe_for_agents` writes it, with logs."""
    from fleetlab.contracts.model import (
        Change,
        ChangeAuthor,
        ComponentState,
        Fleet,
        Robot,
        Site,
        describe_for_agents,
    )

    components = {c: ComponentState(component=c, installed_day=0) for c in Component}
    site = Site(
        id="s1",
        robots=[
            Robot(id=f"r{n:02d}", site_id="s1", components=components) for n in range(1, robots + 1)
        ],
    )
    log = [
        Change(
            day=d % 9,
            scope="s1",
            key=ConfigKey.software_version,
            old="1.0",
            new="1.1",
            author=ChangeAuthor.vendor,
        )
        for d in range(changes)
    ]
    fleet = Fleet(id="w01", sites=[site], change_log=log)
    return describe_for_agents(fleet, True, ["pick up the bowl"])


def test_describe_returns_one_compact_part() -> None:
    whole = description()
    fleet_part = json.loads(describe_part("fleet", whole, "# Data dictionary", False))
    assert set(fleet_part) == {"contracts_version", "fleet", "task_classes", "catalogue_entries"}
    assert "change_log" not in fleet_part["fleet"]
    with_logs = describe_part(DescribePart.fleet, whole, "# Data dictionary", True)
    assert '"change_log"' in with_logs and ": " not in with_logs  # compact JSON
    assert json.loads(describe_part("dictionary", whole, "# Data dictionary", True)) == {
        "dictionary": "# Data dictionary"
    }
    assert set(json.loads(describe_part("vocabulary", whole, "", True))) == {"vocabulary"}
    costs = json.loads(describe_part("costs", whole, "", True))["costs"]
    assert {"verb": "replace", "target": "arm", "hours": 16.0} in costs
    everything = json.loads(describe_part("all", whole, "# Data dictionary", True))
    assert {"dictionary", "fleet", "vocabulary", "costs", "catalogue_entries"} <= set(everything)
    assert "catalogue" not in everything  # the catalogue is a table


def test_a_part_over_the_size_cap_is_refused() -> None:
    whole = description(robots=2, changes=400)  # a long change log
    assert len(describe_part("fleet", whole, "", False)) < DESCRIBE_SIZE_CAP
    with pytest.raises(DescribeTooLarge, match="query the tables change_log"):
        describe_part("fleet", whole, "", True)
