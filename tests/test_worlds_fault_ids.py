"""Opaque composition identities, legacy replay contracts, and the export boundary."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import TASK_CLASSES, all_jobs, toy_fleet, write_toy_run
from pydantic import TypeAdapter, ValidationError

from fleetlab.contracts.ids import FaultId, IdKind, id_number, is_neutral, which_kind
from fleetlab.contracts.model import describe_for_agents
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.world import Patterns
from fleetlab.log import audit_export, export_runs
from fleetlab.worlds.compose import (
    Composer,
    Composition,
    identify,
    make_fault,
    opaque_fault_ids,
    rename_fault_ids,
)
from fleetlab.worlds.recipe import DEVELOPMENT, INTEGRATION


def test_tokens_are_seeded_opaque_nonwords_not_ordinals() -> None:
    handles = [f"f{i:02}" for i in range(1000)]
    tokens = opaque_fault_ids(DEVELOPMENT["D2"].seed, handles)
    assert tokens == opaque_fault_ids(DEVELOPMENT["D2"].seed, handles[::-1])
    assert len(set(tokens.values())) == 1000
    assert tokens["f01"] == "fx6d3e2e809b706d92715cf6aa9fbd5888"
    assert not set(tokens.values()) & set(
        opaque_fault_ids(DEVELOPMENT["D3"].seed, handles).values()
    )
    for token in tokens.values():
        assert re.fullmatch(r"fx[0-9a-f]{32}", token)
        assert any(c.isdigit() for c in token[2:]) and any(c.isalpha() for c in token[2:])
        assert TypeAdapter(FaultId).validate_python(token) == token
        assert which_kind(token) is IdKind.fault
        assert not any(is_neutral(kind, token) for kind in IdKind if kind is not IdKind.fault)
        with pytest.raises(ValueError, match="no ordinal"):
            id_number(IdKind.fault, token)


@pytest.mark.parametrize(
    "token",
    ["fx123abc", "fx" + "a" * 32, "fx" + "1" * 32, "fx" + "A1" * 16, "fx" + "a1" * 16 + "\n"],
)
def test_invalid_or_word_like_tokens_are_refused(token: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(FaultId).validate_python(token)


def test_assignment_is_independent_of_schedule_order_kind_and_scope() -> None:
    recipe = DEVELOPMENT["D2"]
    faults = [
        make_fault("f01", "action_latency", "8 steps", "r01", 0),
        make_fault("f02", "policy_knob", "15 actions per chunk", "r01", 1),
    ]
    original = Composition(
        recipe,
        Composer(recipe).spec(faults, Patterns(interactions=[("f01", "f02")])),
        0,
        {"f01": "interaction", "f02": "interaction"},
        frozenset({"f01"}),
        False,
        {"f01+f02": ["f01", "f02"]},
    )
    new = identify(original)
    inverse = {v: k for k, v in new.fault_id_map.items()}
    assert rename_fault_ids(replace(new, fault_id_map={}), inverse) == original
    changed = replace(
        original,
        spec=original.spec.model_copy(
            update={
                "fault_schedule": [
                    faults[1],
                    make_fault("f01", "gripper_wear", "0.5 wear", "r03", 2),
                ]
            }
        ),
        roles=dict(reversed(list(original.roles.items()))),
    )
    assert identify(changed).fault_id_map == new.fault_id_map
    assert [f.fault_id for f in identify(changed).spec.fault_schedule] == [
        new.fault_id_map["f02"],
        new.fault_id_map["f01"],
    ]
    # Identity allocation covers the schedule even without optional role metadata.
    assert identify(replace(original, roles={})).fault_id_map == new.fault_id_map
    for recipe in (DEVELOPMENT["D1"], *INTEGRATION.values()):
        legacy = replace(original, recipe=recipe)
        assert identify(legacy) is legacy
    # A different seed under a previously used world number is a new composition.
    assert identify(
        replace(
            original,
            recipe=recipe.model_copy(update={"seed": original.recipe.seed, "estimator": "task-v2"}),
        )
    ).fault_id_map


def test_legacy_counted_ids_and_replay_jobs_round_trip_unchanged() -> None:
    for value in ("f00", "f01", "f14", "f100"):
        assert TypeAdapter(FaultId).validate_python(value) == value
        assert id_number(IdKind.fault, value) == int(value[1:])
    for job in all_jobs():
        raw = job.model_dump_json()
        loaded = Job.model_validate_json(raw)
        assert loaded.model_dump_json() == raw
        assert loaded.identity.fault_ids == job.identity.fault_ids
        assert loaded.remove == job.remove
    with pytest.raises(ValidationError):
        TypeAdapter(FaultId).validate_python("f001")


@pytest.fixture
def opaque_export(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    mapping = opaque_fault_ids(DEVELOPMENT["D2"].seed, ["f01", "f02"])
    jobs = rename_fault_ids(all_jobs(), mapping)
    run, export = tmp_path / "run", tmp_path / "export"
    fleet = write_toy_run(run, jobs)
    export_runs([run], export, fleet=fleet, task_classes=TASK_CLASSES)
    return run, export, list(mapping.values())


def test_opaque_ids_are_absent_from_export_description_and_mounted_names(
    opaque_export: tuple[Path, Path, list[str]],
) -> None:
    run, export, tokens = opaque_export
    description = json.dumps(describe_for_agents(toy_fleet(), True, TASK_CLASSES))
    report = audit_export(export, [run], fleet=toy_fleet())
    assert report.clean, report.summary()
    assert set(tokens) <= set(report.terms)
    for token in tokens:
        assert token not in description
        for path in export.rglob("*"):
            assert token not in path.relative_to(export).as_posix()
            if path.is_file():
                assert token.encode() not in path.read_bytes()
                if path.suffix == ".parquet":
                    assert token not in json.dumps(pq.read_table(path).to_pylist(), default=str)


@pytest.mark.parametrize("surface", ["description", "code", "filename", "parquet"])
def test_export_audit_refuses_a_planted_opaque_fault_id(
    opaque_export: tuple[Path, Path, list[str]], surface: str
) -> None:
    run, export, tokens = opaque_export
    token = tokens[0]
    texts = {"description": token} if surface == "description" else None
    code = {"mounted.py": f"value = '{token}'"} if surface == "code" else None
    if surface == "filename":
        (export / f"{token}.txt").write_text("a note")
    if surface == "parquet":
        path = export / "log/events.parquet"
        table = pq.read_table(path)
        details = table["detail"].to_pylist()
        details[0] = f"see {token}"
        pq.write_table(table.set_column(3, "detail", pa.array(details)), path)
    report = audit_export(export, [run], fleet=toy_fleet(), texts=texts, code=code)
    assert not report.clean
    assert token in {hit.term for hit in report.hits}
