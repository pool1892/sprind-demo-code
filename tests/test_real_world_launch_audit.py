"""Read-only launch audits on the retired development fixtures; no agent calls.

These tests deliberately derive the scanner inputs from actual stored worlds,
not the much smaller toy truth table. They skip only where those local records
are unavailable. Nothing is written to the stored worlds or analysis records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from fleetlab.agents.audit import audit_state
from fleetlab.agents.checks import source_audit
from fleetlab.agents.sessions import SESSION_RULES, SessionReport, continuation_prompt
from fleetlab.agents.toolenv import ToolEnv, build_bundle, build_skills
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.world import WorldSpec
from fleetlab.integration.analyse import task_prompt
from fleetlab.integration.chain import SYSTEM
from fleetlab.log._files import read_canary
from fleetlab.log.leaks import code_terms, truth_terms
from fleetlab.skill_library import ANALYSIS_SKILLS, library_dir

PROBES = Path.home() / ".cache/robots-probes"
DEVELOPMENT = {"w41", "w42", "w43"}


def world_inputs(name: str) -> tuple[Fleet, tuple[Path, ...], WorldSpec]:
    assert name in DEVELOPMENT
    files = PROBES / "worlds" / name / "files"
    if not (files / "model.json").is_file():
        pytest.skip("stored development-world audit fixture unavailable")
    fleet = Fleet.model_validate_json((files / "model.json").read_text())
    runs = tuple(Path(p) for p in json.loads((files / "world_truth.json").read_text())["base_runs"])
    return fleet, runs, WorldSpec.model_validate_json((files / "spec.json").read_text())


@pytest.mark.parametrize("ablated", [False, True], ids=["full", "ablated"])
def test_real_world_pipeline_texts_schema_bundle_and_both_libraries(
    tmp_path: Path, ablated: bool
) -> None:
    fleet, runs, spec = world_inputs("w41")
    export = PROBES / "worlds/w41/export-review-20261009T173751Z"
    if not export.is_dir():
        pytest.skip("cleared development export unavailable")
    # A real manifest contributes thousands of seeds as well as planted names.
    assert len(truth_terms(runs, fleet=fleet)) > 3000
    texts = {
        "system": SYSTEM + SESSION_RULES,
        "schema": json.dumps(SessionReport.model_json_schema()),
    }
    for condition in ("logs", "experiments"):
        texts[condition] = task_prompt(spec, condition)
        texts[condition + " continuation"] = continuation_prompt(2, texts[condition])
    bundle = build_bundle(tmp_path)
    skills = build_skills(tmp_path, library_dir(), ANALYSIS_SKILLS, ablate_fault_classes=ablated)
    env = ToolEnv(tmp_path / "python", tmp_path / "site", bundle, skills)
    report = source_audit(export, runs, env, fleet=fleet, texts=texts)
    assert report.passed, (ablated, report.failures())


def _stored_states() -> list[Path]:
    checked = []
    experiment_sessions = 0
    for path in sorted((PROBES / "skills").glob("20261009-r*/*/run.json")):
        metadata = json.loads(path.read_text())
        name = Path(metadata["world_root"]).name
        if name not in DEVELOPMENT or not metadata.get("analysis") or metadata.get("resume_from"):
            continue
        analysis = Path(metadata["analysis"])
        if not analysis.is_dir():
            continue
        checked.append(path)
        experiment_sessions += name == "w41" and metadata["condition"] == "experiments"
    if checked:
        assert len(checked) >= 26 and experiment_sessions >= 2
    return checked


@pytest.mark.parametrize(
    "path",
    _stored_states(),
    ids=lambda path: str(path.relative_to(PROBES)) if isinstance(path, Path) else None,
)
def test_real_world_stored_agent_states_pass(path: Path) -> None:
    metadata = json.loads(path.read_text())
    fleet, runs, _ = world_inputs(Path(metadata["world_root"]).name)
    failures: list[str] = []
    for state in ("memory", "outbox"):
        report = audit_state(Path(metadata["analysis"]) / state, runs, fleet, state)
        failures.extend(f"{state}: {hit.where}: {hit.term}" for hit in report.hits)
    assert not failures, (path, failures)


def test_real_world_planted_names_and_seeds_still_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.log import leaks

    fleet, runs, _ = world_inputs("w41")
    # These retired worlds have only countable fault IDs. Augment the real
    # audit inputs with one synthetic non-countable secret, without changing
    # any stored truth/manifest file or replacing the real world's term list.
    non_countable_id = "x9e52a18c"
    original = leaks._secret_values

    def secret_values(paths: Iterable[Path | str]) -> tuple[set[int], set[str]]:
        seeds, fault_ids = original(paths)
        return seeds, fault_ids | {non_countable_id}

    monkeypatch.setattr(leaks, "_secret_values", secret_values)
    terms = code_terms(runs, fleet=fleet)
    truth = json.loads((PROBES / "worlds/w41/files/world_truth.json").read_text())
    plants = {
        "canary": read_canary(runs[0]),
        "trait": next(iter(fleet.robots()[0].traits)),
        "kind": truth["fault_schedule"][0]["kind"],
        "non-countable fault id": non_countable_id,
        "seed": next(t for t in terms if t.isdigit() and len(t) >= 10),
    }
    for name, value in plants.items():
        assert value and value.lower() in truth_terms(runs, fleet=fleet)
        (tmp_path / "notes.md").write_text(
            "smallest detectable difference; needed for 80% power; healthy robots; "
            "sampling_steps10; gain=2.0\n" + value
        )
        report = audit_state(tmp_path, runs, fleet, "memory")
        assert any(hit.term == value.lower() for hit in report.hits), name


def test_real_public_condition_values_are_accepted_by_the_tool(tmp_path: Path) -> None:
    from fleetlab.toolserver.server import Limits, Paths, ToolBox

    export = PROBES / "worlds/w41/export-review-20261009T173751Z"
    if not export.is_dir():
        pytest.skip("cleared development export unavailable")
    for surface in ("memory", "outbox", "results"):
        (tmp_path / surface).mkdir()
    box = ToolBox(
        Paths(export, tmp_path / "memory", tmp_path / "outbox", tmp_path / "results"),
        limits=Limits(max_experiment_requests=20, duckdb_memory="512MB", duckdb_threads=1),
    )
    try:
        description = json.loads(
            box.call("describe", {"part": "fleet", "include_logs": False}).content[0]["text"]
        )
        checked = set()
        for site in description["fleet"]["sites"]:
            for key, value in site["conditions"].items():
                result = box.call(
                    "request_experiment",
                    {
                        "arms": [
                            {"actions": []},
                            {
                                "actions": [
                                    {
                                        "verb": "change_condition",
                                        "target": {
                                            "scope": site["id"],
                                            "condition_key": key,
                                        },
                                        "value": value,
                                    }
                                ]
                            },
                        ],
                        "robots": [site["robots"][0]["id"]],
                        "episodes_per_arm": 1,
                        "paired_starts": True,
                        "power_note": "Domain check only; no execution.",
                    },
                )
                receipt = json.loads(result.content[0]["text"])
                assert receipt["accepted"], (key, value, receipt)
                checked.add(key)
        assert {"lighting", "object_batch", "placement"} <= checked
    finally:
        box.close()
