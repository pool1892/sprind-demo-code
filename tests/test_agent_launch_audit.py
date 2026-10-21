"""Launch-boundary regressions: no model, login or subprocess is used."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from log_toy import TASK_CLASSES, write_toy_run

from fleetlab.agents import checks, run
from fleetlab.agents.audit import AuditContext
from fleetlab.agents.checks import GateError
from fleetlab.agents.sessions import Analysis
from fleetlab.agents.toolenv import ToolEnv, closure_problems, import_closure
from fleetlab.log import export_runs
from fleetlab.log._files import read_canary


class ReachedLaunch(Exception):
    pass


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[run.ToolAccess, list[Path]]:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    fleet = write_toy_run(tmp_path / "run")
    export_runs([tmp_path / "run"], tmp_path / "export", fleet=fleet, task_classes=TASK_CLASSES)
    for name in ("source", "skills", "python", "site"):
        (tmp_path / name).mkdir()
    (tmp_path / "source" / "module.py").write_text("value = 1\n")
    env = ToolEnv(tmp_path / "python", tmp_path / "site", tmp_path / "source", tmp_path / "skills")
    analysis = Analysis.create(fleet.id, base=tmp_path / "analyses")
    access = analysis.tool_access(
        env, tmp_path / "export", 1, audit=AuditContext((tmp_path / "run",), fleet)
    )
    launched: list[Path] = []

    def prepare(*args: Any, **kwargs: Any) -> run.PreparedCall:
        directory = args[4]
        box = run.box_dir(directory)
        (box / "cwd").mkdir(parents=True)
        (box / "toolbox").mkdir()
        (box / "system.txt").write_text(args[2])
        (box / "schema.json").write_text(json.dumps(args[7]))
        return run.PreparedCall([], {}, args[3], directory, None, 0)

    def execute(*args: Any, **kwargs: Any) -> None:
        directory = args[3]
        assert json.loads((directory / "audit.json").read_text())["passed"]
        assert not (run.box_dir(directory) / "audit.json").exists()
        if kwargs.get("on_start") is not None:
            kwargs["on_start"](directory)
        launched.append(directory)
        raise ReachedLaunch

    monkeypatch.setattr(run, "prepare_call", prepare)
    monkeypatch.setattr(run, "_execute", execute)
    monkeypatch.setattr(checks, "require_verified", lambda *args: {"passed": True})
    return access, launched


def test_clean_call_is_audited_at_actual_launch(launch: tuple[run.ToolAccess, list[Path]]) -> None:
    access, launched = launch
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


def test_direct_call_requires_audit_context(launch: tuple[run.ToolAccess, list[Path]]) -> None:
    access, launched = launch
    with pytest.raises(GateError, match="audit inputs"):
        run.run_agent("claude", "model", "System", "Task", tools=replace(access, audit=None))
    assert not launched


@pytest.mark.parametrize(
    "surface",
    [
        "export",
        "source",
        "skills",
        "memory",
        "outbox",
        "results",
        "system",
        "prompt",
        "schema",
        "box",
    ],
)
def test_each_actual_session_surface_is_checked(
    launch: tuple[run.ToolAccess, list[Path]], surface: str
) -> None:
    access, launched = launch
    assert access.audit is not None
    token = read_canary(access.audit.run_dirs[0])
    assert token
    system, prompt = "System", "Task"
    kwargs: dict[str, Any] = {}
    root = {
        "export": access.export,
        "source": access.env.src,
        "skills": access.env.skills,
        "memory": access.memory,
        "outbox": access.outbox,
        "results": access.results,
    }.get(surface)
    if root is not None:
        (root / "contaminated.txt").write_text(token)
    elif surface == "system":
        system = token
    elif surface == "prompt":
        prompt = token
    elif surface == "schema":
        kwargs["schema"] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
            "description": token,
        }
    else:
        kwargs["_plant"] = {"extra.txt": token}
    with pytest.raises(GateError):
        run.run_agent("claude", "model", system, prompt, tools=access, **kwargs)
    assert not launched


def test_next_call_rechecks_modified_memory(launch: tuple[run.ToolAccess, list[Path]]) -> None:
    access, launched = launch
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert access.audit is not None
    (access.memory / "notes.md").write_text(str(read_canary(access.audit.run_dirs[0])))
    with pytest.raises(GateError):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


PUBLIC_NOTES = (
    "Windows: observed days; success=.6; camera faults: independently supported; "
    "successes128; fault_classes; sampling_steps10; actions_executed_per_chunk30; "
    "image_views4; min_detectable=.25."
)


def _request_record(note: str) -> dict[str, Any]:
    return {
        "experiment_id": "x0001",
        "session": 1,
        "request": {
            "arms": [{"actions": []}, {"actions": []}],
            "robots": ["r01"],
            "episodes_per_arm": 4,
            "paired_starts": True,
            "power_note": note,
        },
    }


def test_agent_notes_and_request_prose_allow_public_words(
    launch: tuple[run.ToolAccess, list[Path]],
) -> None:
    from fleetlab.agents.sessions import QueuedExperiment
    from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest

    access, launched = launch
    (access.memory / "notes.md").write_text(PUBLIC_NOTES)
    request = QueuedExperiment(
        experiment_id="x0001",
        session=1,
        request=ExperimentRequest(
            arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[])],
            robots=["r01"],
            episodes_per_arm=4,
            paired_starts=True,
            power_note=PUBLIC_NOTES,
        ),
    )
    (access.outbox / "experiments.jsonl").write_text(request.model_dump_json() + "\n")
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


@pytest.mark.parametrize(
    "hidden",
    ["toy_camera_wobble", "mount_tolerance_mm", "mountToleranceMm", "raw_toy_camera_wobble_delta"],
)
def test_agent_prose_still_refuses_planted_identifiers(
    launch: tuple[run.ToolAccess, list[Path]], hidden: str
) -> None:
    access, launched = launch
    (access.memory / "notes.md").write_text(PUBLIC_NOTES + "\n" + hidden)
    with pytest.raises(GateError, match="memory audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


@pytest.mark.parametrize("term", ["private_sampling_steps", "min_detectable_hidden"])
def test_new_hidden_trait_is_protected_on_agent_surface(
    launch: tuple[run.ToolAccess, list[Path]], term: str
) -> None:
    from fleetlab.agents.audit import audit_state

    access, _ = launch
    assert access.audit is not None and access.audit.fleet is not None
    fleet = access.audit.fleet
    fleet.sites[0].robots[0].traits[term] = 1
    (access.memory / "notes.md").write_text(PUBLIC_NOTES + "\n" + term)
    report = audit_state(access.memory, access.audit.run_dirs, fleet, "memory")
    assert any(hit.term == term for hit in report.hits)


@pytest.mark.parametrize(
    "prose",
    [
        "world_seed=7; fault_schedule=[]; success_step=4; fault_table=[]",
        "toy-camera-wobble; mount tolerance mm; fault; faults; min_detectable=.2",
        '{"success_step": 4, "fault_table": []}',
    ],
)
def test_agent_prose_allows_record_structure_and_natural_hypotheses(
    launch: tuple[run.ToolAccess, list[Path]], prose: str
) -> None:
    access, launched = launch
    (access.memory / "notes.md").write_text(prose)
    (access.outbox / "diagnosis.json").write_text(json.dumps({"mode_summaries": [prose]}))
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


def test_public_prose_cannot_mask_later_canary(launch: tuple[run.ToolAccess, list[Path]]) -> None:
    access, launched = launch
    assert access.audit is not None
    (access.memory / "notes.md").write_text(
        PUBLIC_NOTES + "\n" + str(read_canary(access.audit.run_dirs[0]))
    )
    with pytest.raises(GateError, match="memory audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


@pytest.mark.parametrize("surface", ["name", "extra", "results", "box", "key", "envelope"])
def test_agent_policy_does_not_expand_to_pipeline_state(
    launch: tuple[run.ToolAccess, list[Path]], surface: str
) -> None:
    access, launched = launch
    kwargs: dict[str, Any] = {}
    # Natural forms pass on agent but refuse on ours. Check files and envelope
    # fields too: merely naming an extra field power_note must not reclassify it.
    marker = "mount tolerance mm"
    if surface == "box":
        kwargs["_plant"] = {"extra.txt": marker}
    elif surface == "results":
        (access.results / "extra.txt").write_text(marker)
    elif surface == "name":
        (access.memory / "mount-tolerance-mm.md").write_text("ordinary text")
    elif surface in {"key", "envelope"}:
        value = _request_record("ordinary")
        value.update({marker: "ordinary"} if surface == "key" else {"power_note": marker})
        (access.outbox / "experiments.jsonl").write_text(json.dumps(value) + "\n")
    else:
        (access.memory / "extra.md").write_text(marker)
    with pytest.raises(GateError):
        run.run_agent("claude", "model", "System", "Task", tools=access, **kwargs)
    assert not launched


def test_unknown_refusal_text_is_not_mounted(launch: tuple[run.ToolAccess, list[Path]]) -> None:
    from fleetlab.toolserver.files import RESULTS_FILE

    access, launched = launch
    (access.results / RESULTS_FILE).write_text(
        json.dumps(
            {
                "experiment_id": "x0001",
                "status": "refused",
                "arms": [],
                "message": "apparently harmless free text",
            }
        )
        + "\n"
    )
    with pytest.raises(GateError, match="public validation"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


def test_seed_and_oracle_contracts_never_enter_tool_closure() -> None:
    closure = import_closure()
    assert not closure_problems(closure)
    for name in ("identity", "oracle", "world", "rollout"):
        assert "fleetlab.contracts." + name not in closure
        assert closure_problems(["fleetlab.contracts." + name])


def test_lazy_contract_exports_keep_the_existing_interface() -> None:
    import fleetlab.contracts as contracts

    for name in contracts.__all__:
        assert getattr(contracts, name) is not None


@pytest.mark.parametrize(
    "override",
    [{"_env": {"HOME": "/somewhere"}}, {"_extra_argv": ["-c", "features.shell_tool=true"]}],
)
def test_analysis_call_refuses_unverified_overrides(
    launch: tuple[run.ToolAccess, list[Path]], override: dict[str, Any]
) -> None:
    access, launched = launch
    with pytest.raises(ValueError, match="overrides"):
        run.run_agent("claude", "model", "System", "Task", tools=access, **override)
    assert not launched


@pytest.mark.parametrize("suffix", ["toy_camera_wobble", "mount_tolerance_mm"])
def test_public_json_prose_still_refuses_hidden_suffixes(
    launch: tuple[run.ToolAccess, list[Path]],
    suffix: str,
) -> None:
    access, launched = launch
    (access.outbox / "diagnosis.json").write_text(
        json.dumps(
            {
                "causes": [],
                "mode_summaries": [PUBLIC_NOTES, suffix],
                "warnings": [],
                "retrain": [],
            }
        )
    )
    with pytest.raises(GateError, match="outbox audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


def test_json_key_is_not_a_prose_exception(launch: tuple[run.ToolAccess, list[Path]]) -> None:
    access, launched = launch
    (access.outbox / "diagnosis.json").write_text(json.dumps({"faults": PUBLIC_NOTES}))
    with pytest.raises(GateError, match="outbox audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


@pytest.mark.parametrize("day,allowed", [(2, True), (3, False)])
def test_launch_checks_experiment_days_even_without_result_references(
    launch: tuple[run.ToolAccess, list[Path]],
    day: int,
    allowed: bool,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    access, launched = launch
    assert access.audit is not None
    path = access.export / "log" / "episodes.parquet"
    table = pq.read_table(path)
    for name, values in (("experiment_id", ["x0001"] * len(table)), ("day", [day] * len(table))):
        idx = table.schema.get_field_index(name)
        table = table.set_column(
            idx, table.schema.field(idx), pa.array(values, type=table.schema.field(idx).type)
        )
    pq.write_table(table, path)
    access = replace(access, audit=replace(access.audit, experiment_days=(2, 2)))
    with pytest.raises(ReachedLaunch if allowed else GateError):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert bool(launched) is allowed


def test_ablated_analysis_library_passes_the_actual_launch_audit(
    launch: tuple[run.ToolAccess, list[Path]],
    tmp_path: Path,
) -> None:
    from fleetlab.agents.toolenv import build_bundle, build_skills
    from fleetlab.contracts.skills import load_skill
    from fleetlab.skill_library import (
        ANALYSIS_SKILLS,
        KNOWLEDGE_BEGIN,
        KNOWLEDGE_END,
        library_dir,
    )

    access, launched = launch
    stripped = build_skills(
        tmp_path / "ablated", library_dir(), ANALYSIS_SKILLS, ablate_fault_classes=True
    )
    full = build_skills(tmp_path / "full", library_dir(), ANALYSIS_SKILLS)
    removed = []
    marked_skills = set()
    for path in full.rglob("SKILL.md"):
        text = path.read_text()
        expected = re.sub(
            rf"(?ms)^{re.escape(KNOWLEDGE_BEGIN)}\n.*?^{re.escape(KNOWLEDGE_END)}\n?",
            "",
            text,
        )
        assert (stripped / path.relative_to(full)).read_text() == expected
        while KNOWLEDGE_BEGIN in text:
            marked_skills.add(path.parent.name)
            _, text = text.split(KNOWLEDGE_BEGIN, 1)
            passage, text = text.split(KNOWLEDGE_END, 1)
            removed.append(passage.strip())
    content = "\n".join(p.read_text() for p in stripped.rglob("*") if p.is_file())
    assert marked_skills == {"episode-cards"}
    assert removed and all(passage not in content for passage in removed)
    assert KNOWLEDGE_BEGIN not in content and KNOWLEDGE_END not in content
    for name in ANALYSIS_SKILLS:
        load_skill(stripped / name)
    access = replace(
        access, env=replace(access.env, src=build_bundle(tmp_path / "bundle"), skills=stripped)
    )
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


def test_repeat_callback_is_after_gate_and_audit(
    launch: tuple[run.ToolAccess, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    access, launched = launch
    start = Mock()
    monkeypatch.setattr(checks, "require_verified", Mock(side_effect=GateError("gate refused")))
    with pytest.raises(GateError, match="gate refused"):
        run.run_agent("claude", "model", "System", "Task", tools=access, on_start=start)
    assert not start.called and not launched
    monkeypatch.setattr(checks, "require_verified", lambda *a: {})
    assert access.audit is not None
    (access.memory / "notes.md").write_text(str(read_canary(access.audit.run_dirs[0])))
    with pytest.raises(GateError, match="memory audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access, on_start=start)
    assert not start.called and not launched
    (access.memory / "notes.md").write_text("ordinary analysis notes")
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access, on_start=start)
    assert start.call_count == 1 and start.call_args.args[0] == launched[0]


def test_failed_process_start_cannot_reserve_a_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    start = Mock()
    monkeypatch.setattr(
        "fleetlab.agents.run.subprocess.Popen", Mock(side_effect=OSError("cannot start"))
    )
    result = run.AgentResult(False, None, {}, 0)
    with pytest.raises(OSError, match="cannot start"):
        run._execute(result, [], {}, tmp_path, "", 1, "codex", False, on_start=start)
    assert not start.called and not (tmp_path / "started.json").exists()


def test_repeat_callback_precedes_task_input_to_a_started_process(tmp_path: Path) -> None:
    """A plain Python stdin echo tests dispatch ordering; no CLI or model is started."""
    import sys

    (run.box_dir(tmp_path) / "cwd").mkdir(parents=True)
    marker = tmp_path / "reserved"
    result = run.AgentResult(False, None, {}, 0)

    def reserve(directory: Path) -> None:
        (directory / "reserved").write_text("reserved")

    run._execute(
        result,
        [
            sys.executable,
            "-c",
            "import pathlib,sys; print(pathlib.Path(sys.stdin.read()).is_file())",
        ],
        {},
        tmp_path,
        str(marker),
        10,
        "codex",
        False,
        on_start=reserve,
    )
    assert result.meta["exit_code"] == 0
    assert (tmp_path / "stdout.jsonl").read_text().strip() == "True"
    assert (tmp_path / "started.json").is_file()


def test_pipeline_texts_never_enter_agent_texts(
    launch: tuple[run.ToolAccess, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.log as log

    access, launched = launch
    prose = {
        "Saved hypothesis",
        "Requested comparison",
        "Diagnosed mode",
        "Fix rationale",
        "Data wanted",
    }
    (access.memory / "notes.md").write_text("Saved hypothesis")
    request = _request_record("Requested comparison")
    request["power_note"] = "Pipeline envelope"
    (access.outbox / "experiments.jsonl").write_text(json.dumps(request) + "\n")
    (access.outbox / "submissions.jsonl").write_text(
        json.dumps(
            {
                "kind": "diagnosis",
                "receipt_id": "diagnosis-1",
                "answer": {"mode_summaries": ["Diagnosed mode"]},
            }
        )
        + "\n"
    )
    (access.outbox / "fix_set.json").write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "rationale": "Fix rationale",
                        "action": {"verb": "request_training_data", "value": "Data wanted"},
                    }
                ]
            }
        )
    )
    (access.results / "message.txt").write_text("Pipeline reply")
    original = log.audit_export
    ours: set[Path] = set()
    agent: set[str] = set()
    checked_texts: dict[str, str] = {}
    checked_code: dict[str, str] = {}

    expected_sources = {
        "notes.md": "Saved hypothesis",
        "experiments.jsonl line 1 at /request/power_note": "Requested comparison",
        "submissions.jsonl line 1 at /answer/mode_summaries/0": "Diagnosed mode",
        "fix_set.json at /actions/0/rationale": "Fix rationale",
        "fix_set.json at /actions/0/action/value": "Data wanted",
    }
    expected_sources = {
        where + " (agent prose).json": text for where, text in expected_sources.items()
    }

    def scan(root: Any, *args: Any, **kwargs: Any) -> Any:
        assert "surface" not in kwargs
        if kwargs.get("agent_texts"):
            # Only separately named agent prose may be lenient. Every label
            # identifies its real file/field; ours cannot be sent in this mapping.
            assert not list(Path(root).iterdir())
            for where, encoded in kwargs["agent_texts"].items():
                assert where in expected_sources
                assert json.loads(encoded) == expected_sources[where]
                agent.add(json.loads(encoded))
            assert not kwargs.get("texts") and not kwargs.get("code")
        else:
            ours.add(Path(root))
            checked_texts.update(kwargs.get("texts") or {})
            checked_code.update(kwargs.get("code") or {})
        return original(root, *args, **kwargs)

    monkeypatch.setattr(log, "audit_export", scan)
    monkeypatch.setattr(checks, "audit_export", scan)
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "Pipeline system", "Pipeline task", tools=access)
    assert agent == prose
    assert {access.export, access.memory, access.outbox, access.results} <= ours
    assert run.box_dir(launched[0]) in ours
    assert {"Pipeline system", "Pipeline task"} <= set(checked_texts.values())
    assert any("schema" in name for name in checked_texts)
    assert any(name.startswith("tool ") for name in checked_texts)
    assert "src/module.py" in checked_code


@pytest.mark.parametrize(
    "record",
    [
        {"mode_summaries/0": "mount tolerance mm", "mode_summaries": ["ordinary"]},
        {"mode_summaries": {"*": "mount tolerance mm"}},
    ],
)
def test_malformed_keys_cannot_impersonate_agent_prose_locations(
    launch: tuple[run.ToolAccess, list[Path]], record: dict[str, Any]
) -> None:
    access, launched = launch
    (access.outbox / "diagnosis.json").write_text(json.dumps(record))
    with pytest.raises(GateError, match="outbox audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


def test_agent_prose_decodes_json_before_checking_identifiers(
    launch: tuple[run.ToolAccess, list[Path]],
) -> None:
    from fleetlab.agents.audit import audit_state

    access, _ = launch
    assert access.audit is not None and access.audit.fleet is not None
    prose = r'{"hypothesis": "toy\u005fcamera\u005fwobble"}'
    (access.outbox / "diagnosis.json").write_text(json.dumps({"mode_summaries": [prose]}))
    report = audit_state(access.outbox, access.audit.run_dirs, access.audit.fleet, "outbox")
    hits = [hit for hit in report.hits if hit.term == "toy_camera_wobble"]
    assert hits and all(hit.where.startswith("diagnosis.json at /mode_summaries/0") for hit in hits)


def test_mixed_measurement_records_remain_strict_trees(
    launch: tuple[run.ToolAccess, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import measure
    from fleetlab.agents.toy import ToyWorld
    from fleetlab.log.leaks import LeakReport

    access, _ = launch
    assert access.audit is not None and access.audit.fleet is not None
    assert access.env.skills is not None
    calls = []

    def scan(root: Path, terms: Any, **kwargs: Any) -> LeakReport:
        assert not kwargs
        calls.append(root)
        return LeakReport(tuple(terms))

    monkeypatch.setattr(measure, "leak_search", scan)
    # Only the run_dir/fleet fields are read; no measurement or model is run.
    toy = ToyWorld(
        base=access.export.parent,
        run_dir=access.audit.run_dirs[0],
        export=access.export,
        fleet=access.audit.fleet,
        skills=access.env.skills,
    )
    measure.blinding([{"run_dir": str(access.results), "analysis": str(access.memory)}], toy)
    assert calls == [access.results, access.memory]


@pytest.mark.parametrize("verb", ["request_training_data", "route_to_teleop", "set"])
@pytest.mark.parametrize("prose", ["mount tolerance mm", "mount_tolerance_mm"])
def test_only_public_text_valued_action_fields_use_agent_policy(
    launch: tuple[run.ToolAccess, list[Path]], verb: str, prose: str
) -> None:
    from fleetlab.agents.audit import audit_state

    access, _ = launch
    assert access.audit is not None and access.audit.fleet is not None
    (access.outbox / "fix_set.json").write_text(
        json.dumps({"actions": [{"action": {"verb": verb, "value": prose}}]})
    )
    report = audit_state(access.outbox, access.audit.run_dirs, access.audit.fleet, "outbox")
    assert report.clean == (verb != "set" and prose == "mount tolerance mm")


def _job_check(sql: str, expects: str) -> str:
    check = {
        "name": "release-flags",
        "purpose": "the release and rest flags of a job's episodes",
        "parameters": [{"name": "job", "type": "str"}],
        "sql": sql,
        "expects": expects,
    }
    return json.dumps({"receipt_id": "check-1", "session": 3, "check": check}) + "\n"


# The job-one run of 21 October lost session 4 to `verdict`, a truth-side key, in the
# agent's own check: an SQL alias and a word of its expectation.
VERDICT_SQL = (
    "select episode_id, case when released and at_rest then 'pass' else 'fail' end as verdict "
    "from diag where job_id = $job"
)


def test_a_job_check_s_own_words_pass_the_launch(
    launch: tuple[run.ToolAccess, list[Path]],
) -> None:
    from fleetlab.toolserver import jobs

    assert {jobs.CHECKS_FILE, jobs.JOB_REQUESTS_FILE, jobs.SAVED_JOBS_FILE, jobs.ACCEPTANCE_FILE,
            jobs.CHECK_RUNS_FILE, jobs.ACCOUNT_FILE} <= set(__import__(
                "fleetlab.agents.audit", fromlist=["_JOB_PROSE"])._JOB_PROSE)  # fmt: skip
    access, launched = launch
    expects = "every delivered episode flags true and verdict=pass; a healthy rate is high"
    (access.outbox / jobs.CHECKS_FILE).write_text(_job_check(VERDICT_SQL, expects))
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


@pytest.mark.parametrize("where", ["sql", "expects"])
@pytest.mark.parametrize("hidden", ["canary", "toy_camera_wobble"])
def test_a_job_check_still_may_not_carry_hidden_identifiers(
    launch: tuple[run.ToolAccess, list[Path]], where: str, hidden: str
) -> None:
    from fleetlab.toolserver import jobs

    access, launched = launch
    assert access.audit is not None
    term = str(read_canary(access.audit.run_dirs[0])) if hidden == "canary" else hidden
    sql = VERDICT_SQL + (f" -- {term}" if where == "sql" else "")
    expects = "verdict=pass" + (f" unless {term}" if where == "expects" else "")
    (access.outbox / jobs.CHECKS_FILE).write_text(_job_check(sql, expects))
    with pytest.raises(GateError, match="outbox audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


def _mount_library_check(access: run.ToolAccess, expects: str, sql: str = VERDICT_SQL) -> None:
    library = access.results / "library"
    library.mkdir(exist_ok=True)
    check = json.loads(_job_check(sql, expects))["check"]
    (library / "saved_checks.json").write_text(json.dumps([check]))
    job = {"name": "loaded-carry", "purpose": "verdict=pass on release", "check": check["name"]}
    (library / "saved_jobs.json").write_text(json.dumps([job]))


def test_a_mounted_library_s_own_words_pass_the_launch(
    launch: tuple[run.ToolAccess, list[Path]],
) -> None:
    """Job one run-4 of 21 October failed at session 1: the mounted library's saved checks
    (run-3's own sql and expects) said `verdict`."""
    access, launched = launch
    _mount_library_check(access, "every delivered episode has verdict=pass")
    with pytest.raises(ReachedLaunch):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert len(launched) == 1


def test_a_mounted_library_check_still_may_not_carry_a_canary(
    launch: tuple[run.ToolAccess, list[Path]],
) -> None:
    access, launched = launch
    assert access.audit is not None
    canary = str(read_canary(access.audit.run_dirs[0]))
    _mount_library_check(access, f"verdict=pass unless {canary}")
    with pytest.raises(GateError, match="results audit"):
        run.run_agent("claude", "model", "System", "Task", tools=access)
    assert not launched


def test_wire_check_audits_claude_s_prompt_not_its_envelope(
    launch: tuple[run.ToolAccess, list[Path]], tmp_path: Path
) -> None:
    """The keys of claude's stream-json message (`role`, `text`) are truth
    terms; the audit sees the prompt inside it, and a truth term in that prompt still fails."""
    from fleetlab.agents.audit import launch_audit

    access, _ = launch
    assert access.audit is not None
    box = tmp_path / "wire-box"
    box.mkdir()

    def envelope(prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]
        return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"

    raw = envelope("USER-PROMPT-PROBE")
    with pytest.raises(GateError, match="truth terms"):
        launch_audit(access, box, {"prompt": raw})
    assert checks._prompt_text("claude", raw) == "USER-PROMPT-PROBE"
    launch_audit(access, box, {"prompt": checks._prompt_text("claude", raw)})
    for leak in ('{"role": "gripper"}', str(read_canary(access.audit.run_dirs[0]))):
        with pytest.raises(GateError, match="truth terms"):
            launch_audit(access, box, {"prompt": checks._prompt_text("claude", envelope(leak))})
    assert checks._prompt_text("codex", "USER-PROMPT-PROBE") == "USER-PROMPT-PROBE"
    extra = json.dumps({"type": "user", "message": {"role": "user", "content": []}, "x": 1})
    with pytest.raises(GateError, match="envelope"):
        checks._prompt_text("claude", extra)
    for empty in ("", " \n"):
        with pytest.raises(GateError, match="no prompt text"):
            checks._prompt_text("claude", envelope(empty))
