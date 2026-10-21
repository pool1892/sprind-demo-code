"""Evaluator/agent boundary checks for the development-world analysis command."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from log_toy import TASK_CLASSES, write_toy_run

from fleetlab.agents.audit import tree_manifest
from fleetlab.agents.checks import GateError, source_audit
from fleetlab.agents.sessions import Analysis, EngineReply, ExperimentQueue, QueuedExperiment
from fleetlab.agents.toolenv import ToolEnv
from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
from fleetlab.contracts.world import DayRange, Windows
from fleetlab.integration import analyse as runner
from fleetlab.integration.chain import HEAVY_RUN, SYSTEM
from fleetlab.log import export_runs
from fleetlab.worlds.mini import mini_spec


def test_prompt_uses_only_public_schedule() -> None:
    spec = mini_spec()
    spec = spec.model_copy(
        update={
            "windows": Windows(
                observed=DayRange(first=2, last=7),
                experiment=DayRange(first=8, last=9),
                scoring=DayRange(first=10, last=11),
            )
        }
    )
    prompt = runner.task_prompt(spec, "logs")
    assert "observed window covers days 2 through 7" in prompt
    assert "experiment window covers days 8 through 9" in prompt
    assert "scoring window covers days 10 through 11" in prompt
    assert "No experiments can be run" in prompt
    assert "sensitivity-by-experiment," not in prompt
    assert spec.world_id not in prompt
    for fault in spec.fault_schedule:
        assert fault.fault_id not in prompt
        assert fault.kind not in prompt
        assert fault.target.scope == "fleet" or fault.target.scope not in prompt
    experiments = runner.task_prompt(spec, "experiments")
    assert "sensitivity-by-experiment," in experiments
    assert "eight A/B requests" in experiments and "60 attempts per arm" in experiments


def test_initial_export_rejects_future_data(tmp_path: Path) -> None:
    fleet = write_toy_run(tmp_path / "run")
    export = tmp_path / "export"
    export_runs([tmp_path / "run"], export, fleet=fleet, task_classes=TASK_CLASSES)
    spec = mini_spec().model_copy(
        update={
            "world_id": fleet.id,
            "windows": Windows(
                observed=DayRange(first=0, last=2),
                experiment=DayRange(first=3, last=3),
                scoring=DayRange(first=4, last=4),
            ),
        }
    )
    runs = [tmp_path / "run"]
    runner.check_observed_export(export, spec, runs, fleet)
    empty_bundle = tmp_path / "empty-bundle"
    empty_bundle.mkdir()
    for condition in ("logs", "experiments"):
        audit = source_audit(
            export,
            runs,
            ToolEnv(empty_bundle, empty_bundle, empty_bundle, None),
            fleet=fleet,
            texts={"system": SYSTEM, "task": runner.task_prompt(spec, condition)},
        )
        assert audit.passed, audit.failures()
    shortened = spec.model_copy(
        update={
            "windows": Windows(
                observed=DayRange(first=0, last=0),
                experiment=DayRange(first=1, last=1),
                scoring=DayRange(first=2, last=2),
            )
        }
    )
    with pytest.raises(ValueError, match="exclusively the observed window"):
        runner.check_observed_export(export, shortened, runs, fleet)
    # Public label and hidden world id deliberately differ after the identity repair.
    runner.check_observed_export(export, spec.model_copy(update={"world_id": "w99"}), runs, fleet)
    with pytest.raises(ValueError, match="fleet ids differ"):
        runner.check_observed_export(export, spec, runs, fleet.model_copy(update={"id": "w99"}))
    # A matching id/day range is insufficient: the visible records must agree too.
    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes = export / "log" / "episodes.parquet"
    table = pq.read_table(episodes)
    field = table.schema.get_field_index("outcome")
    table = table.set_column(field, "outcome", pa.array(["success"] * table.num_rows))
    pq.write_table(table, episodes)
    with pytest.raises(ValueError, match="episode records differ"):
        runner.check_observed_export(export, spec, runs, fleet)


def test_output_cannot_write_inside_stored_world(tmp_path: Path) -> None:
    args = argparse.Namespace(world_root=tmp_path, export=tmp_path / "export", out=tmp_path / "new")
    with pytest.raises(ValueError, match="outside the stored world"):
        runner.analyse(args)


def test_queue_writes_in_new_root_and_audits_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = write_toy_run(tmp_path / "base-run")
    files = tmp_path / "stored" / "files"
    output = tmp_path / "new"
    output.mkdir()
    queue_file = output / "session-01.json"
    reply = EngineReply(results=[], export=output / "engine" / "export")
    env = ToolEnv(tmp_path, tmp_path, tmp_path, None)
    engine = runner.DevelopmentEngine(
        files,
        output,
        10,
        env,
        fleet,
        [tmp_path / "base-run"],
        {"task": "public task"},
        (tmp_path / "cleared-run",),
    )
    run = Mock(
        side_effect=lambda *a, **kw: queue_file.with_suffix(".reply.json").write_text(
            reply.model_dump_json()
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    audit = SimpleNamespace(passed=True, as_json=lambda: {"passed": True})
    source_audit = Mock(return_value=audit)
    monkeypatch.setattr(runner, "source_audit", source_audit)
    queue = ExperimentQueue(
        world_id="w01",
        analysis_id="analysis",
        session=1,
        created_at="now",
        export=str(tmp_path / "export"),
        experiments=[],
    )
    assert engine.run_queue(queue_file, queue) == reply
    command = run.call_args.args[0]
    assert command[0] == str(HEAVY_RUN)
    assert command[command.index("--engine-root") + 1] == str(output / "engine")
    assert command[command.index("--truth") + 1] == str(files / "world_truth.json")
    assert command[command.index("--base-run") + 1] == str(tmp_path / "cleared-run")
    assert source_audit.call_args.kwargs["fleet"] is fleet
    assert source_audit.call_args.args[0] == reply.export
    assert json.loads(queue_file.with_suffix(".audit.json").read_text())["passed"]
    bad = SimpleNamespace(
        passed=False, as_json=lambda: {"passed": False}, failures=lambda: ["trait leaked"]
    )
    source_audit.return_value = bad
    with pytest.raises(GateError, match="trait leaked"):
        engine.run_queue(queue_file, queue)


def test_experiment_base_runs_reproduce_exact_export(tmp_path: Path) -> None:
    fleet = write_toy_run(tmp_path / "run")
    export = tmp_path / "export"
    export_runs([tmp_path / "run"], export, fleet=fleet, task_classes=TASK_CLASSES)
    full_fleet = fleet.model_copy(
        update={
            "maintenance_log": [m.model_copy(update={"day": 99}) for m in fleet.maintenance_log]
        }
    )
    runner.check_base_runs(
        [tmp_path / "run"], export, tmp_path / "good", full_fleet, fleet, TASK_CLASSES
    )
    with pytest.raises(ValueError, match="exact cleared export"):
        runner.check_base_runs(
            [tmp_path / "run"],
            export,
            tmp_path / "wrong-public-fleet",
            full_fleet,
            full_fleet,
            TASK_CLASSES,
        )
    # A different public event string must not quietly reappear on continuation.
    import pyarrow as pa
    import pyarrow.parquet as pq

    events = export / "log" / "events.parquet"
    table = pq.read_table(events)
    column = table.schema.get_field_index("detail")
    table = table.set_column(column, "detail", pa.array(["public correction"] * len(table)))
    pq.write_table(table, events)
    with pytest.raises(ValueError, match="exact cleared export"):
        runner.check_base_runs(
            [tmp_path / "run"], export, tmp_path / "bad", fleet, fleet, TASK_CLASSES
        )


def test_resume_copies_pending_state_and_preserves_call_provenance(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "continued"
    export = tmp_path / "export"
    export.mkdir()
    (export / "public.txt").write_text("observations")
    call = source / "agent-runs" / "call-1"
    call.mkdir(parents=True)
    (call / "audit.json").write_text(
        json.dumps({"passed": True, "manifest": {"export": tree_manifest(export)}})
    )
    analysis = Analysis.create("w01", "experiments", base=source)
    queued = QueuedExperiment(
        experiment_id="x0001",
        session=1,
        request=ExperimentRequest(
            arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[])],
            robots=["r01"],
            episodes_per_arm=4,
            paired_starts=True,
            power_note="test",
        ),
    )
    (analysis.outbox / "experiments.jsonl").write_text(queued.model_dump_json() + "\n")
    (analysis.memory / "notes.md").write_text("Evidence from the completed session")
    session = analysis.root / "sessions" / "session-01.json"
    session.write_text(
        json.dumps(
            {
                "run_dir": str(source / "agent-runs" / "call-1"),
                "number": 1,
                "state": "waiting",
                "queued": ["x0001"],
            }
        )
    )
    analysis.write_queue(1, export)
    expected = dict(
        world_root="world",
        world_hash="hash",
        export=str(export),
        condition="experiments",
        repeat=1,
        model="gpt-6-astra",
        effort="xhigh",
    )
    meta = {
        **expected,
        "state": "waiting",
        "ended": 1,
        "analysis": str(analysis.root),
        "final_export": str(export),
    }
    (source / "run.json").write_text(json.dumps(meta))
    original = tree_manifest(source)
    copied = runner.copy_waiting_analysis(source, output, expected)
    assert copied.analysis_id == analysis.analysis_id
    assert copied.pending() == [queued]
    assert copied.notes() == analysis.notes()
    assert (copied.root / "sessions" / session.name).read_bytes() == session.read_bytes()
    assert tree_manifest(source) == original
    with pytest.raises(ValueError, match="outside its source"):
        runner.copy_waiting_analysis(source, analysis.root / "nested", expected)
    changed = queued.model_copy(
        update={"request": queued.request.model_copy(update={"episodes_per_arm": 5})}
    )
    (analysis.outbox / "experiments.jsonl").write_text(changed.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="completed session or saved queue"):
        runner.copy_waiting_analysis(source, tmp_path / "changed-request", expected)
    (analysis.outbox / "experiments.jsonl").write_text(queued.model_dump_json() + "\n")
    (export / "public.txt").write_text("changed observations")
    with pytest.raises(ValueError, match="export changed"):
        runner.copy_waiting_analysis(source, tmp_path / "changed-export", expected)
    (export / "public.txt").write_text("observations")
    with pytest.raises(ValueError, match="world_hash"):
        runner.copy_waiting_analysis(
            source, tmp_path / "wrong", {**expected, "world_hash": "other"}
        )
    (source / "run.json").write_text(json.dumps({**meta, "ended": None}))
    with pytest.raises(ValueError, match="ended waiting"):
        runner.copy_waiting_analysis(source, tmp_path / "active", expected)
    (source / "run.json").write_text(json.dumps(meta))
    (source / "engine").mkdir()
    (source / "engine" / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="no ended queue checkpoint"):
        runner.copy_waiting_analysis(source, tmp_path / "unverified-engine", expected)
    timing = analysis.root / "queues" / "session-01.timing.json"
    timing.write_text('{"state": "running"}')
    with pytest.raises(ValueError, match="queue has not ended"):
        runner.copy_waiting_analysis(source, tmp_path / "running-engine", expected)
    timing.write_text('{"state": "failed", "ended": 1}')
    copied = runner.copy_waiting_analysis(source, tmp_path / "simulated", expected)
    assert copied.pending() == [queued]
    assert (tmp_path / "simulated" / "engine" / "run.json").is_file()
