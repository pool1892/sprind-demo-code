"""The episode log schema: every column tagged and described, the visible export built from
the allow-list, no measured time in a visible table, and a validator for a run's log tree.

docs/design/contracts.md, section 11.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fleetlab.contracts import log
from fleetlab.contracts.env import VERDICT_REASONS
from fleetlab.contracts.log import (
    JOB_TABLE_NAMES,
    SCHEMA,
    ColumnSpec,
    LogLeakError,
    TableSpec,
    Visibility,
    allowed_columns,
    arrow_schema,
    data_dictionary_markdown,
    export_table_markdown,
    log_tree_problems,
    schema_problems,
    validate_log_tree,
    visible_schema,
)


def test_every_column_has_a_tag_a_type_and_a_one_line_description() -> None:
    assert schema_problems() == []
    for table in SCHEMA.values():
        for column in table.columns:
            assert isinstance(column.visibility, Visibility)
            assert column.description.strip() and "\n" not in column.description


def test_the_note_s_tables_are_declared_in_their_trees() -> None:
    visible = {
        "episodes",
        "steps",
        "events",
        "health",
        "change_log",
        "maintenance",
        "costs",
        "catalogue",
    }
    run_tables = {t.name for t in SCHEMA.values() if t.tree == "log" and t.source == "run"}
    assert run_tables == visible
    # Probes the agent composes (contracts.probes) and the job tables (contracts.jobs):
    # visible tables only an export or a job run's tree holds.
    export_only = {t.name for t in SCHEMA.values() if t.source == "export"}
    assert export_only == {"probe_episodes", "probe_segments", "probe_steps"} | {
        "job_episodes",
        "job_step_events",
        "object_steps",
        "goal_steps",
        "robot_steps",
    }
    assert all(SCHEMA[name].tree == "log" for name in export_only)
    hidden = {
        "identities",
        "faults_active",
        "effective_policy",
        "chain",
        "replays",
        "oracle",
        "decisions",
    }
    assert hidden <= {t.name for t in SCHEMA.values() if t.tree == "truth"}
    assert "calls" not in SCHEMA
    assert "inference_ms" not in {c.name for c in SCHEMA["steps"].columns}
    assert "executed_action" in {c.name for c in SCHEMA["chain"].columns}
    assert "start_state" in {c.name for c in SCHEMA["identities"].columns}
    assert {"params", "nominal"} <= {c.name for c in SCHEMA["faults_active"].columns}
    assert "verdicts" in SCHEMA
    verdicts = SCHEMA["verdicts"]
    assert [c.name for c in verdicts.columns][-1] == "success_step"
    assert "VERDICT_REASONS" in verdicts.column("true_reason").description
    assert len(VERDICT_REASONS) == 6
    fixture = SCHEMA["chain"].column("fixture_hash")
    assert fixture.per_file and fixture.visibility is Visibility.hidden
    assert [c.name for c in SCHEMA["chain"].columns if c.per_file] == ["fixture_hash"]
    assert [c.name for c in SCHEMA["job_outcomes"].columns] == [
        "job_key",
        "episode_id",
        "mode",
        "removed",
        "added",
        "from_snapshot_step",
        "success",
        "steps",
        "placed_step",
        "libero_step",
    ]
    assert SCHEMA["job_outcomes"].tree == "truth" and allowed_columns("job_outcomes") == ()
    assert [c.name for c in SCHEMA["replays"].columns] == ["episode_id", "active_faults", "success"]
    # A replay has its own chain, under its job key (contracts.rollout.Job.key).
    assert SCHEMA["chain"].path == "truth/chain/{job_key}.npz"
    assert SCHEMA["chain"].file_key == "job_key" and SCHEMA["steps"].file_key == "episode_id"
    assert "NaN in row 0" in SCHEMA["chain"].column("executed_action").description


def test_arrow_schemas_cover_both_trees() -> None:
    assert arrow_schema("steps").names == list(allowed_columns("steps"))
    assert arrow_schema("faults_active").names[-2:] == ["params", "nominal"]
    with pytest.raises(ValueError, match="truth side"):
        arrow_schema("faults_active", visible_only=True)


def test_the_visible_export_excludes_every_hidden_column() -> None:
    for name, table in SCHEMA.items():
        hidden = {c.name for c in table.columns if c.visibility is Visibility.hidden}
        assert not hidden & set(allowed_columns(name)), name
        if table.tree == "truth":
            assert allowed_columns(name) == ()
    assert set(visible_schema()) == {t.name for t in SCHEMA.values() if t.tree == "log"}


def test_the_allow_list_is_built_from_tags_not_by_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hidden column slipped into a visible table stays out of the export, and the
    # schema check names it.
    leak = ColumnSpec("scene_seed", pa.int64(), Visibility.hidden, "seed set before the reset")
    episodes = SCHEMA["episodes"]
    patched = {
        **SCHEMA,
        "episodes": TableSpec(
            episodes.name,
            episodes.tree,
            episodes.path,
            episodes.storage,
            episodes.description,
            (*episodes.columns, leak),
        ),
    }
    monkeypatch.setattr(log, "SCHEMA", patched)
    assert "scene_seed" not in allowed_columns("episodes")
    assert any("scene_seed" in p for p in schema_problems(patched))


def test_no_visible_column_is_a_measured_time() -> None:
    for table in visible_schema().values():
        for column in table.columns:
            assert column.clock != "measured", f"{table.name}.{column.name}"
            assert not log.MEASURED_TIME_NAME.search(column.name)
    assert SCHEMA["episodes"].column("started_at").clock == "simulated"
    assert SCHEMA["episodes"].column("duration_s").clock == "simulated"
    measured = ColumnSpec(
        "latency_ms", pa.float64(), Visibility.visible, "call latency", clock="measured"
    )
    table = TableSpec("calls", "log", "log/calls.parquet", "parquet", "calls", (measured,))
    assert len(schema_problems({"calls": table})) == 2


def test_the_data_dictionary_shows_vectors_as_duckdb_readers_think_of_them() -> None:
    text = data_dictionary_markdown()
    assert "| joint_pos | double[7] |" in text
    assert "| commanded_action | float[7] |" in text
    assert "fixed_size_list" not in text


def test_the_data_dictionary_shows_only_visible_columns() -> None:
    text = data_dictionary_markdown()
    # The loop's exports add the probe tables (the default); a job run's tree, the job tables.
    extra = export_table_markdown() + "\n" + export_table_markdown(JOB_TABLE_NAMES)
    assert "## job_episodes" not in export_table_markdown()
    for table in visible_schema().values():
        # The tables only an export holds are documented by the export that holds them.
        shown = text if table.source == "run" else extra
        assert f"## {table.name}" in shown
        assert (f"## {table.name}" in text) == (table.source == "run")
        for column in table.columns:
            assert f"| {column.name} |" in shown
    for word in ("scene_seed", "executed_action", "state_hash", "identities", "fault"):
        assert word not in text


# A run directory, written as the log writer will.


def _row(table: str, overrides: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    values: dict[str, Any] = {
        "episode_id": "e000001",
        "robot_id": "r01",
        "site_id": "s1",
        "day": 3,
        "task_text": "pick up the black bowl",
        "started_at": datetime.time(9, 30),
        "steps": 2,
        "outcome": "success",
        "duration_s": 0.1,
        "video_scene": "video/e000001.scene.mp4",
        "video_wrist": "video/e000001.wrist.mp4",
        "experiment_id": None,
        "step": 0,
        "time_s": 0.0,
        "joint_pos": [0.0] * 7,
        "joint_vel": [0.0] * 7,
        "gripper_qpos": [0.0] * 2,
        "eef_pos": [0.0] * 3,
        "eef_quat": [0.0, 0.0, 0.0, 1.0],
        "commanded_action": [0.0] * 7,
        "chunk_index": 0,
        "chunk_step": 0,
        "frame_index": 0,
        "kind": "grasp",
        "detail": "closed on the bowl",
    }
    values.update(overrides or {})
    return {c: [values[c]] for c in allowed_columns(table)}


def _write(
    path: Path, table: str, extra: dict[str, list[Any]] | None = None, **overrides: Any
) -> None:
    data = _row(table, overrides)
    schema = arrow_schema(table)
    arrays = {name: pa.array(values, type=schema.field(name).type) for name, values in data.items()}
    for name, values in (extra or {}).items():
        arrays[name] = pa.array(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def clean_run(root: Path) -> Path:
    _write(root / "log/episodes.parquet", "episodes")
    _write(root / "log/steps/e000001.parquet", "steps")
    _write(root / "log/events.parquet", "events")
    (root / "log/description.json").write_text(json.dumps({"fleet": {"sites": []}}))
    (root / "log/video").mkdir()
    (root / "log/video/e000001.scene.mp4").write_bytes(b"\0")
    (root / "log/video/e000001.wrist.mp4").write_bytes(b"\0")
    (root / "truth").mkdir()
    (root / "truth/anything.parquet").write_bytes(b"not checked: truth is not exported")
    return root


def test_a_clean_log_tree_passes(tmp_path: Path) -> None:
    validate_log_tree(clean_run(tmp_path))


def test_the_validator_refuses_a_hidden_column(tmp_path: Path) -> None:
    root = clean_run(tmp_path)
    _write(root / "log/episodes.parquet", "episodes", extra={"scene_seed": [12345]})
    problems = log_tree_problems(root)
    assert any("scene_seed" in p and "allow-list" in p for p in problems)
    with pytest.raises(LogLeakError):
        validate_log_tree(root)


def test_the_validator_refuses_a_measured_time(tmp_path: Path) -> None:
    root = clean_run(tmp_path)
    _write(root / "log/steps/e000001.parquet", "steps", extra={"inference_ms": [41.5]})
    problems = log_tree_problems(root)
    assert any("inference_ms" in p and "measured time" in p for p in problems)


def test_the_validator_refuses_identifiers_that_are_not_neutral(tmp_path: Path) -> None:
    root = clean_run(tmp_path)
    _write(root / "log/steps/d2-degrees4-t0-i1.parquet", "steps", episode_id="d2-degrees4-t0-i1")
    _write(root / "log/episodes.parquet", "episodes", robot_id="robot_1")
    problems = log_tree_problems(root)
    assert any("d2-degrees4-t0-i1.parquet: episode id in the name" in p for p in problems)
    assert any("episode_id values not neutral" in p for p in problems)
    assert any("robot_id values not neutral" in p for p in problems)


def test_the_validator_refuses_unknown_files_and_hidden_fields(tmp_path: Path) -> None:
    root = clean_run(tmp_path)
    (root / "log/fault_table.json").write_text("{}")
    (root / "log/description.json").write_text(json.dumps({"robots": [{"traits": {}}]}))
    (root / "log/video/e000001.thermal.mp4").write_bytes(b"\0")
    problems = log_tree_problems(root)
    assert any("fault_table.json: not a file of the visible log" in p for p in problems)
    assert any("hidden field 'traits'" in p for p in problems)
    assert any("unknown camera" in p for p in problems)
