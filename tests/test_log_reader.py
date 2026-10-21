"""The DuckDB reader over an export: views per table, steps and frames, locked file access.

src/fleetlab/log/reader.py and runfiles.py.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest
from log_toy import TASK_CLASSES, frame_value, run_jobs, toy_fleet, write_toy_run

from fleetlab.contracts.env import Camera
from fleetlab.contracts.log import visible_schema
from fleetlab.contracts.oracle import OracleTable
from fleetlab.log import LogReader, export_runs
from fleetlab.log.dictionary import data_dictionary
from fleetlab.log.leaks import GENERIC_TERMS, leak_search
from fleetlab.log.runfiles import open_run, oracle_rows, read_identities


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("reader")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    return base / "run", base / "export"


@pytest.mark.parametrize("materialize", [False, True])
def test_every_visible_table_is_a_view(toy: tuple[Path, Path], materialize: bool) -> None:
    with LogReader(toy[1], materialize=materialize) as reader:
        assert reader.tables == tuple(visible_schema())
        for name, spec in visible_schema().items():
            columns = reader.query(f"SELECT * FROM {name} LIMIT 0").column_names
            assert columns == [c.name for c in spec.columns], name
        counts = reader.query(
            "SELECT e.outcome, count(*) AS n, sum(s.rows) AS rows FROM episodes e "
            "JOIN (SELECT episode_id, count(*) AS rows FROM steps GROUP BY 1) s USING (episode_id) "
            "GROUP BY 1 ORDER BY 1"
        ).to_pylist()
    assert sum(c["n"] for c in counts) == len(run_jobs())
    assert [c["outcome"] for c in counts] == ["failure", "success"]


def test_an_empty_table_still_has_its_columns(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [])  # no episodes: steps has no file at all
    with LogReader(tmp_path) as reader:
        assert reader.query("SELECT count(*) AS n FROM steps").to_pylist() == [{"n": 0}]
        assert "commanded_action" in reader.query("SELECT * FROM steps").column_names


def test_lists_index_from_one_and_the_last_command_is_empty(toy: tuple[Path, Path]) -> None:
    with LogReader(toy[1]) as reader:
        rows = reader.query(
            "SELECT step, commanded_action[1] AS x, joint_pos[7] AS j7 FROM steps "
            "WHERE episode_id = 'e000002' ORDER BY step"
        ).to_pylist()
    assert rows[-1]["x"] is None and all(r["x"] is not None for r in rows[:-1])
    assert all(r["j7"] is not None for r in rows)


def test_frames_come_from_the_nearest_logged_step(toy: tuple[Path, Path]) -> None:
    with LogReader(toy[1]) as reader:
        logged = reader.query(
            "SELECT step, frame_index FROM steps WHERE episode_id = 'e000001' "
            "AND frame_index IS NOT NULL ORDER BY step"
        ).to_pylist()
        assert [r["step"] for r in logged] == [0, 2, 4]
        frames = reader.frames("e000001", [0, 1, 3, 4, 99], Camera.wrist)
    assert [(f.step, f.frame_step, f.frame_index) for f in frames] == [
        (0, 0, 0),
        (1, 0, 0),  # a tie goes to the earlier frame
        (3, 2, 1),
        (4, 4, 2),
        (99, 4, 2),
    ]
    for frame in frames:
        assert frame.image.shape == (16, 16, 3) and frame.image.dtype == np.uint8
        expected = frame_value(1, frame.frame_step, Camera.wrist)
        assert abs(float(frame.image.mean()) - expected) < 3


def test_frames_of_an_episode_without_video(toy: tuple[Path, Path]) -> None:
    with LogReader(toy[1]) as reader:
        with pytest.raises(ValueError, match="has no video"):
            reader.frames("e000002", [0], "scene")
        with pytest.raises(KeyError):
            reader.frames("e000099", [0], "scene")
        with pytest.raises(ValueError, match="not a neutral episode"):
            reader.frames("../truth", [0], "scene")


def test_a_locked_reader_cannot_open_files(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run, export = toy
    canary = str(run / "truth/canary.txt")
    identities = str(run / "truth/identities.parquet")
    written = tmp_path / "out.csv"
    with LogReader(export, materialize=True, lock=True) as reader:
        assert reader.query("SELECT count(*) AS n FROM episodes").to_pylist() == [{"n": 10}]
        for sql in (
            f"SELECT * FROM read_text('{canary}')",
            f"SELECT * FROM read_parquet('{export}/log/episodes.parquet')",
            f"COPY episodes TO '{written}'",
            f"COPY episodes TO '{export}/log/copy.csv'",
            "SET enable_external_access = true",
        ):
            with pytest.raises(duckdb.Error):
                reader.query(sql)
    assert not written.exists() and not (export / "log/copy.csv").exists()
    # On a run directory, views and a lock keep DuckDB inside log/: truth/ stays shut.
    for materialize in (False, True):
        with LogReader(run, materialize=materialize, lock=True) as reader:
            assert reader.query("SELECT count(*) AS n FROM steps").to_pylist()[0]["n"] > 0
            for sql in (
                f"SELECT * FROM read_parquet('{identities}')",
                f"SELECT * FROM read_text('{canary}')",
                f"SELECT * FROM read_text('{run}/run.json')",
            ):
                with pytest.raises(duckdb.Error):
                    reader.query(sql)


def test_description_and_dictionary(toy: tuple[Path, Path]) -> None:
    with LogReader(toy[1]) as reader:
        full = reader.description()
        static = reader.description(include_logs=False)
        assert full["fleet"]["change_log"] and "change_log" not in static["fleet"]
        assert static["task_classes"] == list(TASK_CLASSES)
        assert reader.data_dictionary() == data_dictionary()
    assert toy_fleet().id == full["fleet"]["id"]


def test_the_reader_s_source_names_nothing_of_the_hidden_side(tmp_path: Path) -> None:
    import fleetlab.log.dictionary
    import fleetlab.log.reader

    texts = {
        m.__name__: Path(m.__file__ or "").read_text()
        for m in (fleetlab.log.reader, fleetlab.log.dictionary)
    }
    report = leak_search(tmp_path, GENERIC_TERMS, texts)
    assert report.clean, report.summary()


def test_the_run_reader_sees_the_hidden_tables(toy: tuple[Path, Path]) -> None:
    run, _ = toy
    with open_run(run) as reader:
        rows = reader.query(
            "SELECT e.episode_id, e.outcome, v.true_success, count(f.fault_id) AS faults "
            "FROM episodes e JOIN verdicts v USING (episode_id) "
            "LEFT JOIN faults_active f USING (episode_id) GROUP BY ALL ORDER BY 1"
        ).to_pylist()
    assert len(rows) == 10 and rows[1]["faults"] == 1 and rows[5]["faults"] == 2
    table = OracleTable(read_identities(run), oracle_rows(run))
    assert len(table.fault_ids()) == 2
