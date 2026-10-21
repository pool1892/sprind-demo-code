"""The run writer: every table round-trips, rows off the schema are refused, runs resume.

src/fleetlab/log/writer.py; docs/design/contracts.md, section 11. Toy data from
tests/log_toy.py, no simulator.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from log_toy import (
    EPISODES_PER_DAY,
    POLICY,
    TASK_CLASSES,
    TASKS,
    all_jobs,
    continue_job,
    episode_data,
    fixture_hash_of,
    flat_frame,
    record,
    replay_jobs,
    run_jobs,
    toy_fleet,
    write_job,
    write_toy_run,
)

from fleetlab.contracts import log as contracts_log
from fleetlab.contracts.actions import catalogue_rows, costs_rows
from fleetlab.contracts.env import REASON_PLACED, REASON_STEP_LIMIT, Camera
from fleetlab.contracts.identity import SoftwareVersion, started_at_for
from fleetlab.contracts.log import SCHEMA, ColumnSpec, TableSpec, Visibility, type_name
from fleetlab.contracts.oracle import fault_list_hash
from fleetlab.contracts.rollout import JobMode
from fleetlab.log import (
    EpisodeWriter,
    LogReader,
    ResumeMismatch,
    RowError,
    RunLocked,
    RunWriter,
    SchemaError,
    run_status,
)
from fleetlab.log._files import STAGING_DIR
from fleetlab.log._rows import EPISODE_TABLES, PER_EPISODE_TABLES, RUN_TABLES, check_keys
from fleetlab.log.export import mp4_times
from fleetlab.log.runfiles import full_data_dictionary, read_chain, read_identities, read_table
from fleetlab.video import write_mp4


def _log_files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted((root / "log").rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("toy") / "run"
    write_toy_run(run)
    return run


def test_every_table_of_the_schema_has_one_route() -> None:
    routes = [*PER_EPISODE_TABLES, *EPISODE_TABLES, *RUN_TABLES]
    # The tables only an export holds (the probes) are no run's to write.
    assert sorted(routes) == sorted(t.name for t in SCHEMA.values() if t.source == "run")
    assert set(PER_EPISODE_TABLES) == {"steps", "chain"}
    assert set(RUN_TABLES) == {
        "health",
        "change_log",
        "maintenance",
        "costs",
        "catalogue",
        "oracle",
    }


def test_the_run_has_the_layout_of_the_contracts(toy_run: Path) -> None:
    status = run_status(toy_run, all_jobs())
    assert status.finalized and not status.pending
    assert len(status.done) == len(all_jobs())
    for table in SCHEMA.values():
        if table.file_key is None and table.source == "run":
            assert (toy_run / table.path).is_file(), table.path
    assert (toy_run / "log/description.json").is_file()
    assert (toy_run / "run.json").is_file()
    for job in replay_jobs():
        assert (toy_run / f"truth/chain/{job.key}.npz").is_file()
        assert not (toy_run / f"log/steps/{job.key}.parquet").exists()


def test_every_table_round_trips(toy_run: Path) -> None:
    checked: set[str] = set()
    reader = LogReader(toy_run)
    jobs = {j.key: j for j in all_jobs()}
    runs = run_jobs()

    # steps: values as given, the last command empty, frames numbered.
    for job in runs:
        data = episode_data(job)
        rows = reader.steps(job.identity.episode_id).to_pylist()
        assert len(rows) == len(data.steps)
        frame = 0
        for row, telemetry, frames in zip(rows, data.steps, data.frames, strict=True):
            assert row["step"] == telemetry.step
            assert row["time_s"] == telemetry.time_s
            assert row["joint_pos"] == telemetry.joint_pos.tolist()
            assert row["gripper_qpos"] == telemetry.gripper_qpos.tolist()
            assert row["eef_quat"] == telemetry.eef_quat.tolist()
            if np.isnan(telemetry.commanded_action).all():
                assert row["commanded_action"] is None
            else:
                assert row["commanded_action"] == telemetry.commanded_action.tolist()
            assert (row["chunk_index"], row["chunk_step"]) == (
                telemetry.chunk_index,
                telemetry.chunk_step,
            )
            assert row["frame_index"] == (None if frames is None else frame)
            frame += frames is not None
    checked.add("steps")

    # chain: hashes as given, NaN as row 0's executed action.
    for key, job in jobs.items():
        data = episode_data(job)
        chain = read_table(toy_run, "chain", key).to_pylist()
        assert [r["step"] for r in chain] == [c[0] for c in data.chain]
        for row, (_, state, images, executed) in zip(chain, data.chain, strict=True):
            assert row["state_hash"] == state
            assert row["scene_image_hash"] == images[Camera.scene]
            assert row["wrist_image_hash"] == images[Camera.wrist]
            if executed is None:
                assert all(math.isnan(x) for x in row["executed_action"])
            else:
                assert row["executed_action"] == executed.tolist()
        # The fixture hash: one per file, a 0-d array.
        fixture = read_chain(toy_run, key)["fixture_hash"]
        assert fixture.shape == () and str(fixture) == fixture_hash_of(job)
    checked.add("chain")

    # episodes: identity fields, outcome, length from the last step, videos.
    episodes = {r["episode_id"]: r for r in reader.query("SELECT * FROM episodes").to_pylist()}
    assert sorted(episodes) == sorted(j.identity.episode_id for j in runs)
    for job in runs:
        row, data, identity = episodes[job.identity.episode_id], episode_data(job), job.identity
        assert (row["robot_id"], row["site_id"], row["day"]) == (
            identity.robot_id,
            identity.site_id,
            identity.day,
        )
        assert row["task_text"] == TASKS[identity.task_index]
        started = datetime.time.fromisoformat(started_at_for(identity, EPISODES_PER_DAY))
        assert row["started_at"] == started
        assert started == datetime.time(8 + 2 * identity.episode_index, 0)
        assert row["steps"] == len(data.steps) - 1
        assert row["duration_s"] == data.steps[-1].time_s
        assert row["outcome"] == ("success" if data.success else "failure")
        if job.record_video:
            assert row["video_scene"] == f"video/{identity.episode_id}.scene.mp4"
        else:
            assert row["video_scene"] is None and row["video_wrist"] is None
    checked.add("episodes")

    events = reader.query("SELECT * FROM events ORDER BY episode_id, step").to_pylist()
    assert [(e["episode_id"], e["kind"]) for e in events[:2]] == [
        ("e000001", "grasp"),
        ("e000001", "release"),
    ]
    assert {e["episode_id"] for e in events} == {"e000001", "e000005", "e000009"}
    checked.add("events")

    health = reader.query("SELECT * FROM health").to_pylist()
    assert len(health) == 10 and health[0]["metric"] == "grip_force_n"
    checked.add("health")

    fleet = toy_fleet()
    change = reader.query("SELECT * FROM change_log").to_pylist()
    assert change == [
        {
            "day": 1,
            "scope": "r02",
            "key": "control.action_delay_steps",
            "old": "0",
            "new": "3",
            "author": "ops",
        }
    ]
    maintenance = reader.query("SELECT * FROM maintenance").to_pylist()
    assert maintenance == [
        {"day": 1, "robot_id": "r01", "component": "gripper", "verb": "clean", "note": "pads"}
    ]
    assert len(fleet.change_log) == len(change)
    checked |= {"change_log", "maintenance"}

    # The action costs, written by finalize from the contracts.
    assert reader.query("SELECT * FROM costs").to_pylist() == costs_rows()
    checked.add("costs")

    # The action catalogue, written by write_model from the contracts.
    assert reader.query("SELECT * FROM catalogue").to_pylist() == catalogue_rows(
        fleet, TASK_CLASSES
    )
    checked.add("catalogue")

    # Hidden tables, through the run reader.
    identities = read_identities(toy_run)
    assert identities == [j.identity for j in runs]
    table = read_table(toy_run, "identities").to_pylist()
    assert table[1]["scene_seed"] == runs[1].identity.scene_seed
    assert table[1]["policy_name"] == POLICY.name and table[1]["in_worker"] is False
    checked.add("identities")

    faults = read_table(toy_run, "faults_active").to_pylist()
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in faults:
        by_episode.setdefault(row["episode_id"], []).append(row)
    for job in runs:
        rows = by_episode.get(job.identity.episode_id, [])
        assert [r["fault_id"] for r in rows] == [f.fault_id for f in job.identity.faults]
        for row, fault in zip(rows, job.identity.faults, strict=True):
            assert row["kind"] == fault.kind and row["level"] == fault.level
            assert json.loads(row["params"]) == fault.params
            if fault.target.component is not None:
                assert row["nominal"] is None
                assert row["target_component"] == fault.target.component.value
            else:
                assert json.loads(row["nominal"]) == fault.nominal
    checked.add("faults_active")

    policies = {r["episode_id"]: r for r in read_table(toy_run, "effective_policy").to_pylist()}
    assert policies["e000004"]["sampling_steps"] == 5
    assert policies["e000003"]["sampling_steps"] == POLICY.sampling_steps
    checked.add("effective_policy")

    verdicts = {r["episode_id"]: r for r in read_table(toy_run, "verdicts").to_pylist()}
    assert verdicts["e000003"]["true_success"] is False
    assert verdicts["e000003"]["true_reason"] == REASON_STEP_LIMIT
    assert verdicts["e000003"]["success_step"] is None
    assert verdicts["e000001"]["libero_first_success_step"] == 2
    assert verdicts["e000001"]["true_reason"] == REASON_PLACED
    assert verdicts["e000001"]["success_step"] == len(episode_data(runs[0]).steps) - 1
    checked.add("verdicts")

    replays = read_table(toy_run, "replays").to_pylist()
    expected = [
        {
            "episode_id": j.identity.episode_id,
            "active_faults": fault_list_hash(j.effective_faults),
            "success": not episode_data(j).success,
        }
        for j in replay_jobs()
    ]
    assert replays == expected
    checked.add("replays")

    decisions = read_table(toy_run, "decisions").to_pylist()
    assert decisions == [
        {
            "episode_id": "e000004",
            "step": 2,
            "decision": "continue",
            "hold_steps": 1,
            "decision_ms": 812.5,
            "model_ms": 640.0,
            "rationale": "",
            "call_status": "ok",
            "error": None,
            "retry_budget_steps": None,
        }
    ]
    checked.add("decisions")

    # The oracle table is written whole by whoever computes the gains.
    oracle_rows = [
        {"item": "f01", "item_kind": "fault_effect", "rescued": 1, "broken": 3, "net": -2}
    ]
    with RunWriter(toy_run, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        run.write_table("oracle", oracle_rows)
    assert read_table(toy_run, "oracle").to_pylist() == oracle_rows
    checked.add("oracle")

    # job_outcomes: one row per job of any mode.
    outcomes = {r["job_key"]: r for r in read_table(toy_run, "job_outcomes").to_pylist()}
    assert sorted(outcomes) == sorted(jobs)
    for key, job in jobs.items():
        row, data = outcomes[key], episode_data(job)
        assert row["episode_id"] == job.identity.episode_id
        assert row["mode"] == job.mode.value
        assert json.loads(row["removed"]) == job.remove
        assert json.loads(row["added"]) == []
        if job.mode is JobMode.run:
            assert row["success"] is data.success
            assert row["steps"] == len(data.steps) - 1
            assert row["placed_step"] == (1 if data.success else None)
            assert row["libero_step"] == 2
            assert row["from_snapshot_step"] is None
        else:
            assert row["success"] is not data.success
            assert row["steps"] == data.chain[-1][0]
            assert row["placed_step"] == (3 if row["success"] else None)
            assert row["libero_step"] is None
            snapshot = job.from_snapshot
            assert row["from_snapshot_step"] == (None if snapshot is None else snapshot.step)
    continued = outcomes[continue_job().key]
    assert continued["mode"] == "continue" and continued["from_snapshot_step"] == 2
    order = read_table(toy_run, "job_outcomes").column("job_key").to_pylist()
    assert order == sorted(order, key=lambda k: (k.split(".")[0], k))
    checked.add("job_outcomes")

    run_tables = {t.name for t in SCHEMA.values() if t.source == "run"}
    assert checked == run_tables, "a table of the schema has no round-trip check"


def test_a_hidden_column_in_a_visible_row_is_refused(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [], finalize=False)
    job = run_jobs()[0]
    writer = EpisodeWriter(tmp_path, job, task_text="t", episodes_per_robot_day=EPISODES_PER_DAY)
    telemetry = episode_data(job).steps[0]
    row = {f: getattr(telemetry, f) for f in telemetry.__dataclass_fields__}
    with pytest.raises(RowError, match="has no column 'executed_action'"):
        writer.add_step({**row, "executed_action": [0.0] * 7})
    with pytest.raises(RowError, match="frame_index is assigned by the writer"):
        writer.add_step({**row, "frame_index": 0})
    with pytest.raises(RowError, match="lacks"):
        check_keys("episodes", {"episode_id": "e000001"})
    with pytest.raises(RowError, match="has no column 'scene_seed'"):
        check_keys(
            "episodes", {c.name: None for c in SCHEMA["episodes"].columns} | {"scene_seed": 1}
        )


def test_a_hidden_column_slipped_into_a_visible_table_stops_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = SCHEMA["episodes"]
    leak = ColumnSpec("scene_seed", pa.int64(), Visibility.hidden, "seed set before the reset")
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
    monkeypatch.setattr(contracts_log, "SCHEMA", patched)
    with pytest.raises(SchemaError, match="scene_seed"):
        RunWriter(tmp_path / "run", record(), episodes_per_robot_day=EPISODES_PER_DAY)
    # And a row carrying it is refused: it is not on the allow-list.
    import fleetlab.log._rows as rows

    monkeypatch.setattr(rows, "SCHEMA", patched)
    names = {c.name: None for c in episodes.columns}
    with pytest.raises(RowError, match=r"'scene_seed' is tagged hidden .* not on the allow-list"):
        check_keys("episodes", {**names, "scene_seed": 12345})


def test_types_are_checked(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [], finalize=False)
    with RunWriter(tmp_path, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        with pytest.raises(RowError):
            run.write_table(
                "health",
                [
                    {
                        "day": "Monday",
                        "robot_id": "r01",
                        "component": "arm",
                        "metric": "m",
                        "value": 1.0,
                    }
                ],
            )
        with pytest.raises(RowError, match="written by the episode writers"):
            run.write_table("episodes", [])


def test_visible_tables_do_not_depend_on_the_order_jobs_finish(tmp_path: Path) -> None:
    write_toy_run(tmp_path / "a", all_jobs())
    write_toy_run(tmp_path / "b", list(reversed(all_jobs())))
    a, b = _log_files(tmp_path / "a"), _log_files(tmp_path / "b")
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []  # videos included: the encoding repeats
    assert any(k.endswith(".mp4") for k in a)


def test_a_replay_writes_nothing_visible(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [], finalize=False)
    replay = replay_jobs()[0]
    writer = EpisodeWriter(tmp_path, replay, task_text="t", episodes_per_robot_day=EPISODES_PER_DAY)
    with pytest.raises(RowError, match="writes nothing visible"):
        writer.add_step(episode_data(replay).steps[0])
    with pytest.raises(RowError, match="writes nothing visible"):
        writer.add_event(0, "grasp", "x")
    with pytest.raises(RowError, match="belongs to the run job"):
        writer.set_verdict(True, "x", None)
    for step, state, images, executed in episode_data(replay).chain:
        writer.add_chain(step, state, images, executed)
    with pytest.raises(RowError, match="needs success="):
        writer.finish()
    with pytest.raises(RowError, match="no fixture hash"):
        writer.finish(success=True)
    writer.set_fixture_hash(fixture_hash_of(replay))
    writer.finish(success=True)
    assert not list((tmp_path / "log/steps").iterdir())


def test_a_run_job_must_be_complete(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [], finalize=False)
    job = run_jobs()[0]  # with video
    data = episode_data(job)
    writer = EpisodeWriter(tmp_path, job, task_text="t", episodes_per_robot_day=EPISODES_PER_DAY)
    for telemetry in data.steps:
        writer.add_step(telemetry)  # no frames although record_video is set
    for step, state, images, executed in data.chain:
        writer.add_chain(step, state, images, executed)
    with pytest.raises(RowError, match=r"takes \['success'\] from its steps and verdict"):
        writer.finish(success=True)
    with pytest.raises(RowError, match="needs outcome="):
        writer.finish()
    with pytest.raises(RowError, match="no verdict"):
        writer.finish(outcome=True)
    writer.set_verdict(True, REASON_PLACED, None)
    with pytest.raises(RowError, match="record_video is True but 0 frames"):
        writer.finish(outcome=True)
    due = len(data.chain)
    with pytest.raises(RowError, match=f"chain step {due + 1} given where step {due} is due"):
        writer.add_chain(due + 1, data.chain[0][1], data.chain[0][2], None)
    other = EpisodeWriter(tmp_path, job, task_text="t", episodes_per_robot_day=EPISODES_PER_DAY)
    with pytest.raises(RowError, match="step 1 given where step 0 is due"):
        other.add_step(data.steps[1])
    with pytest.raises(RowError, match="every step needs all cameras"):
        other.add_step(data.steps[0], {Camera.scene: flat_frame(1, 0, Camera.scene)})
    with pytest.raises(RowError, match="not a hash"):
        other.add_chain(0, "not-a-hash", data.chain[0][2], None)


def test_a_killed_run_is_detected_and_completed(tmp_path: Path) -> None:
    jobs = all_jobs()
    straight = tmp_path / "straight"
    write_toy_run(straight, jobs)

    run_dir = tmp_path / "killed"
    write_toy_run(run_dir, jobs[:5], finalize=False)
    # A job killed after its steps and videos but before its chain, and a file half written.
    killed = jobs[6]
    assert killed.record_video
    writer = write_job(run_dir, killed, finish=False)
    writer._write_chain = lambda path: (_ for _ in ()).throw(KeyboardInterrupt())  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        writer.finish(outcome=True)
    (run_dir / STAGING_DIR).mkdir(exist_ok=True)
    (run_dir / STAGING_DIR / "1234.e000008.parquet").write_bytes(b"half")
    status = run_status(run_dir, jobs)
    assert status.partial == (killed.key,)
    assert not status.finalized
    assert (run_dir / f"log/steps/{killed.key}.parquet").is_file()

    with RunWriter(run_dir, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        assert run.resumed
        assert run.discarded == (killed.key,)
        assert run.cleared_files == 1
        assert not (run_dir / f"log/steps/{killed.key}.parquet").exists()
        assert not (run_dir / f"log/video/{killed.key}.scene.mp4").exists()
        pending = run.pending(jobs)
        assert [j.key for j in pending] == [j.key for j in jobs[5:]]
        for job in pending:
            write_job(run_dir, job)
        assert run.finalize().finalized
    assert _log_files(run_dir) == _log_files(straight)
    for table in EPISODE_TABLES:
        assert read_table(run_dir, table).equals(read_table(straight, table)), table


def test_finalize_takes_only_finished_jobs_and_can_run_again(tmp_path: Path) -> None:
    jobs = run_jobs()
    write_toy_run(tmp_path, jobs[:3])
    with RunWriter(tmp_path, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        assert len(LogReader(tmp_path).episode_ids()) == 3
        in_flight = write_job(tmp_path, jobs[3], finish=False)  # buffered, nothing on disk
        assert run.finalize().finalized
        assert len(LogReader(tmp_path).episode_ids()) == 3
        write_job(tmp_path, jobs[4])
        in_flight.finish(outcome=False)
        assert run.finalize().finalized
        assert run.finalize().finalized  # again: nothing changes
    assert LogReader(tmp_path).episode_ids() == [j.identity.episode_id for j in jobs[:5]]


def test_a_resume_must_keep_what_fixes_the_episodes(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [])
    n = EPISODES_PER_DAY
    software = SoftwareVersion(
        lock_hash="lock1", libero_patch_set="p1", mujoco_version="3.3.7", miopen_db_hash="db0"
    )
    others = {
        "software": (record().model_copy(update={"software": software}), n),
        "replay_rules": (record().model_copy(update={"replay_rules": ["batch_size_fixed"]}), n),
        "batch_size": (record().model_copy(update={"batch_size": 8}), n),
        "episodes_per_robot_day": (record(), n + 1),
    }
    for name, (other, per_day) in others.items():
        with pytest.raises(ResumeMismatch, match=name):
            RunWriter(tmp_path, other, episodes_per_robot_day=per_day)
    job = run_jobs()[0]
    with pytest.raises(ResumeMismatch, match="episodes per robot and day"):
        EpisodeWriter(tmp_path, job, task_text="t", episodes_per_robot_day=n + 1)
    with RunWriter(tmp_path, record(), episodes_per_robot_day=n) as run:  # the lock was released
        updated = run.update_record(chain_sample=["e000001"], chain_ok=True, extra={"x": 1})
        assert updated.extra == {"episodes_per_robot_day": n, "x": 1}
        with pytest.raises(ValueError, match="cannot change"):
            run.update_record(extra={"episodes_per_robot_day": 9})
        with pytest.raises(ValueError, match="cannot change"):
            run.update_record(batch_size=2)
    assert json.loads((tmp_path / "run.json").read_text())["chain_ok"] is True
    with RunWriter(tmp_path, record(), episodes_per_robot_day=n) as run:
        assert run.record.extra["x"] == 1  # kept from the file
    # Indices past the day's episodes are arm episodes from 16:00 every three minutes;
    # n + 160 would pass midnight.
    late = job.identity.model_copy(update={"episode_index": n + 160})
    with pytest.raises(ValueError, match="after 23:59:59"):
        EpisodeWriter(
            tmp_path,
            job.model_copy(update={"identity": late}),
            task_text="t",
            episodes_per_robot_day=n,
        )


def test_a_continuation_starts_from_its_snapshot(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [], finalize=False)
    job = continue_job()
    data = episode_data(job)
    writer = EpisodeWriter(tmp_path, job, task_text="t", episodes_per_robot_day=EPISODES_PER_DAY)
    first, state, images, _ = data.chain[0]
    assert first == 2
    with pytest.raises(RowError, match="chain step 0 given where step 2 is due"):
        writer.add_chain(0, state, images, None)
    with pytest.raises(RowError, match=r"first state .* has no executed action"):
        writer.add_chain(2, state, images, np.ones(7, np.float32))
    with pytest.raises(RowError, match="writes nothing visible"):
        writer.add_step(episode_data(run_jobs()[3]).steps[0])
    for step, state_hash, hashes, executed in data.chain:
        writer.add_chain(step, state_hash, hashes, executed)
    writer.set_fixture_hash(fixture_hash_of(job))
    with pytest.raises(RowError, match="the chain ends at step"):
        writer.finish(success=True, steps=99)
    writer.finish(success=True)
    with RunWriter(tmp_path, record(), episodes_per_robot_day=EPISODES_PER_DAY) as run:
        run.finalize()
    assert read_table(tmp_path, "replays").num_rows == 0
    (row,) = read_table(tmp_path, "job_outcomes").to_pylist()
    assert (row["mode"], row["from_snapshot_step"], row["steps"]) == ("continue", 2, 4)
    assert math.isnan(read_table(tmp_path, "chain", job.key).to_pylist()[0]["executed_action"][0])


def test_a_verdict_names_a_known_reason(tmp_path: Path) -> None:
    write_toy_run(tmp_path, [], finalize=False)
    job = run_jobs()[0]
    writer = EpisodeWriter(tmp_path, job, task_text="t", episodes_per_robot_day=EPISODES_PER_DAY)
    with pytest.raises(RowError, match="verdict reason 'timeout' is not one of"):
        writer.set_verdict(False, "timeout", None)
    writer.set_verdict(True, REASON_PLACED, 40, 52, placed_step=45)


def test_one_writer_per_run(tmp_path: Path) -> None:
    with (
        RunWriter(tmp_path, record(), episodes_per_robot_day=EPISODES_PER_DAY),
        pytest.raises(RunLocked),
    ):
        RunWriter(tmp_path, record(), episodes_per_robot_day=EPISODES_PER_DAY)


def test_the_canary_is_planted_on_the_hidden_side(toy_run: Path) -> None:
    import pyarrow.parquet as pq

    token = (toy_run / "truth/canary.txt").read_text().strip()
    assert token.startswith("CANARY-")
    for table in SCHEMA.values():
        if table.tree == "truth" and table.storage == "parquet":
            metadata = pq.read_schema(toy_run / table.path).metadata or {}
            assert metadata[b"fleetlab.canary"] == token.encode(), table.name
        if table.tree == "log" and table.file_key is None and table.source == "run":
            assert pq.read_schema(toy_run / table.path).metadata is None
    assert token not in (toy_run / "log/description.json").read_text()


def test_video_encoding_repeats_and_carries_no_time(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 255, (32, 48, 3), dtype=np.uint8) for _ in range(6)]
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    write_mp4(a, frames, 5, crf=23, preset="veryfast", repeatable=True)
    write_mp4(b, frames, 5, crf=23, preset="veryfast", repeatable=True)
    assert a.read_bytes() == b.read_bytes()
    times = mp4_times(a)
    assert times and not any(times)
    assert b"Lavf" not in a.read_bytes()  # no muxer version tag


def test_the_description_lists_the_task_classes(toy_run: Path) -> None:
    description = json.loads((toy_run / "log/description.json").read_text())
    assert description["task_classes"] == list(TASK_CLASSES)
    assert "traits" not in json.dumps(description)


def test_a_steps_file_holds_one_episode(toy_run: Path) -> None:
    for job in run_jobs():
        steps = read_table(toy_run, "steps", job.identity.episode_id)
        assert set(steps.column("episode_id").to_pylist()) == {job.identity.episode_id}


def test_a_column_without_a_visibility_tag_stops_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = SCHEMA["steps"]
    untagged = ColumnSpec("motor_temp", pa.float64(), None, "motor temperature")  # type: ignore[arg-type]
    patched = {
        **SCHEMA,
        "steps": TableSpec(
            steps.name,
            steps.tree,
            steps.path,
            steps.storage,
            steps.description,
            (*steps.columns, untagged),
        ),
    }
    monkeypatch.setattr(contracts_log, "SCHEMA", patched)
    with pytest.raises(SchemaError, match=r"steps\.motor_temp: no visibility tag"):
        RunWriter(tmp_path / "run", record(), episodes_per_robot_day=EPISODES_PER_DAY)


def test_the_full_dictionary_lists_every_column_with_its_visibility() -> None:
    text = full_data_dictionary()
    for table in SCHEMA.values():
        assert f"## {table.name} ({table.tree})" in text
        for column in table.columns:
            kind = type_name(column.type)
            assert f"| {column.name} | {kind} | {column.visibility.value} |" in text


def test_a_job_s_files_reach_the_disk_before_its_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.log._files as files

    write_toy_run(tmp_path, [], finalize=False)
    events: list[tuple[str, str]] = []
    real_fsync, real_replace = files._fsync, os.replace

    def fsync(path: Path) -> None:
        events.append(("fsync", Path(path).name))
        real_fsync(path)

    def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        events.append(("rename", Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(files, "_fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    write_job(tmp_path, run_jobs()[0])
    renames = [name for kind, name in events if kind == "rename"]
    assert renames[-1] == "e000001.npz" and "e000001.json" in renames
    assert renames.index("e000001.parquet") < renames.index("e000001.npz")
    for i, (kind, name) in enumerate(events):
        if kind == "rename":
            assert events[i - 1][0] == "fsync" and events[i - 1][1].endswith(name)
            assert events[i + 1][0] == "fsync"  # the directory, after the rename
