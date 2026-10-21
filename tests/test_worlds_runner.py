"""The fleet runner at scale on the toy cell: a whole world as one resumable
job (files, both runs, records, events, export with the leak audit, summary), and a resume
after an interruption that reruns only what was lost and gives the same chains."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from engine_helpers import SOFTWARE, config, fault

from fleetlab.contracts.identity import Task
from fleetlab.contracts.model import ComponentState, ConfigVersion, Fleet, Robot, Site
from fleetlab.contracts.rollout import Job, RunLayout
from fleetlab.contracts.vocabulary import Component, ConfigKey
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec
from fleetlab.engine import manifests, toy
from fleetlab.engine.chain import Chain, compare
from fleetlab.log import export_problems, run_status
from fleetlab.worlds import runner
from fleetlab.worlds.generate import World, write_world
from fleetlab.worlds.layout import WorldPaths
from fleetlab.worlds.truth import FaultTable

PER_DAY = 4
ROBOTS = (1, 2, 3)
TASKS = [Task("toy", 0), Task("toy", 1)]


def toy_world() -> World:
    """Three robots at one site, three days (one per window), two toy faults."""
    camera = fault("toy_camera_shift", 1, robot=2, pixels=2)
    delay = fault("toy_delay", 2, robot=3, steps=3)
    windows = Windows(
        observed=DayRange(first=0, last=0),
        experiment=DayRange(first=1, last=1),
        scoring=DayRange(first=2, last=2),
    )
    spec = WorldSpec(
        world_id="w01",
        seed=3,
        size="small",
        sites=1,
        robots_per_site=len(ROBOTS),
        days=3,
        windows=windows,
        fault_schedule=[camera, delay],
        patterns=Patterns(),
        policy=toy.TOY_SPEC,
        tasks=TASKS,
        episodes_per_robot_day=PER_DAY,
    )
    jobs = []
    number = 1
    for day in range(3):
        for robot in ROBOTS:
            faults = {1: [], 2: [camera], 3: [delay]}[robot]
            for index in range(PER_DAY):
                ident = manifests.identity(
                    world_seed=3,
                    episode_number=number,
                    robot=robot,
                    site=1,
                    day=day,
                    episode_index=index,
                    suite="toy",
                    task_index=index % 2,
                    start_state=number % 7,
                    policy=toy.TOY_SPEC,
                    batch_size=4,
                    software=SOFTWARE,
                    image_size=32,
                    max_steps=60,
                    faults=faults,
                )
                jobs.append(Job(identity=ident, record_video=index == 0))
                number += 1

    def make_robot(n: int) -> Robot:
        return Robot(
            id=f"r0{n}",
            site_id="s1",
            components={c: ComponentState(component=c, installed_day=-10) for c in Component},
        )

    fleet = Fleet(
        id="w01",
        sites=[Site(id="s1", robots=[make_robot(n) for n in ROBOTS])],
        config=ConfigVersion(version=1, values={ConfigKey.control_action_delay_steps: 0}),
    )
    health = [
        {
            "day": d,
            "robot_id": f"r0{n}",
            "component": "gripper",
            "metric": "grip_force",
            "value": 60.0 - d - n,
        }
        for d in range(3)
        for n in ROBOTS
    ]
    return World(
        spec=spec,
        software=SOFTWARE,
        fleet=fleet,
        observed_fleet=fleet,
        jobs=jobs,
        health=health,
        task_texts={TASKS[0]: "carry the block", TASKS[1]: "push the block"},
        fault_table=FaultTable(
            world_id="w01",
            windows=windows,
            episodes_per_window={"observed": 12, "experiment": 12, "scoring": 12},
            rule="toy",
            faults=[],
        ),
    )


def engine(tmp_path: Path) -> runner.EngineSettings:
    toy_config = config(tmp_path / "unused", workers=2)
    fields = {
        name: getattr(toy_config, name)
        for name in (
            "device",
            "precision",
            "miopen",
            "env_factory",
            "policy_factory",
            "check_factory",
            "fault_registry",
            "life",
            "server_threads",
        )
    }
    return runner.EngineSettings(workers=2, overrides=fields, contacts=False)


def chains(run_dir: Path, jobs: Sequence[Job]) -> dict[str, Chain]:
    layout = RunLayout(run_dir)
    return {j.key: Chain.load(layout.chain(j)) for j in jobs}


def test_a_world_runs_end_to_end_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = toy_world()
    paths = WorldPaths(tmp_path / "w01", "w01")
    settings = engine(tmp_path)
    record = runner.run_world(world, paths, settings, write=write_world)
    assert record["ready"] is True
    assert set(record["steps"]) == set(runner.STEPS)
    for name in ("observed", "later"):
        assert run_status(paths.run(name)).finalized
        assert record["steps"][name]["chain_ok"] is True
    assert (paths.files / "hash").is_file()
    assert export_problems(paths.export) == []
    assert record["steps"]["export"]["leaks_clean"] is True
    summary = record["summary"]
    assert summary["episodes"] == 36
    assert summary["measured"]["observed"]["logged"]["n"] == 12
    assert summary["gate3"]["catalogue_ratio"]["entries"] > 0
    import pyarrow.parquet as pq

    health = pq.read_table(paths.observed / "log" / "health.parquet").to_pylist()
    assert {r["day"] for r in health} == {0}  # the observed run holds the observed days only
    later_jobs = runner.run_jobs(world, "later")
    before = chains(paths.later, later_jobs)

    # A second call does nothing new: no run, records or events rewritten, no hashed
    # world file touched.
    stamps = {
        p: p.stat().st_mtime_ns
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file() and "runner" not in p.name and ".world-check" not in str(p)
    }
    again = runner.run_world(world, paths, settings, write=write_world)
    assert again["ready"] is True
    assert again["steps"]["later"]["attempts"][-1]["ok"] is True
    assert again["steps"]["observed"]["skipped"] is True
    assert len(again["steps"]["observed-events"]["attempts"]) == 1
    after_stamps = {p: p.stat().st_mtime_ns for p in stamps if p.exists()}
    assert after_stamps == stamps

    # An interruption: the later run lost its last jobs, the export was never built.
    lost = later_jobs[-3:]
    layout = RunLayout(paths.later)
    for job in lost:
        layout.chain(job).unlink()
        (paths.later / "truth" / "parts" / f"{job.key}.json").unlink()
    import shutil

    shutil.rmtree(paths.export)
    resumed = runner.run_world(world, paths, settings, write=write_world)
    assert resumed["ready"] is True
    assert resumed["steps"]["later"]["jobs_run_now"] == len(lost)
    after = chains(paths.later, later_jobs)
    for key, chain in before.items():
        assert compare(after[key], chain).identical, key
    assert export_problems(paths.export) == []
    attempts = resumed["steps"]["later"]["attempts"]
    assert len(attempts) == 3 and all(a["ok"] for a in attempts)

    # Kill after publication, before its result reaches the runner. The pre-publication
    # receipt already binds these bytes, so recovery can reuse it without rebuilding.
    import fleetlab.log

    shutil.rmtree(paths.export)
    real_export = fleetlab.log.export_runs

    def publish_then_die(*args: object, **kwargs: object) -> None:
        real_export(*args, **kwargs)  # type: ignore[arg-type]
        raise SystemExit("killed after export publication")

    monkeypatch.setattr(fleetlab.log, "export_runs", publish_then_die)
    with pytest.raises(SystemExit, match="publication"):
        runner.run_world(world, paths, settings, only="export")
    published = {p: p.read_bytes() for p in paths.export.rglob("*") if p.is_file()}
    assert paths.read_record()["ready"] is False
    # A missing saved clean flag must not prevent using the content-bound receipt.
    runner.note(paths, "export", leaks_clean=None)
    recovered = runner.run_world(world, paths, settings, only="export")
    assert recovered["ready"] is True
    assert recovered["steps"]["export"]["skipped"] is True
    assert recovered["steps"]["export"]["leaks_clean"] is True
    assert {p: p.read_bytes() for p in published} == published
    assert runner.run_world(world, paths, settings, only="export")["ready"] is True

    # A saved clean receipt cannot bless a subsequently modified export.
    events_path = paths.export / "log/events.parquet"
    events = pq.read_table(events_path)
    details = events["detail"].to_pylist()
    details[0] = "gripper_pad_age_days=80"
    import pyarrow as pa

    pq.write_table(events.set_column(3, "detail", pa.array(details)), events_path)
    with pytest.raises(fleetlab.log.ExportRefused, match="gripper_pad_age_days"):
        runner.run_world(world, paths, settings, only="export")
    assert paths.read_record()["ready"] is False
    assert runner.is_ready(paths.read_record(), paths, fleet=world.fleet) is False
    assert paths.read_record()["steps"]["export"]["leaks_clean"] is False


def test_a_changed_world_is_refused_once_episodes_ran(tmp_path: Path) -> None:
    world = toy_world()
    paths = WorldPaths(tmp_path / "w01", "w01")
    settings = engine(tmp_path)
    runner.run_world(world, paths, settings, write=write_world, through="observed")
    other = World(**{**world.__dict__, "spec": world.spec.model_copy(update={"seed": 4})})
    with pytest.raises(runner.WorldChanged):
        runner.run_world(other, paths, settings, write=write_world, through="files")


def test_a_changed_world_before_any_episode_is_rewritten(tmp_path: Path) -> None:
    world = toy_world()
    paths = WorldPaths(tmp_path / "w01", "w01")
    settings = engine(tmp_path)
    first = runner.run_world(world, paths, settings, write=write_world, only="files")
    other = World(**{**world.__dict__, "spec": world.spec.model_copy(update={"seed": 4})})
    second = runner.run_world(other, paths, settings, write=write_world, only="files")
    assert first["steps"]["files"]["hash"] != second["steps"]["files"]["hash"]
    assert (paths.files / "hash").read_text().strip() == second["steps"]["files"]["hash"]


def test_a_broken_chain_is_not_ready_and_not_exported(tmp_path: Path) -> None:
    world = toy_world()
    paths = WorldPaths(tmp_path / "w01", "w01")
    settings = engine(tmp_path)
    runner.run_world(world, paths, settings, write=write_world, through="observed-events")
    record = json.loads((paths.observed / "run.json").read_text())
    record["chain_ok"] = False
    (paths.observed / "run.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="chain_ok"):
        runner.run_world(world, paths, settings, write=write_world, only="export")
    assert not paths.export.exists()
    assert runner.is_ready(paths.read_record(), paths, fleet=world.fleet) is False


def test_one_runner_per_world_root(tmp_path: Path) -> None:
    paths = WorldPaths(tmp_path / "w01", "w01")
    with runner.runner_lock(paths), pytest.raises(runner.RunnerBusy):
        runner.run_world(toy_world(), paths, engine(tmp_path), write=write_world, only="files")


def test_ready_survives_oracle_plan_run_and_resume(tmp_path: Path) -> None:
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.world import world_ready

    world = toy_world()
    paths = WorldPaths(tmp_path / "w01", "w01")
    settings = engine(tmp_path)
    assert runner.run_world(world, paths, settings, write=write_world)["ready"]
    oracle = Oracle.open(paths.root, registry=settings.overrides["fault_registry"])
    oracle.step_plan()
    assert paths.ready and world_ready(paths.root) is None
    result = oracle.step_run(here=True, overrides=dict(settings.overrides))
    assert result["chain_ok"] is True
    assert paths.ready and world_ready(paths.root) is None
    oracle.step_run(here=True, overrides=dict(settings.overrides))
    assert paths.ready and world_ready(paths.root) is None
    (paths.root / "oracle/reader-added.json").write_text('{"future_reader_key": 1}')
    assert paths.ready


def test_false_receipt_is_not_evidence(tmp_path: Path) -> None:
    from fleetlab.log import leaks

    world, paths = checked_fixture(tmp_path)
    checked = leaks.write_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    assert paths.ready
    path = Path(checked["receipt"])
    receipt = json.loads(path.read_text())
    receipt["clean"] = False
    path.write_text(json.dumps(receipt))
    assert not paths.ready


def test_certify_command_and_reader_diagnostic_leave_records_unchanged(tmp_path: Path) -> None:
    from fleetlab.live.data import development_runs
    from fleetlab.log.leaks import main

    _world, paths = checked_fixture(tmp_path)
    record = paths.record.read_bytes()
    assert "python -m fleetlab.log certify" in (paths.readiness_problem or "")
    with pytest.raises(ValueError, match=r"no clean receipt.*python -m fleetlab.log certify"):
        development_runs(paths.root)
    assert main(["certify", str(paths.root)]) == 0
    assert paths.ready
    assert paths.record.read_bytes() == record


def test_one_audit_per_export_runner_pass_with_recorded_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.log.export as exports
    from fleetlab.log import leaks

    world, paths = checked_fixture(tmp_path)
    import shutil

    shutil.rmtree(paths.export)
    real = leaks.audit_export
    calls = []

    def count(*a: object, **k: object) -> object:
        calls.append(a[0])
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(leaks, "audit_export", count)
    monkeypatch.setattr(exports, "audit_export", count)
    monkeypatch.setattr(runner, "_chain_ok", lambda *a: True)
    result = runner.run_world(world, paths, engine(tmp_path), only="export")
    assert len(calls) == 1
    assert result["steps"]["export"]["audit_seconds"] > 0
    assert runner.run_world(world, paths, engine(tmp_path), only="export")["ready"]
    assert len(calls) == 1


def checked_fixture(
    tmp_path: Path, world_id: str = "w01", *, root_name: str | None = None
) -> tuple[World, WorldPaths]:
    from log_toy import toy_fleet, write_toy_run

    from fleetlab.log import export_runs

    fleet = toy_fleet()
    fleet.robots()[0].traits["fleet_only_private_property"] = 19
    observed = fleet.model_copy(deep=True)
    for robot in observed.robots():
        robot.traits.clear()
    world = replace(
        toy_world(),
        spec=toy_world().spec.model_copy(update={"world_id": world_id}),
        fleet=fleet,
        observed_fleet=observed,
    )
    paths = WorldPaths(tmp_path / (root_name or world_id), world_id)
    write_toy_run(paths.observed)
    paths.files.mkdir()
    (paths.files / "spec.json").write_text(world.spec.model_dump_json())
    (paths.files / "model.json").write_text(world.fleet.model_dump_json())
    export_runs([paths.observed], paths.export, fleet=world.observed_fleet, audit_fleet=world.fleet)
    runner.set_fields(
        paths,
        world_id=world_id,
        ready=True,
        steps={
            step: {"done": True, "chain_ok": True, "leaks_clean": True} for step in runner.STEPS
        },
        summary={},
    )
    return world, paths


def test_leaking_run_is_checked_once_and_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    import fleetlab.log.export as exports
    from fleetlab.log import ExportRefused, leaks

    world, paths = checked_fixture(tmp_path)
    shutil.rmtree(paths.export)
    path = paths.observed / "log/events.parquet"
    events = pq.read_table(path)
    details = events["detail"].to_pylist()
    details[0] = "gripper_pad_age_days=80"
    pq.write_table(events.set_column(3, "detail", pa.array(details)), path)
    audited = []
    published = []
    real_audit = leaks.audit_export
    real_replace = os.replace

    def count(*args: object, **kwargs: object) -> object:
        audited.append(args[0])
        return real_audit(*args, **kwargs)  # type: ignore[arg-type]

    def publish(source: Path, target: Path) -> None:
        if target == paths.export:
            published.append(source)
        real_replace(source, target)

    monkeypatch.setattr(leaks, "audit_export", count)
    monkeypatch.setattr(exports, "audit_export", count)
    monkeypatch.setattr(os, "replace", publish)
    monkeypatch.setattr(runner, "_chain_ok", lambda *a: True)
    with pytest.raises(ExportRefused, match="gripper_pad_age_days"):
        runner.run_world(world, paths, engine(tmp_path), only="export")
    assert not paths.export.exists()
    assert not published  # Even publishing and deleting immediately would fail.
    assert len(audited) == 1 and audited[0] != paths.export
    record = paths.read_record()
    assert record["steps"]["export"]["leaks_clean"] is False
    assert record["steps"]["export"]["audit_seconds"] > 0


def test_certify_uses_world_name_when_fleet_id_differs(tmp_path: Path) -> None:
    from fleetlab.log.leaks import main

    world, paths = checked_fixture(tmp_path, world_id="w77")
    assert world.fleet.id != world.spec.world_id == paths.root.name
    record = paths.record.read_bytes()
    assert main(["certify", str(paths.root)]) == 0
    assert paths.ready
    assert paths.record.read_bytes() == record


def test_legacy_flags_need_a_new_receipt_and_bytes_and_terms_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.log import leaks
    from fleetlab.oracle.world import world_ready

    world, paths = checked_fixture(tmp_path)
    original = paths.record.read_bytes()
    assert not paths.ready and world_ready(paths.root) is not None
    receipt = leaks.write_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    assert receipt["clean"] is True
    assert paths.record.read_bytes() == original
    assert paths.ready and world_ready(paths.root) is None
    changed = world.fleet.model_copy(deep=True)
    changed.robots()[0].traits["future_private_property"] = 3
    # Reader writes after certification cannot retroactively change the audited tree.
    assert runner.is_ready(paths.read_record(), paths, fleet=changed)
    with monkeypatch.context() as patch:
        patch.setattr(leaks, "audit_identifier", lambda *a, **k: "upgraded-checker")
        assert not paths.ready
    with (paths.export / "data_dictionary.md").open("a") as handle:
        handle.write("\nordinary new text\n")
    assert not paths.ready


def test_later_unreadable_truth_archive_does_not_change_a_checked_export(tmp_path: Path) -> None:
    from fleetlab.log.leaks import write_export_check

    world, paths = checked_fixture(tmp_path)
    write_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    assert paths.ready
    (paths.observed / "truth/broken.npz").write_bytes(b"incomplete archive")
    assert paths.ready


@pytest.mark.parametrize(
    "selection", [{"only": "summary"}, {"only": "files"}, {"through": "later-events"}]
)
def test_partial_calls_cannot_reuse_a_receipt_for_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: dict[str, str]
) -> None:
    from fleetlab.log.leaks import write_export_check

    world, paths = checked_fixture(tmp_path)
    write_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    assert paths.ready
    (paths.export / "data_dictionary.md").write_text("fleet_only_private_property=19")
    monkeypatch.setattr(runner, "step_files", lambda *a: "unchanged")
    monkeypatch.setattr(runner, "step_run", lambda *a: {"skipped": True})
    monkeypatch.setattr(runner, "_chain_ok", lambda *a: True)
    result = runner.run_world(
        world,
        paths,
        engine(tmp_path),
        summarize=lambda *a: {},
        only=selection.get("only"),
        through=selection.get("through"),
    )
    assert result["ready"] is False and not paths.ready
    assert result["steps"]["export"]["leaks_clean"] is False


def test_runner_hands_full_fleet_to_publication_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import fleetlab.log
    from fleetlab.log import leaks

    world, paths = checked_fixture(tmp_path)
    import shutil

    shutil.rmtree(paths.export)
    monkeypatch.setattr(runner, "_chain_ok", lambda *a: True)
    real_export = fleetlab.log.export_runs

    def observed_export(*args: object, **kwargs: object) -> object:
        assert kwargs.get("fleet") is world.observed_fleet
        assert kwargs.get("audit_fleet") is world.fleet
        return real_export(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fleetlab.log, "export_runs", observed_export)
    assert runner.step_export(world, paths)["leaks_clean"] is True
    real_audit = leaks.audit_export

    def observed_audit(*args: object, **kwargs: object) -> object:
        assert kwargs.get("fleet") is world.fleet
        return real_audit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(leaks, "audit_export", observed_audit)
    target = paths.export / "log/events.parquet"
    table = pq.read_table(target)
    values = table["detail"].to_pylist()
    values[0] = "fleet_only_private_property=19"
    pq.write_table(table.set_column(3, "detail", pa.array(values)), target)
    with pytest.raises(fleetlab.log.ExportRefused, match="fleet_only_private_property"):
        runner.step_export(world, paths)
    assert paths.read_record()["steps"]["export"]["leaks_clean"] is False


def test_recovery_requires_structure_as_well_as_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.log import ExportRefused
    from fleetlab.log.leaks import write_export_check

    world, paths = checked_fixture(tmp_path)
    write_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    (paths.export / "surplus.txt").write_text("hello")
    monkeypatch.setattr(runner, "_chain_ok", lambda *a: True)
    with pytest.raises(ExportRefused, match=r"surplus\.txt"):
        runner.step_export(world, paths)
    assert paths.read_record()["steps"]["export"]["leaks_clean"] is False
