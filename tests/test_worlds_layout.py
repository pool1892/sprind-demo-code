"""Record-derived world identities, including every reader of an arbitrary root."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from log_toy import write_toy_run
from test_live_model import dataset
from test_worlds_runner import checked_fixture

from fleetlab.dashboard.world_page import load_world
from fleetlab.live import __main__ as live_command
from fleetlab.live.data import Dataset, development_runs
from fleetlab.log import leaks
from fleetlab.oracle.world import world_ready
from fleetlab.policies.teleop.measure import require_ready
from fleetlab.worlds import runner
from fleetlab.worlds.layout import WorldPaths


@pytest.fixture
def checked_root(tmp_path: Path) -> WorldPaths:
    world, paths = checked_fixture(tmp_path, world_id="w77", root_name="arbitrary-root")
    write_toy_run(paths.later)
    # Set up readiness using explicit fixture paths, independent of root discovery.
    receipt = leaks.write_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    assert receipt["clean"] and paths.ready
    return paths


def test_oracle_reads_a_root_named_differently_from_its_world(
    checked_root: WorldPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner.is_ready

    def check(record: dict[str, object], paths: WorldPaths) -> bool:
        assert paths == checked_root
        return original(record, paths)

    monkeypatch.setattr(runner, "is_ready", check)
    assert checked_root.root.name != checked_root.world_id
    assert world_ready(checked_root.root) is None


def test_certify_reads_a_root_named_differently_from_its_world(checked_root: WorldPaths) -> None:
    before = checked_root.record.read_bytes()
    assert leaks.main(["certify", str(checked_root.root)]) == 0
    assert checked_root.record.read_bytes() == before
    assert checked_root.ready


def test_live_data_reads_both_runs_by_recorded_world_id(checked_root: WorldPaths) -> None:
    assert development_runs(checked_root.root) == [checked_root.observed, checked_root.later]


def test_teleop_reads_a_root_named_differently_from_its_world(
    checked_root: WorldPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner.readiness_problem

    def check(record: dict[str, object], paths: WorldPaths) -> str | None:
        assert paths == checked_root
        return original(record, paths)

    monkeypatch.setattr(runner, "readiness_problem", check)
    loaded = require_ready(checked_root.root)
    assert loaded["files"] == checked_root.files
    assert loaded["spec"]["world_id"] == checked_root.world_id


def test_live_prepare_world_labels_the_dataset_with_its_recorded_id(
    checked_root: WorldPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "data.npz"
    monkeypatch.setattr(
        sys, "argv", ["live", "prepare-world", str(checked_root.root), "--out", str(out)]
    )

    def build(world: str, runs: list[Path]) -> Dataset:
        assert world == checked_root.world_id
        assert runs == [checked_root.observed, checked_root.later]
        data = dataset()
        for episode in data.episodes:
            episode["world"] = world
        return data

    monkeypatch.setattr(live_command, "build_dataset", build)
    # Run selection is checked above; isolate the CLI's dataset-label decision.
    monkeypatch.setattr(
        live_command, "development_runs", lambda root: [checked_root.observed, checked_root.later]
    )
    live_command.main()
    assert {e["world"] for e in Dataset.load(out).episodes} == {checked_root.world_id}


def test_dashboard_checks_records_even_when_the_root_name_differs(checked_root: WorldPaths) -> None:
    source = load_world(checked_root.root)
    assert source.world_id == checked_root.world_id
    assert [r.directory for r in source.runs] == [checked_root.observed, checked_root.later]
    assert source.ready
    runner = checked_root.read_record()
    runner["world_id"] = "w78"
    checked_root.record.write_text(json.dumps(runner))
    with pytest.raises(ValueError, match="conflicting world_id") as error:
        load_world(checked_root.root)
    assert_named_records(str(error.value), checked_root.root)
    with pytest.raises(ValueError, match="conflicting world_id"):
        _ = source.ready


def assert_named_records(message: str, root: Path) -> None:
    assert str(root / "files/spec.json") in message
    assert str(root / "runner.json") in message


@pytest.mark.parametrize("record", ["files/spec.json", "runner.json", "both"])
@pytest.mark.parametrize("root_name", ["w77", "arbitrary-root"])
def test_resolver_uses_either_record_without_using_the_root_name(
    tmp_path: Path, record: str, root_name: str
) -> None:
    root = tmp_path / root_name
    for filename in ("files/spec.json", "runner.json"):
        if record in (filename, "both"):
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"world_id": "w77"}')
    assert WorldPaths.from_root(root) == WorldPaths(root, "w77")


@pytest.mark.parametrize("records_exist", [False, True])
def test_a_missing_identity_names_both_records(tmp_path: Path, records_exist: bool) -> None:
    if records_exist:
        (tmp_path / "files").mkdir()
        (tmp_path / "files/spec.json").write_text("{}")
        (tmp_path / "runner.json").write_text("{}")
    with pytest.raises(ValueError, match="no world_id") as error:
        WorldPaths.from_root(tmp_path)
    assert_named_records(str(error.value), tmp_path)
    assert_named_records(world_ready(tmp_path) or "", tmp_path)


@pytest.mark.parametrize("reader", ["certify", "oracle", "live", "teleop", "dashboard"])
def test_readers_refuse_conflicting_identities_with_named_records(
    checked_root: WorldPaths, reader: str
) -> None:
    record = checked_root.read_record()
    record["world_id"] = "w78"
    checked_root.record.write_text(json.dumps(record))
    if reader == "oracle":
        message = world_ready(checked_root.root) or ""
        assert "conflicting world_id" in message
    else:
        with pytest.raises(ValueError, match="conflicting world_id") as error:
            if reader == "certify":
                leaks.main(["certify", str(checked_root.root)])
            elif reader == "live":
                development_runs(checked_root.root)
            elif reader == "teleop":
                require_ready(checked_root.root)
            else:
                load_world(checked_root.root)
        message = str(error.value)
    assert_named_records(message, checked_root.root)


@pytest.mark.parametrize("filename", ["world/spec.json", "spec.json"])
def test_the_oracles_legacy_spec_layout_is_still_read(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"world_id": "w77"}')
    assert WorldPaths.from_root(tmp_path).world_id == "w77"
    assert world_ready(tmp_path) is None


@pytest.mark.parametrize("value", ['"../outside"', '""', "12"])
def test_an_invalid_identity_names_its_record(tmp_path: Path, value: str) -> None:
    path = tmp_path / "runner.json"
    path.write_text('{"world_id": ' + value + "}")
    with pytest.raises(ValueError, match="invalid world_id") as error:
        WorldPaths.from_root(tmp_path)
    assert str(path) in str(error.value)
