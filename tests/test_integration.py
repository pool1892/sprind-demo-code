"""The miniature fleet's chain without the simulator: the random baseline's
catalogue, the resolver's current values, the fixtures' manifest."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_worlds import SOFTWARE, TEXTS

from fleetlab.contracts.vocabulary import ConfigKey, Verb
from fleetlab.integration import fixtures
from fleetlab.integration.scoring_step import current_values, filled_catalogue
from fleetlab.worlds.generate import World, generate
from fleetlab.worlds.mini import mini_spec


@pytest.fixture(scope="module")
def world() -> World:
    return generate(mini_spec(), SOFTWARE, TEXTS)


def test_filled_catalogue(world: World) -> None:
    entries = filled_catalogue(world.observed_fleet, world.task_classes)
    assert entries and all(a.is_complete for a in entries)
    sets = [a for a in entries if a.verb is Verb.set]
    fleet_values = world.observed_fleet.config.values
    assert all(a.value == fleet_values[a.target.config_key] for a in sets)  # type: ignore[index]
    delay = [a for a in sets if a.target.config_key is ConfigKey.control_action_delay_steps]
    assert {a.value for a in delay} == {0}
    assert len({a.model_dump_json() for a in entries}) == len(entries)


def test_chain_export_uses_full_fleet_and_rechecks_publication(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    from log_toy import write_toy_run

    import fleetlab.log
    from fleetlab.integration.chain import ChainPaths, step_export

    paths = ChainPaths(tmp_path / "chain")
    full = write_toy_run(paths.observed)
    full.robots()[0].traits["chain_private_property"] = 3
    observed = full.model_copy(deep=True)
    for robot in observed.robots():
        robot.traits.clear()
    made = replace(world, fleet=full, observed_fleet=observed)
    real_export = fleetlab.log.export_runs

    def checked_export(*args: object, **kwargs: object) -> object:
        assert kwargs.get("fleet") is observed
        assert kwargs.get("audit_fleet") is full
        return real_export(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fleetlab.log, "export_runs", checked_export)
    assert step_export(paths, made) == paths.export
    target = paths.export / "log/events.parquet"
    table = pq.read_table(target)
    details = table["detail"].to_pylist()
    details[0] = "chain_private_property=3"
    pq.write_table(table.set_column(3, "detail", pa.array(details)), target)
    with pytest.raises(fleetlab.log.ExportRefused, match="chain_private_property"):
        step_export(paths, made)
    assert json.loads(paths.step_file("export").read_text())["leaks_clean"] is False


def test_current_values(world: World) -> None:
    current = current_values(world.fleet)
    by_robot = {j.identity.robot_id: j.identity for j in world.jobs if j.identity.day == 3}
    assert current(by_robot["r05"])[ConfigKey.control_action_delay_steps] == 17
    assert current(by_robot["r01"])[ConfigKey.policy_sampling_steps] == 5
    assert current(by_robot["r05"])[ConfigKey.policy_sampling_steps] == 10


def test_fixture_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = tmp_path / "chain"
    (chain / "world").mkdir(parents=True)
    (chain / "world" / "spec.json").write_text("{}")
    (chain / "runs" / "w10-observed").mkdir(parents=True)
    (chain / "runs" / "w10-observed" / "run.json").write_text("{}")
    (chain / "runs" / "w10-observed" / ".writer.lock").write_text("")
    monkeypatch.setattr(fixtures, "FIXTURES_ROOT", tmp_path / "fixtures-root")
    monkeypatch.setattr(fixtures, "MANIFEST_DIR", tmp_path / "manifests")
    out = fixtures.copy_fixtures(chain, "mini-test", {"world_id": "w10"})
    assert out == tmp_path / "manifests" / "mini-test.json"
    assert fixtures.verify("mini-test") == []
    copied = tmp_path / "fixtures-root" / "mini-test"
    assert not (copied / "runs" / "w10-observed" / ".writer.lock").exists()
    (copied / "world" / "spec.json").write_text('{"changed": 1}')
    assert fixtures.verify("mini-test") == ["differs: world/spec.json"]
    (copied / "world" / "link").symlink_to(copied / "world" / "spec.json")
    with pytest.raises(ValueError, match="symbolic link"):
        fixtures.verify("mini-test")


def test_fixture_manifest_groups_large_directories(tmp_path: Path) -> None:
    root = tmp_path / "f"
    (root / "runs" / "chain").mkdir(parents=True)
    for i in range(fixtures.GROUP_ABOVE + 1):
        (root / "runs" / "chain" / f"e{i:06d}.npz").write_bytes(bytes([i]))
    (root / "runs" / "run.json").write_text("{}")
    manifest = fixtures.manifest_of(root, "x")
    assert set(manifest["files"]) == {"runs/chain/*", "runs/run.json"}
    assert manifest["files"]["runs/chain/*"]["files"] == fixtures.GROUP_ABOVE + 1
    before = manifest["files"]["runs/chain/*"]["sha256"]
    (root / "runs" / "chain" / "e000003.npz").write_bytes(b"changed")
    assert fixtures.manifest_of(root, "x")["files"]["runs/chain/*"]["sha256"] != before


def test_a_run_missing_only_a_newer_run_table_is_complete(tmp_path: Path) -> None:
    """A run log version that adds a run-level table (contracts batch 9's catalogue) must
    not send the chain back to the GPU: the records step writes the table."""
    import json

    from log_toy import EPISODES_PER_DAY, all_jobs, write_toy_run

    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.integration.chain import run_complete
    from fleetlab.log import RunWriter, run_status

    run_dir = tmp_path / "run"
    write_toy_run(run_dir)
    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps({**json.loads(run_json.read_text()), "chain_ok": True}))
    jobs = all_jobs()
    assert run_complete(run_dir, jobs)
    (run_dir / "log" / "catalogue.parquet").unlink()
    assert run_status(run_dir).missing == ("log/catalogue.parquet",)
    assert run_complete(run_dir, jobs)
    record = RunRecord.model_validate_json(run_json.read_text())
    with RunWriter(run_dir, record, episodes_per_robot_day=EPISODES_PER_DAY) as writer:
        writer.finalize()
    assert not run_status(run_dir).missing
    (run_dir / "truth" / "chain" / f"{jobs[0].key}.npz").unlink()
    assert not run_complete(run_dir, jobs)
