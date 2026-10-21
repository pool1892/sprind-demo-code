"""Measured cause classes, adaptive plans and resumable CPU execution."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from fleetlab.contracts.oracle import OracleRow, OracleTable, fault_list_hash
from fleetlab.contracts.rollout import Job, JobMode, RunRecord
from fleetlab.contracts.vocabulary import FaultClass
from fleetlab.contracts.world import Patterns, seal_world
from fleetlab.engine import software
from fleetlab.oracle import toy
from fleetlab.oracle.causes import (
    curve_observed_estimate,
    measure_causes,
    observed_requests,
)
from fleetlab.oracle.run import RunRefused
from fleetlab.oracle.world import OracleWorld


def fixture_world(*, interaction: bool = False, additive: bool = False) -> OracleWorld:
    jobs = toy.jobs(software.current("none"))
    faults = [
        f.model_copy(
            update={
                "target": f.target.model_copy(update={"scope": "r01"}),
                "schedule": f.schedule.model_copy(update={"onset_day": 0}),
            }
        )
        for f in toy.faults()
    ]
    identities = tuple(
        jobs[0].identity.model_copy(
            update={
                "episode_id": f"e{day * 100 + number + 1:06}",
                "day": day,
                "faults": faults,
            }
        )
        for day in (0, 2)
        for number in range(100)
    )
    table = toy.fault_table(jobs)
    truths = [
        f.model_copy(
            update={
                "identifiable": True,
                "episodes": {"observed": 100, "scoring": 100, "experiment": 0},
                "expected_loss_points": {"scoring": 3.0, "observed": 3.0, "experiment": 0.0},
                "detectable": f.fault_id == "f02",
            }
        )
        for f in table.faults
    ]
    return OracleWorld(
        spec=toy.spec().model_copy(
            update={
                "fault_schedule": faults,
                "patterns": Patterns(
                    decoys=["f03"], interactions=[("f01", "f02")] if interaction else []
                ),
            }
        ),
        identities=identities,
        fault_table=table.model_copy(update={"faults": truths}),
        additive_pairs=(("f01", "f02"),) if additive else (),
        registry=toy.REGISTRY,
    )


def table_of(
    world: OracleWorld,
    gains: Callable[[str, frozenset[str]], int],
    *,
    observed: bool = True,
) -> OracleTable:
    rows = []
    for identity in world.identities:
        ids = sorted(identity.fault_ids)
        window = "observed" if identity.day == 0 else "scoring"
        number = int(identity.episode_id[1:]) % 100
        for k in range(len(ids) + 1):
            for subset in itertools.combinations(ids, k):
                removed = frozenset(subset)
                if window == "observed" and not observed and removed:
                    continue
                remaining = [f for f in identity.faults if f.fault_id not in removed]
                success = bool(removed) and number < gains(window, removed)
                rows.append(
                    OracleRow(
                        episode_id=identity.episode_id,
                        active_faults=fault_list_hash(remaining),
                        success=success,
                    )
                )
    return OracleTable(world.identities, rows)


def test_measurements_replace_curves_and_use_the_whole_window() -> None:
    world = fixture_world()

    def gain(window: str, removed: frozenset[str]) -> int:
        return {"f01": 3, "f02": 1 if window == "scoring" else 3, "f03": 3}.get(
            next(iter(removed)), 0
        )

    measured = measure_causes(world, table_of(world, gain))
    by_id = {f.item: f for f in measured.faults}
    assert measured.complete and measured.cause_count == 3
    assert by_id["f01"].curve_class == "small" and by_id["f01"].measured_class == "cause"
    assert by_id["f02"].qualified_by == ["observed"]
    assert by_id["f02"].windows["scoring"].points == 1
    assert by_id["f02"].windows["observed"].points == 3
    assert by_id["f03"].curve_class == "decoy" and by_id["f03"].measured_class == "cause"
    assert measured.no_decoy_left


def sparse_world(*, interaction: bool = False) -> OracleWorld:
    """100 attempts/window; faults have different, overlapping carrier sets."""
    world = fixture_world(interaction=interaction)
    identities = tuple(
        i.model_copy(
            update={
                "faults": [
                    f
                    for f in i.faults
                    if int(i.episode_id[1:]) % 100 < {"f01": 20, "f02": 40, "f03": 60}[f.fault_id]
                ]
            }
        )
        for i in world.identities
    )
    assert world.fault_table
    faults = [
        f.model_copy(
            update={"episodes": {"observed": {"f01": 20, "f02": 40, "f03": 60}[f.fault_id]}}
        )
        for f in world.fault_table.faults
    ]
    return replace(
        world,
        identities=identities,
        fault_table=world.fault_table.model_copy(update={"faults": faults}),
    )


def test_sparse_carriers_two_points_and_twenty_carriers_qualify() -> None:
    world = sparse_world()
    measured = measure_causes(world, table_of(world, lambda window, removed: 2))
    first = measured.faults[0]
    assert first.observed_carriers == 20
    assert first.windows["scoring"].carrying == 20
    assert first.windows["scoring"].episodes == 100
    assert first.windows["scoring"].points == 2
    assert first.measured_class == "cause" and first.qualified_by == ["scoring"]


def test_removal_subtracts_a_broken_attempt_over_the_whole_window() -> None:
    world = sparse_world()
    rows = []
    for i in world.identities:
        number = int(i.episode_id[1:]) % 100
        for count in range(len(i.faults) + 1):
            for removed in itertools.combinations(sorted(i.fault_ids), count):
                remaining = [f for f in i.faults if f.fault_id not in removed]
                # Three rescues, but the fourth carrying attempt is broken by removal.
                success = number == 4 if "f01" not in removed else number in (1, 2, 3)
                rows.append(
                    OracleRow(
                        episode_id=i.episode_id,
                        active_faults=fault_list_hash(remaining),
                        success=success,
                    )
                )
    measured = measure_causes(world, OracleTable(world.identities, rows))
    m = measured.faults[0].windows["scoring"]
    assert (m.rescued, m.broken, m.net, m.points) == (3, 1, 2, 2)


def test_pending_items_never_enter_the_measured_list() -> None:
    world = sparse_world()
    measured = measure_causes(world, table_of(world, lambda window, removed: 1, observed=False))
    assert not measured.complete
    assert measured.faults[0].measured_class is None
    assert measured.causes == [] and measured.cause_count == 0


def test_pair_carriers_require_both_members() -> None:
    world = sparse_world(interaction=True)
    # Only 19 observed attempts carry both; 21 others carry f02 alone.
    world = replace(
        world,
        identities=tuple(
            i.model_copy(
                update={
                    "faults": [
                        f
                        for f in i.faults
                        if not (
                            i.day == 0 and int(i.episode_id[1:]) % 100 == 19 and f.fault_id == "f01"
                        )
                    ]
                }
            )
            for i in world.identities
        ),
    )
    measured = measure_causes(
        world, table_of(world, lambda window, removed: 2 if {"f01", "f02"} <= removed else 1)
    )
    pair = measured.pairs[0]
    assert pair.windows["scoring"].points == 2
    assert pair.observed_carriers == 19 and pair.measured_class == "small"


@pytest.mark.parametrize("eligibility", [{"identifiable": False}, {"episodes": {"observed": 19}}])
@pytest.mark.parametrize("points", [2, 5])
def test_large_ineligible_decoy_is_on_neither_scoring_list(
    eligibility: dict[str, object], points: int
) -> None:
    world = sparse_world()
    assert world.fault_table
    truths = [
        f.model_copy(update=eligibility) if f.fault_id == "f03" else f
        for f in world.fault_table.faults
    ]
    world = replace(world, fault_table=world.fault_table.model_copy(update={"faults": truths}))
    measured = measure_causes(
        world, table_of(world, lambda window, removed: points if removed == {"f03"} else 0)
    )
    decoy = measured.faults[2]
    assert decoy.windows["scoring"].points == points
    assert decoy.measured_class == "small" and not decoy.qualified_by
    assert not measured.causes and measured.no_decoy_left


def test_eligibility_comes_from_the_fault_table() -> None:
    world = fixture_world()
    assert world.fault_table is not None
    faults = [
        f.model_copy(update={"identifiable": False})
        if f.fault_id == "f01"
        else f.model_copy(update={"episodes": {"observed": 19, "scoring": 100}})
        if f.fault_id == "f02"
        else f
        for f in world.fault_table.faults
    ]
    world = replace(world, fault_table=world.fault_table.model_copy(update={"faults": faults}))
    measured = measure_causes(world, table_of(world, lambda window, removed: 3))
    assert [f.item for f in measured.faults if f.measured_class == "cause"] == ["f03"]


@pytest.mark.parametrize("observed_joint", [False, True])
def test_interaction_only_after_both_members_and_each_adaptive_stage(observed_joint: bool) -> None:
    world = fixture_world(interaction=True)

    def gain(window: str, removed: frozenset[str]) -> int:
        return (
            3 if {"f01", "f02"} <= removed and (window == "observed" or not observed_joint) else 1
        )

    table = table_of(world, gain, observed=False)
    first = measure_causes(world, table)
    requests = observed_requests(world, first, table)
    assert len(requests) == 200  # two removals over 100 observed attempts
    assert all(len(r.faults) == 2 for r in requests)  # no joint jobs before singles decided
    full = table_of(world, gain)
    for r in requests:
        success = full.success(r.episode_id, r.faults)
        assert success is not None
        table.add(
            OracleRow(
                episode_id=r.episode_id, active_faults=fault_list_hash(r.faults), success=success
            )
        )
    second = measure_causes(world, table)
    pair = second.pairs[0]
    if not observed_joint:
        assert pair.qualified_by == ["scoring"]
        assert not observed_requests(world, second, table)
    else:
        joint_requests = observed_requests(world, second, table)
        assert len(joint_requests) == 100 and all(len(r.faults) == 1 for r in joint_requests)
        for r in joint_requests:
            success = full.success(r.episode_id, r.faults)
            assert success is not None
            table.add(
                OracleRow(
                    episode_id=r.episode_id,
                    active_faults=fault_list_hash(r.faults),
                    success=success,
                )
            )
        final = measure_causes(world, table)
        assert final.pairs[0].qualified_by == ["observed"]
        assert final.pairs[0].windows["observed"].points == 3
        assert final.complete


def test_additive_pairs_and_standalone_causes_do_not_get_joint_treatment() -> None:
    for world in (fixture_world(additive=True), fixture_world(interaction=True)):
        table = table_of(world, lambda window, removed: 3)
        measured = measure_causes(world, table)
        assert measured.pairs[0].measured_class == "small"
        assert measured.pairs[0].windows["scoring"].status == "not_applicable"
        assert measured.pairs[0].windows["scoring"].reason == (
            "additive pairs have no joint cause treatment"
            if world.additive_pairs
            else "standalone member qualifies"
        )
        assert not observed_requests(world, measured, table)


def test_data_fault_counts_misreports_in_both_directions_without_replays() -> None:
    world = fixture_world()
    assert world.fault_table is not None
    truths = [
        f.model_copy(update={"fault_class": FaultClass.data}) if f.fault_id == "f01" else f
        for f in world.fault_table.faults
    ]
    world = replace(world, fault_table=world.fault_table.model_copy(update={"faults": truths}))
    recorded = {
        i.episode_id: (int(i.episode_id[1:]) % 100 < 2, int(i.episode_id[1:]) % 100 == 2)
        for i in world.identities
    }
    table = table_of(world, lambda window, removed: 0)
    measured = measure_causes(world, table, recorded)
    data = measured.faults[0]
    assert data.measured_class == "cause" and data.qualified_by == ["scoring", "observed"]
    assert data.windows["scoring"].points == 3 and data.windows["scoring"].net == 1
    assert data.windows["scoring"].metric == "misreported_outcomes"
    assert all(
        "f01" in {f.fault_id for f in r.faults} for r in observed_requests(world, measured, table)
    )


def test_missing_scoring_outcomes_prevent_observed_planning() -> None:
    world = fixture_world()
    table = OracleTable(world.identities)
    with pytest.raises(RunRefused, match="scoring measurements required"):
        observed_requests(world, measure_causes(world, table), table)
    estimate = curve_observed_estimate(world)
    assert estimate["removal_replays"] == 0
    assert "curve estimates only" in str(estimate["basis"])


def test_decoys_and_inert_entries_keep_their_class_below_threshold() -> None:
    world = fixture_world()
    assert world.fault_table is not None
    faults = [
        f.model_copy(update={"role": "inert"}) if f.fault_id == "f02" else f
        for f in world.fault_table.faults
    ]
    world = replace(world, fault_table=world.fault_table.model_copy(update={"faults": faults}))
    measured = measure_causes(world, table_of(world, lambda window, removed: 1))
    assert [f.measured_class for f in measured.faults] == ["small", "inert", "decoy"]
    assert not measured.no_decoy_left
    assert measured.faults[1].windows["observed"].status == "not_needed"


@pytest.mark.parametrize("source_samples", [False, True])
def test_observed_cpu_run_adds_all_seconds_and_resumes_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_samples: bool,
) -> None:
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.log import RunWriter
    from fleetlab.oracle import steps
    from fleetlab.oracle.steps import Oracle

    root = tmp_path / "w99"
    original = toy.write_world(root, software.current("none"))
    # A planted effective delay has no realised effect at this setting.
    manifest = [
        Job(
            identity=j.identity.model_copy(
                update={
                    "faults": [
                        f.model_copy(update={"params": {"steps": 0}}) if f.fault_id == "f02" else f
                        for f in j.identity.faults
                    ]
                }
            )
        )
        for j in original
    ]
    (root / "world/manifest.jsonl").write_text(
        "".join(j.model_dump_json() + "\n" for j in manifest)
    )
    seal_world(root / "world")
    source = root / "runs/toy"
    run(
        RunConfig(
            out=str(source),
            batch_size=4,
            chain_sample=source_samples,
            episodes_per_robot_day=4,
            **toy.engine_overrides(),
        ),
        manifest,
    )
    with RunWriter(
        source,
        RunRecord.model_validate_json((source / "run.json").read_text()),
        episodes_per_robot_day=4,
    ) as writer:
        writer.update_record(chain_ok=True)
    oracle = Oracle.open(root, registry=toy.REGISTRY)
    with pytest.raises(RunRefused, match="scoring measurements required"):
        oracle.step_causes_plan()
    oracle.step_plan()
    from fleetlab.oracle.run import run_here

    run_here(
        oracle.layout.run,
        oracle.stored_manifest(),
        oracle.world_runs(),
        oracle.world.spec.episodes_per_robot_day,
        manifest_file=oracle.layout.jobs,
        world_hash=oracle.world.world_hash,
        overrides=toy.engine_overrides(),
    )
    # This is a pre-round oracle run: scoring rows/proofs exist, no coverage ledger.
    assert not (oracle.layout.root / "verification/comparisons.json").exists()
    before = {p: p.read_bytes() for p in root.rglob("*.npz")}
    with pytest.raises(RunRefused, match="double execution"):
        oracle.step_causes_plan()
    # Simulate a first pass interrupted after finalize but before its old proof.
    record_path = oracle.layout.run / "run.json"
    record = RunRecord.model_validate_json(record_path.read_text())
    record = record.model_copy(update={"chain_ok": None})
    record_path.write_text(record.model_dump_json())
    from fleetlab.engine import fleet

    with monkeypatch.context() as guard:
        guard.setattr(fleet, "run", lambda *args: pytest.fail("all scoring executions exist"))
        verification = oracle.step_verify(here=True, overrides=toy.engine_overrides())
    assert verification["jobs_run_now"] == 0
    assert verification["coverage"][0]["attempts_executed_twice"] == 12
    plan = oracle.step_causes_plan()
    assert plan["removal_replays_remaining"] == 4
    assert plan["second_executions_remaining"] == (0 if source_samples else 4)
    assert plan["observed_episodes"] == [
        j.identity.episode_id for j in manifest[:12] if "f02" in j.identity.fault_ids
    ]
    result = oracle.step_causes_run(here=True, overrides=toy.engine_overrides())
    assert result["jobs_run_now"] == (4 if source_samples else 8)
    coverage = {c["window"]: c for c in result["verification"]["coverage"]}
    assert coverage["observed"]["attempts_executed_twice"] == 4
    assert coverage["scoring"]["attempts_executed_twice"] == 12
    assert "experiment" not in coverage
    assert all(p.read_bytes() == content for p, content in before.items())
    assert oracle.step_causes_plan()["jobs_remaining"] == 0
    monkeypatch.setattr(steps, "run_heavy", lambda *args: pytest.fail("all executions exist"))
    assert oracle.step_causes_run()["jobs_run_now"] == 0
    measured = json.loads((oracle.layout.root / "measured_causes.json").read_text())
    assert measured["complete"] and measured["faults"][1]["windows"]["observed"]["net"] == 0
    report = oracle.step_report(boot=0, draws=0)
    assert report.measured_causes == measured
    assert report.replay_verification["residual_by_window"]["observed"]["changed_replays"] == 4
    assert "Measured and curve-based causes" in oracle.layout.summary.read_text()


def test_coverage_includes_removals_already_found_in_other_runs() -> None:
    from fleetlab.oracle.causes import required_observed_episodes

    world = fixture_world()
    table = table_of(world, lambda window, removed: 1)
    measured = measure_causes(world, table)
    assert not observed_requests(world, measured, table)
    assert len(required_observed_episodes(world, measured)) == 100


def test_complete_truth_is_preserved_and_wrong_world_manifest_is_refused(tmp_path: Path) -> None:
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.world import OracleLayout

    world = fixture_world()
    oracle = Oracle(tmp_path, world, OracleLayout(tmp_path / "oracle", world.world_id))
    full = measure_causes(world, table_of(world, lambda window, removed: 3))
    oracle.write_measured_causes(full)
    path = oracle.layout.root / "measured_causes.json"
    original = path.read_bytes()
    partial = measure_causes(world, table_of(world, lambda window, removed: 1, observed=False))
    with pytest.raises(RunRefused, match="complete measured causes preserved"):
        oracle.write_measured_causes(partial)
    assert path.read_bytes() == original
    wrong = world.window("observed")[0].model_copy(update={"noise_seed": 999})
    with pytest.raises(RunRefused, match="observed manifest"):
        oracle.check_cause_manifest([Job(identity=wrong, mode=JobMode.replay)])


def test_data_fault_with_two_active_data_hooks_is_explicitly_unattributed() -> None:
    world = fixture_world()
    assert world.fault_table
    truths = [
        f.model_copy(update={"fault_class": FaultClass.data}) if f.fault_id != "f03" else f
        for f in world.fault_table.faults
    ]
    world = replace(world, fault_table=world.fault_table.model_copy(update={"faults": truths}))
    measured = measure_causes(
        world,
        table_of(world, lambda window, removed: 0),
        {i.episode_id: (True, False) for i in world.identities},
    )
    assert not measured.complete
    assert measured.faults[0].windows["scoring"].points is None
    assert "cannot be attributed" in str(measured.faults[0].windows["scoring"].reason)


def test_data_decoy_cannot_qualify_by_observed_misreports() -> None:
    world = sparse_world()
    assert world.fault_table
    truths = [
        f.model_copy(update={"fault_class": FaultClass.data}) if f.fault_id == "f03" else f
        for f in world.fault_table.faults
    ]
    world = replace(world, fault_table=world.fault_table.model_copy(update={"faults": truths}))
    measured = measure_causes(
        world,
        table_of(world, lambda window, removed: 0),
        {
            i.episode_id: (False, i.day == 0 and int(i.episode_id[1:]) % 100 < 5)
            for i in world.identities
        },
    )
    decoy = measured.faults[2]
    assert decoy.windows["observed"].points == 5
    assert decoy.windows["scoring"].points == 0
    assert decoy.measured_class == "decoy" and not decoy.qualified_by


def test_logged_outcomes_use_the_actual_run_even_if_a_replay_is_listed_first(
    tmp_path: Path,
) -> None:
    from test_oracle_verification import execute

    from fleetlab.oracle.causes import logged_outcomes
    from fleetlab.oracle.table import RunRows, TableBuild

    job = toy.jobs(software.current("none"))[24]
    left = execute(tmp_path / "original", job, flip=True)
    right = execute(tmp_path / "scratch", job.model_copy(update={"mode": JobMode.replay}))
    build = TableBuild(
        OracleTable([job.identity]),
        [RunRows(right.run, executions=[right]), RunRows(left.run, executions=[left])],
    )
    assert not left.success and right.success
    assert logged_outcomes(build) == {job.identity.episode_id: (False, False)}


def test_measured_truth_is_incomplete_until_baselines_are_verified(tmp_path: Path) -> None:
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.table import TableBuild
    from fleetlab.oracle.world import OracleLayout

    world = fixture_world()
    oracle = Oracle(tmp_path, world, OracleLayout(tmp_path / "oracle", world.world_id))
    build = TableBuild(table_of(world, lambda window, removed: 3), [])
    unverified = oracle.measured_causes(build)
    assert not unverified.complete
    assert unverified.problems == ["scoring: 100 baselines not executed twice"]
    build.verification = {"verified_episodes": [i.episode_id for i in world.scoring()]}
    assert oracle.measured_causes(build).complete
    # When observed removal values were needed, their baselines also need checks.
    build.table = table_of(world, lambda window, removed: 1)
    measured = oracle.measured_causes(build)
    assert not measured.complete
    assert measured.problems == ["observed: 100 baselines not executed twice"]


def test_stage_with_reused_rows_finishes_baseline_checks_and_publishes_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.oracle.__main__ import main
    from fleetlab.oracle.plan import manifest_text
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.table import TableBuild
    from fleetlab.oracle.world import OracleLayout

    world = fixture_world()
    oracle = Oracle(tmp_path, world, OracleLayout(tmp_path / "oracle", world.world_id))
    build = TableBuild(
        table_of(world, lambda window, removed: 1),
        [],
        verification={"verified_episodes": [i.episode_id for i in world.scoring()]},
    )
    monkeypatch.setattr(Oracle, "table", lambda *args, **kwargs: build)
    assert not oracle.measured_causes().complete
    folder = oracle.layout.root / "causes"
    folder.mkdir(parents=True)
    (folder / "jobs.jsonl").write_text(manifest_text([]))
    (folder / "plan.json").write_text(
        json.dumps({"world_hash": world.world_hash, "observed_episodes": []})
    )

    def finish_verification(*args: object, **kwargs: object) -> dict[str, object]:
        build.verification["verified_episodes"] = [i.episode_id for i in world.identities]
        return {"jobs_run_now": 0}

    monkeypatch.setattr(Oracle, "step_verify", finish_verification)
    monkeypatch.setattr(Oracle, "open", lambda *args, **kwargs: oracle)
    monkeypatch.setattr(
        Oracle,
        "step_causes_plan",
        lambda *args, **kwargs: {
            "jobs_remaining": 0,
            "measurement_complete": False,
        },
    )
    assert main(["causes", str(tmp_path), "--here"]) == 0
    truth = json.loads((oracle.layout.root / "measured_causes.json").read_text())
    assert truth["complete"]
