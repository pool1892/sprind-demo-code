"""Outcome-table regressions for the oracle review of 2026-10-09.

Fixtures are synthetic; the command-status tests mock execution, not simulator proof.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from test_oracle import CAMERA, DELAY, SCORING, SOFTWARE, row, toy_table, toy_world, world_spec

from fleetlab.contracts.actions import Action, Budget
from fleetlab.contracts.oracle import Gain, OracleTable, fault_list_hash
from fleetlab.contracts.rollout import Job, JobMode, RunRecord
from fleetlab.contracts.vocabulary import Verb
from fleetlab.oracle.chance import chance_intervals
from fleetlab.oracle.gains import Gains
from fleetlab.oracle.report import compute_report, pattern_checks, summary_markdown
from fleetlab.oracle.steps import Oracle
from fleetlab.oracle.table import TableBuild, rows_of_run
from fleetlab.oracle.world import OracleLayout, OracleWorld, load_world


def pair_world(outcomes: list[tuple[bool, bool, bool, bool]]) -> tuple[OracleWorld, OracleTable]:
    """Each tuple is success with both faults, A only, B only, neither."""
    identities = [
        SCORING[0].model_copy(update={"episode_id": f"e{n:06d}", "faults": [CAMERA, DELAY]})
        for n in range(1, len(outcomes) + 1)
    ]
    rows = [
        row(i, kept, success)
        for i, values in zip(identities, outcomes, strict=True)
        for kept, success in zip(([CAMERA, DELAY], [CAMERA], [DELAY], []), values, strict=True)
    ]
    world = OracleWorld(
        spec=world_spec(interactions=[("f01", "f02")]), identities=tuple(identities)
    )
    return world, OracleTable(identities, rows)


@pytest.mark.parametrize(
    ("outcome", "term", "verdict"),
    [
        ((False, True, True, True), -28, "confirmed"),
        ((False, False, False, True), 28, "not confirmed"),
    ],
)
def test_interaction_verdict_uses_harm_not_complementary_fixes(
    outcome: tuple[bool, bool, bool, bool], term: int, verdict: str
) -> None:
    world, table = pair_world([outcome] * 28)
    estimate = Gains(table, boot=100, seed=7).interaction("f01", "f02", True)
    assert estimate.net == term
    checks = pattern_checks(world, {"fault_effects": [], "interactions": [estimate], "by_day": []})
    assert next(c for c in checks if c.pattern == "interaction").verdict == verdict


def test_interaction_requires_negative_point_as_well_as_interval() -> None:
    # One very small bootstrap can exclude zero by chance although the measured
    # outcome-table term is zero. Both conditions in the verdict must hold.
    world, table = pair_world([(False, True, True, True), (False, False, False, True)])
    estimate = Gains(table, boot=1, seed=4).interaction("f01", "f02", True)
    assert estimate.net == 0 and estimate.high is not None and estimate.high < 0
    checks = pattern_checks(world, {"fault_effects": [], "interactions": [estimate], "by_day": []})
    assert next(c for c in checks if c.pattern == "interaction").verdict == "not confirmed"


def test_replay_without_any_recorded_identity_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.log import runfiles
    from fleetlab.oracle import table as reader

    original = SCORING[4]
    changed = original.model_copy(update={"noise_seed": original.noise_seed + 1})
    job = Job(identity=original, mode=JobMode.replay, remove=["f01"])
    assert job.key == Job(identity=changed, mode=JobMode.replay, remove=["f01"]).key
    record = RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[], chain_ok=True)
    (tmp_path / "run.json").write_text(record.model_dump_json())
    monkeypatch.setattr(reader, "run_refusal", lambda path: None)
    monkeypatch.setattr(runfiles, "read_identities", lambda path: [])
    monkeypatch.setattr(
        runfiles,
        "read_table",
        lambda *args: pa.Table.from_pylist(
            [
                {
                    "episode_id": original.episode_id,
                    "job_key": job.key,
                    "mode": "replay",
                    "removed": '["f01"]',
                    "added": "[]",
                    "from_snapshot_step": None,
                    "success": True,
                }
            ]
        ),
    )
    for identity in (original, changed):
        rows, info = rows_of_run(tmp_path, {identity.episode_id: identity}, "another world")
        assert not rows
        assert info.problems and "identity" in info.problems[0]


def test_policy_failure_on_healthy_robot_can_still_be_fixed_by_a_partial_removal() -> None:
    world, table = pair_world([(False, False, True, False)])
    report, _ = compute_report(world, TableBuild(table, []), boot=50, draws=0)
    assert Gains(table, boot=0).fault_effect("f01").net == 1
    assert report.still_failing == 1
    text = summary_markdown(report)
    assert "policy itself on a healthy robot" in text
    assert "no catalogue action fixes" not in text


def test_missing_fault_free_outcome_marks_remaining_failures_incomplete() -> None:
    identity = SCORING[0].model_copy(update={"faults": []})
    world = OracleWorld(spec=world_spec(), identities=(identity,))
    report, _ = compute_report(world, TableBuild(OracleTable([identity]), []), boot=50, draws=0)
    assert report.all_faults.net == 0
    assert report.still_failing is None
    assert report.problems
    assert "incomplete" in summary_markdown(report).lower()


def test_missing_draw_is_not_dropped_from_chance_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fleetlab.oracle.chance as chance

    monkeypatch.setattr(chance, "random_draws", lambda *args, **kwargs: [[Gain.of(2, 0), "fresh"]])
    result = chance_intervals(
        toy_table(),
        toy_world().truth(),
        None,
        [Action(verb=Verb.recalibrate, target=CAMERA.target)],
        seed=7,
        achievable={"k = 3": 3},
        draws=2,
        budgets=[Budget(kind="count", amount=3)],
    )[0]
    assert result.draws == 2 and result.valued == 1
    assert result.mean_points is None and result.p5_points is None and result.p95_points is None
    assert result.missing is not None


@pytest.mark.parametrize("command", ["run", "all", "fresh"])
@pytest.mark.parametrize("here", [False, True])
def test_failed_proof_returns_nonzero_before_report_or_valuation(
    command: str, here: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fleetlab.oracle.__main__ import main

    oracle = Oracle(tmp_path, toy_world(), OracleLayout(tmp_path))
    oracle.layout.jobs.write_text("")
    monkeypatch.setattr(Oracle, "open", lambda *args, **kwargs: oracle)
    failed = {"chain_ok": False, "proof": {"compared": 34, "identical": 33}}
    monkeypatch.setattr(Oracle, "step_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(Oracle, "step_run", lambda *args, **kwargs: failed)
    monkeypatch.setattr(Oracle, "step_fresh_run", lambda *args, **kwargs: failed)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a failed proof must stop report/valuation")

    monkeypatch.setattr(Oracle, "step_report", forbidden)
    monkeypatch.setattr(Oracle, "step_fresh_value", forbidden)
    args = [command, str(tmp_path)] + (["--here"] if here else [])
    if command == "fresh":
        args += ["--name", "test", "--run-only"]
    assert main(args) != 0


def test_populated_report_with_bootstrap_disabled_renders_unavailable_intervals() -> None:
    report, _ = compute_report(toy_world(), TableBuild(toy_table(), []), boot=0, draws=0)
    assert all(e.low is None and e.high is None for e in report.fault_effects)
    assert "unavailable" in summary_markdown(report)


def test_run_here_fails_when_new_proof_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.contracts.rollout import RunLayout
    from fleetlab.engine import fleet
    from fleetlab.oracle import proof, run

    directory = tmp_path / "run"
    directory.mkdir()
    job = Job(identity=SCORING[0], mode=JobMode.replay)
    text = job.model_dump_json() + "\n"
    (tmp_path / "jobs.jsonl").write_text(text)
    (directory / "run.json").write_text(
        RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[]).model_dump_json()
    )
    (directory / run.ORIGIN_FILE).write_text(
        json.dumps({"world_hash": None, "manifest_sha256": run.manifest_sha(text), "jobs": 1})
    )
    monkeypatch.setattr(RunLayout, "is_done", lambda *args: True)
    from fleetlab.oracle import table

    monkeypatch.setattr(table, "jobs_merged", lambda *args: True)
    failed = RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[], chain_ok=False)
    monkeypatch.setattr(proof, "prove_run", lambda *args, **kwargs: failed)
    monkeypatch.setattr(fleet, "run", lambda *args: pytest.fail("jobs already completed"))
    with pytest.raises(run.RunRefused, match="proof"):
        run.run_here(
            directory,
            [job],
            [],
            4,
            manifest_file=tmp_path / "jobs.jsonl",
            world_hash=None,
        )


def test_missing_all_removed_row_does_not_report_a_definitive_count() -> None:
    table = toy_table()
    del table._rows[(SCORING[4].episode_id, fault_list_hash([]))]
    report, _ = compute_report(toy_world(), TableBuild(table, []), boot=50, draws=0)
    assert report.still_failing is None
    assert report.all_faults.missing


@pytest.mark.parametrize(
    ("synergy", "verdict"),
    [(0, "not contradicted"), (2, "not contradicted"), (3, "contradicted"), (6, "contradicted")],
)
def test_additive_pair_uses_summed_standalone_harm(synergy: int, verdict: str) -> None:
    from fleetlab.oracle.report import ADDITIVE_RELATIVE_TOLERANCE

    # Separate failures from A and B, then failures only when both act.
    outcomes = (
        [(False, False, True, True)] * 100
        + [(False, True, False, True)] * 100
        + [(False, True, True, True)] * (10 * synergy)
    )
    world, table = pair_world(outcomes)
    world = OracleWorld(
        spec=world_spec(), identities=world.identities, additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(table, []), boot=500, draws=0)
    estimate = report.interactions[0]
    assert estimate.detail["summed_harm_net"] == 200
    assert estimate.detail["standalone_harm_net"] == [100, 100]
    assert ADDITIVE_RELATIVE_TOLERANCE == 0.10
    check = next(c for c in report.patterns if c.pattern == "additive_pair")
    assert check.verdict == verdict
    assert report.marked_additive_pairs == [["f01", "f02"]]
    assert estimate.detail["marked"]


def test_additive_pair_with_interval_holding_zero_is_not_contradicted() -> None:
    world, table = pair_world([(False, True, True, True)] + [(True, True, True, True)] * 27)
    world = OracleWorld(
        spec=world_spec(), identities=world.identities, additive_pairs=(("f01", "f02"),)
    )
    report, _ = compute_report(world, TableBuild(table, []), boot=500, draws=0)
    estimate = report.interactions[0]
    assert estimate.zero_within_noise
    assert estimate.detail["summed_harm_net"] == 0
    assert (
        next(c for c in report.patterns if c.pattern == "additive_pair").verdict
        == "not contradicted"
    )


def test_load_world_accepts_and_checks_new_additive_truth(tmp_path: Path) -> None:
    spec = world_spec().model_dump(mode="json")
    spec["patterns"]["additive_pairs"] = [["f01", "f02"]]
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    (tmp_path / "manifest.jsonl").write_text(
        "".join(Job(identity=i).model_dump_json() + "\n" for i in SCORING)
    )
    world = load_world(tmp_path)
    assert world.additive_pairs == (("f01", "f02"),)
    spec["patterns"]["additive_pairs"] = [["f01", "f99"]]
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=r"patterns name faults not in the schedule.*f99"):
        load_world(tmp_path)


def test_marked_pair_without_joint_episodes_is_reported_missing() -> None:
    world = OracleWorld(
        spec=world_spec(interactions=[("f01", "f02")]),
        identities=tuple(SCORING),
        additive_pairs=(("f01", "f02"),),
    )
    report, _ = compute_report(world, TableBuild(toy_table(), []), boot=50, draws=0)
    checks = [c for c in report.patterns if c.pattern in ("interaction", "additive_pair")]
    assert len(checks) == 2 and all(c.verdict == "missing" for c in checks)


def test_chance_bounds_cover_each_possible_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    import fleetlab.oracle.chance as chance

    baseline = [True, False, False]
    identities = [SCORING[i] for i in range(3)]
    table = OracleTable(
        identities, [row(i, i.faults, s) for i, s in zip(identities, baseline, strict=True)]
    )
    drawn: list[Any] = [Gain.of(1, 0), "fresh", "resolution error"]
    monkeypatch.setattr(chance, "random_draws", lambda *args, **kwargs: [drawn])
    kwargs: dict[str, Any] = dict(
        seed=1, draws=3, achievable={"k = 3": 2}, budgets=[Budget(kind="count", amount=3)]
    )
    entries = [Action(verb=Verb.recalibrate, target=CAMERA.target)]
    bounded = chance_intervals(table, toy_world().truth(), None, entries, **kwargs)[0]
    assert bounded.bounds and bounded.skipped_error == 1 and bounded.skipped_fresh_replay == 1
    for a in range(-1, 3):
        for b in range(-1, 3):
            drawn[:] = [
                Gain.of(1, 0),
                Gain.of(max(a, 0), max(-a, 0)),
                Gain.of(max(b, 0), max(-b, 0)),
            ]
            complete = chance_intervals(table, toy_world().truth(), None, entries, **kwargs)[0]
            assert complete.missing is None
            for name, (low, high) in bounded.bounds.items():
                assert low <= getattr(complete, name) <= high


def test_exposed_random_fix_sets_match_the_original_valuation() -> None:
    import numpy as np

    from fleetlab.contracts.actions import cut
    from fleetlab.oracle.valuation import random_draws, random_fix_sets

    entries = [
        Action(verb=Verb.recalibrate, target=CAMERA.target),
        Action(verb=Verb.revert, target=DELAY.target),
    ]
    budgets = [Budget(kind="count", amount=1)]
    rng = np.random.default_rng(7)
    expected = [cut([entries[int(i)] for i in rng.permutation(2)], budgets[0]) for _ in range(10)]
    actual = list(random_fix_sets(entries, budgets, seed=7, draws=10))
    assert [sets[0] for sets in actual] == expected
    from fleetlab.contracts.oracle import value_fix_set

    results = random_draws(
        toy_table(), toy_world().truth(), None, entries, budgets, seed=7, draws=10
    )
    assert results[0] == [value_fix_set(fs, toy_table(), toy_world().truth()) for fs in expected]


def test_a_run_with_no_origin_cannot_acquire_one_by_reproving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.oracle import proof, run

    directory = tmp_path / "run"
    directory.mkdir()
    record = RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[], chain_ok=True)
    (directory / "run.json").write_text(record.model_dump_json())
    job = Job(identity=SCORING[0], mode=JobMode.replay)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(job.model_dump_json() + "\n")
    monkeypatch.setattr(run, "run_complete", lambda *args: True)
    monkeypatch.setattr(proof, "prove_run", lambda *args: record)
    with pytest.raises(run.RunRefused, match="origin"):
        run.run_here(directory, [job], [], 4, manifest_file=manifest, world_hash=None)


def test_an_interrupted_run_keeps_its_independent_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from fleetlab.engine import fleet
    from fleetlab.oracle import proof, run

    directory, manifest = tmp_path / "run", tmp_path / "jobs.jsonl"
    job = Job(identity=SCORING[0], mode=JobMode.replay)
    record = RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[])
    calls = 0

    def execute(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(record.model_dump_json())
        if calls == 1:
            raise RuntimeError("interrupted before proof")
        return SimpleNamespace(record=record)

    monkeypatch.setattr(fleet, "run", execute)
    monkeypatch.setattr(run, "run_complete", lambda *args: False)
    monkeypatch.setattr(
        proof, "prove_run", lambda *args, **kwargs: record.model_copy(update={"chain_ok": True})
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_here(directory, [job], [], 4, manifest_file=manifest, world_hash=None)
    original = manifest.read_text()
    changed = job.model_copy(
        update={"identity": job.identity.model_copy(update={"noise_seed": 42})}
    )
    assert changed.key == job.key
    manifest.write_text(changed.model_dump_json() + "\n")
    with pytest.raises(run.RunRefused, match="another manifest"):
        run.run_here(directory, [changed], [], 4, manifest_file=manifest, world_hash=None)
    assert calls == 1
    manifest.write_text(original)
    assert run.run_here(directory, [job], [], 4, manifest_file=manifest, world_hash=None)[
        "chain_ok"
    ]
    assert calls == 2


@pytest.mark.parametrize("case", ["cached files without record", "different world hash"])
def test_run_resume_refuses_other_unbound_origins(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.oracle import proof, run

    directory, manifest = tmp_path / "run", tmp_path / "jobs.jsonl"
    directory.mkdir()
    job = Job(identity=SCORING[0], mode=JobMode.replay)
    text = job.model_dump_json() + "\n"
    manifest.write_text(text)
    provenance: dict[str, Any] = {
        "world_hash": "old",
        "manifest": "../jobs.jsonl",
        "manifest_sha256": run.manifest_sha(text),
        "jobs": 1,
    }
    record = RunRecord(
        software=SOFTWARE,
        batch_size=4,
        replay_rules=[],
        chain_ok=True,
        extra={"oracle": provenance},
    )
    if case == "different world hash":
        (directory / "run.json").write_text(record.model_dump_json())
    else:
        (directory / "cached-job.parquet").write_text("unknown origin")
    monkeypatch.setattr(run, "run_complete", lambda *args: True)
    monkeypatch.setattr(proof, "prove_run", lambda *args: record)
    with pytest.raises(run.RunRefused, match=r"origin|world hash"):
        run.run_here(directory, [job], [], 4, manifest_file=manifest, world_hash="new")
