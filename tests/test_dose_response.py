"""The dose-response calibration script on a toy engine (fleetlab.dose_response).

The manifests pair every faulted episode with a healthy twin (same task,
start state and seeds), the runs are read through the run layout, and the
statistics reproduce the numbers spikes/xvla-speed section 8 published.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fault_toys import POLICY, SOFTWARE

from fleetlab import dose_response as dr
from fleetlab.contracts.rollout import Job, RunLayout


def test_wilson_mcnemar_and_newcombe_match_the_spikes_numbers() -> None:
    # spikes/xvla-speed section 8, case 3: 200 pairs, 145 (32-bit) against 142 (16-bit),
    # 140 both, 5 only 32-bit, 2 only 16-bit, 53 neither: p 0.45, -1.5 (-4.4 to +1.3) points.
    theta, low, high = dr.newcombe_paired(140, 2, 5, 53)
    assert round(100 * theta, 1) == -1.5
    assert round(100 * low, 1) == -4.4 and round(100 * high, 1) == 1.3
    assert round(dr.mcnemar_exact(5, 2), 2) == 0.45
    # case 2: 400 pairs, 17 against 17 discordant: p 1.00, 0.0 (-2.9 to +2.9).
    theta, low, high = dr.newcombe_paired(204, 17, 17, 162)
    assert theta == 0.0 and round(100 * low, 1) == -2.9 and round(100 * high, 1) == 2.9
    assert dr.mcnemar_exact(17, 17) == 1.0
    lo, hi = dr.wilson(179, 200)
    assert 0.84 < lo < 0.85 and 0.93 < hi < 0.94


def test_a_plan_pairs_every_faulted_episode_with_a_healthy_twin() -> None:
    healthy = dr.plan_jobs(None, 30, POLICY, SOFTWARE)
    faulted = dr.plan_jobs("camera_pose_offset", 30, POLICY, SOFTWARE)
    points = dr.calibration_points("camera_pose_offset")
    assert len(healthy) == 30 and len(faulted) == 30 * len(points)
    assert len({j.key for j in healthy + faulted}) == len(healthy) + len(faulted)
    twins = {dr.split_episode_id(j.identity.episode_id)[1]: j.identity for j in healthy}
    for job in faulted:
        block, index = dr.split_episode_id(job.identity.episode_id)
        assert block == dr.block_of("camera_pose_offset", (block - 1) % dr.POINTS_PER_KIND)
        twin = twins[index]
        assert job.identity.model_dump(exclude={"episode_id", "faults"}) == twin.model_dump(
            exclude={"episode_id", "faults"}
        )
        assert [f.fault_id for f in job.identity.faults] == ["f01"]
    tasks = [j.identity.task_index for j in healthy]
    assert sorted(set(tasks)) == list(range(10)) and tasks.count(0) == 3
    with pytest.raises(ValueError):
        dr.episodes(25)


@pytest.mark.parametrize("kind", dr.CALIBRATED)
def test_every_calibrated_kind_has_a_plan(kind: str) -> None:
    jobs = dr.plan_jobs(kind, 10, POLICY, SOFTWARE)
    assert len(jobs) == 10 * len(dr.calibration_points(kind))
    blocks = {dr.split_episode_id(j.identity.episode_id)[0] for j in jobs}
    assert len(blocks) == len(dr.calibration_points(kind))
    # Blocks of different kinds never collide.
    assert all(dr.CALIBRATED.index(kind) * dr.POINTS_PER_KIND < b for b in blocks)


def toy_success(job: Job) -> tuple[bool, bool]:
    """A toy engine's (true, logged) outcome: healthy 95%; a delay over 15 steps 45%."""
    index = job.identity.episode_index
    u = int.from_bytes(hashlib.sha256(f"toy{index}".encode()).digest()[:8], "big") / 2**64
    level = job.identity.faults[0].level if job.identity.faults else 0.0
    kind = job.identity.faults[0].kind if job.identity.faults else ""
    rate = 0.95
    if kind == "action_latency" and level > 15:
        rate = 0.45
    success = u < rate
    return success, success


def toy_run(jobs: list[Job], run_dir: Path, chain_ok: bool = True) -> None:
    layout = RunLayout(run_dir)
    rows, verdicts = [], []
    for job in jobs:
        true, logged = toy_success(job)
        rows.append(
            {
                "episode_id": job.identity.episode_id,
                "outcome": "success" if logged else "failure",
                "steps": 110 if true else 280,
            }
        )
        verdicts.append(
            {
                "episode_id": job.identity.episode_id,
                "true_success": true,
                "true_reason": "",
                "libero_first_success_step": 100 if true else None,
            }
        )
    for name, table in (("episodes", rows), ("verdicts", verdicts)):
        path = layout.table(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(table), path)
    layout.run_record.write_text(json.dumps({"chain_ok": chain_ok}))


def test_a_toy_calibration_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    healthy_jobs = dr.plan_jobs(None, 100, POLICY, SOFTWARE)
    delay_jobs = dr.plan_jobs("action_latency", 100, POLICY, SOFTWARE)
    toy_run(healthy_jobs, tmp_path / "cal-healthy")
    toy_run(delay_jobs, tmp_path / "cal-delay")
    healthy = dr.read_run(tmp_path / "cal-healthy")
    run = dr.read_run(tmp_path / "cal-delay")
    assert healthy.chain_ok is True and len(healthy.outcomes) == 100
    results = dr.analyse("action_latency", healthy, run)
    by_label = {r.label: r for r in results}
    assert by_label["14 steps"].successes == by_label["14 steps"].n - by_label[
        "14 steps"
    ].broken - (by_label["14 steps"].n - dr.healthy_summary(healthy)["successes"])
    assert by_label["14 steps"].p_value == 1.0 and by_label["14 steps"].broken == 0
    assert by_label["17 steps"].p_value < 1e-6 and by_label["17 steps"].rescued == 0
    assert dr.verdict(results) == "kept"
    # 100 flat pairs cannot exclude a loss of 2 points; about 300 can.
    assert dr.verdict(results[:2]).startswith("flat, not yet a decoy: a loss of up to 3.")
    narrow = [
        r.__class__(**{**r.__dict__, "difference": (0.0, -0.013, 0.013)}) for r in results[:2]
    ]
    assert dr.verdict(narrow).startswith("decoy")
    assert all(dr.decoy_level(r) for r in narrow)
    one = [narrow[0], results[1]]
    assert dr.verdict(one).startswith(f"decoy at {narrow[0].label};")
    assert not any(dr.decoy_level(r) for r in results[:2])
    assert by_label["17 steps"].run_id == "cal-delay"

    code = dr.main(
        [
            "analyse",
            "--healthy",
            str(tmp_path / "cal-healthy"),
            f"action_latency={tmp_path / 'cal-delay'}",
            "--json",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "### action_latency: kept" in printed and "| 17 steps | 100 |" in printed
    assert by_label["17 steps"].healthy_successes == dr.healthy_summary(healthy)["successes"]
    dump = json.loads((tmp_path / "out.json").read_text())
    # A type split over two runs reads as one.
    toy_run(dr.plan_jobs("action_latency", 100, POLICY, SOFTWARE, points=[0]), tmp_path / "a")
    toy_run(dr.plan_jobs("action_latency", 100, POLICY, SOFTWARE, points=[2]), tmp_path / "b")
    merged = dr.merge_runs([dr.read_run(tmp_path / "a"), dr.read_run(tmp_path / "b")])
    labels = [r.label for r in dr.analyse("action_latency", healthy, merged)]
    assert labels == ["8 steps", "17 steps"] and merged.run_id == "a + b"
    with pytest.raises(ValueError, match="share episodes"):
        dr.merge_runs([dr.read_run(tmp_path / "a"), dr.read_run(tmp_path / "a")])
    assert dump["types"]["action_latency"]["verdict"] == "kept"


def test_the_measured_curves_match_the_registry() -> None:
    import fleetlab.faults as faults
    from fleetlab.contracts.vocabulary import FaultClass
    from fleetlab.faults.records import measured

    for kind in dr.CALIBRATED:
        try:
            entry = measured(kind)
        except KeyError:
            assert dr.CALIBRATED.index(kind) >= 12, f"stage 2's {kind} lost its curve"
            continue  # a stage-3 kind before its calibration has run
        assert entry["verdict"] in {"kept", "decoy", "inert", "dropped"}
        labels = [p.label for p in dr.calibration_points(kind)]
        have = [p["label"] for p in entry["points"]]
        assert have == labels[: len(have)], kind  # points are appended, never inserted
        for point in entry["points"]:
            assert point["n"] >= 100
            assert point["successes"] <= point["n"] and point["libero_successes"] <= point["n"]
            d, lo, hi = point["difference"]
            assert lo <= d <= hi
        data = faults.get(kind).fault_class is FaultClass.data
        effects = [
            p for p in entry["points"] if p["p_value"] < 0.05 and (data or p["difference"][0] < 0)
        ]
        assert {p["label"] for p in effects} == {
            p["label"] for p in entry["points"] if p["role"] == "effect"
        }, kind
        if entry["verdict"] == "kept":
            assert effects
        elif entry["verdict"] == "decoy":  # at its confirmed levels; the others small causes
            assert not effects
            decoys = [p for p in entry["points"] if p["role"] == "decoy"]
            assert decoys and all(p["difference"][1] >= -dr.DECOY_MARGIN for p in decoys)
            assert all(p["n"] >= 1000 for p in decoys)
            assert {p["role"] for p in entry["points"]} <= {"decoy", "small_cause"}
        elif entry["verdict"] == "inert":
            assert all(p["broken"] == p["rescued"] == 0 for p in entry["points"])
    with pytest.raises(KeyError):
        measured("config_change")
