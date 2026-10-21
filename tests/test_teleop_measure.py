from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from engine_helpers import fault, ident

from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SnapshotRef
from fleetlab.engine import manifests
from fleetlab.policies.teleop.human import HumanSettings, admit
from fleetlab.policies.teleop.measure import dump, report, require_ready


def table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_an_unready_world_is_refused_before_any_other_file_is_read(tmp_path: Path) -> None:
    dump(tmp_path / "runner.json", {"world_id": "w01", "ready": False})
    with pytest.raises(ValueError, match="world is not read"):
        require_ready(tmp_path)


def synthetic(root: Path) -> None:
    world = root / "world-source"
    jobs = [Job(identity=ident(1)), Job(identity=ident(2, faults=[fault("toy_camera_shift", 1)]))]
    manifests.write(root / "sample.jsonl", jobs)
    settings = HumanSettings()
    dump(
        root / "measurement.json",
        {"world": str(world), "settings": settings.model_dump(mode="json")},
    )
    table(
        world / "runs/as-run/truth/job_outcomes.parquet",
        [
            {
                "job_key": j.key,
                "episode_id": j.identity.episode_id,
                "mode": "run",
                "success": n == 0,
            }
            for n, j in enumerate(jobs)
        ],
    )
    for method in ("timeout", "random"):
        layout = RunLayout(root / method)
        continuations = [
            Job(
                identity=j.identity,
                mode=JobMode.continue_,
                from_snapshot=SnapshotRef(
                    episode_id=j.identity.episode_id, step=25, path=f"{j.key}.25.pkl"
                ),
            )
            for j in jobs
        ]
        admitted = admit(continuations, settings, 14)
        manifests.write(layout.root / "admitted.jsonl", admitted)
        outcomes = []
        metrics = {}
        for n, job in enumerate(admitted):
            counter = Job(
                identity=job.identity, mode=JobMode.replay, from_snapshot=job.from_snapshot
            )
            for key, success in (
                (job.key, n == 1),
                (counter.key, n == 0),
                (job.identity.episode_id, False),
            ):
                outcomes.append({"job_key": key, "success": success})
            metrics[job.key] = {
                "operator_used_s": 16,
                "idle_steps": 60,
                "reaction_steps": 60,
                "fumble_draw": n == 0,
                "fumble_applied": n == 0,
            }
        table(layout.table("job_outcomes"), outcomes)
        dump(layout.root / "human-metrics.json", metrics)


def test_report_counts_rescues_breakages_both_counterfactuals_and_operator_time(
    tmp_path: Path,
) -> None:
    synthetic(tmp_path)
    result = report(tmp_path)
    assert result["no_handover"] == {"episodes": 2, "placed": 1}
    for method in ("timeout", "random"):
        rows = result["methods"][method]
        assert rows["handovers"] == 2 and rows["placed"] == rows["all_placed"] == 1
        assert rows["snapshot_would_succeed"] == rows["as_run_would_succeed"] == 1
        assert rows["rescued_snapshot"] == rows["broken_snapshot"] == 1
        assert rows["snapshot_would_fail"] == rows["as_run_would_fail"] == 1
        assert rows["rescued_as_run"] == 1
        assert rows["operator_reserved_s"] == 180 and rows["operator_used_s"] == 32
        assert rows["reaction_s"] == 6 and rows["fumbles_applied"] == 1
        assert rows["by_fault_type"]["healthy"]["broken_snapshot"] == 1
        assert rows["by_fault_type"]["toy_camera_shift"]["rescued_snapshot"] == 1


def test_report_refuses_a_missing_as_run_denominator(tmp_path: Path) -> None:
    synthetic(tmp_path)
    path = tmp_path / "world-source/runs/as-run/truth/job_outcomes.parquet"
    table(path, [{"episode_id": "e000001", "mode": "run", "success": True}])
    with pytest.raises(ValueError, match="no denominator is silently dropped"):
        report(tmp_path)
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("missing_episode", [False, True])
def test_expert_alone_report_uses_the_same_snapshot_cohort(
    tmp_path: Path, missing_episode: bool
) -> None:
    synthetic(tmp_path)
    for method in ("timeout", "random"):
        source = RunLayout(tmp_path / method)
        layout = RunLayout(tmp_path / "expert-only" / method)
        jobs = [
            job.model_copy(update={"teleoperator_params": None})
            for job in manifests.read(source.root / "admitted.jsonl")
        ]
        if missing_episode:
            jobs = jobs[:1]
        manifests.write(layout.root / "manifest.jsonl", jobs)
        table(layout.table("job_outcomes"), [{"job_key": job.key, "success": True} for job in jobs])
    if missing_episode:
        with pytest.raises(ValueError, match="expert-only denominator differs"):
            report(tmp_path)
    else:
        result = report(tmp_path)
        for method in ("timeout", "random"):
            expert = result["methods"][method]["expert_alone"]
            assert expert["handovers"] == expert["placed"] == 2
            assert expert["rescued_snapshot"] == 1 and expert["broken_snapshot"] == 0
