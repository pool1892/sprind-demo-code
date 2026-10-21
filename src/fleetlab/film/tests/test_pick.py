from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleetlab.film import pick


def _batch(root: Path, rows: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True)
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _row(variant: str, start: int, delivered: bool, **change: object) -> dict[str, object]:
    row: dict[str, object] = {
        "job": "J1",
        "variant": variant,
        "start": start,
        "base_task": 6,
        "init_state": start,
        "scene_seed": 1000 + start,
        "noise_seed": 2000 + start,
        "delivered": delivered,
    }
    row.update(change)
    return row


def test_pairs_by_start_in_spike_batches(tmp_path: Path) -> None:
    first = _batch(tmp_path / "a", [_row("naive", 0, True), _row("naive", 1, False)])
    second = _batch(
        tmp_path / "b",
        [_row("naive", 2, True), _row("release", 0, False), _row("release", 1, True),
         _row("release", 2, True), _row("release", 3, True)],
    )  # fmt: skip
    before = (
        pick.load_side(first, {"variant": "naive"})[0]
        + pick.load_side(second, {"variant": "naive"})[0]
    )
    after, files = pick.load_side(second, {"variant": "release"})
    report = pick.pair(before, after, files).summary()
    assert report["matched_on"] == ["base_task", "init_state", "scene_seed", "noise_seed"]
    assert (report["both_delivered"], report["rescued"], report["broken"]) == (1, 1, 1)
    assert report["only_after"] == 1
    assert report["first_rescued"][0]["episode"] == "J1/naive/s01"
    assert report["first_broken"][1]["episode"] == "J1/release/s00"
    with pytest.raises(ValueError, match="two episodes on the start"):
        pick.pair(before + before, after, files)


def test_other_seeds_are_other_starts(tmp_path: Path) -> None:
    before, files = pick.load_side(_batch(tmp_path / "a", [_row("naive", 0, False)]), {})
    after, _ = pick.load_side(_batch(tmp_path / "b", [_row("release", 0, True, scene_seed=7)]), {})
    report = pick.pair(before, after, files).summary()
    assert report["starts"] == 0 and report["only_before"] == 1 and report["only_after"] == 1
    no_seed = _row("naive", 0, True)
    del no_seed["noise_seed"]
    with pytest.raises(ValueError, match="no noise_seed"):
        pick.load_side(_batch(tmp_path / "c", [no_seed]), {})


def _jobs_run(root: Path, delivered: list[bool], drop: str | None = None) -> Path:
    root.mkdir(parents=True)
    lines = []
    for i, d in enumerate(delivered):
        start: dict[str, object] = {
            "base_task": 6, "init_state": 30 + i, "scene_seed": 500 + i, "noise_seed": 900 + i,
        }  # fmt: skip
        if drop:
            start.pop(drop)
        lines.append(
            {"job_id": "j0001", "episode_id": f"{root.name}{i}", "start": start, "delivered": d}
        )
    (root / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return root


def test_pairs_in_jobs_runs_use_the_whole_identity(tmp_path: Path) -> None:
    before, files = pick.load_side(_jobs_run(tmp_path / "naive", [False, True, False]), {})
    after, more = pick.load_side(_jobs_run(tmp_path / "program", [True, True, False]), {})
    report = pick.pair(before, after, {**files, **more}).summary()
    counts = (report["both_delivered"], report["rescued"], report["broken"], report["neither"])
    assert counts == (1, 1, 0, 1)
    assert report["first_rescued"][1]["episode"] == "program0"
    assert report["first_broken"] is None
    with pytest.raises(ValueError, match="no scene_seed"):
        pick.load_side(_jobs_run(tmp_path / "old", [True], drop="scene_seed"), {})
