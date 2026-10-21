"""Small recorded runs for the film's tests: a fleet log and an episode directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.video import write_mp4


def frames(n: int, width: int, height: int, seed: int) -> list[npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        frame = base.copy()
        frame[: (i * 7) % height, :8] = 255  # something that moves
        out.append(frame)
    return out


def fleet_run(root: Path) -> Path:
    """`<root>/log/` with two episodes: e1 (failure, 40 steps, a frame every 4th step) and e2
    (success, 20 steps)."""
    log = root / "log"
    (log / "video").mkdir(parents=True)
    (log / "steps").mkdir()
    rows = []
    for eid, steps, outcome, seed in (("e1", 40, "failure", 1), ("e2", 20, "success", 2)):
        n_frames = steps // 4 + 1
        write_mp4(
            log / "video" / f"{eid}.scene.mp4",
            frames(n_frames, 64, 64, seed),
            fps=5,
            repeatable=True,
        )
        pq.write_table(
            pa.table(
                {
                    "episode_id": [eid] * (steps + 1),
                    "step": list(range(steps + 1)),
                    "frame_index": [s // 4 if s % 4 == 0 else None for s in range(steps + 1)],
                }
            ),
            log / "steps" / f"{eid}.parquet",
        )
        rows.append(
            {
                "episode_id": eid,
                "robot_id": "r01",
                "site_id": "s1",
                "day": 0,
                "task_text": "put the bowl on the plate",
                "steps": steps,
                "outcome": outcome,
                "video_scene": f"video/{eid}.scene.mp4",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), log / "episodes.parquet")
    return root


def episode_run(root: Path) -> Path:
    """`<root>/J1/naive/s00/` with a 30-step two-skill chain, both cameras side by side."""
    folder = root / "J1" / "naive" / "s00"
    folder.mkdir(parents=True)
    write_mp4(folder / "scene.mp4", frames(31, 128, 64, 3), fps=20, repeatable=True)
    summary = {
        "job": "J1",
        "variant": "naive",
        "start": 0,
        "control_steps": 30,
        "delivered": False,
        "tag": "held_at_switch",
        "steps": [
            {
                "index": 0,
                "instruction": "put the cream cheese in the bowl",
                "start_t": 0,
                "end_t": 12,
                "outcome": "reached",
            },
            {
                "index": 1,
                "instruction": "put the bowl on the plate",
                "start_t": 12,
                "end_t": 30,
                "outcome": "timeout",
            },
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary))
    results = [
        {"job": "J1", "variant": "naive", "start": 0, "delivered": False},
        {"job": "J1", "variant": "naive", "start": 1, "delivered": True},
        {"job": "J1", "variant": "naive", "start": 2, "delivered": True},
        {"job": "J2", "variant": "naive", "start": 0, "delivered": True},
    ]
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    # a chain with a transition motion between its policy steps (control steps 12 to 19)
    folder = root / "J1" / "release" / "s00"
    folder.mkdir(parents=True)
    write_mp4(folder / "scene.mp4", frames(31, 128, 64, 4), fps=20, repeatable=True)
    release = dict(summary, variant="release", delivered=True, tag="delivered")
    release["steps"] = [
        {"instruction": "put the cream cheese in the bowl", "start_t": 0, "end_t": 12},
        {"instruction": "put the bowl on the plate", "start_t": 20, "end_t": 30},
    ]
    (folder / "summary.json").write_text(json.dumps(release))
    more = [{"job": "J1", "variant": "naive", "start": 3, "delivered": True}]
    (root / "more.jsonl").write_text("".join(json.dumps(r) + "\n" for r in more))
    return root


def manifest(
    path: Path, fleet: Path, chain: Path, script: Path | None = None, **extra: object
) -> Path:
    data: dict[str, object] = {
        "format": "fleetlab.film-manifest/1",
        "title": "Test film",
        "fps": 20,
        "stories": {"A": "Make this one job"},
        "beats": [
            {
                "id": "open",
                "title": "Opening",
                "shots": [{"kind": "card", "title": "Test", "seconds": 1}],
            },
            {
                "id": "naive",
                "story": "A",
                "title": "The naive attempt",
                "placeholder": "a test run",
                "panel": {
                    "requirement": ["cream cheese in the bowl", "bowl on the plate"],
                    "program": "naive chain",
                    "current": "auto",
                    "count": {
                        "text": "{k} of {n} delivered",
                        "from": {
                            "table": str(chain / "results.jsonl"),
                            "where": {"job": "J1"},
                            "success": "delivered",
                        },
                    },
                },
                "shots": [
                    {
                        "kind": "clip",
                        "clip": {"run": str(chain), "episode": "J1/naive/s00", "label": "naive"},
                    }
                ],
            },
            {
                "id": "pair",
                "story": "A",
                "title": "Before and after",
                "shots": [
                    {
                        "kind": "pair",
                        "clips": [
                            {"run": str(fleet), "episode": "e1", "label": "before", "speed": 2},
                            {"run": str(fleet), "episode": "e2", "label": "after", "speed": 2},
                        ],
                    }
                ],
            },
            {
                "id": "missing",
                "story": "A",
                "title": "Later",
                "needs": "a run",
                "shots": [{"kind": "slate", "seconds": 1}],
            },
        ],
    }
    if script is not None:
        data["script"] = str(script)
    data.update(extra)
    path.write_text(json.dumps(data))
    return path


def jobs_run(root: Path) -> Path:
    """A jobs runner's run (`fleetlab.jobs.tables`): episode e000101, 24 control steps, a
    frame every 4th step and the last; a policy step, a motion, a skipped policy step."""
    log = root / "log"
    for sub in ("video", "robot_steps"):
        (log / sub).mkdir(parents=True)
    eid, steps = "e000101", 24
    write_mp4(log / "video" / f"{eid}.scene.mp4", frames(7, 64, 64, 5), fps=5, repeatable=True)
    pq.write_table(
        pa.table(
            {
                "episode_id": [eid] * (steps + 1),
                "step": list(range(steps + 1)),
                "frame_index": [s // 4 if s % 4 == 0 else None for s in range(steps + 1)],
            }
        ),
        log / "robot_steps" / f"{eid}.parquet",
    )
    episode = {
        "job_id": "j0001",
        "episode_id": eid,
        "control_steps": steps,
        "delivered": True,
        "video_scene": f"video/{eid}.scene.mp4",
        "video_wrist": None,
    }
    pq.write_table(pa.Table.from_pylist([episode]), log / "job_episodes.parquet")
    events = [
        {"episode_id": eid, "program_step": 0, "kind": "policy", "start_step": 0, "end_step": 10,
         "outcome": "reached", "instruction": "put the cream cheese in the bowl"},
        {"episode_id": eid, "program_step": 1, "kind": "motion", "start_step": 10, "end_step": 18,
         "outcome": "reached", "instruction": None},
        {"episode_id": eid, "program_step": 2, "kind": "policy", "start_step": None,
         "end_step": None, "outcome": "skipped", "instruction": "put the bowl on the plate"},
        {"episode_id": eid, "program_step": None, "kind": "tail", "start_step": 18,
         "end_step": 24, "outcome": "reached", "instruction": "put the cream cheese in the bowl"},
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(events), log / "job_step_events.parquet")
    return root


def rerender(
    folder: Path,
    run: Path,
    original_steps: int = 24,
    overrides: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """A re-render of jobs_run's episode at one frame per control step, and the jobs
    runner's report on it, matching unless `overrides` change it."""
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / "e000101.rerender.scene.128px.mp4"
    write_mp4(video, frames(original_steps + 1, 128, 128, 6), fps=20, repeatable=True)
    report: dict[str, object] = {
        "format": "fleetlab.job-rerender/1",
        "label": "re-render",
        "accepted": True,
        "episode_id": "e000101",
        "job_id": "j0001",
        "run_dir": str(run),
        "original_video": str(run / "log" / "video" / "e000101.scene.mp4"),
        "video": video.name,
        "camera": "scene",
        "frame_every": 1,
        "frames": original_steps + 1,
        "steps": original_steps,
        "hashes_compared": original_steps + 1,
        "hash_mismatches": 0,
        "verdict_matches": True,
    }
    report.update(overrides or {})
    path = folder / "e000101.rerender.scene.128px.json"
    path.write_text(json.dumps(report))
    return video, path
