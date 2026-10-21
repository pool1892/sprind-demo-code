"""A probe's figure for the film, drawn from the probe's recorded tables.

A probe the agent requested during an engagement is recorded in the session's export
(`log/probe_episodes.parquet`, `log/probe_steps/<episode>.parquet`); the engagement's
exhibit does not draw it. This draws it with the loop's own probe figure
(`fleetlab.loop.exhibit.figures.probe_figure`: commanded against measured hand position,
one panel per robot, repetitions overlaid, the rows to the first movement noted), and writes
beside the PNG what it was drawn from (`<png>.json`: the probe, the robots, every table read
with its SHA-256), so that the film's image shot resolves to the recorded probe.

    python -m fleetlab.film probe-figure EXPORT_LOG --probe p0001 \\
        --robot "delayed site=r01" --robot "lagging robot=r06" ... --out FIGURE.png
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["draw"]

FORMAT = "fleetlab.film-probe-figure/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(
    log: Path,
    probe_id: str,
    robots: Sequence[tuple[str, str]],
    out: Path,
    *,
    title: str | None = None,
    axis: str = "x",
) -> dict[str, Any]:
    """Draw probe `probe_id` of the export `log` for `robots` ((label, robot id) panels)."""
    import pyarrow.parquet as pq

    from fleetlab.loop.exhibit.figures import probe_figure

    episodes_file = log / "probe_episodes.parquet"
    if not episodes_file.is_file():
        raise FileNotFoundError(f"{log} holds no probe_episodes.parquet")
    read = {str(episodes_file): _sha256(episodes_file)}
    rows = [r for r in pq.read_table(episodes_file).to_pylist() if r["probe_id"] == probe_id]
    if not rows:
        raise ValueError(f"{episodes_file} has no episodes of probe {probe_id}")
    panels: list[tuple[str, list[list[dict[str, Any]]]]] = []
    shown: list[dict[str, Any]] = []
    for label, robot in robots:
        mine = sorted((r for r in rows if r["robot_id"] == robot), key=lambda r: r["repeat"])
        if not mine:
            raise ValueError(f"probe {probe_id} did not run on {robot}")
        steps = []
        for r in mine:
            path = log / "probe_steps" / f"{r['episode_id']}.parquet"
            read[str(path)] = _sha256(path)
            table = sorted(pq.read_table(path).to_pylist(), key=lambda s: s["step"])
            steps.append(table)
            shown.append({"robot": robot, "repeat": r["repeat"], "episode_id": r["episode_id"]})
        panels.append((f"{label}: {robot}" if label else robot, steps))
    out.parent.mkdir(parents=True, exist_ok=True)
    probe_figure(out, panels, title or f"probe {probe_id}: commanded against measured", "", axis)
    record = {
        "format": FORMAT,
        "probe_id": probe_id,
        "panels": [{"label": label, "robot": robot} for label, robot in robots],
        "episodes": shown,
        "axis": axis,
        "files_read": read,
        "figure_sha256": _sha256(out),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(record, indent=1) + "\n")
    return record
