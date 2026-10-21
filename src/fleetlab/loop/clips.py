"""Before-and-after clips from the reserved panel (docs/design/m3-recipe.md, section 3).

    uv run --locked python -m fleetlab.loop.clips --evaluation LOOP/evaluation --site s2 \\
        [--before original] [--after round-2]

The film the report shows comes from the reserved attempts, chosen by a rule fixed before
any evaluation, and only among the attempts that recorded their videos (in M3 the panel
attempts with index 0 of each robot and day, `EngineSetup.panel_video_indices`). An
attempt is a candidate when both states' runs hold its two videos as files. The rule:

- **at the site:** the first candidate in panel order (`evaluation/panel.json`) at the
  given site (the lasting condition's, from the truth side) that failed under `before`
  and succeeded under `after`;
- **elsewhere:** the first such candidate at any other site.

When no candidate qualifies, the record says so and names no attempt: an attempt without
video is never chosen, and nothing is simulated or replayed to film one. Truth side: the
evaluation is never exported, and the caller passes the site.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["RULE", "Clip", "final_state", "select_clips"]

RULE = (
    "the first attempt in panel order, among those whose videos both states recorded, that "
    "failed under the first state and succeeded under the second: at the given site, and "
    "elsewhere; none when no recorded attempt qualifies"
)
CAMERAS = ("video_scene", "video_wrist")


@dataclass(frozen=True)
class Clip:
    """One chosen attempt: its identity and, per state, its outcome and video files."""

    episode_id: str
    robot_id: str
    site_id: str
    day: int
    episode_index: int
    before: dict[str, Any]
    after: dict[str, Any]


def _attempts(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Each attempt of a state's run: its outcome and the video files that exist."""
    import pyarrow.parquet as pq

    table = run_dir / "log" / "episodes.parquet"
    if not table.is_file():
        raise FileNotFoundError(f"{run_dir}: no episode log (was the state evaluated?)")
    out = {}
    for row in pq.read_table(table, columns=["episode_id", "outcome", *CAMERAS]).to_pylist():
        files = {}
        for camera in CAMERAS:
            relative = row[camera]
            if relative and (run_dir / "log" / relative).is_file():
                files[camera] = str(run_dir / "log" / relative)
        out[row["episode_id"]] = {"outcome": row["outcome"], "videos": files}
    return out


def final_state(evaluation: Path) -> str:
    """The engagement's final state: round 2's if it was evaluated, else round 1's."""
    return "round-2" if (evaluation / "round-2").is_dir() else "round-1"


def select_clips(
    evaluation: Path, site: str, *, before: str = "original", after: str | None = None
) -> dict[str, Any]:
    """The two clips by `RULE` (or why there is none), with the counts behind them."""
    evaluation = Path(evaluation)
    after = after or final_state(evaluation)
    panel: list[Mapping[str, Any]] = json.loads((evaluation / "panel.json").read_text())
    first, second = _attempts(evaluation / before), _attempts(evaluation / after)
    candidates = [
        i
        for i in panel
        if len(first.get(i["episode_id"], {}).get("videos", {})) == len(CAMERAS)
        and len(second.get(i["episode_id"], {}).get("videos", {})) == len(CAMERAS)
    ]
    rescued = [
        i
        for i in candidates
        if first[i["episode_id"]]["outcome"] == "failure"
        and second[i["episode_id"]]["outcome"] == "success"
    ]

    def clip(found: Mapping[str, Any] | None, where: str) -> dict[str, Any]:
        if found is None:
            return {
                "none": f"no recorded attempt {where} failed under {before} and "
                f"succeeded under {after}"
            }
        eid = found["episode_id"]
        return asdict(
            Clip(
                episode_id=eid,
                robot_id=found["robot_id"],
                site_id=found["site_id"],
                day=int(found["day"]),
                episode_index=int(found["episode_index"]),
                before=first[eid],
                after=second[eid],
            )
        )

    at_site = next((i for i in rescued if i["site_id"] == site), None)
    elsewhere = next((i for i in rescued if i["site_id"] != site), None)
    return {
        "rule": RULE,
        "before": before,
        "after": after,
        "site": site,
        "panel_attempts": len(panel),
        "recorded_candidates": len(candidates),
        "rescued_among_recorded": len(rescued),
        "at_site": clip(at_site, f"at {site}"),
        "elsewhere": clip(elsewhere, f"away from {site}"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.clips")
    parser.add_argument("--evaluation", type=Path, required=True, help="LOOP/evaluation")
    parser.add_argument("--site", required=True, help="the lasting condition's site")
    parser.add_argument("--before", default="original")
    parser.add_argument("--after", default=None, help="default: round-2, else round-1")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            select_clips(args.evaluation, args.site, before=args.before, after=args.after), indent=1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
