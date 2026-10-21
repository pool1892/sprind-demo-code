"""Recount two diagnostics of the second reliable job as one JSON record for the record page.

The record page needs 95% intervals for two counts that no recorded file holds as data:
the confirmation runs whose bowl goal stopped holding after the bowl first arrived and
held again by the end (the agent's account gives the count only in prose), and the
reserved runs whose cheese ends touching the table. This reads the public goal table of
the confirmation batch and the coordinator's cheese record, and writes one row per
episode; the binder's existing counts and Wilson intervals do the rest.

    uv run --locked python site/reliability/record/recount_second_job.py OUT.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

COHORT = Path.home() / ".cache/robots-probes/demo/2026-10-20/reliability/plate-cabinet-01"
LOG = COHORT / "run/runs/session-06/run/log"
CONFIRMATION_JOB = "j0019"
BOWL_GOAL = 1
CHEESE = COHORT / "relocated-cheese-final-positions.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out: Path) -> None:
    episodes_file = LOG / "job_episodes.parquet"
    episodes = [r for r in pq.read_table(episodes_file).to_pylist()
                if r["job_id"] == CONFIRMATION_JOB]  # fmt: skip
    sources = {episodes_file: _sha256(episodes_file), CHEESE: _sha256(CHEESE)}
    runs = []
    for episode in sorted(episodes, key=lambda r: r["init_state"]):
        steps_file = LOG / "goal_steps" / f"{episode['episode_id']}.parquet"
        sources[steps_file] = _sha256(steps_file)
        rows = sorted((r for r in pq.read_table(steps_file).to_pylist() if r["goal"] == BOWL_GOAL),
                      key=lambda r: r["step"])  # fmt: skip
        held = [r["step"] for r in rows if r["holds"]]
        first = held[0] if held else None
        lost = first is not None and any(not r["holds"] and r["step"] > first for r in rows)
        runs.append({
            "episode_id": episode["episode_id"],
            "init_state": episode["init_state"],
            "delivered": bool(episode["delivered"]),
            "bowl_goal_first_held_step": first,
            "bowl_goal_lost_after_first_held": lost,
            "bowl_goal_held_at_end": bool(rows and rows[-1]["holds"]),
        })  # fmt: skip
    cheese = [
        {
            "episode_id": e["episode_id"],
            "touches_table": "table" in e["in_contact_with"],
            "touches_plate": "plate_1" in e["in_contact_with"],
            "touched_by_robot": bool(e["in_contact_with_robot"]),
        }
        for e in json.loads(CHEESE.read_text())["episodes"]
    ]
    record = {
        "format": "robots.record-recount/1",
        "what": "second job: bowl-goal losses after arrival in the confirmation runs, from the "
        "public goal table; the cheese's final contacts in the reserved runs, from the "
        "coordinator's record of the public object table",
        "script": "site/reliability/record/recount_second_job.py",
        "script_sha256": _sha256(Path(__file__)),
        "sources": [
            {"file": p.name, "parent": p.parent.name, "sha256": s} for p, s in sources.items()
        ],
        "confirmation_runs": runs,
        "cheese_final": cheese,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    recovered = sum(
        r["bowl_goal_lost_after_first_held"] and r["bowl_goal_held_at_end"] for r in runs
    )
    on_table = sum(c["touches_table"] for c in cheese)
    print(f"wrote {out}: {len(runs)} confirmation runs, {recovered} recovered after a loss; "
          f"cheese on the table in {on_table} of {len(cheese)}")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
