"""Choose the X-VLA hand-overs: which recorded episodes, and after how many of their steps.

    run.sh xvla_select.py

Writes two lists to ~/.cache/robots-probes/teleop-absolute/handovers/:

* `xvla_end.jsonl`: every X-VLA episode in the recordings below that failed,
  handed over as it stood after its 280 steps, the way the SmolVLA failures
  were handed over in spikes/teleop-expert.
* `xvla_mid.jsonl`: 100 episodes per suite drawn at random, handed over after
  a random number of steps between 10 and the step before the episode ended,
  that is in the middle of X-VLA's motion and before LIBERO counted a success.

Recordings: the walking skeleton's runs and the xvla-speed spike's runs in
32-bit floats (the checkpoint as stored), from the baseline probe venv. Their
episodes replay bit for bit in the locked environment (probes/replay_xvla_check.py).
Faults in those runs (scene camera turned, actions late) act on what the
policy sees or when its actions arrive, not on the physics, so the recorded
executed actions replay in a healthy simulator. An episode recorded twice
(same final state hash) is used once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xvla_record import load_index

PROBES = Path.home() / ".cache" / "robots-probes"
SOURCES = [
    PROBES / "skeleton" / "runs" / name
    for name in ("healthy200", "dose_cam_p1", "dose_cam_p2", "dose_delay_p1", "dose_delay_p2", "fleet")
] + [
    PROBES / "xvla-speed" / "runs" / name
    for name in ("h200v_fp32_r4w10", "m200v_fp32_r4w10", "p_cam3_fp32", "p_cam6a_fp32", "p_cam6b_fp32",
                 "p_cam9_fp32", "p_delay17_fp32", "p_obj_fp32")
]
MID_SOURCES = {"libero_spatial": "h200v_fp32_r4w10", "libero_object": "p_obj_fp32"}
MID_PER_SUITE = 100
OUT = PROBES / "teleop-absolute" / "handovers"


def entry(run_dir: Path, row: dict, handover: int, tag: str) -> dict:
    return dict(run_dir=str(run_dir), episode_id=row["episode_id"], suite=row["suite"], task=int(row["task"]),
                init=int(row["init"]), scene_seed=int(row["scene_seed"]), handover_step=int(handover),
                xvla_success=bool(row["success"]), xvla_steps=int(row["n_steps"]), faults=row.get("faults") or [],
                tag=tag)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    end = []
    for run_dir in SOURCES:
        for row in load_index(run_dir):
            if row["success"] or row["final_state_hash"] in seen:
                continue
            seen.add(row["final_state_hash"])
            end.append(entry(run_dir, row, row["n_steps"], f"end:{run_dir.name}"))
    rng = np.random.default_rng(20261007)
    mid = []
    for suite, name in MID_SOURCES.items():
        run_dir = next(p for p in SOURCES if p.name == name)
        rows = [r for r in load_index(run_dir) if r["suite"] == suite]
        for i in rng.choice(len(rows), size=min(MID_PER_SUITE, len(rows)), replace=False):
            row = rows[int(i)]
            handover = int(rng.integers(10, row["n_steps"]))  # 10 .. n_steps - 1
            mid.append(entry(run_dir, row, handover, f"mid:{run_dir.name}"))
    for name, items in (("xvla_end", end), ("xvla_mid", mid)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        suites = {s: sum(1 for i in items if i["suite"] == s) for s in sorted({i["suite"] for i in items})}
        print(name, len(items), suites)
    by_source: dict[str, int] = {}
    for e in end:
        by_source[e["tag"]] = by_source.get(e["tag"], 0) + 1
    print("failures by recording:", by_source)


if __name__ == "__main__":
    main()
