"""What the hand did in each run's later steps, from the traces (light: CPU only).

    uv run --locked python spikes/jobs/inspect_runs.py DIR/J1/naive [DIR/J8/naive ...]

Per run and per later step: at the switch into it, the gripper command, the finger
opening, the hand's distance from where it was at the job's start, and what the
hand touches; during the step, the closest the hand came to the step's object,
the share of steps it touched it, how high it lifted it, and a class for a step
that did not reach its goal: `never_approached` (the hand stayed more than 8 cm
from the object), `no_grasp` (it came close but lifted it less than 2 cm),
`lifted_not_placed`. A fixture's goal (stove knob, drawer) has no lift.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MOVER = {
    "put_the_cream_cheese_in_the_bowl": "cream_cheese_1",
    "put_the_bowl_on_the_plate": "akita_black_bowl_1",
    "put_the_bowl_on_the_stove": "akita_black_bowl_1",
    "put_the_bowl_on_top_of_the_cabinet": "akita_black_bowl_1",
    "open_the_top_drawer_and_put_the_bowl_inside": "akita_black_bowl_1",
    "push_the_plate_to_the_front_of_the_stove": "plate_1",
    "turn_on_the_stove": "flat_stove_1",
    "open_the_middle_drawer_of_the_cabinet": "wooden_cabinet_1",
}


def describe(run: Path) -> tuple[str, list[str]]:
    s = json.loads((run / "summary.json").read_text())
    tr = np.load(run / "trace.npz")
    objects = [str(o) for o in tr["objects"]]
    idx = {o: i for i, o in enumerate(objects)}
    pos, contact, action = tr["pos"], tr["robot_contact"], tr["action"]
    eef, opening = tr["eef"], tr["opening"]
    classes = []
    parts = [f"{run.name} {s['tag']}"]
    for k in range(1, len(s["steps"])):
        rec = s["steps"][k]
        if rec.get("start_t") is None:
            break
        a, b = rec["start_t"], rec["end_t"]
        mover = MOVER[rec["stem"]]
        i = idx[mover]
        sw = a - 1
        touched = [o for o in objects if contact[sw, i := idx[o]]] if a > 0 else []
        i = idx[mover]
        away = np.linalg.norm(eef[sw] - eef[0]) * 100
        near = float(np.linalg.norm(eef[a:b] - pos[a:b, i], axis=1).min()) * 100
        share = float(contact[a:b, i].mean())
        lift = float((pos[a:b, i, 2] - pos[a, i, 2]).max()) * 100
        cls = "reached"
        if rec["outcome"] != "reached":
            if near > 8:
                cls = "never_approached"
            elif mover.startswith(("flat_stove", "wooden_cabinet")) or lift < 2:
                cls = "no_grasp"
            else:
                cls = "lifted_not_placed"
        classes.append(cls)
        parts.append(
            f"| step {k + 1} {cls}: switch grip={action[sw, 6]:+.0f} open={opening[sw]:.3f} "
            f"hand {away:.0f} cm from start, touching {touched}; nearest {near:.0f} cm, "
            f"touch {share:.2f}, lift {lift:.0f} cm"
        )
    return " ".join(parts), classes


def main() -> int:
    for arg in sys.argv[1:]:
        counts: Counter[str] = Counter()
        for run in sorted(Path(arg).glob("s*")):
            if not (run / "summary.json").exists():
                continue
            line, classes = describe(run)
            counts.update(f"step{k + 2}:{c}" for k, c in enumerate(classes))
            print(line)
        print(f"== {arg}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
