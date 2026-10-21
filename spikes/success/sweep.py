"""How much the rest thresholds of placed matter: a sweep over the recorded feature rows.

    .venv/bin/python spikes/success/sweep.py [--expert expert1000] [--replay replay850]

No simulator. For every pair of thresholds (linear speed, angular speed) placed is scored
as in analyze.py: against the end state of the expert's three continuations (80 steps
after LIBERO's check; window 40) and of the policy releases (`open_lift`, 40 steps).
Columns: expert false successes and misses (all continuations), policy false successes and
misses, policy releases not placed within 40 steps, median steps from the check to placed
for the expert (bowl, basket), the expert's slowest placement, and the median for the
policy releases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import defs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="expert1000")
    ap.add_argument("--replay", default="replay850")
    args = ap.parse_args()
    e_dir, r_dir = A.OUT / "expert" / args.expert, A.OUT / "replay" / args.replay
    exp = []
    for r in (json.loads(x) for x in open(e_dir / "summary.jsonl")):
        if r.get("s_l") is None:
            continue
        z = np.load(e_dir / f"{r['suite']}_t{r['task']}_i{r['init']}.npz")
        exp.append((r, [A.branch_rows(z, b, r["s_l"], 80) for b in A.BRANCHES]))
    rel = []
    for r in (json.loads(x) for x in open(r_dir / "summary.jsonl")):
        z = np.load(r_dir / f"{r['run']}__{r['episode_id']}.npz")
        rel.append({k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("open_lift/")})

    def score() -> tuple:
        fs = mi = 0
        offs: dict[str, list[int]] = {"libero_spatial": [], "libero_object": []}
        for r, branches in exp:
            for i, f in enumerate(branches):
                ok = defs.def_d(f)
                fired, end = bool(ok[:41].any()), bool(defs.def_b(f)[-1])
                fs += fired and not end
                mi += end and not fired
                if i == 0 and ok.any():
                    offs[r["suite"]].append(int(np.argmax(ok)))
        pfs = pmi = notpl = 0
        poffs = []
        for f in rel:
            ok = defs.def_d(f)[:41]
            fired, end = bool(ok.any()), bool(defs.def_b(f)[-1])
            pfs += fired and not end
            pmi += end and not fired
            notpl += not fired
            if fired:
                poffs.append(int(np.argmax(ok)))
        return (fs, mi, pfs, pmi, notpl, int(np.median(offs["libero_spatial"])), int(np.median(offs["libero_object"])),
                max(offs["libero_object"]), int(np.median(poffs)))

    print("| linear, cm/s | angular, rad/s | expert false / missed | policy false / missed | policy not placed "
          "| expert median steps, bowl / basket | expert slowest | policy median steps |")
    print("|---|---|---|---|---|---|---|---|")
    lin0, ang0 = defs.REST_LIN, defs.REST_ANG
    for lin in (0.005, 0.01, 0.02, 0.05, 0.10):
        for ang in (0.05, 0.10, 0.15, 0.30, 0.50, 1.0, float("inf")):
            defs.REST_LIN, defs.REST_ANG = lin, ang
            s = score()
            mark = " (chosen)" if (lin, ang) == (lin0, ang0) else ""
            print(f"| {100 * lin:g}{mark} | {ang:g} | {s[0]} / {s[1]} | {s[2]} / {s[3]} | {s[4]} | {s[5]} / {s[6]} | {s[7]} | {s[8]} |")
    defs.REST_LIN, defs.REST_ANG = lin0, ang0


if __name__ == "__main__":
    main()
