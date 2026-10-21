"""Do absolute and relative control drive the expert's arm the same way?

The expert plans the same unit commands in both modes; in absolute control
they become target poses built on the expert's reading of the present pose,
in relative control robosuite builds the same target from its own reading
(teleop/actions.py). This runs the expert from the same start state in both
modes and compares the grip site's path step by step.

    run.sh probes/mode_paths.py [--inits 0] [--tasks 0-9]

Writes results/mode_paths.json.
"""

from __future__ import annotations

import argparse
import json

import fleetlab  # noqa: F401  (torch before mujoco)
import numpy as np

from episode import OUT, forward, make_env, set_mode
from evaluate import parse_range
from teleop.expert import ScriptedExpert


def path_of(env, init: int, seed: int, mode: str) -> tuple[np.ndarray, str, list[str]]:
    env.reset(init_state=init, seed=seed)
    set_mode(env, mode)
    forward(env)
    expert = ScriptedExpert(env._env, control_mode=mode)
    path = [expert.sc.eef_pos()]
    while not expert.done:
        env.step(expert.act())
        path.append(expert.sc.eef_pos())
    return np.array(path), expert.status, [msg for _, msg in expert.events]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="0")
    ap.add_argument("--tasks", default="0-9")
    args = ap.parse_args()
    rows = []
    for suite in ("libero_spatial", "libero_object"):
        for task in parse_range(args.tasks):
            env = make_env(suite, task)
            for init in parse_range(args.inits):
                a, status_a, ev_a = path_of(env, init, 1000 + init, "absolute")
                r, status_r, ev_r = path_of(env, init, 1000 + init, "relative")
                n = min(len(a), len(r))
                dev = np.linalg.norm(a[:n] - r[:n], axis=1)
                over = np.nonzero(dev > 0.001)[0]
                row = dict(suite=suite, task=task, init=init, steps_abs=len(a) - 1, steps_rel=len(r) - 1,
                           status_abs=status_a, status_rel=status_r, same_event_texts=[e[:20] for e in ev_a]
                           == [e[:20] for e in ev_r], max_dev_mm=round(1000 * float(dev.max()), 3),
                           median_dev_mm=round(1000 * float(np.median(dev)), 4),
                           first_step_over_1mm=int(over[0]) if len(over) else None)
                rows.append(row)
                print(json.dumps(row), flush=True)
            env.close()
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "mode_paths.json").write_text(json.dumps(rows, indent=1))
    devs = [r["max_dev_mm"] for r in rows]
    print(f"{len(rows)} episodes; largest deviation {max(devs):.2f} mm, median of maxima {np.median(devs):.2f} mm; "
          f"same outcome {sum(r['status_abs'] == r['status_rel'] for r in rows)}; "
          f"same step count {sum(r['steps_abs'] == r['steps_rel'] for r in rows)}")


if __name__ == "__main__":
    main()
