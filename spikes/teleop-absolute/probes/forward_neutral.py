"""Does the hand-over procedure leave the physics alone?

The take-over calls `mj_forward` before the expert starts, so that MuJoCo's
derived quantities (positions of bodies and sites, contacts) belong to the
present state. robosuite runs with `lite_physics`, which recomputes them at the
start of every physics substep, so the extra call should change nothing that
follows. And a snapshot taken and restored after a reset elsewhere must
continue exactly as the uninterrupted episode.

For recorded X-VLA episodes (bit-for-bit replayable, probes/replay_xvla_check.py)
this plays the recording, interrupts it after step k in one of four ways, plays
the rest, and compares every state hash with the recording:

  forward   mj_forward at step k
  restore   snapshot at step k, reset to another start state and scene seed, 3 steps there, restore
  both      restore, then mj_forward (what the take-over does)
  lean      as restore, but only MuJoCo's state with warm start, the model's small arrays
            (fixture poses), the gripper controller and the sensor clocks are put back;
            the arm controller (its goal pose), the composite controller, the robot's
            buffers, robosuite's counters and LIBERO's object trackers are not

    run.sh probes/forward_neutral.py [--n 12]
"""

from __future__ import annotations

import argparse
import json

import fleetlab  # noqa: F401  (torch before mujoco)
import numpy as np

import snapshot
from episode import EpisodeSpec, forward, make_env, scramble
from xvla_record import h64, load_episode, load_index, mj_state
from xvla_select import OUT as HANDOVERS

HOWS = ("forward", "restore", "both", "lean")
LEAN_SKIP = ("arm", "composite", "robot", "env", "objects")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    items = [json.loads(line) for line in open(HANDOVERS / "xvla_mid.jsonl")]
    rng = np.random.default_rng(5)
    picks = [items[int(i)] for i in rng.choice(len(items), size=args.n, replace=False)]
    results = []
    for item in sorted(picks, key=lambda x: (x["suite"], x["task"])):
        from pathlib import Path

        run_dir = Path(item["run_dir"])
        rec = load_episode(run_dir, item["episode_id"], load_index(run_dir))
        k = item["handover_step"]
        env = make_env(rec.suite, rec.task)
        spec = EpisodeSpec(rec.suite, rec.task, rec.init, rec.scene_seed, "xvla")
        row = dict(episode=rec.episode_id, suite=rec.suite, k=k, steps=len(rec.actions))
        for how in HOWS:
            env.reset(init_state=rec.init, seed=rec.scene_seed)
            for t in range(k):
                env.step(rec.actions[t])
            assert h64(mj_state(env)) == rec.state_hashes[k]
            if how in ("restore", "both", "lean"):
                snap = snapshot.take(env)
                scramble(env, spec)
                snapshot.restore(env, snap, skip=LEAN_SKIP if how == "lean" else ())
            if how in ("forward", "both"):
                forward(env)
            same = h64(mj_state(env)) == rec.state_hashes[k]
            first_diff = None if same else k
            for t in range(k, len(rec.actions)):
                env.step(rec.actions[t])
                if first_diff is None and h64(mj_state(env)) != rec.state_hashes[t + 1]:
                    first_diff = t + 1
            row[how] = "identical" if first_diff is None else f"differs from step {first_diff}"
        env.close()
        print(json.dumps(row), flush=True)
        results.append(row)
    for how in HOWS:
        print(how, sum(r[how] == "identical" for r in results), "of", len(results), "identical to the end")


if __name__ == "__main__":
    main()
