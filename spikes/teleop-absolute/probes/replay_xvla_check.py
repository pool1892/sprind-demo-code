"""Does an X-VLA episode recorded by the walking skeleton replay bit for bit here?

The skeleton (spikes/skeleton) ran X-VLA in the baseline probe venv (MuJoCo
3.3.7, robosuite 1.5.2 port, LeRobot's LIBERO wrapper, absolute control, soft
resets) and stored per episode the executed actions and a hash of MuJoCo's
complete integration state after every step. This probe resets the locked
environment's adapter to the same (task, start state, scene seed), plays the
stored actions and compares the hashes.

    run.sh probes/replay_xvla_check.py RUN EPISODE_ID [EPISODE_ID ...] [--size 64] [--hard]
"""

from __future__ import annotations

import argparse
import json

import fleetlab  # noqa: F401  (torch before mujoco)
import numpy as np

from fleetlab.sim.libero_env import LiberoEnv
from xvla_record import RUNS, load_episode, state_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--hard", action="store_true")
    args = ap.parse_args()
    env, env_task = None, None
    for eid in args.episodes:
        rec = load_episode(RUNS / args.run, eid)
        if env_task != (rec.suite, rec.task):
            if env is not None:
                env.close()
            env = LiberoEnv(rec.suite, rec.task, image_size=args.size, control_mode="absolute",
                            hard_reset=args.hard)
            env_task = (rec.suite, rec.task)
        env.reset(init_state=rec.init, seed=rec.scene_seed)
        hashes = [state_hash(env)]
        for a in rec.actions:
            env.step(a)
            hashes.append(state_hash(env))
        hashes = np.array(hashes, np.uint64)
        same = hashes == rec.state_hashes
        first_diff = int(np.argmin(same)) if not same.all() else None
        print(json.dumps(dict(episode=eid, steps=len(rec.actions), identical=bool(same.all()),
                              first_differing_step=first_diff, success_recorded=rec.success,
                              success_now=bool(env._env.check_success()))), flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
