"""Record real observations once, so that the policy-only checks need no simulator.

    tools/heavy-run --light ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/xvla-speed/make_obs.py [--inits 6] [--moved 2]

For each of the ten `libero_spatial` tasks: the observation after a pinned reset
of start states 0 to inits-1, and for the first `moved` start states one more
observation after 40 control steps in which the arm is driven about 6 cm
sideways and down (so that some observations are not reset poses). Writes
~/.cache/robots-probes/xvla-speed/obs/obs.pkl: a list of
{"task", "init", "phase", "task_description", "obs"}.

One environment at a time in one process, no policy: a light job.
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

import xs_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, default=6)
    ap.add_argument("--moved", type=int, default=2)
    args = ap.parse_args()
    out = C.OUT / "obs"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.perf_counter()
    for task in range(10):
        env = C.make_env(task)
        for init in range(args.inits):
            o = C.reset_pinned(env, init, 1000 + 100 * task + init)
            records.append({"task": task, "init": init, "phase": "reset",
                            "task_description": env.task_description, "obs": o})
            if init < args.moved:
                # absolute control: command the current pose shifted a little, 40 times
                import robosuite.utils.transform_utils as T

                rs = o["robot_state"]
                target = np.concatenate([rs["eef"]["pos"] + np.array([0.04, 0.03, -0.04]), np.zeros(3), [-1.0]])
                # orientation: keep the current one, as an axis-angle vector
                target[3:6] = T.quat2axisangle(np.asarray(rs["eef"]["quat"], dtype=np.float64))
                ctrl_obs = o
                for _ in range(40):
                    ctrl_obs, _, terminated, _, _ = env.step(target.astype(np.float32))
                    if terminated:
                        break
                records.append({"task": task, "init": init, "phase": "moved",
                                "task_description": env.task_description, "obs": ctrl_obs})
        C.make_current(env)
        env.close()
        print(f"task {task}: {len(records)} observations, {time.perf_counter() - t0:.0f} s", flush=True)
    with open(out / "obs.pkl", "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    img = records[0]["obs"]["pixels"]["image"]
    print(f"wrote {len(records)} observations to {out / 'obs.pkl'}; image {img.shape} {img.dtype}; load {C.loadavg()}")


if __name__ == "__main__":
    main()
