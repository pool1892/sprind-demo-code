"""Time defs.features per control step against a control step itself, on one scene per suite.

    spikes/success/run.sh bench bench.py

Builds the expert's environment without a renderer (as expert_tail.py does), resets to
initial state 0 of task 0, and times 2,000 calls of defs.features, 2,000 of LIBERO's own
env.check_success() and 100 control steps with the gripper held open. Output in
~/.cache/robots-probes/success/logs/bench.log.
"""

from __future__ import annotations

import fleetlab  # noqa: F401  (torch before mujoco)
from fleetlab.runtime import configure_process

configure_process()

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "teleop-expert")]

import numpy as np  # noqa: E402

import defs  # noqa: E402
from common import DUMMY, make_env, reset_to_init  # noqa: E402

for suite in ("libero_spatial", "libero_object"):
    env, task, inits = make_env(suite, 0, renderer=False)
    reset_to_init(env, inits[0])
    ix = defs.build_index(env.env)
    m, d = env.env.sim.model._model, env.env.sim.data._data
    t0 = time.perf_counter()
    for _ in range(2000):
        defs.features(m, d, ix)
    tf = (time.perf_counter() - t0) / 2000
    t0 = time.perf_counter()
    for _ in range(2000):
        env.check_success()
    tc = (time.perf_counter() - t0) / 2000
    t0 = time.perf_counter()
    for _ in range(100):
        env.step(np.asarray(DUMMY, float))
    ts = (time.perf_counter() - t0) / 100
    print(f"{suite}: defs.features {tf * 1e3:.3f} ms, env.check_success {tc * 1e3:.3f} ms, "
          f"control step without rendering {ts * 1e3:.2f} ms, contacts in the scene {d.ncon}", flush=True)
