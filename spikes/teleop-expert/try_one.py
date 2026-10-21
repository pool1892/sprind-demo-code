"""Run the expert once, print its event log, optionally save a video.

    run.sh try_one.py SUITE TASK_ID INIT_ID [--scenario S] [--fault F] [--video] [--seed N]

The video shows the take-over only (the disturbance is set up before recording starts),
from the two cameras agentview and frontview side by side.
"""
import argparse
import time

import numpy as np

from common import OUT, make_env, reset_to_init
from disturb import apply_fault, apply_scenario
from runner import run_expert

ap = argparse.ArgumentParser()
ap.add_argument("suite")
ap.add_argument("task", type=int)
ap.add_argument("init", type=int)
ap.add_argument("--scenario", default="init")
ap.add_argument("--fault", default="none")
ap.add_argument("--video", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="debug")
args = ap.parse_args()

env, task, inits = make_env(args.suite, args.task)
reset_to_init(env, inits[args.init], seed=args.seed + args.init)
print(task.language)
# same random stream as evaluate.py, so an episode from a batch can be replayed here
rng = np.random.default_rng([args.seed, args.task, args.init, sum(map(ord, args.suite + args.scenario))])
print("fault:", apply_fault(env, args.fault))
setup = apply_scenario(env, args.scenario, rng)
print("setup:", setup)
t0 = time.time()
(OUT / args.out).mkdir(parents=True, exist_ok=True)
video = str(OUT / args.out / f"{args.suite}_t{args.task}_i{args.init}_{args.scenario}_{args.fault}.mp4") if args.video else None
if setup.get("invalid"):
    print("scenario could not be set up:", setup["invalid"])
else:
    res = run_expert(env, verbose=True, video=video)
    print({k: v for k, v in res.items() if k != "events"}, f"{time.time()-t0:.1f}s")
