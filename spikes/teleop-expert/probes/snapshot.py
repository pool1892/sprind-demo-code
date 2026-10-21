"""Large pictures of an episode at chosen steps: run.sh probes/snapshot.py SUITE TASK INIT STEP,STEP[,..] [CAMERAS]"""
import sys, numpy as np, imageio
from common import make_env, reset_to_init, frame, OUT
from expert import ScriptedExpert
suite, tid, iid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
steps = [int(s) for s in sys.argv[4].split(",")]
cams = sys.argv[5].split(",") if len(sys.argv) > 5 else ["agentview", "frontview", "sideview", "birdview"]
env, task, inits = make_env(suite, tid, use_camera_obs=False)
reset_to_init(env, inits[iid])
print("cameras", [env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)])
ex = ScriptedExpert(env, verbose=True)
rows = []
while ex.t <= max(steps):
    if ex.t in steps:
        rows.append(np.concatenate([frame(env, c, 512) for c in cams], axis=1))
    env.step(ex.act())
out = OUT / "debug" / f"snap_{suite}_{tid}_{iid}.png"
imageio.imwrite(out, np.concatenate(rows, axis=0))
print(out)
