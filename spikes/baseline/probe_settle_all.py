"""Where do the objects come to rest after reset, for every task of the four suites?

No model. For each task this resets LeRobot's LIBERO wrapper on stored initial
state 0 and writes the resting position and tilt of every object, plus the
robot state the policy would see. Run it under two MuJoCo versions and compare
the two files with --compare to see which start states depend on the version.

    python probe_settle_all.py --out settle_mj381.json
    python probe_settle_all.py --compare settle_mj337.json settle_mj381.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
ap.add_argument("--compare", nargs=2)
args = ap.parse_args()

if args.compare:
    a, b = (json.load(open(p)) for p in args.compare)
    print(f"A: mujoco {a['mujoco']}   B: mujoco {b['mujoco']}   (robosuite {a['robosuite']} / {b['robosuite']})")
    n = flagged = 0
    for key in a["tasks"]:
        ta, tb = a["tasks"][key], b["tasks"][key]
        worst, worst_name, worst_tilt = 0.0, "", 0.0
        for name, va in ta["bodies"].items():
            vb = tb["bodies"][name]
            d = float(np.linalg.norm(np.array(va["pos"]) - np.array(vb["pos"]))) * 1000
            if d > worst:
                worst, worst_name, worst_tilt = d, name, abs(va["tilt_deg"] - vb["tilt_deg"])
        eef = float(np.linalg.norm(np.array(ta["state"][:3]) - np.array(tb["state"][:3]))) * 1000
        n += 1
        mark = ""
        if worst > 5.0:
            flagged += 1
            mark = "  <-- differs"
        print(f"{key:18s} largest object shift {worst:6.1f} mm ({worst_name}, tilt differs {worst_tilt:.1f} deg), "
              f"gripper position differs {eef:.1f} mm{mark}")
    print(f"{flagged} of {n} tasks have an object resting more than 5 mm apart between the two versions")
    sys.exit(0)

import mujoco  # noqa: E402
import robosuite  # noqa: E402

from lerobot.envs.libero import LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.processor.env_processor import LiberoProcessorStep  # noqa: E402

step = LiberoProcessorStep()
report = {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__, "tasks": {}}
for suite_name in args.suites.split(","):
    suite = _get_suite(suite_name)
    for task_id in range(len(suite.tasks)):
        # LIBERO draws the exact position of the fixed furniture (stove, cabinet, rack)
        # from NumPy's global generator at every reset; the stored initial state does
        # not contain it. Seed it, or two runs differ by up to 2 cm for that reason alone.
        np.random.seed(1000)
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            observation_width=64,
            observation_height=64,
        )
        obs, _ = env.reset(seed=1000)
        sim = env._env.sim
        bodies = {}
        for b in range(sim.model.nbody):
            name = sim.model.body(b).name
            if name.endswith("_main") and not name.startswith(("robot", "gripper", "mount")):
                z = sim.data.xmat[b].reshape(3, 3)[:, 2]
                bodies[name] = {
                    "pos": [round(float(x), 5) for x in sim.data.xpos[b]],
                    "tilt_deg": round(float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))), 2),
                }
        batched = {
            "pixels": {k: v[None] for k, v in obs["pixels"].items()},
            "robot_state": {
                g: {k: np.asarray(v)[None] for k, v in d.items()} for g, d in obs["robot_state"].items()
            },
        }
        state = step.observation(preprocess_observation(batched))["observation.state"][0].tolist()
        report["tasks"][f"{suite_name}/{task_id}"] = {
            "language": suite.get_task(task_id).language,
            "state": [round(x, 5) for x in state],
            "bodies": bodies,
        }
        env.close()
        print(suite_name, task_id, "done", flush=True)

with open(args.out, "w") as f:
    json.dump(report, f, indent=1)
