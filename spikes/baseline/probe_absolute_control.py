"""Does LIBERO's absolute control mode behave the same on robosuite 1.4.0 and on the 1.5.2 port?

X-VLA commands absolute end-effector poses (position + axis-angle orientation),
not deltas like SmolVLA and pi0.5. This probe needs no model: it resets
libero_spatial task 0 in absolute mode and
  1. commands the pose the robot already has, for 20 steps (it should not move),
  2. commands a pose 5 cm higher and 5 cm to the side, for 40 steps (it should
     arrive there and keep its orientation).
It prints how far the end effector ends up from the commanded pose.

    python probe_absolute_control.py            as LeRobot's wrapper (plus the port shim) sets it up
    python probe_absolute_control.py --world    robosuite 1.5 only: also switch the controller's
                                                input frame from the robot base to the world
"""

from __future__ import annotations

import json
import sys

import numpy as np
import robosuite
from scipy.spatial.transform import Rotation as R

from lerobot.envs.libero import LiberoEnv, _get_suite

suite = _get_suite("libero_spatial")
env = LiberoEnv(
    task_suite=suite,
    task_id=0,
    task_suite_name="libero_spatial",
    obs_type="pixels_agent_pos",
    observation_width=64,
    observation_height=64,
    control_mode="absolute",
)
obs, _ = env.reset(seed=1000)
world = "--world" in sys.argv
if world:
    for robot in env._env.robots:
        robot.part_controllers["right"].input_ref_frame = "world"


def pose(o):
    return np.asarray(o["robot_state"]["eef"]["pos"], float), np.asarray(o["robot_state"]["eef"]["mat"], float)


def err(o, target_pos, target_mat):
    p, m = pose(o)
    ang = np.degrees(R.from_matrix(m @ target_mat.T).magnitude())
    return round(float(np.linalg.norm(p - target_pos)) * 1000, 1), round(float(ang), 2)


p0, m0 = pose(obs)
out = {"robosuite": robosuite.__version__, "input_frame_forced_to_world": world, "start_pos": p0.round(4).tolist()}

hold = np.concatenate([p0, R.from_matrix(m0).as_rotvec(), [-1.0]]).astype(np.float32)
for _ in range(20):
    obs, *_ = env.step(hold)
out["hold_20_steps: position error mm, orientation error deg"] = err(obs, p0, m0)

target = p0 + np.array([0.0, 0.05, 0.05])
move = np.concatenate([target, R.from_matrix(m0).as_rotvec(), [-1.0]]).astype(np.float32)
for _ in range(40):
    obs, *_ = env.step(move)
out["move_5cm_40_steps: position error mm, orientation error deg"] = err(obs, target, m0)
out["end_pos"] = pose(obs)[0].round(4).tolist()
print("ABSPROBE " + json.dumps(out))
env.close()
