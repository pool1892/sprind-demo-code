"""Exploration: print what a LIBERO task exposes that a scripted expert needs."""
import sys

import numpy as np

from common import make_env, reset_to_init, suite_tasks

suite_name = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
task_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

suite, tasks = suite_tasks(suite_name)
for i, t in enumerate(tasks):
    print(i, t.language, "|", t.bddl_file)

env, task, init_states = make_env(suite_name, task_id)
print("n init states", len(init_states), "state dim", init_states.shape)
obs = reset_to_init(env, init_states[0])
e = env.env
print("obj_of_interest", e.obj_of_interest)
print("parsed goal_state", e.parsed_problem["goal_state"])
print("objects", list(e.objects_dict.keys()))
print("fixtures", list(e.fixtures_dict.keys()))
print("object_states", list(e.object_states_dict.keys()))
print("obs keys", [k for k in obs.keys()])
robot = e.robots[0]
ctrl = robot.part_controllers["right"]
print("controller", type(ctrl).__name__, "input_type", ctrl.input_type, "ref", ctrl.input_ref_frame)
for k in ("input_max", "input_min", "output_max", "output_min", "kp", "kd", "control_freq",
          "position_limits", "orientation_limits", "uncouple_pos_ori", "interpolator_pos",
          "impedance_mode", "control_delta"):
    print("  ", k, getattr(ctrl, k, "<none>"))
print("gripper", type(robot.gripper["right"]).__name__ if isinstance(robot.gripper, dict) else type(robot.gripper).__name__)
print("eef pos", obs["robot0_eef_pos"], "quat", obs["robot0_eef_quat"])
print("gripper qpos", obs["robot0_gripper_qpos"])
print("base pos", robot.base_pos, "base ori", robot.base_ori)
sim = e.sim
for name in e.objects_dict:
    obj = e.objects_dict[name]
    bid = sim.model.body_name2id(obj.root_body)
    print(f"  obj {name:40s} pos {np.round(sim.data.body_xpos[bid],3)} quat(wxyz) {np.round(sim.data.body_xquat[bid],3)}"
          f" mass {sim.model.body_subtreemass[bid]:.3f}")
for name in e.fixtures_dict:
    obj = e.fixtures_dict[name]
    try:
        bid = sim.model.body_name2id(obj.root_body)
        print(f"  fix {name:40s} pos {np.round(sim.data.body_xpos[bid],3)}")
    except Exception as ex:  # noqa: BLE001
        print("  fix", name, "err", ex)
print("sites:", [sim.model.site_id2name(i) for i in range(sim.model.nsite)])
print("control_freq", e.control_freq, "model timestep", sim.model.opt.timestep, "horizon", e.horizon)
print("success now", env.check_success())
