"""Probe RoboTwin 2.0 on this machine without CUDA: does a task load, render and step?

Usage: python robotwin_probe.py --root ROBOTWIN_CLONE --out DIR [--task handover_block] [--stub]

Without --stub the script imports the environment code as shipped and records where it
fails. With --stub it puts a do-nothing class in place of CuroboPlanner before the robot
module is imported (CuRobo is NVIDIA-only), then:
- builds the task with the clean config (demo_clean.yml, aloha-agilex) in eval mode for a seed,
- saves the head and wrist camera frames,
- holds the current joint positions for N take_action calls (qpos mode, the mode a
  joint-space policy uses) and times them,
- builds the task a second time with the same seed in a fresh object and compares the
  observations, and the rollout, bit for bit.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def load_args(root, task_name):
    import yaml

    cfg = Path(root) / "env_cfg" / "task_config"
    args = yaml.safe_load(open(cfg / "demo_clean.yml"))
    args["task_name"] = task_name
    args["task_config"] = "demo_clean"
    args["data_type"]["rgb"] = True
    args["data_type"]["qpos"] = True
    args["data_type"]["endpose"] = True
    emb = yaml.safe_load(open(cfg / "_embodiment_config.yml"))
    cams = yaml.safe_load(open(cfg / "_camera_config.yml"))
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cams[head]["h"]
    args["head_camera_w"] = cams[head]["w"]
    robot_file = emb[args["embodiment"][0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    conf = yaml.safe_load(open(Path(root) / robot_file / "config.yml"))
    args["left_embodiment_config"] = conf
    args["right_embodiment_config"] = conf
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    args["eval_video_log"] = False
    return args


def obs_digest(obs):
    h = hashlib.sha256()
    for cam in sorted(obs["observation"]):
        rgb = obs["observation"][cam].get("rgb")
        if rgb is not None:
            h.update(np.ascontiguousarray(rgb).tobytes())
    h.update(np.asarray(obs["joint_action"]["vector"], dtype=np.float64).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="handover_block")
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = str(Path(a.root).resolve())
    os.chdir(root)  # RoboTwin opens ./assets/... relative to its root
    sys.path.insert(0, root)
    result = {"task": a.task, "stub": a.stub, "seed": a.seed}

    import torch  # noqa: F401  RoboTwin imports torch anyway; import it before sapien

    if a.stub:
        # envs/robot/__init__.py imports robot.py, which imports CuroboPlanner from
        # planner.py. Load planner.py under an empty package shell first, put the stub
        # into it, then import the real package, which then finds the patched module.
        import importlib
        import types

        import envs  # noqa: F401

        shell = types.ModuleType("envs.robot")
        shell.__path__ = [os.path.join(root, "envs", "robot")]
        sys.modules["envs.robot"] = shell
        planner = importlib.import_module("envs.robot.planner")

        class CuroboStub:
            """Stands in for CuRobo; a policy in joint space never calls it."""

            def __init__(self, *args, **kwargs):
                self.created = True

            def plan_grippers(self, now_val, target_val):
                # Same as CuroboPlanner.plan_grippers: a linear ramp, no CUDA involved.
                num_step = 200
                return {
                    "num_step": num_step,
                    "per_step": (target_val - now_val) / num_step,
                    "result": np.linspace(now_val, target_val, num_step),
                }

            def __getattr__(self, name):
                raise RuntimeError(f"CuroboStub.{name} called: this path needs CuRobo")

        planner.CuroboPlanner = CuroboStub
        del sys.modules["envs.robot"]
        importlib.import_module("envs.robot")
    try:
        import importlib

        task_cls = getattr(importlib.import_module(f"envs.{a.task}"), a.task)
    except Exception as e:
        result["import_error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    import sapien
    from PIL import Image

    result["sapien"] = sapien.__version__
    args = load_args(root, a.task)
    runs = []
    for rep in range(2):
        t0 = time.perf_counter()
        env = task_cls()
        env.setup_demo(now_ep_num=0, seed=a.seed, is_test=True, **args)
        t_setup = time.perf_counter() - t0
        obs = env.get_obs()
        first = obs_digest(obs)
        if rep == 0:
            for cam, d in obs["observation"].items():
                if "rgb" in d:
                    Image.fromarray(np.asarray(d["rgb"])).save(out / f"{a.task}_{cam}.png")
            result["cameras"] = {c: list(np.asarray(d["rgb"]).shape) for c, d in obs["observation"].items() if "rgb" in d}
            result["joint_action_dim"] = len(obs["joint_action"]["vector"])
        q = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
        rng = np.random.default_rng(0)
        h = hashlib.sha256()
        t0 = time.perf_counter()
        err = None
        try:
            for _ in range(a.steps):
                act = q + rng.normal(0, 0.02, size=q.shape)
                env.take_action(act, action_type="qpos")
                o = env.get_obs()
                h.update(obs_digest(o).encode())
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        t_steps = time.perf_counter() - t0
        runs.append({
            "setup_seconds": t_setup,
            "first_obs_sha256": first,
            "rollout_sha256": h.hexdigest(),
            "take_action_error": err,
            "seconds_per_take_action_with_obs": t_steps / a.steps,
            "take_action_cnt": getattr(env, "take_action_cnt", None),
            "check_success": bool(env.check_success()) if err is None else None,
        })
        try:
            env.close_env()
        except Exception as e:  # noqa: BLE001
            runs[-1]["close_error"] = str(e)
    result["runs"] = runs
    result["same_seed_first_obs_identical"] = runs[0]["first_obs_sha256"] == runs[1]["first_obs_sha256"]
    result["same_seed_rollout_identical"] = runs[0]["rollout_sha256"] == runs[1]["rollout_sha256"]
    result["loadavg"] = os.getloadavg()
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
