"""Probe RoboTwin's RLinf_support branch with planner_backend=mplib (no CuRobo).

Usage: python robotwin_mplib_probe.py --root CLONE --out DIR --task handover_block
           --seeds 100000:100010 [--ee-steps 5]

For each seed, as the evaluation's expert check does: build the task (clean config,
aloha-agilex, eval mode), run the scripted expert (play_once) and record whether it
planned and succeeded, and the time. Then, on the first seed that built:
- time get_obs() alone, 10 calls;
- same seed twice more in this process, one of them after another seed's episode, and
  compare the first observation, the expert's final state and final observation;
- five end-effector actions (take_action(action_type="ee")) around the current pose,
  which go through the planner.
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

    cfg = Path(root) / "task_config"
    args = yaml.safe_load(open(cfg / "demo_clean.yml"))
    args["task_name"] = task_name
    args["task_config"] = "demo_clean"
    for k in ("rgb", "qpos", "endpose"):
        args["data_type"][k] = True
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
    args["planner_backend"] = "mplib"
    return args


def obs_digest(obs):
    h = hashlib.sha256()
    for cam in sorted(obs["observation"]):
        rgb = obs["observation"][cam].get("rgb")
        if rgb is not None:
            h.update(np.ascontiguousarray(rgb).tobytes())
    h.update(np.asarray(obs["joint_action"]["vector"], dtype=np.float64).tobytes())
    return h.hexdigest()


def state_digest(env):
    h = hashlib.sha256()
    for ent in env.scene.get_entities():
        h.update(np.concatenate([ent.pose.p, ent.pose.q]).astype(np.float64).tobytes())
    for art in env.scene.get_all_articulations():
        h.update(np.asarray(art.get_qpos(), dtype=np.float64).tobytes())
    return h.hexdigest()


def parse_seeds(text):
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in text.split(",") if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="handover_block")
    ap.add_argument("--seeds", default="100000:100010")
    ap.add_argument("--ee-steps", type=int, default=5)
    ap.add_argument(
        "--patch-mplib",
        action="store_true",
        help="apply RoboTwin's install-script change to mplib in this process only: "
        "screw planning no longer fails on collision",
    )
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = str(Path(a.root).resolve())
    os.chdir(root)
    sys.path.insert(0, root)

    import importlib

    import torch  # noqa: F401
    import sapien

    if a.patch_mplib:
        # scripts/_install.sh of RoboTwin edits mplib/planner.py with sed:
        #   "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
        # loses "or collide ". Same change, applied to the function in memory.
        import inspect
        import textwrap

        import mplib.planner as mp_planner

        src = textwrap.dedent(inspect.getsource(mp_planner.Planner.plan_screw))
        old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
        assert old in src, "mplib source differs from what RoboTwin patches"
        src = src.replace(old, "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:")
        ns = {}
        exec(compile(src, mp_planner.__file__, "exec"), mp_planner.__dict__, ns)  # noqa: S102
        mp_planner.Planner.plan_screw = ns["plan_screw"]

    from envs.utils.create_actor import UnStableError

    task_cls = getattr(importlib.import_module(f"envs.{a.task}"), a.task)
    args = load_args(root, a.task)
    res = {"task": a.task, "sapien": sapien.__version__, "planner_backend": "mplib",
           "patch_mplib": a.patch_mplib, "episodes": []}

    def episode(seed, ep_num):
        env = task_cls()
        rec = {"seed": seed}
        t0 = time.perf_counter()
        try:
            env.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
        except UnStableError as e:
            rec["unstable"] = str(e)
            return env, rec
        rec["setup_seconds"] = time.perf_counter() - t0
        rec["first_obs"] = obs_digest(env.get_obs())
        t0 = time.perf_counter()
        try:
            env.play_once()
            rec["plan_success"] = bool(env.plan_success)
            rec["check_success"] = bool(env.check_success())
        except Exception as e:  # noqa: BLE001
            rec["expert_error"] = f"{type(e).__name__}: {e}"
            rec["expert_traceback"] = traceback.format_exc()[-1500:]
        rec["expert_seconds"] = time.perf_counter() - t0
        rec["final_state"] = state_digest(env)
        rec["final_obs"] = obs_digest(env.get_obs())
        return env, rec

    first_ok = None
    for i, seed in enumerate(parse_seeds(a.seeds)):
        env, rec = episode(seed, i)
        res["episodes"].append(rec)
        print(json.dumps({k: v for k, v in rec.items() if "traceback" not in k}), flush=True)
        if first_ok is None and "setup_seconds" in rec:
            first_ok = seed
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass

    if first_ok is not None:
        # Observation time, and same-seed repeats in this process.
        env = task_cls()
        env.setup_demo(now_ep_num=0, seed=first_ok, is_test=True, **args)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            obs = env.get_obs()
            times.append(time.perf_counter() - t0)
        res["get_obs_seconds"] = times
        res["cameras"] = {c: list(np.asarray(d["rgb"]).shape) for c, d in obs["observation"].items() if "rgb" in d}
        # End-effector control through the planner.
        ee = []
        try:
            ep = obs["endpose"]
            base = np.concatenate([
                np.asarray(ep["left_endpose"], dtype=np.float64), [ep["left_gripper"]],
                np.asarray(ep["right_endpose"], dtype=np.float64), [ep["right_gripper"]],
            ])
            for k in range(a.ee_steps):
                act = base.copy()
                act[2] += 0.01 * (k + 1)  # left hand up 1 cm per step
                act[10] += 0.01 * (k + 1)  # right hand up
                t0 = time.perf_counter()
                env.take_action(act, action_type="ee")
                ee.append({"seconds": time.perf_counter() - t0,
                           "left_z": float(env.get_obs()["endpose"]["left_endpose"][2])})
            res["ee_control"] = {"base_left_z": float(base[2]), "steps": ee}
        except Exception as e:  # noqa: BLE001
            res["ee_control"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:], "steps": ee}
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
        repeats = []
        for s in [first_ok, first_ok + 1, first_ok]:
            env, rec = episode(s, 0)
            repeats.append(rec)
            try:
                env.close_env()
            except Exception:  # noqa: BLE001
                pass
        res["repeats"] = repeats
        base = next(e for e in res["episodes"] if e["seed"] == first_ok)
        keys = ["first_obs", "final_state", "final_obs", "plan_success", "check_success"]
        res["same_seed_identical_vs_first"] = [
            {k: r.get(k) == base.get(k) for k in keys} for r in (repeats[0], repeats[2])
        ]
    res["loadavg"] = os.getloadavg()
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=2)[:4000])


if __name__ == "__main__":
    main()
