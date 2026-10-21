"""Probe ManiSkill3's two-arm tasks on CPU physics: build, render a frame, check
same-seed determinism, time steps. Usage: python maniskill_probe.py --out DIR
"""

import torch  # noqa: I001  ManiSkill imports torch anyway

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import mani_skill.envs  # noqa: F401

TASKS = ["TwoRobotPickCube-v1", "TwoRobotStackCube-v1", "UnitreeG1TransportBox-v1"]


def flat(x, h):
    if isinstance(x, dict):
        for k in sorted(x):
            h.update(k.encode())
            flat(x[k], h)
    elif isinstance(x, torch.Tensor):
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(x, np.ndarray):
        h.update(np.ascontiguousarray(x).tobytes())


def run(task, seed, steps):
    env = gym.make(task, obs_mode="rgb", sim_backend="physx_cpu", render_mode="rgb_array",
                   num_envs=1)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(0)  # a Dict space for the two-robot tasks
    h = hashlib.sha256()
    flat(obs, h)
    t0 = time.perf_counter()
    for _ in range(steps):
        a = env.action_space.sample()
        obs, *_ = env.step(a)
        flat(obs, h)
    rate = steps / (time.perf_counter() - t0)
    frame = env.render()
    env.close()
    return h.hexdigest(), rate, frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import mani_skill
    import sapien

    res = {"mani_skill": mani_skill.__version__, "sapien": sapien.__version__, "tasks": {}}
    for t in TASKS:
        try:
            h1, r1, frame = run(t, 3, a.steps)
            h2, r2, _ = run(t, 3, a.steps)
            f = frame[0] if frame.ndim == 4 else frame
            f = f.cpu().numpy() if isinstance(f, torch.Tensor) else f
            Image.fromarray(np.asarray(f, dtype=np.uint8)).save(out / f"{t}.png")
            res["tasks"][t] = {"same_seed_identical": h1 == h2, "steps_per_second_rgb_obs": [r1, r2]}
        except Exception as e:  # noqa: BLE001
            res["tasks"][t] = {"error": f"{type(e).__name__}: {e}"}
        print(t, res["tasks"][t], flush=True)
    res["loadavg"] = os.getloadavg()
    (out / "result.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
