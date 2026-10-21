"""Probe BiGym headless: build one task with a head camera, save a frame, check
same-seed determinism and time steps. Usage (MUJOCO_GL=egl): python bigym_probe.py --out DIR
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bigym.action_modes import JointPositionActionMode
from bigym.envs.pick_and_place import PickBox
from bigym.utils.observation_config import CameraConfig, ObservationConfig


def make():
    return PickBox(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[CameraConfig(name="head", rgb=True, depth=False, resolution=(256, 256))]
        ),
        render_mode=None,
    )


def digest(obs):
    h = hashlib.sha256()
    for k in sorted(obs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(obs[k]).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import bigym
    import mujoco

    res = {"bigym": bigym.__version__, "mujoco": mujoco.__version__, "task": "PickBox"}
    hashes = []
    for rep in range(2):
        env = make()
        obs, _ = env.reset(seed=11)
        if rep == 0:
            first11 = digest(obs)
            Image.fromarray(np.moveaxis(obs["rgb_head"], 0, -1)).save(out / "PickBox_head.png")
            res["action_dim"] = int(env.action_space.shape[0])
            res["obs_keys"] = sorted(obs)
        h = hashlib.sha256(digest(obs).encode())
        rng = np.random.default_rng(0)
        low, high = env.action_space.low, env.action_space.high
        mid = (low + high) / 2
        t0 = time.perf_counter()
        for _ in range(a.steps):
            obs, *_ = env.step(mid + 0.05 * (high - low) * rng.uniform(-1, 1, size=low.shape))
            h.update(digest(obs).encode())
        res[f"steps_per_second_rep{rep}"] = a.steps / (time.perf_counter() - t0)
        hashes.append(h.hexdigest())
        env.close()
    env = make()
    obs8, _ = env.reset(seed=12)
    res["different_seed_first_obs_differs"] = digest(obs8) != first11
    res["same_seed_rollout_identical"] = hashes[0] == hashes[1]
    res["loadavg"] = os.getloadavg()
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
