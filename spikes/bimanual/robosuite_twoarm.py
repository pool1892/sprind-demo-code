"""Probe robosuite 1.5.2 two-arm tasks headless: one frame per task, same-seed
determinism, and simulation step rate.

Usage (MUJOCO_GL=egl): python robosuite_twoarm.py --out DIR [--steps 500]

For each of TwoArmLift, TwoArmHandover, TwoArmTransport (two Pandas, "parallel" and,
for Transport, "opposed" as robomimic used it):
- build the env with seed 7, reset, save a 480x640 frame from each camera listed;
- build a second env with seed 7 and compare the reset states (qpos, qvel, object
  poses) and the first rendered frame bit for bit;
- step both envs with the same seeded random actions for 200 steps and compare
  states and frames bit for bit;
- reseed one env in place to a new seed twice and compare (the per-episode reseed
  a harness would do);
- time steps per second without cameras, and with one and two 256x256 cameras.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import robosuite as suite
from PIL import Image

TASKS = [
    ("TwoArmLift", "parallel"),
    ("TwoArmHandover", "parallel"),
    ("TwoArmTransport", "opposed"),
]


def make(task, config, seed, cams=(), size=(256, 256)):
    return suite.make(
        task,
        robots=["Panda", "Panda"],
        env_configuration=config,
        has_renderer=False,
        has_offscreen_renderer=bool(cams),
        use_camera_obs=bool(cams),
        camera_names=list(cams) if cams else "agentview",
        camera_heights=size[0],
        camera_widths=size[1],
        control_freq=20,
        horizon=10_000,
        ignore_done=True,
        seed=seed,
    )


def state_hash(env):
    d = env.sim.data
    s = np.concatenate([d.qpos, d.qvel, d.act if d.act is not None else []]).astype(np.float64)
    return hashlib.sha256(s.tobytes()).hexdigest()


def obs_hash(obs):
    h = hashlib.sha256()
    for k in sorted(obs):
        v = np.asarray(obs[k])
        h.update(k.encode())
        h.update(v.tobytes())
    return h.hexdigest()


def reseed(env, seed):
    """Reset the shared generator in place, so samplers that hold it see the new seed."""
    env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


def rollout(env, n, action_seed):
    rng = np.random.default_rng(action_seed)
    low, high = env.action_spec
    hs = hashlib.sha256()
    for _ in range(n):
        obs, *_ = env.step(rng.uniform(low, high))
        hs.update(obs_hash(obs).encode())
    return hs.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import mujoco

    result = {
        "robosuite": suite.__version__,
        "mujoco": mujoco.__version__,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "loadavg_start": os.getloadavg(),
        "tasks": {},
    }
    for task, config in TASKS:
        r = {"env_configuration": config}
        cams = ("frontview", "agentview") if config == "opposed" else ("frontview", "birdview")
        # Frames.
        env = make(task, config, seed=7, cams=cams, size=(480, 640))
        obs = env.reset()
        for c in cams:
            img = obs[f"{c}_image"][::-1]  # robosuite images are upside down
            Image.fromarray(img).save(out / f"{task}_{c}.png")
        r["action_dim"] = int(env.action_spec[0].shape[0])
        r["obs_keys"] = sorted(obs.keys())
        env.close()

        # Same seed, two envs: reset state, first frame, and a 200-step rollout.
        envs = [make(task, config, seed=7, cams=("frontview",)) for _ in range(2)]
        o = [e.reset() for e in envs]
        r["same_seed_reset_state_identical"] = state_hash(envs[0]) == state_hash(envs[1])
        r["same_seed_reset_obs_identical"] = obs_hash(o[0]) == obs_hash(o[1])
        ro = [rollout(e, 200, action_seed=3) for e in envs]
        r["same_seed_200_steps_obs_identical"] = ro[0] == ro[1]
        r["same_seed_200_steps_state_identical"] = state_hash(envs[0]) == state_hash(envs[1])
        # Different seed: does the start actually change?
        e3 = make(task, config, seed=8, cams=("frontview",))
        e3.reset()
        r["different_seed_reset_state_differs"] = state_hash(e3) != state_hash(envs[0])
        e3.close()
        # In-place reseed of one env, twice, after it already ran an episode.
        a = envs[0]
        hashes = []
        for _ in range(2):
            reseed(a, 1234)
            a.reset()
            hashes.append((state_hash(a), rollout(a, 50, action_seed=5)))
        r["inplace_reseed_twice_identical"] = hashes[0] == hashes[1]
        for e in envs:
            e.close()

        # Step rate.
        rates = {}
        for label, cs in [("no_camera", ()), ("one_camera_256", ("frontview",)),
                          ("two_cameras_256", cams)]:
            e = make(task, config, seed=1, cams=cs)
            e.reset()
            rng = np.random.default_rng(0)
            low, high = e.action_spec
            for _ in range(20):
                e.step(rng.uniform(low, high))
            t0 = time.perf_counter()
            for _ in range(args.steps):
                e.step(rng.uniform(low, high))
            rates[label] = args.steps / (time.perf_counter() - t0)
            e.close()
        r["steps_per_second"] = rates
        result["tasks"][task] = r
        print(task, json.dumps(r), flush=True)
    result["loadavg_end"] = os.getloadavg()
    (out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
