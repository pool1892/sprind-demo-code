"""Run a LeRobot ACT checkpoint on a gym-aloha task, one episode at a time, and log
success, timings and per-episode hashes of every observation and action.

Usage (from the probe venv, MUJOCO_GL=egl):
    python aloha_eval.py --ckpt DIR --task transfer_cube --seeds 1000:1050 --out DIR
        [--threads 4] [--video-seeds 1000,1001] [--repeat]

--repeat runs each seed twice in the same process, with a different seed in between,
to see whether anything carries over from one episode to the next.

Seeding follows lerobot-eval: env.reset(seed=s), seeds counting up from 1000.
Episodes stop at success (reward 4) or after 400 steps (LeRobot's AlohaEnv default).
"""

import torch  # noqa: I001  torch before mujoco

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_aloha  # noqa: F401  registers the environments
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

TASK_IDS = {
    "transfer_cube": "gym_aloha/AlohaTransferCube-v0",
    "insertion": "gym_aloha/AlohaInsertion-v0",
}
MAX_STEPS = 400


def parse_seeds(text: str) -> list[int]:
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in text.split(",") if s]


def batched(obs: dict) -> dict:
    """Add the batch dimension a vector env would add."""
    return {
        "pixels": {k: v[None] for k, v in obs["pixels"].items()},
        "agent_pos": obs["agent_pos"][None],
    }


def run_episode(env, policy, pre, post, seed: int, record: bool) -> dict:
    policy.reset()
    t_ep = time.perf_counter()
    obs, info = env.reset(seed=seed)
    h_obs = hashlib.sha256()
    h_act = hashlib.sha256()
    first_obs = hashlib.sha256(obs["pixels"]["top"].tobytes() + obs["agent_pos"].tobytes()).hexdigest()
    frames = [obs["pixels"]["top"].copy()] if record else None
    t_policy = 0.0
    t_env = 0.0
    model_calls = []
    success = False
    max_reward = 0.0
    steps = 0
    for step in range(MAX_STEPS):
        h_obs.update(obs["pixels"]["top"].tobytes())
        h_obs.update(obs["agent_pos"].tobytes())
        t0 = time.perf_counter()
        will_call = len(policy._action_queue) == 0
        batch = preprocess_observation(batched(obs))
        batch["task"] = [""]
        batch = pre(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = post(action)
        action_np = action.to("cpu").numpy()[0]
        dt = time.perf_counter() - t0
        t_policy += dt
        if will_call:
            model_calls.append(dt)
        h_act.update(action_np.astype(np.float32).tobytes())
        t1 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action_np)
        t_env += time.perf_counter() - t1
        steps = step + 1
        max_reward = max(max_reward, float(reward))
        if record:
            frames.append(obs["pixels"]["top"].copy())
        if info.get("is_success", False):
            success = True
        if terminated or truncated:
            break
    physics = env.unwrapped._env.physics
    state = np.concatenate([physics.data.qpos, physics.data.qvel]).astype(np.float64)
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "max_reward": max_reward,
        "seconds": time.perf_counter() - t_ep,
        "seconds_policy": t_policy,
        "seconds_env": t_env,
        "model_call_seconds": model_calls,
        "first_obs_sha256": first_obs,
        "obs_sha256": h_obs.hexdigest(),
        "act_sha256": h_act.hexdigest(),
        "final_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASK_IDS))
    ap.add_argument("--seeds", default="1000:1050")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--video-seeds", default="")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument(
        "--top-only",
        action="store_true",
        help="render only the 'top' camera, the one the policy sees (gym-aloha renders three)",
    )
    args = ap.parse_args()

    if args.top_only:
        import collections

        from gym_aloha.tasks import sim

        def top_only_observation(self, physics):
            obs = collections.OrderedDict()
            obs["qpos"] = self.get_qpos(physics)
            obs["qvel"] = self.get_qvel(physics)
            obs["env_state"] = self.get_env_state(physics)
            obs["images"] = {"top": physics.render(height=480, width=640, camera_id="top")}
            return obs

        sim.BimanualViperXTask.get_observation = top_only_observation

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_seeds = set(parse_seeds(args.video_seeds)) if args.video_seeds else set()

    t0 = time.perf_counter()
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    env = gym.make(TASK_IDS[args.task], obs_type="pixels_agent_pos", max_episode_steps=MAX_STEPS)
    t_load = time.perf_counter() - t0

    import mujoco
    import dm_control

    meta = {
        "ckpt": args.ckpt,
        "task": args.task,
        "threads": args.threads,
        "top_only": args.top_only,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "dm_control": getattr(dm_control, "__version__", "?"),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "load_seconds": t_load,
        "loadavg_start": os.getloadavg(),
    }
    seeds = parse_seeds(args.seeds)
    order = []
    for s in seeds:
        order.append(s)
        if args.repeat:
            order.extend([s + 10_000, s])
    with open(out / "episodes.jsonl", "w") as f:
        for s in order:
            rec = run_episode(env, policy, pre, post, s, record=s in video_seeds)
            frames = rec.pop("frames")
            if frames is not None:
                import imageio.v2 as imageio

                imageio.mimwrite(out / f"seed{s}.mp4", frames, fps=50, codec="libx264", quality=7)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"seed {s}: success={rec['success']} steps={rec['steps']} "
                f"{rec['seconds']:.2f}s policy {rec['seconds_policy']:.2f}s env {rec['seconds_env']:.2f}s",
                flush=True,
            )
    meta["loadavg_end"] = os.getloadavg()
    meta["total_seconds"] = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
