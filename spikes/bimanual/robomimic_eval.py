"""Run a robomimic checkpoint (here: the Transport diffusion policy) against robosuite
on the CPU, one episode per seed, and log success, timings and hashes.

Usage (MUJOCO_GL=egl):
    python robomimic_eval.py --ckpt FILE --seeds 0:20 --out DIR [--threads 8]
        [--video-seeds 0,1] [--horizon 600] [--ddim 0]

Per episode, the robosuite generator is reset in place to the seed (so the object
placement follows the seed) and torch's global generator is seeded with the seed (the
diffusion sampler draws its noise from it). --ddim N switches the sampler to DDIM with N
steps, a speed variant the checkpoint was not evaluated with.
"""

import torch  # noqa: I001  torch before mujoco

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def parse_seeds(text):
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in text.split(",") if s]


def obs_bytes(obs):
    h = hashlib.sha256()
    for k in sorted(obs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(obs[k]).tobytes())
    return h.digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="0:20")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--video-seeds", default="")
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--ddim", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_seeds = set(parse_seeds(args.video_seeds)) if args.video_seeds else set()

    t0 = time.perf_counter()
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.ckpt, device=device, verbose=False)
    algo = policy.policy
    if args.ddim:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        algo.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True,
            set_alpha_to_one=True, steps_offset=0, prediction_type="epsilon",
        )
        algo.algo_config.unlock()
        algo.algo_config.ddpm.enabled = False
        algo.algo_config.ddim.enabled = True
        algo.algo_config.ddim.num_inference_timesteps = args.ddim
        algo.algo_config.lock()
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=bool(video_seeds), verbose=False
    )
    t_load = time.perf_counter() - t0
    rs_env = env  # unwrap robomimic's frame stacking and env wrapper down to robosuite
    while not hasattr(rs_env, "rng"):
        rs_env = rs_env.env

    import mujoco
    import robosuite

    meta = {
        "ckpt": args.ckpt,
        "threads": args.threads,
        "ddim": args.ddim,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "env_name": env.name,
        "env_kwargs": {k: v for k, v in ckpt_dict["env_metadata"]["env_kwargs"].items()
                       if isinstance(v, (int, float, str, bool, list))},
        "load_seconds": t_load,
        "loadavg_start": os.getloadavg(),
    }
    print(json.dumps(meta), flush=True)

    with open(out / "episodes.jsonl", "w") as f:
        for seed in parse_seeds(args.seeds):
            rs_env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
            torch.manual_seed(seed)
            np.random.seed(seed)
            policy.start_episode()
            t_ep = time.perf_counter()
            obs = env.reset()
            state0 = env.get_state()["states"]
            h_obs, h_act = hashlib.sha256(), hashlib.sha256()
            frames = [] if seed in video_seeds else None
            calls, t_pol, t_env, success, steps = [], 0.0, 0.0, False, 0
            for step in range(args.horizon):
                h_obs.update(obs_bytes(obs))
                will_call = algo.action_queue is None or len(algo.action_queue) == 0
                t1 = time.perf_counter()
                act = policy(ob=obs)
                dt = time.perf_counter() - t1
                t_pol += dt
                if will_call:
                    calls.append(dt)
                h_act.update(np.asarray(act, dtype=np.float64).tobytes())
                t1 = time.perf_counter()
                obs, r, done, _ = env.step(act)
                t_env += time.perf_counter() - t1
                steps = step + 1
                if frames is not None and step % 2 == 0:
                    frames.append(env.render(mode="rgb_array", height=360, width=640, camera_name="frontview"))
                if env.is_success()["task"]:
                    success = True
                    break
            final = env.get_state()["states"]
            rec = {
                "seed": seed, "success": success, "steps": steps,
                "seconds": time.perf_counter() - t_ep, "seconds_policy": t_pol, "seconds_env": t_env,
                "model_call_seconds": calls,
                "first_state_sha256": hashlib.sha256(np.asarray(state0).tobytes()).hexdigest(),
                "obs_sha256": h_obs.hexdigest(), "act_sha256": h_act.hexdigest(),
                "final_state_sha256": hashlib.sha256(np.asarray(final).tobytes()).hexdigest(),
            }
            if frames:
                import imageio.v2 as imageio

                imageio.mimwrite(out / f"seed{seed}.mp4", frames, fps=10, codec="libx264", quality=7)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"seed {seed}: success={success} steps={steps} {rec['seconds']:.1f}s "
                  f"policy {t_pol:.1f}s ({len(calls)} calls) env {t_env:.1f}s", flush=True)
    meta["loadavg_end"] = os.getloadavg()
    meta["total_seconds"] = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
