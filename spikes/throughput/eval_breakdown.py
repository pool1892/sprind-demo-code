"""Where does lerobot-eval spend its time today?

This repeats LeRobot's own rollout loop (lerobot/scripts/lerobot_eval.py,
rollout) for one batch of environments of one task, with a timer around each
stage. Same calls, same order, same defaults as the known 570 episodes/hour
run: ten environments stepped one after another in one process, 360x360
images, a video frame collected at every step.

    tools/heavy-run spikes/throughput/run.sh eval_breakdown.py --tasks 0 1
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

import common
import policy_setup as ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[0])
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--amp", default="fp32")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads and BLAS threads; 0 = defaults")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    import profile_step

    profile_step.instrument()  # splits env.step into physics, controller and rendering
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation

    cpu0, wall0 = common.cpu_seconds(), time.perf_counter()
    pol = ps.Policy("cuda")
    policy = pol.policy
    pol.env_cfg.task_ids = list(args.tasks)
    t0 = time.perf_counter()
    envs = make_env(pol.env_cfg, n_envs=args.n_envs, use_async_envs=False)["libero_spatial"]
    t_make = time.perf_counter() - t0

    # Count real inference calls: select_action only runs the model when its action queue is empty.
    infer = {"n": 0, "t": 0.0}
    orig_chunk = policy._get_action_chunk

    def timed_chunk(*a, **k):
        ps.sync("cuda")
        s = time.perf_counter()
        out = orig_chunk(*a, **k)
        ps.sync("cuda")
        infer["n"] += 1
        infer["t"] += time.perf_counter() - s
        return out

    policy._get_action_chunk = timed_chunk

    results = []
    for task_id, env in envs.items():
        T = defaultdict(float)
        infer["n"], infer["t"] = 0, 0.0
        policy.reset()
        s = time.perf_counter()
        observation, info = env.reset(seed=list(range(args.seed, args.seed + args.n_envs)),
                                      options={NEW_ROLLOUT_OPTION: True})
        T["reset (builds the environments on first use)"] = time.perf_counter() - s
        max_steps = env.call("_max_episode_steps")[0]
        done = np.array([False] * env.num_envs)
        frames, step, active_env_steps = [], 0, 0
        success = np.array([False] * env.num_envs)
        loop0 = time.perf_counter()
        profile_step.T.clear()
        profile_step.N.clear()
        while not np.all(done) and step < max_steps:
            s = time.perf_counter()
            observation = preprocess_observation(observation)
            T["obs to tensors (preprocess_observation)"] += time.perf_counter() - s

            s = time.perf_counter()
            observation["task"] = list(env.call("task_description"))
            observation = pol.env_pre(observation)
            T["LIBERO obs processing (flip images, build state)"] += time.perf_counter() - s

            s = time.perf_counter()
            observation = pol.pre(observation)
            ps.sync("cuda")
            T["policy preprocessing (tokenize, copy to GPU, normalize)"] += time.perf_counter() - s

            s = time.perf_counter()
            before = infer["t"]
            with torch.inference_mode(), ps.autocast(args.amp):
                action = policy.select_action(observation)
            ps.sync("cuda")
            T["select_action outside the model (queue handling)"] += time.perf_counter() - s - (infer["t"] - before)

            s = time.perf_counter()
            action = pol.post(action)
            action = pol.env_post({"action": action})["action"]
            action_numpy = action.to("cpu").numpy()
            T["postprocessing (unnormalize, copy to CPU)"] += time.perf_counter() - s

            s = time.perf_counter()
            active_env_steps += int((~done).sum())
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            T["env.step for the whole batch"] += time.perf_counter() - s

            if not args.no_video:
                s = time.perf_counter()
                frames.append(np.stack([env.envs[i].render() for i in range(env.num_envs)]))
                T["video frame collection"] += time.perf_counter() - s

            # Same extraction as lerobot_eval.rollout.
            if "final_info" in info and isinstance(info["final_info"], dict):
                success |= np.asarray(info["final_info"].get("is_success", [False] * env.num_envs), dtype=bool)
            elif "is_success" in info:
                success |= np.asarray(info["is_success"], dtype=bool)
            done = terminated | truncated | done
            step += 1
        loop = time.perf_counter() - loop0
        T["model inference (action chunks)"] = infer["t"]
        accounted = sum(v for k, v in T.items() if not k.startswith("reset"))
        T["loop bookkeeping (unaccounted)"] = loop - accounted
        total = loop + T["reset (builds the environments on first use)"]
        P = profile_step.T
        n_sim = max(profile_step.N.get("wrapper.step", 0), 1)
        sim_ms = {
            "simulated steps": n_sim,
            "resets inside the loop (autoreset of finished environments)": profile_step.N.get("wrapper.reset", 0),
            "seconds in those resets": P.get("wrapper.reset", 0),
            "whole step": 1000 * P.get("wrapper.step", 0) / n_sim,
            "physics": 1000 * (P.get("physics.step1", 0) + P.get("physics.step2", 0)) / n_sim,
            "controller": 1000 * P.get("controller", 0) / n_sim,
            "rendering": 1000 * sum(v for k, v in P.items() if k.startswith("render.")
                                    and k not in ("render.draw", "render.readback")) / n_sim,
        }
        r = {"task": task_id, "n_envs": env.num_envs, "loop_iterations": step, "active_env_steps": active_env_steps,
             "inference_calls": infer["n"], "inference_s_per_call": infer["t"] / max(infer["n"], 1),
             "successes": int(success.sum()), "total_s": total, "stages_s": dict(T), "sim_ms_per_step": sim_ms,
             "episodes_per_hour": 3600 * env.num_envs / total, "loadavg": common.loadavg()}
        results.append(r)
        print(f"\n== task {task_id}: {env.num_envs} episodes in {total:.1f} s = {r['episodes_per_hour']:.0f} episodes/hour; "
              f"{step} loop iterations, {active_env_steps} environment steps needed, "
              f"{infer['n']} inference calls ({1000 * r['inference_s_per_call']:.0f} ms each), "
              f"{r['successes']} successes; load {r['loadavg']}")
        for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
            print(f"   {k:58s} {v:7.2f} s  {100 * v / total:5.1f}%   {1000 * v / max(step, 1):7.2f} ms per loop iteration")
        print(f"   inside env.step: {n_sim} environment steps were simulated, {active_env_steps} were needed "
              f"(up to each episode's end); {sim_ms['resets inside the loop (autoreset of finished environments)']} "
              f"resets of finished environments took {sim_ms['seconds in those resets']:.1f} s; per simulated step: "
              + ", ".join(f"{k} {sim_ms[k]:.2f} ms" for k in ("whole step", "physics", "controller", "rendering")))
        env.close()
    wall = time.perf_counter() - wall0
    summary = {"make_env_s": t_make, "policy_load_s": pol.load_s, "wall_s": wall,
               "cpu_cores_mean": (common.cpu_seconds() - cpu0) / wall, "pss_mib": common.proc_mem_mib().get("Pss"),
               "gtt_gib": common.gtt_gib()}
    print(f"\npolicy load {pol.load_s:.1f} s, process used {summary['cpu_cores_mean']:.2f} CPU cores on average, "
          f"Pss {summary['pss_mib']:.0f} MiB, GTT {summary['gtt_gib']:.2f} GiB")
    print("saved", common.save(f"eval_breakdown{'_novideo' if args.no_video else ''}{args.tag}.json",
                               {"args": vars(args), "results": results, "summary": summary}))


if __name__ == "__main__":
    main()
