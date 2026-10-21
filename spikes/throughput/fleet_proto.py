"""Prototype of the recommended layout, measured end to end.

N simulation workers, each a separate process with one LIBERO environment, and
one inference process (this one) that owns the only copy of the policy on the
GPU. A worker sends its observation when it needs new actions and waits; the
server takes every request that is waiting at that moment, runs them as one
batch and returns an action chunk to each worker. Workers finish an episode as
soon as the task succeeds, and move on to the next one on their own.

The workload is lerobot-eval's: for each task of libero_spatial, episodes
0..9 with seeds 1000..1009 and LIBERO's stored initial states 0..9.

    tools/heavy-run spikes/throughput/run.sh fleet_proto.py --workers 8 --rounds 2
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import statistics
import time

import common

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def worker(idx, args, jobs, req_q, conn, res_q):
    import numpy as np

    if args.torch_threads:
        import torch

        torch.set_num_threads(args.torch_threads)
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS

    max_steps = TASK_SUITE_MAX_STEPS[args.suite]
    if args.trim:
        import ctypes
        import gc

        libc = ctypes.CDLL("libc.so.6")
    t = {"build": 0.0, "reset": 0.0, "step": 0.0, "wait": 0.0}
    cpu0, born = common.cpu_seconds(), time.perf_counter()
    peak_pss = 0.0
    while True:
        try:
            task_id, episodes = jobs.get_nowait()
        except queue.Empty:
            break
        s = time.perf_counter()
        env = common.make_env(task_id=task_id, res=args.res, suite_name=args.suite, hard_reset=not args.soft)
        first = True
        for ep in episodes:
            s_reset = time.perf_counter()
            obs, info = env.reset(seed=args.seed + ep)
            if args.trim:
                # A hard reset rebuilds model and renderer; the old ones are only freed by a
                # garbage-collector pass, and the allocator keeps the pages unless told to return them.
                gc.collect()
                libc.malloc_trim(0)
            if first:  # the first reset builds the simulator
                t["build"] += time.perf_counter() - s
                first = False
            else:
                t["reset"] += time.perf_counter() - s_reset
            peak_pss = max(peak_pss, common.proc_mem_mib().get("Pss", 0.0))
            ep_t0 = time.perf_counter()
            steps, calls, success, done = 0, 0, False, False
            while not done:
                s_wait = time.perf_counter()
                req_q.put((idx, obs, env.task_description))
                chunk = conn.recv()
                t["wait"] += time.perf_counter() - s_wait
                calls += 1
                s_step = time.perf_counter()
                for a in chunk[: args.n_action_steps]:
                    obs, _, terminated, _, info = env.step(a)
                    steps += 1
                    if terminated or steps >= max_steps:
                        success = bool(info.get("is_success", False))
                        done = True
                        break
                t["step"] += time.perf_counter() - s_step
            res_q.put(("episode", {"worker": idx, "task": task_id, "episode": ep, "success": success,
                                   "steps": steps, "calls": calls, "t_end": time.time(),
                                   "episode_s": time.perf_counter() - ep_t0}))
        peak_pss = max(peak_pss, common.proc_mem_mib().get("Pss", 0.0))
        env.close()
    res_q.put(("worker", {"worker": idx, **t, "alive_s": time.perf_counter() - born,
                          "cpu_s": common.cpu_seconds() - cpu0, "peak_pss_mib": peak_pss}))
    req_q.put((idx, None, None))  # tell the server this worker is finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1, help="how many times to run all tasks x episodes")
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--res", type=int, default=common.DEFAULT_RES)
    ap.add_argument("--amp", default="fp32")
    ap.add_argument("--n-action-steps", type=int, default=50, help="actions executed from each chunk")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--linger-ms", type=float, default=0.0, help="wait this long for more requests before a call")
    ap.add_argument("--soft", action="store_true", help="hard_reset=False in the workers")
    ap.add_argument("--trim", action="store_true", help="collect garbage and return freed memory after each reset")
    ap.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP threads per worker (0 = library default)")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--server-threads", type=int, default=0, help="torch CPU threads in the inference process")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    jobs, req_q, res_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    n_jobs = 0
    for _ in range(args.rounds):
        for task_id in args.tasks:
            jobs.put((task_id, list(range(args.episodes))))
            n_jobs += 1
    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    if args.threads:
        for v in THREAD_VARS:
            os.environ[v] = str(args.threads)
    conns, procs = [], []
    avail0, gtt0, load0 = common.mem_available_gib(), common.gtt_gib(), common.loadavg()
    for i in range(args.workers):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(i, args, jobs, req_q, child, res_q))
        p.start()
        conns.append(parent)
        procs.append(p)
    for v, val in saved.items():  # the inference process keeps its own thread settings
        if val is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = val

    import numpy as np
    import torch

    import policy_setup as ps

    if args.server_threads:
        torch.set_num_threads(args.server_threads)
    pol = ps.Policy("cuda")
    t_loaded = time.perf_counter()
    print(f"policy loaded {pol.load_s:.1f} s after {t_loaded - t_start:.1f} s; {args.workers} workers starting; "
          f"load {load0}", flush=True)

    alive = args.workers
    batches, infer_s, pre_s, post_s = [], 0.0, 0.0, 0.0
    gtt_peak, avail_min = common.gtt_gib(), common.mem_available_gib()
    cpu0, serve0 = common.cpu_seconds(), time.perf_counter()
    while alive:
        try:
            reqs = [req_q.get(timeout=5)]
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in procs):
                raise SystemExit(f"a worker died: {[p.exitcode for p in procs]}")
            continue
        if args.linger_ms:
            time.sleep(args.linger_ms / 1000)
        while len(reqs) < args.max_batch:
            try:
                reqs.append(req_q.get_nowait())
            except queue.Empty:
                break
        finished = [r for r in reqs if r[1] is None]
        alive -= len(finished)
        reqs = [r for r in reqs if r[1] is not None]
        if not reqs:
            continue
        s = time.perf_counter()
        batch = pol.preprocess(ps.stack_obs([r[1] for r in reqs]), [r[2] for r in reqs])
        ps.sync("cuda")
        s1 = time.perf_counter()
        out = pol.chunks(batch, args.amp)
        ps.sync("cuda")
        s2 = time.perf_counter()
        acts = pol.to_env_actions(out)
        for r, a in zip(reqs, acts):
            conns[r[0]].send(a)
        s3 = time.perf_counter()
        pre_s += s1 - s
        infer_s += s2 - s1
        post_s += s3 - s2
        batches.append(len(reqs))
        gtt_peak = max(gtt_peak, common.gtt_gib())
        avail_min = min(avail_min, common.mem_available_gib())
    serve = time.perf_counter() - serve0
    server_cpu = (common.cpu_seconds() - cpu0) / serve
    server_pss = common.proc_mem_mib().get("Pss", 0.0)

    episodes, workers = [], []
    while len(workers) < args.workers:
        kind, payload = res_q.get(timeout=60)
        (episodes if kind == "episode" else workers).append(payload)
    for p in procs:
        p.join(60)
    total = time.perf_counter() - t_start

    n = len(episodes)
    steps = sum(e["steps"] for e in episodes)
    ends = sorted(e["t_end"] for e in episodes)
    # Steady state: the middle 80% of episode completions, which leaves out start-up and the tail
    # where some workers have already run out of jobs.
    lo, hi = int(0.1 * n), int(0.9 * n) - 1
    steady = 3600 * (hi - lo) / (ends[hi] - ends[lo]) if hi > lo and ends[hi] > ends[lo] else float("nan")
    by_task: dict[int, list[bool]] = {}
    for e in episodes:
        by_task.setdefault(e["task"], []).append(e["success"])
    agg = {k: sum(w[k] for w in workers) for k in ("build", "reset", "step", "wait")}
    alive_s = sum(w["alive_s"] for w in workers)
    out = {
        "args": vars(args), "episodes": n, "successes": sum(e["success"] for e in episodes),
        "success_by_task": {k: sum(v) for k, v in sorted(by_task.items())},
        "total_s": total, "episodes_per_hour_overall": 3600 * n / total,
        "episodes_per_hour_steady": steady,
        "env_steps": steps, "env_steps_per_s_overall": steps / total,
        "mean_steps_per_episode": steps / n, "mean_calls_per_episode": sum(e["calls"] for e in episodes) / n,
        "inference_calls": len(batches), "mean_batch": statistics.mean(batches), "max_batch": max(batches),
        "inference_s_total": infer_s, "inference_s_per_call": infer_s / len(batches),
        "inference_ms_per_env": 1000 * infer_s / sum(batches),
        "preprocess_s_total": pre_s, "postprocess_and_send_s_total": post_s,
        "server_busy_fraction": (infer_s + pre_s + post_s) / serve, "server_cpu_cores": server_cpu,
        "server_pss_mib": server_pss,
        "worker_time_share": {k: v / alive_s for k, v in agg.items()},
        "worker_ms_per_step": 1000 * agg["step"] / steps,
        "worker_reset_s_mean": agg["reset"] / max(n - n_jobs, 1),
        "worker_build_s_mean": agg["build"] / n_jobs,
        "worker_cpu_cores_mean": sum(w["cpu_s"] for w in workers) / alive_s,
        "worker_peak_pss_mib_mean": statistics.mean(w["peak_pss_mib"] for w in workers),
        "worker_peak_pss_mib_max": max(w["peak_pss_mib"] for w in workers),
        "gtt_base_gib": gtt0, "gtt_peak_gib": gtt_peak,
        "mem_available_before_gib": avail0, "mem_available_min_gib": avail_min,
        "loadavg_before": load0, "loadavg_after": common.loadavg(),
    }
    print(f"\n{n} episodes in {total:.1f} s including start-up = {out['episodes_per_hour_overall']:.0f} episodes/hour; "
          f"steady state {steady:.0f} episodes/hour")
    print(f"success {out['successes']}/{n}, by task {out['success_by_task']}")
    print(f"mean {out['mean_steps_per_episode']:.0f} steps and {out['mean_calls_per_episode']:.1f} inference requests "
          f"per episode; {steps / total:.0f} env steps/s overall")
    print(f"inference: {len(batches)} calls, mean batch {out['mean_batch']:.1f} (max {out['max_batch']}), "
          f"{1000 * out['inference_s_per_call']:.0f} ms per call, {out['inference_ms_per_env']:.1f} ms per env; "
          f"server busy {100 * out['server_busy_fraction']:.0f}% of the time, {server_cpu:.2f} CPU cores, "
          f"Pss {server_pss:.0f} MiB")
    print("worker time: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in out["worker_time_share"].items())
          + f"; {out['worker_ms_per_step']:.2f} ms per step, reset {out['worker_reset_s_mean']:.2f} s, "
          f"build {out['worker_build_s_mean']:.2f} s, {out['worker_cpu_cores_mean']:.2f} CPU cores per worker")
    print(f"memory: worker peak Pss mean {out['worker_peak_pss_mib_mean']:.0f} MiB (max {out['worker_peak_pss_mib_max']:.0f}), "
          f"GTT {gtt0:.2f} -> peak {gtt_peak:.2f} GiB, MemAvailable {avail0:.1f} -> min {avail_min:.1f} GiB; "
          f"load before {load0.split()[0]}, after {out['loadavg_after'].split()[0]}")
    name = f"fleet_proto_w{args.workers}_{args.amp}_k{args.n_action_steps}{'_soft' if args.soft else ''}{'_trim' if args.trim else ''}{f'_st{args.server_threads}' if args.server_threads else ''}{args.tag}.json"
    out["episode_list"] = episodes
    out["worker_list"] = workers
    print("saved", common.save(name, out))


if __name__ == "__main__":
    main()
