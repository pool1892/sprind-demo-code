"""Do rendering and inference slow each other down on the shared GPU?

K simulation workers (forked, each with its own rendering context, as in
sim_scaling.py --mode fork) and one policy in this process. Four phases of
equal length, back to back so they see the same machine load:

  1. simulation only          -> environment steps per second
  2. inference only           -> milliseconds per call at the given batch size
  3. both at the same time    -> both numbers again
  4. simulation only, again   -> shows drift from other jobs on the machine
  5. inference only, again
  6. everything idle          -> what a loaded but waiting policy process costs

    tools/heavy-run spikes/throughput/run.sh contention.py --workers 8 --batch 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time

import common
from rocm_idle import thread_cpu


def sim_worker(idx, env, res, run, quit_, counters, render):
    import numpy as np

    from sim_scaling import actions_for

    sim = env.sim
    if render:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen

        MjRenderContextOffscreen(sim, device_id=-1)
    acts = actions_for(idx)
    n = 0
    while not quit_.value:
        if not run.value:
            time.sleep(0.005)
            continue
        env.step(acts[n % len(acts)])
        if render:
            for cam in ("agentview", "robot0_eye_in_hand"):
                np.ascontiguousarray(sim.render(camera_name=cam, width=res, height=res)[::-1])
        n += 1
        counters[idx] = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--amp", default="fp32")
    ap.add_argument("--res", type=int, default=common.DEFAULT_RES)
    ap.add_argument("--phase", type=float, default=10.0)
    ap.add_argument("--no-render", dest="render", action="store_false")
    args = ap.parse_args()

    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    import policy_setup as ps

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    t = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    env = ControlEnv(bddl_file_name=bddl, camera_heights=args.res, camera_widths=args.res, control_freq=20,
                     hard_reset=False, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                     horizon=10**9)
    env.reset()
    ctx = mp.get_context("fork")
    run, quit_ = ctx.Value("i", 0), ctx.Value("i", 0)
    counters = ctx.Array("q", args.workers, lock=False)
    procs = [ctx.Process(target=sim_worker, args=(i, env, args.res, run, quit_, counters, args.render))
             for i in range(args.workers)]
    for p in procs:
        p.start()

    # Only now does this process touch the GPU: a GL or ROCm context must not be inherited by a fork.
    obs, task = ps.sample_obs(args.res)
    pol = ps.Policy("cuda")
    batch = pol.preprocess(ps.stack_obs([obs] * args.batch), [task] * args.batch)
    for _ in range(2):
        pol.chunks(batch, args.amp)
    ps.sync("cuda")
    time.sleep(3)  # workers have their rendering contexts by now

    def phase(sim: bool, infer: bool) -> dict:
        run.value = int(sim)
        time.sleep(0.5)
        c0, t0 = sum(counters), time.perf_counter()
        cpu0 = common.cpu_seconds()
        thr0 = thread_cpu(os.getpid())
        calls, busy = [], []
        while time.perf_counter() - t0 < args.phase:
            if infer:
                s = time.perf_counter()
                pol.chunks(batch, args.amp)
                ps.sync("cuda")
                calls.append(time.perf_counter() - s)
            else:
                time.sleep(0.25)
            busy.append(common.gpu_busy())
        wall = time.perf_counter() - t0
        thr1 = thread_cpu(os.getpid())
        hot = {n: round((thr1[n] - thr0.get(n, 0.0)) / wall, 2) for n in thr1}
        hot = dict(sorted(((n, v) for n, v in hot.items() if v > 0.02), key=lambda kv: -kv[1])[:4])
        return {"sim": sim, "infer": infer, "busiest_threads_cores": hot, "steps_per_s": (sum(counters) - c0) / wall,
                "call_ms": 1000 * statistics.median(calls) if calls else None, "calls": len(calls),
                "gpu_busy_pct": statistics.mean(busy), "server_cpu_cores": (common.cpu_seconds() - cpu0) / wall,
                "gtt_gib": common.gtt_gib(), "loadavg": common.loadavg()}

    rows = [phase(True, False), phase(False, True), phase(True, True), phase(True, False), phase(False, True),
            phase(False, False)]
    quit_.value = 1
    for p in procs:
        p.join(30)
    for r in rows:
        what = {(True, False): "simulation only", (False, True): "inference only", (True, True): "both",
                (False, False): "all idle"}[(r["sim"], r["infer"])]
        print(f"{what:16s}: {r['steps_per_s']:7.0f} env steps/s; "
              + (f"{r['call_ms']:.0f} ms per call ({r['calls']} calls); " if r["call_ms"] else "no inference; ")
              + f"GPU busy {r['gpu_busy_pct']:.0f}%, inference process {r['server_cpu_cores']:.2f} cores, "
              f"GTT {r['gtt_gib']:.2f} GiB, load {r['loadavg'].split()[0]}; threads {r['busiest_threads_cores']}")
    print("saved", common.save(f"contention_w{args.workers}_b{args.batch}_{args.amp}{'' if args.render else '_norender'}.json",
                               {"args": vars(args), "rows": rows}))


if __name__ == "__main__":
    main()
