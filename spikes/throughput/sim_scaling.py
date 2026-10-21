"""How does simulation throughput scale with the number of worker processes?

Each worker owns one LIBERO environment (libero_spatial, two cameras) and steps
it with small random actions. All workers start stepping together and run for
the same wall-clock time. No policy is involved.

Two ways of creating the workers:

  --mode spawn   Independent processes that each build the environment through
                 LeRobot's wrapper, exactly as lerobot-eval does. This is the
                 honest per-worker memory figure. Optionally a reset every
                 --episode-steps steps, timed separately.

  --mode fork    The parent builds one environment without a renderer and forks
                 the workers, so they share the model's memory. Each worker then
                 opens its own GPU rendering context and renders both cameras
                 after every step. Same work per step, a fraction of the RAM,
                 which is what makes 16 to 32 workers fit on this machine.
                 --no-render drops the rendering to isolate the CPU part.

    tools/heavy-run spikes/throughput/run.sh sim_scaling.py --mode fork --workers 1 2 4 8 12 16 24 32
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import common

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def actions_for(seed: int, n: int = 4096):
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.3, 0.3, size=(n, 7)).astype(np.float32)
    a[:, 6] = np.sign(a[:, 6])
    return a


def pin(idx: int, mode: str):
    """cores: one worker per physical core first, then the sibling threads."""
    if mode == "none":
        return
    ncpu = os.cpu_count() or 1
    os.sched_setaffinity(0, {idx % ncpu})


def stepping_loop(step_fn, reset_fn, idx, barrier, duration, episode_steps, out, extra):
    import profile_step

    acts = actions_for(idx)
    mem_ready = common.proc_mem_mib()
    barrier.wait()
    profile_step.T.clear()
    profile_step.N.clear()
    cpu0, t0 = common.cpu_seconds(), time.perf_counter()
    steps = resets = 0
    reset_time = 0.0
    while True:
        step_fn(acts[steps % len(acts)])
        steps += 1
        if episode_steps and steps % episode_steps == 0:
            r0 = time.perf_counter()
            reset_fn(steps)
            reset_time += time.perf_counter() - r0
            resets += 1
        if time.perf_counter() - t0 >= duration:
            break
    wall = time.perf_counter() - t0
    T = profile_step.T
    out.put({
        "idx": idx, "steps": steps, "wall": wall, "cpu": common.cpu_seconds() - cpu0,
        "resets": resets, "reset_time": reset_time,
        "ms": {k: 1000 * v / steps for k, v in T.items()},
        "mem_ready": mem_ready, "mem_end": common.proc_mem_mib(),
        "threads": len(os.listdir("/proc/self/task")), **extra,
    })
    barrier.wait()  # stay alive until the parent has read system-wide memory


def spawn_worker(idx, args, barrier, out):
    pin(idx, args.pin)
    if args.torch_threads:
        import torch

        torch.set_num_threads(args.torch_threads)
    import profile_step

    profile_step.instrument()
    t0 = time.perf_counter()
    env = common.make_env(task_id=idx % 10, res=args.res, hard_reset=not args.soft, episode_index=idx)
    env.reset(seed=idx)
    build = time.perf_counter() - t0
    stepping_loop(env.step, lambda s: env.reset(seed=s), idx, barrier, args.duration, args.episode_steps, out,
                  {"build_s": build})


def fork_worker(idx, args, barrier, out, env):
    pin(idx, args.pin)
    import numpy as np

    sim = env.sim
    if args.render:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen

        # Creating the context registers it with the simulation (sim.add_render_context).
        MjRenderContextOffscreen(sim, device_id=-1)

    def step(a):
        env.step(a)
        if args.render:
            for cam in ("agentview", "robot0_eye_in_hand"):
                img = sim.render(camera_name=cam, width=args.res, height=args.res)
                np.ascontiguousarray(img[::-1])  # robosuite flips each image

    stepping_loop(step, lambda s: None, idx, barrier, args.duration, 0, out, {"build_s": 0.0})


def run_one(n: int, args) -> dict:
    out = mp.get_context("spawn").Queue() if args.mode == "spawn" else mp.get_context("fork").Queue()
    ctx = mp.get_context(args.mode)
    barrier = ctx.Barrier(n + 1)
    gtt0, avail0, load0 = common.gtt_gib(), common.mem_available_gib(), common.loadavg()

    env = None
    if args.mode == "fork":
        import numpy as np
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv

        import profile_step

        if not getattr(run_one, "_instrumented", False):
            profile_step.instrument()
            run_one._instrumented = True
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = ControlEnv(bddl_file_name=bddl, camera_heights=args.res, camera_widths=args.res, control_freq=20,
                         hard_reset=False, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                         horizon=10**9)
        env.reset()
        for _ in range(10):
            env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        procs = [ctx.Process(target=fork_worker, args=(i, args, barrier, out, env)) for i in range(n)]
    else:
        procs = [ctx.Process(target=spawn_worker, args=(i, args, barrier, out)) for i in range(n)]

    for p in procs:
        p.start()
    barrier.wait(timeout=600)  # every worker has built its environment
    gtt_ready, avail_ready = common.gtt_gib(), common.mem_available_gib()
    busy = []
    t_end = time.time() + args.duration
    while time.time() < t_end:
        busy.append(common.gpu_busy())
        time.sleep(0.5)
    res = [out.get(timeout=600) for _ in range(n)]
    gtt1, avail1 = common.gtt_gib(), common.mem_available_gib()
    barrier.wait(timeout=600)
    for p in procs:
        p.join(timeout=60)
    if env is not None:
        env.close()

    steps = sum(r["steps"] for r in res)
    step_wall = sum(r["wall"] - r["reset_time"] for r in res)
    keys = sorted({k for r in res for k in r["ms"]})
    ms = {k: sum(r["ms"].get(k, 0) * r["steps"] for r in res) / steps for k in keys}
    render_ms = sum(v for k, v in ms.items() if k.startswith("render.") and k not in ("render.draw", "render.readback"))
    n_resets = sum(r["resets"] for r in res)
    row = {
        "workers": n,
        "steps_per_s_total": sum(r["steps"] / r["wall"] for r in res),
        "steps_per_s_stepping_only": sum(r["steps"] / (r["wall"] - r["reset_time"]) for r in res),
        "ms_per_step_worker": 1000 * step_wall / steps,
        "physics_ms": ms.get("physics.step1", 0) + ms.get("physics.step2", 0),
        "controller_ms": ms.get("controller", 0),
        "render_ms": render_ms,
        "reset_s_mean": (sum(r["reset_time"] for r in res) / n_resets) if n_resets else None,
        "cpu_per_worker": sum(r["cpu"] / r["wall"] for r in res) / n,
        "pss_mib_per_worker": sum(r["mem_end"].get("Pss", 0) for r in res) / n,
        "rss_mib_per_worker": sum(r["mem_end"].get("Rss", 0) for r in res) / n,
        "gtt_gib_per_worker": (gtt_ready - gtt0) / n,
        "gtt_gib_total": max(gtt_ready, gtt1),
        "mem_available_drop_gib_per_worker": (avail0 - min(avail_ready, avail1)) / n,
        "gpu_busy_pct_mean": sum(busy) / max(len(busy), 1),
        "threads_per_worker": res[0]["threads"],
        "build_s_mean": sum(r["build_s"] for r in res) / n,
        "loadavg_before": load0,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("spawn", "fork"), default="spawn")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--res", type=int, default=common.DEFAULT_RES)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--episode-steps", type=int, default=0, help="spawn mode: reset every this many steps")
    ap.add_argument("--soft", action="store_true", help="spawn mode: hard_reset=False")
    ap.add_argument("--no-render", dest="render", action="store_false", help="fork mode: no cameras")
    ap.add_argument("--threads", type=int, default=0, help="set OMP/OpenBLAS/MKL thread variables to this")
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--pin", choices=("none", "cores"), default="none")
    ap.add_argument("--gtt-stop", type=float, default=17.0, help="do not start a run above this GTT use (GiB)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.threads:
        for v in THREAD_VARS:
            os.environ[v] = str(args.threads)
    rows = []
    for n in args.workers:
        # Predict GPU memory for this size and refuse to overshoot: from the last size if there is
        # one, otherwise from the 0.9 GiB per rendering worker measured earlier.
        renders = args.mode == "spawn" or args.render
        per = max(rows[-1]["gtt_gib_per_worker"], 0.0) if rows else (0.9 if renders else 0.0)
        if common.gtt_gib() + per * n > args.gtt_stop:
            print(f"skip {n} workers: predicted GTT {common.gtt_gib() + per * n:.1f} GiB > {args.gtt_stop}")
            continue
        row = run_one(n, args)
        rows.append(row)
        print(f"{n:3d} workers: {row['steps_per_s_stepping_only']:7.0f} steps/s "
              f"({row['ms_per_step_worker']:.2f} ms/step per worker: physics {row['physics_ms']:.2f}, "
              f"controller {row['controller_ms']:.2f}, render {row['render_ms']:.2f}) "
              f"cpu/worker {row['cpu_per_worker']:.2f}  Pss/worker {row['pss_mib_per_worker']:.0f} MiB  "
              f"GTT/worker {row['gtt_gib_per_worker']:.2f} GiB (total {row['gtt_gib_total']:.1f})  "
              f"avail drop/worker {row['mem_available_drop_gib_per_worker']:.2f} GiB  "
              f"gpu busy {row['gpu_busy_pct_mean']:.0f}%  "
              + (f"reset {row['reset_s_mean']:.2f} s  " if row["reset_s_mean"] else "")
              + f"load {row['loadavg_before']}", flush=True)
        time.sleep(2)
    name = f"sim_scaling_{args.mode}{'' if args.render else '_norender'}_{args.res}{args.tag}.json"
    print("saved", common.save(name, {"args": vars(args), "rows": rows}))


if __name__ == "__main__":
    main()
