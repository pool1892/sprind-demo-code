"""Run a manifest of episodes in the multi-process layout under the replay rules.

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/skeleton/fleet.py --manifest M.jsonl --out RUN_DIR [--video] [--life 16]

Layout (from spikes/throughput/fleet_proto.py): this process holds the only copy
of X-VLA on the GPU and serves action chunks; one simulation worker process per
row of the inference batch holds one LIBERO environment.

The replay rules (spikes/replay/README.md) as applied here:

1. Pinned scene: sk_common.reset_pinned seeds numpy right before every reset.
2. Per-episode noise: the server draws each row's noise from the episode's
   (noise seed, chunk number); see policy_xvla.py.
3. Fixed batch size, fixed row: every call computes exactly --rows rows. An
   episode's row is part of its identity (the manifest gives it) and the worker
   for that row is the only one that runs it. Rows with no request waiting are
   filled with a copy of a waiting row and their output is dropped.
4. One action queue per episode, in the worker; an episode that ends leaves the
   loop and nothing is reset automatically.
5. (Snapshots in mid-episode are not used here: every replay starts at a reset.)
6. One rendering context at a time: one environment per worker process; a
   worker closes its environment before it builds the next task's.

What a worker writes per episode, under RUN_DIR:

  log/steps/<id>.parquet        the per-step table a fleet would log
  log/video/<id>.scene.mp4      both cameras at 5 frames per second (--video)
  log/video/<id>.wrist.mp4
  truth/chain/<id>.npz          per-step hashes of simulator state and images,
                                full-precision robot state, commanded and
                                executed actions: what replay checks compare

and this process writes log/episodes.parquet, truth/episodes.jsonl and run.json.
Nothing under log/ carries a fault description, a seed or simulator internals.

Two things in here exist only for speed and do not touch what an episode
contains. A call costs the same however many of its rows are in use, so after
a call the server waits up to --linger seconds for the rows it has just served
before it starts the next one. And a worker writes an episode's files only
after it has sent the next request, while it would be waiting anyway.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
FRAME_EVERY = 4  # control runs at 20 steps per second, so every fourth step is 5 frames per second
VIDEO_FPS = 5
GTT_START_LIMIT_GIB = 17.0

STEP_COLUMNS = (
    ["episode_id", "step", "t_s"]
    + [f"joint_pos_{i}" for i in range(7)]
    + [f"joint_vel_{i}" for i in range(7)]
    + ["gripper_pos_0", "gripper_pos_1", "gripper_vel_0", "gripper_vel_1"]
    + ["eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw"]
    + ["cmd_x", "cmd_y", "cmd_z", "cmd_rx", "cmd_ry", "cmd_rz", "cmd_gripper"]
    + ["chunk", "chunk_step"]
)


def write_mp4(path: Path, frames: list, fps: int = VIDEO_FPS) -> None:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def worker(row: int, life: int, jobs: list[dict], req_q, conn, out_dir: str, video: bool):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    import faults as F
    import sk_common as C

    C.torch.set_num_threads(1)
    out = Path(out_dir)
    born = time.perf_counter()
    t_sum = {"build": 0.0, "reset": 0.0, "step": 0.0, "wait": 0.0, "write": 0.0}
    env, env_task = None, None
    deferred = None  # writes the previous episode's files
    for job in jobs:
        eid = job["episode_id"]
        t_ep = time.perf_counter()
        built = False
        if env_task != job["task"]:
            if env is not None:
                C.make_current(env)
                env.close()  # rule 6: never two scenes in one process
            s = time.perf_counter()
            env = C.make_env(job["task"], job.get("suite", C.SUITE))
            env._ensure_env()
            env_task = job["task"]
            built = True
            t_sum["build"] += time.perf_counter() - s
        flts = F.active(job.get("faults"))

        def before_reset(e, flts=flts):
            for f in flts:
                f.apply_cell(e)

        s = time.perf_counter()
        obs = C.reset_pinned(env, job["init"], job["scene_seed"], before_reset=before_reset)
        t_sum["reset"] += time.perf_counter() - s
        for f in flts:
            f.new_episode()
        fixtures = C.fixture_hash(env)

        proprio, state_hash, img_hash, cmd, exe, chunk_of, chunk_step_of, frames = [], [], [], [], [], [], [], []
        infer_ms = []

        def record(o, t):
            rs = o["robot_state"]
            proprio.append(
                np.concatenate(
                    [rs["joints"]["pos"], rs["joints"]["vel"], rs["gripper"]["qpos"], rs["gripper"]["qvel"],
                     rs["eef"]["pos"], rs["eef"]["quat"]]
                ).astype(np.float64)
            )
            state_hash.append(C.h64(C.mj_state(env)))
            img_hash.append([C.h64(o["pixels"][k]) for k in C.CAMERAS])
            if video and t % FRAME_EVERY == 0:
                # stored upright: LIBERO renders both cameras rotated by 180 degrees
                frames.append((t, o["pixels"]["image"][::-1, ::-1].copy(), o["pixels"]["image2"][::-1, ::-1].copy()))

        record(obs, 0)
        max_steps = job.get("max_steps", env._max_episode_steps)
        t, chunk_index, k, pending, done, success = 0, 0, 0, [], False, False
        while not done and t < max_steps:
            if not pending:
                s = time.perf_counter()
                req_q.put(("req", row, obs, env.task_description, job["noise_seed"], chunk_index))
                if deferred is not None:
                    deferred()
                    deferred = None
                pending = list(conn.recv())
                w = time.perf_counter() - s
                t_sum["wait"] += w
                infer_ms.append(1000 * w)
                chunk_index += 1
                k = 0
            a_cmd = pending.pop(0)
            a_exe = a_cmd
            for f in flts:
                a_exe = f.on_action(a_exe)
            s = time.perf_counter()
            obs, _, terminated, _, info = env.step(a_exe)
            t_sum["step"] += time.perf_counter() - s
            cmd.append(a_cmd)
            exe.append(a_exe)
            chunk_of.append(chunk_index - 1)
            chunk_step_of.append(k)
            k += 1
            t += 1
            record(obs, t)
            if terminated:
                done = True
                success = bool(info["is_success"])
        for f in flts:
            f.undo_cell(env)
        if video and frames[-1][0] != t:  # always keep the last image of the episode
            frames.append((t, obs["pixels"]["image"][::-1, ::-1].copy(), obs["pixels"]["image2"][::-1, ::-1].copy()))

        # Writing the files is put off until this worker waits for its next chunk: the inference
        # process is the bottleneck, and a worker that asks late misses a whole call.
        wall_sim = time.perf_counter() - t_ep
        def finish(job=job, eid=eid, t=t, obs=obs, proprio=proprio, cmd=cmd, exe=exe, chunk_of=chunk_of,
                   chunk_step_of=chunk_step_of, frames=frames, state_hash=state_hash, img_hash=img_hash,
                   fixtures=fixtures, success=success, chunk_index=chunk_index, infer_ms=infer_ms, built=built,
                   wall_sim=wall_sim):
            s = time.perf_counter()
            proprio_a = np.stack(proprio)
            cmd_a = np.stack(cmd).astype(np.float32)
            exe_a = np.stack(exe).astype(np.float32)
            n = t + 1  # one row per state; the last row has no command
            cols = {"episode_id": pa.array([eid] * n), "step": pa.array(np.arange(n, dtype=np.int16)),
                    "t_s": pa.array((np.arange(n) * C.CONTROL_DT).astype(np.float32))}
            for name, col in zip(STEP_COLUMNS[3:28], proprio_a.T.astype(np.float32)):
                cols[name] = pa.array(col)
            pad = np.full((1, 7), np.nan, np.float32)
            for name, col in zip(STEP_COLUMNS[28:35], np.concatenate([cmd_a, pad]).T):
                cols[name] = pa.array(col, mask=np.isnan(col))
            last = np.array([True] * t + [False])
            cols["chunk"] = pa.array(np.array(chunk_of + [0], np.int16), mask=~last)
            cols["chunk_step"] = pa.array(np.array(chunk_step_of + [0], np.int16), mask=~last)
            steps_path = out / "log" / "steps" / f"{eid}.parquet"
            pq.write_table(pa.table(cols), steps_path, compression="zstd")
            sizes = {"steps_parquet": steps_path.stat().st_size, "video_scene": 0, "video_wrist": 0}
            if video:
                for name, idx in (("scene", 1), ("wrist", 2)):
                    p = out / "log" / "video" / f"{eid}.{name}.mp4"
                    write_mp4(p, [f[idx] for f in frames])
                    sizes[f"video_{name}"] = p.stat().st_size
            chain_path = out / "truth" / "chain" / f"{eid}.npz"
            np.savez_compressed(
                chain_path, state_hash=np.array(state_hash, np.uint64), img_hash=np.array(img_hash, np.uint64),
                proprio=proprio_a, cmd=cmd_a, exe=exe_a, fixtures=np.array([fixtures], np.uint64),
            )
            sizes["truth_chain"] = chain_path.stat().st_size
            t_sum["write"] += time.perf_counter() - s
            req_q.put(("episode", {
                **job, "success": success, "n_steps": t, "chunks": chunk_index, "fixture_hash": int(fixtures),
                "final_state_hash": int(state_hash[-1]), "frames": len(frames), "bytes": sizes,
                "infer_wait_ms": [round(x, 1) for x in infer_ms], "wall_s": wall_sim,
                "env_built": built, "worker_pid": os.getpid(), "worker_life": life, "t_end": time.time(),
                "worker_pss_mib": C.pss_mib(), "worker_gpu_mib": C.gpu_mib_of(),
            }))

        deferred = finish
    if deferred is not None:
        deferred()
    if env is not None:
        C.make_current(env)
        env.close()
    req_q.put(("bye", row, {"row": row, "life": life, "episodes": len(jobs), "alive_s": time.perf_counter() - born,
                            "pss_mib_end": C.pss_mib(), **t_sum}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true", help="write MP4 of both cameras for every episode")
    ap.add_argument("--life", type=int, default=16, help="episodes after which a worker process is replaced")
    ap.add_argument("--linger", type=float, default=0.65,
                    help="seconds after a call during which the server waits for the rows it just served")
    ap.add_argument("--tasks", default=None, help="run only these tasks of the manifest, e.g. 0-4")
    args = ap.parse_args()

    t_start = time.perf_counter()
    jobs = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.tasks:
        lo_t, _, hi_t = args.tasks.partition("-")
        jobs = [j for j in jobs if int(lo_t) <= j["task"] <= int(hi_t or lo_t)]
    ids = [j["episode_id"] for j in jobs]
    assert len(set(ids)) == len(ids), "episode ids in a manifest must be unique"
    rows = {j["rows"] for j in jobs}
    assert len(rows) == 1, "one batch size per run: the batch size is part of an episode's identity"
    n_rows = rows.pop()
    assert n_rows <= 8, "at most eight simulation workers on this machine"
    by_row: dict[int, list[dict]] = {r: [] for r in range(n_rows)}
    for j in jobs:
        by_row[j["row"]].append(j)

    out = Path(args.out)
    for sub in ("log/steps", "log/video", "truth/chain"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    import sk_common as C

    gtt0, load0, avail0 = C.gtt_gib(), C.loadavg(), C.mem_available_gib()
    if gtt0 > GTT_START_LIMIT_GIB:
        print(f"GPU memory in use is {gtt0:.1f} GiB, above {GTT_START_LIMIT_GIB} GiB. Not starting.", flush=True)
        sys.exit(3)

    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    conns: dict[int, object] = {}
    procs: dict[int, object] = {}
    lives: dict[int, int] = {r: 0 for r in by_row}

    def spawn(row: int) -> bool:
        share, by_row[row] = by_row[row][: args.life], by_row[row][args.life :]
        if not share:
            return False
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        for v in THREAD_VARS:
            os.environ[v] = "1"
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(row, lives[row], share, req_q, child, str(out), args.video))
        p.start()
        for v, val in saved.items():  # the inference process keeps its own thread settings
            os.environ.pop(v, None) if val is None else os.environ.__setitem__(v, val)
        lives[row] += 1
        conns[row], procs[row] = parent, p
        return True

    alive = sum(spawn(r) for r in sorted(by_row))

    peaks = {"gtt_gib": gtt0, "pss_total_mib": 0.0, "pss_server_mib": 0.0, "pss_worker_max_mib": 0.0,
             "gpu_own_total_mib": 0.0, "gpu_server_mib": 0.0, "mem_available_min_gib": avail0, "samples": 0}
    stop = threading.Event()

    def monitor():
        while not stop.wait(1.0):
            peaks["gtt_gib"] = max(peaks["gtt_gib"], C.gtt_gib())
            peaks["mem_available_min_gib"] = min(peaks["mem_available_min_gib"], C.mem_available_gib())
            server = C.pss_mib()
            workers = [C.pss_mib(p.pid) for p in list(procs.values()) if p.is_alive()]
            gpu_server = C.gpu_mib_of()
            gpu_workers = sum(C.gpu_mib_of(p.pid) for p in list(procs.values()) if p.is_alive())
            peaks["pss_total_mib"] = max(peaks["pss_total_mib"], server + sum(workers))
            peaks["pss_server_mib"] = max(peaks["pss_server_mib"], server)
            peaks["pss_worker_max_mib"] = max([peaks["pss_worker_max_mib"], *workers])
            peaks["gpu_own_total_mib"] = max(peaks["gpu_own_total_mib"], gpu_server + gpu_workers)
            peaks["gpu_server_mib"] = max(peaks["gpu_server_mib"], gpu_server)
            peaks["samples"] += 1

    threading.Thread(target=monitor, daemon=True).start()

    import torch

    from policy_xvla import Policy

    pol = Policy("cuda")
    print(f"policy loaded in {pol.load_s:.1f} s; {len(jobs)} episodes, {n_rows} rows, GTT {gtt0:.1f} GiB, load {load0}",
          flush=True)

    episodes, byes, calls = [], [], []
    waiting: dict[int, tuple] = {}
    state = {"alive": alive}
    infer_s = prep_s = 0.0
    serve0 = last_call_end = time.perf_counter()

    def handle(m) -> None:
        if m[0] == "req":
            waiting[m[1]] = m
        elif m[0] == "episode":
            episodes.append(m[1])
            if len(episodes) % 10 == 0:
                done = sum(e["success"] for e in episodes)
                print(f"{len(episodes)}/{len(jobs)} episodes, {done} successful, "
                      f"{time.perf_counter() - t_start:.0f} s", flush=True)
        elif m[0] == "bye":
            byes.append(m[2])
            procs[m[1]].join(30)
            if not spawn(m[1]):
                state["alive"] -= 1

    while state["alive"]:
        try:
            handle(req_q.get(timeout=5))
        except queue.Empty:
            bad = {r: p.exitcode for r, p in procs.items() if p.exitcode not in (None, 0)}
            if bad:
                for p in procs.values():
                    p.kill()
                raise SystemExit(f"a worker died: {bad}")
            continue
        # A call costs the same however many rows are in use, so give the workers that were
        # served last time a moment to come back before the next call starts.
        deadline = last_call_end + args.linger
        while True:
            try:
                left = deadline - time.perf_counter()
                full = len(waiting) >= state["alive"]
                handle(req_q.get_nowait() if (full or left <= 0 or not waiting) else req_q.get(timeout=left))
            except queue.Empty:
                break
        if not waiting:
            continue
        s = time.perf_counter()
        filler = next(iter(waiting.values()))
        reqs = [waiting.get(r, filler) for r in range(n_rows)]
        batch = pol.batch([r[2] for r in reqs], [r[3] for r in reqs])
        noise = torch.stack([pol.noise(r[4], r[5]) for r in reqs])
        torch.cuda.synchronize()
        s1 = time.perf_counter()
        acts = pol.to_env_actions(pol.chunks(batch, noise))
        torch.cuda.synchronize()
        s2 = time.perf_counter()
        for r in waiting:
            conns[r].send(acts[r])
        calls.append((len(waiting), s2 - s1))
        prep_s += s1 - s
        infer_s += s2 - s1
        waiting.clear()
        last_call_end = time.perf_counter()
    serve_s = time.perf_counter() - serve0
    stop.set()
    for p in procs.values():
        p.join(30)
    total = time.perf_counter() - t_start

    # ---- outputs -------------------------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq

    order = {eid: i for i, eid in enumerate(ids)}
    episodes.sort(key=lambda e: order[e["episode_id"]])
    task_text = {}
    from lerobot.envs.libero import _get_suite

    for suite_name in {e.get("suite", C.SUITE) for e in episodes}:
        suite = _get_suite(suite_name)
        for i in range(len(suite.tasks)):
            task_text[(suite_name, i)] = suite.get_task(i).language
    visible = {
        "episode_id": [e["episode_id"] for e in episodes],
        "robot_id": [e["robot"] for e in episodes],
        "task_id": [e["task"] for e in episodes],
        "task": [task_text[(e.get("suite", C.SUITE), e["task"])] for e in episodes],
        "start_state_id": [e["init"] for e in episodes],
        "n_steps": [e["n_steps"] for e in episodes],
        "duration_s": [round(e["n_steps"] * C.CONTROL_DT, 2) for e in episodes],
        "success": [e["success"] for e in episodes],
    }
    pq.write_table(pa.table(visible), out / "log" / "episodes.parquet", compression="zstd")
    with open(out / "truth" / "episodes.jsonl", "w") as fh:
        for e in episodes:
            fh.write(json.dumps(e) + "\n")

    n = len(episodes)
    ends = sorted(e["t_end"] for e in episodes)
    lo, hi = int(0.1 * n), int(0.9 * n) - 1
    steady = 3600 * (hi - lo) / (ends[hi] - ends[lo]) if hi > lo and ends[hi] > ends[lo] else float("nan")
    steps = sum(e["n_steps"] for e in episodes)
    agg = {k: sum(b[k] for b in byes) for k in ("build", "reset", "step", "wait", "write")}
    alive_s = sum(b["alive_s"] for b in byes)
    size_keys = ("steps_parquet", "video_scene", "video_wrist", "truth_chain")
    import mujoco
    import robosuite

    import lerobot

    run = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest": str(args.manifest), "video": args.video,
        "rows": n_rows, "life": args.life, "linger_s": args.linger, "tasks": args.tasks, "episodes": n, "successes": sum(e["success"] for e in episodes),
        "total_s": total, "policy_load_s": pol.load_s, "serve_s": serve_s,
        "episodes_per_hour_overall": 3600 * n / total, "episodes_per_hour_steady": steady,
        "env_steps": steps, "mean_steps": steps / n, "mean_chunks": sum(e["chunks"] for e in episodes) / n,
        "inference_calls": len(calls), "mean_rows_filled": sum(c[0] for c in calls) / len(calls),
        "inference_s_per_call": infer_s / len(calls), "inference_s_total": infer_s, "prep_s_total": prep_s,
        "server_busy_fraction": (infer_s + prep_s) / serve_s,
        "worker_time_share": {k: v / alive_s for k, v in agg.items()},
        "worker_ms_per_step": 1000 * agg["step"] / steps,
        "worker_reset_s_mean": agg["reset"] / n,
        "worker_build_s_mean": agg["build"] / max(1, sum(e["env_built"] for e in episodes)),
        "worker_write_s_mean": agg["write"] / n,
        "env_builds": sum(e["env_built"] for e in episodes), "worker_processes": len(byes),
        "bytes_per_episode": {k: sum(e["bytes"][k] for e in episodes) / n for k in size_keys},
        "bytes_episodes_table": (out / "log" / "episodes.parquet").stat().st_size,
        "peaks": peaks, "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gtt_start_gib": gtt0, "load_start": load0, "load_end": C.loadavg(),
        "mem_available_start_gib": avail0,
        "versions": {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
                     "lerobot": lerobot.__version__, "torch": torch.__version__, "policy": "lerobot/xvla-libero"},
        "worker_lives": byes,
    }
    (out / "run.json").write_text(json.dumps(run, indent=1))
    b = run["bytes_per_episode"]
    print(f"\n{n} episodes in {total:.0f} s = {run['episodes_per_hour_overall']:.0f} per hour including start-up, "
          f"{steady:.0f} per hour in steady state; success {run['successes']}/{n}")
    print(f"mean {run['mean_steps']:.0f} steps and {run['mean_chunks']:.1f} chunks per episode; "
          f"{len(calls)} calls of {1000 * run['inference_s_per_call']:.0f} ms, {run['mean_rows_filled']:.1f} of "
          f"{n_rows} rows in use; server busy {100 * run['server_busy_fraction']:.0f}%")
    print("worker time: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in run["worker_time_share"].items())
          + f"; {run['worker_ms_per_step']:.1f} ms per step, reset {run['worker_reset_s_mean']:.2f} s")
    print(f"peaks: process memory {peaks['pss_total_mib'] / 1024:.1f} GiB (server {peaks['pss_server_mib'] / 1024:.1f}, "
          f"largest worker {peaks['pss_worker_max_mib'] / 1024:.2f}), GPU memory of these processes "
          f"{peaks['gpu_own_total_mib'] / 1024:.1f} GiB, machine-wide {peaks['gtt_gib']:.1f} GiB "
          f"(started at {gtt0:.1f}); load {load0} -> {run['load_end']}")
    print(f"bytes per episode: steps {b['steps_parquet']:.0f}, scene video {b['video_scene']:.0f}, "
          f"wrist video {b['video_wrist']:.0f}, truth chain {b['truth_chain']:.0f}")


if __name__ == "__main__":
    main()
