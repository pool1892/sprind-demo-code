"""spikes/xvla-speed/fleet.py with one addition: the policy keeps acting after LIBERO's check.

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python spikes/success/fleet_tail.py \
        --manifest M.jsonl --out RUN_DIR --rows 4 --workers 10 --policy fp16_res32 --success-tail 60

Copied from spikes/xvla-speed/fleet.py as of commit a4c7d73. Every change is marked
`success-tail:`. With `--success-tail N` an episode does not end when LIBERO's check
first holds: the policy goes on for N more control steps (or to the step limit), and
the episode record gets `libero_success_step`, the step at which the check first held.
`success` stays LIBERO's verdict (the check held at some step), so the run's own
summary is comparable with the original runs; the stricter definitions are scored
afterwards from the replayed states (spikes/success/replay_policy.py, analyze.py
--policy-run). Everything else, including the replay rules, is the original's. The
original's description follows.

Run a manifest of episodes in the multi-process layout, with free rows and more workers than rows.

Copied from spikes/skeleton/fleet.py and changed in three places.

1. Rows are free. The skeleton tied every episode to one row of the inference
   batch and every row to one worker, because for SmolVLA a chunk depends on its
   row. For X-VLA as run here it does not (rows_check.py), so a request takes any
   free row: the server serves the requests that have waited longest, up to
   `--rows` of them per call. The batch size stays fixed, because a chunk does
   depend on it; rows nobody asked for are filled with a copy of row 0 and their
   output is dropped. An episode's identity therefore still contains the batch
   size (`rows`) and no longer a row.
2. More workers than rows (`--workers`). While a call runs, the workers that are
   not in it simulate; when it ends, their requests are already waiting, so the
   next call starts full and at once.
3. Work is handed out, not pre-assigned. A worker asks for its next episode and
   gets one of the task it has built, if any is left, and otherwise one of the
   task with the most episodes left per worker on it. A small thread answers
   these requests so that a worker never waits for a running inference call.

The replay rules (spikes/replay/README.md) otherwise as in the skeleton: scene
pinned by seeding numpy before every reset; each chunk's starting noise drawn on
the CPU from (noise seed, chunk number); one action queue per episode; one
environment per worker process, closed before the next task's is built. Soft
resets; a worker process is replaced after `--life` episodes, and the first
lives are staggered so that the workers are not all replaced at once.

`--model smolvla` runs the second policy through the same loop. Its chunks do
depend on the row (rows_check.py --model smolvla), so with free rows its runs
are a throughput measurement, not replayable episodes.

`--policy` picks the inference settings (POLICIES below) or takes Policy options
as JSON. Outputs are the skeleton's: log/ (what a fleet would log), truth/
(identity, fault set, per-step hash chain) and run.json.
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

# success-tail: the original's helper modules (xs_common, faults, policy_fast, ...) live there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "xvla-speed"))

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
FRAME_EVERY = 4  # control runs at 20 steps per second, so every fourth step is 5 frames per second
VIDEO_FPS = 5
GTT_START_LIMIT_GIB = 17.0
GTT_ABORT_GIB = 21.0
MEM_AVAILABLE_FLOOR_GIB = 8.0
WORKER_GIB = 2.1  # process memory of one worker late in its life (skeleton: 1.76 to 2.1)
MAX_WORKERS = 12

HALF = {"transformer": "fp16", "vision": "fp16", "encoder": "fp16"}
POLICIES = {
    # today's arithmetic, bit for bit (variants.py: fp32_pinned), with the convolution choice pinned
    "fp32": {"hoist": True, "fast_post": True, "benchmark": False},
    # the whole model in float16, weights stored transposed so that a row's position does not matter
    "fp16": {"half": HALF, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False},
    # recommended (README): as fp16, but in the action transformer only the attention and MLP sub-layers
    # compute in float16; each token's running sum and the layer norms stay in 32 bits
    "fp16_res32": {"half": HALF, "residual32": True, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False},
    "bf16": {"half": {k: "bf16" for k in HALF}, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False},
}
# The second policy (--model smolvla): lerobot/smolvla_libero as the baseline spike corrected it, 10 actions per chunk.
SMOLVLA_POLICIES = {
    "fp32": {"n_action_steps": 10},
    "fp32_lt": {"n_action_steps": 10, "linear_t": True},
    "fp16": {"n_action_steps": 10, "half": "fp16", "linear_t": True},
}

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


def worker(wid: int, life: int, req_q, job_q, conn, out_dir: str, video: bool):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    import faults as F
    import xs_common as C

    C.torch.set_num_threads(1)
    out = Path(out_dir)
    born = time.perf_counter()
    t_sum = {"build": 0.0, "reset": 0.0, "step": 0.0, "wait": 0.0, "write": 0.0, "ask": 0.0}
    env, env_task = None, None
    deferred = None  # writes the previous episode's files
    n_done = 0
    while True:
        s = time.perf_counter()
        job_q.put((wid, env_task))
        job = conn.recv()
        t_sum["ask"] += time.perf_counter() - s
        if job is None:
            break
        eid = job["episode_id"]
        t_ep = time.perf_counter()
        built = False
        task_key = (job.get("suite", C.SUITE), job["task"])
        if env_task != task_key:
            if env is not None:
                C.make_current(env)
                env.close()  # rule 6: never two scenes in one process
            s = time.perf_counter()
            env = C.make_env(job["task"], job.get("suite", C.SUITE), control_mode=job.get("control_mode", "absolute"))
            env._ensure_env()
            env_task = task_key
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
        success_tail = job.get("success_tail", 0)  # success-tail
        libero_step = None  # success-tail
        while not done and t < max_steps:
            if not pending:
                s = time.perf_counter()
                req_q.put(("req", wid, obs, env.task_description, job["noise_seed"], chunk_index))
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
            # success-tail: note where LIBERO's check first held and keep going for success_tail steps.
            if libero_step is None and info["is_success"]:
                libero_step = t
                success = True
            if terminated and not (success_tail and info["is_success"]):
                done = True
                success = success or bool(info["is_success"])
            if libero_step is not None and success_tail and t >= libero_step + success_tail:
                done = True
        for f in flts:
            f.undo_cell(env)
        if video and frames[-1][0] != t:  # always keep the last image of the episode
            frames.append((t, obs["pixels"]["image"][::-1, ::-1].copy(), obs["pixels"]["image2"][::-1, ::-1].copy()))
        n_done += 1

        # Writing the files is put off until this worker waits for its next chunk.
        wall_sim = time.perf_counter() - t_ep
        def finish(job=job, eid=eid, t=t, proprio=proprio, cmd=cmd, exe=exe, chunk_of=chunk_of,
                   chunk_step_of=chunk_step_of, frames=frames, state_hash=state_hash, img_hash=img_hash,
                   fixtures=fixtures, success=success, chunk_index=chunk_index, infer_ms=infer_ms, built=built,
                   wall_sim=wall_sim, n_done=n_done, libero_step=libero_step):  # success-tail: libero_step
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
                "libero_success_step": libero_step,  # success-tail
                "final_state_hash": int(state_hash[-1]), "frames": len(frames), "bytes": sizes,
                "infer_wait_ms": [round(x, 1) for x in infer_ms], "wall_s": wall_sim,
                "env_built": built, "worker": wid, "worker_pid": os.getpid(), "worker_life": life,
                "worker_episode": n_done, "t_end": time.time(),
                "worker_pss_mib": C.pss_mib(), "worker_gpu_mib": C.gpu_mib_of(),
            }))

        deferred = finish
    if deferred is not None:
        deferred()
    if env is not None:
        C.make_current(env)
        env.close()
    req_q.put(("bye", wid, {"worker": wid, "life": life, "episodes": n_done, "alive_s": time.perf_counter() - born,
                            "pss_mib_end": C.pss_mib(), **t_sum}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, required=True, help="size of every inference batch")
    ap.add_argument("--workers", type=int, required=True, help="simulation worker processes")
    ap.add_argument("--model", default="xvla", choices=["xvla", "smolvla"])
    ap.add_argument("--policy", default="fp16_res32", help=f"one of {sorted(POLICIES)} or Policy options as JSON")
    ap.add_argument("--steps", type=int, default=None, help="sampling steps (changes the policy's behaviour)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="actions executed per chunk (changes behaviour)")
    ap.add_argument("--views", type=int, default=None, help="3: without LeRobot's surplus empty image view (changes behaviour)")
    ap.add_argument("--video", action="store_true", help="write MP4 of both cameras for every episode")
    ap.add_argument("--life", type=int, default=16, help="episodes after which a worker process is replaced")
    ap.add_argument("--no-stagger", action="store_true", help="give every worker the full first life")
    ap.add_argument("--linger", type=float, default=0.25,
                    help="seconds after a call during which the server waits for a full batch")
    ap.add_argument("--server-threads", type=int, default=4, help="PyTorch CPU threads of the inference process")
    ap.add_argument("--tasks", default=None, help="run only these tasks of the manifest, e.g. 0-4")
    ap.add_argument("--limit", type=int, default=None, help="run only the first so many episodes")
    ap.add_argument("--success-tail", type=int, default=0,  # success-tail
                    help="control steps the policy keeps acting after LIBERO's check first holds")
    args = ap.parse_args()

    t_start = time.perf_counter()
    jobs = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.tasks:
        lo_t, _, hi_t = args.tasks.partition("-")
        jobs = [j for j in jobs if int(lo_t) <= j["task"] <= int(hi_t or lo_t)]
    if args.limit:
        jobs = jobs[: args.limit]
    ids = [j["episode_id"] for j in jobs]
    assert len(set(ids)) == len(ids), "episode ids in a manifest must be unique"
    n_rows, n_workers = args.rows, min(args.workers, len(jobs))
    assert 1 <= n_rows <= 16 and 1 <= n_workers <= MAX_WORKERS, "at most twelve simulation workers on this machine"
    named = POLICIES if args.model == "xvla" else SMOLVLA_POLICIES
    policy_opts = dict(named[args.policy]) if args.policy in named else json.loads(args.policy)
    if args.steps:
        policy_opts["steps"] = args.steps
    if args.n_action_steps:
        policy_opts["n_action_steps"] = args.n_action_steps
    if args.views:
        policy_opts["views"] = args.views
    policy_tag = json.dumps(policy_opts, sort_keys=True) if args.model == "xvla" else json.dumps(
        {"model": args.model, **policy_opts}, sort_keys=True)
    for j in jobs:  # the batch size and the inference settings belong to an episode's identity
        j["rows"] = n_rows
        j["policy"] = policy_tag
        if args.success_tail:  # success-tail
            j["success_tail"] = args.success_tail
        j.pop("row", None)
        if args.model == "smolvla":
            j["control_mode"] = "relative"

    out = Path(args.out)
    for sub in ("log/steps", "log/video", "truth/chain"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    import xs_common as C

    gtt0, load0, avail0 = C.gtt_gib(), C.loadavg(), C.mem_available_gib()
    if gtt0 > GTT_START_LIMIT_GIB:
        print(f"GPU memory in use is {gtt0:.1f} GiB, above {GTT_START_LIMIT_GIB} GiB. Not starting.", flush=True)
        sys.exit(3)
    need = n_workers * WORKER_GIB + 4.0
    if avail0 - need < MEM_AVAILABLE_FLOOR_GIB:
        print(f"{avail0:.1f} GiB of memory available; {n_workers} workers and the policy need about {need:.0f} GiB "
              f"and {MEM_AVAILABLE_FLOOR_GIB:.0f} GiB must stay free. Not starting.", flush=True)
        sys.exit(3)

    ctx = mp.get_context("spawn")
    req_q, job_q = ctx.Queue(), ctx.Queue()
    conns: dict[int, object] = {}
    procs: dict[int, object] = {}
    lives: dict[int, int] = {w: 0 for w in range(n_workers)}

    # ---- handing out work ---------------------------------------------------------------------
    lock = threading.Lock()
    pending = list(jobs)
    given: dict[int, int] = {w: 0 for w in range(n_workers)}      # episodes given to the current process of a slot
    on_task: dict[int, tuple | None] = {w: None for w in range(n_workers)}

    def pick(wid: int, built_task) -> dict | None:
        """Next episode for a worker, or None when its process should end (work done, or its life is over)."""
        with lock:
            if not pending or given[wid] >= life_of[wid]:
                on_task[wid] = None
                return None
            key = lambda j: (j.get("suite", C.SUITE), j["task"])  # noqa: E731
            same = next((j for j in pending if built_task is not None and key(j) == tuple(built_task)), None)
            if same is None:
                left: dict = {}
                for j in pending:
                    left[key(j)] = left.get(key(j), 0) + 1
                crowd: dict = {}
                for w, k in on_task.items():
                    if k is not None and w != wid:
                        crowd[k] = crowd.get(k, 0) + 1
                best = max(left, key=lambda k: (left[k] / (1 + crowd.get(k, 0)), -list(left).index(k)))
                same = next(j for j in pending if key(j) == best)
            pending.remove(same)
            given[wid] += 1
            on_task[wid] = key(same)
            return same

    def dispatcher():
        while True:
            m = job_q.get()
            if m is None:
                return
            wid, built_task = m
            conns[wid].send(pick(wid, built_task))

    # First lives are staggered so that the workers are not all replaced in the same few seconds.
    life_of = {w: args.life if args.no_stagger else max(1, round(args.life * (w + 1) / n_workers))
               for w in range(n_workers)}

    def spawn(wid: int) -> bool:
        with lock:
            if not pending:
                return False
            given[wid] = 0
            if lives[wid] > 0:
                life_of[wid] = args.life
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        for v in THREAD_VARS:
            os.environ[v] = "1"
        parent, child = ctx.Pipe()
        conns[wid] = parent
        p = ctx.Process(target=worker, args=(wid, lives[wid], req_q, job_q, child, str(out), args.video))
        p.start()
        for v, val in saved.items():  # the inference process keeps its own thread settings
            os.environ.pop(v, None) if val is None else os.environ.__setitem__(v, val)
        lives[wid] += 1
        procs[wid] = p
        return True

    alive = sum(spawn(w) for w in range(n_workers))
    threading.Thread(target=dispatcher, daemon=True).start()

    peaks = {"gtt_gib": gtt0, "pss_total_mib": 0.0, "pss_server_mib": 0.0, "pss_worker_max_mib": 0.0,
             "gpu_own_total_mib": 0.0, "gpu_server_mib": 0.0, "mem_available_min_gib": avail0, "samples": 0}
    stop = threading.Event()
    abort = {"why": None}

    def monitor():
        while not stop.wait(1.0):
            gtt, avail = C.gtt_gib(), C.mem_available_gib()
            peaks["gtt_gib"] = max(peaks["gtt_gib"], gtt)
            peaks["mem_available_min_gib"] = min(peaks["mem_available_min_gib"], avail)
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
            if gtt > GTT_ABORT_GIB or avail < MEM_AVAILABLE_FLOOR_GIB - 2:
                abort["why"] = f"GPU memory {gtt:.1f} GiB, memory available {avail:.1f} GiB"

    threading.Thread(target=monitor, daemon=True).start()

    import torch

    if args.model == "smolvla":
        from policy_smolvla import Policy
    else:
        from policy_fast import Policy

    torch.set_num_threads(args.server_threads)
    suites = sorted({j.get("suite", C.SUITE) for j in jobs})
    pol = Policy("cuda", suite=suites[0] if len(suites) == 1 else C.SUITE, opts=policy_opts)
    print(f"policy loaded in {pol.load_s:.1f} s ({args.policy}); {len(jobs)} episodes, {n_rows} rows, {n_workers} workers, "
          f"GTT {gtt0:.1f} GiB, {avail0:.0f} GiB available, load {load0}", flush=True)

    episodes, byes, calls = [], [], []
    waiting: dict[int, tuple] = {}  # worker -> request, in order of arrival
    state = {"alive": alive}
    infer_s = prep_s = 0.0
    serve0 = last_call_end = time.perf_counter()

    def handle(m) -> None:
        if m[0] == "req":
            waiting[m[1]] = m
        elif m[0] == "episode":
            episodes.append(m[1])
            if len(episodes) % 20 == 0:
                done = sum(e["success"] for e in episodes)
                print(f"{len(episodes)}/{len(jobs)} episodes, {done} successful, "
                      f"{time.perf_counter() - t_start:.0f} s", flush=True)
        elif m[0] == "bye":
            byes.append(m[2])
            procs[m[1]].join(30)
            if not spawn(m[1]):
                state["alive"] -= 1

    while state["alive"]:
        if abort["why"]:
            for p in procs.values():
                p.kill()
            raise SystemExit(f"stopped to protect the machine: {abort['why']}")
        try:
            handle(req_q.get(timeout=5))
        except queue.Empty:
            bad = {w: p.exitcode for w, p in procs.items() if p.exitcode not in (None, 0)}
            if bad:
                for p in procs.values():
                    p.kill()
                raise SystemExit(f"a worker died: {bad}")
            continue
        # A call costs the same however many rows are in use, so wait a moment for a full batch.
        deadline = last_call_end + args.linger
        while True:
            try:
                left = deadline - time.perf_counter()
                full = len(waiting) >= min(n_rows, state["alive"])
                handle(req_q.get_nowait() if (full or left <= 0 or not waiting) else req_q.get(timeout=left))
            except queue.Empty:
                break
        if not waiting:
            continue
        s = time.perf_counter()
        served = list(waiting)[:n_rows]  # the requests that have waited longest
        reqs = [waiting[w] for w in served]
        reqs += [reqs[0]] * (n_rows - len(reqs))
        batch = pol.batch([r[2] for r in reqs], [r[3] for r in reqs])
        noise = torch.stack([pol.noise(r[4], r[5]) for r in reqs])
        torch.cuda.synchronize()
        s1 = time.perf_counter()
        acts = pol.to_env_actions(pol.chunks_staged(batch, noise))
        torch.cuda.synchronize()
        s2 = time.perf_counter()
        for i, w in enumerate(served):
            conns[w].send(acts[i])
            del waiting[w]
        calls.append((len(served), s2 - s1, s1 - s, s - serve0, len(waiting)))
        prep_s += s1 - s
        infer_s += s2 - s1
        last_call_end = time.perf_counter()
    serve_s = time.perf_counter() - serve0
    stop.set()
    job_q.put(None)
    for p in procs.values():
        p.join(30)
    total = time.perf_counter() - t_start

    # ---- outputs -------------------------------------------------------------
    import numpy as np
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
    agg = {k: sum(b[k] for b in byes) for k in ("build", "reset", "step", "wait", "write", "ask")}
    alive_s = sum(b["alive_s"] for b in byes)
    size_keys = ("steps_parquet", "video_scene", "video_wrist", "truth_chain")
    filled = np.array([c[0] for c in calls])
    call_s = np.array([c[1] for c in calls])
    # the middle of the run, where every worker has work: calls that started between the 10% and 90% marks
    t_calls = np.array([c[3] for c in calls])
    mid = (t_calls >= 0.1 * serve_s) & (t_calls <= 0.9 * serve_s)
    import mujoco
    import robosuite

    import lerobot

    run = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest": str(args.manifest), "video": args.video,
        "rows": n_rows, "workers": n_workers, "model": args.model, "policy": args.policy, "policy_opts": policy_opts,
        "life": args.life, "staggered_first_lives": not args.no_stagger, "linger_s": args.linger, "server_threads": args.server_threads, "tasks": args.tasks,
        "episodes": n, "successes": sum(e["success"] for e in episodes),
        "total_s": total, "policy_load_s": pol.load_s, "serve_s": serve_s,
        "episodes_per_hour_overall": 3600 * n / total, "episodes_per_hour_steady": steady,
        "env_steps": steps, "mean_steps": steps / n, "mean_chunks": sum(e["chunks"] for e in episodes) / n,
        "inference_calls": len(calls), "mean_rows_filled": float(filled.mean()),
        "share_of_rows_in_use": float(filled.sum() / (len(calls) * n_rows)),
        "share_of_rows_in_use_middle": float(filled[mid].sum() / (mid.sum() * n_rows)) if mid.any() else float("nan"),
        "calls_by_rows_filled": {int(k): int((filled == k).sum()) for k in range(1, n_rows + 1)},
        "mean_requests_left_waiting_after_a_call": float(np.mean([c[4] for c in calls])),
        "inference_s_per_call": infer_s / len(calls), "inference_s_per_call_median": float(np.median(call_s)),
        "inference_s_total": infer_s, "prep_s_total": prep_s, "prep_ms_per_call": 1000 * prep_s / len(calls),
        "server_busy_fraction": (infer_s + prep_s) / serve_s,
        "worker_time_share": {k: v / alive_s for k, v in agg.items()},
        "worker_ms_per_step": 1000 * agg["step"] / steps,
        "worker_reset_s_mean": agg["reset"] / n,
        "worker_build_s_mean": agg["build"] / max(1, sum(e["env_built"] for e in episodes)),
        "worker_write_s_mean": agg["write"] / n,
        "chunk_wait_ms_median": float(np.median([w for e in episodes for w in e["infer_wait_ms"]])),
        "env_builds": sum(e["env_built"] for e in episodes), "worker_processes": len(byes),
        "bytes_per_episode": {k: sum(e["bytes"][k] for e in episodes) / n for k in size_keys},
        "bytes_episodes_table": (out / "log" / "episodes.parquet").stat().st_size,
        "peaks": peaks, "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gtt_start_gib": gtt0, "load_start": load0, "load_end": C.loadavg(),
        "mem_available_start_gib": avail0, "miopen_db": C.MIOPEN_DB,
        "versions": {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
                     "lerobot": lerobot.__version__, "torch": torch.__version__,
                     "policy": "lerobot/xvla-libero" if args.model == "xvla" else "lerobot/smolvla_libero"},
        "worker_lives": byes,
        # per call: seconds since the server started serving, rows in use, call seconds, preparation
        # seconds, requests left waiting when it ended
        "calls": [[round(c[3], 3), c[0], round(c[1], 4), round(c[2], 4), c[4]] for c in calls],
        "idle_s_between_calls": float(serve_s - infer_s - prep_s),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print(f"\n{n} episodes in {total:.0f} s = {run['episodes_per_hour_overall']:.0f} per hour including start-up, "
          f"{steady:.0f} per hour in steady state; success {run['successes']}/{n}")
    print(f"mean {run['mean_steps']:.0f} steps and {run['mean_chunks']:.1f} chunks per episode; "
          f"{len(calls)} calls of {1000 * run['inference_s_per_call']:.0f} ms plus {run['prep_ms_per_call']:.0f} ms of "
          f"preparation, {100 * run['share_of_rows_in_use']:.0f}% of {n_rows} rows in use "
          f"({100 * run['share_of_rows_in_use_middle']:.0f}% in the middle of the run); server busy "
          f"{100 * run['server_busy_fraction']:.0f}%")
    print("worker time: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in run["worker_time_share"].items())
          + f"; {run['worker_ms_per_step']:.1f} ms per step, reset {run['worker_reset_s_mean']:.2f} s, "
          f"{run['env_builds']} environment builds, {len(byes)} worker processes")
    print(f"peaks: process memory {peaks['pss_total_mib'] / 1024:.1f} GiB (server {peaks['pss_server_mib'] / 1024:.1f}, "
          f"largest worker {peaks['pss_worker_max_mib'] / 1024:.2f}), GPU memory of these processes "
          f"{peaks['gpu_own_total_mib'] / 1024:.1f} GiB, machine-wide {peaks['gtt_gib']:.1f} GiB "
          f"(started at {gtt0:.1f}), least memory available {peaks['mem_available_min_gib']:.1f} GiB; "
          f"load {load0} -> {run['load_end']}")


if __name__ == "__main__":
    main()
