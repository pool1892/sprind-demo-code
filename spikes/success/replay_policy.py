"""Replay recorded policy episodes without the policy and read the state at LIBERO's check.

    run.sh --baseline LOG replay_policy.py --run RUN_DIR [--run RUN_DIR ...] --name NAME
                                           [--only-success] [--tail 40] [--workers 4]

The fleet runs of spikes/skeleton and spikes/xvla-speed log the robot's telemetry but not
the objects. They do keep, per episode, the start (task, initial state, scene seed) and
every action as it reached the simulator (`truth/chain/*.npz`, key `exe`), plus a hash of
MuJoCo's full state after every step. Stepping the same environment with the same actions
reproduces the episode without the policy, and the hashes say whether it did, bit for bit,
at every step. Every replayed state is read with defs.features.

After the last recorded step (where LIBERO's check ended the episode) one branch runs on
for `tail` steps: `open_now`, the arm keeps its last target and the gripper opens. That
is what happens if the policy simply let go at that moment; it is not what the policy
would have done, which needs the policy (see the README).

Runs in the environment the episodes were recorded in (baseline-venv-mj337, `--baseline`
in run.sh); the set-up below is copied from spikes/skeleton/sk_common.py.
Output: ~/.cache/robots-probes/success/replay/NAME/{summary.jsonl, <run>__<episode>.npz}.
"""

from __future__ import annotations

import os
from pathlib import Path

PROBES = Path.home() / ".cache" / "robots-probes"
OUT = PROBES / "success"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROBES / "baseline" / "libero-config-mj337"))
os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402, F401, I001  (torch before mujoco)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
RES = 360
_patched = False


def patch_absolute_world() -> None:
    """Absolute commands in world coordinates, as in the recorded runs (sk_common.py)."""
    global _patched
    if _patched:
        return
    import lerobot.envs.libero as ll

    orig = ll._set_delta

    def _set_delta(robot, use_delta: bool) -> None:
        orig(robot, use_delta)
        ctrl = ll._arm_controller(robot)
        if not use_delta and hasattr(ctrl, "input_ref_frame"):
            ctrl.input_ref_frame = "world"

    ll._set_delta = _set_delta
    _patched = True


def make_env(task_id: int, suite_name: str, control_mode: str):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    if control_mode == "absolute":
        patch_absolute_world()
    return LiberoEnv(
        task_suite=_get_suite(suite_name), task_id=task_id, task_suite_name=suite_name,
        obs_type="pixels_agent_pos", observation_height=RES, observation_width=RES,
        episode_index=0, n_envs=1, control_mode=control_mode, hard_reset=False,
    )


def make_current(env) -> None:
    if getattr(env, "_env", None) is not None:
        ctx = getattr(env._env.env.sim, "_render_context_offscreen", None)
        if ctx is not None:
            ctx.gl_ctx.make_current()


def reset_pinned(env, init: int, scene_seed: int):
    env._ensure_env()
    make_current(env)
    rs = env._env.env
    if rs.hard_reset and hasattr(rs, "object_property_initializers"):
        rs.object_property_initializers.clear()
    np.random.seed(scene_seed)
    env.init_state_id = init
    obs, _ = env.reset(seed=scene_seed)
    return obs


def h64(env) -> int:
    sim = env._env.env.sim
    m, d = sim.model._model, sim.data._data
    out = np.empty(mujoco.mj_stateSize(m, SPEC), np.float64)
    mujoco.mj_getState(m, d, out, SPEC)
    return int(np.frombuffer(hashlib.sha256(out.tobytes()).digest()[:8], np.uint64)[0])


TAIL_BRANCHES = ("open_now", "open_lift")


def tail_action(kind: str, i: int, last: np.ndarray, absolute: bool) -> np.ndarray:
    """One action of a scripted tail after the last recorded step.

    open_now:  the arm keeps its last target, the gripper opens.
    open_lift: the same for 10 steps, then the target rises by 1 cm per step up to 10 cm.
    """
    a = last.copy() if absolute else np.zeros(7)
    a[6] = -1.0
    if kind == "open_lift" and i >= 10:
        if absolute:
            a[2] += min(0.10, 0.01 * (i - 9))
        else:
            a[2] = 0.2 if i < 20 else 0.0  # 1 cm per step for 10 steps in relative units
    return a


def replay_one(env, ep: dict, chain, tail: int, frames: dict | None = None):
    """Replay one episode, then each tail branch from the state at its last step."""
    import defs

    exe = chain["exe"]
    hashes = chain["state_hash"]
    absolute = ep.get("control_mode", "absolute") == "absolute"
    last = np.asarray(exe[-1], np.float64)
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {"n_steps": len(exe)}
    for branch in ("recorded",) + (TAIL_BRANCHES if tail > 0 else ()):
        obs = reset_pinned(env, ep["init"], ep["scene_seed"])
        e = env._env.env
        ix = defs.build_index(e)
        rows: list[dict] = []
        shots: list = []

        def rec():
            f = defs.features(e.sim.model._model, e.sim.data._data, ix)
            f["libero_env"] = float(bool(env._env.check_success()))
            rows.append(f)

        def grab(o):
            if frames is not None:
                shots.append(np.concatenate([o["pixels"]["image"][::-1, ::-1], o["pixels"]["image2"][::-1, ::-1]], axis=1))

        if branch == "recorded":
            match = [h64(env) == int(hashes[0])]
            rec()
            grab(obs)
            for t, a in enumerate(exe):
                obs, *_ = env.step(np.asarray(a))
                match.append(h64(env) == int(hashes[t + 1]))
                rec()
                grab(obs)
            match_a = np.array(match)
            summary["hash_match_all"] = bool(match_a.all())
            summary["hash_first_mismatch"] = int(np.argmin(match_a)) if not match_a.all() else None
            prefix = ""
        else:
            for a in exe:
                obs, *_ = env.step(np.asarray(a))
            summary[f"hash_ok_{branch}"] = h64(env) == int(hashes[-1])
            rec()
            grab(obs)
            for i in range(tail):
                obs, *_ = env.step(tail_action(branch, i, last, absolute))
                rec()
                grab(obs)
            prefix = branch + "/"
        for k in rows[0]:
            arrays[prefix + k] = np.array([r[k] for r in rows], np.float64)
        if frames is not None:
            frames[branch] = shots
    lib, lib_env = arrays["libero"], arrays["libero_env"]
    summary["replica_mismatch"] = int(np.sum(lib != lib_env)) + sum(
        int(np.sum(arrays[f"{b}/libero"] != arrays[f"{b}/libero_env"])) for b in TAIL_BRANCHES if f"{b}/libero" in arrays)
    summary["libero_at_end"] = bool(lib_env[-1] > 0.5)
    summary["s_l"] = int(np.argmax(lib_env > 0.5)) if (lib_env > 0.5).any() else None
    return summary, arrays, len(lib)


def worker(job: dict) -> list[dict]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    torch.set_num_threads(1)
    out_dir = Path(job["out"])
    rows, env, env_key = [], None, None
    for ep in job["episodes"]:
        t0 = time.time()
        key = (ep["suite"], ep["task"], ep.get("control_mode", "absolute"))
        row = {k: ep.get(k) for k in ("run", "episode_id", "suite", "task", "init", "scene_seed", "success", "faults", "variant",
                                      "libero_success_step")}
        row["control_mode"] = key[2]
        try:
            if env_key != key:
                if env is not None:
                    make_current(env)
                    env.close()
                env = make_env(ep["task"], ep["suite"], key[2])
                env._ensure_env()
                env_key = key
            chain = np.load(ep["chain"])
            summary, arrays, n_rec = replay_one(env, ep, chain, job["tail"])
            row.update(summary)
            np.savez_compressed(out_dir / f"{ep['run']}__{ep['episode_id']}.npz", n_rec=n_rec, **arrays)
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
    if env is not None:
        make_current(env)
        env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="a run directory with truth/episodes.jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="at most this many episodes per run (0: all)")
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-job", type=int, default=12)
    args = ap.parse_args()
    assert args.workers <= 4, "a light job runs at most four simulator processes"

    out_dir = OUT / "replay" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists already; choose another name")
    out_dir.mkdir(parents=True)
    episodes = []
    for run in args.run:
        run_dir = Path(run).expanduser()
        eps = [json.loads(line) for line in open(run_dir / "truth" / "episodes.jsonl")]
        if args.only_success:
            eps = [e for e in eps if e["success"]]
        if args.limit:
            eps = eps[: args.limit]
        for e in eps:
            e["run"] = run_dir.name
            e["chain"] = str(run_dir / "truth" / "chain" / f"{e['episode_id']}.npz")
            e.setdefault("variant", "")
        episodes.extend(eps)
    episodes.sort(key=lambda e: (e["suite"], e["task"], e.get("control_mode", "absolute"), e["run"], e["episode_id"]))
    jobs = []
    for k in range(0, len(episodes), args.per_job):
        jobs.append(dict(episodes=episodes[k : k + args.per_job], tail=args.tail, out=str(out_dir)))
    meta = dict(vars(args), when=time.strftime("%F %T"), episodes=len(episodes), jobs=len(jobs))
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool, open(out_dir / "summary.jsonl", "w") as f:
        for part in pool.imap_unordered(worker, jobs, chunksize=1):
            for r in part:
                f.write(json.dumps(r, default=str) + "\n")
                n += 1
            f.flush()
            print(f"{n}/{len(episodes)} episodes, {time.time() - t0:.0f} s", flush=True)
    meta["wall_s"] = round(time.time() - t0, 1)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"# {args.name}: {n} episodes in {meta['wall_s']} s -> {out_dir}")


if __name__ == "__main__":
    main()
