"""Run the scripted expert past LIBERO's check and record the features of defs.py at every step.

    run.sh expert_tail.py --name NAME [--suites libero_spatial,libero_object] [--tasks 0-9]
                          [--inits 0-19] [--tail 80] [--workers 4] [--chunk 4]

For every episode (task, initial state) the expert of spikes/teleop-expert runs from
LIBERO's initial state in relative control, exactly as in that spike. Let s_L be the
first control step after which LIBERO's check is true. Three branches leave from the
same state at s_L (the later two by resetting and replaying the expert's own actions up
to s_L, checked bit for bit through a hash of MuJoCo's full state):

  expert     the expert carries on: sets the object down, lets go, backs off; once it
             has stopped, the arm holds still with the gripper open.
  open_now   the arm holds still where it is and the gripper opens at once: a policy
             that lets go the moment LIBERO's check fires.
  carry_off  the gripper stays closed, the arm rises 15 cm and then moves 15 cm toward
             the robot: a policy that never lets go.

Each branch runs `tail` control steps past s_L. The features of defs.py and LIBERO's
own `check_success()` are recorded at every step from the reset on. Output under
~/.cache/robots-probes/success/expert/NAME/: one .npz per episode, and summary.jsonl.
"""

from __future__ import annotations

import fleetlab  # noqa: F401  (torch before mujoco)
from fleetlab.runtime import configure_process

configure_process()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERT_DIR = HERE.parent / "teleop-expert"
OUT = Path.home() / ".cache" / "robots-probes" / "success"
BRANCHES = ("expert", "open_now", "carry_off")


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def state_hash(env) -> str:
    import mujoco
    import numpy as np

    m, d = env.env.sim.model._model, env.env.sim.data._data
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    out = np.empty(mujoco.mj_stateSize(m, spec), np.float64)
    mujoco.mj_getState(m, d, out, spec)
    return hashlib.sha256(out.tobytes()).hexdigest()[:16]


class Recorder:
    """Feature rows of one branch, one per state."""

    def __init__(self, env):
        import defs

        self.env = env
        self.ix = defs.build_index(env.env)
        self.rows: list[dict[str, float]] = []

    def __call__(self) -> dict[str, float]:
        import defs

        e = self.env.env
        f = defs.features(e.sim.model._model, e.sim.data._data, self.ix)
        f["libero_env"] = float(bool(self.env.check_success()))
        self.rows.append(f)
        return f

    def arrays(self):
        import numpy as np

        return {k: np.array([r[k] for r in self.rows], np.float64) for k in self.rows[0]}


def scripted_branch(env, kind: str, n: int, rec: Recorder, frames=None, grab=None) -> list:
    """Run one of the two scripted branches for n steps from the current state."""
    import numpy as np

    e = env.env
    m, d = e.sim.model._model, e.sim.data._data
    import mujoco

    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
    p0 = d.site_xpos[site].copy()
    base = np.array(e.robots[0].base_pos[:2])
    toward = base - p0[:2]
    toward /= max(np.linalg.norm(toward), 1e-9)
    actions = []
    for i in range(n):
        a = np.zeros(7)
        if kind == "open_now":
            a[6] = -1.0
        elif kind == "carry_off":
            goal = p0 + np.array([0, 0, 0.15])
            if i >= 20:
                goal[:2] += 0.15 * toward
            a[:3] = np.clip(1.5 * (goal - d.site_xpos[site]) / 0.05, -1, 1)
            a[6] = 1.0
        else:
            raise ValueError(kind)
        env.step(a)
        actions.append(a)
        rec()
        if grab is not None:
            grab()
    return actions


def run_episode(env, init_state, seed: int, tail: int, max_steps: int = 600, branches=BRANCHES, video=None):
    """One episode and its branches. Returns (summary dict, arrays dict)."""
    import numpy as np
    from common import DUMMY, frame, reset_to_init
    from expert import ScriptedExpert

    frames: dict[str, list] = {}

    def grabber(name):
        if video is None:
            return None
        frames.setdefault(name, [])

        def grab():
            frames[name].append(np.concatenate([frame(env, c, 256) for c in ("agentview", "frontview")], axis=1))

        return grab

    out: dict[str, np.ndarray] = {}
    reset_to_init(env, init_state, seed=seed)
    rec = Recorder(env)
    rec()
    g = grabber("expert")
    if g:
        g()
    expert = ScriptedExpert(env, max_steps=max_steps)
    actions, s_l, done_at, h_at_sl = [], None, None, None
    t = 0
    while True:
        if s_l is not None and t >= s_l + tail:
            break
        if s_l is None and (expert.done or t >= max_steps):
            break
        a = expert.act() if not expert.done else np.asarray(DUMMY, float)
        if expert.done and done_at is None:
            done_at = t
        env.step(a)
        t += 1
        actions.append(np.asarray(a, float))
        f = rec()
        if g:
            g()
        if s_l is None and f["libero_env"] > 0.5:
            s_l = t
            h_at_sl = state_hash(env)
    if expert.done and done_at is None:
        done_at = t
    for k, v in rec.arrays().items():
        out[f"expert/{k}"] = v
    out["expert/actions"] = np.array(actions)
    summary = {
        "s_l": s_l,
        "steps": t,
        "expert_status": expert.status,
        "expert_done_at": done_at,
        "attempts": expert.attempt,
        "events": [f"{tt}: {msg}" for tt, msg in expert.events],
        "replica_mismatch": {},
        "replay_ok": {},
    }
    if s_l is not None:
        for b in branches:
            if b == "expert":
                continue
            reset_to_init(env, init_state, seed=seed)
            for a in actions[:s_l]:
                env.step(a)
            summary["replay_ok"][b] = state_hash(env) == h_at_sl
            r = Recorder(env)
            r()
            gb = grabber(b)
            if gb:
                gb()
            acts = scripted_branch(env, b, tail, r, grab=gb)
            for k, v in r.arrays().items():
                out[f"{b}/{k}"] = v
            out[f"{b}/actions"] = np.array(acts)
    for b in branches:
        if f"{b}/libero" in out:
            summary["replica_mismatch"][b] = int(np.sum(out[f"{b}/libero"] != out[f"{b}/libero_env"]))
    return summary, out, frames


def worker(job: dict) -> list[dict]:
    sys.path[:0] = [str(HERE), str(EXPERT_DIR)]
    import numpy as np
    from common import make_env_bddl

    rows = []
    out_dir = Path(job["out"])
    try:
        env = make_env_bddl(job["bddl"], renderer=False)
    except Exception:  # noqa: BLE001
        return [dict(job_error=traceback.format_exc(), suite=job["suite"], task=job["task"])]
    for iid, init_state in job["inits"]:
        t0 = time.time()
        row = dict(suite=job["suite"], task=job["task"], init=iid, seed=job["seed"] + iid)
        try:
            summary, arrays, _ = run_episode(env, np.asarray(init_state), job["seed"] + iid, job["tail"])
            row.update(summary)
            np.savez_compressed(out_dir / f"{job['suite']}_t{job['task']}_i{iid}.npz", **arrays)
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--suites", default="libero_spatial,libero_object")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0-19")
    ap.add_argument("--tail", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4, help="episodes per worker process")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.workers <= 4, "a light job runs at most four simulator processes"

    sys.path[:0] = [str(HERE), str(EXPERT_DIR)]
    from common import bddl_path, load_init_states, suite_tasks

    out_dir = OUT / "expert" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists already; choose another name")
    out_dir.mkdir(parents=True)
    inits = parse_range(args.inits)
    jobs = []
    for suite_name in args.suites.split(","):
        suite, _ = suite_tasks(suite_name)
        for tid in parse_range(args.tasks):
            states = load_init_states(suite, tid)
            for k in range(0, len(inits), args.chunk):
                part = inits[k : k + args.chunk]
                jobs.append(
                    dict(suite=suite_name, task=tid, bddl=bddl_path(suite, tid), seed=args.seed, tail=args.tail,
                         inits=[(i, states[i].tolist()) for i in part], out=str(out_dir))
                )
    meta = dict(vars(args), when=time.strftime("%F %T"), jobs=len(jobs), episodes=len(jobs) and sum(len(j["inits"]) for j in jobs))
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    n = 0
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool, open(out_dir / "summary.jsonl", "w") as f:
        for part in pool.imap_unordered(worker, jobs, chunksize=1):
            for r in part:
                f.write(json.dumps(r, default=str) + "\n")
                n += 1
            f.flush()
            print(f"{n} episodes, {time.time() - t0:.0f} s", flush=True)
    meta["wall_s"] = round(time.time() - t0, 1)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"# {args.name}: {n} episodes in {meta['wall_s']} s -> {out_dir}")


if __name__ == "__main__":
    main()
