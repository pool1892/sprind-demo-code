"""Question 3: save the state in mid-episode, restore it, continue, compare.

    python save_restore.py record --out NAME [--actions FILE.npy] [--k 97] [--task 0] [--init 0]
    python save_restore.py fresh  --ref NAME [--seed 5]

'record' plays an action script, takes a snapshot after step k, plays on to
the end (the reference), then restores the snapshot into the same simulator
in several ways and replays the rest each time. 'fresh' does the restore in a
new process, into a scene that was reset with a different seed.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

import common as C
import engine as E
import snapshot as S


def play(env, obs, actions):
    """Play actions; return per-step states, proprio, images and success flags (including the start)."""
    states, proprio = [C.get_mj_state(env)], [C.flat_proprio(obs)]
    imgs = {k: [obs["pixels"][k].copy()] for k in C.CAMERAS}
    succ = []
    for a in actions:
        obs, _, terminated, _, info = env.step(a)
        states.append(C.get_mj_state(env))
        proprio.append(C.flat_proprio(obs))
        for k in C.CAMERAS:
            imgs[k].append(obs["pixels"][k].copy())
        succ.append(bool(info["is_success"]))
        if terminated:
            break
    return {
        "states": np.stack(states),
        "proprio": np.stack(proprio),
        "image": np.stack(imgs["image"]),
        "image2": np.stack(imgs["image2"]),
        "success": np.array(succ),
    }, obs


def verdict(ref, got, nphys: int) -> str:
    """nphys: number of leading state entries that are time, positions and velocities."""
    parts, exact = [], True
    n = min(len(ref["states"]), len(got["states"]))
    for label, a, b in (
        ("physics", ref["states"][:, :nphys], got["states"][:, :nphys]),
        ("solver+controls", ref["states"][1:, nphys:], got["states"][1:, nphys:]),
        ("proprio", ref["proprio"], got["proprio"]),
    ):
        if len(a) == len(b) and a.tobytes() == b.tobytes():
            parts.append(f"{label} identical")
        else:
            exact = False
            m = min(len(a), len(b))
            i = C.first_diff(a, b)
            d = np.abs(a[:m] - b[:m]).max(axis=1)
            off = 1 if label == "solver+controls" else 0
            parts.append(
                f"{label} DIFFER from step +{i + off} (there {d[i]:.1e}, at the end {d[-1]:.1e}, largest {d.max():.1e})"
            )
    for k in C.CAMERAS:
        d = np.abs(ref[k][:n].astype(np.int16) - got[k][:n].astype(np.int16)).reshape(n, -1)
        bad = np.flatnonzero(d.max(axis=1))
        if len(bad) == 0 and len(ref[k]) == len(got[k]):
            parts.append(f"{k} identical")
        else:
            exact = False
            parts.append(
                f"{k} DIFFER in {len(bad)}/{n} frames from +{bad[0] if len(bad) else n} "
                f"(largest {int(d.max())}/255, {100 * (d[bad] > 0).mean():.2f}% of values)"
            )
    same_outcome = bool(ref["success"].any()) == bool(got["success"].any())
    return ("EXACT" if exact else "NOT EXACT") + " | " + "; ".join(parts) + (
        "" if same_outcome else "; OUTCOME DIFFERS"
    )


def record(args):
    actions = np.load(args.actions) if args.actions else C.synthetic_actions()
    env = C.make_env(args.suite, args.task, observation_height=360, observation_width=360)
    spec = E.Spec(task=args.task, init=args.init, seed=args.seed, noise_seed=0, suite=args.suite)
    obs = E.reset_episode(env, spec)
    k = args.k
    head, obs_k = play(env, obs, actions[:k])
    assert len(head["states"]) == k + 1, "episode ended before the snapshot step"
    snap = S.take(env, obs_k)
    ref, _ = play(env, obs_k, actions[k:])
    print(f"reference: {len(ref['states']) - 1} steps after step {k}, success {bool(ref['success'].any())}")

    m = C.sim_of(env).model._model
    nphys = 1 + m.nq + m.nv

    def trial(label, target_env, skip=(), forward=False, libero=False):
        for g in skip:
            if g != "obs" and len(skip) == 1:
                names = S.stale(target_env, snap, g)
                shown = ", ".join(names[:12]) + (f", ... ({len(names)} in all)" if len(names) > 12 else "")
                print(f"  [{g}: values that differ from the snapshot before the restore: {shown or 'none'}]")
        if libero:
            o = S.libero_restore(target_env, snap)
        else:
            o = S.restore(target_env, snap, skip=skip, forward=forward)
            if o is None:
                o = obs_k
        got, _ = play(target_env, o, actions[k:])
        print(f"  {label:<46} {verdict(ref, got, nphys)}", flush=True)

    print("restore later in the same process (the simulator had run on to the end of the episode):")
    trial("everything", env)
    trial("everything, then mj_forward", env, forward=True)
    for g in S.GROUPS:
        if g in ("mujoco", "obs", "model"):
            continue
        trial(f"everything except {g}", env, skip=(g,))
    lean = ("arm", "composite", "robot", "observables", "env", "objects")
    trial("lean set: MuJoCo+warm start, gripper, model, obs", env, skip=lean)
    trial("MuJoCo state only (with warm start)", env, skip=tuple(g for g in S.GROUPS if g not in ("mujoco", "warmstart")))
    trial("LIBERO's own set_state (positions, velocities)", env, libero=True)

    print("restore after a hard reset with another seed (new model, moved fixtures, new controllers):")
    other = E.Spec(task=args.task, init=(args.init + 3) % 50, seed=args.seed + 77, noise_seed=0, suite=args.suite)
    E.reset_episode(env, other)
    changed = [n for n, v in S.model_arrays(env).items() if n in snap["model"] and not np.array_equal(v, snap["model"][n])]
    big = {n: C.sha(v) for n, v in S.model_arrays(env).items() if n in snap["model_big_hash"]}
    print(f"  model fields that differ from the snapshot after that reset: {changed}")
    print(f"  large model arrays (meshes, textures) identical: {big == snap['model_big_hash']}")
    trial("everything", env)
    E.reset_episode(env, other)
    trial("lean set: MuJoCo+warm start, gripper, model, obs", env, skip=lean)
    E.reset_episode(env, other)
    trial("everything except model", env, skip=("model",))
    E.reset_episode(env, other)
    trial("everything except arm", env, skip=("arm",))
    E.reset_episode(env, other)
    trial("everything except gripper", env, skip=("gripper",))
    E.reset_episode(env, other)
    trial("everything except observables", env, skip=("observables",))
    E.reset_episode(env, other)
    trial("everything except robot", env, skip=("robot",))
    E.reset_episode(env, other)
    trial("everything except env", env, skip=("env",))
    E.reset_episode(env, other)
    trial("everything except warmstart", env, skip=("warmstart",))
    E.reset_episode(env, other)
    trial("LIBERO's own set_state (positions, velocities)", env, libero=True)

    with open(C.OUT / f"{args.out}.pkl", "wb") as f:
        pickle.dump({"snap": snap, "ref": ref, "actions": actions, "k": k, "args": vars(args), "nphys": nphys}, f)
    size = (C.OUT / f"{args.out}.pkl").stat().st_size
    snap_only = len(pickle.dumps(snap))
    print(f"snapshot is {snap_only / 1e6:.2f} MB pickled (file with reference: {size / 1e6:.1f} MB)")
    env.close()


def fresh(args):
    with open(C.OUT / f"{args.ref}.pkl", "rb") as f:
        d = pickle.load(f)
    a = d["args"]
    env = C.make_env(a["suite"], a["task"], observation_height=360, observation_width=360)
    E.reset_episode(env, E.Spec(task=a["task"], init=(a["init"] + 5) % 50, seed=args.seed, noise_seed=0, suite=a["suite"]))
    obs = S.restore(env, d["snap"])
    got, _ = play(env, obs, d["actions"][d["k"] :])
    print(f"fresh process, scene reset with seed {args.seed}, everything restored: {verdict(d['ref'], got, d['nphys'])}")
    E.reset_episode(env, E.Spec(task=a["task"], init=(a["init"] + 5) % 50, seed=args.seed, noise_seed=0, suite=a["suite"]))
    obs = S.libero_restore(env, d["snap"])
    got, _ = play(env, obs, d["actions"][d["k"] :])
    print(f"fresh process, LIBERO's own set_state: {verdict(d['ref'], got, d['nphys'])}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--actions", default=None)
    r.add_argument("--k", type=int, default=97)
    r.add_argument("--suite", default="libero_spatial")
    r.add_argument("--task", type=int, default=0)
    r.add_argument("--init", type=int, default=0)
    r.add_argument("--seed", type=int, default=1000)
    f = sub.add_parser("fresh")
    f.add_argument("--ref", required=True)
    f.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    {"record": record, "fresh": fresh}[args.cmd](args)
