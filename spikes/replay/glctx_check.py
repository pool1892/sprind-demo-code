"""Rendering when one process holds more than one environment, or changes what it renders.

robosuite gives every environment its own OpenGL context but never makes it
current again after creating it, so every later render and every later
clean-up runs in whichever context was created last. This script measures what
that does and whether an explicit switch repairs it.

    python glctx_check.py ref          # each scene alone in its process: the reference images
    python glctx_check.py both         # two scenes alive in one process, stock behaviour
    python glctx_check.py both-switch  # the same with the context made current before each use
    python glctx_check.py rebuild      # one scene closed, another built, in one process
    python glctx_check.py visual       # camera, light, colour and texture changed on a live model

A is libero_spatial task 0, B is libero_10 task 3 (a different set of objects),
A2 is libero_spatial task 0 again with a different scene seed.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

import common as C
import engine as E

SCENES = {
    "A": E.Spec(task=0, init=0, seed=1000, noise_seed=0, suite="libero_spatial"),
    "B": E.Spec(task=3, init=0, seed=1000, noise_seed=0, suite="libero_10"),
    "A2": E.Spec(task=0, init=0, seed=7, noise_seed=0, suite="libero_spatial"),
}
ACTS = C.synthetic_actions(12, seed=1)
FIRST, MORE = ACTS[:6], ACTS[6:]


def build(name):
    s = SCENES[name]
    return C.make_env(s.suite, s.task, observation_height=360, observation_width=360)


def current(env):
    C.sim_of(env)._render_context_offscreen.gl_ctx.make_current()


def frames(env, name, switch=False, reset=True, acts=FIRST):
    """Reset (optionally) and take six steps; return the agent-view images."""
    out = []
    if switch:
        env._ensure_env()
        current(env)
    if reset:
        out.append(E.reset_episode(env, SCENES[name])["pixels"]["image"].copy())
    for a in acts:
        if switch:
            current(env)
        out.append(env.step(a)[0]["pixels"]["image"].copy())
    return np.stack(out)


def report(label, got, ref):
    n = len(got)
    assert len(ref) == n
    d = got.astype(np.int16) - ref.astype(np.int16)
    wrong = (np.abs(d).max(axis=-1) > 0).reshape(n, -1).mean(axis=1)
    verdict = "identical to the reference" if wrong.max() == 0 else (
        f"WRONG: {100 * wrong.mean():.1f}% of pixels differ on average (worst frame {100 * wrong.max():.1f}%)"
    )
    print(f"  {label}: {verdict}", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "ref":
        name = sys.argv[2]
        env = build(name)
        np.save(C.OUT / f"glctx_ref_{name}.npy", np.concatenate([frames(env, name), frames(env, name, reset=False, acts=MORE)]))
        print(f"reference for {name} stored")
        return
    full = {n: np.load(C.OUT / f"glctx_ref_{n}.npy") for n in SCENES}
    ref = {n: v[:7] for n, v in full.items()}  # reset and the first six steps
    more = {n: v[7:] for n, v in full.items()}  # the next six steps, without a reset

    if mode in ("both", "both-switch"):
        sw = mode == "both-switch"
        print(f"two different scenes alive in one process ({'context switched before each use' if sw else 'stock'}):")
        a, b = build("A"), build("B")
        fa = frames(a, "A", sw)
        fb = frames(b, "B", sw)
        report("A, reset and stepped before B existed in the renderer", fa, ref["A"])
        report("B, built second", fb, ref["B"])
        report("A again, stepped after B was built (no reset)", frames(a, "A", sw, reset=False, acts=MORE), more["A"])
        report("A after a hard reset of A", frames(a, "A", sw), ref["A"])
        report("B stepped after A's hard reset (no reset)", frames(b, "B", sw, reset=False, acts=MORE), more["B"])
        report("B after a hard reset of B", frames(b, "B", sw), ref["B"])
        print(f"the same task twice in one process, different fixture layouts ({'switched' if sw else 'stock'}):")
        a.close()
        b.close()
        a, a2 = build("A"), build("A2")
        fa = frames(a, "A", sw)
        fa2 = frames(a2, "A2", sw)
        report("A", fa, ref["A"])
        report("A2", fa2, ref["A2"])
        report("A after a hard reset", frames(a, "A", sw), ref["A"])
        report("A2 after A's hard reset (no reset)", frames(a2, "A2", sw, reset=False, acts=MORE), more["A2"])
    elif mode == "rebuild":
        print("one environment at a time: closed, then another built in the same process:")
        a = build("A")
        report("A, first in the process", frames(a, "A"), ref["A"])
        a.close()
        b = build("B")
        report("B, built after A was closed", frames(b, "B"), ref["B"])
        b.close()
        a = build("A")
        report("A, built after B was closed", frames(a, "A"), ref["A"])
        report("A after one more hard reset", frames(a, "A"), ref["A"])
    elif mode == "visual":
        print("visual parameters changed on a live model (one environment in the process):")
        env = build("A")
        base = frames(env, "A")[-1]
        sim = C.sim_of(env)
        m, con = sim.model._model, sim._render_context_offscreen.con
        rs = C.rs_env_of(env)

        def shot():
            sim.forward()  # camera and light poses are recomputed with the kinematics
            return env._format_raw_obs(rs._get_observations(force_update=True))["pixels"]["image"].copy()

        still = shot()
        cam = m.camera("agentview").id

        def check(label, change, restore, upload=None):
            change()
            after = shot()
            if upload and (after == still).all():
                print(f"  {label}: NOT picked up by the next render without an upload")
                upload()
                after = shot()
            restore()
            if upload:
                upload()
            back = shot()
            changed = 100 * (np.abs(after.astype(int) - still.astype(int)).max(-1) > 0).mean()
            print(
                f"  {label}: {'picked up' if changed > 0 else 'NOT picked up'} ({changed:.0f}% of pixels changed); "
                f"after undoing it the image is {'identical to before' if (back == still).all() else 'NOT the same as before'}"
            )

        old = m.cam_pos[cam].copy()
        check("camera position", lambda: m.cam_pos.__setitem__(cam, old + [0.0, 0.05, 0.03]), lambda: m.cam_pos.__setitem__(cam, old))
        old_l = m.light_diffuse.copy()
        check("light colour", lambda: m.light_diffuse.__setitem__(slice(None), old_l * 0.4), lambda: m.light_diffuse.__setitem__(slice(None), old_l))
        old_g = m.geom_rgba.copy()
        check("geom colours", lambda: m.geom_rgba.__setitem__((slice(None), 0), old_g[:, 0] * 0.5), lambda: m.geom_rgba.__setitem__(slice(None), old_g))
        old_m = m.mat_rgba.copy()
        check("material colours", lambda: m.mat_rgba.__setitem__((slice(None), 1), old_m[:, 1] * 0.5), lambda: m.mat_rgba.__setitem__(slice(None), old_m))
        old_t = m.tex_data.copy()

        def upload_all():
            for t in range(m.ntex):
                mujoco.mjr_uploadTexture(m, con, t)

        check("texture pixels", lambda: m.tex_data.__setitem__(slice(None), 255 - old_t), lambda: m.tex_data.__setitem__(slice(None), old_t), upload=upload_all)
        del base
        print("the same changes survive resets?")
        m.cam_pos[cam] = old + [0.0, 0.05, 0.03]
        moved = shot()
        for hard in (False, True):
            env.hard_reset = hard
            rs.hard_reset = hard
            E.reset_episode(env, SCENES["A"])
            m2 = C.sim_of(env).model._model
            kept = np.allclose(m2.cam_pos[cam], old + [0.0, 0.05, 0.03])
            print(f"  camera position after a {'hard' if hard else 'soft'} reset: {'kept' if kept else 'back to the default'}")
        del moved


if __name__ == "__main__":
    main()
