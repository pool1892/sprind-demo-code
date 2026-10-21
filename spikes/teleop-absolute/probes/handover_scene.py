"""What does the scene look like when X-VLA hands over? For explaining failed take-overs.

For each named campaign row (an X-VLA take-over), replays the recording to the
hand-over step and prints: what the task's object touches, what else touches
the target, how far every movable object has moved, and the position of every
fixture joint (drawers, doors) against the start of the episode.

    run.sh probes/handover_scene.py c_xvla_end [EPISODE_ID@STEP ...]   (default: every failure)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fleetlab  # noqa: F401  (torch before mujoco)
import mujoco
import numpy as np

from episode import OUT, make_env
from teleop.expert import ScriptedExpert
from xvla_record import h64, load_episode, load_index, mj_state


def contacts_of(sc, name: str) -> list[str]:
    """Names of objects, fixtures or 'robot'/'table' that touch `name`."""
    mine = set(sc.geoms[name])
    out = set()
    for i in range(sc.d.ncon):
        c = sc.d.contact[i]
        if c.dist > 0.0:
            continue
        other = c.geom2 if c.geom1 in mine else c.geom1 if c.geom2 in mine else None
        if other is None or other in mine:
            continue
        if other in sc.robot_geoms:
            out.add("robot")
            continue
        for n, gs in sc.geoms.items():
            if other in gs:
                out.add(n)
                break
        else:
            out.add("table")
    return sorted(out)


def fixture_joints(sc) -> dict[str, float]:
    out = {}
    for name in sc.names:
        if name in sc.env.objects_dict:
            continue
        root = sc.root[name]
        for j in range(sc.m.njnt):
            if sc.m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE) and sc._descends(
                sc.m.jnt_bodyid[j], root
            ):
                out[mujoco.mj_id2name(sc.m, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(sc.d.qpos[sc.m.jnt_qposadr[j]])
    return out


def main() -> None:
    batch = sys.argv[1]
    rows = [json.loads(line) for line in open(OUT / "results" / f"{batch}.jsonl")]
    wanted = set(sys.argv[2:])
    picks = [r for r in rows if (f"{r['xvla_episode']}@{r['handover_step']}" in wanted) or (not wanted and not r["success"] and not str(r["status"]).startswith("skipped"))]
    for r in sorted(picks, key=lambda r: (r["suite"], r["task"])):
        run_dir = Path(r["xvla_run"])
        rec = load_episode(run_dir, r["xvla_episode"], load_index(run_dir))
        env = make_env(rec.suite, rec.task)
        env.reset(init_state=rec.init, seed=rec.scene_seed)
        ex = ScriptedExpert(env._env)
        start = {n: ex.sc.body_pos(n) for n in ex.e.objects_dict}
        joints0 = fixture_joints(ex.sc)
        for t in range(r["handover_step"]):
            env.step(rec.actions[t])
        assert h64(mj_state(env)) == rec.state_hashes[r["handover_step"]]
        env._env.sim.forward()
        sc = ex.sc
        moved = {n: round(float(np.linalg.norm(sc.body_pos(n) - p)), 3) for n, p in start.items()}
        joints = {k: (round(v0, 3), round(fixture_joints(sc)[k], 3)) for k, v0 in joints0.items()
                  if abs(fixture_joints(sc)[k] - v0) > 0.002}
        print(json.dumps(dict(
            episode=f"{r['xvla_episode']}@{r['handover_step']}", task=r["task"], reason=r.get("reason"),
            instruction=env.task_description, object=ex.obj, target=ex.container,
            object_touches=contacts_of(sc, ex.obj), target_touches=contacts_of(sc, ex.container),
            moved_m={k: v for k, v in moved.items() if v > 0.01}, fixture_joints_moved=joints,
        )), flush=True)
        env.close()


if __name__ == "__main__":
    main()
