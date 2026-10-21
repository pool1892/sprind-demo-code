"""Physics sanity check of the per-step goal evaluation (light: no policy, no GPU model).

On the base task of J1 (put the cream cheese in the bowl), start 0:

1. at the start no goal of J1 holds, and this runner's evaluation of the base
   task's goal agrees with the env's own LIBERO check;
2. the cream cheese moved to 6 cm above the bowl's centre, before any physics
   step: `On cream_cheese_1 akita_black_bowl_1` does not hold, because LIBERO's
   `On` needs contact (`check_ontop`), not only a pose above the bowl;
3. after 40 held control steps it rests in the bowl: the first goal holds, the
   second (`On akita_black_bowl_1 plate_1`) does not, the env's check agrees;
4. from a fresh reset, the bowl moved above the plate and settled: the second
   goal holds and the first does not.

It also builds the two tasks whose goal is unary (turn on the stove, open the
middle drawer), which `build_index` alone refuses, and evaluates every
goal-suite goal on each start state.

    tools/heavy-run --light uv run --locked python spikes/jobs/goals_check.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import fleetlab  # noqa: F401  first: torch before mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner


def hold(obs: Any) -> np.ndarray:
    """Absolute control: keep the commanded pose, gripper open."""
    from scipy.spatial.transform import Rotation

    eef = obs["robot_state"]["eef"]
    a = np.zeros(7, np.float32)
    a[:3] = eef["pos"]
    a[3:6] = Rotation.from_matrix(np.asarray(eef["mat"], np.float64)).as_rotvec()
    a[6] = -1.0
    return a


def move(env: Any, name: str, xyz: np.ndarray) -> None:
    import mujoco

    rs = env._rs
    model, data = env._sim.model._model, env._sim.data._data
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rs.objects_dict[name].joints[-1])
    q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data.qpos[q : q + 3] = xyz
    data.qvel[v : v + 6] = 0.0
    mujoco.mj_forward(model, data)


def section(text: str, name: str) -> str:
    """One top-level section of a BDDL file, from its opening '(:name' to its closing paren."""
    start = text.index(f"(:{name}")
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return " ".join(text[start : i + 1].split())
    raise ValueError(name)


def shared_scene() -> dict[str, Any]:
    """Whether the ten goal-suite BDDL files share regions, fixtures, objects and init."""
    texts = {stem: (runner.BDDL_DIR / f"{stem}.bddl").read_text() for stem in runner.plan_tasks()}
    first = next(iter(texts.values()))
    out: dict[str, Any] = {}
    for name in ("regions", "fixtures", "objects", "init"):
        ref = section(first, name)
        out[name] = all(section(t, name) == ref for t in texts.values())
    out["goals"] = {stem: section(t, "goal") for stem, t in texts.items()}
    return out


def frame(obs: Any) -> np.ndarray:
    return np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from fleetlab.engine import software
    from fleetlab.policies.xvla import nominal_spec

    cat = runner.Catalogue()
    spec, soft = nominal_spec(runner.PRECISION), software.current("none")
    report: dict[str, Any] = {"bench_index_by_plan_number": {}, "shared_scene": shared_scene()}
    scene = report["shared_scene"]
    assert all(scene[k] for k in ("regions", "fixtures", "objects", "init")), scene
    for n, stem in enumerate(cat.stems):
        report["bench_index_by_plan_number"][n] = [stem, cat.bench[stem], cat.language[stem]]

    # Every base task builds, unary goals included; every goal evaluates on its start states.
    report["starts"] = {}
    for number in (2, 7, 6, 4, 8):
        env = runner.make_env(cat.bench[cat.stems[number]])
        goals = [cat.goal(i, env._rs) for i in range(10)]
        rows = []
        for start in (0, 1, 2):
            obs = env.reset(runner.start_identity(start, goals[number].bench, spec, soft), [])
            values = [runner.holds(env._rs, g.predicates) for g in goals]
            own = bool(env._inner._env.check_success())
            rows.append({"start": start, "goals": values, "env_check": own})
            assert own == values[number], (number, start, own, values[number])
        report["starts"][number] = {
            "stem": cat.stems[number],
            "predicates": [list(p) for p in goals[number].predicates],
            "named": list(goals[number].named),
            "rows": rows,
        }
        env.close()

    # The physics check on J1's base task.
    env = runner.make_env(cat.bench[cat.stems[6]])
    first, second = cat.goal(6, env._rs), cat.goal(4, env._rs)
    ident = runner.start_identity(0, first.bench, spec, soft)
    obs = env.reset(ident, [])
    rs = env._rs

    def state(label: str) -> dict[str, Any]:
        return {
            "label": label,
            "cheese_in_bowl": runner.holds(rs, first.predicates),
            "bowl_on_plate": runner.holds(rs, second.predicates),
            "env_check_cheese_in_bowl": bool(env._inner._env.check_success()),
        }

    checks = [state("start 0 after reset")]
    poses = env.privileged_state().object_poses
    bowl = poses["akita_black_bowl_1"][:3].copy()
    move(env, "cream_cheese_1", bowl + np.array([0.0, 0.0, 0.06]))
    checks.append(state("cheese moved 6 cm above the bowl, no physics step yet"))
    images = {}
    for _ in range(40):
        obs = env.step(hold(obs)).observation
    checks.append(state("after 40 held steps (cheese settled in the bowl)"))
    images["cheese_in_bowl"] = frame(obs)
    obs = env.reset(ident, [])
    poses = env.privileged_state().object_poses
    plate = poses["plate_1"][:3].copy()
    move(env, "akita_black_bowl_1", plate + np.array([0.0, 0.0, 0.05]))
    for _ in range(40):
        obs = env.step(hold(obs)).observation
    checks.append(state("fresh reset, bowl moved above the plate, after 40 held steps"))
    images["bowl_on_plate"] = frame(obs)
    env.close()
    report["physics"] = checks
    expected = [
        (False, False),
        (False, False),
        (True, False),
        (False, True),
    ]
    got = [(c["cheese_in_bowl"], c["bowl_on_plate"]) for c in checks]
    report["physics_as_expected"] = got == expected
    report["env_check_agrees"] = all(
        c["cheese_in_bowl"] == c["env_check_cheese_in_bowl"] for c in checks
    )
    (args.out / "goals_check.json").write_text(json.dumps(report, indent=1))
    if args.figures is not None:
        import imageio.v2 as imageio

        args.figures.mkdir(parents=True, exist_ok=True)
        strip = np.concatenate([images["cheese_in_bowl"], images["bowl_on_plate"]], axis=1)
        imageio.imwrite(args.figures / "goals_check.png", strip[::2, ::2])
    print(json.dumps(report["shared_scene"], indent=1))
    print(
        json.dumps(
            {
                "physics": checks,
                "as_expected": report["physics_as_expected"],
                "env_check_agrees": report["env_check_agrees"],
            },
            indent=1,
        )
    )
    for number, block in report["starts"].items():
        print(
            number,
            block["stem"],
            block["predicates"],
            block["named"],
            block["rows"][0]["goals"],
        )
    return 0 if report["physics_as_expected"] and report["env_check_agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
