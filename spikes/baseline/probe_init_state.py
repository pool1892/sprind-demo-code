"""What does the policy see at the start of an episode? No model, no GPU job.

For one LIBERO task this resets LeRobot's environment wrapper on the first few
stored initial states and records
  * where the task's objects have come to rest (for libero_spatial task 5: is
    the bowl still on the ramekin?),
  * the 8-number robot state the policy receives,
  * the two camera images exactly as the policy receives them (after LeRobot's
    180 degree flip), with their mean brightness.
With --dataset it also pulls the first frame of a training episode of the same
task from a local copy of lerobot/libero, so the two can be compared.

    python probe_init_state.py --suite libero_spatial --task 5 --out DIR \
        [--n 3] [--size 256] [--dataset ~/.cache/robots-probes/baseline/ds/lerobot_libero]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--task", type=int, default=5)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default=None)
args = ap.parse_args()

import mujoco  # noqa: E402
import robosuite  # noqa: E402
from PIL import Image  # noqa: E402

from lerobot.envs.libero import LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.processor.env_processor import LiberoProcessorStep  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
suite = _get_suite(args.suite)
step = LiberoProcessorStep()
report = {
    "mujoco": mujoco.__version__,
    "robosuite": robosuite.__version__,
    "suite": args.suite,
    "task": args.task,
    "language": suite.get_task(args.task).language,
    "size": args.size,
    "inits": [],
}


def tilt_deg(xmat: np.ndarray) -> float:
    z = xmat.reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))


for i in range(args.n):
    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task,
        task_suite_name=args.suite,
        obs_type="pixels_agent_pos",
        observation_width=args.size,
        observation_height=args.size,
        episode_index=i,
        n_envs=1,
    )
    obs, _ = env.reset(seed=1000 + i)
    sim = env._env.sim
    bodies = {}
    for b in range(sim.model.nbody):
        name = sim.model.body(b).name
        if name.endswith("_main") and not name.startswith(("robot", "gripper", "mount")):
            bodies[name] = {
                "pos": [round(float(x), 4) for x in sim.data.xpos[b]],
                "tilt_deg": round(tilt_deg(sim.data.xmat[b]), 1),
            }
    # Same path as lerobot-eval: numpy obs -> tensors -> LIBERO processor step.
    batched = {
        "pixels": {k: v[None] for k, v in obs["pixels"].items()},
        "robot_state": {
            g: {k: np.asarray(v)[None] for k, v in d.items()} for g, d in obs["robot_state"].items()
        },
    }
    t = preprocess_observation(batched)
    t = step.observation(t)
    state = t["observation.state"][0].tolist()
    entry = {"init": i, "state": [round(x, 4) for x in state], "bodies": bodies, "images": {}}
    for key in ("observation.images.image", "observation.images.image2"):
        img = (t[key][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        name = f"env_init{i}_{key.split('.')[-1]}.png"
        Image.fromarray(img).save(out / name)
        entry["images"][key] = {
            "file": name,
            "shape": list(img.shape),
            "mean": round(float(img.mean()), 2),
            "mean_rgb": [round(float(x), 2) for x in img.reshape(-1, 3).mean(0)],
        }
    report["inits"].append(entry)
    env.close()

if args.dataset:
    import pandas as pd

    root = Path(args.dataset)
    info = json.loads((root / "meta/info.json").read_text())
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task_index = int(tasks.loc[report["language"], "task_index"])
    eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    found = []
    for (ci, fi), grp in eps.groupby(["data/chunk_index", "data/file_index"]):
        df = pd.read_parquet(root / info["data_path"].format(chunk_index=ci, file_index=fi))
        first = df[df["frame_index"] == 0]
        for _, r in first[first["task_index"] == task_index].iterrows():
            found.append((int(r["episode_index"]), r["observation.state"].tolist(), r["action"].tolist()))
        if len(found) >= args.n:
            break
    report["dataset"] = {"task_index": task_index, "fps": info["fps"], "episodes": []}
    for ep_idx, state, action in found[: args.n]:
        e = eps[eps["episode_index"] == ep_idx].iloc[0]
        entry = {"episode": ep_idx, "length": int(e["length"]), "state": [round(x, 4) for x in state],
                 "first_action": [round(x, 4) for x in action], "images": {}}
        for key in ("observation.images.image", "observation.images.image2"):
            video = root / info["video_path"].format(
                video_key=key,
                chunk_index=int(e[f"videos/{key}/chunk_index"]),
                file_index=int(e[f"videos/{key}/file_index"]),
            )
            ts = float(e[f"videos/{key}/from_timestamp"])
            name = f"dataset_ep{ep_idx}_{key.split('.')[-1]}.png"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{ts + 0.001:.4f}", "-i", str(video),
                 "-frames:v", "1", str(out / name)],
                check=True,
            )
            img = np.asarray(Image.open(out / name).convert("RGB"))
            entry["images"][key] = {
                "file": name,
                "shape": list(img.shape),
                "mean": round(float(img.mean()), 2),
                "mean_rgb": [round(float(x), 2) for x in img.reshape(-1, 3).mean(0)],
            }
        report["dataset"]["episodes"].append(entry)

(out / "probe.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
