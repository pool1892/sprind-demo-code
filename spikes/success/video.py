"""Videos of single episodes, with the state of the definitions written on every frame.

    run.sh LOG video.py expert SUITE TASK INIT [--before 30]
    run.sh --baseline LOG video.py policy RUN_DIR EPISODE_ID [--before 30] [--tail 40]

`expert` runs the scripted expert again on one episode of expert_tail.py (rendering does
not change the physics; the step at which LIBERO's check fires is printed for comparison
with the campaign's summary) and writes one video per branch. `policy` replays one
recorded policy episode (bit for bit, checked) and writes the recorded part and each tail
branch. Videos start `before` steps ahead of LIBERO's check. Each video gets a strip of
eight frames beside it as a PNG, for looking at over SSH.
Output: ~/.cache/robots-probes/success/videos/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path.home() / ".cache" / "robots-probes" / "success" / "videos"
FLAGS = (("a_libero", "LIBERO"), ("b_released", "released"), ("d_placed", "placed"), ("c_rest_k20", "rest K20"))


def annotate(frames, f, offset0: int):
    import numpy as np
    from PIL import Image, ImageDraw

    import defs

    ok = {name: defs.DEFINITIONS[name](f) for name, _ in FLAGS}
    out = []
    for i, fr in enumerate(frames):
        img = Image.fromarray(np.asarray(fr))
        dr = ImageDraw.Draw(img)
        t = i + offset0
        text = f"step {'+' if t >= 0 else ''}{t} from LIBERO's check   " + "   ".join(
            f"{label} {'yes' if ok[name][i] else 'no'}" for name, label in FLAGS)
        text2 = (f"gripper {1000 * f['gap'][i]:.0f} mm  touch {'yes' if f['robot_touch'][i] > 0.5 else 'no'}  "
                 f"speed {100 * f['lin_speed'][i]:.1f} cm/s")
        dr.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
        dr.text((4, 2), text, fill=(255, 255, 255))
        dr.text((4, 16), text2, fill=(255, 255, 0))
        out.append(np.asarray(img))
    return out


def write(path: Path, frames) -> None:
    import imageio.v2 as imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2] for fr in frames]
    imageio.mimwrite(path, frames, fps=20, codec="libx264", macro_block_size=1)
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    strip = np.concatenate([np.concatenate([frames[i] for i in idx[:4]], axis=1),
                            np.concatenate([frames[i] for i in idx[4:]], axis=1)], axis=0)
    imageio.imwrite(path.with_name(path.stem + "_strip.png"), strip)
    print(path, len(frames), "frames")


def expert_video(args) -> None:
    import fleetlab  # noqa: F401
    from fleetlab.runtime import configure_process

    configure_process()
    sys.path[:0] = [str(HERE), str(HERE.parent / "teleop-expert")]
    import numpy as np
    from common import bddl_path, load_init_states, make_env_bddl, suite_tasks
    from expert_tail import run_episode

    suite, _ = suite_tasks(args.suite)
    states = load_init_states(suite, args.task)
    env = make_env_bddl(bddl_path(suite, args.task), renderer=True)
    summary, arrays, frames = run_episode(env, np.asarray(states[args.init]), args.seed + args.init, args.tail, video=True)
    s_l = summary["s_l"]
    print("s_L", s_l, "expert", summary["expert_status"], "replays identical", summary["replay_ok"])
    for b, fr in frames.items():
        f = {k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(b + "/") and not k.endswith("actions")}
        if b == "expert":
            lo = max(0, s_l - args.before)
            fr, f, off = fr[lo:], {k: v[lo:] for k, v in f.items()}, lo - s_l
        else:
            off = 0
        write(OUT / f"{args.suite}_t{args.task}_i{args.init}_{b}.mp4", annotate(fr, f, off))


def policy_video(args) -> None:
    sys.path.insert(0, str(HERE))
    import json

    import numpy as np
    import replay_policy as R

    run = Path(args.run).expanduser()
    ep = next(json.loads(x) for x in open(run / "truth" / "episodes.jsonl") if json.loads(x)["episode_id"] == args.episode)
    chain = np.load(run / "truth" / "chain" / f"{args.episode}.npz")
    env = R.make_env(ep["task"], ep["suite"], ep.get("control_mode", "absolute"))
    env._ensure_env()
    frames: dict = {}
    summary, arrays, n_rec = R.replay_one(env, ep, chain, args.tail, frames=frames)
    print({k: v for k, v in summary.items()})
    s_l = summary["s_l"]
    lo = max(0, s_l - args.before)
    f = {k: v[lo:] for k, v in arrays.items() if "/" not in k}
    write(OUT / f"{run.name}__{args.episode}_recorded.mp4", annotate(frames["recorded"][lo:], f, lo - s_l))
    for b in R.TAIL_BRANCHES:
        if b in frames:
            f = {k.split("/", 1)[1]: v for k, v in arrays.items() if k.startswith(b + "/")}
            write(OUT / f"{run.name}__{args.episode}_{b}.mp4", annotate(frames[b], f, n_rec - 1 - s_l))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    e = sub.add_parser("expert")
    e.add_argument("suite")
    e.add_argument("task", type=int)
    e.add_argument("init", type=int)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tail", type=int, default=80)
    e.add_argument("--before", type=int, default=30)
    p = sub.add_parser("policy")
    p.add_argument("run")
    p.add_argument("episode")
    p.add_argument("--tail", type=int, default=40)
    p.add_argument("--before", type=int, default=30)
    args = ap.parse_args()
    expert_video(args) if args.kind == "expert" else policy_video(args)


if __name__ == "__main__":
    main()
