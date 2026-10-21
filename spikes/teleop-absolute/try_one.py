"""Run one episode, print its record and event log; optionally write a video.

    run.sh try_one.py SUITE TASK INIT [--scenario init|KIND] [--mode absolute|relative]
                      [--no-restore] [--seed-base 1000] [--video PATH.mp4] [--size 256]
    run.sh try_one.py --xvla RUN_DIR EPISODE_ID [--handover N] [--video PATH.mp4]

A video shows both cameras side by side (the scene camera and the wrist camera)
at 10 frames per second, with a bar on top that is orange before the hand-over
(set-up or X-VLA) and blue while the expert acts. A strip of eight frames is
written beside it as a PNG, for looking at over SSH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fleetlab  # noqa: F401  (torch before mujoco)
import numpy as np

from episode import EpisodeSpec, Recorder, make_env, run_episode
from fleetlab.video import write_mp4
from xvla_record import load_episode, load_index


def write_strip(frames: list[np.ndarray], path: Path, n: int = 8) -> None:
    import imageio.v2 as imageio

    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    rows = [np.concatenate([frames[i] for i in idx[r : r + 4]], axis=1) for r in range(0, n, 4)]
    imageio.imwrite(path, np.concatenate(rows, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?")
    ap.add_argument("task", nargs="?", type=int)
    ap.add_argument("init", nargs="?", type=int)
    ap.add_argument("--scenario", default="init")
    ap.add_argument("--mode", default="absolute")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--xvla", nargs=2, metavar=("RUN_DIR", "EPISODE_ID"))
    ap.add_argument("--handover", type=int, default=None, help="steps of the recording before the hand-over")
    ap.add_argument("--video", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()

    recording = None
    if args.xvla:
        run_dir = Path(args.xvla[0])
        recording = load_episode(run_dir, args.xvla[1], load_index(run_dir))
        spec = EpisodeSpec(recording.suite, recording.task, recording.init, recording.scene_seed, "xvla",
                           control_mode=args.mode, restore=not args.no_restore, xvla_run=str(run_dir),
                           xvla_episode=recording.episode_id,
                           handover_step=args.handover if args.handover is not None else len(recording.actions))
    else:
        spec = EpisodeSpec(args.suite, args.task, args.init, args.seed_base + args.init, args.scenario,
                           control_mode=args.mode, restore=not args.no_restore and args.scenario != "init")
    env = make_env(spec.suite, spec.task, image_size=args.size if args.video else 64)
    recorder = Recorder(args.every) if args.video else None
    row = run_episode(env, spec, recording, recorder)
    env.close()
    events = row.pop("events", [])
    print(json.dumps(row, default=str, indent=1))
    for e in events:
        print("  ", e)
    if recorder is not None and recorder.frames:
        video = Path(args.video)
        write_mp4(video, recorder.frames, fps=20 // args.every)
        write_strip(recorder.frames, video.with_name(video.stem + "_strip.png"))
        print("video:", video)


if __name__ == "__main__":
    main()
