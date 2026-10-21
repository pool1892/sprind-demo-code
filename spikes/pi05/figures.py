"""The pictures and the camera-fault check of the report. Reads only run outputs; needs av, numpy, Pillow.

    python figures.py sheet RUN EPISODE_ID OUT.png     8 scene-camera frames of an episode side by side, half size
    python figures.py pair RUN_A EP_A RUN_B EP_B OUT.png   the first scene image of two episodes side by side
    python figures.py camcheck                         does the camera fault reach the policy? (results/cam_fault_reaches_policy.txt)

These were first written as one-off commands during the spike and collected
here afterwards; they produce the same files.

camcheck compares robot 2's 40 healthy episodes in `h200v_pi05_r1w3` with
their twins in `c6av_pi05_r1w3`, which differ only by the camera fault: after
the reset, the scene image must differ and the wrist image, the simulator state
and the furniture must not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RUNS = Path.home() / ".cache" / "robots-probes" / "pi05" / "runs"
RESULTS = Path.home() / ".cache" / "robots-probes" / "pi05" / "results"


def frames(run: str, eid: str) -> list:
    import av

    return [f.to_ndarray(format="rgb24") for f in av.open(str(RUNS / run / "log" / "video" / f"{eid}.scene.mp4")).decode(video=0)]


def sheet(run: str, eid: str, out: str) -> None:
    from PIL import Image

    fr = frames(run, eid)
    idx = np.linspace(0, len(fr) - 1, 8).astype(int)
    row = np.concatenate([fr[i] for i in idx], axis=1)
    Image.fromarray(row).resize((row.shape[1] // 2, row.shape[0] // 2)).save(out)
    print(f"{out}: {len(fr)} frames, shown {list(idx)}")


def pair(run_a: str, ep_a: str, run_b: str, ep_b: str, out: str) -> None:
    from PIL import Image

    Image.fromarray(np.concatenate([frames(run_a, ep_a)[0], frames(run_b, ep_b)[0]], axis=1)).save(out)
    print(out)


def camcheck() -> None:
    n = scene = wrist = fix = state = 0
    for t in range(10):
        for i in range(4):
            a = np.load(RUNS / "h200v_pi05_r1w3" / "truth" / "chain" / f"h2-t{t}-i{i}.npz")
            b = np.load(RUNS / "c6av_pi05_r1w3" / "truth" / "chain" / f"c6-t{t}-i{i}.npz")
            n += 1
            scene += a["img_hash"][0][0] != b["img_hash"][0][0]
            wrist += a["img_hash"][0][1] == b["img_hash"][0][1]
            fix += a["fixtures"][0] == b["fixtures"][0]
            state += a["state_hash"][0] == b["state_hash"][0]
    line = f"{n} scene differs {scene} wrist same {wrist} fixtures same {fix} state same {state}"
    (RESULTS / "cam_fault_reaches_policy.txt").write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"sheet": sheet, "pair": pair, "camcheck": camcheck}[cmd](*args)
