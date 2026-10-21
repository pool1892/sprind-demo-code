"""Build the tiny made-up dataset the spike's tools serve.

Creates, inside WORKSPACE (what the agent's tools may serve):

    fleet.duckdb             tables episodes, steps, events
    frames/<episode>/<t>.png one frame per second, each showing a 3-digit code

and, inside TRUTH_DIR (which the agent must never see):

    frame_codes.json         the codes, so a test can check the model saw them

The story in the data: robot r3's wrist camera lags from episode e09 on, and
its episodes fail from then. The canned answer of `request_experiment` agrees.
None of this is real; it only gives the tools something to return.

    python make_toy_data.py WORKSPACE TRUTH_DIR
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import duckdb
from PIL import Image, ImageDraw, ImageFont

ROBOTS = ["r1", "r2", "r3"]
SECONDS = 8  # frames at t = 0..7
SIZE = (256, 192)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def draw_frame(path: Path, episode: str, t: int, code: int, failed: bool) -> None:
    img = Image.new("RGB", SIZE, (236, 236, 228))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 150, SIZE[0], SIZE[1]], fill=(120, 110, 100))  # table
    cube_x = 60 + (5 * t if failed else 18 * t)
    d.rectangle([cube_x, 126, cube_x + 24, 150], fill=(200, 40, 40))  # cube
    grip_x = 60 + 18 * t
    d.rectangle([grip_x + 4, 20, grip_x + 20, 110], fill=(40, 60, 160))  # arm
    d.text((8, 6), f"{episode} t={t}s", fill=(0, 0, 0), font=font(14))
    d.text((150, 40), str(code), fill=(0, 0, 0), font=font(40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def main() -> None:
    ws = Path(sys.argv[1])
    truth = Path(sys.argv[2])
    ws.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261006)

    db_path = ws / "fleet.duckdb"
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE episodes (episode_id TEXT, robot_id TEXT, task TEXT,"
        " duration_s DOUBLE, success BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE steps (episode_id TEXT, t DOUBLE, gripper_force_n DOUBLE,"
        " ee_z_m DOUBLE, camera_latency_ms DOUBLE)"
    )
    con.execute(
        "CREATE TABLE events (episode_id TEXT, t DOUBLE, level TEXT, source TEXT,"
        " message TEXT)"
    )

    codes: dict[str, dict[str, int]] = {}
    for i in range(1, 13):
        episode = f"e{i:02d}"
        robot = ROBOTS[(i - 1) % 3]
        lagging = robot == "r3" and i >= 9
        failed = lagging or i == 4  # one unrelated failure as noise
        con.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
            [episode, robot, "pick_cube", float(SECONDS), not failed],
        )
        for step in range(SECONDS * 5):
            t = step / 5
            latency = rng.gauss(180, 15) if lagging else rng.gauss(35, 4)
            force = rng.gauss(2.0 if failed else 7.5, 0.4) if t > 3 else 0.0
            con.execute(
                "INSERT INTO steps VALUES (?, ?, ?, ?, ?)",
                [episode, t, round(force, 2), round(0.30 - 0.02 * t, 3), round(latency, 1)],
            )
        con.execute(
            "INSERT INTO events VALUES (?, 0.0, 'info', 'policy', 'episode start')",
            [episode],
        )
        if lagging:
            con.execute(
                "INSERT INTO events VALUES (?, 2.4, 'warn', 'camera_wrist',"
                " 'frame older than 150 ms')",
                [episode],
            )
        if failed:
            con.execute(
                "INSERT INTO events VALUES (?, 6.0, 'error', 'task', 'cube not lifted')",
                [episode],
            )
        codes[episode] = {}
        for t in range(SECONDS):
            code = rng.randint(100, 999)
            codes[episode][str(t)] = code
            draw_frame(ws / "frames" / episode / f"{t}.png", episode, t, code, failed)
    con.close()
    (truth / "frame_codes.json").write_text(json.dumps(codes, indent=1))
    print(f"wrote {db_path}, {12 * SECONDS} frames, codes in {truth}")


if __name__ == "__main__":
    main()
