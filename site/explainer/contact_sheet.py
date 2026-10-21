"""A contact sheet of a rendered cut: every beat at a few moments, labelled, for review.

    uv run --locked python site/explainer/contact_sheet.py --film <film.mp4> --out sheet.jpg \
        [--at 0.25,0.6,0.92] [--width 480]

Reads the beat starts from film/public/data/timeline.json (the same arithmetic as the
composition) and writes one row per beat.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FPS = 30
T = 14
CUT = {"title"}


def starts(beats):
    out, at = [], 0
    for i, b in enumerate(beats):
        if i > 0 and b["id"] not in CUT:
            at -= T
        out.append(at / FPS)
        at += round(b["seconds"] * FPS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--film", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--at", default="0.25,0.6,0.92", help="fractions of each beat")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--beats", help="comma-separated beat ids (default all)")
    args = ap.parse_args()
    tl = json.loads((HERE / "film" / "public" / "data" / "timeline.json").read_text())
    beats = tl["beats"]
    at = starts(beats)
    fracs = [float(x) for x in args.at.split(",")]
    only = set(args.beats.split(",")) if args.beats else None
    w = args.width
    h = w * 9 // 16
    rows = []
    font = ImageFont.load_default(size=18)
    with tempfile.TemporaryDirectory(dir="/var/tmp") as tmp:
        for b, s in zip(beats, at, strict=True):
            if only and b["id"] not in only:
                continue
            tiles = []
            for f in fracs:
                t = s + b["seconds"] * f
                p = Path(tmp) / f"{b['id']}-{f}.jpg"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        f"{t:.2f}",
                        "-i",
                        str(args.film),
                        "-frames:v",
                        "1",
                        "-vf",
                        f"scale={w}:{h}",
                        str(p),
                    ],
                    check=True,
                )
                tiles.append(
                    Image.open(p).convert("RGB")
                    if p.is_file()
                    else Image.new("RGB", (w, h), "grey")
                )
            rows.append((b["id"], s, tiles))
    label_w = 170
    sheet = Image.new("RGB", (label_w + len(fracs) * (w + 6), len(rows) * (h + 6)), "white")
    d = ImageDraw.Draw(sheet)
    for r, (bid, s, tiles) in enumerate(rows):
        y = r * (h + 6)
        d.text((8, y + 8), bid, fill="black", font=font)
        d.text((8, y + 32), f"{int(s // 60)}:{s % 60:04.1f}", fill="grey", font=font)
        for c, im in enumerate(tiles):
            sheet.paste(im, (label_w + c * (w + 6), y))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=85)
    print(f"contact sheet: {len(rows)} beats -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
