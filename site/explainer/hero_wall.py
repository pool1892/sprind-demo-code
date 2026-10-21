"""The picture behind the public page's opening: the agent's two jobs on all 40 locked exam
layouts, one tile per recorded run, graded into the page's dark indigo, with a green light on
each tile for its verdict (every one completed).

    uv run python site/explainer/hero_wall.py

Reads the film's bound data (site/explainer/film/public/data/film-data.json) and its footage,
and writes the wall, eight tiles by five, to site/static at two sizes: hero-wall.jpg (416-pixel
tiles, sharp on a high-density laptop screen) and hero-wall-small.jpg (208-pixel tiles, for
phones), with hero-wall.json, the provenance of every tile: its clip, layout, verdict and the
SHA-256 of the re-render it shows. A tile whose run did not complete, or whose re-render was not
verified against its recording, stops the build.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
DATA = HERE / "film" / "public" / "data" / "film-data.json"
FOOTAGE = HERE / "film" / "public"
STATIC = HERE.parent / "static"

MOSAICS = ("r1-job", "r2-job")  # the saved jobs on the locked layouts, as in the film
AT = 0.45  # the moment shown, as a share of each run: mid-task, the arm at work
COLS = 8
SIZES = {"hero-wall.jpg": 416, "hero-wall-small.jpg": 208}  # file, tile width in pixels
SHADOW, MID, HIGH = (12, 14, 22), (52, 62, 120), (226, 230, 255)  # the grade's three tones
LIGHT = (18, 200, 120)  # completed, the page's green, brightened to read on the dark


def grade(im: Image.Image) -> Image.Image:
    """Luminance mapped onto three tones: the page's near-black, an indigo, a pale blue."""
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    lum = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2])[..., None]
    lum = np.clip((lum - 0.06) / 0.9, 0, 1) ** 1.15
    s, m, h = (np.array(c, dtype=np.float32) / 255.0 for c in (SHADOW, MID, HIGH))
    low = s + (m - s) * np.clip(lum * 2, 0, 1)
    high = m + (h - m) * np.clip(lum * 2 - 1, 0, 1)
    out = np.where(lum < 0.5, low, high)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def frame(clip: Path, at: float, work: Path) -> Image.Image:
    seconds = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(clip)],
            check=True, capture_output=True, text=True,
        ).stdout
    )  # fmt: skip
    still = work / f"{clip.stem}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds * at:.3f}", "-i", str(clip),
         "-frames:v", "1", str(still)],
        check=True,
    )  # fmt: skip
    return Image.open(still).convert("RGB")


def arrange(stills: list[Image.Image], tile: int) -> Image.Image:
    """The graded tiles at `tile` pixels, in rows of COLS, a green light on each; gutters,
    corners and lights in proportion to the 208-pixel drawing."""
    k = tile / 208
    gutter, radius = round(8 * k), round(10 * k)
    rows = -(-len(stills) // COLS)
    width = COLS * tile + (COLS + 1) * gutter
    height = rows * tile + (rows + 1) * gutter
    wall = Image.new("RGBA", (width, height), (*SHADOW, 255))
    lights = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(lights)
    mask = Image.new("L", (tile, tile), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile - 1, tile - 1), radius, fill=255)
    for i, still in enumerate(stills):
        x = gutter + (i % COLS) * (tile + gutter)
        y = gutter + (i // COLS) * (tile + gutter)
        wall.paste(still.resize((tile, tile), Image.Resampling.LANCZOS), (x, y), mask)
        cx, cy, glow, light = x + 18 * k, y + 18 * k, 14 * k, 6 * k
        draw.ellipse((cx - glow, cy - glow, cx + glow, cy + glow), fill=(18, 162, 108, 70))
        draw.ellipse((cx - light, cy - light, cx + light, cy + light), fill=(*LIGHT, 255))
    return Image.alpha_composite(wall, lights.filter(ImageFilter.GaussianBlur(0.6 * k)))


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    tiles = [data["clips"][c] for name in MOSAICS for c in data["mosaics"][name]]
    for t in tiles:
        if not (t.get("delivered") and t.get("verified") and t.get("hash_mismatches") == 0):
            raise SystemExit(f"{t['id']}: not a completed, verified run; the wall shows those")
    largest = max(SIZES.values())  # graded once, at the largest size
    with tempfile.TemporaryDirectory() as tmp:
        frames = [frame(FOOTAGE / t["src"], AT, Path(tmp)) for t in tiles]
    stills = [grade(f.resize((largest, largest), Image.Resampling.LANCZOS)) for f in frames]
    for name, tile in SIZES.items():
        wall = arrange(stills, tile).convert("RGB")
        wall.save(STATIC / name, quality=74, optimize=True, progressive=True)
        print(f"{STATIC / name}: {len(tiles)} tiles of {tile} px, {wall.width}x{wall.height}")
    (STATIC / "hero-wall.json").write_text(
        json.dumps(
            {
                "made_by": "site/explainer/hero_wall.py",
                "shows": "the saved jobs on the locked exam layouts, one tile per recorded run, "
                f"at {AT:.0%} of the run; the green light is the run's verdict",
                "files": {name: f"{px}-pixel tiles, {COLS} across" for name, px in SIZES.items()},
                "tiles": [
                    {"clip": t["id"], "layout": t["start"], "completed": bool(t["delivered"]),
                     "rerender_sha256": t["sha256"]}
                    for t in tiles
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )  # fmt: skip


if __name__ == "__main__":
    main()
