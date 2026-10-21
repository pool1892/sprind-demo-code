"""Paired film: one attempt that failed before a change and the same attempt after it.

The pairs are chosen by a fixed rule, never by looking: in an experiment, the first pair
(by position in a paired arm) where the control arm failed and the changed arm succeeded
on the same task and start; on the reserved panel, the first attempt (by episode id)
rescued between the original and the final state, and the first one broken, if any.

Each pair becomes:
- an animated GIF of both scene videos side by side (the review site renders images; its
  sanitizer drops `<video>`, so a GIF is the moving picture that shows on the page);
- a still strip of six evenly spaced frames from each, one row per attempt;
- the two scene videos as MP4 files, linked from the page (the review site serves them as
  video/mp4 when followed).

The videos are the ones the log writer already wrote (`log/video/<episode>.scene.mp4`).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

__all__ = ["ClipPair", "RenderedClip", "render_pair", "video_path"]

STRIP_FRAMES = 6
SCALE = 2  # 360-pixel videos become 180-pixel tiles
HEADER = 18
GAP = 4
MAX_VIDEO_BYTES = 8 * 2**20
GIF_STEP = 2  # every second frame, each shown twice as long: half the size, same speed
GIF_COLORS = 64  # a palette per frame, no dithering: about 1.2 MB for 14 s of two views


@dataclass(frozen=True)
class ClipPair:
    item: str
    before_log: Path  # the `log/` directory holding the episodes row and its video
    before_row: Mapping[str, Any]
    after_log: Path
    after_row: Mapping[str, Any]
    before_label: str
    after_label: str
    rule: str  # how this pair was chosen
    files: tuple[str, ...]


@dataclass(frozen=True)
class RenderedClip:
    pair: ClipPair
    gif: str | None
    strip: str | None
    videos: tuple[str, ...]
    problems: tuple[str, ...]


def video_path(log: Path, row: Mapping[str, Any], camera: str = "scene") -> Path | None:
    relative = row.get(f"video_{camera}")
    if not relative:
        return None
    path = (log / str(relative)).resolve()
    if not path.is_relative_to(log.resolve()) or not path.is_file():
        return None
    return path


def _frames(path: Path) -> tuple[list[npt.NDArray[np.uint8]], float]:
    reader: Any = imageio.get_reader(path, format="FFMPEG")  # type: ignore[arg-type]
    try:
        fps = float(reader.get_meta_data().get("fps") or 5.0)
        frames = [np.asarray(f, dtype=np.uint8)[::SCALE, ::SCALE, :3] for f in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{path.name} has no frames")
    return frames, fps


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """DejaVu Sans from matplotlib's own files (always installed with it)."""
    import matplotlib

    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(path), 11)
    except OSError:
        return ImageFont.load_default()


FONT = _font()


def _tile(image: npt.NDArray[np.uint8], label: str, color: tuple[int, int, int]) -> Image.Image:
    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w, h + HEADER), (255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, w, HEADER - 2), fill=color)
    draw.text((4, 2), label, fill=(255, 255, 255), font=FONT)
    return canvas


def _side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    canvas = Image.new(
        "RGB", (left.width + GAP + right.width, max(left.height, right.height)), (255, 255, 255)
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + GAP, 0))
    return canvas


BEFORE_COLOR = (82, 81, 78)  # neutral ink: the state before
AFTER_COLOR = (42, 120, 214)  # series blue: the state after


def _outcome(row: Mapping[str, Any]) -> str:
    return str(row.get("outcome", "unknown"))


def render_pair(pair: ClipPair, out: Path, stem: str) -> RenderedClip:
    """Write `<stem>.gif`, `<stem>.strip.png` and the two MP4s under `out`."""
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    before = video_path(pair.before_log, pair.before_row)
    after = video_path(pair.after_log, pair.after_row)
    if before is None or after is None:
        missing = sorted(
            {
                str(r["episode_id"])
                for r, v in ((pair.before_row, before), (pair.after_row, after))
                if v is None
            }
        )
        return RenderedClip(pair, None, None, (), (f"no scene video for {', '.join(missing)}",))
    try:
        a, fps_a = _frames(before)
        b, fps_b = _frames(after)
    except (OSError, ValueError) as error:
        return RenderedClip(pair, None, None, (), (f"a video could not be read ({error})",))
    label_a = f"{pair.before_label}: {_outcome(pair.before_row)}"
    label_b = f"{pair.after_label}: {_outcome(pair.after_row)}"
    fps = fps_a or 5.0
    length = max(len(a), len(b))
    frames = []
    for i in range(length):
        left = _tile(a[min(i, len(a) - 1)], label_a, BEFORE_COLOR)
        right = _tile(b[min(i, len(b) - 1)], label_b, AFTER_COLOR)
        frames.append(_side_by_side(left, right))
    gif = out / f"{stem}.gif"
    duration = round(GIF_STEP * 1000 / (fps_a or fps_b or 5.0))
    shown = [
        f.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for f in frames[::GIF_STEP]
    ]
    shown[0].save(
        gif, save_all=True, append_images=shown[1:], duration=duration, loop=0, optimize=True
    )
    strip_rows = []
    for frames_x, label, color in ((a, label_a, BEFORE_COLOR), (b, label_b, AFTER_COLOR)):
        k = min(STRIP_FRAMES, len(frames_x))
        picks = [round(i * (len(frames_x) - 1) / max(1, k - 1)) for i in range(k)]
        tiles = [
            _tile(frames_x[p], label if j == 0 else f"t = {p / fps:.1f} s", color)
            for j, p in enumerate(picks)
        ]
        row = tiles[0]
        for tile in tiles[1:]:
            row = _side_by_side(row, tile)
        strip_rows.append(row)
    width = max(r.width for r in strip_rows)
    strip = Image.new("RGB", (width, sum(r.height for r in strip_rows) + GAP), (255, 255, 255))
    strip.paste(strip_rows[0], (0, 0))
    strip.paste(strip_rows[1], (0, strip_rows[0].height + GAP))
    strip_file = out / f"{stem}.strip.png"
    strip.save(strip_file, optimize=True)
    videos = []
    for path, which in ((before, "before"), (after, "after")):
        if path.stat().st_size > MAX_VIDEO_BYTES:
            problems.append(f"the {which} video is over {MAX_VIDEO_BYTES // 2**20} MB, not copied")
            continue
        target = out / f"{stem}.{which}.mp4"
        shutil.copyfile(path, target)
        videos.append(target.name)
    return RenderedClip(pair, gif.name, strip_file.name, tuple(videos), tuple(problems))


def first_rescued(
    control: Sequence[Mapping[str, Any] | None], arm: Sequence[Mapping[str, Any] | None]
) -> int | None:
    """The first position where the control attempt failed and the arm's succeeded on the
    same task."""
    for i, (c, a) in enumerate(zip(control, arm, strict=False)):
        if (
            c is not None
            and a is not None
            and c.get("outcome") == "failure"
            and a.get("outcome") == "success"
            and c.get("task_text") == a.get("task_text")
        ):
            return i
    return None
