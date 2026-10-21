"""Video files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt

# ffmpeg options that make an encode repeatable: one encoder thread, no version strings or
# other metadata in the container. Two encodes of the same frames then give the same bytes
# (the skeleton's differing files came through PyAV, see src/fleetlab/log/README.md).
REPEATABLE_PARAMS: tuple[str, ...] = (
    "-threads",
    "1",
    "-fflags",
    "+bitexact",
    "-flags:v",
    "+bitexact",
    "-map_metadata",
    "-1",
)


def write_mp4(
    path: Path,
    frames: Sequence[npt.NDArray[np.uint8]],
    fps: int,
    *,
    crf: int | None = None,
    preset: str | None = None,
    repeatable: bool = False,
) -> None:
    """Write RGB frames (height x width x 3, even height and width) as an H.264 MP4.

    `crf` and `preset` are libx264's settings; without `crf` imageio's default quality
    applies. `repeatable` encodes with one thread and without metadata, so that the same
    frames give the same file.
    """
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[str] = []
    if crf is not None:
        params += ["-crf", str(crf)]
    if preset is not None:
        params += ["-preset", preset]
    if repeatable:
        params += list(REPEATABLE_PARAMS)
    # macro_block_size=1 keeps the frame size; the default pads it to a multiple of 16.
    imageio.mimwrite(
        path,
        list(frames),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        quality=None if crf is not None else 5,
        ffmpeg_params=params or None,
    )
