from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from fleetlab.video import write_mp4


def test_write_mp4_keeps_frame_count_and_size(tmp_path: Path) -> None:
    # 360 is not a multiple of 16: the size must survive anyway.
    frames = [np.full((360, 720, 3), 10 * i, dtype=np.uint8) for i in range(5)]
    path = tmp_path / "nested" / "clip.mp4"

    write_mp4(path, frames, fps=20)

    decoded = imageio.mimread(path)
    assert len(decoded) == 5
    assert decoded[0].shape == (360, 720, 3)
    assert abs(int(decoded[4].mean()) - 40) <= 2


def test_write_mp4_refuses_an_empty_episode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        write_mp4(tmp_path / "clip.mp4", [], fps=20)
