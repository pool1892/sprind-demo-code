"""ffmpeg: frames in, H.264 and AAC MP4 files out; audio decoded and mixed.

There is no system ffmpeg on this machine; the binary that `imageio_ffmpeg` ships in the
environment is used (7.0.2 on 19 October 2026: libx264 and the native AAC encoder, no
drawtext, which is why every frame is drawn in Python and piped in raw).

One encoder process writes every output from the same frames: the full-resolution master
and the deliverable for the web, 1280 x 720 with its video bitrate capped (`WEB_MAXRATE`)
so that a three-minute film stays under 80 MB and far under GitHub Pages' 100 MB a file.
`+faststart` puts the index at the front so a browser starts playing before the download
ends. With `repeatable` the encoders run on one thread with bit-exact flags and no
metadata (`fleetlab.video.REPEATABLE_PARAMS`), so the same frames give the same bytes.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from fleetlab.video import REPEATABLE_PARAMS

__all__ = ["AUDIO_RATE", "Encoder", "Output", "decode_audio", "ffmpeg", "write_wav"]

AUDIO_RATE = 48_000
WEB_MAXRATE_KBPS = 3000  # video; with 128 kb/s audio at most about 70 MB for three minutes
WEB_LIMIT_BYTES = 100 * 10**6  # GitHub Pages refuses files over 100 MB
WEB_TARGET_BYTES_3MIN = 80 * 10**6


def ffmpeg() -> str:
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg(), "-version"], capture_output=True, text=True, check=False)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"


@dataclass(frozen=True)
class Output:
    path: Path
    width: int
    height: int
    crf: int
    preset: str
    maxrate_kbps: int | None  # a cap on the video bitrate, for the web deliverable
    audio_kbps: int
    role: str  # master | web | draft


def write_wav(path: Path, samples: npt.NDArray[np.int16]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_RATE)
        handle.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())


def decode_audio(path: Path) -> npt.NDArray[np.int16]:
    """An audio file as mono 16-bit samples at `AUDIO_RATE`."""
    out = subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_RATE),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(out.stdout, dtype="<i2").astype(np.int16)


class Encoder:
    """Raw RGB frames on stdin, the audio track from a WAV file, every output in one pass."""

    def __init__(
        self,
        outputs: list[Output],
        width: int,
        height: int,
        fps: int,
        audio: Path,
        *,
        repeatable: bool = False,
        threads: int = 4,
    ) -> None:
        self.width, self.height = width, height
        self.frames = 0
        cmd = [
            ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-i", str(audio),
        ]  # fmt: skip
        for out in outputs:
            cmd += ["-map", "0:v", "-map", "1:a"]
            if (out.width, out.height) != (width, height):
                cmd += ["-vf", f"scale={out.width}:{out.height}:flags=lanczos"]
            cmd += [
                "-c:v", "libx264", "-preset", out.preset, "-crf", str(out.crf),
                "-pix_fmt", "yuv420p", "-profile:v", "high",
                "-g", str(fps * 4),
            ]  # fmt: skip
            if out.maxrate_kbps:
                cmd += ["-maxrate", f"{out.maxrate_kbps}k", "-bufsize", f"{2 * out.maxrate_kbps}k"]
            cmd += ["-c:a", "aac", "-b:a", f"{out.audio_kbps}k", "-ar", str(AUDIO_RATE)]
            if repeatable:
                cmd += [*REPEATABLE_PARAMS, "-flags:a", "+bitexact"]
            else:
                cmd += ["-threads", str(threads), "-map_metadata", "-1"]
            cmd += ["-movflags", "+faststart", str(out.path)]
        self.command = cmd
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert self._process.stdin is not None
        self._stdin: IO[bytes] = self._process.stdin

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame of {frame.shape}, expected {(self.height, self.width, 3)}")
        self._stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        self._stdin.close()
        code = self._process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with {code}")

    def abort(self) -> None:
        try:
            self._stdin.close()
        finally:
            self._process.kill()
            self._process.wait()


def probe(path: Path) -> dict[str, str]:
    """The streams ffmpeg reports for a file (codec lines), for provenance."""
    out = subprocess.run(
        [ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, check=False
    )
    lines = [ln.strip() for ln in out.stderr.splitlines() if "Stream #" in ln or "Duration:" in ln]
    return {f"line{i}": ln for i, ln in enumerate(lines)}
