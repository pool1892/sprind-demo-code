"""Resolving a clip to its recorded video, and the control steps to the video's frames.

Three layouts of a recorded run are understood:

- **fleet log** (`fleetlab.log`): `<run>/log/episodes.parquet` holds the episode's row,
  its `video_scene` (relative to `log/`) and outcome; `log/steps/<episode>.parquet`
  maps each control step to the frame logged at it (`frame_index`, empty between
  frames), and a step without a frame shows the latest earlier one, as the log reader
  does.
- **episode directory** (the jobs spike's runner and its successors): `<run>/<episode>/`
  holds `scene.mp4` with one frame per control step from the reset on (frame 0 is the
  reset), both cameras side by side, scene on the left; and `summary.json` with the
  outcome and the program's steps (`steps[].start_t`, `end_t`, `instruction`; a step is
  in force from its start step up to, not including, its end step; control steps between
  two policy steps that no step claims are shown as a transition motion).
- **jobs run** (`fleetlab.jobs.tables`, lane J): `<run>/log/job_episodes.parquet` holds the
  episode's row (`delivered`, `video_scene` relative to `log/`), `log/robot_steps/
  <episode>.parquet` the frame of each control step, `log/job_step_events.parquet` the
  program's steps with their start and end control steps (a skipped step is listed and
  never in force).
- **video file**: a video named directly, with its frames `steps_per_frame` control steps
  apart (default one).

A re-render is used only for a jobs run, and only with the jobs runner's own report
(`fleetlab.job-rerender/1`): accepted, no hash mismatch, the delivery verdict matched;
for the same run, episode, original video and camera; covering every control step of the
original (its hashes compared for the reset and every step); naming the video beside it,
whose frames fit the report's cadence, whose SHA-256 is the one the report names if it
names one, and which was not written after the report; the original video's SHA-256, where
the report names it (`original_video_sha256`), is the original's. Otherwise the original is shown
and the refusal is recorded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio_ffmpeg  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from fleetlab.film.manifest import ClipRef, CountSpec, ManifestError

__all__ = ["ResolvedClip", "count_value", "frames", "resolve", "sha256"]

_HASHES: dict[tuple[str, int, int], str] = {}


def sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def video_info(path: Path) -> VideoInfo:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        meta: dict[str, Any] = next(reader)
    finally:
        reader.close()
    width, height = meta["size"]
    count, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    return VideoInfo(int(width), int(height), float(meta.get("fps") or 0.0), int(count))


def frames(path: Path, info: VideoInfo) -> Iterator[npt.NDArray[np.uint8]]:
    """The video's frames in order, as height x width x 3 arrays."""
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        next(reader)
        for raw in reader:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
    finally:
        reader.close()


@dataclass(frozen=True)
class ProgramStep:
    index: int
    text: str
    start: int  # first control step in force
    end: int  # last control step in force
    outcome: str | None


@dataclass
class ResolvedClip:
    ref: ClipRef
    layout: str  # fleet-log | episode-dir | video-file
    run: str  # as named in the manifest
    episode: str
    video: Path  # the video the film shows
    video_sha256: str
    info: VideoInfo
    frame_of_step: list[int]  # index: control step from 0; value: frame index
    start: int
    end: int
    crop: tuple[int, int, int, int]
    outcome: str | None
    program: list[ProgramStep] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)  # the source rows, for provenance
    files: dict[str, str] = field(default_factory=dict)  # path -> SHA-256 read
    rerender: dict[str, Any] | None = None  # what happened to a requested re-render
    placeholder: bool = False  # a bare video, without a recorded run
    original: Path | None = None  # the original video, when a re-render is shown

    @property
    def shown_steps(self) -> int:
        return self.end - self.start + 1

    def output_frames(self, fps: int) -> int:
        """Output frames at the film's rate: one control step of 50 ms per 1/20 s at speed 1."""
        steps_per_output = self.ref.speed * 20.0 / fps
        return max(1, int(np.ceil(self.shown_steps / steps_per_output)))

    def step_at(self, k: int, fps: int) -> int:
        step = self.start + int(k * self.ref.speed * 20.0 / fps)
        return min(step, self.end)

    def frame_at(self, k: int, fps: int) -> int:
        step = self.step_at(k, fps)
        return self.frame_of_step[min(step, len(self.frame_of_step) - 1)]

    def program_index(self, step: int) -> int | None:
        for p in self.program:
            if p.start <= step <= p.end:
                return p.index
        if self.program and step > self.program[-1].end:
            return len(self.program)  # past the last step: the program has finished
        return None

    def provenance(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.ref.label,
            "layout": self.layout,
            "run": self.run,
            "episode": self.episode,
            "video": str(self.video),
            "video_sha256": self.video_sha256,
            "video_frames": self.info.frames,
            "video_fps": self.info.fps,
            "control_steps_shown": [self.start, self.end],
            "speed": self.ref.speed,
            "crop": list(self.crop),
            "outcome": self.outcome,
            "files_read": self.files,
            "record": self.record,
        }
        if self.ref.note:
            out["note"] = self.ref.note
        if self.placeholder:
            out["placeholder"] = True
            out["run"] = None
            out["episode"] = None
        if self.rerender is not None:
            out["rerender"] = self.rerender
        if self.original is not None:
            out["original_video"] = str(self.original)
            out["original_sha256"] = sha256(self.original)
        return out


def _read(path: Path, files: dict[str, str]) -> Path:
    if not path.is_file():
        raise ManifestError(f"{path} does not exist")
    files[str(path)] = sha256(path)
    return path


def _log_dir(run: Path, table: str = "episodes.parquet") -> Path | None:
    for candidate in (run / "log", run):
        if (candidate / table).is_file():
            return candidate
    return None


def _frames_of_steps(steps_file: Path, column: str = "step") -> list[int]:
    """Control step -> the frame logged at it or, between frames, the latest earlier one."""
    import pyarrow.parquet as pq

    data = pq.read_table(steps_file, columns=[column, "frame_index"]).to_pydict()
    pairs = sorted(zip(data[column], data["frame_index"], strict=True))
    mapping: list[int] = []
    last = 0
    for step, frame in pairs:
        while len(mapping) < step:
            mapping.append(last)
        if frame is not None:
            last = int(frame)
        mapping.append(last)
    return mapping


def _jobs(ref: ClipRef, log: Path, files: dict[str, str]) -> dict[str, Any]:
    """A run of the jobs runner (`fleetlab.jobs.tables`): `job_episodes`, `job_step_events`,
    `robot_steps/<episode>.parquet` with the frame of each control step."""
    import pyarrow.parquet as pq

    table = _read(log / "job_episodes.parquet", files)
    rows = [r for r in pq.read_table(table).to_pylist() if r.get("episode_id") == ref.episode]
    if not rows:
        raise ManifestError(f"episode {ref.episode} is not in {table}")
    row = rows[0]
    relative = row.get(f"video_{ref.camera}")
    if not relative:
        raise ManifestError(f"episode {ref.episode} in {log} has no {ref.camera} video")
    video = (log / str(relative)).resolve()
    if not video.is_relative_to(log.resolve()):
        raise ManifestError(f"episode {ref.episode}: its video lies outside {log}")
    mapping: list[int] = []
    steps_file = log / "robot_steps" / f"{ref.episode}.parquet"
    if steps_file.is_file():
        mapping = _frames_of_steps(_read(steps_file, files))
    program: list[ProgramStep] = []
    events = log / "job_step_events.parquet"
    if events.is_file():
        _read(events, files)
        mine = [
            e
            for e in pq.read_table(events).to_pylist()
            if e.get("episode_id") == ref.episode and e.get("program_step") is not None
        ]
        for e in sorted(mine, key=lambda e: int(e["program_step"])):
            kind = str(e.get("kind") or "")
            text = str(e.get("instruction") or "") if kind == "policy" else kind
            start, end = e.get("start_step"), e.get("end_step")
            if start is None or end is None:  # skipped: listed, never in force
                program.append(ProgramStep(len(program), text, -1, -2, e.get("outcome")))
            else:
                program.append(
                    ProgramStep(len(program), text, int(start), int(end) - 1, e.get("outcome"))
                )
    delivered = row.get("delivered")
    outcome = None if delivered is None else "delivered" if delivered else "not delivered"
    keep = (
        "job_id", "episode_id", "program_digest", "base_task", "init_state", "control_steps",
        "tail_steps", "delivered", "delivery_detail", "order_kept", "order_detail",
    )  # fmt: skip
    return {
        "video": video,
        "mapping": mapping,
        "outcome": outcome,
        "steps": int(row.get("control_steps") or len(mapping) or 0),
        "record": {k: _plain(row.get(k)) for k in keep if k in row},
        "program": program,
    }


def _fleet(ref: ClipRef, log: Path, files: dict[str, str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = _read(log / "episodes.parquet", files)
    rows = [r for r in pq.read_table(table).to_pylist() if r.get("episode_id") == ref.episode]
    if not rows:
        raise ManifestError(f"episode {ref.episode} is not in {table}")
    row = rows[0]
    relative = row.get(f"video_{ref.camera}")
    if not relative:
        raise ManifestError(f"episode {ref.episode} in {log} has no {ref.camera} video")
    video = (log / str(relative)).resolve()
    if not video.is_relative_to(log.resolve()):
        raise ManifestError(f"episode {ref.episode}: its video lies outside {log}")
    steps_file = log / "steps" / f"{ref.episode}.parquet"
    mapping: list[int] = []
    if steps_file.is_file():
        mapping = _frames_of_steps(_read(steps_file, files))
    keep = (
        "episode_id", "robot_id", "site_id", "day", "task_text", "steps", "outcome",
        "experiment_id", "duration_s",
    )  # fmt: skip
    record = {k: _plain(row.get(k)) for k in keep if k in row}
    return {
        "video": video,
        "mapping": mapping,
        "outcome": row.get("outcome"),
        "steps": int(row.get("steps") or len(mapping) or 0),
        "record": record,
        "program": [],
    }


def _episode_dir(ref: ClipRef, run: Path, files: dict[str, str]) -> dict[str, Any] | None:
    folder = run / str(ref.episode)
    video = folder / f"{ref.camera}.mp4"
    if not video.is_file():
        video = folder / "scene.mp4"
    if not video.is_file():
        return None
    if not (folder / "summary.json").is_file():
        raise ManifestError(f"{folder}: no summary.json, so not a recorded episode")
    summary: dict[str, Any] = json.loads(_read(folder / "summary.json", files).read_text())
    program: list[ProgramStep] = []
    for i, step in enumerate(summary.get("steps") or []):
        start, end = step.get("start_t"), step.get("end_t")
        if start is None or end is None:
            continue
        text = str(
            step.get("instruction") or step.get("stem") or step.get("kind") or f"step {i + 1}"
        )
        if program and int(start) > program[-1].end + 1:
            # control steps between two policy steps that no step claims: a transition motion
            gap = ProgramStep(
                len(program), "transition motion", program[-1].end + 1, int(start) - 1, None
            )
            program.append(gap)
        program.append(
            ProgramStep(len(program), text, int(start), int(end) - 1, step.get("outcome"))
        )
    outcome = summary.get("tag")
    if outcome is None and "delivered" in summary:
        outcome = "delivered" if summary["delivered"] else "not delivered"
    keep = (
        "job", "variant", "start", "base_task", "base_stem", "init_state", "scene_seed",
        "noise_seed", "control_steps", "delivered", "tag", "reason", "end_hash",
    )  # fmt: skip
    return {
        "video": video,
        "mapping": [],  # one frame per control step from the reset
        "outcome": outcome,
        "steps": int(summary.get("control_steps") or 0),
        "record": {k: _plain(summary.get(k)) for k in keep if k in summary},
        "program": program,
    }


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


RERENDER_FORMAT = "fleetlab.job-rerender/1"  # the jobs runner's report (`fleetlab.jobs.replay`)


def _report(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, f"the report could not be read ({error})"
    if not isinstance(record, dict) or record.get("format") != RERENDER_FORMAT:
        return None, f"not a {RERENDER_FORMAT} report from the jobs runner's replay"
    return record, ""


def _expected_frames(steps: int, every: int) -> int:
    """The jobs replay keeps the reset, every `every`-th state and the last."""
    return steps // every + 1 + (1 if steps % every else 0)


def _same(a: object, b: Path) -> bool:
    try:
        return Path(str(a)).expanduser().resolve() == b.expanduser().resolve()
    except (OSError, ValueError):
        return False


def _check_rerender(
    clip: ResolvedClip, video: Path, path: Path
) -> tuple[str | None, dict[str, Any]]:
    """None when the report vouches for this video of this recording, else why not."""
    if clip.layout != "jobs-run":
        return "re-renders are accepted only for jobs runs, from the replay's report", {}
    record, why = _report(path)
    if record is None:
        return why, {}
    if not (
        record.get("accepted") is True
        and record.get("hash_mismatches") == 0
        and record.get("verdict_matches") is True
    ):
        return f"the report says {record.get('label') or 'not accepted'}", record
    run = clip.ref.run
    assert run is not None
    steps = len(clip.frame_of_step) - 1
    checks = [
        (record.get("episode_id") == clip.episode, f"it is for episode {record.get('episode_id')}"),
        (_same(record.get("run_dir"), run), "it is for another run"),
        (_same(record.get("original_video"), clip.video), "it names another original video"),
        (
            record.get("camera", "scene") == clip.ref.camera,
            f"it is for the {record.get('camera')} camera",
        ),
        (record.get("steps") == steps, f"it covers {record.get('steps')} steps of {steps}"),
        (
            record.get("hashes_compared") == steps + 1,
            f"it compared {record.get('hashes_compared')} state hashes of {steps + 1}",
        ),
        (record.get("video") == video.name, f"it names the video {record.get('video')}"),
        (video.parent.resolve() == path.parent.resolve(), "the video is not beside its report"),
    ]
    for ok, problem in checks:
        if not ok:
            return problem, record
    every = record.get("frame_every")
    if not isinstance(every, int) or every < 1:
        return "it names no frame cadence", record
    info = video_info(video)
    if record.get("frames") != info.frames or info.frames != _expected_frames(steps, every):
        return f"the video has {info.frames} frames, the report {record.get('frames')}", record
    if record.get("original_video_sha256") not in (None, sha256(clip.video)):
        return "the original video's SHA-256 is not the one the report names", record
    if record.get("video_sha256") not in (None, sha256(video)):
        return "the video's SHA-256 is not the one the report names", record
    if record.get("video_sha256") is None and video.stat().st_mtime > path.stat().st_mtime + 2:
        return "the video was written after its report", record
    return None, record


def resolve(ref: ClipRef) -> ResolvedClip:
    files: dict[str, str] = {}
    if ref.run is not None:
        run = ref.run.expanduser()
        if not run.exists():
            raise ManifestError(f"run {run} does not exist")
        log = _log_dir(run)
        jobs_log = _log_dir(run, "job_episodes.parquet")
        if jobs_log is not None:
            layout, found = "jobs-run", _jobs(ref, jobs_log, files)
        elif log is not None:
            layout, found = "fleet-log", _fleet(ref, log, files)
        else:
            maybe = _episode_dir(ref, run, files)
            if maybe is None:
                raise ManifestError(
                    f"{run}: neither a fleet log nor an episode directory {ref.episode}"
                )
            layout, found = "episode-dir", maybe
        run_name, episode = str(ref.run), str(ref.episode)
    else:
        assert ref.video is not None
        if not ref.video.is_file():
            raise ManifestError(f"video {ref.video} does not exist")
        layout = "video-file"
        found = {
            "video": ref.video,
            "mapping": [],
            "outcome": None,
            "steps": 0,
            "record": {},
            "program": [],
        }
        # no recorded run: nothing is made up for it; it is a placeholder, and says so
        run_name, episode = str(ref.video), "(a bare video)"
    video: Path = found["video"]
    info = video_info(video)
    if info.frames <= 0:
        raise ManifestError(f"{video} has no frames")
    mapping: list[int] = found["mapping"]
    if not mapping:
        per = ref.steps_per_frame or 1
        steps = found["steps"] or (info.frames - 1) * per
        mapping = [min(s // per, info.frames - 1) for s in range(steps + 1)]
    mapping = [min(f, info.frames - 1) for f in mapping]
    start = ref.start if ref.start is not None else 0
    end = ref.end if ref.end is not None else len(mapping) - 1
    if start >= len(mapping):
        raise ManifestError(
            f"{run_name} {episode}: start step {start} is past its {len(mapping)} steps"
        )
    end = min(end, len(mapping) - 1)
    crop = ref.crop
    if crop is None:
        if layout == "episode-dir" and info.width == 2 * info.height:
            x = 0 if ref.camera == "scene" else info.height
            crop = (x, 0, info.height, info.height)
        else:
            crop = (0, 0, info.width, info.height)
    x, y, w, h = crop
    if x + w > info.width or y + h > info.height or w < 2 or h < 2:
        raise ManifestError(
            f"{run_name} {episode}: crop {crop} is outside the {info.width}x{info.height} video"
        )
    clip = ResolvedClip(
        ref=ref,
        layout=layout,
        run=run_name,
        episode=episode,
        video=video,
        video_sha256=sha256(video),
        info=info,
        frame_of_step=mapping,
        start=start,
        end=end,
        crop=crop,
        outcome=found["outcome"],
        program=found["program"],
        record=found["record"],
        files=files,
    )
    clip.placeholder = layout == "video-file"
    if ref.rerender is not None:
        _apply_rerender(clip)
    return clip


def _apply_rerender(clip: ResolvedClip) -> None:
    rerender = clip.ref.rerender
    assert rerender is not None
    status: dict[str, Any] = {
        "video": str(rerender.video),
        "verification": str(rerender.verification),
    }
    problem: str | None = None
    record: dict[str, Any] = {}
    if not rerender.video.is_file():
        problem = "the re-render video does not exist"
    elif not rerender.verification.is_file():
        problem = "no verification report"
    else:
        status["verification_sha256"] = sha256(rerender.verification)
        status["video_sha256"] = sha256(rerender.video)
        problem, record = _check_rerender(clip, rerender.video, rerender.verification)
    status["used"] = problem is None
    if record:  # what the replay compared, as its report says
        status["comparison"] = {
            k: record.get(k)
            for k in (
                "label", "hashes_compared", "recorded_hashes", "hash_mismatches",
                "verdict_matches", "recorded_end_state_hash", "end_state_hash", "image_size",
                "recorded_image_size", "original_video_sha256",
            )
        }  # fmt: skip
    status["reason"] = problem or (
        "accepted by the replay's report: the same run, episode, original video and camera, "
        "every state hash and the delivery verdict matched"
    )
    if problem is None:
        info = video_info(rerender.video)
        every = int(record["frame_every"])
        steps = len(clip.frame_of_step) - 1
        mapping = [min(s // every, info.frames - 1) for s in range(steps + 1)]
        mapping[-1] = info.frames - 1
        sx, sy = info.width / clip.info.width, info.height / clip.info.height
        x, y, w, h = clip.crop
        clip.original = clip.video
        clip.video = rerender.video
        clip.video_sha256 = sha256(rerender.video)
        clip.crop = (round(x * sx), round(y * sy), round(w * sx), round(h * sy))
        clip.info = info
        clip.frame_of_step = mapping
    clip.rerender = status


# ---- counts --------------------------------------------------------------------------------


def _rows(table: Path) -> list[dict[str, Any]]:
    if table.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_table(table).to_pylist())
    rows = []
    for line in table.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def count_value(spec: CountSpec) -> dict[str, Any]:
    """The count's text on screen and where it came from."""
    if not spec.tables:
        return {"text": spec.text, "detail": spec.detail, "kind": "literal", "source": spec.source}
    rows: list[dict[str, Any]] = []
    for table in spec.tables:
        if not table.is_file():
            raise ManifestError(f"count table {table} does not exist")
        rows += [r for r in _rows(table) if all(r.get(k) == v for k, v in spec.where.items())]
    if spec.key:
        seen: set[tuple[Any, ...]] = set()
        for r in rows:
            identity = tuple(r.get(k) for k in spec.key)
            if identity in seen:
                raise ManifestError(f"count {spec.text!r}: the row {identity} appears twice")
            seen.add(identity)
    if isinstance(spec.success, str):
        k = sum(1 for r in rows if r.get(spec.success) is True)
        rule = f"{spec.success} is true"
    else:
        assert spec.success is not None
        k = sum(1 for r in rows if all(r.get(c) == v for c, v in spec.success.items()))
        rule = " and ".join(f"{c} = {v!r}" for c, v in spec.success.items())
    n = len(rows)
    return {
        "text": spec.text.format(k=k, n=n),
        "detail": None if spec.detail is None else spec.detail.format(k=k, n=n),
        "kind": "computed",
        "k": k,
        "n": n,
        "tables": [{"path": str(t), "sha256": sha256(t)} for t in spec.tables],
        "where": spec.where,
        "key": list(spec.key),
        "success": rule,
        "source": spec.source,
    }
