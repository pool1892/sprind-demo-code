"""Assembling the film from a manifest: timeline, frames, narration track, files beside it.

`build(manifest, out)` writes into `out`:

- `film.mp4`: the deliverable, 1280 x 720, H.264 and AAC, bitrate capped for the web;
- `film-master.mp4`: the same film at 1920 x 1080 without the cap (not with `draft`);
- `captions.vtt` (the narration as captions, also burned in), `chapters.vtt`,
  `chapters.json` (each beat's start, end and footage status);
- `poster.jpg`, and `timeline.png` with `timeline.md`: one picture and one row per beat,
  saying which beats show the record's footage, placeholders, or no footage yet;
- `manifest.json` (the manifest as used) and `provenance.json`: every output with its
  SHA-256, every clip with its run, episode, control steps and the SHA-256 of the video and
  of every record read to find it, every count with the table it was computed from.

A beat lasts as long as its shots, or as long as its narration with a lead-in and a tail,
whichever is longer; the extra time holds the last shot. Narration audio comes from a
directory of `<beat id>.mp3` files (`tools/film-narrate`); without one the captions are
timed at a normal reading pace and the sound track is silent.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from fleetlab.film import compose, encode, narration
from fleetlab.film.compose import Canvas, PanelState, TileSpec
from fleetlab.film.manifest import Beat, Manifest, ManifestError, Panel, Shot
from fleetlab.film.public import short_ref
from fleetlab.film.sources import ResolvedClip, VideoInfo, count_value, frames, resolve, sha256

__all__ = ["BuildResult", "build", "plan"]

LEAD_S = 0.5  # narration starts this long into its beat
TAIL_S = 0.8  # and the beat runs on this long after it ends
MASTER = (1920, 1080)
WEB = (1280, 720)
DRAFT = (640, 360)


@dataclass
class ShotPlan:
    beat: Beat
    shot: Shot
    clips: list[ResolvedClip]
    panel: Panel
    counts: list[dict[str, Any]]
    frames: int
    start: int = 0
    files: dict[str, str] = field(default_factory=dict)  # stills and listings read
    text: str | None = None  # a listing's text
    messages: list[tuple[str, str]] = field(default_factory=list)  # who, what
    calls: list[dict[str, Any]] = field(default_factory=list)  # the transcript items shown
    picture: Image.Image | None = None


@dataclass
class BeatPlan:
    beat: Beat
    shots: list[ShotPlan]
    spoken: narration.Beat | None
    audio: Path | None
    samples: npt.NDArray[np.int16] | None
    narration_s: float
    start: int = 0
    frames: int = 0
    cues: list[narration.Cue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.beat.missing:
            return "no footage"
        if any(s.shot.kind == "slate" for s in self.shots):
            return (
                "placeholder, partly no footage" if self.beat.placeholder else "partly no footage"
            )
        if self.beat.placeholder:
            return "placeholder"
        if all(s.shot.kind == "card" for s in self.shots):
            return "card"
        return "record"


@dataclass
class BuildResult:
    out: Path
    outputs: list[Path]
    seconds: float
    beats: list[BeatPlan]
    warnings: list[str]
    provenance: dict[str, Any]


def plan(
    manifest: Manifest,
    *,
    narration_dir: Path | None = None,
    beats: set[str] | None = None,
    story: str | None = None,
) -> tuple[list[BeatPlan], list[str]]:
    """Resolve every clip and count and lay the beats out on the timeline (no rendering)."""
    from fleetlab.film import public

    public.configure(manifest.aliases, list(manifest.forbid))
    fps = manifest.fps
    warnings: list[str] = []
    script: dict[str, narration.Beat] = {}
    if manifest.script is not None:
        if manifest.script.is_file():
            script = {b.id: b for b in narration.load_script(manifest.script)}
        else:
            warnings.append(f"the narration script {manifest.script} does not exist")
    voice = _voice(narration_dir) if narration_dir is not None else None
    problems: list[str] = []
    plans: list[BeatPlan] = []
    frame = 0
    # numbers on cards and labels come from the bound script's bindings, never typed
    bound = (_bindings(manifest.script) or {}) if manifest.script is not None else {}
    values = {k: str(v["value"]) for k, v in (bound.get("values") or {}).items()}
    for beat in manifest.beats:
        if beats is not None and beat.id not in beats:
            continue
        if story is not None and beat.story not in (None, story):
            continue
        shots: list[ShotPlan] = []
        for shot in beat.shots:
            shot = _filled(shot, values)
            clips = [resolve(ref) for ref in shot.clips]
            for clip in clips:
                if clip.rerender is not None and not clip.rerender["used"]:
                    warnings.append(
                        f"beat {beat.id}: re-render refused for {clip.episode} "
                        f"({clip.rerender['reason']}); the original is shown"
                    )
            panel = shot.panel or beat.panel
            counts = [count_value(c) for c in panel.counts]
            if shot.kind in ("clip", "pair"):
                n = max(c.output_frames(fps) + round(c.ref.hold * fps) for c in clips)
            else:
                assert shot.seconds is not None
                n = max(1, round(shot.seconds * fps))
            sp = ShotPlan(beat, shot, clips, panel, counts, n)
            if shot.kind == "image":
                assert shot.image is not None
                if not shot.image.is_file():
                    raise FileNotFoundError(f"beat {beat.id}: image {shot.image} does not exist")
                sp.files[str(shot.image)] = sha256(shot.image)
                sp.picture = Image.open(shot.image).convert("RGB")
            if shot.kind in ("text", "messages", "calls") and shot.file is not None:
                if not shot.file.is_file():
                    raise FileNotFoundError(f"beat {beat.id}: file {shot.file} does not exist")
                sp.files[str(shot.file)] = sha256(shot.file)
                value: Any = shot.file.read_text(encoding="utf-8", errors="replace")
                if shot.field is not None:
                    value = _field(json.loads(value), shot.field, f"beat {beat.id}")
                if shot.kind == "calls":
                    sp.messages, sp.calls = _calls(shot, f"beat {beat.id}")
                elif shot.kind == "text":
                    if not isinstance(value, str):
                        value = json.dumps(value, indent=2, ensure_ascii=False)
                    if shot.lines is not None:
                        first, last = shot.lines
                        value = "\n".join(value.splitlines()[first - 1 : last])
                    sp.text = value
                else:
                    sp.messages = _messages(value, shot.lines, f"beat {beat.id}", shot.text_key)
            elif shot.kind == "text":
                sp.text = shot.text
            elif shot.kind == "messages":
                sp.messages = list(shot.messages)
            shots.append(sp)
        spoken = script.get(beat.id) if beat.narrated else None
        if beat.narrated and script and spoken is None:
            warnings.append(f"beat {beat.id} has no narration in the script")
        if spoken is not None and spoken.unbound:
            names = ", ".join("{" + n + "}" for n in spoken.unbound)
            warnings.append(f"beat {beat.id} narrates unbound number(s) {names}")
        audio: Path | None = None
        samples = None
        seconds = 0.0
        if spoken is not None and spoken.text:
            if voice is not None and narration_dir is not None:
                audio = _checked_audio(narration_dir, voice, spoken, problems, warnings)
            if audio is not None:
                samples = encode.decode_audio(audio)
                seconds = len(samples) / encode.AUDIO_RATE
            else:
                seconds = narration.spoken_seconds(spoken)
        total = sum(s.frames for s in shots)
        needed = round((LEAD_S + seconds + TAIL_S) * fps) if seconds else 0
        if needed > total:
            shots[-1].frames += needed - total  # the last shot holds
            total = needed
        bp = BeatPlan(beat, shots, spoken, audio, samples, seconds, start=frame, frames=total)
        at = frame
        for s in shots:
            s.start = at
            at += s.frames
        if spoken is not None and seconds:
            bp.cues = narration.cues(spoken, seconds, offset=frame / fps + LEAD_S)
        plans.append(bp)
        frame += total
    if problems:
        raise NarrationError("the narration does not match the script:\n  " + "\n  ".join(problems))
    if not plans:
        raise ValueError("no beats selected")
    hits = [f"{where}: {', '.join(sorted(set(found)))}" for where, found in _forbidden(plans)]
    if hits:
        raise ManifestError("the film would show names this cut forbids:\n  " + "\n  ".join(hits))
    return plans, warnings


def _forbidden(plans: list[BeatPlan]) -> list[tuple[str, list[str]]]:
    """Every text the film draws or speaks, checked against the cut's `forbid` patterns."""
    from fleetlab.film.public import forbidden

    out = []
    for bp in plans:
        b = bp.beat
        texts = [b.title, b.subtitle or "", b.placeholder or "", b.needs or ""]
        if bp.spoken is not None:
            texts.append(bp.spoken.text)
        for sp in bp.shots:
            sh = sp.shot
            texts += [sh.title or "", sh.subtitle or "", sh.label or "", sh.note or ""]
            texts += [sp.text or ""] + [f"{a} {t}" for a, t in sp.messages]
            texts += [str(sh.file.name) if sh.file else ""]
            pn = sp.panel
            texts += [*pn.requirement, pn.rule or "", pn.bar or "", pn.program or "", *pn.steps]
            texts += [f"{c.get('text')} {c.get('detail')} {c.get('source')}" for c in sp.counts]
            for clip in sp.clips:
                texts += [clip.ref.label, clip.ref.note or "", *(p.text for p in clip.program)]
            texts += [seg for seg, _ in _source_segments(sp)]
        found = [h for t in texts for h in forbidden(t)]
        if found:
            out.append((b.id, found))
    return out


class NarrationError(ValueError):
    pass


def _voice(folder: Path) -> dict[str, Any]:
    """`voice.json` of a narration folder (`tools/film-narrate render`)."""
    path = folder / "voice.json"
    if not path.is_file():
        raise NarrationError(f"{folder} has no voice.json: not a recording by tools/film-narrate")
    record = json.loads(path.read_text())
    if record.get("format") != narration.VOICE_FORMAT:
        raise NarrationError(f"{path} is not a {narration.VOICE_FORMAT} record")
    return dict(record)


def _checked_audio(
    folder: Path,
    voice: dict[str, Any],
    spoken: narration.Beat,
    problems: list[str],
    warnings: list[str],
) -> Path | None:
    """The beat's recording if voice.json names it and both digests match; a recording
    that voice.json does not name, or that was made from other words or changed since, is
    a problem that stops the build."""
    entry = (voice.get("beats") or {}).get(spoken.id)
    if entry is None:
        stray = [
            p.name
            for p in (folder / f"{spoken.id}.mp3", folder / f"{spoken.id}.wav")
            if p.is_file()
        ]
        if stray:
            problems.append(f"{spoken.id}: {', '.join(stray)} is not in voice.json")
        else:
            warnings.append(f"beat {spoken.id} has no recorded voice; captions only")
        return None
    audio = folder / str(entry.get("audio", ""))
    if entry.get("text_sha256") != narration.text_digest(spoken):
        problems.append(f"{spoken.id}: recorded from other words than the script's (re-record it)")
        return None
    if not audio.is_file():
        problems.append(f"{spoken.id}: {audio.name} named in voice.json is missing")
        return None
    if sha256(audio) != entry.get("audio_sha256"):
        problems.append(f"{spoken.id}: {audio.name} changed since it was recorded")
        return None
    return audio


# ---- drawing --------------------------------------------------------------------------------


AUTHORS = {"lead": "Shift lead", "agent": "Agent", "host": "Host", "operator": "Operator"}


def _field(data: Any, path: str, where: str) -> Any:
    """A dotted path into JSON: object keys, and list indexes as numbers."""
    value = data
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif (
            isinstance(value, list)
            and part.lstrip("-").isdigit()
            and -len(value) <= int(part) < len(value)
        ):
            value = value[int(part)]
        else:
            raise ValueError(f"{where}: {path!r} does not lead anywhere in the file (at {part!r})")
    return value


def _messages(
    value: Any, lines: tuple[int, int] | None, where: str, text_key: str | None = None
) -> list[tuple[str, str]]:
    """Messages from a record: a list of {author|from, text}; `lines` picks some, 1-based."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{where}: the messages are not a list")
    if lines is not None:
        value = value[lines[0] - 1 : lines[1]]
    out = []
    for m in value:
        if not isinstance(m, dict):
            raise ValueError(f"{where}: a message is not an object")
        who = str(m.get("author") or m.get("from") or m.get("kind") or "")
        text = m.get(text_key) if text_key else m.get("text")
        if (
            text is None and not text_key and "name" in m
        ):  # a library index entry: its name and purpose
            text = str(m["name"]) + (f": {m['purpose']}" if m.get("purpose") else "")
        out.append((AUTHORS.get(who, who), str(text or "")))
    if not out:
        raise ValueError(f"{where}: no messages selected")
    return out


def _call_text(arguments: Any) -> str:
    """A tool call's arguments as one short line: its SQL, its name and parameters, or JSON."""
    if not isinstance(arguments, dict):
        return " ".join(str(arguments).split())[:300]
    if isinstance(arguments.get("sql"), str):
        head = f"{arguments['name']}: " if isinstance(arguments.get("name"), str) else ""
        return (head + " ".join(arguments["sql"].split()))[:420]
    spec = arguments.get("check") or arguments.get("spec")
    if isinstance(spec, dict) and isinstance(spec.get("sql"), str):
        return (f"{spec.get('name', '')}: " + " ".join(spec["sql"].split()))[:420]
    return " ".join(json.dumps(arguments, ensure_ascii=False).split())[:300]


def _calls(shot: Any, where: str) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """The agent's completed tool calls from its session transcript (Codex `--json` events:
    `item.completed` items of type `mcp_tool_call`), optionally only some tools; `lines`
    picks among them, 1-based. A refused or failed call says so."""
    items: list[dict[str, Any]] = []
    for line in shot.file.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "mcp_tool_call"
            and (not shot.tools or item.get("tool") in shot.tools)
        ):
            items.append(item)
    if shot.lines is not None:
        items = items[shot.lines[0] - 1 : shot.lines[1]]
    if not items:
        raise ValueError(f"{where}: no tool calls selected from {shot.file}")
    messages, shown = [], []
    for item in items:
        text = _call_text(item.get("arguments"))
        if item.get("error") or item.get("status") not in (None, "completed"):
            text += f"  (not completed: {str(item.get('error') or item.get('status'))[:120]})"
        messages.append((f"Agent · {item.get('tool')}", text))
        shown.append({"id": item.get("id"), "tool": item.get("tool")})
    return messages, shown


def _count_source(count: dict[str, Any], viewer: bool = False) -> str | None:
    """Where a count on screen comes from: its tables by public identifier and the first
    eight hex digits of their SHA-256 (the full ones are in the provenance), or the source a
    literal count states. A viewer cut leaves the tables out of the frame."""
    tables = count.get("tables") or []
    if tables and viewer:
        return None
    if tables:
        return "from " + ", ".join(f"{short_ref(t['path'], 3)} {t['sha256'][:8]}" for t in tables)
    if count.get("source"):
        return f"source: {count['source']}"
    return None


def _panel_state(sp: ShotPlan, k: int, fps: int, viewer: bool = False) -> PanelState:
    panel = sp.panel
    steps = panel.steps
    current: int | None = None
    outcomes: tuple[str | None, ...] = ()
    if panel.current == "auto" and sp.clips:
        clip = sp.clips[min(panel.follow, len(sp.clips) - 1)]
        if not steps:
            steps = tuple(p.text for p in clip.program)
        frames_shown = clip.output_frames(fps)
        step = clip.step_at(min(k, frames_shown - 1), fps)
        current = clip.program_index(step)
        outcomes = tuple(p.outcome for p in clip.program)
    elif isinstance(panel.current, int):
        current = panel.current
    counts = tuple((str(c["text"]), c.get("detail"), _count_source(c, viewer)) for c in sp.counts)
    return PanelState(
        requirement=panel.requirement,
        rule=panel.rule,
        bar=panel.bar,
        program=panel.program,
        steps=steps,
        current=current,
        outcomes=outcomes,
        counts=counts,
    )


def _caption(cues: list[narration.Cue], t: float) -> str | None:
    for cue in cues:
        if cue.start <= t < cue.end:
            return cue.text
    return None


def _outcome_text(clip: ResolvedClip) -> str:
    if clip.outcome is None:
        return ""
    return str(clip.outcome).replace("_", " ")


def _tile_specs(sp: ShotPlan) -> list[TileSpec]:
    specs = []
    for i, clip in enumerate(sp.clips):
        label = clip.ref.label or clip.episode
        outcome = _outcome_text(clip)
        if outcome:
            label = f"{label}: {outcome}"
        if clip.original is not None:
            label += "  ·  re-render"
        if sp.shot.kind == "pair":
            color = compose.BEFORE if i == 0 else compose.AFTER
        else:
            color = compose.SERIES
        specs.append(TileSpec(label, color))
    return specs


def _badges(sp: ShotPlan) -> list[tuple[str, compose.RGB]]:
    badges: list[tuple[str, compose.RGB]] = []
    if sp.shot.kind == "slate":
        badges.append(("NO FOOTAGE YET", compose.MISSING))
    elif sp.beat.placeholder and sp.shot.kind != "card":
        badges.append(("PLACEHOLDER", compose.PLACEHOLDER))
    if any(c.original is not None for c in sp.clips):
        badges.append(("RE-RENDER", compose.SERIES))
    if any(c.rerender is not None and not c.rerender["used"] for c in sp.clips):
        badges.append(("ORIGINAL: RE-RENDER REFUSED", compose.MISSING))
    return badges


def _source_segments(sp: ShotPlan, viewer: bool = False) -> list[tuple[str, compose.RGB]]:
    """The line under the shot. A viewer cut drops the run ids, steps and digests and keeps
    the placeholder, missing-footage and note texts."""
    segments: list[tuple[str, compose.RGB]] = []
    if sp.beat.placeholder:
        segments.append((f"Placeholder: {sp.beat.placeholder}.", compose.PLACEHOLDER))
    if sp.clips and not viewer:
        parts = []
        for i, clip in enumerate(sp.clips):
            who = ("before: " if i == 0 else "after: ") if len(sp.clips) == 2 else ""
            parts.append(
                f"{who}{short_ref(clip.run)} · {clip.episode} · steps {clip.start}-{clip.end}"
                f" · {clip.video_sha256[:10]}"
            )
        segments.append(("   ".join(parts), compose.DIM))
    elif sp.files and not viewer:
        segments.append(
            ("  ".join(f"{Path(p).name} · {h[:10]}" for p, h in sp.files.items()), compose.DIM)
        )
    elif sp.shot.kind == "slate":
        segments.append(("No recorded run yet.", compose.DIM))
    if sp.shot.note:
        segments.append((sp.shot.note, compose.DIM))
    for clip in sp.clips:
        if clip.ref.note:
            segments.append((clip.ref.note, compose.DIM))
    return segments


def _base(
    canvas: Canvas,
    manifest: Manifest,
    sp: ShotPlan,
    state: PanelState,
    caption: str | None,
    shown: int,
) -> npt.NDArray[np.uint8]:
    beat, shot = sp.beat, sp.shot
    if shot.kind == "card":
        image = canvas.card(shot.title or beat.title, shot.subtitle, shot.note, beat.story)
        canvas.caption(image, caption)
        return np.asarray(image, dtype=np.uint8)
    image = canvas.blank()
    wide = state.empty
    canvas.header(
        image, beat.story, manifest.story_name(beat.story), beat.title, beat.subtitle, _badges(sp)
    )
    if shot.kind in ("clip", "pair"):
        canvas.tile_headers(image, canvas.tiles(len(sp.clips), wide), _tile_specs(sp))
    elif shot.kind == "slate":
        canvas.slate(image, beat.needs, beat.title, wide)
    elif shot.kind == "image":
        assert sp.picture is not None
        canvas.still(image, sp.picture, shot.label, wide)
    elif shot.kind == "text":
        title = shot.title or (shot.file.name if shot.file else None)
        canvas.listing(image, title, sp.text or "", wide)
    elif shot.kind in ("messages", "calls"):
        canvas.messages(image, sp.messages, shown, wide)
    canvas.side_panel(image, state)
    canvas.source_line(image, _source_segments(sp, manifest.sources_on_screen == "viewer"))
    canvas.caption(image, caption)
    return np.asarray(image, dtype=np.uint8)


class _Stream:
    """A clip's frames read forward as the film needs them."""

    def __init__(self, clip: ResolvedClip) -> None:
        self.clip = clip
        info: VideoInfo = clip.info
        self._it: Iterator[npt.NDArray[np.uint8]] = frames(clip.video, info)
        self.index = -1
        self.frame: npt.NDArray[np.uint8] | None = None
        self._tile: tuple[int, int, npt.NDArray[np.uint8]] | None = None

    def at(self, index: int) -> npt.NDArray[np.uint8]:
        while self.index < index:
            try:
                self.frame = next(self._it)
            except StopIteration:
                break
            self.index += 1
        assert self.frame is not None
        return self.frame

    def tile(self, index: int, side: int) -> npt.NDArray[np.uint8]:
        if self._tile is not None and self._tile[0] == index and self._tile[1] == side:
            return self._tile[2]
        tile = compose.fit_tile(self.at(index), self.clip.crop, side)
        self._tile = (index, side, tile)
        return tile

    def close(self) -> None:
        close = getattr(self._it, "close", None)
        if close is not None:
            close()


def _render(
    plans: list[BeatPlan],
    manifest: Manifest,
    canvas: Canvas,
    encoder: encode.Encoder,
    samples: dict[int, str],
) -> dict[str, Image.Image]:
    fps = manifest.fps
    viewer = manifest.sources_on_screen == "viewer"
    grabbed: dict[str, Image.Image] = {}
    for bp in plans:
        for sp in bp.shots:
            streams = [_Stream(c) for c in sp.clips]
            wide = _panel_state(sp, 0, fps, viewer).empty
            boxes = canvas.tiles(len(sp.clips), wide) if sp.clips else []
            cache: dict[tuple[PanelState, str | None, int], npt.NDArray[np.uint8]] = {}
            n_messages = len(sp.messages)
            try:
                for k in range(sp.frames):
                    index = sp.start + k
                    t = index / fps
                    state = _panel_state(sp, k, fps, viewer)
                    caption = _caption(bp.cues, t)
                    shown = 0
                    if n_messages:
                        spread = max(1, int(sp.frames * 0.7))
                        shown = min(n_messages, 1 + k * n_messages // spread)
                    key = (state, caption, shown)
                    base = cache.get(key)
                    if base is None:
                        if len(cache) > 256:
                            cache.clear()
                        base = cache[key] = _base(canvas, manifest, sp, state, caption, shown)
                    frame = base
                    if streams:
                        frame = base.copy()
                        for stream, box in zip(streams, boxes, strict=True):
                            clip = stream.clip
                            last = clip.output_frames(fps) - 1
                            source = clip.frame_at(min(k, last), fps)
                            frame[box.y : box.y + box.h, box.x : box.x + box.w] = stream.tile(
                                source, box.w
                            )
                    encoder.write(frame)
                    if index in samples:
                        grabbed[samples[index]] = Image.fromarray(frame)
            finally:
                for stream in streams:
                    stream.close()
    return grabbed


# ---- files beside the film ------------------------------------------------------------------


def _audio_track(plans: list[BeatPlan], total_frames: int, fps: int) -> npt.NDArray[np.int16]:
    length = round(total_frames / fps * encode.AUDIO_RATE)
    track = np.zeros(length, dtype=np.int16)
    for bp in plans:
        if bp.samples is None:
            continue
        at = round((bp.start / fps + LEAD_S) * encode.AUDIO_RATE)
        part = bp.samples[: max(0, length - at)]
        track[at : at + len(part)] = part
    return track


def _chapters(plans: list[BeatPlan], fps: int) -> list[dict[str, Any]]:
    return [
        {
            "id": bp.beat.id,
            "story": bp.beat.story,
            "title": bp.beat.title,
            "start_s": round(bp.start / fps, 3),
            "end_s": round((bp.start + bp.frames) / fps, 3),
            "status": bp.status,
            "placeholder": bp.beat.placeholder,
            "needs": bp.beat.needs if bp.status != "record" else None,
        }
        for bp in plans
    ]


def _chapters_vtt(chapters: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for i, ch in enumerate(chapters, 1):
        lines += [
            str(i),
            f"{narration.stamp(ch['start_s'])} --> {narration.stamp(ch['end_s'])}",
            ch["title"],
            "",
        ]
    return "\n".join(lines)


def _contact_sheet(plans: list[BeatPlan], grabbed: dict[str, Image.Image], fps: int) -> Image.Image:
    thumb_w, thumb_h, label_h, gap, per_row = 384, 216, 46, 12, 4
    rows = (len(plans) + per_row - 1) // per_row
    sheet = Image.new(
        "RGB",
        (per_row * (thumb_w + gap) + gap, rows * (thumb_h + label_h + gap) + gap),
        compose.BG,
    )
    draw = ImageDraw.Draw(sheet)
    title_font, small = compose.font(14, "bold"), compose.font(12)
    colors = {
        "record": compose.GOOD,
        "card": compose.DIM,
        "placeholder": compose.PLACEHOLDER,
        "no footage": compose.MISSING,
    }
    for i, bp in enumerate(plans):
        x = gap + (i % per_row) * (thumb_w + gap)
        y = gap + (i // per_row) * (thumb_h + label_h + gap)
        picture = grabbed.get(bp.beat.id)
        if picture is not None:
            sheet.paste(picture.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
        start = bp.start / fps
        label = compose.ellipsize(
            f"{bp.beat.id}  {start:5.1f} s  {bp.beat.title}", title_font, thumb_w
        )
        draw.text((x, y + thumb_h + 4), label, fill=compose.TEXT, font=title_font)
        color = colors.get(bp.status, compose.MISSING)
        draw.text((x, y + thumb_h + 24), bp.status.upper(), fill=color, font=small)
    return sheet


def _timeline_md(plans: list[BeatPlan], fps: int, title: str) -> str:
    lines = [
        f"# Timeline: {title}",
        "",
        "| Beat | Story | Title | Start (s) | Length (s) | Footage | Clips | Needs |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for bp in plans:
        clips = "; ".join(
            f"{short_ref(c.run)} {c.episode} steps {c.start}-{c.end}"
            for s in bp.shots
            for c in s.clips
        )
        status = bp.status + (f" ({bp.beat.placeholder})" if bp.beat.placeholder else "")
        needs = bp.beat.needs if bp.status != "record" and bp.beat.needs else ""
        lines.append(
            f"| {bp.beat.id} | {bp.beat.story or ''} | {bp.beat.title} | {bp.start / fps:.1f} | "
            f"{bp.frames / fps:.1f} | {status} | {clips or 'none'} | {needs} |"
        )
    return "\n".join(lines) + "\n"


def _code_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--",
                "src/fleetlab/film",
                "site",
                "tools",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"checkout": str(root), "commit": commit, "film_code_modified": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"checkout": str(root), "commit": None, "film_code_modified": None}


def _shot_record(sp: ShotPlan, fps: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": sp.shot.kind,
        "start_s": round(sp.start / fps, 3),
        "end_s": round((sp.start + sp.frames) / fps, 3),
    }
    if sp.clips:
        record["clips"] = [c.provenance() for c in sp.clips]
    if sp.counts:
        record["counts"] = sp.counts
    if sp.files:
        record["files"] = sp.files
    if sp.shot.lines is not None:
        record["lines"] = list(sp.shot.lines)
    if sp.shot.kind in ("messages", "calls"):
        record["messages"] = [{"from": a, "text": b} for a, b in sp.messages]
    if sp.shot.kind == "calls":
        record["calls"] = sp.calls
    if sp.shot.field is not None:
        record["field"] = sp.shot.field
    if sp.shot.kind in ("card",):
        record["title"] = sp.shot.title or sp.beat.title
    return record


class FinalCutError(ValueError):
    pass


def final_cut_problems(plans: list[BeatPlan]) -> list[str]:
    """What a final cut may not hold: placeholders, beats without footage, bare videos."""
    problems = []
    for bp in plans:
        if bp.beat.placeholder:
            problems.append(f"{bp.beat.id}: a placeholder ({bp.beat.placeholder})")
        if bp.spoken is not None and bp.spoken.unbound:
            names = ", ".join("{" + n + "}" for n in bp.spoken.unbound)
            problems.append(f"{bp.beat.id}: the narration's {names} is bound to no record")
        for sp in bp.shots:
            from fleetlab.film.fields import placeholders

            texts = (sp.shot.title, sp.shot.subtitle, sp.shot.label, sp.shot.note)
            left = [n for t in texts if t for n in placeholders(t)]
            if left:
                problems.append(f"{bp.beat.id}: a card shows unbound {', '.join(left)}")
            if sp.shot.kind == "slate":
                problems.append(f"{bp.beat.id}: no footage yet")
            for clip in sp.clips:
                if clip.placeholder:
                    problems.append(f"{bp.beat.id}: a bare video without its recorded run")
    return problems


def _filled(shot: Shot, values: dict[str, str]) -> Shot:
    """The shot with `{name}` in its title, subtitle, label and note replaced by the bound
    value of that name (the narration's `bindings.json`); an unbound name stays visible."""
    import dataclasses

    from fleetlab.film.fields import PLACEHOLDER

    def fill(text: str | None) -> str | None:
        if text is None:
            return None
        return PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)

    return dataclasses.replace(
        shot,
        title=fill(shot.title),
        subtitle=fill(shot.subtitle),
        label=fill(shot.label),
        note=fill(shot.note),
    )


def _bindings(script: Path) -> dict[str, Any] | None:
    """`bindings.json` beside a bound script (`python -m fleetlab.film bind`), if any."""
    path = script.parent / "bindings.json"
    if not path.is_file():
        return None
    try:
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return record


def _binding_provenance(script: Path | None) -> dict[str, Any] | None:
    """Where the narration's numbers came from: each bound value with its record field and
    file hash, and the names left unbound."""
    if script is None or not script.is_file():
        return None
    record = _bindings(script)
    if record is None:
        return None
    path = script.parent / "bindings.json"
    return {
        "path": str(path),
        "sha256": sha256(path),
        "matches_script": record.get("bound_script_sha256") == sha256(script),
        "values": record.get("values") or {},
        "unbound": sorted(record.get("unbound") or {}),
    }


def script_binding_problems(script: Path) -> list[str]:
    """A final cut narrates a script bound to its records: `bindings.json` beside it, made
    for exactly this file, with no number left unbound."""
    from fleetlab.film.fields import BINDINGS_FORMAT

    if not script.is_file():
        return [f"the narration script {script} does not exist"]
    record = _bindings(script)
    if record is None or record.get("format") != BINDINGS_FORMAT:
        return [f"{script} is not a bound script (no {BINDINGS_FORMAT} bindings.json beside it)"]
    problems = []
    if record.get("bound_script_sha256") != sha256(script):
        problems.append(f"{script} was changed after it was bound")
    if record.get("unbound"):
        names = ", ".join("{" + n + "}" for n in record["unbound"])
        problems.append(f"the narration leaves {names} unbound")
    return problems


def build(
    manifest: Manifest,
    out: Path,
    *,
    narration_dir: Path | None = None,
    draft: bool = False,
    repeatable: bool = False,
    beats: set[str] | None = None,
    story: str | None = None,
    final: bool = False,
) -> BuildResult:
    fps = manifest.fps
    plans, warnings = plan(manifest, narration_dir=narration_dir, beats=beats, story=story)
    if final:
        problems = final_cut_problems(plans)
        if manifest.script is not None and any(bp.spoken is not None for bp in plans):
            problems += script_binding_problems(manifest.script)
        if problems:
            raise FinalCutError("not a final cut:\n  " + "\n  ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    total = sum(bp.frames for bp in plans)
    size = DRAFT if draft else MASTER
    canvas = Canvas(*size)
    if draft:
        outputs = [
            encode.Output(
                out / "film.mp4",
                *DRAFT,
                crf=28,
                preset="veryfast",
                maxrate_kbps=800,
                audio_kbps=96,
                role="draft",
            )
        ]
    else:
        outputs = [
            encode.Output(
                out / "film-master.mp4", *MASTER, crf=16, preset="slow",
                maxrate_kbps=None, audio_kbps=192, role="master",
            ),
            encode.Output(
                out / "film.mp4", *WEB, crf=20, preset="slow",
                maxrate_kbps=encode.WEB_MAXRATE_KBPS, audio_kbps=128, role="web",
            ),
        ]  # fmt: skip
    samples: dict[int, str] = {}
    poster_beat = manifest.poster if manifest.poster in {bp.beat.id for bp in plans} else None
    for bp in plans:
        shot = next((s for s in bp.shots if s.shot.kind != "card"), bp.shots[0])
        samples[shot.start + shot.frames // 2] = bp.beat.id
        if poster_beat is None and any(s.clips for s in bp.shots):
            poster_beat = bp.beat.id
    with tempfile.TemporaryDirectory(prefix="film-") as scratch:
        wav = Path(scratch) / "track.wav"
        encode.write_wav(wav, _audio_track(plans, total, fps))
        encoder = encode.Encoder(outputs, *size, fps, wav, repeatable=repeatable)
        try:
            grabbed = _render(plans, manifest, canvas, encoder, samples)
        except BaseException:
            encoder.abort()
            raise
        encoder.close()
    seconds = total / fps
    # files beside the film
    all_cues = [c for bp in plans for c in bp.cues]
    (out / "captions.vtt").write_text(narration.vtt(all_cues), encoding="utf-8")
    chapters = _chapters(plans, fps)
    (out / "chapters.vtt").write_text(_chapters_vtt(chapters), encoding="utf-8")
    narration.write_json(out / "chapters.json", chapters)
    poster = grabbed.get(poster_beat or plans[0].beat.id) or next(iter(grabbed.values()), None)
    if poster is not None:
        poster.resize(WEB if not draft else DRAFT, Image.Resampling.LANCZOS).save(
            out / "poster.jpg", quality=88
        )
    _contact_sheet(plans, grabbed, fps).save(out / "timeline.png", optimize=True)
    (out / "timeline.md").write_text(_timeline_md(plans, fps, manifest.title), encoding="utf-8")
    shutil.copyfile(manifest.path, out / "manifest.json")
    output_records = []
    for o in outputs:
        size_bytes = o.path.stat().st_size
        output_records.append(
            {
                "role": o.role,
                "file": o.path.name,
                "sha256": sha256(o.path),
                "bytes": size_bytes,
                "width": o.width,
                "height": o.height,
                "fps": fps,
                "seconds": round(seconds, 3),
                "video": f"H.264 high, crf {o.crf}, preset {o.preset}"
                + (f", at most {o.maxrate_kbps} kb/s" if o.maxrate_kbps else ""),
                "audio": f"AAC {o.audio_kbps} kb/s, {encode.AUDIO_RATE} Hz mono",
                "streams": encode.probe(o.path),
            }
        )
        if o.role == "web":
            if size_bytes > encode.WEB_LIMIT_BYTES:
                warnings.append(
                    f"{o.path.name} is {size_bytes / 1e6:.1f} MB, over GitHub Pages' 100 MB"
                )
            elif size_bytes > encode.WEB_TARGET_BYTES_3MIN * max(1.0, seconds / 180):
                warnings.append(
                    f"{o.path.name} is {size_bytes / 1e6:.1f} MB, over the 80 MB for three minutes"
                )
    voice = _voice(narration_dir) if narration_dir is not None else None
    provenance: dict[str, Any] = {
        "format": "fleetlab.film-provenance/1",
        "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "command": "python -m fleetlab.film build",
        "code": _code_state(),
        "ffmpeg": encode.ffmpeg_version(),
        "manifest": {"path": str(manifest.path), "sha256": sha256(manifest.path)},
        "script": (
            {"path": str(manifest.script), "sha256": sha256(manifest.script)}
            if manifest.script is not None and manifest.script.is_file()
            else None
        ),
        "bindings": _binding_provenance(manifest.script),
        "narration": (
            {"dir": str(narration_dir), "voice": voice}
            if narration_dir is not None
            else "none: captions timed at a reading pace of "
            f"{narration.WORDS_PER_MINUTE} words a minute, silent sound track"
        ),
        "final": final,
        "options": {
            "draft": draft,
            "repeatable": repeatable,
            "beats": sorted(beats) if beats else None,
            "story": story,
            "sources_on_screen": manifest.sources_on_screen,
        },
        "outputs": output_records,
        "seconds": round(seconds, 3),
        "beats": [
            {
                **ch,
                "subtitle": bp.beat.subtitle,
                "narration": (
                    None
                    if bp.spoken is None
                    else {
                        "words": bp.spoken.words,
                        "seconds": round(bp.narration_s, 3),
                        "voice": str(bp.audio) if bp.audio else None,
                        "voice_sha256": sha256(bp.audio) if bp.audio else None,
                        "timing": "measured audio" if bp.audio else "estimated reading pace",
                        "text": bp.spoken.text,
                    }
                ),
                "shots": [_shot_record(sp, fps) for sp in bp.shots],
            }
            for bp, ch in zip(plans, chapters, strict=True)
        ],
        "warnings": warnings,
    }
    narration.write_json(out / "provenance.json", provenance)
    return BuildResult(out, [o.path for o in outputs], seconds, plans, warnings, provenance)
