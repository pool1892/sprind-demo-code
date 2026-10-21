"""The film manifest (`fleetlab.film-manifest/1`): beats, their shots, captions and side panel.

A manifest is JSON. Paths are relative to the manifest's directory unless absolute; `~`
is expanded. The outline:

    {"format": "fleetlab.film-manifest/1", "title": ..., "subtitle": ...,
     "script": "../narration/script.md",            # the narration, by beat id
     "stories": {"A": "Make this one job", ...},
     "beats": [
       {"id": "a1-skills", "story": "A", "title": "The skills", "subtitle": ...,
        "placeholder": "why the footage is not the record's yet" | null,
        "needs": "what footage the beat must show",
        "panel": {...},                              # the side panel, see `Panel`
        "shots": [
          {"kind": "clip", "clip": {...}},           # one scene view, see `ClipRef`
          {"kind": "pair", "clips": [{...}, {...}]}, # before and after, side by side
          {"kind": "card", "title": ..., "subtitle": ..., "seconds": 3},
          {"kind": "slate", "seconds": 5},           # no footage yet: says what it needs
          {"kind": "image", "image": "figure.png", "seconds": 5, "label": ...},
          {"kind": "text", "file": "check.sql" | null, "text": ... | null, "seconds": 6,
           "field": "sql", "lines": [first, last]},  # a listing, long lines wrapped
          {"kind": "messages", "file": "run.json", "field": "state.message_log",
           "lines": [1, 3]},                         # messages read from the record
          {"kind": "messages", "messages": [{"from": ..., "text": ...}], "seconds": 8},
          {"kind": "calls", "file": "agent-runs/<run>/stdout.jsonl", "tools": ["query"],
           "lines": [1, 3], "seconds": 8}]}]}             # the agent's tool calls, read
                                                         # from its session transcript

A clip names its recorded run and episode and, optionally, the control steps it starts
and ends on. The panel shows the requirement (requested goals, the order rule, the
approved bar), the program's step list with the step in force, and a count; a count is
either computed from a run's table (`from`) or carries the source it was read from.

Under each shot the film draws a small source line naming the footage's run, episode,
control steps and video digest, and under each computed count its table and digest. A cut
for viewers can set `"sources_on_screen": "viewer"` (the default is `"full"`): the frames
then leave those identifiers and digests out and keep the notes, placeholder and
missing-footage texts; the provenance, the timeline and every verification are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FORMAT",
    "Beat",
    "ClipRef",
    "CountSpec",
    "Manifest",
    "ManifestError",
    "Panel",
    "Rerender",
    "Shot",
    "load",
]

FORMAT = "fleetlab.film-manifest/1"
SOURCES_ON_SCREEN = ("full", "viewer")
SHOT_KINDS = ("clip", "pair", "card", "slate", "image", "text", "messages", "calls")
DEFAULT_SECONDS = {
    "card": 3.0, "slate": 5.0, "image": 5.0, "text": 6.0, "messages": 8.0, "calls": 8.0,
}  # fmt: skip


class ManifestError(ValueError):
    pass


def _path(base: Path, value: object, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: a path is required")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def _opt_str(data: dict[str, Any], key: str, where: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"{where}.{key}: text expected")
    return value


def _opt_int(data: dict[str, Any], key: str, where: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{where}.{key}: a non-negative whole number expected")
    return value


def _number(data: dict[str, Any], key: str, where: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{where}.{key}: a positive number expected")
    return float(value)


def _known(data: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = sorted(set(data) - keys)
    if unknown:
        raise ManifestError(f"{where}: unknown field(s) {', '.join(unknown)}")


@dataclass(frozen=True)
class Rerender:
    """A verified re-render from the jobs runner's replay, used instead of the original."""

    video: Path
    verification: Path


@dataclass(frozen=True)
class ClipRef:
    label: str
    run: Path | None = None  # a run directory, or its `log/` directory
    episode: str | None = None  # the episode id within the run
    video: Path | None = None  # a bare video: placeholder beats only, never the record
    start: int | None = None  # the first control step shown
    end: int | None = None  # the last control step shown
    speed: float = 1.0  # control steps advanced per output frame at the film's rate
    hold: float = 0.0  # seconds the last frame stays on screen
    crop: tuple[int, int, int, int] | None = None  # x, y, width, height in source pixels
    camera: str = "scene"
    steps_per_frame: int | None = None  # for a video named directly
    rerender: Rerender | None = None
    note: str | None = None  # shown under the scene, e.g. "placeholder: lane S spike run"

    @staticmethod
    def parse(base: Path, data: object, where: str) -> ClipRef:
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: an object expected")
        _known(
            data,
            {
                "label", "run", "episode", "video", "start", "end", "speed", "hold", "crop",
                "camera", "steps_per_frame", "rerender", "note",
            },
            where,
        )  # fmt: skip
        label = _opt_str(data, "label", where) or ""
        run = _path(base, data["run"], f"{where}.run") if data.get("run") else None
        video = _path(base, data["video"], f"{where}.video") if data.get("video") else None
        episode = _opt_str(data, "episode", where)
        if run is None and video is None:
            raise ManifestError(f"{where}: a clip names its recorded run and episode")
        if run is not None and not episode:
            raise ManifestError(f"{where}: a clip from a run needs its episode id")
        if run is None and _opt_int(data, "steps_per_frame", where) is None:
            raise ManifestError(
                f"{where}: a video without its run states its frame cadence (steps_per_frame)"
            )
        crop = data.get("crop")
        if crop is not None:
            if (
                not isinstance(crop, list)
                or len(crop) != 4
                or not all(isinstance(v, int) and v >= 0 for v in crop)
            ):
                raise ManifestError(f"{where}.crop: [x, y, width, height] expected")
            crop = (crop[0], crop[1], crop[2], crop[3])
        rerender = None
        if data.get("rerender") is not None:
            raw = data["rerender"]
            if not isinstance(raw, dict):
                raise ManifestError(f"{where}.rerender: an object expected")
            rerender = Rerender(
                _path(base, raw.get("video"), f"{where}.rerender.video"),
                _path(base, raw.get("verification"), f"{where}.rerender.verification"),
            )
        start, end = _opt_int(data, "start", where), _opt_int(data, "end", where)
        if start is not None and end is not None and end < start:
            raise ManifestError(f"{where}: end comes before start")
        camera = _opt_str(data, "camera", where) or "scene"
        if camera not in ("scene", "wrist"):
            raise ManifestError(f"{where}.camera: scene or wrist")
        hold = data.get("hold", 0.0)
        if not isinstance(hold, int | float) or isinstance(hold, bool) or hold < 0:
            raise ManifestError(f"{where}.hold: a non-negative number expected")
        return ClipRef(
            label=label,
            run=run,
            episode=episode,
            video=video,
            start=start,
            end=end,
            speed=_number(data, "speed", where, 1.0),
            hold=float(hold),
            crop=crop,
            camera=camera,
            steps_per_frame=_opt_int(data, "steps_per_frame", where),
            rerender=rerender,
            note=_opt_str(data, "note", where),
        )


@dataclass(frozen=True)
class CountSpec:
    """A number on screen. `text` may hold `{k}` and `{n}` when the count is computed."""

    text: str
    detail: str | None = None
    source: str | None = None  # where a literal count was read from
    tables: tuple[Path, ...] = ()  # JSON-lines or Parquet tables to compute from, rows joined
    where: dict[str, Any] = field(default_factory=dict)  # column equals value
    success: str | dict[str, Any] | None = None  # a true column, or column equals value
    key: tuple[str, ...] = ()  # columns that identify a row; a repeated row is an error

    @staticmethod
    def parse(base: Path, data: object, where: str) -> CountSpec:
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: an object expected")
        _known(data, {"text", "detail", "source", "from"}, where)
        text = _opt_str(data, "text", where)
        if not text:
            raise ManifestError(f"{where}.text is required")
        source = _opt_str(data, "source", where)
        computed = data.get("from")
        if computed is None:
            if not source:
                raise ManifestError(f"{where}: a literal count needs its source")
            return CountSpec(text, _opt_str(data, "detail", where), source)
        if not isinstance(computed, dict):
            raise ManifestError(f"{where}.from: an object expected")
        _known(computed, {"table", "where", "success", "key"}, f"{where}.from")
        filters = computed.get("where") or {}
        success = computed.get("success")
        if not isinstance(filters, dict) or not isinstance(success, str | dict):
            raise ManifestError(f"{where}.from: where is an object; success a column or object")
        raw = computed.get("table")
        tables = raw if isinstance(raw, list) else [raw]
        key = computed.get("key") or []
        if not isinstance(key, list) or not all(isinstance(k, str) for k in key):
            raise ManifestError(f"{where}.from.key: a list of column names")
        return CountSpec(
            text,
            _opt_str(data, "detail", where),
            source,
            tuple(_path(base, t, f"{where}.from.table") for t in tables),
            dict(filters),
            success,
            tuple(key),
        )


@dataclass(frozen=True)
class Panel:
    requirement: tuple[str, ...] = ()  # the requested goals, in the shift lead's words
    rule: str | None = None  # the order rule
    bar: str | None = None  # the approved acceptance condition
    program: str | None = None  # the program's name
    steps: tuple[str, ...] = ()  # the program's steps; empty with `current: auto` reads them
    current: int | str | None = None  # an index, "auto" (from the clip's record) or none
    counts: tuple[CountSpec, ...] = ()  # `count` in the manifest: one object or a list
    follow: int = 0  # which clip of a pair drives `auto`

    @staticmethod
    def parse(base: Path, data: object, where: str, inherit: Panel | None = None) -> Panel:
        if data is None:
            return inherit or Panel()
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: an object expected")
        _known(
            data,
            {"requirement", "rule", "bar", "program", "steps", "current", "count", "follow"},
            where,
        )
        start = inherit or Panel()

        def lines(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            if key not in data:
                return default
            value = data[key]
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ManifestError(f"{where}.{key}: a list of text expected")
            return tuple(value)

        current = data.get("current", start.current)
        if current is not None and current != "auto" and not isinstance(current, int):
            raise ManifestError(f"{where}.current: a step index, auto, or null")
        counts = start.counts
        if "count" in data:
            raw = data["count"]
            items = [] if raw is None else raw if isinstance(raw, list) else [raw]
            counts = tuple(
                CountSpec.parse(base, c, f"{where}.count[{i}]") for i, c in enumerate(items)
            )
        follow = data.get("follow", start.follow)
        if not isinstance(follow, int) or follow not in (0, 1):
            raise ManifestError(f"{where}.follow: 0 or 1")
        return Panel(
            requirement=lines("requirement", start.requirement),
            rule=data.get("rule", start.rule),
            bar=data.get("bar", start.bar),
            program=data.get("program", start.program),
            steps=lines("steps", start.steps),
            current=current,
            counts=counts,
            follow=follow,
        )

    @property
    def empty(self) -> bool:
        return not (
            self.requirement or self.rule or self.bar or self.program or self.steps or self.counts
        )


@dataclass(frozen=True)
class Shot:
    kind: str
    clips: tuple[ClipRef, ...] = ()
    seconds: float | None = None
    title: str | None = None
    subtitle: str | None = None
    label: str | None = None
    image: Path | None = None
    file: Path | None = None
    text: str | None = None
    field: str | None = None  # a dotted path into a JSON `file`: what is shown
    lines: tuple[int, int] | None = None  # lines (text) or messages shown, 1-based, inclusive
    messages: tuple[tuple[str, str], ...] = ()
    panel: Panel | None = None
    note: str | None = None
    tools: tuple[str, ...] = ()  # a calls shot: only these tools' calls
    text_key: str | None = None  # a messages shot from a file: show only this key of each

    @staticmethod
    def parse(base: Path, data: object, where: str, panel: Panel) -> Shot:
        if not isinstance(data, dict):
            raise ManifestError(f"{where}: an object expected")
        kind = data.get("kind")
        if kind not in SHOT_KINDS:
            raise ManifestError(f"{where}.kind: one of {', '.join(SHOT_KINDS)}")
        _known(
            data,
            {"kind", "clip", "clips", "seconds", "title", "subtitle", "label", "image",
             "file", "field", "text", "lines", "messages", "panel", "note", "tools",
             "text_key"},
            where,
        )  # fmt: skip
        clips: tuple[ClipRef, ...] = ()
        if kind == "clip":
            clips = (ClipRef.parse(base, data.get("clip"), f"{where}.clip"),)
        elif kind == "pair":
            raw = data.get("clips")
            if not isinstance(raw, list) or len(raw) != 2:
                raise ManifestError(f"{where}.clips: two clips, before and after")
            clips = tuple(ClipRef.parse(base, c, f"{where}.clips[{i}]") for i, c in enumerate(raw))
        seconds = None
        if kind in DEFAULT_SECONDS:
            seconds = _number(data, "seconds", where, DEFAULT_SECONDS[kind])
        elif "seconds" in data:
            raise ManifestError(f"{where}.seconds: a clip's length comes from its steps")
        image = _path(base, data["image"], f"{where}.image") if kind == "image" else None
        file = _path(base, data["file"], f"{where}.file") if data.get("file") else None
        text = _opt_str(data, "text", where)
        if kind == "text" and file is None and text is None:
            raise ManifestError(f"{where}: a text shot needs a file or text")
        if kind == "calls" and file is None:
            raise ManifestError(
                f"{where}: a calls shot reads the agent's session transcript (file)"
            )
        tools = data.get("tools") or []
        if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
            raise ManifestError(f"{where}.tools: a list of tool names")
        if tools and kind != "calls":
            raise ManifestError(f"{where}.tools: only a calls shot filters tools")
        field_path = _opt_str(data, "field", where)
        if field_path is not None and (file is None or kind not in ("text", "messages")):
            raise ManifestError(
                f"{where}.field: a path into the JSON file of a text or messages shot"
            )
        lines = data.get("lines")
        if lines is not None:
            if (
                kind not in ("text", "messages", "calls")
                or not isinstance(lines, list)
                or len(lines) != 2
                or not all(isinstance(v, int) and v >= 1 for v in lines)
                or lines[1] < lines[0]
            ):
                raise ManifestError(
                    f"{where}.lines: [first, last], 1-based, of a text, messages or calls shot"
                )
            lines = (lines[0], lines[1])
        messages: tuple[tuple[str, str], ...] = ()
        if kind == "messages" and file is None:
            raw = data.get("messages")
            if not isinstance(raw, list) or not raw:
                raise ManifestError(f"{where}.messages: a list of {{from, text}}, or a file")
            messages = tuple((str(m.get("from", "")), str(m.get("text", ""))) for m in raw)
        shot_panel = (
            Panel.parse(base, data.get("panel"), f"{where}.panel", panel)
            if "panel" in data
            else None
        )
        return Shot(
            kind=kind,
            clips=clips,
            seconds=seconds,
            title=_opt_str(data, "title", where),
            subtitle=_opt_str(data, "subtitle", where),
            label=_opt_str(data, "label", where),
            image=image,
            file=file,
            text=text,
            field=field_path,
            lines=lines,
            messages=messages,
            panel=shot_panel,
            note=_opt_str(data, "note", where),
            tools=tuple(tools),
            text_key=_opt_str(data, "text_key", where),
        )


@dataclass(frozen=True)
class Beat:
    id: str
    story: str | None
    title: str
    subtitle: str | None
    placeholder: str | None
    needs: str | None
    panel: Panel
    shots: tuple[Shot, ...]
    narrated: bool

    @property
    def missing(self) -> bool:
        """No footage at all: only slates, or nothing."""
        return all(s.kind == "slate" for s in self.shots)


@dataclass(frozen=True)
class Manifest:
    path: Path
    title: str
    subtitle: str | None
    script: Path | None
    stories: dict[str, str]
    beats: tuple[Beat, ...]
    fps: int
    raw: dict[str, Any]
    poster: str | None = None  # the beat whose middle frame is the poster
    aliases: dict[str, str] = field(default_factory=dict)  # private name -> public, in refs
    forbid: tuple[str, ...] = ()  # patterns no drawn text and no site file may hold
    sources_on_screen: str = "full"  # "viewer": no run ids or digests drawn in the frames

    def story_name(self, story: str | None) -> str | None:
        return None if story is None else self.stories.get(story, story)


def parse(data: object, path: Path) -> Manifest:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ManifestError(f"{path}: not a {FORMAT} manifest")
    _known(
        data,
        {
            "format",
            "title",
            "subtitle",
            "script",
            "stories",
            "beats",
            "fps",
            "notes",
            "poster",
            "public",
            "sources_on_screen",
        },
        "manifest",
    )
    base = path.parent
    from fleetlab.film.public import public_rules

    try:
        aliases, forbid = public_rules(data.get("public"))
    except ValueError as error:
        raise ManifestError(f"manifest.{error}") from None
    beats_raw = data.get("beats")
    if not isinstance(beats_raw, list) or not beats_raw:
        raise ManifestError("manifest.beats: a list of beats is required")
    stories = data.get("stories") or {}
    if not isinstance(stories, dict):
        raise ManifestError("manifest.stories: an object of story names")
    beats: list[Beat] = []
    seen: set[str] = set()
    for i, raw in enumerate(beats_raw):
        where = f"beats[{i}]"
        if not isinstance(raw, dict):
            raise ManifestError(f"{where}: an object expected")
        _known(
            raw,
            {
                "id",
                "story",
                "title",
                "subtitle",
                "placeholder",
                "needs",
                "panel",
                "shots",
                "narrated",
            },
            where,
        )
        beat_id = raw.get("id")
        if not isinstance(beat_id, str) or not beat_id or beat_id in seen:
            raise ManifestError(f"{where}.id: a unique id is required")
        seen.add(beat_id)
        where = f"beat {beat_id}"
        panel = Panel.parse(base, raw.get("panel"), f"{where}.panel")
        shots_raw = raw.get("shots")
        if not isinstance(shots_raw, list) or not shots_raw:
            raise ManifestError(f"{where}.shots: at least one shot")
        shots = tuple(
            Shot.parse(base, s, f"{where}.shots[{j}]", panel) for j, s in enumerate(shots_raw)
        )
        if not raw.get("placeholder") and any(c.run is None for s in shots for c in s.clips):
            raise ManifestError(
                f"{where}: a clip without its recorded run and episode is allowed only in a "
                "beat marked as a placeholder"
            )
        beats.append(
            Beat(
                id=beat_id,
                story=_opt_str(raw, "story", where),
                title=_opt_str(raw, "title", where) or beat_id,
                subtitle=_opt_str(raw, "subtitle", where),
                placeholder=_opt_str(raw, "placeholder", where),
                needs=_opt_str(raw, "needs", where),
                panel=panel,
                shots=shots,
                narrated=bool(raw.get("narrated", True)),
            )
        )
    fps = data.get("fps", 20)
    if not isinstance(fps, int) or not 5 <= fps <= 60:
        raise ManifestError("manifest.fps: a whole number from 5 to 60")
    sources_on_screen = data.get("sources_on_screen", "full")
    if sources_on_screen not in SOURCES_ON_SCREEN:
        raise ManifestError(f"manifest.sources_on_screen: one of {', '.join(SOURCES_ON_SCREEN)}")
    return Manifest(
        path=path,
        title=str(data.get("title") or "Untitled"),
        subtitle=data.get("subtitle"),
        script=_path(base, data["script"], "manifest.script") if data.get("script") else None,
        stories={str(k): str(v) for k, v in stories.items()},
        beats=tuple(beats),
        fps=fps,
        raw=data,
        poster=_opt_str(data, "poster", "manifest"),
        aliases=aliases,
        forbid=tuple(forbid),
        sources_on_screen=sources_on_screen,
    )


def load(path: Path) -> Manifest:
    path = path.expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path}: not JSON ({error})") from error
    return parse(data, path)
