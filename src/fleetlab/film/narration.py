"""The narration script: beats of plain text, their spoken length, captions and voice requests.

`site/narration/script.md` holds one section per beat, headed `## <beat id>: <title>`.
Everything under the heading up to the next one is read aloud, except HTML comments
(`<!-- ... -->`, notes for the editor) and lines starting with `>` (stage directions). Text
before the first beat heading is not read.

Without a recorded voice the spoken length is estimated at a normal reading pace
(`WORDS_PER_MINUTE`); with one it is the audio's measured length. Captions split each beat
into cues of at most two short lines, timed in proportion to their characters over that
length, so a caption track exists whether or not a voice was recorded.

The voice requests follow ElevenLabs' text-to-speech API (checked on 19 October 2026:
`POST /v1/text-to-speech/{voice_id}`, body `text`, `model_id`, `voice_settings`, `seed`,
`previous_text`, `next_text`). No key is ever part of a request body; `tools/film-narrate`
sends it as a header read from the environment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "WORDS_PER_MINUTE",
    "Beat",
    "Cue",
    "cues",
    "parse_script",
    "pick_model",
    "pick_voices",
    "request_bodies",
    "spoken_seconds",
    "stamp",
    "vtt",
]

WORDS_PER_MINUTE = 150  # a normal narration pace; about 2.5 words a second
SENTENCE_PAUSE_S = 0.3  # the pause a reader leaves after a sentence
LINE_CHARS = 42
MODEL_DEFAULT = "eleven_multilingual_v2"
VOICE_SETTINGS: dict[str, Any] = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 1.0,
}
SEED = 20261020  # best-effort repeatable voice output

_HEADING = re.compile(r"^##\s+([A-Za-z0-9][A-Za-z0-9_.-]*)\s*:\s*(.+?)\s*$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Beat:
    id: str
    title: str
    text: str  # what is read aloud, whitespace collapsed
    notes: tuple[str, ...]  # the HTML comments under the beat, for the editor

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def unbound(self) -> list[str]:
        """Placeholders (`{name}`) left in the spoken text: numbers not yet bound to a record
        (`fleetlab.film.fields`). Such a beat may be shown as captions in a rehearsal; it is
        never sent to a voice and never goes into a final cut."""
        from fleetlab.film.fields import placeholders

        return placeholders(self.text)

    @property
    def sentences(self) -> list[str]:
        return [s.strip() for s in _SENTENCE.split(self.text) if s.strip()]


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str  # one or two lines joined by a newline


def parse_script(text: str) -> list[Beat]:
    """The beats of a narration script in their order; duplicate ids are an error."""
    beats: list[Beat] = []
    current: tuple[str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is None:
            return
        raw = "\n".join(body)
        notes = tuple(" ".join(m.group(0)[4:-3].split()) for m in _COMMENT.finditer(raw))
        kept = [
            line for line in _COMMENT.sub("", raw).splitlines() if not line.lstrip().startswith(">")
        ]
        spoken = " ".join(" ".join(kept).split())
        beats.append(Beat(current[0], current[1], spoken, notes))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            current = (match.group(1), match.group(2))
            body = []
        elif line.startswith("# ") or line.startswith("## "):
            flush()
            current = None
            body = []
        elif current is not None:
            body.append(line)
    flush()
    seen: set[str] = set()
    for beat in beats:
        if beat.id in seen:
            raise ValueError(f"beat {beat.id!r} appears twice in the script")
        seen.add(beat.id)
    return beats


def spoken_seconds(beat: Beat) -> float:
    """Estimated length read aloud at `WORDS_PER_MINUTE`, with a pause after each sentence."""
    if not beat.text:
        return 0.0
    return beat.words * 60.0 / WORDS_PER_MINUTE + SENTENCE_PAUSE_S * len(beat.sentences)


def _wrap(text: str, width: int = LINE_CHARS) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    return lines


def _fits(text: str) -> bool:
    return len(_wrap(text)) <= 2


def _pieces(sentence: str) -> list[str]:
    """A sentence split into pieces of at most two caption lines, at clause breaks where
    it can and between words where it must."""
    if _fits(sentence):
        return [sentence]
    parts = re.split(r"(?<=[,;:])\s+|\s+(?=-\s)", sentence)
    pieces: list[str] = []
    piece = ""
    for part in parts:
        trial = f"{piece} {part}" if piece else part
        if piece and not _fits(trial):
            pieces.append(piece)
            piece = part
        else:
            piece = trial
    if piece:
        pieces.append(piece)
    out: list[str] = []
    for p in pieces:
        if _fits(p):
            out.append(p)
            continue
        lines = _wrap(p)
        out += [" ".join(lines[i : i + 2]) for i in range(0, len(lines), 2)]
    return out


def cues(beat: Beat, seconds: float, offset: float = 0.0) -> list[Cue]:
    """Caption cues for a beat read over `seconds`, starting at `offset`."""
    pieces = [p for s in beat.sentences for p in _pieces(s)]
    if not pieces or seconds <= 0:
        return []
    weights = [len(p) + 12 for p in pieces]  # a constant per cue stands for the pauses
    total = float(sum(weights))
    out: list[Cue] = []
    t = offset
    for piece, weight in zip(pieces, weights, strict=True):
        length = seconds * weight / total
        out.append(Cue(round(t, 3), round(t + length, 3), "\n".join(_wrap(piece))))
        t += length
    return out


def stamp(seconds: float) -> str:
    """A WebVTT time stamp."""
    ms = round(max(seconds, 0.0) * 1000)
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def vtt(all_cues: Sequence[Cue]) -> str:
    """A WebVTT caption file."""
    lines = ["WEBVTT", ""]
    for i, cue in enumerate(all_cues, 1):
        lines += [str(i), f"{stamp(cue.start)} --> {stamp(cue.end)}", cue.text, ""]
    return "\n".join(lines)


def request_bodies(
    beat: Beat,
    *,
    model: str = MODEL_DEFAULT,
    previous: Beat | None = None,
    following: Beat | None = None,
) -> list[dict[str, Any]]:
    """The JSON bodies for one beat, fullest first; a request refused as invalid (422) is
    sent again with the next, plainer one (not every model takes every optional field)."""
    full: dict[str, Any] = {
        "text": beat.text,
        "model_id": model,
        "voice_settings": dict(VOICE_SETTINGS),
        "seed": SEED,
    }
    if previous is not None and previous.text:
        full["previous_text"] = previous.text
    if following is not None and following.text:
        full["next_text"] = following.text
    plain = {"text": beat.text, "model_id": model, "voice_settings": dict(VOICE_SETTINGS)}
    bare = {"text": beat.text, "model_id": model}
    out: list[dict[str, Any]] = []
    for body in (full, plain, bare):
        if body not in out:
            out.append(body)
    return out


# ---- voices and models from the account's listing --------------------------------------


def _labels(voice: Mapping[str, Any]) -> dict[str, str]:
    raw = voice.get("labels") or {}
    return {str(k).lower(): str(v).lower() for k, v in dict(raw).items() if v is not None}


def _narration_score(voice: Mapping[str, Any]) -> int:
    labels = _labels(voice)
    text = " ".join(
        [
            labels.get("use_case", ""),
            labels.get("use case", ""),
            labels.get("description", ""),
            str(voice.get("description") or "").lower(),
        ]
    )
    score = 0
    for word, points in (("narrat", 4), ("documentary", 3), ("audiobook", 2), ("deep", 1)):
        if word in text:
            score += points
    for word in ("news", "informative", "educational", "conversational", "calm", "warm"):
        if word in text:
            score += 1
    for word in ("character", "animation", "video game", "child", "meme", "asmr"):
        if word in text:
            score -= 3
    return score


def pick_voices(listing: Mapping[str, Any], count: int = 2) -> list[dict[str, Any]]:
    """Up to `count` premade voices labelled male and American, best suited to narration first.

    The listing is the JSON of `GET /v1/voices`. A voice qualifies only on its labels
    (`gender` male, `accent` American); suitability for narration ranks them by their use
    case and description. Ties keep the listing's order.
    """
    voices = list(listing.get("voices") or [])
    chosen: list[tuple[int, int, dict[str, Any]]] = []
    for position, voice in enumerate(voices):
        labels = _labels(voice)
        if str(voice.get("category", "premade")).lower() != "premade":
            continue
        if labels.get("gender") != "male" or "american" not in labels.get("accent", ""):
            continue
        chosen.append((-_narration_score(voice), position, dict(voice)))
    chosen.sort(key=lambda c: (c[0], c[1]))
    return [
        {
            "voice_id": v.get("voice_id"),
            "name": v.get("name"),
            "labels": _labels(v),
            "description": v.get("description"),
            "narration_score": -score,
        }
        for score, _, v in chosen[:count]
    ]


# The models in order of narration quality for English, best first (ElevenLabs' model
# overview, read on 19 October 2026): v4 is the highest-fidelity model and the one it
# recommends for audiobook production; Multilingual v2 is the most stable on long-form
# output; v3 is the previous generation's expressive model. Flash trades quality for speed.
MODEL_PREFERENCE = (
    "eleven_v4",
    "eleven_multilingual_v2",
    "eleven_v3",
    "eleven_flash_v2_5",
)


def pick_model(listing: Sequence[Mapping[str, Any]] | None) -> str:
    """The best text-to-speech model the account lists (`GET /v1/models`).

    Without a listing, or when none of the preferred models is offered, Multilingual v2.
    """
    if not listing:
        return MODEL_DEFAULT
    offered = {
        str(m.get("model_id"))
        for m in listing
        if m.get("can_do_text_to_speech", True) and "english" in _languages(m)
    }
    for model in MODEL_PREFERENCE:
        if model in offered:
            return model
    return MODEL_DEFAULT


def _languages(model: Mapping[str, Any]) -> str:
    languages = model.get("languages")
    if not languages:
        return "english"  # a model that lists no languages is not excluded for that
    return " ".join(str(dict(lang).get("name", "")).lower() for lang in languages)


def load_script(path: Path) -> list[Beat]:
    return parse_script(path.read_text(encoding="utf-8"))


def summary(beats: Sequence[Beat]) -> dict[str, Any]:
    rows = [
        {"id": b.id, "title": b.title, "words": b.words, "seconds": round(spoken_seconds(b), 1)}
        for b in beats
    ]
    return {
        "words_per_minute": WORDS_PER_MINUTE,
        "beats": rows,
        "words": sum(r["words"] for r in rows),
        "seconds": round(sum(spoken_seconds(b) for b in beats), 1),
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---- what a recording was made from ------------------------------------------------------

VOICE_FORMAT = "fleetlab.film-voice/1"


def text_digest(beat: Beat) -> str:
    """The SHA-256 of the words a beat's recording was made from."""
    import hashlib

    return hashlib.sha256(beat.text.encode("utf-8")).hexdigest()


def voice_record(
    out: Path, script: Path, voice: str, model: str, output_format: str, beat_ids: Sequence[str]
) -> dict[str, Any]:
    """`voice.json` for the beats this run recorded: each with the digest of its text and of
    its audio, so a build can refuse a recording made from other words or changed since."""
    import datetime as dt
    import hashlib

    beats = {b.id: b for b in load_script(script)}
    recorded: dict[str, Any] = {}
    for beat_id in beat_ids:
        if beat_id not in beats:
            raise ValueError(f"beat {beat_id} is not in {script}")
        audio = out / f"{beat_id}.mp3"
        if not audio.is_file():
            raise ValueError(f"{audio} was not recorded")
        recorded[beat_id] = {
            "text_sha256": text_digest(beats[beat_id]),
            "audio": audio.name,
            "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        }
    record = {
        "format": VOICE_FORMAT,
        "voice_id": voice,
        "model_id": model,
        "output_format": output_format,
        "recorded": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "beats": recorded,
    }
    write_json(out / "voice.json", record)
    return record
