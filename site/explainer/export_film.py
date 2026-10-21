"""Turn the rendered explainer into a film folder the site builder publishes.

    uv run --locked python site/explainer/export_film.py --render film/out/explainer.mp4 \
        --out ~/.cache/robots-probes/demo/explainer/film/<cut> [--poster-s 15]

Writes, beside each other (the layout `python -m fleetlab.film site` reads):

- `film-master.mp4` (1920x1080, as rendered) and `film.mp4` (1280x720, H.264 and AAC, video
  capped at 3 Mb/s, fast start, for the web);
- `poster.jpg`, `captions.vtt` (word-timed from the narration's character timestamps),
  `chapters.vtt` and `chapters.json` (one chapter per beat, with its act);
- `provenance.json`: every beat with its start and end, its narration, and the recorded clips
  it shows (run, episode, the original recording and, for a re-render, its verification), the
  bound script, the voice, the data file and every output with its SHA-256;
- `manifest.json`: the title and the public rules of the cut.

The clips of each beat are listed in `site/explainer/beat-clips.json` (beat id -> clip ids of
film-data's `clips`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "film" / "public"
FPS = 30
T = 14  # transition frames, as in film/src/theme.ts durations.transition
CUT_ENTRIES = {"title"}  # beats.tsx ENTRY 'cut'

ACTS = [
    ("cold-open", "Why we ran these experiments"),
    ("station", "The test bench and the rules"),
    ("request", "Experiment one: a new job from existing skills"),
    ("b-fleet", "Experiment two: faults planted in a fleet"),
    ("results", "What we learned, and what comes next"),
]


def script_titles() -> dict[str, str]:
    """Each beat's chapter title: the heading of its section in the script (`## id: Title`)."""
    text = (HERE / "script.md").read_text()
    return {
        m.group(1): m.group(2).strip() for m in re.finditer(r"^## ([a-z0-9-]+): (.+)$", text, re.M)
    }


TITLES = script_titles()


PROBES = Path.home() / ".cache" / "robots-probes"


def local(ref: str) -> Path:
    """A local path from a public identifier `probes:<path under ~/.cache/robots-probes>`."""
    return PROBES / ref[len("probes:") :] if ref.startswith("probes:") else Path(ref)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def starts(beats: list[dict[str, Any]]) -> list[float]:
    out, at = [], 0
    for i, b in enumerate(beats):
        if i > 0 and b["id"] not in CUT_ENTRIES:
            at -= T
        out.append(at / FPS)
        at += round(b["seconds"] * FPS)
    return out


def words_of(alignment: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Words with start and end times from character timestamps, break tags removed."""
    chars = alignment.get("characters") or []
    st = alignment.get("character_start_times_seconds") or []
    en = alignment.get("character_end_times_seconds") or []
    words: list[tuple[str, float, float]] = []
    cur, cs, ce, in_tag = "", 0.0, 0.0, False
    for c, s, e in zip(chars, st, en, strict=False):
        if c == "<":
            in_tag = True
        if in_tag:
            if c == ">":
                in_tag = False
            continue
        if c.isspace():
            if cur:
                words.append((cur, cs, ce))
            cur = ""
            continue
        if not cur:
            cs = s
        cur += c
        ce = e
    if cur:
        words.append((cur, cs, ce))
    return words


def captions(beats: list[dict[str, Any]], at: list[float], voice: Path | None) -> str:
    """Caption cues of at most about two short lines: break at sentence ends, else at a comma
    or colon past 34 characters, never inside "1 of 20"; a cue ends where the next begins."""
    cues: list[tuple[float, float, str]] = []
    for b, t0 in zip(beats, at, strict=True):
        if not b.get("voice") or voice is None:
            continue
        info = json.loads((voice / f"{b['id']}.json").read_text())
        words = words_of(info["alignment"])
        base = t0 + b["voice_offset"]
        cue: list[tuple[str, float, float]] = []
        for i, w in enumerate(words):
            cue.append(w)
            text = " ".join(x[0] for x in cue)
            nxt = words[i + 1][0] if i + 1 < len(words) else ""
            sentence = re.search(r"[.?!]$", w[0]) is not None
            soft = re.search(r"[,:;]$", w[0]) is not None
            glued = re.fullmatch(r"\d+", w[0]) is not None and nxt == "of"
            if (
                (sentence and len(text) >= 12)
                or (soft and len(text) >= 34)
                or (len(text) >= 62 and not glued)
                or i == len(words) - 1
            ):
                cues.append((base + cue[0][1], base + cue[-1][2] + 0.25, text))
                cue = []
    lines = ["WEBVTT", ""]
    for n, (s, e, text) in enumerate(cues, 1):
        if n < len(cues):
            e = min(e, cues[n][0] - 0.02)
        lines += [str(n), f"{clock(s)} --> {clock(e)}", wrap(text), ""]
    return "\n".join(lines)


def wrap(text: str, width: int = 42) -> str:
    if len(text) <= width:
        return text
    cut = text.rfind(" ", 0, len(text) // 2 + 8)
    return text[:cut] + "\n" + text[cut + 1 :] if cut > 0 else text


def clock(s: float) -> str:
    h, rem = divmod(max(0.0, s), 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", type=Path, required=True, help="the master MP4 Remotion rendered")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--voice", type=Path, help="the narration folder (for captions); default from the timeline"
    )
    ap.add_argument(
        "--bound", type=Path, help="the bound script's folder (recorded in the provenance)"
    )
    ap.add_argument(
        "--poster-s",
        type=float,
        default=None,
        help="poster frame time; default the title beat's middle",
    )
    ap.add_argument(
        "--title", default="From skills to jobs: an AI agent engineers a robot's new work"
    )
    args = ap.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    timeline = json.loads((PUBLIC / "data" / "timeline.json").read_text())
    film = json.loads((PUBLIC / "data" / "film-data.json").read_text())
    beats = timeline["beats"]
    at = starts(beats)
    total = at[-1] + beats[-1]["seconds"]
    voice = args.voice.expanduser() if args.voice else None
    if voice is None and timeline.get("voice_dir"):
        voice = Path(timeline["voice_dir"])

    # the videos
    # the audio normalised to -16 LUFS integrated, -1.5 dB true peak (the web's usual level)
    loud = "loudnorm=I=-16:TP=-1.5:LRA=7"
    master = out / "film-master.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(args.render.expanduser()),
            "-c:v",
            "copy",
            "-af",
            loud,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(master),
        ]
    )
    web = out / "film.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(master),
            "-vf",
            "scale=1280:720:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-maxrate",
            "3M",
            "-bufsize",
            "6M",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(web),
        ]
    )
    title_i = next((i for i, b in enumerate(beats) if b["id"] == "title"), 0)
    poster_s = (
        args.poster_s
        if args.poster_s is not None
        else at[title_i] + beats[title_i]["seconds"] * 0.6
    )
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{poster_s:.2f}",
            "-i",
            str(master),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out / "poster.jpg"),
        ]
    )

    # captions and chapters
    (out / "captions.vtt").write_text(captions(beats, at, voice))
    act_of: dict[str, str] = {}
    current = ACTS[0][1]
    for b in beats:
        for first, name in ACTS:
            if b["id"] == first:
                current = name
        act_of[b["id"]] = current
    chapters = []
    for i, (b, s) in enumerate(zip(beats, at, strict=True)):
        e = at[i + 1] if i + 1 < len(beats) else total
        chapters.append(
            {
                "id": b["id"],
                "story": act_of[b["id"]],
                "title": TITLES.get(b["id"], b["id"]),
                "start_s": round(s, 2),
                "end_s": round(e, 2),
                "status": "record",
                "placeholder": None,
                "needs": None,
            }
        )
    (out / "chapters.json").write_text(json.dumps(chapters, indent=1) + "\n")
    vtt = ["WEBVTT", ""]
    for i, c in enumerate(chapters, 1):
        vtt += [str(i), f"{clock(c['start_s'])} --> {clock(c['end_s'])}", c["title"], ""]
    (out / "chapters.vtt").write_text("\n".join(vtt))

    # provenance: every beat's recorded clips
    beat_clips = json.loads((HERE / "beat-clips.json").read_text()).get("beats", {})
    clips = film.get("clips", {})
    prov_beats = []
    for c in chapters:
        shots = []
        ids = beat_clips.get(c["id"], [])
        if ids:
            entries = []
            for cid in ids:
                cl = clips.get(cid)
                if not cl:
                    continue
                video = local(cl.get("file", ""))
                run_dir = local(cl.get("run", ""))
                original = run_dir / "log" / "video" / f"{cl['episode']}.scene.mp4"
                # an original recording of any camera (scene or wrist) is its own source
                recorded = (run_dir / "log" / "video").resolve()
                is_original = video.resolve().parent == recorded and video.name.startswith(
                    f"{cl['episode']}."
                )
                entry: dict[str, Any] = {
                    "label": cl.get("label"),
                    "layout": "explainer",
                    "run": str(run_dir),
                    "episode": cl.get("episode"),
                    "video": str(video),
                    "video_sha256": cl.get("sha256"),
                    "video_fps": cl.get("fps"),
                    "control_steps_shown": [
                        cl.get("first_step", 0),
                        cl.get("first_step", 0)
                        + (cl.get("frames", 1) - 1) * cl.get("frame_every", 1),
                    ],
                    "outcome": None
                    if cl.get("delivered") is None
                    else ("delivered" if cl["delivered"] else "not delivered"),
                }
                if not is_original and (
                    original.is_file() or cl.get("hashes_compared") is not None
                ):
                    if (
                        original.is_file()
                    ):  # a run recorded without video (the host's single-skill screen) has none
                        entry["original_video"] = str(original)
                        entry["original_sha256"] = sha(original)
                    comparison = None
                    if (
                        cl.get("hashes_compared") is not None
                    ):  # the framed and mosaic batches' verification
                        comparison = {
                            "accepted": True,
                            "hashes_compared": cl["hashes_compared"],
                            "hash_mismatches": cl.get("hash_mismatches"),
                            "camera": cl.get("camera"),
                            "report": cl.get("report"),
                        }
                    else:
                        report = (
                            next(
                                iter(sorted(video.parent.glob(f"{cl['episode']}.rerender*.json"))),
                                None,
                            )
                            if video.parent.is_dir()
                            else None
                        )
                        if report is not None:
                            r = json.loads(report.read_text())
                            comparison = {
                                k: r.get(k)
                                for k in (
                                    "accepted",
                                    "hashes_compared",
                                    "hash_mismatches",
                                    "verdict_matches",
                                    "camera",
                                    "image_size",
                                )
                            }
                    entry["rerender"] = {"used": True, "reason": None, "comparison": comparison}
                entries.append(entry)
            shots.append(
                {"kind": "clip", "start_s": c["start_s"], "end_s": c["end_s"], "clips": entries}
            )
        else:
            shots.append(
                {"kind": "card", "start_s": c["start_s"], "end_s": c["end_s"], "title": c["title"]}
            )
        tb = next((b for b in beats if b["id"] == c["id"]), {})
        narration = None
        if tb.get("text"):
            narration = {
                "words": len(tb["text"].split()),
                "seconds": tb.get("voice_seconds"),
                "text": tb["text"],
                "voice": str(voice / f"{c['id']}.mp3") if voice and tb.get("voice") else None,
                "voice_sha256": sha(voice / f"{c['id']}.mp3")
                if voice and tb.get("voice")
                else None,
                "timing": "measured audio" if tb.get("voice") else "estimated",
            }
        prov_beats.append({**c, "subtitle": None, "narration": narration, "shots": shots})
    outputs = []
    for name, role in (
        ("film-master.mp4", "master"),
        ("film.mp4", "web"),
        ("poster.jpg", "poster"),
        ("captions.vtt", "captions"),
        ("chapters.vtt", "chapters"),
        ("chapters.json", "chapters"),
    ):
        p = out / name
        outputs.append({"role": role, "file": name, "sha256": sha(p), "bytes": p.stat().st_size})
    provenance = {
        "format": "fleetlab.film-provenance/1",
        "built": subprocess.run(
            ["date", "-Iseconds"], capture_output=True, text=True
        ).stdout.strip(),
        "command": "site/explainer: Remotion render, then export_film.py",
        "code": {
            "checkout": str(HERE.parents[1]),
            "commit": subprocess.run(
                ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
            "film_code_modified": bool(
                subprocess.run(
                    ["git", "-C", str(HERE), "status", "--porcelain", "site/explainer"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
        "manifest": {"title": args.title},
        "script": str(args.bound) if args.bound else None,
        "narration": str(voice) if voice else None,
        "data": {
            "film_data_sha256": sha(PUBLIC / "data" / "film-data.json"),
            "timeline_sha256": sha(PUBLIC / "data" / "timeline.json"),
            "inputs": film.get("provenance", {}).get("inputs", []),
        },
        "final": True,
        "options": {},
        "outputs": outputs,
        "seconds": round(total, 2),
        "beats": prov_beats,
        "warnings": [],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mayfly.explainer-cut/1",
                "title": args.title,
                "public": {
                    "aliases": {"w80": "w08", "w81": "w09"},
                    "forbid": ["(?<![A-Za-z0-9])w8[0-9](?![0-9])"],
                },
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"film folder {out}: {total:.1f} s, {len(chapters)} chapters, "
        f"{sum(len(s.get('clips', [])) for b in prov_beats for s in b['shots'])} clip entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
