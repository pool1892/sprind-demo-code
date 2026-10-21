"""Record the explainer's narration with ElevenLabs, one file per beat, with timestamps.

Reads the bound script (`<bound>/page/index.md`, every number already bound), and for each
beat that has narration asks ElevenLabs' text-to-speech "with timestamps" endpoint for the
audio and the start time of every character, so the film can cue its graphics to the words.
Writes `<out>/<beat>.mp3` and `<out>/<beat>.json` (seconds, alignment, the request without the
key) and `<out>/voice.json` (voice, model, settings, the script's SHA-256).

The key is read from the environment variable ELEVENLABS_API_KEY only: it is sent in a request
header and never printed or written anywhere:

    python3 site/explainer/narrate.py --bound <bound dir> --out <dir>

Standard library only. Without the key it lists the requests it would send and stops.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get("ELEVENLABS_API_BASE", "https://api.elevenlabs.io")
VOICE = "nPczCjzI2devNBz1zQrb"  # Brian, chosen by ear on 19 October
MODEL = "eleven_multilingual_v2"  # the model of every earlier narration


def beats(script: str) -> list[tuple[str, str]]:
    body = re.sub(r"<!--.*?-->", "", script, flags=re.S)
    out = []
    for part in re.split(r"^## ", body, flags=re.M)[1:]:
        head, _, text = part.partition("\n")
        out.append((head.split(":")[0].strip(), " ".join(text.split())))
    return out


def paced(text: str, pause: float) -> str:
    """Short pauses between sentences: a break after each full stop, question mark or colon."""

    def brk(seconds: float) -> str:
        return f' <break time="{seconds:.2f}s" /> '

    out = re.sub(r"([.?!])\s+(?=[A-Z0-9\"“])", lambda m: m.group(1) + brk(pause), text)
    return re.sub(r":\s+", lambda m: ":" + brk(pause * 0.6), out)


def speak(key: str, text: str, previous: str, following: str, settings: dict | None) -> dict:
    payload = {
        "text": text,
        "model_id": MODEL,
        "voice_settings": settings or None,
        "previous_text": previous or None,
        "next_text": following or None,
    }
    req = urllib.request.Request(
        f"{API}/v1/text-to-speech/{VOICE}/with-timestamps?output_format=mp3_44100_192",
        data=json.dumps({k: v for k, v in payload.items() if v is not None}).encode(),
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:400].replace(key, "[key removed]")
            if error.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            raise SystemExit(f"narrate: HTTP {error.code}: {detail}") from None
        except urllib.error.URLError as error:
            if attempt < 3:
                time.sleep(4 * (attempt + 1))
                continue
            raise SystemExit(f"narrate: {error.reason}") from None
    raise SystemExit("narrate: no answer")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bound", type=Path, required=True, help="the bound script's folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="beat ids to (re)record; default all")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--similarity", type=float, default=0.75)
    ap.add_argument("--style", type=float, default=0.0)
    ap.add_argument(
        "--plain",
        action="store_true",
        help="no voice settings and no neighbouring text (the earlier narrations' request)",
    )
    ap.add_argument("--no-context", action="store_true", help="no neighbouring text")
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds of pause after each sentence (a <break> tag)",
    )
    args = ap.parse_args()
    page = (args.bound.expanduser() / "page" / "index.md").read_text()
    spoken = [(b, t) for b, t in beats(page) if t]
    settings = {
        "stability": args.stability,
        "similarity_boost": args.similarity,
        "style": args.style,
        "use_speaker_boost": True,
        "speed": args.speed,
    }
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    out = args.out.expanduser()
    if not key:
        for b, t in spoken:
            print(f"would record {b}: {len(t.split())} words")
        print("narrate: ELEVENLABS_API_KEY is not set; nothing was sent.", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for i, (beat, text) in enumerate(spoken):
        if args.only and beat not in args.only:
            continue
        context = not (args.plain or args.no_context)
        previous = spoken[i - 1][1] if i > 0 and context else ""
        following = spoken[i + 1][1] if i + 1 < len(spoken) and context else ""
        said = paced(text, args.pause) if args.pause > 0 else text
        answer = speak(key, said, previous, following, None if args.plain else settings)
        audio = base64.b64decode(answer["audio_base64"])
        (out / f"{beat}.mp3").write_bytes(audio)
        align = answer.get("alignment") or {}
        ends = align.get("character_end_times_seconds") or [0.0]
        seconds = float(ends[-1])
        total += seconds
        (out / f"{beat}.json").write_text(
            json.dumps(
                {
                    "beat": beat,
                    "text": text,
                    "seconds": seconds,
                    "alignment": align,
                    "normalized_alignment": answer.get("normalized_alignment"),
                    "voice_id": VOICE,
                    "model_id": MODEL,
                    "voice_settings": settings,
                    "mp3_sha256": hashlib.sha256(audio).hexdigest(),
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"recorded {beat}: {seconds:.2f} s")
    (out / "voice.json").write_text(
        json.dumps(
            {
                "voice_id": VOICE,
                "voice": "Brian",
                "model_id": MODEL,
                "voice_settings": settings,
                "script": str(args.bound),
                "script_sha256": hashlib.sha256(page.encode()).hexdigest(),
                "beats": [b for b, _ in spoken],
                "recorded_seconds": round(total, 2),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"narration: {total:.1f} s in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
