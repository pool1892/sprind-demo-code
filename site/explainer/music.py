"""Compose the explainer's music bed with ElevenLabs Music (instrumental, sized to the film).

    python3 site/explainer/music.py --seconds 285 --out <dir>/music.mp3

The key comes from the environment variable ELEVENLABS_API_KEY only; it is sent in a header
and never printed or written. Writes the MP3 and a JSON beside it with the
prompt, the length and the file's SHA-256. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get("ELEVENLABS_API_BASE", "https://api.elevenlabs.io")

PROMPT = (
    "Instrumental background score for a calm, confident technology explainer film about robots "
    "and AI engineering. Warm felt piano and soft analog synth pulses over a light, steady beat "
    "around 96 BPM; airy pads; optimistic and curious, never dramatic; sits under a narrator "
    "without competing (no lead melody in the speech range, no vocals). A gentle lift around the "
    "middle and a quiet, resolved ending."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        print("music: ELEVENLABS_API_KEY is not set; nothing was sent.", file=sys.stderr)
        return 2
    body = {
        "prompt": args.prompt,
        "music_length_ms": int(args.seconds * 1000),
        "model_id": "music_v1",
    }
    req = urllib.request.Request(
        f"{API}/v1/music?output_format=mp3_44100_192",
        data=json.dumps(body).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:600].replace(key, "[key removed]")
        print(f"music: HTTP {error.code}: {detail}", file=sys.stderr)
        return 1
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)
    meta = {
        "prompt": args.prompt,
        "music_length_ms": body["music_length_ms"],
        "model_id": body["model_id"],
        "bytes": len(audio),
        "sha256": hashlib.sha256(audio).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"music: {len(audio)} bytes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
