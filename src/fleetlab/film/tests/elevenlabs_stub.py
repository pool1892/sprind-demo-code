"""A local stand-in for ElevenLabs' API, for testing `tools/film-narrate` without a key.

    python -m fleetlab.film.tests.elevenlabs_stub PORT_FILE LOG_FILE

Serves on 127.0.0.1 at a free port (written to PORT_FILE). Every request must carry the
header `xi-api-key` equal to the environment variable `STUB_KEY`, else 401. GET /v1/voices
and /v1/models answer with small listings; POST /v1/text-to-speech/<voice> answers with an
MP3 of a tone as long as the text would take to read, except that a request for
`eleven_v4` with `previous_text` is refused with 422 (so the plainer fallback is
exercised). LOG_FILE gets one JSON line per request: method, path, whether the key
matched, and the body's fields; never the key.

A file `<LOG_FILE>.mode`, read at every request, changes the behaviour: `reflect` echoes
the received key back (in the voice listing's descriptions, in a 500 for /v1/models and in
the 422 refusals), as a misbehaving server might; `fail` answers every text-to-speech
request with 500.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

VOICES: dict[str, Any] = {
    "voices": [
        {"voice_id": "f1", "name": "Rachel", "category": "premade",
         "labels": {"gender": "female", "accent": "american", "use_case": "narration"}},
        {"voice_id": "m-uk", "name": "George", "category": "premade",
         "labels": {"gender": "male", "accent": "british", "use_case": "narration"}},
        {"voice_id": "m-game", "name": "Callum", "category": "premade",
         "labels": {"gender": "male", "accent": "american", "use_case": "characters, video games"}},
        {"voice_id": "m-brian", "name": "Brian", "category": "premade",
         "labels": {"gender": "male", "accent": "american", "use_case": "narration",
                    "description": "deep"}},
        {"voice_id": "m-adam", "name": "Adam", "category": "premade",
         "labels": {"gender": "male", "accent": "american", "use_case": "narration"}},
        {"voice_id": "m-clone", "name": "My clone", "category": "cloned",
         "labels": {"gender": "male", "accent": "american", "use_case": "narration"}},
    ]
}  # fmt: skip
MODELS: list[dict[str, Any]] = [
    {
        "model_id": "eleven_multilingual_v2",
        "can_do_text_to_speech": True,
        "languages": [{"name": "English"}],
    },
    {"model_id": "eleven_v4", "can_do_text_to_speech": True, "languages": [{"name": "English"}]},
    {
        "model_id": "eleven_flash_v2_5",
        "can_do_text_to_speech": True,
        "languages": [{"name": "English"}],
    },
]


def tone(seconds: float) -> bytes:
    import imageio_ffmpeg  # type: ignore[import-untyped]

    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=220:duration={seconds:.2f}", "-ac", "1", "-ar", "44100",
         "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"],
        capture_output=True,
        check=True,
    )  # fmt: skip
    return out.stdout


class Handler(BaseHTTPRequestHandler):
    log_file: Path
    key: str

    def _mode(self) -> str:
        path = self.log_file.with_name(self.log_file.name + ".mode")
        return path.read_text().strip() if path.is_file() else ""

    def _log(self, entry: dict[str, Any]) -> None:
        with self.log_file.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        ok = self.headers.get("xi-api-key") == self.key
        if not ok:
            self._send(401, b'{"detail": "invalid api key"}')
        return ok

    def do_GET(self) -> None:
        authorised = self._authorised()
        self._log({"method": "GET", "path": self.path, "key_ok": authorised})
        if not authorised:
            return
        reflect = self._mode() == "reflect"
        if self.path.startswith("/v1/voices"):
            listing = json.loads(json.dumps(VOICES))
            if reflect:
                listing["voices"][0]["description"] = f"seen with key {self.key}"
            self._send(200, json.dumps(listing).encode())
        elif self.path.startswith("/v1/models"):
            if reflect:
                self._send(500, f'{{"detail": "trouble with key {self.key}"}}'.encode())
            else:
                self._send(200, json.dumps(MODELS).encode())
        else:
            self._send(404, b'{"detail": "not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        authorised = self._authorised()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
        self._log(
            {"method": "POST", "path": self.path, "key_ok": authorised, "fields": sorted(body)}
        )
        if not authorised:
            return
        if not self.path.startswith("/v1/text-to-speech/"):
            self._send(404, b'{"detail": "not found"}')
            return
        mode = self._mode()
        if mode == "fail":
            self._send(500, b'{"detail": "service unavailable"}')
            return
        if body.get("model_id") == "eleven_v4" and "previous_text" in body:
            echo = f" for key {self.key}" if mode == "reflect" else ""
            message = f'{{"detail": [{{"msg": "previous_text is not supported{echo}"}}]}}'
            self._send(422, message.encode())
            return
        words = len(str(body.get("text", "")).split())
        self._send(200, tone(max(0.5, words / 2.5)), "audio/mpeg")

    def log_message(self, format: str, *args: Any) -> None:  # quiet
        return


def main() -> int:
    port_file, log_file = Path(sys.argv[1]), Path(sys.argv[2])
    Handler.log_file = log_file
    Handler.key = os.environ["STUB_KEY"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port_file.write_text(str(server.server_address[1]))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
