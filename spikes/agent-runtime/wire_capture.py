"""Record what a headless agent CLI would send to its model, without sending it.

A tiny HTTP server on 127.0.0.1. Point the CLI at it instead of at the vendor
(`ANTHROPIC_BASE_URL` for claude, `-c openai_base_url=...` for codex), run one
call, and read the captured request body: the system prompt, the messages and
the list of tools the model is offered. The server answers every request with
an error, so no model is called and nothing is billed.

Request headers are never stored: they carry the login token.

    python wire_capture.py OUT.jsonl        # prints the port, then serves
"""

from __future__ import annotations

import gzip
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import zstandard  # type: ignore
except ImportError:  # codex compresses request bodies with zstd
    zstandard = None


def main() -> None:
    out_path = sys.argv[1]

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            encoding = (self.headers.get("content-encoding") or "").lower()
            note = ""
            try:
                if encoding == "gzip":
                    raw = gzip.decompress(raw)
                elif encoding == "zstd":
                    if zstandard is None:
                        note = "zstd body, zstandard not installed"
                        raw = b""
                    else:
                        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
            except Exception as exc:  # keep serving whatever arrives
                note = f"decode failed: {exc}"
                raw = b""
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                body = {"_unparsed_bytes": len(raw)}
            with open(out_path, "a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "method": self.command,
                            "path": self.path,
                            "upgrade": self.headers.get("upgrade"),
                            "content_encoding": encoding,
                            "note": note,
                            "body": body,
                        }
                    )
                    + "\n"
                )
            payload = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "captured by wire_capture.py, not forwarded",
                    },
                }
            ).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = do_HEAD = _handle

        def log_message(self, *args) -> None:  # silence the access log
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
