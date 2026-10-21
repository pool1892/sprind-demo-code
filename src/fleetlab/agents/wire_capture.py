"""Record what a CLI would send to its model, without sending it (agent-runtime spike).

A small HTTP server on 127.0.0.1. Point a CLI at it instead of at the vendor
(`ANTHROPIC_BASE_URL` for claude, `-c openai_base_url=...` for codex), run one
call, and read the captured request body: the system prompt, the messages and
the tools on offer. Every request is answered with an error, so no model is
called and nothing is billed.

Request headers are never stored: they carry the login token.

    python -m fleetlab.agents.wire_capture OUT.jsonl     # prints the port, then serves
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def decode(raw: bytes, encoding: str) -> tuple[bytes, str]:
    """The body without its content encoding (codex compresses with zstd)."""
    try:
        if encoding == "gzip":
            return gzip.decompress(raw), ""
        if encoding == "zstd":
            done = subprocess.run(
                ["zstd", "-d", "-c", "-q"], input=raw, capture_output=True, timeout=30
            )
            if done.returncode != 0:
                return b"", f"zstd failed: {done.stderr[-200:]!r}"
            return done.stdout, ""
    except Exception as exc:  # keep serving whatever arrives
        return b"", f"decode failed: {exc}"
    return raw, ""


def main() -> None:
    out_path = sys.argv[1]

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            encoding = (self.headers.get("content-encoding") or "").lower()
            raw, note = decode(raw, encoding)
            body: Any
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                body = {"_unparsed_bytes": len(raw)}
            with open(out_path, "a") as handle:
                record = {
                    "method": self.command,
                    "path": self.path,
                    "content_encoding": encoding,
                    "note": note,
                    "body": body,
                }
                handle.write(json.dumps(record) + "\n")
            payload = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "captured by wire_capture, not forwarded",
                    },
                }
            ).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = do_HEAD = _handle

        def log_message(self, *args: Any) -> None:  # no access log
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
