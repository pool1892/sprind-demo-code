"""The part of the Model Context Protocol the two agent CLIs use, on stdin and stdout.

Both `claude -p` and `codex exec` start a tool server as a child process and
talk JSON-RPC 2.0 to it, one message per line: `initialize`, the
`notifications/initialized` notification, `tools/list`, `tools/call`, and
`ping`. That is all this module implements. It is written here instead of
taken from the `mcp` package so that the tool server's sandbox holds as little
code as possible (the `mcp` package pulls in an HTTP stack) and so that every
line the agent's tools run through can be read in this repository.

Protocol versions: the server answers with the version the client asked for
when it knows it, and with the newest it knows otherwise; the subset used here
is the same in all of them.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

__all__ = ["PROTOCOL_VERSIONS", "StdioServer", "Tool", "ToolResult", "image_part", "text_part"]

# Newest first. The CLIs measured (claude 2.1.293, codex 0.161.0) ask for one of these.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(png_base64: str, mime_type: str = "image/png") -> dict[str, Any]:
    return {"type": "image", "data": png_base64, "mimeType": mime_type}


@dataclass(frozen=True)
class ToolResult:
    """What a tool call returns: content parts (text, images) and whether it failed."""

    content: list[dict[str, Any]]
    is_error: bool = False

    @classmethod
    def text(cls, text: str, is_error: bool = False) -> ToolResult:
        return cls([text_part(text)], is_error)


@dataclass(frozen=True)
class Tool:
    """A tool as offered to the client: name, description, JSON schema of its arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult] = field(compare=False)


class StdioServer:
    """Answers JSON-RPC messages for a fixed set of tools."""

    def __init__(
        self,
        name: str,
        version: str,
        tools: Sequence[Tool],
        instructions: str | None = None,
        fallback: Callable[[str, dict[str, Any]], ToolResult] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.tools = {tool.name: tool for tool in tools}
        self.instructions = instructions
        # Answers (and records) a call to a tool that is not offered.
        self.fallback = fallback
        self._write_lock = threading.Lock()

    # One message in, at most one message out.

    def handle(self, message: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        if isinstance(message, list):  # a batch (protocol 2025-03-26 only)
            answers = [a for a in (self._one(m) for m in message) if a is not None]
            return answers or None
        return self._one(message)

    def _one(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        if "method" not in message:
            return None  # a response to something we never asked: ignore
        method = message["method"]
        request_id = message.get("id")
        params = message.get("params") or {}
        if request_id is None:
            return None  # a notification: initialized, cancelled, progress ...
        try:
            if method == "initialize":
                return _result(request_id, self._initialize(params))
            if method == "ping":
                return _result(request_id, {})
            if method == "tools/list":
                return _result(
                    request_id, {"tools": [self._describe(t) for t in self.tools.values()]}
                )
            if method == "tools/call":
                return self._call(request_id, params)
        except Exception as error:  # a bug in a tool must not kill the server
            return _error(request_id, INVALID_PARAMS, f"{type(error).__name__}: {error}")
        return _error(request_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        answer: dict[str, Any] = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": self.version},
        }
        if self.instructions:
            answer["instructions"] = self.instructions
        return answer

    @staticmethod
    def _describe(tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }

    def _call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self.tools.get(name) if isinstance(name, str) else None
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(request_id, INVALID_PARAMS, "arguments must be an object")
        if tool is None:
            if self.fallback is None or not isinstance(name, str):
                return _error(request_id, INVALID_PARAMS, f"unknown tool: {name!r}")
            result = self.fallback(name, arguments)
            return _result(request_id, {"content": result.content, "isError": result.is_error})
        result = tool.handler(arguments)
        return _result(request_id, {"content": result.content, "isError": result.is_error})

    # The loop.

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Read messages until stdin closes; answer each on stdout, one line per message."""
        stdin = stdin or _utf8(sys.stdin)
        stdout = stdout or _utf8(sys.stdout)
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self._send(stdout, _error(None, PARSE_ERROR, "not JSON"))
                continue
            answer = self.handle(message)
            if answer is not None:
                self._send(stdout, answer)

    def _send(self, stdout: TextIO, answer: dict[str, Any] | list[dict[str, Any]]) -> None:
        text = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            stdout.write(text + "\n")
            stdout.flush()


def _utf8(stream: TextIO) -> TextIO:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    return stream


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
