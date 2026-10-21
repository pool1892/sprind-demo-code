"""Show exactly what each backend would send to its model. Calls no model.

Starts `wire_capture.py`, points the CLI at it, and prints for every
combination of backend and sandbox: the tools on offer, the instruction text,
and whether anything of the machine leaks into the request (home directory,
the account's e-mail address, a CLAUDE.md / AGENTS.md planted in the working
directory, the user's global instruction files).

    python wire_check.py            # both backends, sandbox none and bwrap
    python wire_check.py claude     # one backend
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import toy
from agent_runtime import run_agent

TRIPWIRES = {
    "CLAUDE.md": "Project marker TRIPWIRE-CLAUDE-MD. Always mention it.\n",
    "AGENTS.md": "Project marker TRIPWIRE-AGENTS-MD. Always mention it.\n",
}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def leak_markers() -> dict[str, str]:
    """Strings that must not show up in a request, with a label each."""
    marks = {
        "planted CLAUDE.md": "TRIPWIRE-CLAUDE-MD",
        "planted AGENTS.md": "TRIPWIRE-AGENTS-MD",
        "home directory path": str(Path.home()),
        "repository name": "code/robots",
    }
    account = Path.home() / ".claude.json"
    if account.exists():
        email = (json.loads(account.read_text()).get("oauthAccount") or {}).get("emailAddress")
        if email:
            marks["account e-mail"] = email
    for label, path in {
        "user's ~/.codex/AGENTS.md": Path.home() / ".codex/AGENTS.md",
        "user's ~/.claude/CLAUDE.md": Path.home() / ".claude/CLAUDE.md",
    }.items():
        if path.exists():
            words = [w for w in re.findall(r"[A-Za-z]{6,}", path.read_text())]
            if len(words) > 12:
                marks[label] = " ".join(words[8:11])  # an unlikely three-word run
    return marks


def tool_names(body: dict) -> list[str]:
    names = []

    def walk(tool, prefix=""):
        if tool.get("type") == "namespace":
            for sub in tool.get("tools", []):
                walk(sub, tool.get("name", "") + ".")
        else:
            names.append(prefix + (tool.get("name") or tool.get("type") or "?"))
            if tool.get("name") == "exec":  # codex code mode: tools nested inside it
                for nested in re.findall(r"^### `([^`]+)`", tool.get("description", ""), flags=re.M):
                    names.append(f"{prefix}exec>{nested}")

    for tool in body.get("tools") or []:
        walk(tool)
    for item in body.get("input") or []:
        if item.get("type") == "additional_tools":
            for tool in item.get("tools", []):
                walk(tool)
    return names


def texts(body: dict) -> list[tuple[str, str]]:
    out = []
    for block in body.get("system") or []:
        out.append(("system", block.get("text", "")))
    if body.get("instructions"):
        out.append(("instructions", body["instructions"]))
    for msg in (body.get("messages") or []) + (body.get("input") or []):
        content = msg.get("content")
        if isinstance(content, str):
            out.append((msg.get("role", "?"), content))
        for part in content if isinstance(content, list) else []:
            if part.get("text"):
                out.append((msg.get("role", "?"), part["text"]))
    return out


def main() -> None:
    marks = leak_markers()
    wanted = sys.argv[1:] or ["claude", "codex"]
    for backend, model in (("claude", "opus"), ("codex", "gpt-6.1-sol")):
        if backend not in wanted:
            continue
        for sandbox in ("none", "bwrap"):
            capture = Path(tempfile.mkstemp(suffix=".jsonl")[1])
            server = subprocess.Popen(
                [sys.executable, str(toy.HERE / "wire_capture.py"), str(capture)],
                stdout=subprocess.PIPE, text=True,
            )
            port = server.stdout.readline().strip()
            base = f"http://127.0.0.1:{port}"
            extra, env = [], {}
            if backend == "claude":
                env = {"ANTHROPIC_BASE_URL": base}
            else:
                extra = ["-c", f'openai_base_url="{base}/v1"', "-c", f'chatgpt_base_url="{base}/backend-api/"']
            try:
                run_agent(
                    backend, model, "SYSTEM-PROMPT-PROBE", "USER-PROMPT-PROBE",
                    tools=toy.tool_server(), schema=SCHEMA, sandbox=sandbox, timeout=90,
                    _extra_argv=extra, _env=env, _plant=TRIPWIRES, _record=False,
                )
            finally:
                time.sleep(0.2)
                server.terminate()
            body = None
            for line in capture.read_text().splitlines():
                candidate = json.loads(line).get("body")
                if isinstance(candidate, dict) and ("messages" in candidate or "input" in candidate):
                    body = candidate
                    break
            capture.unlink()
            print(f"\n=== {backend} / sandbox={sandbox}")
            if body is None:
                print("    no model request captured")
                continue
            blob = json.dumps(body)
            print(f"    request size: {len(blob)} bytes, model: {body.get('model')}")
            print(f"    tools offered: {tool_names(body)}")
            for role, text in texts(body):
                shown = text if len(text) < 160 else text[:157] + "..."
                print(f"    [{role}, {len(text)} chars] {shown!r}")
            for label, needle in marks.items():
                print(f"    leak check, {label}: {'LEAKS' if needle in blob else 'absent'}")


if __name__ == "__main__":
    main()
