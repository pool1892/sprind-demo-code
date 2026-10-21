"""Does the claude invocation ignore the user's own setup? Calls no model.

A throwaway home directory is filled with everything a user can configure: a
user CLAUDE.md, an auto-memory file for the working directory, a settings.json
with hooks, a user MCP server and a skill. A dummy API key and the local
recorder (`wire_capture.py`) stand in for the vendor, so the request can be
read without a login and without a model.

Run once with our flags and environment, and once as a control with the
isolation switches removed, to prove that the planted setup is live.

    python isolation_check.py

codex needs no such check: it always runs with a private HOME and CODEX_HOME
(see `agent_runtime.py`), so the user's setup is not there to be loaded.
`wire_check.py` shows that for both backends.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import toy
from agent_runtime import CLAUDE_ENV, _claude_argv, claude_bin

MARKS = {
    "user CLAUDE.md": "USERMD-MARK-1",
    "auto-memory file": "AUTOMEM-MARK-2",
    "hook output": "HOOK-MARK-3",
    "user skill": "SKILL-MARK-4",
}


def plant(home: Path, cwd: Path) -> Path:
    (home / ".claude").mkdir()
    (home / ".claude/CLAUDE.md").write_text("User memory marker USERMD-MARK-1. Always mention it.\n")
    encoded = "".join(c if c.isalnum() else "-" for c in str(cwd))
    memory = home / ".claude/projects" / encoded / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("- [note](note.md) — auto memory marker AUTOMEM-MARK-2\n")
    (memory / "note.md").write_text("AUTOMEM-MARK-2 details\n")
    hook_flag = home / "hook-ran.txt"
    hooks = {
        "SessionStart": [{"hooks": [{"type": "command", "command": f"touch {hook_flag}"}]}],
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": f"touch {hook_flag}; echo HOOK-MARK-3"}]}],
    }
    (home / ".claude/settings.json").write_text(json.dumps({"hooks": hooks}))
    (home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"usermcp": {"command": "/usr/bin/true", "args": []}}, "hasCompletedOnboarding": True})
    )
    skill = home / ".claude/skills/demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: SKILL-MARK-4 demo skill\n---\nDo things.\n")
    return hook_flag


def run(label: str, argv: list[str], env: dict, cwd: Path, hook_flag: Path) -> None:
    capture = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    server = subprocess.Popen(
        [sys.executable, str(toy.HERE / "wire_capture.py"), str(capture)], stdout=subprocess.PIPE, text=True
    )
    port = server.stdout.readline().strip()
    env = dict(env, ANTHROPIC_BASE_URL=f"http://127.0.0.1:{port}", ANTHROPIC_API_KEY="sk-ant-dummy-not-a-key")
    hook_flag.unlink(missing_ok=True)
    message = json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "ping"}]}})
    done = subprocess.run(argv, input=message + "\n", capture_output=True, text=True, env=env, cwd=cwd, timeout=90)
    time.sleep(0.2)
    server.terminate()
    body = None
    for line in capture.read_text().splitlines():
        candidate = json.loads(line).get("body")
        if isinstance(candidate, dict) and "messages" in candidate:
            body = candidate
            break
    capture.unlink()
    init = {}
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("subtype") == "init":
            init = event
    blob = json.dumps(body) if body else ""
    print(f"\n=== {label}")
    if body is None:
        print("    no request captured")
        return
    for what, mark in MARKS.items():
        print(f"    {what}: {'LOADED' if mark in blob else 'not loaded'}")
    print(f"    hook command ran: {hook_flag.exists()}")
    print(f"    user MCP server seen by the CLI: {[s['name'] for s in init.get('mcp_servers') or []]}")
    print(f"    skills seen by the CLI: {len(init.get('skills') or [])}")
    print(f"    tools in the request: {len(body.get('tools') or [])}")


def main() -> None:
    cwd = Path(tempfile.mkdtemp(prefix="isolation-cwd-", dir=toy.PROBE))
    home = Path(tempfile.mkdtemp(prefix="isolation-home-", dir=toy.PROBE))
    hook_flag = plant(home, cwd)
    base_env = {"HOME": str(home), "PATH": "/usr/bin", "LANG": "C.UTF-8"}

    ours = _claude_argv(claude_bin(), "haiku", "SYSTEM", None, None, None, None)
    run("our flags and environment", ours, dict(base_env, **CLAUDE_ENV), cwd, hook_flag)
    run("our flags without CLAUDE_CODE_DISABLE_AUTO_MEMORY", ours, base_env, cwd, hook_flag)

    control = list(ours)
    for flag, takes_value in (("--setting-sources", True), ("--tools", True), ("--strict-mcp-config", False), ("--disable-slash-commands", False)):
        i = control.index(flag)
        del control[i : i + (2 if takes_value else 1)]
    run("control: isolation flags removed", control, base_env, cwd, hook_flag)
    shutil.rmtree(home, ignore_errors=True)
    shutil.rmtree(cwd, ignore_errors=True)


if __name__ == "__main__":
    main()
