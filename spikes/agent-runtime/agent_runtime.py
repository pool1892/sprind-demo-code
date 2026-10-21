"""Call an analysis agent through the subscription command-line tools.

One function, `run_agent`, drives either `claude -p` (Anthropic) or
`codex exec` (OpenAI) as a child process and gives back a schema-checked
answer, the token count and the wall-clock time. The agent gets a system
prompt, a user prompt, images, and exactly the tools of one tool server
(`toy_tools.py` in this spike). Nothing of the user's or the project's setup is
loaded: no CLAUDE.md or AGENTS.md, no memory, no skills, no hooks, no plugins,
no other MCP servers.

Three levels of walling-in, chosen with `sandbox=`:

    "none"    The CLI runs as the user. The model is blind only because it has
              no tool that reads files.
    "bwrap"   The CLI runs in a bubblewrap sandbox that shows the system
              libraries, the login file and nothing else of the home directory.
              The tool server runs in a second sandbox inside it that shows only
              the workspace and has no network.
    "docker"  The same layout in a Docker container.

Everything here was measured with claude 2.1.292 and codex 0.160.1. Both tools
update themselves almost daily, so the flag set is pinned to a binary path and
`codex` is run with `--strict-config`, which turns a renamed setting into a
loud error instead of a silently re-opened door.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

HOME = Path.home()
STATE = Path(
    os.environ.get("ROBOTS_AGENT_STATE", HOME / ".cache/robots-probes/agent-runtime/state")
)

# Where the model's world lives inside a sandbox. Neutral names on purpose:
# both CLIs can tell the model its working directory.
BOX_HOME = "/home/agent"
BOX_WORK = "/work"
BOX_RUN = "/run/agent"
BOX_TOOLS = "/opt/tools"
BOX_CLI = "/opt/cli"

DOCKER_IMAGE = os.environ.get("ROBOTS_AGENT_IMAGE", "debian:stable-slim")


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def claude_bin() -> Path:
    """The real binary, not the `~/.local/bin/claude` launcher script."""
    return _real(
        Path(os.environ.get("ROBOTS_CLAUDE_BIN", HOME / ".local/share/mise/installs/claude/latest/claude"))
    )


def codex_bin() -> Path:
    return _real(
        Path(os.environ.get("ROBOTS_CODEX_BIN", HOME / ".local/share/mise/installs/codex/latest/bin/codex"))
    )


@dataclass
class ToolServer:
    """The one tool server the agent may use: a Python MCP server on stdio."""

    script: Path  # e.g. toy_tools.py
    python: Path  # interpreter of a venv that has `mcp` installed
    workspace: Path  # the exported data the tools may serve
    tools: list[str]  # tool names the model may call
    name: str = "robots"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    ok: bool
    result: Any  # the structured answer, checked against the schema
    usage: dict  # input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, ...
    seconds: float  # wall clock of the whole child process
    error: str | None = None  # one of ERROR_KINDS, or None
    error_detail: str = ""
    text: str = ""  # the raw final message
    tool_calls: list = field(default_factory=list)
    model: str = ""  # the model id claude reported; for codex the one we asked for
    meta: dict = field(default_factory=dict)
    run_dir: str = ""

    def __iter__(self):  # allows: result, usage, seconds = run_agent(...)
        return iter((self.result, self.usage, self.seconds))


ERROR_KINDS = (
    "timeout",  # we killed the CLI after `timeout` seconds
    "rate_limit",  # the subscription's usage limit or a 429
    "auth",  # login missing or expired
    "overloaded",  # vendor-side 5xx / overloaded
    "refusal",  # the model declined
    "malformed_output",  # no answer, not JSON, or JSON that fails the schema
    "tool_server",  # our MCP server did not start
    "bad_request",  # unknown model, schema the vendor rejects: our bug, not worth retrying
    "cli_error",  # anything else
)


# --------------------------------------------------------------------------
# The two command lines
# --------------------------------------------------------------------------


def _claude_argv(binary, model, system, tool_cmd, tools, schema, effort) -> list[str]:
    argv = [
        str(binary),
        "-p",
        "--model", model,
        "--system-prompt", system,  # replaces Claude Code's own system prompt
        "--tools", "",  # no built-in tools: no Bash, Read, Write, WebFetch, Agent
        "--strict-mcp-config",  # only the MCP server named below, none of the user's
        "--setting-sources", "",  # no user/project/local settings: no CLAUDE.md, hooks, plugins
        "--disable-slash-commands",  # no skills
        "--no-session-persistence",  # leave no transcript in ~/.claude/projects
        "--permission-mode", "dontAsk",  # whatever is not allowed below is denied, never asked
        "--input-format", "stream-json",  # the prompt arrives on stdin, so it can carry images
        "--output-format", "stream-json",
        "--verbose",  # required by stream-json output
    ]
    if tools is not None:
        config = {"mcpServers": {tools.name: {"command": tool_cmd[0], "args": tool_cmd[1:]}}}
        argv += ["--mcp-config", json.dumps(config)]
        argv += ["--allowedTools", ",".join(f"mcp__{tools.name}__{t}" for t in tools.tools)]
    if schema is not None:
        argv += ["--json-schema", json.dumps(schema)]
    if effort:
        argv += ["--effort", effort]
    return argv


CODEX_FEATURES_OFF = [
    "shell_tool", "unified_exec",  # no shell
    "apps", "plugins", "multi_agent", "goals", "sleep_tool",
    "view_image", "image_generation", "browser_use", "computer_use",
    "tool_suggest", "skill_search", "hooks",
]


def _toml(value) -> str:
    return json.dumps(value)  # JSON strings, numbers, booleans and lists are valid TOML


def _codex_argv(binary, model, system_file, tool_cmd, tools, schema_file, effort, image_files, last_file, cwd) -> list[str]:
    def c(key, value):
        return ["-c", f"{key}={_toml(value)}"]

    argv = [
        str(binary), "exec",
        "--ignore-user-config",  # not ~/.codex/config.toml (belt and braces: CODEX_HOME is private too)
        "--ignore-rules",
        "--ephemeral",  # no session file
        "--skip-git-repo-check",
        "--strict-config",  # an unknown setting is an error, not a silent no-op
        "-s", "read-only",
        "-C", str(cwd),
        "-m", model,
        "--json",
        "-o", str(last_file),
    ]
    for image in image_files:
        argv += ["-i", str(image)]
    argv += c("model_instructions_file", str(system_file))  # replaces Codex's 20 kB of instructions
    argv += c("project_doc_max_bytes", 0)  # no AGENTS.md from the working directory
    argv += c("web_search", "disabled")
    argv += c("agents.enabled", False)
    argv += c("skills.include_instructions", False)
    argv += c("tools.experimental_request_user_input.enabled", False)
    argv += c("include_environment_context", False)  # do not tell the model cwd, shell, date
    argv += c("include_permissions_instructions", False)
    argv += c("include_apps_instructions", False)
    argv += c("include_collaboration_mode_instructions", False)
    # Codex's models work in "code mode": one built-in `exec` tool runs
    # JavaScript in a bare V8 isolate (no file system, no network) and reaches
    # other tools through it. Keep the built-in namespaces out of it and offer
    # our tools as ordinary direct tool calls.
    argv += c("features.code_mode.excluded_tool_namespaces", ["functions", "clock"])
    for feature in CODEX_FEATURES_OFF:
        argv += ["--disable", feature]
    if tools is not None:
        ns = f"mcp_servers.{tools.name}"
        argv += c(f"{ns}.command", tool_cmd[0])
        argv += c(f"{ns}.args", tool_cmd[1:])
        argv += c(f"{ns}.required", True)  # fail the call if the tool server does not start
        argv += c(f"{ns}.enabled_tools", tools.tools)
        argv += c(f"{ns}.default_tools_approval_mode", "approve")  # nobody is there to click
        argv += c("features.code_mode.direct_only_tool_namespaces", [f"mcp__{tools.name}"])
    if schema_file is not None:
        argv += ["--output-schema", str(schema_file)]
    if effort:
        argv += c("model_reasoning_effort", effort)
    return argv


# --------------------------------------------------------------------------
# Sandboxes
# --------------------------------------------------------------------------

# Switches that cannot be given as flags. Measured with a planted memory file:
# `--setting-sources ""` stops CLAUDE.md, hooks, skills and user MCP servers,
# but claude still loads the auto-memory index of the working directory
# (~/.claude/projects/<cwd>/memory/MEMORY.md) unless this variable is set.
CLAUDE_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "DISABLE_AUTOUPDATER": "1",
}


def _cli_env(backend: str, home: str, codex_home: str | None = None) -> dict[str, str]:
    """The whole environment of the CLI. Built from nothing: no variable of a
    surrounding Claude Code session (CLAUDECODE, CLAUDE_CODE_*, AI_AGENT ...)
    and none of the user's shell reaches the child."""
    env = {"HOME": home, "PATH": "/usr/bin", "LANG": "C.UTF-8"}
    if backend == "claude":
        env.update(CLAUDE_ENV)
    else:
        env["CODEX_HOME"] = codex_home or f"{home}/.codex"
    return env


SYSTEM_RO = [  # what a dynamically linked binary and TLS need, nothing else
    ("/usr", "/usr"),
    ("/etc/ssl", "/etc/ssl"),
    ("/etc/ca-certificates", "/etc/ca-certificates"),
]


def _python_mounts(python: Path) -> list[tuple[str, str]]:
    """(host dir, path in the sandbox) pairs a venv interpreter needs.

    `venv/bin/python` is a symlink, often through an alias directory
    (`cpython-3.12-...` -> `cpython-3.12.15-...`). Every directory on that
    chain has to exist in the sandbox under the name the link uses.
    """
    venv = python.parent.parent
    mounts = [(str(venv), str(venv))]
    hop = python
    for _ in range(8):
        if not hop.is_symlink():
            break
        hop = (hop.parent / os.readlink(hop)).absolute()
        base = hop.parent.parent  # .../bin/python3.12 -> the install directory
        pair = (os.path.realpath(base), str(base))
        if pair not in mounts:
            mounts.append(pair)
    return mounts


def _tool_bwrap(tools: ToolServer, box_script: str, log_in_box: str | None) -> list[str]:
    """The tool server's own sandbox: the workspace, Python, no network, no home."""
    cmd = ["bwrap", "--die-with-parent", "--unshare-all"]
    for src, dst in [("/usr", "/usr")]:
        cmd += ["--ro-bind", src, dst]
    cmd += ["--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64"]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for _, dst in _python_mounts(tools.python):
        cmd += ["--ro-bind", dst, dst]  # already mounted under this name in the outer sandbox
    cmd += ["--ro-bind", BOX_TOOLS, BOX_TOOLS, "--ro-bind", BOX_WORK, BOX_WORK]
    if log_in_box:
        cmd += ["--bind", str(Path(log_in_box).parent), str(Path(log_in_box).parent)]
    cmd += ["--chdir", BOX_WORK, "--"]
    cmd += [str(tools.python), box_script, "--workspace", BOX_WORK]
    if log_in_box:
        cmd += ["--log", log_in_box]
    return cmd + list(tools.extra_args)


def _codex_box_home() -> Path:
    """A CODEX_HOME for sandboxed calls. Holds caches, never the login."""
    path = STATE / "codex-box-home"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _login_mount(backend: str) -> tuple[str, str]:
    """(file on the host, where the CLI looks for it inside the sandbox)."""
    if backend == "claude":
        return str(HOME / ".claude/.credentials.json"), f"{BOX_HOME}/.claude/.credentials.json"
    return str(HOME / ".codex/auth.json"), f"{BOX_HOME}/.codex/auth.json"


def _bwrap_prefix(backend, binary: Path, tools: ToolServer | None, run_dir: Path) -> list[str]:
    cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--share-net"]
    for src, dst in SYSTEM_RO:
        cmd += ["--ro-bind", src, dst]
    cmd += ["--ro-bind", os.path.realpath("/etc/resolv.conf"), "/etc/resolv.conf"]
    cmd += ["--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64"]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    cmd += ["--tmpfs", "/home", "--dir", BOX_HOME]
    host_login, box_login = _login_mount(backend)
    if backend == "codex":
        # codex fills an empty CODEX_HOME on first use (model catalogue, bundled
        # skills, databases; about a second), so its private one is kept.
        cmd += ["--bind", str(_codex_box_home()), str(Path(box_login).parent)]
    else:
        cmd += ["--dir", str(Path(box_login).parent)]  # claude starts fast from nothing
    # The login file itself is bound read-write so that a token refresh inside
    # the sandbox lands in the real file.
    cmd += ["--bind", host_login, box_login]
    cmd += ["--ro-bind", str(binary.parent), BOX_CLI]
    cmd += ["--bind", str(run_dir), BOX_RUN]
    if tools is not None:
        cmd += ["--ro-bind", str(tools.workspace), BOX_WORK]
        # The script alone, not its directory: the directory is part of the
        # repository the agent must not see.
        cmd += ["--ro-bind", str(tools.script), f"{BOX_TOOLS}/{tools.script.name}"]
        for src, dst in _python_mounts(tools.python):
            cmd += ["--ro-bind", src, dst]
    else:
        cmd += ["--dir", BOX_WORK]
    cmd += ["--chdir", BOX_RUN + "/cwd"]
    cmd += ["--clearenv"]
    for key, value in _cli_env(backend, BOX_HOME).items():
        cmd += ["--setenv", key, value]
    return cmd + ["--"]


def _docker_prefix(backend, binary: Path, tools: ToolServer | None, run_dir: Path, name: str) -> list[str]:
    host_login, box_login = _login_mount(backend)
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp", "--tmpfs", f"{BOX_HOME}:uid={os.getuid()},gid={os.getgid()},mode=700",
        "--memory", "2g", "--pids-limit", "256",
        *[arg for key, value in _cli_env(backend, BOX_HOME).items() for arg in ("-e", f"{key}={value}")],
        *(["-v", f"{_codex_box_home()}:{Path(box_login).parent}"] if backend == "codex" else []),
        "-v", f"{host_login}:{box_login}",
        "-v", "/etc/ssl:/etc/ssl:ro", "-v", "/etc/ca-certificates:/etc/ca-certificates:ro",
        "-v", f"{binary.parent}:{BOX_CLI}:ro",
        "-v", f"{run_dir}:{BOX_RUN}",
        "-w", BOX_RUN + "/cwd",
    ]
    if tools is not None:
        cmd += ["-v", f"{tools.workspace}:{BOX_WORK}:ro", "-v", f"{tools.script}:{BOX_TOOLS}/{tools.script.name}:ro"]
        for src, dst in _python_mounts(tools.python):
            cmd += ["-v", f"{src}:{dst}:ro"]
    return cmd + [DOCKER_IMAGE]


# --------------------------------------------------------------------------
# Reading what the CLIs print
# --------------------------------------------------------------------------


def _classify(text: str) -> str:
    low = text.lower()
    # OpenAI's filter answers instead of the model: "This content was flagged for possible ..."
    if any(s in low for s in ("was flagged for", "content policy", "violates our usage policies", "usage policy")):
        return "refusal"
    if "mcp server" in low and ("failed to initialize" in low or "handshaking" in low):
        return "tool_server"
    if any(s in low for s in ("rate limit", "usage limit", "429", "too many requests", "limit reached")):
        return "rate_limit"
    if any(s in low for s in ("is not supported when using codex", "issue with the selected model", "invalid_json_schema", "invalid schema")):
        return "bad_request"
    if any(s in low for s in ("401", "unauthorized", "not logged in", "login", "authentication", "oauth token", "invalid api key")):
        return "auth"
    if any(s in low for s in ("overloaded", "529", "503", "502", "500 internal", "server_error", "service unavailable")):
        return "overloaded"
    return "cli_error"


def _parse_claude(lines: list[str], res: AgentResult) -> None:
    pending: dict[str, dict] = {}
    final = None
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            res.model = ev.get("model", "")
            res.meta["offered_tools"] = ev.get("tools")
            res.meta["mcp_servers"] = ev.get("mcp_servers")
            res.meta["cli_version"] = ev.get("claude_code_version")
        elif kind == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    call = {"name": block.get("name"), "arguments": block.get("input")}
                    pending[block.get("id")] = call
                    res.tool_calls.append(call)
        elif kind == "user":
            content = ev.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    parts = block.get("content")
                    kinds = [p.get("type") for p in parts] if isinstance(parts, list) else ["text"]
                    pending[block["tool_use_id"]]["result_parts"] = kinds
                    pending[block["tool_use_id"]]["is_error"] = bool(block.get("is_error"))
        elif kind == "rate_limit_event":
            res.meta["rate_limit"] = ev.get("rate_limit_info")
        elif kind == "result":
            final = ev
    if final is None:
        servers = res.meta.get("mcp_servers") or []
        if any(s.get("status") == "failed" for s in servers):
            res.error, res.error_detail = "tool_server", json.dumps(servers)
        else:
            res.error, res.error_detail = "cli_error", "no result event"
        return
    usage = final.get("usage") or {}
    res.usage = {
        # Claude reports cached input separately; add it up so both vendors mean the same thing.
        "input_tokens": usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0),
        "cached_input_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        "cost_usd_list_price": final.get("total_cost_usd"),
    }
    res.meta.update(
        num_turns=final.get("num_turns"),
        cli_ms=final.get("duration_ms"),
        api_ms=final.get("duration_api_ms"),
        ttft_ms=final.get("ttft_ms"),
        stop_reason=final.get("stop_reason"),
        subtype=final.get("subtype"),
        terminal_reason=final.get("terminal_reason"),
        api_error_status=final.get("api_error_status"),
        permission_denials=final.get("permission_denials"),
    )
    res.text = final.get("result") or ""
    if final.get("structured_output") is not None:
        res.result = final["structured_output"]
    servers = res.meta.get("mcp_servers") or []
    if any(s.get("status") not in ("connected", None) for s in servers):
        res.error, res.error_detail = "tool_server", json.dumps(servers)
    elif final.get("is_error"):
        status = final.get("api_error_status")
        detail = f"{final.get('subtype')} {status} {res.text}"
        if final.get("stop_reason") == "refusal":
            res.error = "refusal"
        elif status == 429:
            res.error = "rate_limit"
        elif status in (401, 403):
            res.error = "auth"
        elif status and status >= 500:
            res.error = "overloaded"
        elif status in (400, 404):
            res.error = "bad_request"
        elif "structured_output" in str(final.get("subtype")):
            res.error = "malformed_output"
        else:
            res.error = _classify(detail)
        res.error_detail = detail[:500]
    elif final.get("stop_reason") == "refusal":
        res.error, res.error_detail = "refusal", res.text[:500]


def _parse_codex(lines: list[str], res: AgentResult, last_file: Path) -> None:
    failed = None
    errors = []
    usage = None
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("type")
        item = ev.get("item") or {}
        if kind == "item.completed":
            if item.get("type") == "mcp_tool_call":
                result = item.get("result") or {}
                parts = [p.get("type") for p in (result.get("content") or [])]
                res.tool_calls.append(
                    {
                        "name": f"mcp__{item.get('server')}__{item.get('tool')}",
                        "arguments": item.get("arguments"),
                        "result_parts": parts,
                        "is_error": item.get("status") != "completed" or bool(item.get("error")),
                    }
                )
            elif item.get("type") == "agent_message":
                res.text = item.get("text") or res.text
            elif item.get("type") == "error":
                errors.append(item.get("message", ""))
            elif item.get("type") not in ("reasoning",):
                # Anything else (command_execution, file_change, web_search ...) would
                # mean a built-in tool ran. Record it so a test can assert it never does.
                res.meta.setdefault("builtin_items", []).append(
                    {k: item.get(k) for k in ("type", "command", "status", "tool", "name") if k in item}
                )
        elif kind == "turn.completed":
            usage = ev.get("usage") or {}
        elif kind == "turn.failed":
            failed = (ev.get("error") or {}).get("message", "")
        elif kind == "error":
            errors.append(ev.get("message", ""))
    res.model = res.meta.get("requested_model", "")  # codex --json does not report the model
    if usage:
        res.usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "cached_input_tokens": usage.get("cached_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_output_tokens", 0),
        }
    if last_file.exists():
        res.text = last_file.read_text() or res.text
    if failed is not None or (usage is None and errors):
        detail = failed if failed is not None else " | ".join(errors)
        res.error, res.error_detail = _classify(detail), detail[:500]
    if errors:
        res.meta["stream_errors"] = errors[:5]


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def run_agent(
    backend: str,
    model: str,
    system: str,
    prompt: str,
    images: list[str | Path] | None = None,
    tools: ToolServer | None = None,
    schema: dict | None = None,
    *,
    effort: str | None = None,
    timeout: float = 300.0,
    sandbox: str = "none",
    keep_run_dir: bool = True,
    _extra_argv: list[str] | None = None,  # appended to the CLI's arguments
    _argv_hook=None,  # rewrites the CLI's arguments (the canary test's deliberately unsafe variants)
    _env: dict | None = None,  # extra environment for the CLI (wire_check.py points it at a recorder)
    _plant: dict | None = None,  # files to put into the working directory (tripwires)
    _record: bool = True,  # False for runs that never reach a model
) -> AgentResult:
    """Run one agent call. Returns an `AgentResult`; unpacks as (result, usage, seconds).

    backend   "claude" or "codex"
    model     e.g. "opus", "haiku", "gpt-6.1-sol", "gpt-6-luna"
    images    paths of PNG/JPEG files shown to the model with the prompt
    tools     the tool server, or None for a call without tools
    schema    JSON Schema the answer must satisfy, or None for free text
    effort    reasoning effort ("low" ... "max"); None keeps the model's default
    """
    assert backend in ("claude", "codex")
    assert sandbox in ("none", "bwrap", "docker")
    images = [Path(p) for p in (images or [])]
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = STATE / "runs" / run_id
    (run_dir / "cwd").mkdir(parents=True)
    boxed = sandbox != "none"
    inside = (lambda p: BOX_RUN + "/" + str(Path(p).relative_to(run_dir))) if boxed else str

    binary = claude_bin() if backend == "claude" else codex_bin()
    box_binary = f"{BOX_CLI}/{binary.name}" if boxed else str(binary)

    # The tool server's log gets a directory of its own, so that its sandbox
    # can write there without seeing the prompt, the schema or the output.
    (run_dir / "toolbox").mkdir()
    tool_log = run_dir / "toolbox" / "tool_calls.jsonl"
    tool_log.touch()
    tool_cmd = None
    if tools is not None:
        if sandbox == "bwrap":
            tool_cmd = _tool_bwrap(tools, f"{BOX_TOOLS}/{tools.script.name}", inside(tool_log))
        elif sandbox == "docker":  # no second sandbox inside the container
            tool_cmd = [
                str(tools.python), f"{BOX_TOOLS}/{tools.script.name}",
                "--workspace", BOX_WORK, "--log", inside(tool_log),
            ] + list(tools.extra_args)
        else:
            tool_cmd = [
                str(tools.python), str(tools.script),
                "--workspace", str(tools.workspace), "--log", str(tool_log),
            ] + list(tools.extra_args)

    stdin_text = ""
    env: dict[str, str]
    if backend == "claude":
        argv = _claude_argv(box_binary, model, system, tool_cmd, tools, schema, effort)
        content = []
        for image in images:
            media = mimetypes.guess_type(image.name)[0] or "image/png"
            data = base64.b64encode(image.read_bytes()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": prompt})
        stdin_text = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
        env = _cli_env("claude", str(HOME))
    else:
        system_file = run_dir / "system.txt"
        system_file.write_text(system)
        schema_file = None
        if schema is not None:
            schema_file = run_dir / "schema.json"
            schema_file.write_text(json.dumps(schema))
        image_files = []
        for i, image in enumerate(images):
            copy = run_dir / f"image-{i:02d}{image.suffix}"
            shutil.copyfile(image, copy)
            image_files.append(inside(copy))
        last_file = run_dir / "last_message.txt"
        argv = _codex_argv(
            box_binary, model, inside(system_file), tool_cmd, tools,
            inside(schema_file) if schema_file else None, effort, image_files,
            inside(last_file), inside(run_dir / "cwd"),
        )
        stdin_text = prompt
        # A private HOME and CODEX_HOME: without them codex loads ~/.codex/AGENTS.md
        # and the skills in ~/.agents/skills even under --ignore-user-config.
        private_home = STATE / "codex-private-home"
        codex_home = private_home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        link = codex_home / "auth.json"
        if sandbox == "none" and not link.is_symlink():
            # A symlink, not a copy: codex rewrites auth.json in place when it
            # refreshes the token, so the real file stays the single source.
            # The price: this directory now holds a door to the real login
            # file. Nothing that walks it may write through links (this spike
            # learned that the hard way). The sandboxed modes
            # need no link, and since 2026-10-07 do not create one: there the
            # file is only mounted inside the sandbox.
            link.symlink_to(HOME / ".codex/auth.json")
        env = _cli_env("codex", str(private_home), str(codex_home))

    if _argv_hook:
        argv = _argv_hook(argv)
    if _extra_argv:
        argv += _extra_argv

    for name, text in (_plant or {}).items():
        (run_dir / "cwd" / name).write_text(text)
    env.update(_env or {})

    container = f"robots-agent-{run_id}"
    if sandbox == "bwrap":
        prefix = _bwrap_prefix(backend, binary, tools, run_dir)
        for key, value in (_env or {}).items():
            prefix[-1:-1] = ["--setenv", key, value]
        argv = prefix + argv
        env = {"PATH": "/usr/bin"}
    elif sandbox == "docker":
        prefix = _docker_prefix(backend, binary, tools, run_dir, container)
        for key, value in (_env or {}).items():
            prefix[-1:-1] = ["-e", f"{key}={value}"]
        argv = prefix + argv
        env = {"PATH": "/usr/bin", "HOME": str(HOME)}

    (run_dir / "argv.json").write_text(json.dumps(argv, indent=1))
    res = AgentResult(ok=False, result=None, usage={}, seconds=0.0, run_dir=str(run_dir))
    res.meta["backend"], res.meta["sandbox"], res.meta["requested_model"] = backend, sandbox, model
    res.meta["effort"] = effort

    started = time.perf_counter()
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=run_dir / "cwd", text=True, start_new_session=True,
    )
    stamped: list[tuple[float, str]] = []  # (seconds since start, line) for every event
    err_parts: list[str] = []

    def read_out() -> None:
        for line in proc.stdout:
            stamped.append((time.perf_counter() - started, line))
            # claude carries on without tools when the tool server fails to
            # start, and the model then guesses. Stop it at once instead.
            if backend == "claude" and tools is not None and '"subtype":"init"' in line:
                try:
                    servers = json.loads(line).get("mcp_servers") or []
                except ValueError:
                    servers = []
                if any(s.get("status") == "failed" for s in servers):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    readers = [threading.Thread(target=read_out), threading.Thread(target=lambda: err_parts.append(proc.stderr.read()))]
    for reader in readers:
        reader.start()
    try:
        proc.stdin.write(stdin_text)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the CLI and the tool server it started
        except ProcessLookupError:
            pass
        if sandbox == "docker":
            subprocess.run(["docker", "kill", container], capture_output=True)
        proc.wait()
    for reader in readers:
        reader.join(timeout=10)
    res.seconds = round(time.perf_counter() - started, 3)
    out, err = "".join(line for _, line in stamped), "".join(err_parts)
    (run_dir / "stdout.jsonl").write_text(out)
    (run_dir / "stderr.txt").write_text(err)
    res.meta["exit_code"] = proc.returncode

    # Where the time went: until the CLI was ready to call the model, until
    # the answer was complete, and the CLI's own shutdown after that.
    def first(marker: str) -> float | None:
        return next((round(t, 3) for t, line in stamped if marker in line), None)

    def last(marker: str) -> float | None:
        return next((round(t, 3) for t, line in reversed(stamped) if marker in line), None)

    if backend == "claude":
        res.meta["phases"] = {"ready_s": first('"subtype":"init"'), "answer_s": last('"type":"result"')}
    else:
        res.meta["phases"] = {"ready_s": first('"turn.started"'), "answer_s": last('"turn.completed"')}

    lines = out.splitlines()
    if backend == "claude":
        _parse_claude(lines, res)
    else:
        _parse_codex(lines, res, run_dir / "last_message.txt")
    res.meta["tool_log"] = [json.loads(l) for l in tool_log.read_text().splitlines() if l.strip()]

    if timed_out:
        res.error, res.error_detail = "timeout", f"killed after {timeout}s"
    elif res.error is None and proc.returncode != 0:
        res.error, res.error_detail = _classify(err or ""), (err or "")[-500:]

    if res.error is None:
        if schema is None:
            res.result = res.text
        else:
            if res.result is None:  # codex (and claude without structured_output): parse the text
                try:
                    res.result = json.loads(res.text)
                except ValueError:
                    res.error, res.error_detail = "malformed_output", f"not JSON: {res.text[:300]!r}"
            if res.error is None:
                try:
                    jsonschema.validate(res.result, schema)
                except jsonschema.ValidationError as exc:
                    res.error, res.error_detail = "malformed_output", exc.message[:300]
    res.ok = res.error is None

    summary = asdict(res)
    (run_dir / "result.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(STATE / ("calls.jsonl" if _record else "dry_runs.jsonl"), "a") as fh:  # the ledger
        fh.write(
            json.dumps(
                {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": run_id, "backend": backend,
                    "model": model, "effort": effort, "sandbox": sandbox, "seconds": res.seconds,
                    "ok": res.ok, "error": res.error, "usage": res.usage,
                    "tool_calls": len(res.tool_calls), "images": len(images),
                }
            )
            + "\n"
        )
    if not keep_run_dir and res.ok:
        shutil.rmtree(run_dir, ignore_errors=True)
    return res


RETRY = {
    # error kind: (retry?, wait in seconds before each further attempt)
    "timeout": (True, [0]),  # once more, straight away
    "malformed_output": (True, [0]),  # once more: a fresh sample usually parses
    "overloaded": (True, [5, 20, 60]),
    "cli_error": (True, [2]),
    "tool_server": (True, [1]),
    "rate_limit": (False, []),  # do not hammer a subscription limit: wait for the reset or switch vendor
    "auth": (False, []),  # needs a human to log in: `claude` / `codex login`
    "refusal": (False, []),  # the same prompt will be refused again; record it as an answer
    "bad_request": (False, []),
}


MAX_ATTEMPTS = 4  # however the kinds of failure mix


def run_agent_with_retry(*args, **kwargs) -> AgentResult:
    """`run_agent`, repeated according to RETRY. The last result is returned.

    The loop itself was only dry-run with faked results, never against a
    live failure.
    """
    attempts = []
    res = run_agent(*args, **kwargs)
    attempts.append(res.error)
    while not res.ok and len(attempts) < MAX_ATTEMPTS:
        retry, waits = RETRY.get(res.error or "cli_error", (False, []))
        tried = attempts.count(res.error) - 1
        if not retry or tried >= len(waits):
            break
        time.sleep(waits[tried])
        res = run_agent(*args, **kwargs)
        attempts.append(res.error)
    res.meta["attempts"] = attempts
    return res
