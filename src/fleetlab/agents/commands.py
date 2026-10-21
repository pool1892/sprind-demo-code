"""The command lines and environments of `claude -p` and `codex exec`.

The flag sets of the agent-runtime spike ("The exact
invocations"), each part measured there with the wire and isolation checks:
no built-in tools, only our tool server, nothing of the user's or the
project's setup, our system prompt in place of the CLI's, the answer forced
into a JSON schema. `checks.wire_check` re-verifies them for the pinned
versions; `--strict-config` makes codex refuse a setting it does not know,
so a renamed setting is a loud error instead of a silently re-opened door.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

__all__ = [
    "CLAUDE_ENV",
    "CODEX_FEATURES_OFF",
    "SERVICE_TIERS",
    "claude_argv",
    "cli_env",
    "codex_argv",
]

SERVER = "robots"  # the tool server's name: tools appear as mcp__robots__<tool>

# Switches that cannot be given as flags. `--setting-sources ""` stops
# CLAUDE.md, hooks, skills and user MCP servers, but not the auto-memory index
# of the working directory; this variable does (spike, isolation_check.py).
CLAUDE_ENV = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "DISABLE_AUTOUPDATER": "1",
}

CODEX_FEATURES_OFF = (
    "shell_tool",
    "unified_exec",  # no shell
    "apps",
    "plugins",
    "multi_agent",
    "goals",
    "sleep_tool",
    "view_image",
    "image_generation",
    "browser_use",
    "computer_use",
    "tool_suggest",
    "skill_search",
    "hooks",
    # Off by default in 0.161.0, switched off explicitly: the slot homes persist between
    # calls and worlds, and nothing of one call may be remembered in the next (review
    # finding S2). No shell, so no shell snapshots either.
    "memories",
    "shell_snapshot",
)

# Codex's service tiers ("fast" is the priority tier; the user's own config
# holds "default"). Passed on every codex call, because the private CODEX_HOME
# inherits nothing. Codex's config also accepts "flex", but the service refuses
# it on the subscription ("Unsupported service_tier: flex", 2026-10-07).
SERVICE_TIERS = ("default", "fast")

# A tool call may take this long before codex gives up on it (a query is
# interrupted by the server after 60 s).
CODEX_TOOL_TIMEOUT_S = 120


def cli_env(
    backend: str,
    home: str,
    *,
    thinking_tokens: int | None = None,
    api_keys: dict[str, str] | None = None,
) -> dict[str, str]:
    """The whole environment of the CLI, built from nothing.

    No variable of a surrounding Claude Code session (CLAUDECODE,
    CLAUDE_CODE_SESSION_ID, AI_AGENT ...) and nothing of the user's shell
    reaches the child. API keys, when given, are passed here and only here:
    through the process environment, never on a command line or in a file.
    """
    env = {"HOME": home, "PATH": "/usr/bin", "LANG": "C.UTF-8"}
    if backend == "claude":
        env.update(CLAUDE_ENV)
        if thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(thinking_tokens)
    else:
        env["CODEX_HOME"] = f"{home}/.codex"
    env.update(api_keys or {})
    return env


def claude_argv(
    binary: str,
    model: str,
    system: str,
    tool_command: Sequence[str] | None,
    tools: Sequence[str],
    schema: dict[str, Any] | None,
    effort: str | None,
) -> list[str]:
    argv = [
        binary,
        "-p",
        "--model", model,
        "--system-prompt", system,  # replaces Claude Code's own system prompt
        "--tools", "",  # no built-in tools: no Bash, Read, Write, WebFetch, Agent
        "--strict-mcp-config",  # only the MCP server named below, none of the user's
        "--setting-sources", "",  # no user, project or local settings: no CLAUDE.md, hooks, plugins
        "--disable-slash-commands",  # no skills of the CLI's own
        "--no-session-persistence",  # no transcript
        "--permission-mode", "dontAsk",  # what is not allowed below is denied, never asked
        "--input-format", "stream-json",  # the prompt arrives on stdin and can carry images
        "--output-format", "stream-json",
        "--verbose",  # required by stream-json output
    ]  # fmt: skip
    if tool_command is not None:
        command = list(tool_command)
        config = {"mcpServers": {SERVER: {"command": command[0], "args": command[1:]}}}
        argv += ["--mcp-config", json.dumps(config)]
        argv += ["--allowedTools", ",".join(f"mcp__{SERVER}__{t}" for t in tools)]
    if schema is not None:
        argv += ["--json-schema", json.dumps(schema)]
    if effort:
        argv += ["--effort", effort]
    return argv


def _toml(value: Any) -> str:
    return json.dumps(value)  # JSON strings, numbers, booleans and lists are valid TOML


def codex_argv(
    binary: str,
    model: str,
    system_file: str,
    tool_command: Sequence[str] | None,
    tools: Sequence[str],
    schema_file: str | None,
    effort: str | None,
    image_files: Sequence[str],
    last_file: str,
    cwd: str,
    service_tier: str,
) -> list[str]:
    if service_tier not in SERVICE_TIERS:
        raise ValueError(f"service_tier is one of {SERVICE_TIERS}, not {service_tier!r}")

    def c(key: str, value: Any) -> list[str]:
        return ["-c", f"{key}={_toml(value)}"]

    argv = [
        binary, "exec",
        "--ignore-user-config",  # belt and braces: CODEX_HOME is private too
        "--ignore-rules",
        "--ephemeral",  # no session file
        "--skip-git-repo-check",
        "--strict-config",  # an unknown setting is an error, not a silent no-op
        "-s", "read-only",
        "-C", cwd,
        "-m", model,
        "--json",
        "-o", last_file,
    ]  # fmt: skip
    for image in image_files:
        argv += ["-i", image]
    argv += c("model_instructions_file", system_file)  # replaces Codex's own instructions
    argv += c("project_doc_max_bytes", 0)  # no AGENTS.md from the working directory
    argv += c("web_search", "disabled")
    argv += c("agents.enabled", False)
    argv += c("skills.include_instructions", False)
    argv += c("tools.experimental_request_user_input.enabled", False)
    argv += c("include_environment_context", False)  # no working directory, shell, date
    argv += c("include_permissions_instructions", False)
    argv += c("include_apps_instructions", False)
    argv += c("include_collaboration_mode_instructions", False)
    argv += c("service_tier", service_tier)
    # Codex's models work in "code mode": a built-in `exec` tool runs JavaScript in
    # a bare V8 isolate and reaches other tools through it. Keep the built-in
    # namespaces out of it and offer our tools as ordinary direct calls.
    argv += c("features.code_mode.excluded_tool_namespaces", ["functions", "clock"])
    for feature in CODEX_FEATURES_OFF:
        argv += ["--disable", feature]
    if tool_command is not None:
        command = list(tool_command)
        ns = f"mcp_servers.{SERVER}"
        argv += c(f"{ns}.command", command[0])
        argv += c(f"{ns}.args", command[1:])
        argv += c(f"{ns}.required", True)  # fail the call if the tool server does not start
        argv += c(f"{ns}.enabled_tools", list(tools))
        argv += c(f"{ns}.default_tools_approval_mode", "approve")  # nobody is there to click
        argv += c(f"{ns}.tool_timeout_sec", CODEX_TOOL_TIMEOUT_S)
        argv += c("features.code_mode.direct_only_tool_namespaces", [f"mcp__{SERVER}"])
    if schema_file is not None:
        argv += ["--output-schema", schema_file]
    if effort:
        argv += c("model_reasoning_effort", effort)
    return argv
