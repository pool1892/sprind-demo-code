"""One agent call: `run_agent` drives `claude -p` or `codex exec` in the two sandboxes.

    from fleetlab.agents import run_agent, ToolAccess
    res = run_agent("claude", "opus", SYSTEM, PROMPT, images=[...], tools=access,
                    schema=Diagnosis)               # a pydantic model or a strict JSON schema
    res.ok, res.answer, res.result, res.usage, res.seconds, res.error, res.tool_log

The CLI runs in the CLI sandbox with the login file mounted alone and the
network on; the tool server (`fleetlab.toolserver`) runs in a sandbox of its
own inside it, with the export, the skills and the analysis' memory, and no
login and no network (`sandbox`). There is no unsandboxed mode: without a
sandbox a CLI needs the user's home or a link to the login file in a private
one, and such a link once destroyed the codex login on this machine.

The answer is checked twice: the CLI is given the strict JSON schema
(claude `--json-schema`, codex `--output-schema`), and the runtime validates
what comes back, with the pydantic model when one is given (claude does not
enforce a schema reliably: the spike saw prose after six turns).

Every call leaves a run directory with the command line, the CLI's event
stream, the tool server's call log (`box/toolbox/tool_calls.jsonl`, next to the
answer in `result.json`) and one line in the ledger (`ledger.jsonl` under the
run root's parent).
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from fleetlab.agents.audit import AuditContext
from fleetlab.agents.cli import Backend, agent_root, binary, pinned_version
from fleetlab.agents.commands import claude_argv, cli_env, codex_argv
from fleetlab.agents.sandbox import (
    BOX_CLI,
    BOX_HOME,
    BOX_RUN,
    TOOL_LOG_NAME,
    ToolMounts,
    cli_layout,
    tool_server_command,
)
from fleetlab.agents.toolenv import ToolEnv
from fleetlab.contracts.tools import json_schema_strict, strict_schema_problems
from fleetlab.toolserver.server import default_tool_names

__all__ = [
    "ERROR_KINDS",
    "MAX_ATTEMPTS",
    "RETRY",
    "AgentResult",
    "Auth",
    "PreparedCall",
    "ToolAccess",
    "answer_schema",
    "attempt_record",
    "box_dir",
    "call_log",
    "lost_runtime_answer",
    "prepare_call",
    "run_agent",
    "run_agent_with_retry",
]

Auth = Literal["subscription", "api_key"]

ERROR_KINDS = (
    "timeout",  # killed after `timeout` seconds
    "rate_limit",  # the subscription's usage limit or a 429
    "auth",  # login missing or expired: stop the batch and log in again
    "overloaded",  # vendor-side 5xx
    "refusal",  # the model or the vendor's filter declined
    "malformed_output",  # no answer, not JSON, or JSON that fails the schema
    "tool_server",  # our tool server did not start
    "bad_request",  # unknown model, schema the vendor rejects: our bug
    "cli_error",  # anything else
)

# The API-key route. The keys are read from these variables of the runtime's own
# environment at call time and passed to the CLI through its process
# environment; they are never written to a file or a command line. Named with
# a prefix so that a key someone has in their shell for another purpose never
# switches a run to paid billing by accident.
API_KEY_VARIABLES: dict[Backend, tuple[str, tuple[str, ...]]] = {
    "claude": ("ROBOTS_ANTHROPIC_API_KEY", ("ANTHROPIC_API_KEY",)),
    "codex": ("ROBOTS_OPENAI_API_KEY", ("CODEX_API_KEY", "OPENAI_API_KEY")),
}


@dataclass(frozen=True)
class ToolAccess:
    """What the tool server of a call serves: the export and one analysis' directories."""

    env: ToolEnv
    export: Path
    memory: Path
    outbox: Path
    results: Path
    # None: the tool server's default offer, every tool of the contracts but the opt-in
    # ones (contracts.tools.OPT_IN_TOOLS); the CLI is allowed exactly what is offered.
    tools: tuple[str, ...] | None = None
    session: int = 1
    max_experiment_episodes: int | None = None
    audit: AuditContext | None = None
    learned: Path | None = None  # a frozen library of learned skills, mounted read-only
    # Further budget limits for the tool server (toolserver.server.Limits); None keeps its
    # default. The round driver sets them per round (fleetlab.loop.driver).
    max_experiment_requests: int | None = None
    max_episodes_per_arm: int | None = None
    budget_first_session: int | None = None

    @property
    def enabled(self) -> tuple[str, ...]:
        return self.tools if self.tools is not None else default_tool_names()


@dataclass
class AgentResult:
    ok: bool
    result: Any  # the structured answer as JSON (or the text when no schema was given)
    usage: dict[str, Any]  # input_tokens (cached included), cached_input_tokens, output_tokens, ...
    seconds: float  # wall clock of the whole child process
    error: str | None = None  # one of ERROR_KINDS, or None
    error_detail: str = ""
    text: str = ""  # the raw final message
    answer: Any = None  # the pydantic model instance, when the schema was a model
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # as the CLI reported them
    tool_log: list[dict[str, Any]] = field(default_factory=list)  # as the tool server recorded them
    model: str = ""  # the model id claude reported; for codex the one asked for
    meta: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""

    def __iter__(self) -> Iterator[Any]:  # result, usage, seconds = run_agent(...)
        return iter((self.result, self.usage, self.seconds))


def lost_runtime_answer(result: AgentResult, *, with_tools: bool = False) -> bool:
    """Entry 11: surrounding software failed without delivering an answer.

    A tool-server startup loss counts only for a call given tools, before any
    server work was recorded. No-tool callers retain the baseline's definition.
    """
    if result.ok or answer_present(result, with_tools=with_tools):
        return False
    if result.error in {"timeout", "overloaded"}:
        return True
    if result.error == "tool_server":
        return with_tools and not result.tool_log
    # Other CLI errors (bad executable, process permissions, etc.) are not
    # identified transport failures.
    return result.error == "cli_error" and any(
        term in result.error_detail.lower()
        for term in (
            "transport error",
            "connection reset",
            "connection refused",
            "connection closed",
            "stream disconnected",
            "error sending request",
            "network is unreachable",
            "dns error",
            "tls handshake",
        )
    )


def answer_present(result: AgentResult, *, with_tools: bool = False) -> bool:
    """Tool sessions require a final message; generic/no-tool text keeps its meaning."""
    final_text = bool(result.text) and (
        not with_tools or result.meta.get("codex_final_message", True)
    )
    return result.result is not None or final_text or result.answer is not None


def attempt_record(result: AgentResult, *, with_tools: bool = False) -> dict[str, Any]:
    """The same per-start receipt for scored baselines and analysis sessions."""
    return {
        "ok": result.ok,
        "error": result.error,
        "detail": result.error_detail,
        "run_dir": result.run_dir,
        "seconds": result.seconds,
        "usage": result.usage,
        "runtime_meta": result.meta,
        "answer_present": answer_present(result, with_tools=with_tools),
    }


# Answer schemas.


def answer_schema(
    schema: type[BaseModel] | dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, type[BaseModel] | None]:
    """The strict JSON schema to give the CLI, and the model to validate with."""
    if schema is None:
        return None, None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return json_schema_strict(schema), schema
    if not isinstance(schema, dict):
        raise TypeError("schema is a pydantic model or a JSON schema")
    problems = strict_schema_problems(schema)
    if problems:
        raise ValueError("the schema is not strict: " + "; ".join(problems))
    return schema, None


def schema_errors(value: Any, schema: dict[str, Any], where: str = "$") -> list[str]:
    """Where a JSON value breaks a strict schema (the keywords strict mode uses)."""
    if "anyOf" in schema:
        options = [schema_errors(value, option, where) for option in schema["anyOf"]]
        return [] if any(not o for o in options) else [f"{where}: matches no option"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{where}: {value!r} is not one of {schema['enum']}"]
    kind = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if kinds and not any(checks[k](value) for k in kinds):
        return [f"{where}: expected {kind}, got {type(value).__name__}"]
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        return [f"{where}: {value!r} does not match {schema['pattern']}"]
    errors: list[str] = []
    if isinstance(value, dict) and "properties" in schema:
        missing = set(schema["properties"]) - set(value)
        extra = set(value) - set(schema["properties"])
        errors += [f"{where}.{k}: missing" for k in sorted(missing)]
        errors += [f"{where}.{k}: not allowed" for k in sorted(extra)]
        for key, sub in schema["properties"].items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{where}[{i}]")
    return errors


# Reading what the CLIs print.


def classify(text: str) -> str:
    """The kind of a failure from an error message."""
    low = text.lower()
    # OpenAI's filter answers instead of the model: "This content was flagged for possible ..."
    if any(
        s in low
        for s in (
            "was flagged for",
            "content policy",
            "violates our usage policies",
            "usage policy",
        )
    ):
        return "refusal"
    if "mcp server" in low and ("failed to initialize" in low or "handshaking" in low):
        return "tool_server"
    if any(
        s in low for s in ("rate limit", "usage limit", "429", "too many requests", "limit reached")
    ):
        return "rate_limit"
    if any(
        s in low
        for s in (
            "is not supported when using codex",
            "issue with the selected model",
            "invalid_json_schema",
            "invalid schema",
            "unknown configuration",
            "unknown variant",
            "unknown field",
            "invalid_request_error",
            "unsupported service_tier",
        )
    ):
        return "bad_request"
    if any(
        s in low
        for s in (
            "401",
            "unauthorized",
            "not logged in",
            "authentication",
            "oauth token",
            "invalid api key",
        )
    ):
        return "auth"
    if any(
        s in low
        for s in (
            "overloaded",
            "529",
            "503",
            "502",
            "500 internal",
            "server_error",
            "service unavailable",
        )
    ):
        return "overloaded"
    return "cli_error"


def window(info: dict[str, Any]) -> dict[str, Any]:
    """The subscription window a claude call reports: its status ("allowed",
    "allowed_warning" from about 94 %, "rejected"), which window, how full, when it resets."""
    windows = info.get("unifiedWindows") or {}
    return {
        "status": info.get("status"),
        "type": info.get("rateLimitType"),
        "utilization": {name: w.get("utilization") for name, w in windows.items()},
        "resets_at": info.get("resetsAt"),
    }


def parse_claude(lines: Sequence[str], res: AgentResult) -> None:
    pending: dict[str, dict[str, Any]] = {}
    final: dict[str, Any] | None = None
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
            info = ev.get("rate_limit_info") or {}
            res.meta["rate_limit"] = info
            res.meta["window"] = window(info)
        elif kind == "result":
            final = ev
    servers = res.meta.get("mcp_servers") or []
    if final is None:
        if any(s.get("status") == "failed" for s in servers):
            res.error, res.error_detail = "tool_server", json.dumps(servers)
        else:
            res.error, res.error_detail = "cli_error", "no result event"
        return
    usage = final.get("usage") or {}
    res.usage = {
        # Claude reports cached input apart; add it up so that both vendors mean the same.
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
        stop_reason=final.get("stop_reason"),
        subtype=final.get("subtype"),
        api_error_status=final.get("api_error_status"),
        permission_denials=final.get("permission_denials"),
    )
    res.text = final.get("result") or ""
    if final.get("structured_output") is not None:
        res.result = final["structured_output"]
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
            res.error = classify(detail)
        res.error_detail = detail[:500]
    elif final.get("stop_reason") == "refusal":
        res.error, res.error_detail = "refusal", res.text[:500]


def parse_codex(lines: Sequence[str], res: AgentResult, last_file: Path) -> None:
    failed: str | None = None
    errors: list[str] = []
    usage: dict[str, Any] | None = None
    final_message = False
    unlabelled_message = False
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
                detail = str((item.get("error") or {}).get("message", ""))
                if "transport closed" in detail.lower():
                    res.error, res.error_detail = "tool_server", detail[:500]
            elif item.get("type") == "agent_message":
                res.text = item.get("text") or res.text
                channel = item.get("channel") or item.get("phase")
                unlabelled_message = channel is None and bool(item.get("text"))
                final_message = channel in ("final", "final_answer") and bool(item.get("text"))
            elif item.get("type") == "error":
                errors.append(item.get("message", ""))
            elif item.get("type") not in ("reasoning",):
                # Anything else (command_execution, file_change, web_search ...) means a
                # built-in tool ran. Recorded, so that a check can assert it never does.
                res.meta.setdefault("builtin_items", []).append(
                    {
                        k: item.get(k)
                        for k in ("type", "command", "status", "tool", "name")
                        if k in item
                    }
                )
        elif kind == "turn.completed":
            usage = ev.get("usage") or {}
            final_message = final_message or unlabelled_message
        elif kind == "turn.failed":
            failed = (ev.get("error") or {}).get("message", "")
        elif kind == "error":
            errors.append(ev.get("message", ""))
    res.model = res.meta.get("requested_model", "")  # codex --json does not name the model
    if usage:
        res.usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "cached_input_tokens": usage.get("cached_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_output_tokens", 0),
        }
    if last_file.exists():
        last_text = last_file.read_text()
        res.text = last_text or res.text
        final_message = final_message or bool(last_text)
    # Keep generic callers' text behaviour. Scored sessions use this provenance
    # to distinguish an interim stream message from a delivered final answer.
    res.meta["codex_final_message"] = final_message
    if failed is not None or (usage is None and errors):
        detail = failed if failed is not None else " | ".join(errors)
        res.error, res.error_detail = classify(detail), detail[:500]
    if errors:
        res.meta["stream_errors"] = errors[:5]


def login_placeholder_problem(codex_home: Path) -> str | None:
    """Why the slot's `auth.json` is not the empty mount point bwrap made; None if it is.

    The login file is mounted over it inside the sandbox, so on the host it stays a
    0-byte file. Anything else would be a link or a copy of the login outside the
    login store, which must never exist."""
    path = codex_home / "auth.json"
    if path.is_symlink():
        return f"{path} is a link"
    if path.exists() and path.stat().st_size != 0:
        return f"{path} holds {path.stat().st_size} bytes: a copy of the login on disk"
    return None


def box_dir(run_dir: Path) -> Path:
    """The part of a run directory that is mounted into the CLI sandbox (/run/agent)."""
    return run_dir / "box"


def call_log(run_dir: Path) -> Path:
    """The tool server's call log of a run, next to the answer."""
    return box_dir(run_dir) / "toolbox" / TOOL_LOG_NAME


# Codex homes: one per concurrent call, so that parallel calls do not share
# codex's databases. Each is held through a lock file while a call runs.

CODEX_SLOTS = 16


@contextlib.contextmanager
def codex_home_slot(root: Path) -> Iterator[Path]:
    homes = root / "codex-homes"
    homes.mkdir(parents=True, exist_ok=True)
    while True:
        for k in range(CODEX_SLOTS):
            lock = (homes / f"{k:02d}.lock").open("w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            home = homes / f"{k:02d}"
            home.mkdir(exist_ok=True)
            try:
                yield home
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
            return
        time.sleep(0.2)


# The entry point.


def _api_keys(backend: Backend) -> dict[str, str]:
    source, targets = API_KEY_VARIABLES[backend]
    key = os.environ.get(source)
    if not key:
        raise RuntimeError(f"auth='api_key' needs {source} in the runtime's environment")
    return dict.fromkeys(targets, key)


@dataclass
class PreparedCall:
    """Everything a call runs with, built without running it (what the tests check)."""

    argv: list[str]  # bwrap (the CLI sandbox) -- the CLI and its flags
    env: dict[str, str]  # the CLI sandbox's whole environment (API keys only here)
    stdin: str
    run_dir: Path
    tool_command: list[str] | None  # bwrap (the tool sandbox) -- the tool server
    images: int


def prepare_call(
    backend: Backend,
    model: str,
    system: str,
    prompt: str,
    run_dir: Path,
    images: Sequence[str | Path] = (),
    tools: ToolAccess | None = None,
    schema_json: dict[str, Any] | None = None,
    *,
    effort: str | None = None,
    service_tier: str = "default",
    thinking_tokens: int | None = None,
    auth: Auth = "subscription",
    cli_version: str | None = None,
    codex_home: Path | None = None,
    extra_argv: Sequence[str] = (),
    env_overrides: dict[str, str] | None = None,
) -> PreparedCall:
    """The command line, environment and input of one call; writes the files the CLI reads."""
    if backend not in ("claude", "codex"):
        raise ValueError(f"unknown backend {backend!r}")
    # Only `box/` is mounted into the sandbox (at /run/agent): what the CLI reads and
    # writes. The command line, the prompt record, the CLI's output and the result stay
    # beside it, so that no host path sits inside the sandbox (review finding S6).
    box = box_dir(run_dir)
    (box / "cwd").mkdir(parents=True, exist_ok=True)
    (box / "toolbox").mkdir(exist_ok=True)
    call_log(run_dir).touch()
    version = cli_version or pinned_version(backend)
    cli_binary = binary(backend, version)
    box_binary = f"{BOX_CLI}/{cli_binary.name}"
    api_keys = _api_keys(backend) if auth == "api_key" else None

    mounts = None
    tool_command = None
    enabled: tuple[str, ...] = ()
    if tools is not None:
        mounts = ToolMounts(
            env=tools.env,
            export=tools.export,
            memory=tools.memory,
            outbox=tools.outbox,
            results=tools.results,
            log_dir=box / "toolbox",
            learned=tools.learned,
        )
        enabled = tools.enabled
        tool_command = tool_server_command(
            mounts,
            session=tools.session,
            enabled=list(tools.enabled) if tools.tools is not None else None,
            max_experiment_episodes=tools.max_experiment_episodes,
            max_experiment_requests=tools.max_experiment_requests,
            max_episodes_per_arm=tools.max_episodes_per_arm,
            budget_first_session=tools.budget_first_session,
        )

    def inside(path: Path) -> str:
        return f"{BOX_RUN}/{path.relative_to(box).as_posix()}"

    image_paths = [Path(p) for p in images]
    if backend == "claude":
        argv = claude_argv(box_binary, model, system, tool_command, enabled, schema_json, effort)
        content: list[dict[str, Any]] = []
        for image in image_paths:
            media = mimetypes.guess_type(image.name)[0] or "image/png"
            data = base64.b64encode(image.read_bytes()).decode()
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
            )
        content.append({"type": "text", "text": prompt})
        message = {"type": "user", "message": {"role": "user", "content": content}}
        stdin_text = json.dumps(message) + "\n"
    else:
        (box / "system.txt").write_text(system)
        schema_file = None
        if schema_json is not None:
            schema_file = box / "schema.json"
            schema_file.write_text(json.dumps(schema_json))
        image_files = []
        for i, image in enumerate(image_paths):
            copy = box / f"image-{i:02d}{image.suffix}"
            shutil.copyfile(image, copy)
            image_files.append(inside(copy))
        argv = codex_argv(
            box_binary,
            model,
            inside(box / "system.txt"),
            tool_command,
            enabled,
            inside(schema_file) if schema_file else None,
            effort,
            image_files,
            inside(box / "last_message.txt"),
            inside(box / "cwd"),
            service_tier,
        )
        stdin_text = prompt
    argv += list(extra_argv)
    env = cli_env(backend, BOX_HOME, thinking_tokens=thinking_tokens, api_keys=api_keys)
    env.update(env_overrides or {})
    layout = cli_layout(
        backend,
        cli_binary,
        box,
        mounts,
        subscription=auth == "subscription",
        codex_home=codex_home,
    )
    return PreparedCall(
        argv=[*layout.argv(), *argv],
        env=env,
        stdin=stdin_text,
        run_dir=run_dir,
        tool_command=tool_command,
        images=len(image_paths),
    )


def run_agent(
    backend: Backend,
    model: str,
    system: str,
    prompt: str,
    images: Sequence[str | Path] = (),
    tools: ToolAccess | None = None,
    schema: type[BaseModel] | dict[str, Any] | None = None,
    sandbox: Literal["bwrap"] = "bwrap",
    *,
    effort: str | None = None,
    service_tier: str = "default",
    thinking_tokens: int | None = None,
    auth: Auth = "subscription",
    timeout: float = 600.0,
    run_root: Path | None = None,
    cli_version: str | None = None,
    label: str = "",
    _extra_argv: Sequence[str] = (),
    _env: dict[str, str] | None = None,
    _plant: dict[str, str] | None = None,
    _record: bool = True,
    on_start: Callable[[Path], None] | None = None,
) -> AgentResult:
    """Run one agent call and return an `AgentResult` (it unpacks as result, usage, seconds).

    backend        "claude" or "codex"
    model          as the CLI names it: opus, fable, haiku; gpt-6.1-sol, gpt-6-astra, gpt-6-luna
    images         PNG or JPEG files shown with the prompt
    tools          the tool server's export and analysis directories; None for no tools
    schema         a pydantic model (validated with it) or a strict JSON schema; None: free text
    effort         reasoning effort ("low" ... "max"); None keeps the model's default
    service_tier   codex only, always passed: "default" or "fast"
    thinking_tokens  claude only: MAX_THINKING_TOKENS (0 switches Haiku's default thinking off)
    auth           "subscription" (the login file, mounted alone) or "api_key" (from
                   ROBOTS_ANTHROPIC_API_KEY / ROBOTS_OPENAI_API_KEY; no login file mounted)
    """
    if sandbox != "bwrap":
        raise ValueError("calls run in the bwrap sandbox only (see the module docstring)")
    if tools is not None and (_extra_argv or _env):
        raise ValueError("analysis launches do not accept CLI or environment overrides")
    schema_json, schema_model = answer_schema(schema)
    root = agent_root()
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = (run_root or root / "runs") / run_id
    run_dir.mkdir(parents=True)
    version = cli_version or pinned_version(backend)
    res = AgentResult(ok=False, result=None, usage={}, seconds=0.0, run_dir=str(run_dir))
    res.meta.update(
        backend=backend,
        requested_model=model,
        cli_version=version,
        effort=effort,
        service_tier=service_tier if backend == "codex" else None,
        thinking_tokens=thinking_tokens,
        auth=auth,
        label=label,
        tool_env=tools.env.digest() if tools is not None else None,
    )
    (run_dir / "prompt.json").write_text(
        json.dumps({"system": system, "prompt": prompt, "images": [str(p) for p in images]})
    )
    slot: contextlib.AbstractContextManager[Path | None]
    slot = codex_home_slot(root) if backend == "codex" else contextlib.nullcontext()
    with slot as codex_home:
        call = prepare_call(
            backend,
            model,
            system,
            prompt,
            run_dir,
            images,
            tools,
            schema_json,
            effort=effort,
            service_tier=service_tier,
            thinking_tokens=thinking_tokens,
            auth=auth,
            cli_version=version,
            codex_home=codex_home,
            extra_argv=_extra_argv,
            env_overrides=_env,
        )
        for name, text in (_plant or {}).items():
            (box_dir(run_dir) / "cwd" / name).write_text(text)
        (run_dir / "argv.json").write_text(json.dumps(call.argv, indent=1))
        if tools is not None:
            from fleetlab.agents.audit import launch_audit
            from fleetlab.agents.checks import GateError, codex_home_rows, require_verified

            if version != pinned_version(backend):
                raise GateError("agent launch requires the verified pinned CLI version")
            require_verified(backend, tools.env)

            if codex_home is not None:
                problems = [
                    target
                    for target, observed, expected in codex_home_rows(codex_home)
                    if observed != expected
                ]
                if problems:
                    raise GateError("Codex home isolation: " + "; ".join(problems))

            audit_started = time.perf_counter()
            try:
                receipt = launch_audit(
                    tools,
                    box_dir(run_dir),
                    {"system": system, "prompt": prompt, "schema": json.dumps(schema_json)},
                )
            finally:
                (run_dir / "audit-time.json").write_text(
                    json.dumps(
                        {
                            "seconds": time.perf_counter() - audit_started,
                        }
                    )
                )
            (run_dir / "audit.json").write_text(json.dumps(receipt, indent=1))
        _execute(
            res,
            call.argv,
            call.env,
            run_dir,
            call.stdin,
            timeout,
            backend,
            tools is not None,
            on_start=on_start,
        )
        placeholder = login_placeholder_problem(codex_home) if codex_home is not None else None
    lines = (run_dir / "stdout.jsonl").read_text().splitlines()
    if backend == "claude":
        parse_claude(lines, res)
    else:
        parse_codex(lines, res, box_dir(run_dir) / "last_message.txt")
    tool_log = call_log(run_dir)
    res.tool_log = [json.loads(line) for line in tool_log.read_text().splitlines() if line.strip()]
    if placeholder is not None:
        # A login copy on disk outside the login store: stop the batch, report it,
        # touch nothing.
        res.error, res.error_detail = "auth", placeholder
    elif res.meta.get("timed_out"):
        res.error, res.error_detail = "timeout", f"killed after {timeout}s"
    elif res.error is None and res.meta["exit_code"] != 0:
        err = (run_dir / "stderr.txt").read_text()
        res.error, res.error_detail = classify(err), err[-500:]
    if res.error is None:
        _check_answer(res, schema_json, schema_model)
    res.ok = res.error is None
    (run_dir / "result.json").write_text(json.dumps(_summary(res), indent=1, default=str))
    _ledger(root, run_id, res, call.images, _record)
    return res


def _execute(
    res: AgentResult,
    argv: list[str],
    env: dict[str, str],
    run_dir: Path,
    stdin_text: str,
    timeout: float,
    backend: str,
    with_tools: bool,
    *,
    on_start: Callable[[Path], None] | None = None,
) -> None:
    started = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=box_dir(run_dir) / "cwd",
        text=True,
        start_new_session=True,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    stamped: list[tuple[float, str]] = []
    err_parts: list[str] = []
    stdout, stderr = proc.stdout, proc.stderr

    def kill() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)

    def read_out() -> None:
        for line in stdout:
            stamped.append((time.perf_counter() - started, line))
            # claude carries on without tools when the tool server fails to start,
            # and the model then guesses. Stop it at once instead.
            if backend == "claude" and with_tools and '"subtype":"init"' in line:
                try:
                    servers = json.loads(line).get("mcp_servers") or []
                except ValueError:
                    servers = []
                if any(s.get("status") == "failed" for s in servers):
                    kill()

    readers = [
        threading.Thread(target=read_out),
        threading.Thread(target=lambda: err_parts.append(stderr.read())),
    ]
    for reader in readers:
        reader.start()
    # Gate and audit have passed and the model process exists. Reserve the
    # run's ordinal before sending its task; failed starts consume no ordinal.
    try:
        if on_start is not None:
            on_start(run_dir)
        (run_dir / "started.json").write_text(json.dumps({"started_at": time.time()}))
    except BaseException:
        kill()
        proc.wait()
        for reader in readers:
            reader.join(timeout=10)
        raise
    with contextlib.suppress(BrokenPipeError):
        proc.stdin.write(stdin_text)
        proc.stdin.close()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        res.meta["timed_out"] = True
        kill()  # the CLI, its sandbox and the tool server with it
        proc.wait()
    for reader in readers:
        reader.join(timeout=10)
    res.seconds = round(time.perf_counter() - started, 3)
    (run_dir / "stdout.jsonl").write_text("".join(line for _, line in stamped))
    (run_dir / "stderr.txt").write_text("".join(err_parts))
    res.meta["exit_code"] = proc.returncode

    def first(marker: str) -> float | None:
        return next((round(t, 3) for t, line in stamped if marker in line), None)

    def last(marker: str) -> float | None:
        return next((round(t, 3) for t, line in reversed(stamped) if marker in line), None)

    # Until the CLI was ready to call the model, until the answer was complete.
    if backend == "claude":
        res.meta["phases"] = {
            "ready_s": first('"subtype":"init"'),
            "answer_s": last('"type":"result"'),
        }
    else:
        res.meta["phases"] = {
            "ready_s": first('"turn.started"'),
            "answer_s": last('"turn.completed"'),
        }


def _check_answer(
    res: AgentResult, schema: dict[str, Any] | None, model: type[BaseModel] | None
) -> None:
    if schema is None:
        res.result = res.text
        return
    if res.result is None:  # codex, and claude without structured_output: parse the text
        try:
            res.result = json.loads(res.text)
        except ValueError:
            res.error, res.error_detail = "malformed_output", f"not JSON: {res.text[:300]!r}"
            return
    errors = schema_errors(res.result, schema)
    if errors:
        res.error, res.error_detail = "malformed_output", "; ".join(errors)[:500]
        return
    if model is not None:
        try:
            res.answer = model.model_validate(res.result)
        except ValidationError as exc:
            res.error, res.error_detail = "malformed_output", str(exc)[:500]


def _summary(res: AgentResult) -> dict[str, Any]:
    data = asdict(res)
    data["answer"] = None if res.answer is None else res.answer.model_dump(mode="json")
    return data


_ledger_lock = threading.Lock()


def _ledger(root: Path, run_id: str, res: AgentResult, images: int, record: bool) -> None:
    line = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": run_id,
        "run_dir": res.run_dir,
        "backend": res.meta.get("backend"),
        "model": res.meta.get("requested_model"),
        "model_reported": res.model,
        "cli_version": res.meta.get("cli_version"),
        "effort": res.meta.get("effort"),
        "service_tier": res.meta.get("service_tier"),
        "auth": res.meta.get("auth"),
        "label": res.meta.get("label"),
        "seconds": res.seconds,
        "answer_s": (res.meta.get("phases") or {}).get("answer_s"),
        "api_ms": res.meta.get("api_ms"),
        "ok": res.ok,
        "error": res.error,
        "usage": res.usage,
        "tool_calls": len(res.tool_calls),
        "tool_server_calls": len(res.tool_log),
        "images": images,
        "window": res.meta.get("window"),
    }
    with _ledger_lock, (root / ("ledger.jsonl" if record else "dry_runs.jsonl")).open("a") as fh:
        fh.write(json.dumps(line) + "\n")


RETRY: dict[str, tuple[bool, list[float]]] = {
    # Generic callers only; scored ex-post sessions use lost_runtime_answer.
    # error kind: (retry?, wait in seconds before each further attempt)
    "timeout": (True, [0]),
    "malformed_output": (True, [0]),  # a fresh sample usually parses
    "overloaded": (True, [5, 20, 60]),
    "cli_error": (True, [2]),
    "tool_server": (True, [1]),
    "rate_limit": (False, []),  # wait for the window's reset or switch vendor; never hammer
    "auth": (False, []),  # needs a human: stop the batch
    "refusal": (False, []),  # the same prompt is refused again; it is the answer
    "bad_request": (False, []),  # our bug
}
MAX_ATTEMPTS = 4


def run_agent_with_retry(*args: Any, **kwargs: Any) -> AgentResult:
    """`run_agent`, repeated by the kind of failure (RETRY), at most MAX_ATTEMPTS times.

    The last result is returned, with every attempt's error and run directory in
    `meta["attempts"]`.
    """
    attempts: list[dict[str, Any]] = []
    res = run_agent(*args, **kwargs)
    attempts.append({"error": res.error, "run_dir": res.run_dir, "seconds": res.seconds})
    while not res.ok and len(attempts) < MAX_ATTEMPTS:
        retry, waits = RETRY.get(res.error or "cli_error", (False, []))
        tried = sum(1 for a in attempts if a["error"] == res.error) - 1
        if not retry or tried >= len(waits):
            break
        time.sleep(waits[tried])
        res = run_agent(*args, **kwargs)
        attempts.append({"error": res.error, "run_dir": res.run_dir, "seconds": res.seconds})
    res.meta["attempts"] = attempts
    return res
