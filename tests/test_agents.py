"""The agent runtime without a model: command lines and sandboxes checked for forbidden mounts,
the stream parsers and the retry rules, answers checked against their schemas, the sessions
protocol with its queue for the engine, memory per world, and the gate.

src/fleetlab/agents/; docs/design/contracts.md, sections 12 and 13.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest
from log_toy import TASK_CLASSES, run_jobs, write_toy_run

from fleetlab.agents import run as runmod
from fleetlab.agents.batch import Batch
from fleetlab.agents.checks import GateError, repository_roots, require_verified
from fleetlab.agents.cli import BACKENDS, Backend, binary, pinned_version
from fleetlab.agents.commands import CLAUDE_ENV, SERVICE_TIERS, claude_argv, codex_argv
from fleetlab.agents.run import (
    AgentResult,
    ToolAccess,
    answer_schema,
    classify,
    parse_claude,
    parse_codex,
    prepare_call,
    run_agent,
    run_agent_with_retry,
    schema_errors,
)
from fleetlab.agents.sandbox import (
    BOX_EXPORT,
    BOX_HOME,
    BOX_LOGIN,
    BOX_RUN,
    LOGIN_FILES,
    Mount,
    ToolMounts,
    cli_layout,
    dangerous_links,
    layout_problems,
    parse_bwrap,
    tool_layout,
)
from fleetlab.agents.sessions import (
    AgentSpec,
    Analysis,
    EngineReply,
    ExperimentQueue,
    SessionReport,
    run_analysis,
)
from fleetlab.agents.toolenv import ToolEnv, launcher_argv, project_tool_env
from fleetlab.agents.toy import write_toy_skill
from fleetlab.contracts.tools import ArmResult, Diagnosis, ExperimentResult
from fleetlab.log import export_runs
from fleetlab.toolserver.server import Paths, ToolBox, tool_names

HOME = Path.home()


@dataclass(frozen=True)
class World:
    run: Path
    export: Path
    env: ToolEnv
    after: Path  # an export in which e000002 and e000004 are the arms of x0001


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    base = tmp_path_factory.mktemp("agents")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    write_toy_skill(base / "skills")
    arms = []
    for job in run_jobs():
        if job.identity.episode_id in ("e000002", "e000004"):
            marked = job.identity.model_copy(update={"experiment_id": "x0001"})
            job = job.model_copy(update={"identity": marked})
        arms.append(job)
    write_toy_run(base / "run-after", arms)
    export_runs([base / "run-after"], base / "after", fleet=fleet, task_classes=TASK_CLASSES)
    env = project_tool_env(base / "env", base / "skills")
    return World(base / "run", base / "export", env, base / "after")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private agent root, so that no test touches the real one."""
    path = tmp_path / "agent-root"
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(path))
    monkeypatch.delenv("ROBOTS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROBOTS_OPENAI_API_KEY", raising=False)
    yield path


def access(world: World, analysis: Analysis, session: int = 1) -> ToolAccess:
    return analysis.tool_access(placed_env(world), world.export, session)


def placed_env(world: World) -> ToolEnv:
    """The tool environment as the runtime places it: every directory outside home and repository
    (the interpreter and packages here are stand-ins: nothing is run)."""
    base = world.env.src.parent
    for name in ("python", "site"):
        (base / name).mkdir(exist_ok=True)
    interpreter = base / "python" / "bin" / f"python{world.env.python_version}"
    interpreter.parent.mkdir(exist_ok=True)
    if not interpreter.exists():
        interpreter.write_bytes(b"")
    return ToolEnv(
        python=base / "python", site=base / "site", src=world.env.src, skills=world.env.skills
    )


def test_the_tool_environment_digest_follows_the_interpreter_and_the_content(
    world: World, tmp_path: Path
) -> None:
    env = placed_env(world)
    first = env.digest()
    assert placed_env(world).digest() == first
    interpreter = env.python / "bin" / f"python{env.python_version}"
    interpreter.write_bytes(b"another interpreter")
    assert env.digest() != first  # the binary, not the directory's name
    interpreter.write_bytes(b"")
    assert env.digest() == first
    other = ToolEnv(python=env.python, site=env.site, src=tmp_path / "src-other", skills=env.skills)
    assert other.digest() != first


# The command lines.


def test_claude_command_line_is_the_spike_flag_set() -> None:
    argv = claude_argv(
        "/opt/cli/claude", "opus", "SYS", ["bwrap", "x"], ("query",), {"type": "object"}, "low"
    )
    for flag in (
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--verbose",
    ):
        assert flag in argv
    pairs = dict(itertools.pairwise(argv))
    assert pairs["--tools"] == ""
    assert pairs["--setting-sources"] == ""
    assert pairs["--permission-mode"] == "dontAsk"
    assert pairs["--allowedTools"] == "mcp__robots__query"
    assert pairs["--effort"] == "low"
    config = json.loads(pairs["--mcp-config"])
    assert config == {"mcpServers": {"robots": {"command": "bwrap", "args": ["x"]}}}
    assert CLAUDE_ENV["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


def test_codex_command_line_carries_the_service_tier_and_no_shell() -> None:
    argv = codex_argv(
        "/opt/cli/codex", "gpt-6-luna", "/run/agent/system.txt", ["bwrap", "y"], ("query",),
        "/run/agent/schema.json", None, ["/run/agent/image-00.png"], "/run/agent/last.txt",
        "/run/agent/cwd", "fast",
    )  # fmt: skip
    joined = " ".join(argv)
    for part in (
        "--ignore-user-config",
        "--strict-config",
        "--ephemeral",
        '-c service_tier="fast"',
        "--disable shell_tool",
        "--disable unified_exec",
        "-c include_environment_context=false",
        "-c project_doc_max_bytes=0",
        '-c mcp_servers.robots.enabled_tools=["query"]',
        '-c features.code_mode.direct_only_tool_namespaces=["mcp__robots"]',
    ):
        assert part in joined, part
    with pytest.raises(ValueError, match="service_tier"):
        codex_argv("c", "m", "s", None, (), None, None, [], "l", "d", "turbo")
    assert set(SERVICE_TIERS) == {"default", "fast"}


EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def _inner(call_argv: list[str], backend: Backend) -> list[str]:
    """The tool server's command line, as the CLI is told to start it."""
    if backend == "claude":
        config = json.loads(call_argv[call_argv.index("--mcp-config") + 1])
        server = config["mcpServers"]["robots"]
        return [server["command"], *server["args"]]
    values = {}
    for i, arg in enumerate(call_argv):
        if arg == "-c" and call_argv[i + 1].startswith("mcp_servers.robots."):
            key, value = call_argv[i + 1].split("=", 1)
            values[key] = json.loads(value)
    return [values["mcp_servers.robots.command"], *values["mcp_servers.robots.args"]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_constructed_call_keeps_the_sandbox_rules(
    world: World, root: Path, backend: Backend
) -> None:
    analysis = Analysis.create("w01", base=root / "analyses")
    run_dir = root / "runs" / "one"
    run_dir.mkdir(parents=True)
    codex_home = root / "codex-homes" / "00"
    codex_home.mkdir(parents=True)
    call = prepare_call(
        backend, "model", "SYSTEM", "PROMPT", run_dir, tools=access(world, analysis),
        schema_json=EMPTY_SCHEMA, codex_home=codex_home if backend == "codex" else None,
    )  # fmt: skip
    forbidden = [*repository_roots(), world.run / "truth"]
    outer, cli = parse_bwrap(call.argv)
    problems = layout_problems(
        call.argv, "cli", backend=backend, forbidden_roots=forbidden,
        allowed_home=[binary(backend).parent], env=call.env,
    )  # fmt: skip
    assert problems == []
    inner = _inner(cli, backend)
    assert inner[:4] == ["/usr/bin/env", "-i", "PATH=/usr/bin", "LANG=C.UTF-8"]  # PID 1 clean
    assert inner[4] == "bwrap"
    inner_problems = layout_problems(inner, "tool", backend=backend, forbidden_roots=forbidden)
    assert inner_problems == []
    inner_layout, server = parse_bwrap(inner)
    assert server[:3] == launcher_argv(world.env.python_version)[:3]
    assert not inner_layout.network and inner_layout.clearenv
    # The login file: alone, read-write, in the CLI sandbox only.
    login = [m for m in outer.mounts if m.dst == BOX_LOGIN[backend]]
    assert login == [Mount("bind", str(LOGIN_FILES[backend]), BOX_LOGIN[backend])]
    assert not [
        m
        for m in inner_layout.mounts
        if "login" in m.dst or ".claude" in m.dst or ".codex" in m.dst
    ]
    assert outer.network
    assert outer.die_with_parent
    assert not inner_layout.die_with_parent
    # Only the run directory's box/ is mounted; the command line and the CLI's output
    # stay outside, so no host path is inside the sandbox (review finding S6).
    run_mounts = [m for m in outer.mounts if m.dst == BOX_RUN]
    assert run_mounts == [Mount("bind", str(run_dir / "box"), BOX_RUN)]
    assert (run_dir / "box" / "cwd").is_dir() and (run_dir / "box" / "toolbox").is_dir()
    assert call.env["HOME"] == BOX_HOME
    assert "CLAUDECODE" not in call.env
    for m in outer.mounts:
        assert (
            m.src is None
            or not m.src.startswith(str(HOME / ".claude/"))
            or m.src == str(LOGIN_FILES["claude"])
        )


def test_the_api_key_route_mounts_no_login_and_keeps_the_key_off_the_command_line(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = root / "runs" / "api"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="ROBOTS_ANTHROPIC_API_KEY"):
        prepare_call("claude", "opus", "S", "P", run_dir, auth="api_key")
    monkeypatch.setenv("ROBOTS_ANTHROPIC_API_KEY", "sk-test-not-a-key")
    call = prepare_call("claude", "opus", "S", "P", run_dir, auth="api_key")
    assert call.env["ANTHROPIC_API_KEY"] == "sk-test-not-a-key"
    assert "sk-test-not-a-key" not in json.dumps(call.argv)
    layout, _ = parse_bwrap(call.argv)
    assert not [m for m in layout.mounts if m.dst == BOX_LOGIN["claude"]]


@dataclass
class Bad:
    name: str
    argv: list[str]
    role: str


def _bad_layouts(world: World, root: Path) -> list[Bad]:
    mounts = ToolMounts(
        placed_env(world), world.export, root / "m", root / "o", root / "r", root / "l"
    )
    for d in ("m", "o", "r", "l"):
        (root / d).mkdir(parents=True, exist_ok=True)
    good = tool_layout(mounts, nested=False)
    networked = tool_layout(mounts, nested=False)
    networked.network = True
    inherited = tool_layout(mounts, nested=False)
    inherited.clearenv = False
    with_login = tool_layout(mounts, nested=False)
    with_login.mounts.append(Mount("ro-bind", str(LOGIN_FILES["codex"]), "/work/x"))
    store = cli_layout("claude", binary("claude"), root / "run", None, subscription=True)
    store.mounts.append(Mount("ro-bind", str(HOME / ".claude"), "/home/agent/.claude"))
    repo = cli_layout("claude", binary("claude"), root / "run", None, subscription=True)
    repo.mounts.append(Mount("ro-bind", str(repository_roots()[0]), "/opt/repo"))
    home = cli_layout("claude", binary("claude"), root / "run", None, subscription=True)
    home.mounts.append(Mount("ro-bind", str(HOME), "/opt/home"))
    secret = tool_layout(mounts, nested=False)
    secret.setenv["OPENAI_API_KEY"] = "x"
    writable = tool_layout(mounts, nested=False)
    writable.mounts.append(Mount("bind", str(world.export), BOX_EXPORT))
    pid1 = tool_layout(mounts, nested=False)
    pid1.parent_env = None
    return [
        Bad("network in the tool sandbox", networked.argv(), "tool"),
        Bad("inherited environment", inherited.argv(), "tool"),
        Bad("login file in the tool sandbox", with_login.argv(), "tool"),
        Bad("login store directory", store.argv(), "cli"),
        Bad("the repository", repo.argv(), "cli"),
        Bad("the home directory", home.argv(), "cli"),
        Bad("a key on the command line", secret.argv(), "tool"),
        Bad("a writable export", writable.argv(), "tool"),
        Bad("bwrap keeps the CLI's environment", pid1.argv(), "tool"),
        Bad("good", good.argv(), "tool"),
    ]


def test_forbidden_mounts_are_found(world: World, root: Path) -> None:
    (root / "run").mkdir(parents=True)
    for bad in _bad_layouts(world, root):
        problems = layout_problems(
            bad.argv, bad.role,  # type: ignore[arg-type]
            backend="claude", forbidden_roots=repository_roots(),
            allowed_home=[binary("claude").parent],
        )  # fmt: skip
        if bad.name == "good":
            assert problems == [], problems
        else:
            assert problems, bad.name


def test_a_link_toward_a_login_in_a_writable_mount_is_found(root: Path) -> None:
    # The link points at the sandbox's login path, which does not exist on the host;
    # no link into the real login store is ever made, not even in a test.
    directory = root / "codex-home"
    (directory / "tmp").mkdir(parents=True)
    (directory / "tmp" / "auth.json").symlink_to(BOX_LOGIN["codex"])
    (directory / "tmp" / "apply_patch").symlink_to("/opt/cli/codex")  # codex's own: harmless
    assert dangerous_links(directory) == [directory / "tmp" / "auth.json"]
    layout = cli_layout(
        "codex", binary("codex"), root, None, subscription=True, codex_home=directory
    )
    problems = layout_problems(
        layout.argv(), "cli", backend="codex", allowed_home=[binary("codex").parent]
    )
    assert any("toward a login" in p for p in problems)


# What the CLIs print.


def _lines(*events: dict[str, Any]) -> list[str]:
    return [json.dumps(e) for e in events]


def _result() -> AgentResult:
    return AgentResult(ok=False, result=None, usage={}, seconds=0.0)


TOOL_USE = {
    "type": "tool_use",
    "id": "t1",
    "name": "mcp__robots__query",
    "input": {"sql": "SELECT 1"},
}
TOOL_RESULT = {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text"}]}


def test_claude_stream_gives_answer_tokens_and_tool_calls() -> None:
    res = _result()
    parse_claude(
        _lines(
            {
                "type": "system",
                "subtype": "init",
                "model": "claude-opus-5-5",
                "tools": ["StructuredOutput"],
                "mcp_servers": [{"name": "robots", "status": "connected"}],
                "claude_code_version": "2.1.293",
            },
            {"type": "assistant", "message": {"content": [TOOL_USE]}},
            {"type": "user", "message": {"content": [TOOL_RESULT]}},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "",
                "structured_output": {"a": 1},
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5},
                "total_cost_usd": 0.01,
                "duration_api_ms": 500,
                "num_turns": 2,
            },
        ),
        res,
    )  # fmt: skip
    assert res.error is None
    assert res.result == {"a": 1}
    assert res.usage["input_tokens"] == 100
    assert res.usage["cached_input_tokens"] == 90
    assert res.tool_calls == [
        {
            "name": "mcp__robots__query",
            "arguments": {"sql": "SELECT 1"},
            "result_parts": ["text"],
            "is_error": False,
        }
    ]
    assert res.meta["api_ms"] == 500


@pytest.mark.parametrize(
    ("final", "kind"),
    [
        ({"is_error": True, "api_error_status": 429}, "rate_limit"),
        ({"is_error": True, "api_error_status": 401}, "auth"),
        ({"is_error": True, "api_error_status": 529}, "overloaded"),
        ({"is_error": True, "api_error_status": 404, "result": "model not found"}, "bad_request"),
        ({"is_error": False, "stop_reason": "refusal"}, "refusal"),
    ],
)
def test_claude_failures_by_kind(final: dict[str, Any], kind: str) -> None:
    res = _result()
    parse_claude(_lines({"type": "result", "subtype": "error", **final}), res)
    assert res.error == kind


def test_claude_without_its_tool_server_is_a_tool_server_failure() -> None:
    res = _result()
    parse_claude(
        _lines(
            {
                "type": "system",
                "subtype": "init",
                "mcp_servers": [{"name": "robots", "status": "failed"}],
            }
        ),
        res,
    )
    assert res.error == "tool_server"


MCP_CALL = {
    "type": "mcp_tool_call",
    "server": "robots",
    "tool": "query",
    "arguments": {"sql": "x"},
    "result": {"content": [{"type": "text"}]},
    "status": "completed",
}
CODEX_USAGE = {
    "input_tokens": 100,
    "cached_input_tokens": 50,
    "output_tokens": 7,
    "reasoning_output_tokens": 3,
}


def test_codex_stream_and_its_failures(tmp_path: Path) -> None:
    res = _result()
    res.meta["requested_model"] = "gpt-6-luna"
    last = tmp_path / "last.txt"
    last.write_text('{"answer": "pong"}')
    parse_codex(
        _lines(
            {"type": "item.completed", "item": MCP_CALL},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
            {"type": "turn.completed", "usage": CODEX_USAGE},
        ),
        res,
        last,
    )  # fmt: skip
    assert res.error is None
    assert res.text == '{"answer": "pong"}'
    assert res.usage == {
        "input_tokens": 100,
        "cached_input_tokens": 50,
        "output_tokens": 7,
        "reasoning_tokens": 3,
    }
    assert res.tool_calls[0]["name"] == "mcp__robots__query"
    assert res.meta["builtin_items"] == [{"type": "command_execution", "command": "ls"}]
    failed = _result()
    parse_codex(
        _lines(
            {
                "type": "turn.failed",
                "error": {"message": "This content was flagged for possible cybersecurity risk"},
            }
        ),
        failed,
        tmp_path / "none",
    )
    assert failed.error == "refusal"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("stream error: 401 Unauthorized", "auth"),
        ("Not logged in · Please run /login", "auth"),
        ("You've hit your usage limit", "rate_limit"),
        (
            "The 'gpt-x' model is not supported when using Codex with a ChatGPT account",
            "bad_request",
        ),
        ("invalid_json_schema", "bad_request"),
        ('{"status":400,"error":{"type":"invalid_request_error","message":"x"}}', "bad_request"),
        ("MCP server robots failed to initialize", "tool_server"),
        ("503 Service Unavailable", "overloaded"),
        ("something else", "cli_error"),
    ],
)
def test_error_texts_are_classified(text: str, kind: str) -> None:
    assert classify(text) == kind


# Answers.


def test_answer_schemas_from_models_and_strict_dicts() -> None:
    schema, model = answer_schema(Diagnosis)
    assert model is Diagnosis
    assert schema is not None and schema["additionalProperties"] is False
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises(ValueError, match="not strict"):
        answer_schema(loose)
    strict = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert schema_errors({"a": 1}, strict) == []
    assert schema_errors({"a": "1"}, strict) == ["$.a: expected integer, got str"]
    assert schema_errors({"a": 1, "b": 2}, strict) == ["$.b: not allowed"]
    assert schema_errors({}, strict) == ["$.a: missing"]


def test_a_wrong_answer_is_malformed_output(root: Path) -> None:
    res = _result()
    res.text = '{"status": "maybe", "summary": "x"}'
    schema, model = answer_schema(SessionReport)
    runmod._check_answer(res, schema, model)
    assert res.error == "malformed_output"
    res = _result()
    res.text = '{"status": "done", "summary": "x"}'
    runmod._check_answer(res, schema, model)
    assert res.error is None
    assert res.answer == SessionReport(status="done", summary="x")


def test_retries_follow_the_kind_of_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    script: list[str | None] = []

    def fake(*args: Any, **kwargs: Any) -> AgentResult:
        error = script.pop(0)
        return AgentResult(ok=error is None, result=None, usage={}, seconds=0.0, error=error)

    monkeypatch.setattr(runmod, "run_agent", fake)
    monkeypatch.setattr("fleetlab.agents.run.time.sleep", lambda s: None)
    script[:] = ["timeout", None]
    assert run_agent_with_retry().ok
    script[:] = ["auth", None]
    res = run_agent_with_retry()
    assert res.error == "auth" and [a["error"] for a in res.meta["attempts"]] == ["auth"]
    script[:] = ["overloaded"] * 5
    res = run_agent_with_retry()
    assert len(res.meta["attempts"]) == runmod.MAX_ATTEMPTS
    script[:] = ["malformed_output", "malformed_output", None]
    assert run_agent_with_retry().error == "malformed_output"


def test_run_agent_refuses_to_run_unsandboxed(root: Path) -> None:
    with pytest.raises(ValueError, match="bwrap"):
        run_agent("claude", "opus", "S", "P", sandbox="none")  # type: ignore[arg-type]


# Sessions, memory, the queue.


def test_memory_is_per_world_and_never_reused(root: Path) -> None:
    first = Analysis.create("w01", base=root / "analyses", analysis_id="a1")
    (first.memory / "notes.md").write_text("w01 notes")
    second = Analysis.create("w02", base=root / "analyses", analysis_id="a2")
    assert second.notes() == ""
    with pytest.raises(FileExistsError):
        Analysis.create("w01", base=root / "analyses", analysis_id="a1")
    with pytest.raises(ValueError):
        Analysis.create("world-1", base=root / "analyses")


class FakeEngine:
    def __init__(self, export: Path) -> None:
        self.export = export
        self.queues: list[tuple[Path, ExperimentQueue]] = []

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        self.queues.append((queue_file, queue))
        results = [
            ExperimentResult(
                experiment_id=q.experiment_id,
                status="done",
                arms=[
                    ArmResult(arm_index=0, episode_ids=["e000002"]),
                    ArmResult(arm_index=1, episode_ids=["e000004"]),
                ],
                message="done",
            )
            for q in queue.experiments
        ]
        return EngineReply(results=results, export=self.export)


REQUEST = {
    "arms": [
        {"actions": []},
        {
            "actions": [
                {
                    "verb": "recalibrate",
                    "target": {
                        "scope": "r02",
                        "component": "scene_camera",
                        "config_key": None,
                        "condition_key": None,
                    },
                    "value": None,
                }
            ]
        },
    ],
    "robots": ["r02"],
    "episodes_per_arm": 1,
    "paired_starts": True,
    "power_note": "-",
}
DIAGNOSIS = {
    "causes": [],
    "mode_summaries": ["nothing found"],
    "warnings": [],
    "retrain": [],
}


def test_the_sessions_protocol_with_a_queue_for_the_engine(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scripted agent: session 1 writes memory and queues an experiment, session 2 sees the
    result and its notes and submits; the tool calls go through the real ToolBox."""
    seen: list[dict[str, Any]] = []

    def fake_agent(
        backend: str,
        model: str,
        system: str,
        prompt: str,
        *args: Any,
        tools: ToolAccess,
        **kwargs: Any,
    ) -> AgentResult:
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            ),
            session=tools.session,
        )
        try:
            notes = json.loads(box.call("memory_read", {}).content[0]["text"])["notes"]
            results = json.loads(box.call("experiment_results", {}).content[0]["text"])
            seen.append(
                {
                    "session": tools.session,
                    "notes": notes,
                    "results": results,
                    "prompt": prompt,
                    "system": system,
                }
            )
            if tools.session == 1:
                box.call("memory_write", {"notes": "asked x0001 about r02"})
                box.call("request_experiment", REQUEST)
                report = {"status": "waiting", "summary": "queued"}
            else:
                box.call("submit_diagnosis", DIAGNOSIS)
                report = {"status": "done", "summary": "submitted"}
        finally:
            box.close()
        return AgentResult(
            ok=True, result=report, usage={"input_tokens": 1}, seconds=0.1, run_dir=str(root)
        )

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    engine = FakeEngine(world.after)
    record = run_analysis(
        analysis, world.export, AgentSpec("claude", "haiku"), "SYSTEM", "TASK", world.env, engine
    )
    assert [s.state for s in record.sessions] == ["waiting", "done"]
    assert record.sessions[0].queued == ["x0001"]
    assert record.sessions[1].submitted == ["diagnosis-1"]
    assert seen[0]["notes"] == "" and seen[0]["results"] == []
    assert seen[1]["notes"] == "asked x0001 about r02"
    assert seen[1]["results"][0]["status"] == "done"
    assert "session 2" in seen[1]["prompt"] and "TASK" in seen[1]["prompt"]
    assert 'status "waiting"' in seen[0]["system"]
    queue_file, queue = engine.queues[0]
    on_disk = ExperimentQueue.model_validate_json(queue_file.read_text())
    assert on_disk == queue
    assert on_disk.format == "fleetlab.experiment-queue/1"
    assert on_disk.world_id == "w01" and on_disk.session == 1
    assert on_disk.rules["both_arms_always_simulated"] is True
    assert [q.experiment_id for q in on_disk.experiments] == ["x0001"]
    assert analysis.pending() == []
    assert analysis.diagnosis() == Diagnosis.model_validate(DIAGNOSIS)
    saved = json.loads((analysis.root / "sessions" / "session-01.json").read_text())
    assert saved["state"] == "waiting" and saved["queue_file"] == str(queue_file)


def test_an_engine_whose_arm_episodes_lack_the_experiment_id_is_refused(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents.sessions import EngineError, arm_problems

    def fake_agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            )
        )
        box.call("request_experiment", REQUEST)
        box.close()
        return AgentResult(
            ok=True, result={"status": "waiting", "summary": ""}, usage={}, seconds=0.0
        )

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    with pytest.raises(EngineError, match="e000002 carries experiment_id None"):
        run_analysis(
            analysis, world.export, AgentSpec("claude", "haiku"), "S", "T", world.env,
            FakeEngine(world.export), max_sessions=1,
        )  # fmt: skip
    assert analysis.pending()  # nothing recorded from a broken reply
    reply = FakeEngine(world.after).run_queue(Path(), ExperimentQueue.model_validate(
        {"world_id": "w01", "analysis_id": "a", "session": 1, "created_at": "-", "export": "-",
         "experiments": [q.model_dump() for q in analysis.requests()]}
    ))  # fmt: skip
    assert arm_problems(reply) == []


def test_without_an_engine_the_analysis_stops_waiting_with_the_queue_written(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            )
        )
        box.call("request_experiment", REQUEST)
        box.close()
        return AgentResult(
            ok=True, result={"status": "waiting", "summary": ""}, usage={}, seconds=0.0
        )

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    record = run_analysis(
        analysis, world.export, AgentSpec("codex", "gpt-6-luna"), "S", "T", world.env, None
    )
    assert record.state == "waiting"
    assert record.sessions[0].queue_file is not None
    assert len(analysis.pending()) == 1
    with pytest.raises(ValueError, match="no request"):
        analysis.record_results(
            [ExperimentResult(experiment_id="x0009", status="done", arms=[], message="")]
        )


def test_a_failed_session_is_not_retried_after_it_queued(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        calls.append(1)
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            )
        )
        box.call("request_experiment", REQUEST)
        box.close()
        return AgentResult(ok=False, result=None, usage={}, seconds=0.0, error="timeout")

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    record = run_analysis(
        analysis, world.export, AgentSpec("claude", "opus"), "S", "T", world.env, None
    )
    assert len(calls) == 1
    assert record.sessions[0].state == "waiting"
    assert not record.sessions[0].rerun and not record.sessions[0].lost_twice
    assert [a["error"] for a in record.sessions[0].attempts] == ["timeout"]
    assert len(analysis.requests()) == 1


@pytest.mark.parametrize(
    ("error", "detail", "tool_log", "starts"),
    [
        ("timeout", "killed after 600s", [], 2),
        ("overloaded", "503", [], 2),
        ("cli_error", "transport error: connection reset", [], 2),
        ("cli_error", "executable not found", [], 1),
        ("tool_server", "failed to initialize", [], 2),
        ("tool_server", "transport closed", [{"tool": "query"}], 1),
        ("malformed_output", "not JSON", [], 1),
        ("rate_limit", "429", [], 1),
        ("auth", "401", [], 1),
        ("refusal", "content policy", [], 1),
        ("bad_request", "unknown model", [], 1),
    ],
)
def test_scored_session_starts_by_error_kind(
    world: World,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
    detail: str,
    tool_log: list[dict[str, Any]],
    starts: int,
) -> None:
    calls = []

    def fake_agent(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append((args, kwargs))
        return AgentResult(
            ok=False, result=None, usage={"output_tokens": 3}, seconds=4,
            error=error, error_detail=detail, tool_log=tool_log,
            run_dir=f"start-{len(calls)}",
        )  # fmt: skip

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    record = run_analysis(
        analysis, world.export, AgentSpec("codex", "gpt-6-astra"), "S", "T", world.env, None
    )
    assert len(calls) == starts
    assert all(call == calls[0] for call in calls)  # unchanged prompts, tools and conditions
    assert len(record.sessions) == 1
    session = record.sessions[0]
    assert session.state == "failed" and session.error == error
    assert session.rerun is (starts == 2) and session.lost_twice is (starts == 2)
    assert session.seconds == starts * 4
    assert [a["run_dir"] for a in session.attempts] == [f"start-{n}" for n in range(1, starts + 1)]
    assert all(a["error"] == error and a["detail"] == detail for a in session.attempts)
    assert all(a["seconds"] == 4 and a["answer_present"] is False for a in session.attempts)
    saved = json.loads((analysis.root / "sessions" / "session-01.json").read_text())
    assert saved == asdict(session)


@pytest.mark.parametrize("error", ["timeout", "overloaded", "cli_error", "tool_server"])
@pytest.mark.parametrize("answer_surface", ["result", "text", "answer"])
def test_scored_session_never_restarts_an_answer_that_arrived(
    world: World,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
    answer_surface: str,
) -> None:
    calls = []

    def fake_agent(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append(1)
        res = AgentResult(
            ok=False, result=None, usage={}, seconds=1, error=error,
            error_detail="transport error: connection reset",
        )  # fmt: skip
        setattr(res, answer_surface, "unreadable answer" if answer_surface == "text" else {})
        return res

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    record = run_analysis(
        analysis, world.export, AgentSpec("codex", "gpt-6-astra"), "S", "T", world.env, None
    )
    assert calls == [1]
    session = record.sessions[0]
    assert session.attempts[0]["answer_present"] is True
    assert not session.rerun and not session.lost_twice


@pytest.mark.parametrize("replacement_error", [None, "malformed_output", "timeout"])
def test_scored_session_uses_only_the_replacement_result(
    world: World,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_error: str | None,
) -> None:
    calls = []

    def fake_agent(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append((args, kwargs))
        if len(calls) == 1:
            return AgentResult(
                ok=False, result=None, usage={}, seconds=7, error="overloaded", run_dir="lost"
            )
        assert len(calls) == 2
        return AgentResult(
            ok=replacement_error is None,
            result=(
                {"status": "done", "summary": "replacement"}
                if replacement_error is None else None
            ),
            text="not JSON" if replacement_error == "malformed_output" else "",
            usage={"output_tokens": 5}, seconds=2, error=replacement_error, run_dir="replacement",
        )  # fmt: skip

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    record = run_analysis(
        analysis, world.export, AgentSpec("codex", "gpt-6-astra"), "S", "T", world.env, None
    )
    assert len(calls) == 2 and calls[0] == calls[1]
    session = record.sessions[0]
    assert session.error == replacement_error and session.run_dir == "replacement"
    assert session.ok is (replacement_error is None) and session.seconds == 9
    assert session.usage == {"output_tokens": 5}
    assert session.rerun and session.lost_twice is (replacement_error == "timeout")
    assert session.report == (
        {"status": "done", "summary": "replacement"} if replacement_error is None else None
    )


@pytest.mark.parametrize("left_behind", ["request", "submission"])
def test_replacement_work_is_judged_like_first_start_work(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch, left_behind: str
) -> None:
    calls = []

    def fake_agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        calls.append(1)
        if len(calls) == 2:
            box = ToolBox(
                Paths(
                    export=tools.export,
                    memory=tools.memory,
                    outbox=tools.outbox,
                    results=tools.results,
                )
            )
            box.call(
                "request_experiment" if left_behind == "request" else "submit_diagnosis",
                REQUEST if left_behind == "request" else DIAGNOSIS,
            )
            box.close()
        return AgentResult(ok=False, result=None, usage={}, seconds=1, error="timeout")

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    engine = FakeEngine(world.after)
    record = run_analysis(
        analysis,
        world.export,
        AgentSpec("codex", "gpt-6-astra"),
        "S",
        "T",
        world.env,
        engine,
        max_sessions=1,
    )
    assert calls == [1, 1]
    assert len(record.sessions) == 1 and not record.sessions[0].lost_twice
    assert record.sessions[0].rerun
    if left_behind == "request":
        assert record.state == "waiting" and len(engine.queues) == 1
        assert not analysis.pending()
    else:
        assert record.state == "done" and not engine.queues
        assert analysis.diagnosis() is not None


@pytest.mark.parametrize("error", ["auth", "rate_limit"])
def test_a_stop_error_after_queueing_cannot_run_experiments_or_another_session(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    calls = []

    def fake_agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        calls.append(1)
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            )
        )
        box.call("request_experiment", REQUEST)
        box.close()
        return AgentResult(ok=False, result=None, usage={}, seconds=0.0, error=error)

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    engine = FakeEngine(world.after)
    record = run_analysis(
        analysis, world.export, AgentSpec("codex", "gpt-6-astra"), "S", "T", world.env, engine
    )
    assert calls == [1]
    assert not engine.queues
    assert len(record.sessions) == 1
    assert (record.sessions[0].state, record.sessions[0].error) == ("waiting", error)
    assert len(analysis.pending()) == 1


# The gate.


def test_a_batch_refuses_an_unchecked_cli(world: World, root: Path) -> None:
    with pytest.raises(GateError, match="have not run"):
        require_verified("claude", world.env)
    with pytest.raises(GateError):
        Batch(world.env, ["claude"])


def test_a_batch_refuses_a_failed_or_stale_record(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import checks

    monkeypatch.setattr(checks, "binary_sha256", lambda path: "abc")
    verified = root / "verified"
    verified.mkdir(parents=True)
    version = pinned_version("codex")
    record = {
        "passed": False,
        "sha256": "abc",
        "tool_env": world.env.digest(),
        "implementation": checks.implementation_digest(),
        "checks": [{"name": "wire", "passed": False}],
    }
    (verified / f"codex-{version}.json").write_text(json.dumps(record))
    with pytest.raises(GateError, match="wire"):
        require_verified("codex", world.env)
    (verified / f"codex-{version}.json").write_text(
        json.dumps({**record, "passed": True, "sha256": "old"})
    )
    with pytest.raises(GateError, match="binary changed"):
        require_verified("codex", world.env)
    (verified / f"codex-{version}.json").write_text(
        json.dumps({**record, "passed": True, "tool_env": "other"})
    )
    with pytest.raises(GateError, match="tool environment"):
        require_verified("codex", world.env)
    (verified / f"codex-{version}.json").write_text(json.dumps({**record, "passed": True}))
    assert require_verified("codex", world.env)["passed"]
    batch = Batch(world.env, ["codex"], concurrency=8)
    assert batch.records["codex"]["passed"]
    with pytest.raises(ValueError, match="concurrency"):
        Batch(world.env, ["codex"], concurrency=9)


def test_a_batch_stops_on_a_login_error(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import checks

    monkeypatch.setattr(
        "fleetlab.agents.batch.require_verified", lambda backend, env: {"passed": True}
    )
    _ = checks
    batch = Batch(world.env, ["claude"], concurrency=1)
    errors = [None, "auth", None, None]
    out = batch.map(
        lambda e: AgentResult(
            ok=e is None,
            result=None,
            usage={"input_tokens": 1, "cost_usd_list_price": 0.25},
            seconds=0,
            error=e,
        ),
        errors,
    )
    assert [r is None for r in out] == [False, False, True, True]
    assert batch.stopped is not None and batch.stopped.startswith("auth")
    assert Batch.usage(out) == {"input_tokens": 2, "cost_usd_list_price": 0.5}


def test_tool_names_follow_the_contracts() -> None:
    assert tool_names() == (
        "query", "get_frames", "describe", "list_skills", "read_skill", "request_experiment",
        "experiment_results", "submit_diagnosis", "submit_fix_set", "memory_read", "memory_write",
        "write_learned_skill",
        "experiment_budget", "earlier_decisions", "submit_decision",  # loop.md, section 8
        "request_probe", "probe_results",  # loop.md, section 10; opt-in (OPT_IN_TOOLS)
        # The job tools, opt-in (contracts.tools.JOB_TOOLS; docs/design/jobs.md).
        "operator_state", "submit_acceptance_proposal", "submit_job_program", "job_results",
        "save_job", "write_check", "run_check", "job_library", "submit_job_account",
    )  # fmt: skip


def test_the_subscription_window_is_read_from_claude_s_stream() -> None:
    res = _result()
    info = {
        "status": "allowed_warning",
        "rateLimitType": "five_hour",
        "resetsAt": 1791427800,
        "unifiedWindows": {
            "five_hour": {"utilization": 0.96, "resetsAt": 1791427800},
            "seven_day": {"utilization": 0.47, "resetsAt": 1791500400},
        },
    }
    parse_claude(_lines({"type": "rate_limit_event", "rate_limit_info": info}), res)
    assert res.meta["window"] == {
        "status": "allowed_warning",
        "type": "five_hour",
        "utilization": {"five_hour": 0.96, "seven_day": 0.47},
        "resets_at": 1791427800,
    }


def test_a_batch_stops_before_the_subscription_window_is_full(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fleetlab.agents.batch.require_verified", lambda backend, env: {"passed": True}
    )
    batch = Batch(world.env, ["claude"], concurrency=1, stop_at_window=0.95)

    def call(used: float) -> AgentResult:
        res = AgentResult(ok=True, result=None, usage={}, seconds=0)
        res.meta["window"] = {"status": "allowed", "utilization": {"five_hour": used}}
        return res

    out = batch.map(call, [0.90, 0.96, 0.50])
    assert [r is None for r in out] == [False, False, True]
    assert batch.stopped is not None and "96%" in batch.stopped


def test_a_batch_after_the_gate_runs_the_checks_and_audits_its_own_inputs(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents.checks import CheckReport

    gated: list[list[str]] = []

    def fake_verified(backend: str, env: ToolEnv) -> dict[str, Any]:
        if not gated:
            raise GateError("no record")
        return {"passed": True}

    def fake_gate(backends: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        gated.append(list(backends))
        return {}

    def fake_login(backend: str) -> CheckReport:
        report = CheckReport("auth", backend)
        report.add("logged in", "CLI sandbox", True, True)
        return report

    monkeypatch.setattr("fleetlab.agents.batch.require_verified", fake_verified)
    monkeypatch.setattr("fleetlab.agents.batch.run_gate", fake_gate)
    monkeypatch.setattr("fleetlab.agents.batch.auth_check", fake_login)
    fleet = __import__("log_toy").toy_fleet()
    batch = Batch.after_gate(
        world.env, ["claude"], world.export, [world.run], fleet=fleet, texts={"task": "find it"}
    )
    assert gated == [["claude"]]
    Batch.after_gate(world.env, ["claude"], world.export, [world.run], fleet=fleet, rerun=False)
    assert gated == [["claude"]]  # the record is valid now: no second gate
    Batch.after_gate(world.env, ["claude"], world.export, [world.run], fleet=fleet)
    assert gated == [["claude"], ["claude"]]  # by default, before every batch
    assert batch.records["claude"]["passed"]
    with pytest.raises(GateError, match="source audit"):
        Batch.after_gate(
            world.env,
            ["claude"],
            world.export,
            [world.run],
            fleet=fleet,
            texts={"task": "look for toy_camera_wobble"},  # a fault kind of the toy run
        )


def test_the_gate_s_layout_check_and_codex_home_check_pass_on_a_placed_environment(
    world: World, root: Path
) -> None:
    import sqlite3

    from fleetlab.agents.checks import isolation_check, layout_check

    assert not isolation_check("codex").passed  # no slot home yet: nothing was checked
    for backend in BACKENDS:
        report = layout_check(backend, placed_env(world), world.export, [world.run / "truth"])
        assert report.passed, report.failures()
    home = root / "codex-homes" / "00"
    home.mkdir(parents=True, exist_ok=True)  # the layout check held this slot
    (home / "auth.json").write_bytes(b"")  # bwrap's mount placeholder
    (home / ".sqlite-maintenance.lock").write_bytes(b"")
    con = sqlite3.connect(home / "memories_1.sqlite")
    con.execute("CREATE TABLE _sqlx_migrations (version INTEGER)")
    con.execute("INSERT INTO _sqlx_migrations VALUES (1)")
    con.execute("CREATE TABLE consolidation_progress (singleton INTEGER, max_thread_count INTEGER)")
    con.execute("INSERT INTO consolidation_progress VALUES (1, 0)")
    con.execute("CREATE TABLE stage1_outputs (text TEXT)")
    con.commit()
    assert isolation_check("codex").passed, isolation_check("codex").failures()
    for plant, undo in (
        (lambda: (home / "AGENTS.md").write_text("planted"), lambda: (home / "AGENTS.md").unlink()),
        (
            lambda: (home / "history.jsonl").write_text("{}"),
            lambda: (home / "history.jsonl").unlink(),
        ),
        (
            lambda: (home / "auth.json").write_bytes(b"x"),
            lambda: (home / "auth.json").write_bytes(b""),
        ),
        (
            lambda: (home / ".sqlite-maintenance.lock").write_bytes(b"remembered"),
            lambda: (home / ".sqlite-maintenance.lock").write_bytes(b""),
        ),
        (
            lambda: execute(con, "INSERT INTO stage1_outputs VALUES ('remembered')"),
            lambda: execute(con, "DELETE FROM stage1_outputs"),
        ),
    ):
        plant()
        assert not isolation_check("codex").passed
        undo()
    assert isolation_check("codex").passed
    con.close()


@pytest.mark.parametrize("kind", ["directory", "link", "dangling link"])
def test_sqlite_maintenance_lock_refuses_nonregular_files(tmp_path: Path, kind: str) -> None:
    from fleetlab.agents.checks import codex_home_rows

    lock = tmp_path / ".sqlite-maintenance.lock"
    if kind == "directory":
        lock.mkdir()
    else:
        target = tmp_path / "absent"
        if kind == "link":
            target = tmp_path / "installation_id"
            target.write_bytes(b"")
        lock.symlink_to(target)
    rows = codex_home_rows(tmp_path)
    assert any(name.startswith("SQLite maintenance") and got != want for name, got, want in rows)


def execute(con: Any, sql: str) -> None:
    con.execute(sql)
    con.commit()


def test_the_login_placeholder_must_stay_empty(root: Path) -> None:
    from fleetlab.agents.run import login_placeholder_problem

    home = root / "home"
    home.mkdir(parents=True)
    assert login_placeholder_problem(home) is None  # absent
    (home / "auth.json").write_bytes(b"")
    assert login_placeholder_problem(home) is None
    (home / "auth.json").write_bytes(b"{}")
    assert "copy of the login" in (login_placeholder_problem(home) or "")
    (home / "auth.json").unlink()
    (home / "auth.json").symlink_to(BOX_LOGIN["codex"])  # a path that exists only in a sandbox
    assert "link" in (login_placeholder_problem(home) or "")


def test_the_agent_root_must_be_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.agents.cli import agent_root

    fresh = tmp_path / "fresh"
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(fresh))
    assert agent_root() == fresh
    planted = tmp_path / "planted"
    planted.mkdir(mode=0o755)
    planted.chmod(0o755)
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(planted))
    with pytest.raises(PermissionError, match="mode 755"):
        agent_root()
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(link))
    with pytest.raises(PermissionError, match="symbolic link"):
        agent_root()
    a_file = tmp_path / "a-file"
    a_file.write_text("")
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(a_file))
    with pytest.raises(PermissionError, match="not a directory"):
        agent_root()


def test_only_the_chosen_skills_are_mounted(tmp_path: Path) -> None:
    from fleetlab.agents.toolenv import build_skills

    library = tmp_path / "library"
    write_toy_skill(library)
    other = library / "assess-risk"
    other.mkdir()
    (other / "SKILL.md").write_text(
        (library / "triage" / "SKILL.md").read_text().replace("name: triage", "name: assess-risk")
    )
    every = build_skills(tmp_path / "out", library)
    assert sorted(p.name for p in every.iterdir()) == ["assess-risk", "triage"]
    chosen = build_skills(tmp_path / "out", library, ["triage"])
    assert sorted(p.name for p in chosen.iterdir()) == ["triage"]
    assert chosen != every  # named by content
    with pytest.raises(ValueError, match="no skills"):
        build_skills(tmp_path / "out", library, ["scope"])


def test_a_session_that_submitted_and_then_failed_is_done_with_its_error(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        calls.append(1)
        box = ToolBox(
            Paths(
                export=tools.export, memory=tools.memory, outbox=tools.outbox, results=tools.results
            )
        )
        box.call("submit_diagnosis", DIAGNOSIS)
        box.close()
        return AgentResult(ok=False, result=None, usage={}, seconds=0.0, error="timeout")

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", fake_agent)
    analysis = Analysis.create("w01", base=root / "analyses")
    record = run_analysis(
        analysis, world.export, AgentSpec("claude", "opus"), "S", "T", world.env, None
    )
    assert len(calls) == 1  # not retried: it left a submission behind
    session = record.sessions[0]
    assert (session.state, session.ok, session.error) == ("done", False, "timeout")
    assert not session.rerun and not session.lost_twice
    assert analysis.diagnosis() is not None


def test_an_analysis_run_mounts_the_analysis_skills_of_the_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import toolenv
    from fleetlab.skill_library import ANALYSIS_SKILLS

    monkeypatch.setattr(toolenv, "_copy_python", lambda base: tmp_path / "python")
    monkeypatch.setattr(toolenv, "_site", lambda base, python: tmp_path / "site")
    env = toolenv.build_tool_env(root=tmp_path / "root", role="analysis")
    assert env.skills is not None
    assert sorted(p.name for p in env.skills.iterdir()) == sorted(ANALYSIS_SKILLS)
    with pytest.raises(KeyError):
        toolenv.build_tool_env(root=tmp_path / "root", role="teleoperator")


def test_parallel_gate_receipts_do_not_replace_another_environment(
    world: World, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import checks

    env = placed_env(world)
    monkeypatch.setattr(checks, "binary_sha256", lambda path: "binary")
    version = pinned_version("codex")
    record = {
        "passed": True,
        "sha256": "binary",
        "tool_env": env.digest(),
        "implementation": checks.implementation_digest(),
    }
    own = checks._record_path("codex", version, env)
    own.write_text(json.dumps(record))
    # A different checkout's older gate still writes the shared legacy filename.
    checks._record_path("codex", version).write_text(json.dumps({**record, "tool_env": "other"}))
    assert require_verified("codex", env)["tool_env"] == env.digest()
