"""The gate on both tool offers (after the probe run stopped at the gate).

The tool server leaves the opt-in tools (`OPT_IN_TOOLS`, the probes) out of its default
offer. The gate's wire check compares the default offer against every tool but those, and
checks apart from it an offer that names every tool: there each opt-in tool must reach the
model under our server's name and with our server's description (`offer_check`), and the
tool server must dispatch its calls (`dispatch_rows`), while the default offer refuses them.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from log_toy import TASK_CLASSES, write_toy_run

from fleetlab.agents import checks
from fleetlab.agents.checks import (
    DEFAULT_OFFER,
    DISPATCH_CALLS,
    EVERY_TOOL,
    CheckReport,
    dispatch_rows,
    offer_check,
)
from fleetlab.agents.run import ToolAccess
from fleetlab.contracts.tools import JOB_TOOLS, OPT_IN_TOOLS, TOOLS
from fleetlab.log import export_runs
from fleetlab.toolserver.__main__ import INSTRUCTIONS, build, parse
from fleetlab.toolserver.mcp import StdioServer
from fleetlab.toolserver.server import (
    SERVER_NAME,
    SERVER_VERSION,
    Limits,
    ToolBox,
    default_tool_names,
    tool_description,
    tool_names,
)

SPECS = {spec.name: spec for spec in TOOLS}


def ours(tool: str) -> str:
    return tool_description(SPECS[tool], Limits())


def codex_body(
    tools: Sequence[str], *, described: dict[str, str] | None = None, in_exec: Sequence[str] = ()
) -> dict[str, Any]:
    """A request as codex 0.161 sends it: its own code-mode tools and ours in namespaces."""
    exec_text = "Run JavaScript.\n" + "".join(f"### `{n}`\nnested\n" for n in in_exec)
    mine = [
        {"type": "function", "name": t, "description": (described or {}).get(t, ours(t)),
         "parameters": {"type": "object"}, "strict": False}
        for t in tools
    ]  # fmt: skip
    own = [
        {"type": "custom", "name": "exec", "description": exec_text},
        {"type": "custom", "name": "wait", "description": "wait"},
        {"type": "custom", "name": "request_user_input_async", "description": "ask"},
    ]
    return {
        "model": "gpt-6-luna",
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "description": "", "tools": own},
                {"type": "namespace", "name": "mcp__robots", "description": "", "tools": mine},
            ]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "USER-PROMPT-PROBE"}]},
        ],
    }  # fmt: skip


def claude_body(tools: Sequence[str], *, described: dict[str, str] | None = None) -> dict[str, Any]:
    """A request as claude 2.1.293 sends it: StructuredOutput and ours, prefixed."""
    mine = [
        {"name": f"mcp__robots__{t}", "description": (described or {}).get(t, ours(t)),
         "input_schema": {"type": "object"}}
        for t in tools
    ]  # fmt: skip
    return {
        "model": "haiku",
        "system": [{"type": "text", "text": "SYSTEM-PROMPT-PROBE"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "USER-PROMPT-PROBE"}]}],
        "tools": [{"name": "StructuredOutput", "description": "answer", "input_schema": {}}, *mine],
    }


def body_of(backend: str, tools: Sequence[str], **kw: Any) -> dict[str, Any]:
    return codex_body(tools, **kw) if backend == "codex" else claude_body(tools, **kw)


def checked(backend: str, body: dict[str, Any], offer: Sequence[str] | None) -> CheckReport:
    report = CheckReport("wire", backend)
    offer_check(report, backend, body, offer)  # type: ignore[arg-type]
    return report


def failed(report: CheckReport) -> list[str]:
    return [r.target for r in report.rows if not r.ok]


def test_the_default_offer_is_every_tool_but_the_opt_in_ones(tmp_path: Path) -> None:
    assert {"request_probe", "probe_results", *JOB_TOOLS} == OPT_IN_TOOLS
    assert set(OPT_IN_TOOLS) <= set(tool_names())
    assert default_tool_names() == tuple(t for t in tool_names() if t not in OPT_IN_TOOLS)
    # What the runtime allows the CLI is what the tool server offers, and what the launch
    # audit records: the default offer unless the runtime names the tools.
    access = ToolAccess(env=None, export=tmp_path, memory=tmp_path, outbox=tmp_path,  # type: ignore[arg-type]
                        results=tmp_path)  # fmt: skip
    assert access.enabled == default_tool_names()
    every = ToolAccess(env=None, export=tmp_path, memory=tmp_path, outbox=tmp_path,  # type: ignore[arg-type]
                       results=tmp_path, tools=tool_names())  # fmt: skip
    assert every.enabled == tool_names()


@pytest.mark.parametrize("backend", ["claude", "codex"])
@pytest.mark.parametrize("offer", [None, tool_names()], ids=["default", "every-tool"])
def test_each_offer_passes_on_what_the_cli_sends_for_it(
    backend: str, offer: tuple[str, ...] | None
) -> None:
    sent = default_tool_names() if offer is None else offer
    report = checked(backend, body_of(backend, sent), offer)
    assert report.passed, report.failures()
    targets = [r.target for r in report.rows]
    probes = [t for t in targets if t.startswith(("request_probe", "probe_results"))]
    # The opt-in tools' own rows only where they are offered: route and description each.
    assert len(probes) == (0 if offer is None else 4)


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_a_request_unlike_its_offer_fails(backend: str) -> None:
    # The probe run's failure, the other way round: an offer naming the probes that the
    # request does not carry fails on the tools and on each probe tool's own rows.
    report = checked(backend, body_of(backend, default_tool_names()), tool_names())
    assert "our tools" in failed(report)
    assert {t.split(":")[0] for t in failed(report) if ":" in t} == set(OPT_IN_TOOLS)
    # The probes in a request on the default offer: refused wherever they appear (codex's
    # list of what may stand beside its code-mode tools is the offer, so there too).
    report = checked(backend, body_of(backend, tool_names()), None)
    besides = ["built-in tools besides code mode"] if backend == "codex" else []
    assert failed(report) == [
        *besides,
        "our tools",
        "opt-in tools not asked for, anywhere in the request",
    ]
    # A probe tool whose description is not our server's.
    altered = body_of(backend, tool_names(), described={"request_probe": "Queue a probe."})
    assert failed(checked(backend, altered, tool_names())) == [
        "request_probe: the description our tool server gives"
    ]


def test_a_probe_tool_reachable_inside_exec_is_not_dispatched_to_our_server() -> None:
    nested = [t for t in tool_names() if t != "request_probe"]
    body = codex_body(nested, in_exec=["mcp__robots.request_probe"])
    assert failed(checked("codex", body, tool_names())) == [
        "tools reachable inside exec",
        "request_probe: offered under our server's name, where its calls are dispatched",
        "request_probe: the description our tool server gives",
    ]


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("gate-offers")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    return base / "export"


def served(root: Path, export: Path, tools: Sequence[str] | None) -> dict[str, Any]:
    """What the tool server, built as `python -m fleetlab.toolserver` builds it, answers
    the dispatch check's messages (the gate sends the same ones to it in its sandbox)."""
    for part in ("memory", "outbox", "results"):
        (root / part).mkdir(parents=True, exist_ok=True)
    argv = [f"--{p}={root / p}" for p in ("memory", "outbox", "results")]
    argv += [f"--export={export}", f"--log={root / 'calls.jsonl'}"]
    if tools is not None:
        argv.append("--tools=" + ",".join(tools))
    toolbox: ToolBox = build(parse(argv))
    server = StdioServer(
        SERVER_NAME, SERVER_VERSION, toolbox.mcp_tools(), INSTRUCTIONS, fallback=toolbox.call
    )
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    messages += [
        {"jsonrpc": "2.0", "id": n, "method": "tools/call",
         "params": {"name": name, "arguments": arguments}}
        for n, (name, arguments) in enumerate(DISPATCH_CALLS, 3)
    ]  # fmt: skip
    out = io.StringIO()
    try:
        server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in messages)), out)
    finally:
        toolbox.close()
    replies = {r["id"]: r for r in map(json.loads, out.getvalue().splitlines())}
    calls = {}
    for n, (name, _) in enumerate(DISPATCH_CALLS, 3):
        result = replies[n]["result"]
        text = "\n".join(p.get("text", "") for p in result["content"])
        calls[name] = {"isError": bool(result["isError"]), "text": text}
    return {"listed": replies[2]["result"]["tools"], "calls": calls}


def test_the_tool_server_dispatches_the_opt_in_tools_only_where_offered(
    tmp_path: Path, export: Path
) -> None:
    observed = {
        DEFAULT_OFFER: served(tmp_path / "default", export, None),
        EVERY_TOOL: served(tmp_path / "every", export, tool_names()),
    }
    assert "is not available" in observed[DEFAULT_OFFER]["calls"]["request_probe"]["text"]
    receipt = json.loads(observed[EVERY_TOOL]["calls"]["request_probe"]["text"])
    assert receipt["accepted"] and receipt["probe_id"] == "p0001"  # its handler ran
    queued = json.loads(observed[EVERY_TOOL]["calls"]["probe_results"]["text"])
    assert [p["status"] for p in queued] == ["queued"]
    report = CheckReport("dispatch", "codex")
    dispatch_rows(report, observed)
    assert report.passed, report.failures()
    assert len(report.rows) == 2 * 1 + 2 * 1 + 2 * 2  # listed per offer, two calls, two rows
    # The offers the other way round fail on every row.
    swapped = CheckReport("dispatch", "codex")
    dispatch_rows(
        swapped, {DEFAULT_OFFER: observed[EVERY_TOOL], EVERY_TOOL: observed[DEFAULT_OFFER]}
    )
    assert not any(r.ok for r in swapped.rows), [r.target for r in swapped.rows if r.ok]


def test_the_gate_checks_both_offers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []

    def report(name: str) -> CheckReport:
        r = CheckReport(name, "codex")
        r.add("fake", "test", True, True)
        return r

    def wire(backend: str, model: str, env: Any, export: Path, audit: Any = None, *,
             tools: Sequence[str] | None = None) -> CheckReport:  # fmt: skip
        calls.append(("wire", tools))
        return report("wire" if tools is None else "wire, opt-in tools offered")

    def dispatch(backend: str, env: Any, export: Path) -> CheckReport:
        calls.append(("dispatch", None))
        return report("dispatch")

    class Env:
        def digest(self) -> str:
            return "env"

    for name in ("layout_check", "static_check", "lifetime_check"):
        monkeypatch.setattr(checks, name, lambda *a, _n=name, **k: report(_n))
    monkeypatch.setattr(checks, "source_audit", lambda *a, **k: report("source audit"))
    monkeypatch.setattr(checks, "isolation_check", lambda *a, **k: report("isolation"))
    monkeypatch.setattr(checks, "auth_check", lambda *a, **k: report("auth"))
    monkeypatch.setattr(checks, "wire_check", wire)
    monkeypatch.setattr(checks, "dispatch_check", dispatch)
    monkeypatch.setattr(checks, "pinned_version", lambda backend: "0.0")
    monkeypatch.setattr(checks, "binary", lambda *a: tmp_path / "cli")
    monkeypatch.setattr(checks, "installed_version", lambda path: "0.0")
    monkeypatch.setattr(checks, "binary_sha256", lambda path: "sha")
    monkeypatch.setattr(checks, "implementation_digest", lambda: "impl")
    monkeypatch.setattr(checks, "_record_path", lambda *a: tmp_path / "record.json")
    record = checks.run_gate(["codex"], Env(), tmp_path, [tmp_path])["codex"]  # type: ignore[arg-type]
    assert calls == [("wire", None), ("wire", tool_names()), ("dispatch", None)]
    names = [c["name"] for c in record["checks"]]
    assert {"wire", "wire, opt-in tools offered", "dispatch"} <= set(names)
    assert record["passed"]
