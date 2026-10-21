"""The agents' tool server against the toy export: every tool, the call log, the MCP protocol,
the import graph, and the leak properties in process and inside the real tool sandbox.

src/fleetlab/toolserver/; docs/design/contracts.md, section 12.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from log_toy import TASK_CLASSES, write_toy_run
from PIL import Image

from fleetlab.agents.sandbox import ToolMounts, tool_server_command
from fleetlab.agents.toolenv import (
    FORBIDDEN_MODULES,
    closure_problems,
    import_closure,
    project_tool_env,
)
from fleetlab.agents.toy import write_toy_skill
from fleetlab.contracts.tools import (
    OPT_IN_TOOLS,
    TOOLS,
    Receipt,
    json_schema_strict,
    strict_schema_problems,
)
from fleetlab.log import export_runs
from fleetlab.log._files import read_canary
from fleetlab.toolserver.files import EXPERIMENTS_FILE, RESULTS_FILE, append_line
from fleetlab.toolserver.mcp import PROTOCOL_VERSIONS, StdioServer
from fleetlab.toolserver.server import Limits, Paths, ToolBox, tool_names

TOOLSERVER = Path(__file__).resolve().parents[1] / "src" / "fleetlab" / "toolserver"


@dataclass(frozen=True)
class Toy:
    run: Path
    export: Path
    skills: Path
    canary: str


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Toy:
    base = tmp_path_factory.mktemp("toolserver")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    write_toy_skill(base / "skills")
    token = read_canary(base / "run")
    assert token
    return Toy(base / "run", base / "export", base / "skills", token)


def analysis_paths(root: Path, toy: Toy) -> Paths:
    for part in ("memory", "outbox", "results"):
        (root / part).mkdir(parents=True, exist_ok=True)
    return Paths(
        export=toy.export,
        memory=root / "memory",
        outbox=root / "outbox",
        results=root / "results",
        skills=toy.skills,
        log=root / "calls.jsonl",
    )


@pytest.fixture
def box(tmp_path: Path, toy: Toy) -> Iterator[ToolBox]:
    toolbox = ToolBox(analysis_paths(tmp_path, toy))
    yield toolbox
    toolbox.close()


def text(result: Any) -> str:
    parts = [p["text"] for p in result.content if p["type"] == "text"]
    return "\n".join(parts)


def data(result: Any) -> Any:
    return json.loads(text(result))


EXPERIMENT = {
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
    "episodes_per_arm": 4,
    "paired_starts": True,
    "power_note": "a first look",
}


def diagnosis(episode: str = "e000002") -> dict[str, Any]:
    return {
        "causes": [
            {
                "target": {
                    "scope": "r02",
                    "component": "scene_camera",
                    "config_key": None,
                    "condition_key": None,
                },
                "direction": "degraded",
                "confidence": 0.5,
                "evidence": [{"episode_id": episode, "step": 2, "table": None, "query": None}],
                "failure_mode": "misplaced grasps",
            }
        ],
        "mode_summaries": ["misplaced grasps on one robot"],
        "warnings": [],
        "retrain": [],
    }


# The tools, one by one.


def test_query_answers_from_the_visible_tables(box: ToolBox) -> None:
    out = data(
        box.call("query", {"sql": "SELECT robot_id, count(*) FROM episodes GROUP BY 1 ORDER BY 1"})
    )
    assert out == {
        "columns": ["robot_id", "count_star()"],
        "rows": [["r01", 5], ["r02", 5]],
        "truncated": False,
    }


def test_query_truncates_at_the_row_limit(tmp_path: Path, toy: Toy) -> None:
    box = ToolBox(analysis_paths(tmp_path, toy), limits=Limits(max_rows=3))
    out = data(box.call("query", {"sql": "SELECT episode_id FROM episodes ORDER BY 1"}))
    assert len(out["rows"]) == 3
    assert out["truncated"] is True


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE x AS SELECT 1",
        "DROP TABLE episodes",
        "SELECT 1; SELECT 2",
        "COPY episodes TO 'out.csv'",
        "SET enable_external_access = true",
        "ATTACH 'other.duckdb'",
        "INSTALL httpfs",
        "PRAGMA enable_profiling",
        "CALL pragma_version()",
        # DuckDB calls these EXPLAIN and runs the statement inside (review finding S1).
        "EXPLAIN ANALYZE INSERT INTO episodes SELECT * FROM episodes",
        "EXPLAIN ANALYZE CREATE TABLE x AS SELECT 1",
        "EXPLAIN ANALYZE DELETE FROM episodes",
        "EXPLAIN SELECT * FROM episodes",
    ],
)
def test_query_refuses_anything_but_one_select(box: ToolBox, sql: str) -> None:
    result = box.call("query", {"sql": sql})
    assert result.is_error
    rows = data(box.call("query", {"sql": "SELECT count(*) FROM episodes"}))["rows"]
    assert rows == [[10]]  # nothing changed


def test_a_planted_hidden_file_is_unreadable_through_query(box: ToolBox, toy: Toy) -> None:
    canary_file = toy.run / "truth" / "canary.txt"
    assert canary_file.read_text().strip() == toy.canary
    hidden_table = next((toy.run / "truth").glob("*.parquet"))
    attempts = [
        f"SELECT * FROM read_text('{canary_file}')",
        f"SELECT * FROM read_text('{toy.export}/../run/truth/canary.txt')",
        f"SELECT * FROM read_csv('{canary_file}', header=false)",
        f"SELECT * FROM read_parquet('{hidden_table}')",
        f"SELECT * FROM glob('{toy.run}/**')",
        f"SELECT * FROM read_text('{toy.export}/data_dictionary.md')",  # even inside the export
        f"SELECT * FROM read_blob('/proc/self/root{canary_file}')",
        f"SELECT * FROM read_parquet('{toy.export}/log/episodes.parquet')",
    ]
    for sql in attempts:
        result = box.call("query", {"sql": sql})
        assert result.is_error, sql
        assert toy.canary not in text(result)
    log = (box.paths.log or Path()).read_text()
    assert toy.canary not in log


def test_get_frames_returns_png_images_at_the_logged_frames(box: ToolBox) -> None:
    result = box.call("get_frames", {"episode_id": "e000001", "steps": [0, 3], "camera": "scene"})
    assert not result.is_error
    kinds = [p["type"] for p in result.content]
    assert kinds == ["text", "image", "text", "image"]
    image = Image.open(io.BytesIO(base64.b64decode(result.content[1]["data"])))
    assert image.size == (16, 16)
    assert "logged frame at step 2" in result.content[2]["text"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"episode_id": "e000002", "steps": [0], "camera": "scene"}, "no video"),
        ({"episode_id": "e000099", "steps": [0], "camera": "scene"}, "no episode"),
        ({"episode_id": "e000001", "steps": list(range(9)), "camera": "scene"}, "at most 8"),
        ({"episode_id": "../run", "steps": [0], "camera": "scene"}, "invalid arguments"),
    ],
)
def test_get_frames_refusals(box: ToolBox, arguments: dict[str, Any], message: str) -> None:
    result = box.call("get_frames", arguments)
    assert result.is_error
    assert message in text(result)


def test_describe_with_and_without_the_logs(box: ToolBox) -> None:
    # One part per call, as compact JSON; the catalogue is a table.
    full = data(box.call("describe", {"part": "fleet", "include_logs": True}))
    static = data(box.call("describe", {"part": "fleet", "include_logs": False}))
    assert "change_log" in full["fleet"] and "change_log" not in static["fleet"]
    assert static["catalogue_entries"] > 0 and "catalogue" not in static
    dictionary = data(box.call("describe", {"part": "dictionary", "include_logs": False}))
    assert dictionary["dictionary"].startswith("# Data dictionary")
    everything = data(box.call("describe", {"part": "all", "include_logs": True}))
    assert {"dictionary", "fleet", "vocabulary", "costs"} <= set(everything)
    rows = data(box.call("query", {"sql": "SELECT count(*) AS n FROM catalogue"}))["rows"]
    assert rows == [[static["catalogue_entries"]]]
    missing = box.call("describe", {"include_logs": True})  # the part is required
    assert missing.is_error


def test_skills_are_listed_and_read(box: ToolBox) -> None:
    manifests = data(box.call("list_skills", {}))
    assert [m["name"] for m in manifests] == ["triage"]
    body = text(box.call("read_skill", {"name": "triage"}))
    assert body.startswith("---\nname: triage")
    missing = box.call("read_skill", {"name": "scope"})
    assert missing.is_error


def test_experiments_are_queued_with_ids_and_reported(box: ToolBox) -> None:
    first = Receipt.model_validate(data(box.call("request_experiment", EXPERIMENT)))
    second = Receipt.model_validate(data(box.call("request_experiment", EXPERIMENT)))
    assert (first.accepted, first.receipt_id) == (True, "x0001")
    assert second.receipt_id == "x0002"
    queued = data(box.call("experiment_results", {}))
    assert [(r["experiment_id"], r["status"]) for r in queued] == [
        ("x0001", "queued"),
        ("x0002", "queued"),
    ]
    result = {
        "experiment_id": "x0001",
        "status": "done",
        "arms": [
            {"arm_index": 0, "episode_ids": ["e000002"]},
            {"arm_index": 1, "episode_ids": ["e000004"]},
        ],
        "message": "done",
    }
    append_line(box.paths.results / RESULTS_FILE, result)
    # Observed episodes are not experiment arms, and these arms are too short.
    assert box.call("experiment_results", {}).is_error
    result = {
        "experiment_id": "x0001",
        "status": "refused",
        "arms": [],
        "message": "request_not_runnable",
    }
    (box.paths.results / RESULTS_FILE).write_text(json.dumps(result) + "\n")
    reported = data(box.call("experiment_results", {}))
    assert reported[0] == result
    assert reported[1]["status"] == "queued"


def test_experiment_refusals(tmp_path: Path, toy: Toy) -> None:
    box = ToolBox(analysis_paths(tmp_path, toy), limits=Limits(max_experiment_episodes=10))
    unknown = {**EXPERIMENT, "robots": ["r09"]}
    receipt = Receipt.model_validate(data(box.call("request_experiment", unknown)))
    assert not receipt.accepted
    assert "no robot 'r09'" in receipt.problems[0]
    teleop = json.loads(json.dumps(EXPERIMENT))
    teleop["arms"][1]["actions"] = [
        {
            "verb": "route_to_teleop",
            "target": {
                "scope": "fleet",
                "component": None,
                "config_key": None,
                "condition_key": None,
            },
            "value": "put",
        }
    ]
    assert not data(box.call("request_experiment", teleop))["accepted"]
    assert data(box.call("request_experiment", EXPERIMENT))["accepted"]  # 8 of 10 episodes
    over = Receipt.model_validate(data(box.call("request_experiment", EXPERIMENT)))
    assert not over.accepted
    assert "budget is 10 episodes" in over.problems[0]
    assert "at most 8 requests" in over.problems[0]  # the receipt says the limits
    assert len((box.paths.outbox / EXPERIMENTS_FILE).read_text().splitlines()) == 1
    # The historical default of 60 per arm is the tool server's, not the schema's.
    too_many = data(box.call("request_experiment", {**EXPERIMENT, "episodes_per_arm": 61}))
    assert not too_many["accepted"] and "at most 60 episodes per arm" in too_many["problems"][0]


def test_the_experiment_budget_of_the_contracts(tmp_path: Path, toy: Toy) -> None:
    box = ToolBox(analysis_paths(tmp_path, toy))
    assert (box.limits.max_experiment_requests, box.limits.max_experiment_episodes) == (8, 960)
    description = next(t.description for t in box.mcp_tools() if t.name == "request_experiment")
    assert "at most 8 requests and 960 episodes per analysis" in description
    assert "A robot-day holds at most 160 experiment-arm episodes" in description
    full = {**EXPERIMENT, "episodes_per_arm": 60}
    receipts = [data(box.call("request_experiment", full)) for _ in range(9)]
    assert [r["accepted"] for r in receipts] == [True] * 8 + [False]
    assert "budget is 8 requests" in receipts[8]["problems"][0]
    raised = ToolBox(
        analysis_paths(tmp_path / "raised", toy), limits=Limits(max_experiment_episodes=2000)
    )
    assert raised.limits.max_experiment_requests == 8


def test_an_arm_larger_than_60_runs_within_a_round_budget(tmp_path: Path, toy: Toy) -> None:
    """The loop limits an arm only by what is left of the round's attempts."""
    from fleetlab.loop.driver import AttemptBudget

    budget = AttemptBudget(attempts=200)
    chosen = budget.limits()
    limits = Limits(
        max_experiment_episodes=budget.attempts,
        max_experiment_requests=chosen["max_experiment_requests"],
        max_episodes_per_arm=chosen["max_episodes_per_arm"],
        budget_first_session=chosen["budget_first_session"],
    )
    box = ToolBox(analysis_paths(tmp_path, toy), limits=limits)
    big = data(box.call("request_experiment", {**EXPERIMENT, "episodes_per_arm": 80}))
    assert big == {"accepted": True, "receipt_id": "x0001", "problems": []}
    over = data(box.call("request_experiment", {**EXPERIMENT, "episodes_per_arm": 30}))
    assert (
        not over["accepted"]
        and "200 episodes per analysis; 160 are requested" in (over["problems"][0])
    )
    status = data(box.call("experiment_budget", {}))
    assert (status["attempts_used"], status["attempts_left"], status["max_per_arm"]) == (
        160,
        40,
        200,
    )
    box.close()


def test_a_runtime_sets_a_tighter_budget_and_a_later_first_session(
    tmp_path: Path, toy: Toy
) -> None:
    """The round driver's budget: fewer requests, fewer episodes per arm, and a count that
    starts at the round's first session (fleetlab.loop.driver)."""
    from fleetlab.toolserver.__main__ import build, parse

    paths = analysis_paths(tmp_path, toy)
    tight = Limits(max_experiment_requests=2, max_episodes_per_arm=5, max_experiment_episodes=40)
    box = ToolBox(paths, limits=tight)
    description = next(t.description for t in box.mcp_tools() if t.name == "request_experiment")
    assert "at most 2 requests and 40 episodes per analysis, at most 5 episodes per arm" in (
        description
    )
    wide = data(box.call("request_experiment", {**EXPERIMENT, "episodes_per_arm": 6}))
    assert not wide["accepted"] and "at most 5 episodes per arm" in wide["problems"][0]
    assert [data(box.call("request_experiment", EXPERIMENT))["accepted"] for _ in range(3)] == [
        True,
        True,
        False,
    ]
    box.close()
    # Requests of sessions before `budget_first_session` do not count: both requests above
    # were made in session 1, so a count from session 2 holds neither of them. (A real later
    # session also needs their results first; the driver's CPU test covers that path.)
    later = ToolBox(paths, session=1, limits=replace(tight, budget_first_session=2))
    receipt = data(later.call("request_experiment", EXPERIMENT))
    assert receipt == {"accepted": True, "receipt_id": "x0003", "problems": []}
    description = next(t.description for t in later.mcp_tools() if t.name == "request_experiment")
    assert "counted from session 2" in description
    later.close()
    # The command line carries every limit to the server inside the sandbox.
    args = parse(
        [
            *("--export", "e", "--memory", "m", "--outbox", "o", "--results", "r"),
            *("--max-experiment-requests", "2", "--max-episodes-per-arm", "5"),
            *("--budget-first-session", "3"),
        ]
    )
    limits = build(args).limits
    assert (limits.max_experiment_requests, limits.max_episodes_per_arm) == (2, 5)
    assert (limits.budget_first_session, limits.max_experiment_episodes) == (3, 960)


def test_submissions_are_checked_and_stored(box: ToolBox) -> None:
    receipt = data(box.call("submit_diagnosis", diagnosis()))
    assert receipt == {"accepted": True, "receipt_id": "diagnosis-1", "problems": []}
    assert json.loads((box.paths.outbox / "diagnosis.json").read_text()) == diagnosis()
    wrong = data(box.call("submit_diagnosis", diagnosis("e000099")))
    assert not wrong["accepted"]
    assert "e000099" in wrong["problems"][0]
    fix_set = {"actions": ranked_actions("r02", 10)}
    assert data(box.call("submit_fix_set", fix_set))["accepted"]
    short = box.call("submit_fix_set", {"actions": ranked_actions("r02", 9)})
    assert short.is_error and "invalid arguments" in text(short)  # ten at least
    elsewhere = data(box.call("submit_fix_set", {"actions": ranked_actions("r09", 10)}))
    assert not elsewhere["accepted"]
    assert "r09" in elsewhere["problems"][0]
    lines = (box.paths.outbox / "submissions.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["diagnosis", "fix_set"]


def ranked_actions(robot: str, n: int) -> list[dict[str, Any]]:
    """n distinct component actions on one robot, best first."""
    pairs = [
        (verb, component)
        for verb in ("replace", "service")
        for component in ("scene_camera", "wrist_camera", "gripper", "arm", "controller")
    ]
    return [
        {
            "action": {
                "verb": verb,
                "target": {
                    "scope": robot,
                    "component": component,
                    "config_key": None,
                    "condition_key": None,
                },
                "value": None,
            },
            "expected_gain": float(n - i),
            "uncertainty": {"low": 0.0, "high": float(n - i + 1)},
            "rationale": "-",
        }
        for i, (verb, component) in enumerate(pairs[:n])
    ]


def test_memory_round_trip_and_limit(tmp_path: Path, toy: Toy) -> None:
    box = ToolBox(analysis_paths(tmp_path, toy), limits=Limits(max_notes_bytes=10))
    assert data(box.call("memory_read", {})) == {"notes": ""}
    assert data(box.call("memory_write", {"notes": "r02 low"}))["accepted"]
    assert data(box.call("memory_read", {})) == {"notes": "r02 low"}
    assert not data(box.call("memory_write", {"notes": "x" * 11}))["accepted"]
    assert data(box.call("memory_read", {})) == {"notes": "r02 low"}


def test_the_call_log_holds_every_call_with_its_full_result(box: ToolBox) -> None:
    box.call("query", {"sql": "SELECT count(*) AS n FROM episodes"})
    box.call("get_frames", {"episode_id": "e000001", "steps": [0], "camera": "wrist"})
    box.call("query", {"sql": "DROP TABLE episodes"})
    log = [json.loads(line) for line in (box.paths.log or Path()).read_text().splitlines()]
    assert [e["tool"] for e in log] == ["query", "get_frames", "query"]
    assert [e["seq"] for e in log] == [1, 2, 3]
    assert log[0]["result"] == ['{"columns": ["n"], "rows": [[10]], "truncated": false}']
    assert len(log[1]["images"]) == 1
    assert log[2]["ok"] is False
    assert "DROP" in log[2]["error"]


def test_a_tool_not_enabled_is_neither_offered_nor_callable(tmp_path: Path, toy: Toy) -> None:
    box = ToolBox(analysis_paths(tmp_path, toy), enabled=["query", "describe"])
    assert [t.name for t in box.mcp_tools()] == ["query", "describe"]
    assert box.call("memory_write", {"notes": "x"}).is_error
    with pytest.raises(ValueError, match="unknown tools"):
        ToolBox(analysis_paths(tmp_path, toy), enabled=["read_truth"])


def test_every_contract_tool_is_served_with_a_strict_schema(box: ToolBox) -> None:
    tools = box.mcp_tools()
    offered = [spec.name for spec in TOOLS if spec.name not in OPT_IN_TOOLS]
    assert [t.name for t in tools] == offered
    assert list(tool_names()) == [spec.name for spec in TOOLS]
    for tool in tools:
        assert strict_schema_problems(tool.input_schema) == [], tool.name


# The protocol.


def test_mcp_handshake_list_and_call(box: ToolBox) -> None:
    server = StdioServer("robots", "1", box.mcp_tools())
    init = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert isinstance(init, dict)
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    newer = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2099-01-01"},
        }
    )
    assert isinstance(newer, dict)
    assert newer["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert isinstance(listed, dict)
    first = listed["result"]["tools"][0]
    assert first["name"] == "query"
    assert first["inputSchema"] == json_schema_strict(TOOLS[0].arguments)
    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "query", "arguments": {"sql": "SELECT 1 AS one"}},
        }
    )
    assert isinstance(called, dict)
    assert called["result"]["isError"] is False
    assert json.loads(called["result"]["content"][0]["text"])["rows"] == [[1]]
    unknown = server.handle({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
    assert isinstance(unknown, dict)
    assert unknown["error"]["code"] == -32601
    logged = StdioServer("robots", "1", box.mcp_tools(), fallback=box.call)
    hidden = logged.handle(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "read_truth", "arguments": {"path": "/"}},
        }
    )
    assert isinstance(hidden, dict) and hidden["result"]["isError"] is True
    last = json.loads((box.paths.log or Path()).read_text().splitlines()[-1])
    assert (last["tool"], last["ok"], last["arguments"]) == ("read_truth", False, {"path": "/"})
    batch = server.handle(
        [{"jsonrpc": "2.0", "id": 5, "method": "ping"}, {"jsonrpc": "2.0", "method": "x"}]
    )
    assert batch == [{"jsonrpc": "2.0", "id": 5, "result": {}}]


def _server_args(paths: Paths) -> list[str]:
    return [
        "--export", str(paths.export),
        "--memory", str(paths.memory),
        "--outbox", str(paths.outbox),
        "--results", str(paths.results),
        "--skills", str(paths.skills),
        "--log", str(paths.log),
    ]  # fmt: skip


def _talk(
    command: list[str], messages: list[dict[str, Any]], env: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    stdin = "".join(json.dumps(m) + "\n" for m in messages)
    done = subprocess.run(
        command, input=stdin, capture_output=True, text=True, timeout=120, env=env
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return [json.loads(line) for line in done.stdout.splitlines()]


SESSION: list[dict[str, Any]] = [
    {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "query", "arguments": {"sql": "SELECT count(*) FROM episodes"}},
    },
]


def test_the_server_speaks_on_stdin_and_stdout(tmp_path: Path, toy: Toy) -> None:
    paths = analysis_paths(tmp_path, toy)
    answers = _talk([sys.executable, "-m", "fleetlab.toolserver", *_server_args(paths)], SESSION)
    assert [a["id"] for a in answers] == [0, 1, 2]
    assert len(answers[1]["result"]["tools"]) == len(TOOLS) - len(OPT_IN_TOOLS)
    assert json.loads(answers[2]["result"]["content"][0]["text"])["rows"] == [[10]]


# What the tool server imports, and what its sandbox shows.


def test_the_import_closure_holds_no_truth_module() -> None:
    closure = import_closure()
    assert "fleetlab.toolserver.server" in closure
    assert "fleetlab.log.reader" in closure
    assert closure_problems(closure) == []
    for name in closure:
        assert not any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_MODULES), name


ALLOWED_IMPORTS = (
    "fleetlab.contracts",
    "fleetlab.log",
    "fleetlab.toolserver",
)


def test_the_tool_server_source_imports_only_contracts_and_the_log() -> None:
    for path in sorted(TOOLSERVER.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if name.startswith("fleetlab"):
                    assert name.startswith(ALLOWED_IMPORTS), f"{path.name} imports {name}"


def test_the_bundle_holds_the_closure_and_nothing_of_the_truth(tmp_path: Path) -> None:
    env = project_tool_env(tmp_path)
    files = sorted(p.relative_to(env.src).as_posix() for p in env.src.rglob("*.py"))
    assert "fleetlab/toolserver/server.py" in files
    assert not [
        f for f in files if f.startswith(("fleetlab/faults", "fleetlab/sim", "fleetlab/agents"))
    ]
    # Only the reader's side of the log package (review finding M6).
    for name in ("runfiles", "writer", "export", "leaks", "_rows", "_files"):
        assert f"fleetlab/log/{name}.py" not in files, name
    assert "fleetlab/video.py" not in files
    assert "fleetlab/log/reader.py" in files


def test_importing_the_log_package_loads_only_the_reader() -> None:
    code = (
        "import sys, fleetlab.log; "
        "print(sorted(m for m in sys.modules if m.startswith('fleetlab.log')))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    expected = "['fleetlab.log', 'fleetlab.log.dictionary', 'fleetlab.log.reader']"
    assert done.stdout.strip() == expected
    lazily = "import fleetlab.log as l; print(l.export_runs.__module__, l.RunWriter.__module__)"
    done = subprocess.run(
        [sys.executable, "-c", lazily], capture_output=True, text=True, check=True
    )
    assert done.stdout.split() == ["fleetlab.log.export", "fleetlab.log.writer"]


bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")


@bwrap
def test_the_server_runs_in_its_sandbox_and_cannot_reach_the_truth(
    tmp_path: Path, toy: Toy
) -> None:
    env = project_tool_env(tmp_path / "env", toy.skills)
    paths = analysis_paths(tmp_path / "analysis", toy)
    (tmp_path / "log").mkdir()
    mounts = ToolMounts(
        env=env,
        export=toy.export,
        memory=paths.memory,
        outbox=paths.outbox,
        results=paths.results,
        log_dir=tmp_path / "log",
    )
    command = tool_server_command(
        mounts, session=1, enabled=None, max_experiment_episodes=None, nested=False
    )
    canary_file = toy.run / "truth" / "canary.txt"
    attempts = [
        f"SELECT * FROM read_text('{canary_file}')",
        "SELECT * FROM read_text('/proc/self/environ')",
        "SELECT * FROM glob('/**')",
    ]
    calls: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 10 + i,
            "method": "tools/call",
            "params": {"name": "query", "arguments": {"sql": sql}},
        }
        for i, sql in enumerate(attempts)
    ]
    calls.append(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {"name": "memory_write", "arguments": {"notes": "inside"}},
        }
    )
    calls.append(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {"name": "list_skills", "arguments": {}},
        }
    )
    calls.append(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {"name": "read_truth", "arguments": {}},
        }
    )
    answers = _talk(command, [*SESSION, *calls], env={"PATH": "/usr/bin"})
    by_id = {a["id"]: a for a in answers}
    assert json.loads(by_id[2]["result"]["content"][0]["text"])["rows"] == [[10]]
    for i in range(len(attempts)):
        assert by_id[10 + i]["result"]["isError"] is True
    assert toy.canary not in json.dumps(answers)
    assert (paths.memory / "notes.md").read_text() == "inside"
    assert "triage" in by_id[21]["result"]["content"][0]["text"]
    log = (tmp_path / "log" / "tool_calls.jsonl").read_text()
    assert toy.canary not in log
    assert len(log.splitlines()) == 1 + len(attempts) + 3  # the call to read_truth too
    assert by_id[22]["result"]["isError"] is True
    assert command[:4] == ["/usr/bin/env", "-i", "PATH=/usr/bin", "LANG=C.UTF-8"]


def _mini_exports() -> list[Path]:
    from fleetlab.integration import fixtures

    return sorted(
        p / "export"
        for p in fixtures.FIXTURES_ROOT.glob("mini-v*")
        if (p / "export" / "log").is_dir()
    )


@pytest.mark.skipif(not _mini_exports(), reason="needs the miniature fleet's fixtures")
@pytest.mark.parametrize("export", _mini_exports(), ids=lambda p: p.parent.name)
def test_every_part_of_the_miniature_fleets_description_fits_one_tool_result(
    tmp_path: Path, export: Path
) -> None:
    """Claude Code refused the miniature fleet's whole description (92,874
    characters). Every part, with and without the logs, stays under the cap. mini-v1 was
    exported before the catalogue became a table: its description.json still holds the
    catalogue, which describe never returns, and no costs."""
    from fleetlab.contracts.tools import DESCRIBE_SIZE_CAP, DescribePart

    paths = Paths(
        export=export, memory=tmp_path / "m", outbox=tmp_path / "o", results=tmp_path / "r"
    )
    for d in (paths.memory, paths.outbox, paths.results):
        d.mkdir()
    box = ToolBox(paths)
    try:
        for part in DescribePart:
            for logs in (True, False):
                result = box.call("describe", {"part": part.value, "include_logs": logs})
                assert not result.is_error, (part, logs, text(result)[:200])
                assert len(text(result)) <= DESCRIBE_SIZE_CAP, (part, logs, len(text(result)))
                json.loads(text(result))
    finally:
        box.close()


def test_a_part_over_the_cap_is_refused_with_a_pointer(
    box: ToolBox, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.contracts.tools as contract

    small = len(text(box.call("describe", {"part": "dictionary", "include_logs": False})))
    monkeypatch.setattr(contract, "DESCRIBE_SIZE_CAP", small)
    assert not box.call("describe", {"part": "dictionary", "include_logs": False}).is_error
    refused = box.call("describe", {"part": "all", "include_logs": True})
    assert refused.is_error
    message = text(refused)
    assert "describe(part='all')" in message
    for pointer in ("dictionary, fleet, vocabulary, costs", "change_log", "catalogue"):
        assert pointer in message
    described = next(t.description for t in box.mcp_tools() if t.name == "describe")
    assert "at most 30,000 characters" in described and "catalogue" in described


@pytest.mark.parametrize(
    "verb,key,value,scope,reason",
    [
        ("set", "policy.precision", "not_a_real_precision", "r01", "fp16_res32"),
        ("set", "policy.precision", "fp32", "r01", "fixed for a loaded policy"),
        ("revert", "camera.scene.calibration", None, "r01", "no recorded change"),
        ("set", "control.mode", "sideways", "r01", "absolute or relative"),
        ("set", "control.action_delay_steps", -5, "r01", "nonnegative integer"),
        ("change_condition", "lighting", "bright", "s1", "finite signed stops"),
    ],
)
def test_invalid_actions_are_refused_before_debit(
    box: ToolBox,
    verb: str,
    key: str,
    value: Any,
    scope: str,
    reason: str,
) -> None:
    req = json.loads(json.dumps(EXPERIMENT))
    req["arms"][1]["actions"] = [
        {
            "verb": verb,
            "target": {
                "scope": scope,
                "component": None,
                "config_key": None if verb == "change_condition" else key,
                "condition_key": key if verb == "change_condition" else None,
            },
            "value": value,
        }
    ]
    receipt = data(box.call("request_experiment", req))
    assert not receipt["accepted"] and reason in " ".join(receipt["problems"])
    assert not (box.paths.outbox / EXPERIMENTS_FILE).exists()
    good = data(box.call("request_experiment", EXPERIMENT))
    assert good["accepted"] and good["receipt_id"] == "x0001"


def test_identical_arms_and_uncovered_action_are_refused(box: ToolBox) -> None:
    req = {**EXPERIMENT, "arms": [{"actions": []}, {"actions": []}]}
    assert "distinct" in " ".join(data(box.call("request_experiment", req))["problems"])
    req = json.loads(json.dumps(EXPERIMENT))
    req["arms"][1]["actions"][0]["target"]["scope"] = "r02"
    req["robots"] = ["r01"]
    assert "covers none" in " ".join(data(box.call("request_experiment", req))["problems"])
    assert not (box.paths.outbox / EXPERIMENTS_FILE).exists()


def test_all_actions_in_all_arms_are_checked(box: ToolBox) -> None:
    req = json.loads(json.dumps(EXPERIMENT))
    req["arms"].append(
        {
            "actions": [
                {
                    "verb": "set",
                    "target": {
                        "scope": "r01",
                        "config_key": "control.mode",
                        "component": None,
                        "condition_key": None,
                    },
                    "value": "sideways",
                }
            ]
        }
    )
    refused = data(box.call("request_experiment", req))
    assert not refused["accepted"] and "arm 2" in " ".join(refused["problems"])
    assert not (box.paths.outbox / EXPERIMENTS_FILE).exists()


def test_engine_refusal_releases_budget_without_reusing_ids(tmp_path: Path, toy: Toy) -> None:
    box = ToolBox(analysis_paths(tmp_path, toy), limits=Limits(max_experiment_requests=1))
    assert data(box.call("request_experiment", EXPERIMENT))["receipt_id"] == "x0001"
    assert not data(box.call("request_experiment", EXPERIMENT))["accepted"]
    append_line(
        box.paths.results / RESULTS_FILE,
        {
            "experiment_id": "x0001",
            "status": "refused",
            "arms": [],
            "message": "invalid_public_request",
        },
    )
    assert data(box.call("request_experiment", EXPERIMENT))["receipt_id"] == "x0002"
    box.close()


def test_public_component_glossary_is_served_for_existing_exports(box: ToolBox) -> None:
    stored = (box.paths.export / "log" / "description.json").read_bytes()
    for part in ("all", "vocabulary"):
        description = data(box.call("describe", {"part": part, "include_logs": False}))
        controller = next(
            c for c in description["vocabulary"]["components"] if c["name"] == "controller"
        )
        assert "outcome detector" in controller["description"]
        assert "control.mode" in description["vocabulary"]["experiment_value_domains"]
        assert "no_recorded_change" in description["vocabulary"]["experiment_refusal_codes"]
    assert (box.paths.export / "log" / "description.json").read_bytes() == stored


def test_invalid_stored_refusal_cannot_refund_budget(box: ToolBox) -> None:
    assert data(box.call("request_experiment", EXPERIMENT))["accepted"]
    append_line(
        box.paths.results / RESULTS_FILE,
        {"experiment_id": "x0001", "status": "refused", "arms": [], "message": "hidden detail"},
    )
    assert box.call("request_experiment", EXPERIMENT).is_error
    assert len((box.paths.outbox / EXPERIMENTS_FILE).read_text().splitlines()) == 1


def test_completion_and_recording_semantics_are_public_for_existing_exports(box: ToolBox) -> None:
    from fleetlab.contracts.env import PLACED_THRESHOLDS
    from fleetlab.toolserver.public_system import COMPLETION_CRITERION

    before = (box.paths.export / "data_dictionary.md").read_bytes()
    for part in ("all", "dictionary"):
        dictionary = data(box.call("describe", {"part": part, "include_logs": False}))["dictionary"]
        assert COMPLETION_CRITERION in dictionary
        assert "no body of the robot" in dictionary and "still held" in dictionary
        assert str(PLACED_THRESHOLDS["linear_m_s"]) + " m/s" in dictionary
        assert str(PLACED_THRESHOLDS["angular_rad_s"]) + " rad/s" in dictionary
        assert str(PLACED_THRESHOLDS["tail_steps"]) + " further control steps" in dictionary
        assert "record alone does not change physical success" in dictionary
    assert (box.paths.export / "data_dictionary.md").read_bytes() == before
