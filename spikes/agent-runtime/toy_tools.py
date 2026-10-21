"""The toy tool server: the only things an analysis agent can do.

An MCP server (Model Context Protocol, the plug-in interface both `claude` and
`codex` speak) on stdin/stdout. The agent CLI starts it as a child process and
offers its three tools to the model:

    query(sql)                    read-only SQL on WORKSPACE/fleet.duckdb
    get_frames(episode, times)    camera frames as images
    request_experiment(spec)      a canned experiment result

Blinding lives here as much as in the sandbox. DuckDB can read any file the
process can (`read_text('/etc/passwd')`, `read_csv`, `ATTACH`, `COPY ... TO`),
so `query` switches external access off and locks the setting. `--unsafe`
leaves it on, only so the canary test can show the hole.

    python toy_tools.py --workspace WORKSPACE [--log FILE] [--unsafe]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

try:  # mcp 2.x
    from mcp.server.mcpserver import Image, MCPServer
except ImportError:  # mcp 1.x called the same class FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp import Image

MAX_ROWS = 200
MAX_FRAMES = 8

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", required=True)
parser.add_argument("--log", help="append one JSON line per tool call")
parser.add_argument("--unsafe", action="store_true", help="leave DuckDB file access on")
args = parser.parse_args()
WORKSPACE = Path(args.workspace).resolve()

mcp = MCPServer("robots")


def log_call(tool: str, arguments: dict, seconds: float, note: str = "") -> None:
    if not args.log:
        return
    with open(args.log, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "at": time.time(),
                    "tool": tool,
                    "arguments": arguments,
                    "seconds": round(seconds, 4),
                    "note": note,
                }
            )
            + "\n"
        )


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(WORKSPACE / "fleet.duckdb"), read_only=True)
    if not args.unsafe:
        # No reading or writing of other files, no extension downloads, and the
        # model cannot switch it back on.
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
    return con


@mcp.tool(structured_output=False)
def query(sql: str) -> str:
    """Run one read-only SQL statement (DuckDB dialect) on the fleet log.

    Tables: episodes(episode_id, robot_id, task, duration_s, success),
    steps(episode_id, t, gripper_force_n, ee_z_m, camera_latency_ms),
    events(episode_id, t, level, source, message).
    Returns JSON with `columns` and `rows`, at most 200 rows.
    """
    started = time.perf_counter()
    try:
        con = connect()
        try:
            cur = con.execute(sql)
            columns = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(MAX_ROWS + 1) if cur.description else []
        finally:
            con.close()
        truncated = len(rows) > MAX_ROWS
        out = json.dumps(
            {"columns": columns, "rows": rows[:MAX_ROWS], "truncated": truncated},
            default=str,
        )
        log_call("query", {"sql": sql}, time.perf_counter() - started)
        return out
    except Exception as exc:
        log_call("query", {"sql": sql}, time.perf_counter() - started, f"error: {exc}")
        return json.dumps({"error": str(exc)})


@mcp.tool(structured_output=False)
def get_frames(episode: str, times: list[float]) -> list:
    """Return camera frames of one episode, nearest to each time in seconds.

    Frames exist once per second from t=0 to t=7. At most 8 frames per call.
    """
    started = time.perf_counter()
    folder = (WORKSPACE / "frames" / episode).resolve()
    # The episode name comes from the model: keep it inside the frames folder.
    if folder.parent != (WORKSPACE / "frames").resolve() or not folder.is_dir():
        log_call("get_frames", {"episode": episode, "times": times}, 0.0, "unknown episode")
        return [f"unknown episode: {episode!r}"]
    available = sorted(int(p.stem) for p in folder.glob("*.png"))
    out: list = []
    for t in times[:MAX_FRAMES]:
        nearest = min(available, key=lambda a: abs(a - t))
        out.append(f"frame of {episode} at t={nearest}s:")
        out.append(Image(path=str(folder / f"{nearest}.png")))
    log_call(
        "get_frames",
        {"episode": episode, "times": times},
        time.perf_counter() - started,
    )
    return out


@mcp.tool(structured_output=False)
def request_experiment(spec: str) -> str:
    """Ask the simulator to rerun episodes with one thing changed.

    `spec` describes the change in a sentence. Returns the measured result.
    """
    started = time.perf_counter()
    result = {
        "experiment_id": "exp-0042",
        "spec_received": spec,
        "episodes_run": 20,
        "success_rate_before": 0.0,
        "success_rate_after": 0.95,
        "note": "canned result from the toy tool server",
    }
    log_call("request_experiment", {"spec": spec}, time.perf_counter() - started)
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run()  # stdio transport
