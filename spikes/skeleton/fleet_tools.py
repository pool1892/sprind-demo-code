"""The tool server of the walking skeleton: the only things the analysis agent can do.

An MCP server on stdin/stdout (see spikes/agent-runtime/README.md). The agent's
command-line tool starts it as a child process, inside a sandbox that shows it
the exported workspace and nothing else, and offers its two tools to the model:

    query(sql)                    one read-only SQL statement on the fleet log
    get_frames(episode, times)    camera frames of one episode as images

    python fleet_tools.py --workspace WORKSPACE [--log FILE]

WORKSPACE holds episodes.parquet, steps.parquet and video/<episode>.scene.mp4
and .wrist.mp4. This file is readable from inside the sandbox, so it contains
nothing about how the data was produced.

`query` is hardened in three ways. The two tables are copied into an in-memory
database at start-up, after which DuckDB's access to files and to the network
is switched off and that setting is locked. Only a single SELECT-type statement
is accepted. Results are cut to 200 rows and 40,000 characters.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

import av
import duckdb
from PIL import Image as PILImage

try:  # mcp 2.x
    from mcp.server.mcpserver import Image, MCPServer
except ImportError:  # mcp 1.x called the same class FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp import Image

MAX_ROWS = 200
MAX_CHARS = 40_000
MAX_FRAMES = 8
STEP_S = 0.05  # one control step
FRAME_EVERY = 4  # one stored frame every fourth control step

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", required=True)
parser.add_argument("--log", help="append one JSON line per tool call")
args = parser.parse_args()
WORKSPACE = Path(args.workspace).resolve()

mcp = MCPServer("robots")


def log_call(tool: str, arguments: dict, seconds: float, note: str = "") -> None:
    if not args.log:
        return
    with open(args.log, "a") as fh:
        fh.write(json.dumps({"at": time.time(), "tool": tool, "arguments": arguments,
                             "seconds": round(seconds, 4), "note": note}) + "\n")


def _open_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    for table in ("episodes", "steps"):
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{WORKSPACE / (table + '.parquet')}')")
    # From here on: no files, no network, no extensions, and the model cannot switch it back.
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


DB = _open_db()
EPISODES = {row[0]: row[1] for row in DB.execute("SELECT episode_id, n_steps FROM episodes").fetchall()}


@mcp.tool(structured_output=False)
def query(sql: str) -> str:
    """Run one read-only SQL statement (DuckDB dialect, SELECT only) on the fleet log.

    Table episodes, one row per episode: episode_id, robot_id, task_id, task (the
    instruction given to the robot), start_state_id (which of the stored object
    layouts of that task), n_steps, duration_s, success.

    Table steps, one row per control step (20 per second): episode_id, step, t_s
    (robot time in seconds), joint_pos_0..6 and joint_vel_0..6 (arm joints, rad
    and rad/s), gripper_pos_0..1 and gripper_vel_0..1 (finger joints, m and m/s),
    eef_x/y/z (measured gripper position in the world frame, m), eef_qx/qy/qz/qw
    (measured gripper orientation), cmd_x/y/z (the position the policy commanded
    at this step, world frame, m), cmd_rx/ry/rz (commanded orientation as an
    axis-angle vector, rad), cmd_gripper (-1 open, +1 close), chunk and chunk_step
    (which policy call the command came from and its position in that call's 30
    actions). The row of an episode's last state has no command.

    Returns JSON with `columns` and `rows`: at most 200 rows, so aggregate in SQL.
    """
    started = time.perf_counter()
    try:
        statements = duckdb.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("exactly one SELECT statement is allowed")
        cur = DB.cursor()
        try:
            cur.execute(statements[0])
            columns = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(MAX_ROWS + 1) if cur.description else []
        finally:
            cur.close()
        truncated = len(rows) > MAX_ROWS
        rows = [[round(v, 5) if isinstance(v, float) else v for v in row] for row in rows[:MAX_ROWS]]
        out = json.dumps({"columns": columns, "rows": rows, "truncated": truncated}, default=str)
        if len(out) > MAX_CHARS:
            out = json.dumps({"error": f"result has {len(out)} characters, the limit is {MAX_CHARS}; "
                                       "select fewer columns or rows, or aggregate"})
        log_call("query", {"sql": sql}, time.perf_counter() - started, f"{len(rows)} rows")
        return out
    except Exception as exc:
        log_call("query", {"sql": sql}, time.perf_counter() - started, f"error: {exc}")
        return json.dumps({"error": str(exc)})


def _decode(path: Path, wanted: set[int]) -> dict[int, PILImage.Image]:
    out = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_image()
            if i >= max(wanted):
                break
    return out


@mcp.tool(structured_output=False)
def get_frames(episode: str, times: list[float]) -> list:
    """Return camera frames of one episode, nearest to each time in seconds.

    Each returned image is 720x360: the fixed scene camera on the left, the wrist
    camera on the right. Frames are stored five times per second from t=0, plus
    the last instant of the episode (ask for a time beyond the end to get it).
    At most 8 times per call.
    """
    started = time.perf_counter()
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", episode or "") or episode not in EPISODES:
        log_call("get_frames", {"episode": episode, "times": times}, 0.0, "unknown episode")
        return [f"unknown episode: {episode!r}"]
    n_steps = EPISODES[episode]
    n_frames = n_steps // FRAME_EVERY + 1 + (1 if n_steps % FRAME_EVERY else 0)
    wanted = []
    for t in list(times)[:MAX_FRAMES]:
        step = max(0.0, float(t)) / STEP_S
        wanted.append(n_frames - 1 if step >= n_steps else min(n_frames - 1, int(round(step / FRAME_EVERY))))
    try:
        scene = _decode(WORKSPACE / "video" / f"{episode}.scene.mp4", set(wanted))
        wrist = _decode(WORKSPACE / "video" / f"{episode}.wrist.mp4", set(wanted))
    except Exception as exc:
        log_call("get_frames", {"episode": episode, "times": times}, time.perf_counter() - started, f"error: {exc}")
        return [f"could not read the video of {episode}: {exc}"]
    out: list = []
    for j in wanted:
        if j not in scene or j not in wrist:
            out.append(f"no frame {j} in the video of {episode}")
            continue
        both = PILImage.new("RGB", (scene[j].width + wrist[j].width, scene[j].height))
        both.paste(scene[j], (0, 0))
        both.paste(wrist[j], (scene[j].width, 0))
        buf = io.BytesIO()
        both.save(buf, format="JPEG", quality=88)
        t_frame = min(j * FRAME_EVERY, n_steps) * STEP_S
        out.append(f"{episode} at t={t_frame:.2f}s (scene camera left, wrist camera right):")
        out.append(Image(data=buf.getvalue(), format="jpeg"))
    log_call("get_frames", {"episode": episode, "times": times}, time.perf_counter() - started, f"{len(wanted)} frames")
    return out


if __name__ == "__main__":
    mcp.run()  # stdio transport
