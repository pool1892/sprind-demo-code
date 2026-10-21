"""What the two sandboxes show, and whether the tools work inside them. No model is called.

    python sandbox_check.py WORKSPACE MAIN_RUN_DIR

Builds the same two nested `bwrap` sandboxes that `run_agent(..., sandbox="bwrap")`
builds (through agent_runtime's own functions), and runs three things in them:

1. a plain process inside the tool sandbox: can it read the truth directory, the
   canary, this repository, the home directory, the login file, the network?
2. the same from the command-line tool's sandbox;
3. a real MCP session with the tool server through both sandboxes: list the tools,
   run a query, try three ways of reaching files through SQL, fetch frames.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import diagnose  # sets ROBOTS_AGENT_STATE before agent_runtime is imported
import agent_runtime as R

PROBE = r"""
import json, os, socket, sys
paths = json.loads(sys.argv[1])
out = {}
for name, p in paths.items():
    try:
        if os.path.isdir(p):
            out[name] = "LISTABLE: " + ",".join(sorted(os.listdir(p))[:6])
        else:
            with open(p, "rb") as f:
                out[name] = "READABLE: " + repr(f.read(24))
    except Exception as e:
        out[name] = "hidden (" + type(e).__name__ + ")"
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3).close()
    out["network"] = "OPEN"
except Exception as e:
    out["network"] = "closed (" + type(e).__name__ + ")"
out["/"] = sorted(os.listdir("/"))
out["/work"] = sorted(os.listdir("/work"))[:5]
out["/home"] = sorted(os.listdir("/home")) if os.path.isdir("/home") else "absent"
print(json.dumps(out))
"""

CLIENT = r"""
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

cmd = json.loads(sys.argv[1])

def show(res):
    out = []
    for c in res.content:
        if getattr(c, "type", "") == "text":
            out.append(c.text[:300])
        else:
            out.append(f"<{c.type} {getattr(c, 'mimeType', '')} {len(getattr(c, 'data', ''))} base64 chars>")
    return out

async def main():
    report = {}
    async with stdio_client(StdioServerParameters(command=cmd[0], args=cmd[1:])) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            report["tools"] = [t.name for t in tools.tools]
            first = None
            for name, sql in [
                ("count", "SELECT robot_id, count(*) AS episodes, sum(success::int) AS successes FROM episodes GROUP BY 1 ORDER BY 1"),
                ("first", "SELECT episode_id FROM episodes ORDER BY episode_id LIMIT 1"),
                ("steps", "SELECT count(*) AS n, count(cmd_x) AS with_command FROM steps"),
                ("read a file", "SELECT * FROM read_csv_auto('/etc/hostname')"),
                ("read text", "SELECT content FROM read_text('/proc/self/environ')"),
                ("write a file", "COPY episodes TO '/tmp/x.csv'"),
                ("unlock", "SET enable_external_access = true"),
                ("two statements", "SELECT 1; SELECT 2"),
                ("attach", "ATTACH '/work/episodes.parquet' AS x"),
            ]:
                res = await s.call_tool("query", {"sql": sql})
                report[name] = show(res)
                if name == "first":
                    first = json.loads(res.content[0].text)["rows"][0][0]
            res = await s.call_tool("get_frames", {"episode": first, "times": [0, 1.0, 999]})
            report["frames"] = show(res)
            res = await s.call_tool("get_frames", {"episode": "../../etc/passwd", "times": [0]})
            report["frames of a path"] = show(res)
    print(json.dumps(report))

asyncio.run(main())
"""


def main():
    ws, run_dir_truth = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    tools = diagnose.tool_server(ws)
    run_dir = R.STATE / "static" / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6])
    (run_dir / "cwd").mkdir(parents=True)
    (run_dir / "toolbox").mkdir()
    (run_dir / "client.py").write_text(CLIENT)
    outer = R._bwrap_prefix("claude", R.claude_bin(), tools, run_dir)
    inner = R._tool_bwrap(tools, f"{R.BOX_TOOLS}/{tools.script.name}", f"{R.BOX_RUN}/toolbox/tool_calls.jsonl")
    inner_head = inner[: inner.index("--") + 1]
    py = str(tools.python)
    repo = Path(__file__).resolve().parents[2]
    paths = {
        "truth directory": str(run_dir_truth / "truth"),
        "fault table": str(run_dir_truth / "truth" / "fault_table.json"),
        "canary": str(run_dir_truth / "truth" / "canary.txt"),
        "run directory": str(run_dir_truth),
        "skeleton cache": str(diagnose.BASE),
        "repository": str(repo),
        "repository AGENTS.md": str(repo / "AGENTS.md"),
        "home directory": str(Path.home()),
        "login file (real path)": str(Path.home() / ".claude" / ".credentials.json"),
        "login file (sandbox path)": f"{R.BOX_HOME}/.claude/.credentials.json",
        "prompt and output directory": R.BOX_RUN,
        "tool script": f"{R.BOX_TOOLS}/{tools.script.name}",
        "workspace": R.BOX_WORK,
    }
    report = {}
    for name, cmd in (
        ("tool sandbox", outer + inner_head + [py, "-c", PROBE, json.dumps(paths)]),
        ("command-line tool's sandbox", outer + [py, "-c", PROBE, json.dumps(paths)]),
        ("tool session through both sandboxes", outer + [py, f"{R.BOX_RUN}/client.py", json.dumps(inner)]),
    ):
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=run_dir / "cwd", env={"PATH": "/usr/bin"}, timeout=120)
        try:
            report[name] = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            report[name] = {"failed": p.returncode, "stdout": p.stdout[-500:], "stderr": p.stderr[-1500:]}
    print(json.dumps(report, indent=1))
    (diagnose.BASE / "sandbox_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
