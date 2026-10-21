"""Start the tool server on stdin and stdout (see the package docstring)."""

from __future__ import annotations

import argparse
import faulthandler
import os
from collections.abc import Sequence
from pathlib import Path

from fleetlab.toolserver.mcp import StdioServer
from fleetlab.toolserver.server import SERVER_NAME, SERVER_VERSION, Limits, Paths, ToolBox

INSTRUCTIONS = (
    "Tools over one fleet's exported log. Every call is recorded. Requested experiments run "
    "after the session ends."
)


def parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.toolserver")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--outbox", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--skills", type=Path)
    parser.add_argument("--learned", type=Path, help="a frozen library of learned skills")
    parser.add_argument("--log", type=Path, help="append one JSON line per tool call")
    parser.add_argument("--tools", help="comma-separated tool names; default all")
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--max-experiment-episodes", type=int)
    parser.add_argument("--max-experiment-requests", type=int)
    parser.add_argument("--max-episodes-per-arm", type=int)
    parser.add_argument("--budget-first-session", type=int)
    return parser.parse_args(argv)


# The budget limits a runtime may set on the command line (`Limits` field names).
LIMIT_FLAGS = (
    "max_experiment_episodes",
    "max_experiment_requests",
    "max_episodes_per_arm",
    "budget_first_session",
)


def build(args: argparse.Namespace) -> ToolBox:
    paths = Paths(
        export=args.export,
        memory=args.memory,
        outbox=args.outbox,
        results=args.results,
        skills=args.skills,
        log=args.log,
        learned=args.learned,
    )
    enabled = [t for t in args.tools.split(",") if t] if args.tools else None
    chosen = {
        name: value for name in LIMIT_FLAGS if (value := getattr(args, name, None)) is not None
    }
    return ToolBox(paths, enabled, session=args.session, limits=Limits(**chosen))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse(argv)
    # MCP transports can otherwise discard a server's final stderr on failure.
    # Keep diagnostics beside the tool log, away from agent-readable tools.
    diagnostics = None
    if args.log is not None:
        diagnostics = args.log.with_name("server.stderr").open("a")
        os.dup2(diagnostics.fileno(), 2)
        faulthandler.enable(file=diagnostics)
    toolbox = build(args)
    try:
        server = StdioServer(
            SERVER_NAME, SERVER_VERSION, toolbox.mcp_tools(), INSTRUCTIONS, fallback=toolbox.call
        )
        server.serve()
    finally:
        toolbox.close()
        if diagnostics is not None:
            faulthandler.disable()
            diagnostics.close()


if __name__ == "__main__":
    main()
