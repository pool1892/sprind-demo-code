"""Shared fixtures of the spike's test scripts: paths, the toy tool server, schemas."""

from __future__ import annotations

import os
from pathlib import Path

from agent_runtime import ToolServer

HERE = Path(__file__).resolve().parent
PROBE = Path(os.environ.get("ROBOTS_AGENT_PROBE", Path.home() / ".cache/robots-probes/agent-runtime"))
WORKSPACE = PROBE / "workspace"  # what the agent may see, through its tools
TRUTH = PROBE / "truth"  # what it must never see
CANARY = PROBE / "secret" / "canary.txt"
PYTHON = PROBE / "venv" / "bin" / "python"

TOOL_NAMES = ["query", "get_frames", "request_experiment"]


def tool_server(*extra_args: str) -> ToolServer:
    return ToolServer(
        script=HERE / "toy_tools.py",
        python=PYTHON,
        workspace=WORKSPACE,
        tools=TOOL_NAMES,
        extra_args=list(extra_args),
    )


# Strongest and cheapest tier of each vendor, as the two CLIs name them.
MODELS = {
    ("claude", "strong"): "opus",
    ("claude", "cheap"): "haiku",
    ("codex", "strong"): "gpt-6.1-sol",
    ("codex", "cheap"): "gpt-6-luna",
}

SYSTEM = (
    "You are an analysis agent for a fleet of robot arms. You work only through the tools "
    "you are given. Answer briefly and only with what you measured."
)

# ---- three reference tasks -------------------------------------------------

TRIVIAL_PROMPT = "Reply with the single word: pong"

IMAGES_PROMPT = (
    "Each of the attached camera frames shows a 3-digit code in large type. "
    "Return the codes in the order the frames were given."
)
IMAGES_SCHEMA = {
    "type": "object",
    "properties": {"codes": {"type": "array", "items": {"type": "integer"}}},
    "required": ["codes"],
    "additionalProperties": False,
}

SESSION_PROMPT = (
    "Do these three things with your tools, then answer.\n"
    "1. Query the fleet log: which robot has the lowest success rate, and which logged "
    "signal separates its failed episodes from the rest?\n"
    "2. Get the camera frames of episode e09 at t=2 and t=5 and read the 3-digit code "
    "printed in each frame.\n"
    "3. Request one experiment that would test your explanation and report what came back."
)
SESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "robot_id": {"type": "string"},
        "suspected_cause": {"type": "string"},
        "frame_codes": {"type": "array", "items": {"type": "integer"}},
        "experiment_id": {"type": "string"},
        "success_rate_after": {"type": "number"},
    },
    "required": ["robot_id", "suspected_cause", "frame_codes", "experiment_id", "success_rate_after"],
    "additionalProperties": False,
}


def frame_codes() -> dict:
    import json

    return json.loads((TRUTH / "frame_codes.json").read_text())


def eight_frames() -> list[Path]:
    return [WORKSPACE / "frames" / "e01" / f"{t}.png" for t in range(8)]


def check_images(result) -> bool:
    truth = frame_codes()["e01"]
    return bool(result) and result.get("codes") == [truth[str(t)] for t in range(8)]


def check_session(result) -> bool:
    truth = frame_codes()["e09"]
    return (
        bool(result)
        and result.get("robot_id") == "r3"
        and result.get("frame_codes") == [truth["2"], truth["5"]]
        and result.get("experiment_id") == "exp-0042"
        and abs(result.get("success_rate_after", 0) - 0.95) < 1e-6
    )
