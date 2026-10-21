"""The agent runtime: sandboxed `claude -p` and `codex exec` calls with our tools.

See README.md in this directory. The tool server they use is `fleetlab.toolserver`.

| module | what |
|---|---|
| `run` | `run_agent`, `run_agent_with_retry`: one call, its answer, tokens, time, tool log |
| `sessions` | analyses in sessions, memory per world, the experiment queue for the engine |
| `batch` | `Batch`: the gate first, then calls at bounded concurrency |
| `checks` | the free pre-batch checks and the gate on the pinned CLI versions |
| `sandbox` | the two bubblewrap layouts and their static check |
| `commands` | the CLIs' command lines and environments |
| `toolenv` | the tool server's interpreter, packages, source bundle and skills |
| `cli` | pinned versions, the real binaries, the agent root |
| `wire_capture` | the local recorder the wire check points a CLI at |
| `toy` | the toy world of the checks and the measured tier |
| `measure` | `make check-agents`: real calls on the toy world, into README.md |
"""

from __future__ import annotations

from fleetlab.agents.run import AgentResult, ToolAccess, run_agent, run_agent_with_retry

__all__ = ["AgentResult", "ToolAccess", "run_agent", "run_agent_with_retry"]
