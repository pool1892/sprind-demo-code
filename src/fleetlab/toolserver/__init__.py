"""The agents' tool server: the tools of docs/design/contracts.md section 12, as an MCP server.

It runs inside the inner sandbox of `fleetlab.agents` with the export
mounted read-only and no network, and its source is readable there. It
therefore depends on `fleetlab.contracts` (tools, vocabulary, skills, ids)
and `fleetlab.log` only, and never imports `fleetlab.faults` or anything else
that holds truth; `tests/test_toolserver.py` checks the import graph.

    python -m fleetlab.toolserver --export DIR --memory DIR --outbox DIR --results DIR \
        [--skills DIR] [--log FILE] [--tools query,describe,...] [--session N]

| module | what |
|---|---|
| `mcp` | the part of the Model Context Protocol the CLIs use, on stdin and stdout |
| `server` | `ToolBox`: the eleven tools, the call log |
| `files` | the analysis directory's files (memory, outbox, results) and their formats |
| `jobs` | the opt-in job tools: the request, acceptance, programs, checks, library, account |
"""

from __future__ import annotations
