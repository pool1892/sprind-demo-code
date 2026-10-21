"""The files the tool server reads and writes besides the export, and their formats.

One analysis of one world (by one agent configuration) owns a directory; the
runtime (`fleetlab.agents.sessions`) creates it and mounts its parts into the
tool sandbox:

    memory/notes.md                  the agent's notes (memory_write replaces them)    read-write
    outbox/experiments.jsonl         one line per accepted request_experiment          read-write
    outbox/submissions.jsonl         one line per accepted submission                  read-write
    outbox/diagnosis.json            the latest accepted diagnosis                     read-write
    outbox/fix_set.json              the latest accepted fix set                       read-write
    outbox/learned.jsonl             one line per accepted write_learned_skill         read-write
    outbox/probes.jsonl              one line per accepted request_probe               read-write
    results/experiments.jsonl        one ExperimentResult per line, by the runtime     read-only
    results/probes.jsonl             one ProbeOutcome per line, by the runtime         read-only
    learned/<name>/SKILL.md          a frozen library of learned skills, if given      read-only

The tool server writes the outbox and the memory; only the runtime writes the
results. Every write replaces a file through a temporary file and a rename, or
appends one line, so a killed session never leaves half a record.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "DIAGNOSIS_FILE",
    "EXPERIMENTS_FILE",
    "FIX_SET_FILE",
    "LEARNED_FILE",
    "NOTES_FILE",
    "PROBES_FILE",
    "RESULTS_FILE",
    "SUBMISSIONS_FILE",
    "append_line",
    "read_lines",
    "replace_text",
]

NOTES_FILE = "notes.md"  # in memory/
EXPERIMENTS_FILE = "experiments.jsonl"  # in outbox/
SUBMISSIONS_FILE = "submissions.jsonl"  # in outbox/
DIAGNOSIS_FILE = "diagnosis.json"  # in outbox/
FIX_SET_FILE = "fix_set.json"  # in outbox/
LEARNED_FILE = "learned.jsonl"  # in outbox/: learned skills, the latest line per name counts
RESULTS_FILE = "experiments.jsonl"  # in results/
PROBES_FILE = "probes.jsonl"  # in outbox/: accepted probe requests; in results/: outcomes


def read_lines(path: Path, *, strict: bool = False) -> Iterator[dict[str, Any]]:
    """The JSON objects of a JSON-lines file, none if it does not exist."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict):
                    yield record
                elif strict:
                    raise ValueError("JSON-lines record must be an object")


def append_line(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as one line, written with a single call."""
    data = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def replace_text(path: Path, text: str) -> None:
    """Replace a file's text through a temporary file in the same directory and a rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
