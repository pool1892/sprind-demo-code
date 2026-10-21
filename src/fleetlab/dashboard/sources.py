"""What the progress page reads besides the runs: beads, the heavy-job queue, the nightly note.

Each reader returns its rows or, when the source cannot be read, a short
reason that the page prints instead. Nothing here writes anywhere: `bd list`
and `bd human list` are read-only, and the queue is read from `/proc`.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Bead", "Progress", "QueueJob", "read_beads", "read_human", "read_queue"]

GATE_TITLE = re.compile(r"^Gate \d+\b")
COMMAND_WIDTH = 160


@dataclass(frozen=True)
class Bead:
    id: str
    title: str
    status: str
    priority: int
    issue_type: str
    labels: tuple[str, ...] = ()
    closed_on: str = ""  # the date part of closed_at

    @property
    def is_gate(self) -> bool:
        return bool(GATE_TITLE.match(self.title))

    @staticmethod
    def from_json(row: dict[str, Any]) -> Bead:
        return Bead(
            id=str(row["id"]),
            title=str(row.get("title", "")),
            status=str(row.get("status", "")),
            priority=int(row.get("priority", 4)),
            issue_type=str(row.get("issue_type", "")),
            labels=tuple(sorted(row.get("labels") or ())),
            closed_on=str(row.get("closed_at") or "")[:10],
        )


@dataclass(frozen=True)
class QueueJob:
    """A job started through tools/heavy-run: running (holds the lock) or waiting for it."""

    pid: int
    state: str  # "running" or "waiting"
    command: str


@dataclass(frozen=True)
class Progress:
    """Everything the progress page shows besides the runs; a string where a source failed."""

    beads: list[Bead] | str
    human: list[Bead] | str
    queue: list[QueueJob] | str
    nightly: tuple[str, str] | None = None  # (file name, Markdown) of the last comparison


def _bd(arguments: list[str], cwd: Path | None) -> list[dict[str, Any]] | str:
    try:
        done = subprocess.run(
            ["bd", *arguments, "--json"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"bd could not be run: {error}"
    if done.returncode != 0:
        first = (done.stderr.strip().splitlines() or ["no message"])[0]
        return f"bd {' '.join(arguments)} failed: {first}"
    try:
        rows = json.loads(done.stdout or "[]")
    except json.JSONDecodeError as error:
        return f"bd {' '.join(arguments)} gave no JSON: {error}"
    if not isinstance(rows, list):
        return f"bd {' '.join(arguments)} gave {type(rows).__name__}, not a list"
    return rows


def read_beads(cwd: Path | None = None) -> list[Bead] | str:
    """Every bead, closed ones included (`bd list --all -n 0 --json`)."""
    rows = _bd(["list", "--all", "-n", "0"], cwd)
    return rows if isinstance(rows, str) else [Bead.from_json(r) for r in rows]


def read_human(cwd: Path | None = None) -> list[Bead] | str:
    """The beads waiting for a human (`bd human list --json`), closed ones left out."""
    rows = _bd(["human", "list"], cwd)
    if isinstance(rows, str):
        return rows
    return [b for b in (Bead.from_json(r) for r in rows) if b.status != "closed"]


def _cmdline(proc: Path, pid: int) -> list[str]:
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _children(proc: Path, pid: int) -> list[int]:
    try:
        text = (proc / str(pid) / "task" / str(pid) / "children").read_text()
    except OSError:
        return []
    return [int(p) for p in text.split()]


def _comm(proc: Path, pid: int) -> str:
    try:
        return (proc / str(pid) / "comm").read_text().strip()
    except OSError:
        return ""


def _job_command(argv: list[str]) -> str:
    """The command a heavy-run invocation runs: what follows its own options."""
    rest = argv[2:]
    while rest:
        if rest[0] in ("-m", "-g") and len(rest) > 1:
            rest = rest[2:]
        elif rest[0] == "--":
            rest = rest[1:]
            break
        else:
            break
    text = " ".join(rest)
    return text if len(text) <= COMMAND_WIDTH else text[: COMMAND_WIDTH - 3] + "..."


def read_queue(proc: Path = Path("/proc")) -> list[QueueJob] | str:
    """Heavy jobs of tools/heavy-run: the one running and those waiting for the lock.

    A heavy invocation stays a shell for the job's life (it watches GPU
    memory); while it waits for the lock its child is `flock`. A light
    invocation replaces itself with systemd-run and does not queue, so it is
    not listed. The lock is not first come, first served: the waiting jobs are
    listed by process id, which is not the order they will run in.
    """
    if not proc.is_dir():
        return "no /proc on this machine"
    jobs = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = _cmdline(proc, pid)
        if len(argv) < 2 or Path(argv[0]).name not in {"sh", "dash", "bash"}:
            continue
        if Path(argv[1]).name != "heavy-run" or "--light" in argv[2:3]:
            continue
        waiting = any(_comm(proc, child) == "flock" for child in _children(proc, pid))
        jobs.append(QueueJob(pid, "waiting" if waiting else "running", _job_command(argv)))
    return sorted(jobs, key=lambda j: (j.state != "running", j.pid))


def read_nightly(path: Path | None) -> tuple[str, str] | None:
    if path is None:
        return None
    return path.name, path.read_text()


def gather(
    cwd: Path | None, *, beads: bool = True, queue: bool = True, nightly: Path | None = None
) -> Progress:
    """Read every source; the ones switched off say so on the page."""
    off = "switched off for this build"
    return Progress(
        beads=read_beads(cwd) if beads else off,
        human=read_human(cwd) if beads else off,
        queue=read_queue() if queue else off,
        nightly=read_nightly(nightly),
    )
