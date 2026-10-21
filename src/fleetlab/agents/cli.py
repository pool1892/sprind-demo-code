"""The two agent CLIs: pinned versions, the real binaries, and where the runtime keeps its state.

Both CLIs replace themselves almost daily on this machine, and
the blinding rests on flag sets verified for one version each. So every call
names a fixed binary under mise's install directory, never `latest` and never
the launcher scripts in `~/.local/bin` (they run `mise use -g` on every
start), and a batch refuses to start on a version whose free checks have not
passed (`fleetlab.agents.checks`).

To move to a new version: change `PINNED`, run `make check-agents` (or
`python -m fleetlab.agents.checks`), read the result, commit.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

__all__ = [
    "BACKENDS",
    "PINNED",
    "Backend",
    "agent_root",
    "binary",
    "binary_sha256",
    "installed_version",
    "pinned_version",
    "root_problem",
]

Backend = Literal["claude", "codex"]
BACKENDS: tuple[Backend, ...] = ("claude", "codex")

# The versions the flag sets and sandboxes are verified for. An environment
# variable ROBOTS_CLAUDE_VERSION / ROBOTS_CODEX_VERSION overrides one for a
# trial; the gate still has to pass for it.
PINNED: dict[Backend, str] = {"claude": "2.1.293", "codex": "0.161.0"}

MISE_INSTALLS = Path.home() / ".local/share/mise/installs"


def agent_root() -> Path:
    """Where the runtime keeps its state: outside the repository and outside the home directory.

    `ROBOTS_AGENT_ROOT` overrides it. The default is `/var/tmp/robots-agents-<uid>`,
    created private to the user. Because `/var/tmp` is writable by everyone, an
    existing path is refused unless it is a directory, not a link, owned by this
    user and of mode 0700: someone else could have planted it first (review finding S3).
    """
    configured = os.environ.get("ROBOTS_AGENT_ROOT")
    root = Path(configured) if configured else Path("/var/tmp") / f"robots-agents-{os.getuid()}"
    if not root.is_symlink() and not root.exists():
        root.mkdir(mode=0o700, parents=True)
    problem = root_problem(root)
    if problem:
        raise PermissionError(f"agent root {root}: {problem}")
    return root


def root_problem(root: Path) -> str | None:
    """Why `root` cannot hold the runtime's state; None when it can."""
    if root.is_symlink():
        return "is a symbolic link"
    if not root.is_dir():
        return "is not a directory"
    info = root.stat()
    if info.st_uid != os.getuid():
        return f"is owned by uid {info.st_uid}, not {os.getuid()}"
    if stat.S_IMODE(info.st_mode) != 0o700:
        return f"has mode {stat.S_IMODE(info.st_mode):o}, not 700"
    return None


def pinned_version(backend: Backend) -> str:
    return os.environ.get(f"ROBOTS_{backend.upper()}_VERSION") or PINNED[backend]


def binary(backend: Backend, version: str | None = None) -> Path:
    """The real binary of a pinned version (not `latest`, not the launcher script)."""
    version = version or pinned_version(backend)
    if backend == "claude":
        path = MISE_INSTALLS / "claude" / version / "claude"
    else:
        path = MISE_INSTALLS / "codex" / version / "bin" / "codex"
    if path.is_symlink():
        raise ValueError(f"{path} is a link; the runtime calls real binaries only")
    return path


def installed_version(path: Path) -> str:
    """What the binary says its version is, run with an empty home so it touches no login."""
    with tempfile.TemporaryDirectory(prefix="cli-version-") as home:
        env = {
            "HOME": home,
            "PATH": "/usr/bin",
            "LANG": "C.UTF-8",
            "CODEX_HOME": home,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        }
        done = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, env=env, timeout=60
        )
    words = done.stdout.split()
    # "2.1.293 (Claude Code)" and "codex-cli 0.161.0"
    for word in words:
        if word[:1].isdigit():
            return word
    raise ValueError(f"{path} --version said {done.stdout!r} {done.stderr[-200:]!r}")


def binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
