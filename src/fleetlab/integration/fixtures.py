"""The miniature fleet's fixtures: the chain's files copied outside the repository, with
a manifest of hashes inside it.

The stage-4 agents develop against these files: the world, the runs, the export, the
analysis with every session's tool log, the oracle and the score. They live in
`FIXTURES_ROOT/<version>/` (outside the repository, by design) and the manifest
`fixtures/<version>.json` in the repository names every file with its size and the first
16 hexadecimal digits of its SHA-256 (a directory of many files as one entry over its
files' hashes), and a digest over the whole list, so that a copy can be checked
(`verify`) and a changed fixture shows in review.

Copied: everything under the chain's root but the run directories' writer lock files
and anything whose name starts with `.world-check`. Symbolic links are refused: a link
in a fixture could point into a login store.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FIXTURES_ROOT = Path.home() / ".cache" / "robots-fixtures"
MANIFEST_DIR = Path(__file__).resolve().parents[3] / "fixtures"
SKIP_NAMES = frozenset({".writer.lock"})
PARTS = (
    "world",
    "runs",
    "export",
    "oracle",
    "fresh",
    "score",
    "analyses",
    "agent-runs",
    "steps",
)


def _skip(path: Path) -> bool:
    return path.name in SKIP_NAMES or path.name.startswith(".world-check")


def _files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(_skip(Path(p)) for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"{path} is a symbolic link; fixtures hold none")
        if path.is_file():
            yield path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


GROUP_ABOVE = 40  # a directory holding more files than this directly is listed as one entry


def manifest_of(root: Path, version: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every file under `root` with its size and hash prefix, and a digest over the list.

    A directory that holds more than `GROUP_ABOVE` files directly (steps, videos, chains,
    staged rows) is one entry instead, `<dir>/*`: its file count, bytes and the SHA-256
    prefix over its sorted lines of name, size and file hash; this keeps the manifest in
    the repository small and still pins every byte.
    """
    every: dict[str, dict[str, Any]] = {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in _files(root)
    }
    by_dir: dict[str, list[str]] = {}
    for rel in every:
        by_dir.setdefault(rel.rsplit("/", 1)[0] if "/" in rel else "", []).append(rel)
    files: dict[str, dict[str, Any]] = {}
    for directory, members in by_dir.items():
        if len(members) <= GROUP_ABOVE:
            files.update({rel: every[rel] for rel in members})
            continue
        lines = "".join(
            f"{rel.rsplit('/', 1)[-1]}\t{every[rel]['bytes']}\t{every[rel]['sha256']}\n"
            for rel in sorted(members)
        )
        files[f"{directory}/*"] = {
            "files": len(members),
            "bytes": sum(int(every[rel]["bytes"]) for rel in members),
            "sha256": hashlib.sha256(lines.encode()).hexdigest()[:16],
        }
    listing = json.dumps(every, sort_keys=True, separators=(",", ":")).encode()
    by_part: dict[str, dict[str, int]] = {}
    for rel, info in every.items():
        part = rel.split("/", 1)[0]
        entry = by_part.setdefault(part, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += int(info["bytes"])
    return {
        "version": version,
        "root": f"~/.cache/robots-fixtures/{version}",
        **(extra or {}),
        "digest": hashlib.sha256(listing).hexdigest(),
        "parts": by_part,
        "files": dict(sorted(files.items())),
    }


def copy_fixtures(chain_root: Path, version: str, extra: dict[str, Any] | None = None) -> Path:
    """Copy the chain's files to the fixture directory and write the manifest; return its path.

    The fixture directory is replaced as a whole (copied beside it, then swapped), so a
    failed copy leaves the earlier fixtures in place.
    """
    dest = FIXTURES_ROOT / version
    stage = FIXTURES_ROOT / f".{version}.copy"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for part in PARTS:
        source = chain_root / part
        if not source.exists():
            continue
        for path in _files(source):
            target = stage / path.relative_to(chain_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    manifest = manifest_of(stage, version, extra)
    old = FIXTURES_ROOT / f".{version}.old"
    shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
    stage.rename(dest)
    shutil.rmtree(old, ignore_errors=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{version}.json"
    out.write_text(manifest_text(manifest))
    return out


def manifest_text(manifest: dict[str, Any]) -> str:
    """The manifest as JSON with one line per file, so that a diff shows the files changed."""
    head = {k: v for k, v in manifest.items() if k != "files"}
    lines = json.dumps(head, indent=1).rstrip("}").rstrip() + ',\n "files": {\n'
    entries = [
        f"  {json.dumps(path)}: {json.dumps(info, separators=(', ', ': '))}"
        for path, info in sorted(manifest["files"].items())
    ]
    return lines + ",\n".join(entries) + "\n }\n}\n"


def verify(version: str, root: Path | None = None) -> list[str]:
    """The differences between the fixture directory and its manifest; empty when they agree."""
    manifest = json.loads((MANIFEST_DIR / f"{version}.json").read_text())
    base = root or FIXTURES_ROOT / version
    actual = manifest_of(base, version)
    problems: list[str] = []
    want, have = manifest["files"], actual["files"]
    for rel in sorted(set(want) | set(have)):
        if rel not in have:
            problems.append(f"missing: {rel}")
        elif rel not in want:
            problems.append(f"not in the manifest: {rel}")
        elif want[rel] != have[rel]:
            problems.append(f"differs: {rel}")
    return problems
