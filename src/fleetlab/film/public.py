"""What the public site may say about runs and files, and what it must never contain.

The film's full provenance names local paths; it stays with the records. The site gets
stable identifiers instead (`public_ref`): a run or file under the results roots is named by
its path below the root, with the root as a label (`probes:` for
`~/.cache/robots-probes`, `loop:` for `/var/tmp/robots-loop`), a file in a checkout of this
repository by its path in the repository (`repo:`), anything else by its name and a digest
of its full path (`file:`). With the SHA-256 the provenance records, an identifier finds the
file again on this machine without the site saying where it lies.

`scan` finds what must not be published: home directories and other local paths, this
machine's host name as part of an address, the tailnet and the review hub, tracker ids, e-mail
addresses. The site build refuses a bundle in which any of them occurs.

A cut may add its own rules (`configure`, from a manifest's or a site description's
`public` object): `aliases` replace a private name inside an identifier with its public
one (a fleet's internal world id in a run directory's name by the fleet's public id), and
`forbid` lists patterns that must appear nowhere: not in the site's files, and not in any
text the film draws (`forbidden`).
"""

from __future__ import annotations

import hashlib
import re
import socket
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Hit",
    "configure",
    "forbidden",
    "public_ref",
    "public_rules",
    "scan",
    "scan_tree",
    "short_ref",
]

_ALIASES: list[tuple[re.Pattern[str], str]] = []
_FORBID: list[tuple[str, re.Pattern[str]]] = []


def configure(aliases: dict[str, str] | None = None, forbid: list[str] | None = None) -> None:
    """Set the cut's own rules (replacing earlier ones): `aliases` map a private name to its
    public one inside identifiers, as a whole word; `forbid` are regular expressions."""
    _ALIASES.clear()
    _FORBID.clear()
    for private, public in (aliases or {}).items():
        _ALIASES.append(
            (re.compile(rf"(?<![A-Za-z0-9]){re.escape(private)}(?![A-Za-z0-9])"), public)
        )
    for pattern in forbid or []:
        _FORBID.append((f"a name this cut forbids ({pattern})", re.compile(pattern)))


def public_rules(spec: object) -> tuple[dict[str, str], list[str]]:
    """`aliases` and `forbid` of a `public` object, checked."""
    if spec is None:
        return {}, []
    if not isinstance(spec, dict) or set(spec) - {"aliases", "forbid"}:
        raise ValueError("public: an object with aliases and forbid")
    aliases, forbid = spec.get("aliases") or {}, spec.get("forbid") or []
    if not isinstance(aliases, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
    ):
        raise ValueError("public.aliases: private name to public name")
    if not isinstance(forbid, list) or not all(isinstance(f, str) for f in forbid):
        raise ValueError("public.forbid: a list of regular expressions")
    for f in forbid:
        try:
            re.compile(f)
        except re.error as error:
            raise ValueError(
                f"public.forbid: {f!r} is not a regular expression ({error})"
            ) from None
    return dict(aliases), list(forbid)


def forbidden(text: str) -> list[str]:
    """What of the cut's forbidden names occurs in `text`."""
    return [m.group(0) for _, pattern in _FORBID for m in pattern.finditer(text)]


def _alias(ref: str) -> str:
    for pattern, public in _ALIASES:
        ref = pattern.sub(public, ref)
    return ref


ROOTS = (
    (Path.home() / ".cache" / "robots-probes", "probes"),
    (Path("/var/tmp/robots-loop"), "loop"),
)
# A checkout of this repository: the main one or a worktree beside it (`robots-wt-*`).
_CHECKOUT = re.compile(r"/code/robots(?:-[A-Za-z0-9_-]+)?/(.+)$")

TEXT_SUFFIXES = {
    ".html", ".htm", ".json", ".jsonl", ".md", ".txt", ".vtt", ".css", ".js", ".sql",
    ".csv", ".tsv", ".yaml", ".yml", ".xml", ".svg", "",
}  # fmt: skip


def public_ref(path: str | Path) -> str:
    """A stable identifier for a run, table or file that names no local path."""
    raw = str(path)
    expanded = Path(raw).expanduser()
    for root, label in ROOTS:
        try:
            return _alias(f"{label}:{expanded.relative_to(root).as_posix()}")
        except ValueError:
            continue
    match = _CHECKOUT.search(expanded.as_posix())
    if match:
        return _alias(f"repo:{match.group(1)}")
    digest = hashlib.sha256(expanded.as_posix().encode()).hexdigest()[:8]
    return _alias(f"file:{expanded.name}#{digest}")


def short_ref(path: str | Path, parts: int = 2) -> str:
    """The identifier's label and its last `parts` components, for a line on screen."""
    ref = public_ref(path)
    label, _, rest = ref.partition(":")
    pieces = rest.split("/")
    return f"{label}:{'/'.join(pieces[-parts:])}" if len(pieces) > parts else ref


def _host_patterns() -> list[tuple[str, re.Pattern[str]]]:
    host = socket.gethostname().split(".")[0].lower()
    if not host:
        return []
    h = re.escape(host)
    return [
        (
            "this machine's host name",
            re.compile(rf"(?i)(?://|@|\b){h}(?:\.[a-z0-9-]+)+\b|//{h}[:/]"),
        ),
    ]


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("a home directory", re.compile(r"/home/[^\s\"'<>]*|/Users/[^\s\"'<>]+")),
    ("a local path", re.compile(r"(?<![\w.])(?:/var/tmp|/tmp|/run/user|/root)/[^\s\"'<>]*")),
    ("a home-relative path", re.compile(r"(?<![\w/])~/[^\s\"'<>]+")),
    ("the tailnet", re.compile(r"(?i)[a-z0-9-]+\.ts\.net|\btail[0-9a-f]{6}\b")),
    ("the review hub", re.compile(r"(?i)\bhub\.[a-z0-9-]+\.[a-z]|\bhub (?:publish|batch|check)\b")),
    ("a tracker id", re.compile(r"\brobots-[a-z0-9]{2,4}(?:\.[0-9]+)?\b(?!-)")),
    ("an e-mail address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*[a-z]{2,}")),
]


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    what: str
    text: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.what}: {self.text}"


def scan(text: str, name: str = "") -> list[Hit]:
    hits = []
    patterns = PATTERNS + _host_patterns() + _FORBID
    for number, line in enumerate(text.splitlines(), 1):
        for what, pattern in patterns:
            for match in pattern.finditer(line):
                hits.append(Hit(name, number, what, match.group(0)[:80]))
    return hits


def scan_tree(root: Path) -> list[Hit]:
    """Every text file below `root`; binary files (video, images) are not read."""
    hits: list[Hit] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += scan(text, path.relative_to(root).as_posix())
    return hits
