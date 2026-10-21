"""The skill format (docs/design/contracts.md, section 13).

A skill is a directory `skills/<name>/` with `SKILL.md` (front matter and the
procedure) and optional helper scripts that run inside the tool sandbox. Skills
reach the agent through the tool server (`list_skills`, `read_skill`) or inline
in the system prompt, not through the CLIs' own skill loading.

Skills hold general procedures and general knowledge about fault classes only:
no identifiers or values of a specific world. `validate_skill_general` checks
that mechanically: no neutral identifier of any kind (`r07`, `e000123`, ...)
and no term from a truth vocabulary the caller supplies (fault kinds, values).
It finds names, not every value a skill could encode; review covers the rest.

Choices this module makes where the note is silent:

- The front matter keys are `name`, `description`, `when_to_use`,
  `tools_used`, `inputs`, `outputs`, `version`; the name is lower case with
  hyphens and equals the directory's name.
- The front matter is a restricted YAML subset, parsed here so that no YAML
  library is needed: `key: value` scalars (optionally quoted, continued on
  indented lines), `[a, b]` flow lists and `- item` block lists.
- `tools_used` must name tools of `contracts.tools.TOOLS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.ids import ID_PATTERNS

SKILL_FILE = "SKILL.md"
SKILL_NAME_PATTERN = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"
_LIST_KEYS = frozenset({"tools_used", "inputs", "outputs"})


class SkillManifest(ContractModel):
    """The front matter of a skill's SKILL.md.

    A learned skill (`contracts.learned`, docs/design/loop.md section 9) adds
    `origin: learned`, `applies_when` and `provenance`; a base skill leaves them
    out, and its manifest dumps as before with `exclude_none`.
    """

    name: Annotated[str, Field(pattern=SKILL_NAME_PATTERN, max_length=64)]
    description: Annotated[str, Field(min_length=1, max_length=1024)]
    when_to_use: str
    tools_used: list[str]
    inputs: list[str]
    outputs: list[str]
    version: str
    origin: Literal["learned"] | None = None
    applies_when: str | None = None
    provenance: str | None = None


@dataclass(frozen=True)
class Skill:
    """A loaded skill: its manifest, the procedure text, and the helper files beside it."""

    root: Path
    manifest: SkillManifest
    body: str
    helpers: tuple[Path, ...]


def parse_front_matter(text: str) -> tuple[dict[str, str | list[str]], str]:
    """Split a SKILL.md into its front matter (as a dict) and its body.

    Raises ValueError for a missing front matter or a line outside the subset.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{SKILL_FILE} starts with a front matter between --- lines")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("the front matter is not closed by a --- line") from None
    values: dict[str, str | list[str]] = {}
    key: str | None = None
    for number, line in enumerate(lines[1:end], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if match:
            key, rest = match[1], match[2].strip()
            if key in values:
                raise ValueError(f"line {number}: {key} appears twice")
            if rest.startswith("[") and rest.endswith("]"):
                values[key] = [_unquote(item) for item in rest[1:-1].split(",") if item.strip()]
            elif rest:
                values[key] = _unquote(rest)
            else:
                values[key] = []
            continue
        item = re.match(r"^\s+-\s+(.*)$", line)
        if item and key is not None and isinstance(values[key], list):
            current = values[key]
            assert isinstance(current, list)
            current.append(_unquote(item[1]))
            continue
        if line.startswith((" ", "\t")) and key is not None and isinstance(values[key], str):
            values[key] = f"{values[key]} {line.strip()}"
            continue
        raise ValueError(f"line {number}: not a front matter line: {line!r}")
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return values, body


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_skill(directory: Path | str) -> Skill:
    """Read a skill directory and validate its manifest. Raises ValueError when it is invalid."""
    from fleetlab.contracts.tools import TOOLS  # tools lists skills, so it imports this module

    root = Path(directory)
    values, body = parse_front_matter((root / SKILL_FILE).read_text())
    for key in _LIST_KEYS:
        if isinstance(values.get(key), str):
            values[key] = [str(values[key])]
    manifest = SkillManifest.model_validate(values)
    if manifest.name != root.name:
        raise ValueError(f"skill {manifest.name!r} lives in a directory named {root.name!r}")
    known = {tool.name for tool in TOOLS}
    unknown = sorted(set(manifest.tools_used) - known)
    if unknown:
        raise ValueError(f"skill {manifest.name} uses tools that do not exist: {unknown}")
    helpers = tuple(sorted(p for p in root.rglob("*") if p.is_file() and p.name != SKILL_FILE))
    return Skill(root=root, manifest=manifest, body=body, helpers=helpers)


# A neutral identifier of any kind, as a word: the letter and the digits of
# `contracts.ids.ID_PATTERNS` without the anchors.
_ID_WORD = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(p[1:-1] for p in ID_PATTERNS.values()) + r")(?![A-Za-z0-9_])"
)


def neutral_identifiers(text: str) -> list[str]:
    """Every word of `text` that has the form of a neutral identifier (`r07`, `s2`, `x0001`)."""
    return [match[1] for match in _ID_WORD.finditer(text)]


def validate_skill_general(
    directory: Path | str, forbidden_terms: list[str] | tuple[str, ...] = ()
) -> list[str]:
    """Why a skill is not general, as a list of problems; empty when it passes.

    Fails on an invalid manifest, on any neutral identifier (a world's robots,
    sites, episodes, faults, experiments) and on any of `forbidden_terms` (the
    truth vocabulary: fault kinds, parameter names, values), in every text file
    of the skill. Terms match case-insensitively, as whole words where they
    begin and end with a letter or digit.
    """
    root = Path(directory)
    problems: list[str] = []
    try:
        load_skill(root)
    except (OSError, ValueError) as error:
        problems.append(f"{root.name}: {error}")
    patterns = [(term, _term_pattern(term)) for term in forbidden_terms if term.strip()]
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue  # not a text file
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(text.splitlines(), start=1):
            for match in _ID_WORD.finditer(line):
                problems.append(f"{relative}:{number}: identifier {match[1]!r}")
            for term, pattern in patterns:
                if pattern.search(line):
                    problems.append(f"{relative}:{number}: forbidden term {term!r}")
    return problems


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    start = r"(?<![A-Za-z0-9_])" if term.strip()[0].isalnum() else ""
    end = r"(?![A-Za-z0-9_])" if term.strip()[-1].isalnum() else ""
    return re.compile(start + escaped + end, re.IGNORECASE)
