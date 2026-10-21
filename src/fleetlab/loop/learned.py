"""Retained engineering learning: learned skills kept, checked, frozen and mounted.

docs/design/loop.md, section 9. During an investigation the analysis agent
writes diagnostic skills with the tool `write_learned_skill`
(`contracts.learned`); the tool server checks each draft for what needs no
truth and appends it to the analysis' outbox (`outbox/learned.jsonl`, the
latest line per name counts). After the investigation the host:

1. reads the drafts (`stored_drafts`);
2. renders each into the skill format with its provenance: a label for the
   fleet (never its identifier), the round, the session and the day it was
   kept (`provenance_line`);
3. checks each rendered skill against the truth side of the fleet it was
   written on (`fleetlab.agents.audit.audit_learned_skill`: the agent-text
   leak rules, no neutral identifier, no world-specific term, the learned
   front matter and sections), refusing any that fails;
4. freezes the skills that pass into a library (`freeze`): a directory named
   by its digest, with `skills/<name>/SKILL.md` (read-only files, the part
   that is mounted) and `library.json` beside it (the host's record: digest,
   sources with their world ids, checks; never mounted). A skill that carries
   a probe (`contracts.learned.LearnedProbe`) also gets its recipe,
   `recipes/<name>.json` beside `skills/`: the validated probe, host side and
   never mounted (a mounted learned skill holds its `SKILL.md` and nothing
   else, `agents.audit.audit_learned_skill`), with its sha256 in the record; the
   `SKILL.md` renders the same probe for the agent. `FrozenLibrary.recipe` reads
   it back for the command that runs it on named robots (`fleetlab.loop.probes`).

A later investigation receives the library when its analysis is created
(`Analysis.create(..., learned=library.skills)`); every session mounts it
read-only at `/learned`, the launch audit checks it again against the truth
side of that later fleet, and `list_skills` and `read_skill` serve it after
the unchanged base skills, with `origin`, `applies_when` and `provenance`
in each manifest. A library can carry the skills of an earlier library
forward (`previous`); a new draft of the same name replaces the old one.

    report = freeze([Source(analysis_root, "fleet A", 1, runs, fleet)], parent)
    library = report.library                  # FrozenLibrary, or None if nothing passed
    later = Analysis.create(world_id, base=..., learned=library.skills)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.agents.audit import audit_learned_skill, tree_digest
from fleetlab.contracts.learned import (
    LearnedProbe,
    LearnedSkillDraft,
    draft_problems,
    render_learned_skill,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.skills import SKILL_FILE, neutral_identifiers
from fleetlab.toolserver.files import LEARNED_FILE, read_lines

__all__ = [
    "LEARNING_INSTRUCTIONS",
    "LIBRARY_FORMAT",
    "RECIPES_DIR",
    "RECIPE_FORMAT",
    "FreezeReport",
    "FrozenLibrary",
    "LearnedRefused",
    "Source",
    "StoredDraft",
    "base_skill_names",
    "freeze",
    "provenance_line",
    "stored_drafts",
    "with_learning",
]

LIBRARY_FORMAT = "fleetlab.learned-library/1"
LIBRARY_RECORD = "library.json"
RECIPES_DIR = "recipes"  # beside skills/: each probe-carrying skill's recipe (host side)
RECIPE_FORMAT = "fleetlab.probe-recipe/1"

# The paragraph of the agent's instructions that says when and how to write a
# learned skill. The task prompt carries it (`with_learning`), in both arms of
# a reuse comparison alike.
LEARNING_INSTRUCTIONS = """\
Keep what you learn for the next investigation. Once the evidence from this fleet \
has established an explanation, or ruled one out, write it down with \
write_learned_skill as a competent engineer would want it the next time a similar \
fleet shows a similar pattern: when it applies (the pattern in the records, in \
general terms), the queries that showed it (SQL with parameters such as $site_id, \
and what each returned here), the competing explanations and the observation that \
tells them apart, and the experiment that settles the question. Name no robot, \
site, episode or experiment of this fleet, because the next fleet has other ones: \
make them parameters. Write it before you finish; to revise it, write it again \
under the same name. The host checks it and keeps it after the investigation. If \
list_skills shows a skill with origin learned, it was written in an earlier \
investigation of another fleet: read its provenance, and test whether it applies \
here before you rely on it."""


def with_learning(task: str) -> str:
    """A task prompt with the learning paragraph appended."""
    return f"{task.rstrip()}\n\n{LEARNING_INSTRUCTIONS}\n"


class LearnedRefused(ValueError):
    """A learned skill failed the host's check; the message lists why (host-only text)."""


@dataclass(frozen=True)
class Source:
    """One investigation whose learned skills go into a library, with its truth side."""

    analysis: Path  # the analysis directory (its outbox holds the drafts)
    fleet_label: str  # how the provenance names the fleet: a label, never its identifier
    round: int | None  # the round of the loop it belongs to, if any
    runs: tuple[Path, ...]  # the fleet's runs with their truth side, for the leak rules
    fleet: Fleet  # the full fleet model, for the leak rules
    # An engagement's round of each session (`fleetlab.loop.reuse_run.engagement_source`);
    # a draft's provenance then names the round its session belonged to.
    session_rounds: Mapping[int, int] | None = None

    def round_of(self, session: int) -> int | None:
        if self.session_rounds is not None and session in self.session_rounds:
            return self.session_rounds[session]
        return self.round


@dataclass(frozen=True)
class StoredDraft:
    """The latest accepted draft of one learned skill in an analysis' outbox."""

    name: str
    receipt_id: str
    session: int
    draft: LearnedSkillDraft


def stored_drafts(analysis: Path) -> dict[str, StoredDraft]:
    """The latest accepted draft per name, in the order the names were first written."""
    out: dict[str, StoredDraft] = {}
    for record in read_lines(analysis / "outbox" / LEARNED_FILE, strict=True):
        draft = LearnedSkillDraft.model_validate(record["draft"])
        out[draft.name] = StoredDraft(
            draft.name, str(record["receipt_id"]), int(record["session"]), draft
        )
    return out


def provenance_line(source: Source, stored: StoredDraft, kept_on: str) -> str:
    """Where a learned skill comes from, as the later agent reads it."""
    round_number = source.round_of(stored.session)
    round_text = f", round {round_number}" if round_number is not None else ""
    return (
        f"learned by the analysis agent on {source.fleet_label}{round_text}, session "
        f"{stored.session}; checked and kept on {kept_on}"
    )


def base_skill_names(skills: Path | None = None) -> frozenset[str]:
    """The names a learned skill may not take: every skill of the base library."""
    from fleetlab.skill_library import ROLES, library_dir

    names = {name for role in ROLES.values() for name in role}
    root = skills if skills is not None else library_dir()
    if root.is_dir():
        names |= {p.name for p in root.iterdir() if (p / SKILL_FILE).is_file()}
    return frozenset(names)


@dataclass(frozen=True)
class FrozenLibrary:
    """A frozen library: `skills/` is mounted, `library.json` is the host's record."""

    root: Path

    @property
    def skills(self) -> Path:
        return self.root / "skills"

    @property
    def record(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.root / LIBRARY_RECORD).read_text())
        return data

    @property
    def digest(self) -> str:
        return str(self.record["digest"])

    @property
    def names(self) -> list[str]:
        return [skill["name"] for skill in self.record["skills"]]

    def verify(self) -> FrozenLibrary:
        """Raise ValueError when the mounted part no longer has the recorded digest, or a
        recipe no longer the recorded sha256."""
        if self.record.get("format") != LIBRARY_FORMAT:
            raise ValueError(f"{self.root}: not a learned library")
        if tree_digest(self.skills) != self.digest:
            raise ValueError(f"{self.root}: the skills changed since they were frozen")
        for entry in self.record["skills"]:
            recipe = entry.get("recipe")
            if recipe is not None and _sha256(self.root / recipe["path"]) != recipe["sha256"]:
                raise ValueError(f"{self.root}: the recipe of {entry['name']} changed")
        return self

    def recipe(self, name: str) -> LearnedProbe | None:
        """The probe a skill of the library carries, validated; None if it carries none."""
        entry = next((e for e in self.record["skills"] if e["name"] == name), None)
        if entry is None:
            raise KeyError(f"no skill {name!r} in {self.root} (it has {self.names})")
        recipe = entry.get("recipe")
        if recipe is None:
            return None
        path = self.root / recipe["path"]
        if _sha256(path) != recipe["sha256"]:
            raise ValueError(f"{self.root}: the recipe of {name} changed")
        data = json.loads(path.read_text())
        if data.get("format") != RECIPE_FORMAT or data.get("skill") != name:
            raise ValueError(f"{path}: not the probe recipe of {name}")
        return LearnedProbe.model_validate(data["probe"])

    @classmethod
    def load(cls, root: Path) -> FrozenLibrary:
        return cls(Path(root)).verify()


@dataclass
class FreezeReport:
    """What a freeze kept and refused; refusals carry the host's reasons (never mounted)."""

    library: FrozenLibrary | None
    kept: list[str] = field(default_factory=list)
    carried: list[str] = field(default_factory=list)
    refused: dict[str, list[str]] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _recipe_text(name: str, probe: LearnedProbe) -> str:
    """A probe recipe's file: the format, the skill it belongs to and the probe."""
    data = {"format": RECIPE_FORMAT, "skill": name, "probe": probe.model_dump(mode="json")}
    return json.dumps(data, indent=1) + "\n"


def _write_skill(staging: Path, name: str, text: str) -> Path:
    directory = staging / name
    directory.mkdir(parents=True)
    path = directory / SKILL_FILE
    path.write_text(text, encoding="utf-8")
    return directory


def freeze(
    sources: Sequence[Source],
    parent: Path,
    *,
    previous: FrozenLibrary | None = None,
    base_names: Iterable[str] | None = None,
    kept_on: str | None = None,
    drop_refused: bool = False,
) -> FreezeReport:
    """Check the sources' learned skills and freeze those that pass into a library.

    Raises `LearnedRefused` when a skill fails its check, unless `drop_refused`,
    in which case the report lists it and the library keeps the others. The
    library is written once as `<parent>/learned-<digest16>/`; freezing the
    same skills again returns the same directory. `previous` carries an
    earlier library's skills forward unless a source writes one of the same
    name. Returns a report whose `library` is None when no skill is kept.
    """
    kept_on = kept_on or time.strftime("%Y-%m-%d")
    base = frozenset(base_names) if base_names is not None else base_skill_names()
    for source in sources:
        problems = neutral_identifiers(source.fleet_label)
        if problems:
            raise ValueError(f"the fleet label {source.fleet_label!r} holds identifiers {problems}")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".learned-{uuid.uuid4().hex[:8]}"
    (staging / "skills").mkdir(parents=True)
    report = FreezeReport(library=None)
    entries: dict[str, dict[str, Any]] = {}
    try:
        if previous is not None:
            previous.verify()
            for entry in previous.record["skills"]:
                shutil.copytree(previous.skills / entry["name"], staging / "skills" / entry["name"])
                if entry.get("recipe") is not None:
                    (staging / RECIPES_DIR).mkdir(exist_ok=True)
                    shutil.copy2(previous.root / entry["recipe"]["path"], staging / RECIPES_DIR)
                entries[entry["name"]] = {**entry, "carried_from": previous.digest}
                report.carried.append(entry["name"])
        for source in sources:
            world_id = json.loads((source.analysis / "analysis.json").read_text())["world_id"]
            for name, stored in stored_drafts(source.analysis).items():
                problems = draft_problems(stored.draft)
                if name in base:
                    problems.append(f"{name}: a base skill has this name")
                text = render_learned_skill(stored.draft, provenance_line(source, stored, kept_on))
                check = staging / "check" / uuid.uuid4().hex[:8]
                problems += audit_learned_skill(
                    _write_skill(check, name, text), source.runs, source.fleet
                )
                if problems:
                    report.refused[name] = problems
                    continue
                if name in entries:
                    shutil.rmtree(staging / "skills" / name)
                    (staging / RECIPES_DIR / f"{name}.json").unlink(missing_ok=True)
                    report.carried = [n for n in report.carried if n != name]
                _write_skill(staging / "skills", name, text)
                recipe = None
                if stored.draft.probe is not None:
                    (staging / RECIPES_DIR).mkdir(exist_ok=True)
                    path = staging / RECIPES_DIR / f"{name}.json"
                    path.write_text(_recipe_text(name, stored.draft.probe), encoding="utf-8")
                    recipe = {"path": f"{RECIPES_DIR}/{name}.json", "sha256": _sha256(path)}
                entries[name] = {
                    "name": name,
                    "provenance": provenance_line(source, stored, kept_on),
                    "source": {
                        "analysis": str(source.analysis),
                        "world_id": world_id,
                        "fleet_label": source.fleet_label,
                        "round": source.round_of(stored.session),
                        "session": stored.session,
                        "receipt_id": stored.receipt_id,
                    },
                    "replaces": entries.get(name, {}).get("provenance"),
                    "carried_from": None,
                    "recipe": recipe,
                }
                if name not in report.kept:
                    report.kept.append(name)
        if report.refused and not drop_refused:
            lines = [f"{n}: {p}" for n, problems in report.refused.items() for p in problems]
            raise LearnedRefused("learned skills refused: " + "; ".join(lines))
        shutil.rmtree(staging / "check", ignore_errors=True)
        if not entries:
            return report
        for path in (staging / "skills").rglob(SKILL_FILE):
            path.chmod(0o444)
        if (staging / RECIPES_DIR).is_dir():
            for path in (staging / RECIPES_DIR).glob("*.json"):
                path.chmod(0o444)
        digest = tree_digest(staging / "skills")
        for name, entry in entries.items():
            with (staging / "skills" / name / SKILL_FILE).open("rb") as stream:
                entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        record = {
            "format": LIBRARY_FORMAT,
            "digest": digest,
            "kept_on": kept_on,
            "previous": previous.digest if previous is not None else None,
            "skills": [entries[name] for name in sorted(entries)],
            "checks": "contracts.learned.draft_problems and agents.audit.audit_learned_skill "
            "against each source's runs and fleet",
        }
        (staging / LIBRARY_RECORD).write_text(json.dumps(record, indent=1) + "\n")
        target = parent / f"learned-{digest[:16]}"
        if target.is_dir():
            report.library = FrozenLibrary.load(target)
            return report
        staging.rename(target)
        report.library = FrozenLibrary.load(target)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
