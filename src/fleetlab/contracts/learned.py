"""Learned diagnostic skills: what an analysis agent keeps for a later investigation.

docs/design/loop.md, section 9. During an investigation the analysis agent can
write down what a competent engineer would want to know the next time a
similar fleet shows a similar pattern. It does so with the tool
`write_learned_skill`, whose argument is a `LearnedSkillDraft`. The draft is
rendered into the existing skill format (contracts, section 13): a `SKILL.md`
with front matter and SQL blocks that a later agent copies, fills in and runs
through `query`. Beside the usual front matter a learned skill carries
`origin: learned`, `applies_when` and `provenance`, and its body has the
sections Applies when, Parameters, Evidence, Competing explanations, Next
experiment and Provenance.

What this module checks needs no knowledge of a world, so the tool server
(inside the sandbox) and the host run the same rules (`draft_problems`):

- no neutral identifier anywhere in the draft: a particular robot, site,
  episode or experiment of the fleet the skill was written on is a parameter,
  written `$name` in SQL and `{name}` in prose and declared under
  `parameters`, because the next fleet has other ones;
- every SQL is one statement whose `$parameters` are declared, without a code
  fence inside it; every `{placeholder}` in prose is declared;
- the fields are filled and within their sizes.

The tool server adds what needs the fleet's data (each SQL is one SELECT and
runs once on this fleet with the values of `check_with`, which are never
stored), and the host adds what needs the truth side (the leak rules) when it
freezes the skill into a library (`fleetlab.loop.learned`).

A skill can carry the probe that showed its pattern (`LearnedProbe`, the optional
`probe` of a draft): the request of `request_probe` with its robots as parameters
of the skill (`robot_parameters`), checked against the probe's public bounds when
it is written. The skill renders it as a JSON block under "Next experiment"; the
host keeps it beside the frozen library as a recipe that a command runs on named
robots (`fleetlab.loop.probes`), with the skill's evidence SQL reading the result
(`$probe` bound to the probe's id). A draft without a probe validates, dumps and
renders exactly as before the field existed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import Field, StringConstraints, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.probes import (
    PROBE_BOUNDS,
    PROBE_NAME_PATTERN,
    ProbeProgram,
    ProbeRequest,
    ProbeSegment,
)
from fleetlab.contracts.skills import SKILL_NAME_PATTERN, neutral_identifiers

__all__ = [
    "LEARNED_FILE_FORMAT",
    "LEARNED_SECTIONS",
    "MAX_LEARNED_SKILLS",
    "PARAMETER_PATTERN",
    "PROBE_PARAMETER",
    "CheckValue",
    "LearnedAlternative",
    "LearnedEvidence",
    "LearnedParameter",
    "LearnedProbe",
    "LearnedSkillDraft",
    "LearnedSkillReceipt",
    "draft_problems",
    "evidence_sql",
    "placeholders",
    "probe_block",
    "reads_probe",
    "render_learned_skill",
    "sql_parameters",
    "typed_value",
]

PARAMETER_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
# At most this many distinct learned skills per analysis; a skill is updated by
# writing it again under the same name.
MAX_LEARNED_SKILLS = 3
LEARNED_FILE_FORMAT = "fleetlab.learned-skill/1"
# The body's sections, in order; each is a level-two heading of the SKILL.md.
LEARNED_SECTIONS = (
    "Applies when",
    "Parameters",
    "Evidence",
    "Competing explanations",
    "Next experiment",
    "Provenance",
)
# Sizes, in characters, so that a skill stays one readable tool result.
MAX_TEXT = 2000
MAX_SQL = 4000
MAX_RENDERED = 24_000
PROVENANCE_PENDING = "added by the host when the investigation ends"

_SQL_PARAMETER = re.compile(r"(?<![A-Za-z0-9_$])\$([a-z][a-z0-9_]*)")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
# The parameter through which a skill's evidence SQL reads the probe the skill carries:
# whoever runs the probe again binds it to that probe's id.
PROBE_PARAMETER = "probe"
_EVIDENCE_BLOCK = re.compile(r"```sql\n-- helper: (evidence-[0-9]+)\n(.*?)\n```", re.DOTALL)
_PROBE_STEPS = re.compile(r"(?<![A-Za-z0-9_])probe_steps(?![A-Za-z0-9_])", re.IGNORECASE)


def reads_probe(sql: str) -> bool:
    """Whether an evidence query analyses a carried probe: it reads the measurements
    (`probe_steps`) and takes the probe's id (`$probe`)."""
    return PROBE_PARAMETER in sql_parameters(sql) and _PROBE_STEPS.search(sql) is not None


class LearnedParameter(ContractModel):
    """A value the skill needs from the fleet it is used on, such as the site under suspicion."""

    name: Annotated[str, Field(pattern=PARAMETER_PATTERN)]
    meaning: str


class LearnedEvidence(ContractModel):
    """One query of the skill: what it shows, the SQL, and what it returned where it was learned."""

    claim: str  # what the query shows, in general terms; identifiers as {name}
    sql: str  # one SELECT over the visible tables; parameters as $name
    result: str  # what it returned on the fleet it was written on; identifiers as {name}


class LearnedAlternative(ContractModel):
    """A competing explanation of the same pattern, and the observation that tells them apart."""

    explanation: str
    how_to_tell: str


class CheckValue(ContractModel):
    """A value of this fleet for one parameter, used once to run the SQL; never stored."""

    name: Annotated[str, Field(pattern=PARAMETER_PATTERN)]
    value: str


class LearnedProbe(ContractModel):
    """The probe a learned skill carries, so that a later investigation can run it again:
    request_probe's request with robot_parameters, parameters of the skill in order, in
    place of its robots; {name} in the purpose and the expectation stands for a
    parameter's value. The skill's evidence SQL reads it with $probe, the probe's id."""

    name: Annotated[str, StringConstraints(pattern=PROBE_NAME_PATTERN)]
    purpose: str = Field(max_length=PROBE_BOUNDS.max_text)  # the question it answers
    expectation: str = Field(max_length=PROBE_BOUNDS.max_text)  # what would answer it
    robot_parameters: list[Annotated[str, Field(pattern=PARAMETER_PATTERN)]] = Field(
        min_length=1, max_length=PROBE_BOUNDS.max_robots
    )
    repeats: int = Field(ge=1, le=PROBE_BOUNDS.max_repeats)  # probe episodes per robot
    segments: list[ProbeSegment] = Field(min_length=1, max_length=PROBE_BOUNDS.max_segments)

    @model_validator(mode="after")
    def _valid(self) -> LearnedProbe:
        if len(set(self.robot_parameters)) != len(self.robot_parameters):
            raise ValueError("robot_parameters names each parameter once")
        ProbeProgram(segments=list(self.segments))  # the program's own bounds
        return self

    def request(self, robots: Sequence[str]) -> ProbeRequest:
        """The probe on `robots`, one for each robot parameter in order, as `request_probe`
        takes it; each `{parameter}` of the purpose and the expectation filled in."""
        named = list(robots)
        if len(named) != len(self.robot_parameters):
            raise ValueError(
                f"the probe takes {len(self.robot_parameters)} robots "
                f"({', '.join(self.robot_parameters)}), not {len(named)}"
            )
        values = dict(zip(self.robot_parameters, named, strict=True))

        def fill(text: str) -> str:
            return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)

        return ProbeRequest(
            name=self.name,
            purpose=fill(self.purpose),
            expectation=fill(self.expectation),
            robots=named,
            repeats=self.repeats,
            segments=list(self.segments),
        )


class LearnedSkillDraft(ContractModel):
    """The argument of `write_learned_skill`: a diagnostic skill for a later investigation."""

    name: Annotated[str, Field(pattern=SKILL_NAME_PATTERN, max_length=64)]
    description: Annotated[str, Field(min_length=1, max_length=1024)]
    applies_when: str  # the pattern in the records that makes the skill relevant
    parameters: list[LearnedParameter] = Field(max_length=8)
    evidence: list[LearnedEvidence] = Field(min_length=1, max_length=8)
    alternatives: list[LearnedAlternative] = Field(min_length=1, max_length=8)
    next_experiment: str  # the experiment that settles the question, and how to read it
    check_with: list[CheckValue] = Field(max_length=8)
    # The probe that showed the pattern, to run again (null: none). Left out of every dump
    # when absent, so a draft without one is stored as before the field existed.
    probe: LearnedProbe | None = Field(default=None, exclude_if=lambda value: value is None)


class LearnedSkillReceipt(ContractModel):
    """The answer to `write_learned_skill`: accepted or not, why, and the skill as it is kept."""

    accepted: bool
    receipt_id: str
    problems: list[str]
    skill_md: str


def sql_parameters(sql: str) -> list[str]:
    """The `$name` parameters of a SQL text, in order of first use."""
    return list(dict.fromkeys(_SQL_PARAMETER.findall(sql)))


def placeholders(text: str) -> list[str]:
    """The `{name}` placeholders of a prose text, in order of first use."""
    return list(dict.fromkeys(_PLACEHOLDER.findall(text)))


def typed_value(value: str) -> int | float | str:
    """A check value as the SQL receives it: an integer, a number, or a string."""
    text = value.strip()
    if re.fullmatch(r"-?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"-?[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?", text):
        return float(text)
    return value


def _texts(draft: LearnedSkillDraft) -> list[tuple[str, str]]:
    """Every stored text of a draft with its field path (check values are not stored)."""
    out = [
        ("name", draft.name),
        ("description", draft.description),
        ("applies_when", draft.applies_when),
        ("next_experiment", draft.next_experiment),
    ]
    for i, parameter in enumerate(draft.parameters):
        out += [(f"parameters[{i}].name", parameter.name)]
        out += [(f"parameters[{i}].meaning", parameter.meaning)]
    for i, item in enumerate(draft.evidence):
        out += [(f"evidence[{i}].{k}", getattr(item, k)) for k in ("claim", "sql", "result")]
    for i, alternative in enumerate(draft.alternatives):
        out += [
            (f"alternatives[{i}].{k}", getattr(alternative, k))
            for k in ("explanation", "how_to_tell")
        ]
    if draft.probe is not None:
        out += [(f"probe.{k}", getattr(draft.probe, k)) for k in ("name", "purpose", "expectation")]
    return out


_KIND_WORDS = {
    "r": "robot",
    "s": "site",
    "e": "episode",
    "x": "experiment",
    "w": "fleet",
    "f": "fault",
}


def draft_problems(draft: LearnedSkillDraft) -> list[str]:
    """Why a draft cannot be kept, as messages the agent can act on; empty when it passes.

    Checks what needs no data: identifiers, declared parameters, sizes and
    filled fields. SQL statement types and execution are the tool server's.
    """
    problems: list[str] = []
    declared = [p.name for p in draft.parameters]
    if len(set(declared)) != len(declared):
        problems.append("parameters: each parameter is declared once")
    for where, text in _texts(draft):
        if not text.strip():
            problems.append(f"{where}: empty")
        limit = MAX_SQL if where.endswith(".sql") else MAX_TEXT
        if len(text) > limit:
            problems.append(f"{where}: {len(text)} characters, at most {limit}")
        for word in dict.fromkeys(neutral_identifiers(text)):
            kind = _KIND_WORDS.get(word[0], "neutral")
            problems.append(
                f"{where}: {word!r} has the form of a {kind} identifier. Identifiers of this "
                "fleet differ on the next one: write a parameter instead ($name in SQL, {name} "
                "in prose) and declare it under parameters"
                + (" (or rename the SQL alias)" if where.endswith(".sql") else "")
            )
        if where.endswith(".sql"):
            if "```" in text:
                problems.append(f"{where}: a code fence inside the SQL")
            undeclared = [n for n in sql_parameters(text) if n not in declared]
            if undeclared:
                problems.append(
                    f"{where}: parameters {undeclared} are not declared under parameters"
                )
        elif not where.endswith(".name"):
            undeclared = [n for n in placeholders(text) if n not in declared]
            if undeclared:
                problems.append(
                    f"{where}: placeholders {undeclared} are not declared under parameters"
                )
    used = {n for item in draft.evidence for n in sql_parameters(item.sql)}
    used |= {n for _, text in _texts(draft) for n in placeholders(text)}
    if draft.probe is not None:
        robots = draft.probe.robot_parameters
        used |= set(robots)
        undeclared = [n for n in robots if n not in declared]
        if undeclared:
            problems.append(
                f"probe.robot_parameters: {undeclared} are not declared under parameters"
            )
        if PROBE_PARAMETER in robots:
            problems.append(
                f"probe.robot_parameters: ${PROBE_PARAMETER} is the probe's own id, not a robot"
            )
        if not any(reads_probe(item.sql) for item in draft.evidence):
            problems.append(
                "probe: no evidence SQL analyses its measurements; at least one evidence "
                f"query reads probe_steps for the probe's episodes, taking ${PROBE_PARAMETER} "
                "(the probe's id, declared under parameters), for example through "
                f"probe_episodes.probe_id = ${PROBE_PARAMETER}"
            )
    unused = [n for n in declared if n not in used]
    if unused:
        problems.append(f"parameters {unused} are declared but used nowhere")
    known = set(declared)
    problems += [
        f"check_with: {c.name!r} is not a declared parameter"
        for c in draft.check_with
        if c.name not in known
    ]
    if not problems and len(render_learned_skill(draft)) > MAX_RENDERED:
        problems.append(f"the skill is longer than {MAX_RENDERED} characters; shorten it")
    return problems


_PROBE_BLOCK = re.compile(r"\n### The probe\n.*?```json\n(.*?)\n```", re.DOTALL)


def probe_block(skill_md: str) -> dict[str, Any] | None:
    """The probe a rendered learned skill carries, as its JSON block holds it; None if none."""
    found = _PROBE_BLOCK.search(skill_md)
    if found is None:
        return None
    data: dict[str, Any] = json.loads(found.group(1))
    return data


def evidence_sql(skill_md: str) -> list[tuple[str, str]]:
    """The evidence SQL of a rendered learned skill, in order: (helper name, SQL)."""
    return [(m.group(1), m.group(2)) for m in _EVIDENCE_BLOCK.finditer(skill_md)]


def _scalar(text: str) -> str:
    """A front matter value on one line, quoted so that the parser keeps it a string."""
    return '"' + " ".join(text.split()) + '"'


def render_learned_skill(draft: LearnedSkillDraft, provenance: str | None = None) -> str:
    """The SKILL.md of a draft. `provenance` is the host's line; None while it is a draft.

    Deterministic: the same draft and provenance give the same bytes, which
    the library's digest relies on.
    """
    stamp = provenance if provenance is not None else PROVENANCE_PENDING
    inputs = [f"{p.name}: {' '.join(p.meaning.split())}" for p in draft.parameters]
    lines = [
        "---",
        f"name: {draft.name}",
        f"description: {_scalar(draft.description)}",
        f"when_to_use: {_scalar(draft.applies_when)}",
        "tools_used: [query, request_experiment, request_probe]"
        if draft.probe is not None
        else "tools_used: [query, request_experiment]",
    ]
    if inputs:
        lines += ["inputs:", *(f"  - {_scalar(item)}" for item in inputs)]
    else:
        lines += ["inputs: []"]
    lines += [
        "outputs:",
        '  - "which of the competing explanations the evidence supports on this fleet"',
        '  - "the next experiment to run, with its arms"',
        'version: "1"',
        "origin: learned",
        f"applies_when: {_scalar(draft.applies_when)}",
        f"provenance: {_scalar(stamp)}",
        "---",
        f"# {draft.name} (learned)",
        "",
        "Written by an analysis agent during an earlier investigation and kept for later",
        "ones. It is not one of the base procedures: check that it applies here before",
        "relying on it, and where this fleet's records disagree with it, they win.",
        "",
        "## Applies when",
        "",
        draft.applies_when.strip(),
        "",
        "## Parameters",
        "",
    ]
    if draft.parameters:
        lines += [f"- `{p.name}`: {p.meaning.strip()}" for p in draft.parameters]
        lines += [
            "",
            "In a SQL block, replace each `$name` by a value of this fleet (an identifier",
            "in single quotes, a day as a number) and run the block with `query`.",
        ]
    else:
        lines += ["None."]
    lines += ["", "## Evidence", ""]
    for i, item in enumerate(draft.evidence, start=1):
        lines += [
            f"### {i}. {' '.join(item.claim.split())}",
            "",
            "```sql",
            f"-- helper: evidence-{i}",
            item.sql.strip(),
            "```",
            "",
            f"Where it was learned: {item.result.strip()}",
            "",
        ]
    lines += ["## Competing explanations", ""]
    for i, alternative in enumerate(draft.alternatives, start=1):
        lines += [
            f"{i}. {alternative.explanation.strip()}",
            f"   How to tell: {alternative.how_to_tell.strip()}",
        ]
    lines += [
        "",
        "## Next experiment",
        "",
        draft.next_experiment.strip(),
        "",
    ]
    if draft.probe is not None:
        robots = ", ".join(f"`{n}`" for n in draft.probe.robot_parameters)
        lines += [
            "### The probe",
            "",
            "Request it with `request_probe` as below, with one robot for each of",
            f"`robot_parameters` ({robots}), in that order, as `robots`; the evidence",
            f"SQL reads it with `${PROBE_PARAMETER}`, the probe's id.",
            "",
            "```json",
            json.dumps(draft.probe.model_dump(mode="json"), indent=1),
            "```",
            "",
        ]
    lines += [
        "## Provenance",
        "",
        stamp,
        "",
    ]
    return "\n".join(lines)
