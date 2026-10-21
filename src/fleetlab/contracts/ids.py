"""Neutral identifiers and scopes (docs/design/contracts.md, principle 6).

Every identifier an agent can see carries no information: a letter for the
kind and a number, `e000123`, `r07`, `s2`. The map from identifier to identity
lives on the truth side. The number is formatted to a fixed minimum width, and
only that canonical form is accepted: `r7` and `r007` are refused, so that one
robot has exactly one name.

This module is the home of the identifier rules. It is separate from
`contracts.identity` (which re-exports it) only so that every other contracts
module can use the types without an import cycle.

Kinds beyond the three the note names: `world` (the world and fleet id, also
neutral), `fault` (truth side only), `experiment` (the ids `request_experiment`
returns to an agent) and `probe` (the ids `request_probe` returns, `p0001`). New
fault ids are opaque `fx` tokens, never visible. The historical counted form remains
valid for stored identities and their replays.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, StringConstraints


class IdKind(StrEnum):
    """The kinds of neutral identifier."""

    world = "world"
    episode = "episode"
    robot = "robot"
    site = "site"
    fault = "fault"
    experiment = "experiment"
    probe = "probe"


# Prefix letter and minimum number of digits per kind. `e000123`, `r07`, `s2` are the
# note's examples; the other widths are chosen to the same pattern.
_PREFIX: dict[IdKind, str] = {
    IdKind.world: "w",
    IdKind.episode: "e",
    IdKind.robot: "r",
    IdKind.site: "s",
    IdKind.fault: "f",
    IdKind.experiment: "x",
    IdKind.probe: "p",
}
_WIDTH: dict[IdKind, int] = {
    IdKind.world: 2,
    IdKind.episode: 6,
    IdKind.robot: 2,
    IdKind.site: 1,
    IdKind.fault: 2,
    IdKind.experiment: 4,
    IdKind.probe: 4,
}

# The pattern each kind matches, as a JSON schema `pattern` (no canonical-form check).
ID_PATTERNS: dict[IdKind, str] = {
    kind: rf"^{_PREFIX[kind]}[0-9]{{{_WIDTH[kind]},}}$" for kind in IdKind
}
OPAQUE_FAULT_PATTERN = r"fx[0-9a-f]{32}"
ID_PATTERNS[IdKind.fault] = rf"^(f[0-9]{{2,}}|{OPAQUE_FAULT_PATTERN})$"
_COMPILED = {kind: re.compile(pattern) for kind, pattern in ID_PATTERNS.items()}

# The word "fleet" or a site or robot id.
SCOPE_PATTERN = rf"^(fleet|{ID_PATTERNS[IdKind.site][1:-1]}|{ID_PATTERNS[IdKind.robot][1:-1]})$"
FLEET_SCOPE = "fleet"


def neutral_id(kind: IdKind | str, number: int) -> str:
    """The neutral identifier of the given kind and number, for example `("robot", 7) -> "r07"`."""
    kind = IdKind(kind)
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        raise ValueError(f"an identifier number is a non-negative integer, not {number!r}")
    return f"{_PREFIX[kind]}{number:0{_WIDTH[kind]}d}"


def id_number(kind: IdKind | str, value: str) -> int:
    """The number inside a neutral identifier; raises ValueError for anything else."""
    check_neutral(kind, value)
    if value.startswith("fx"):
        raise ValueError("an opaque fault identifier has no ordinal number")
    return int(value[1:])


def is_neutral(kind: IdKind | str, value: object) -> bool:
    """Whether `value` is a neutral identifier of `kind` in its canonical form."""
    kind = IdKind(kind)
    if not isinstance(value, str) or not _COMPILED[kind].fullmatch(value):
        return False
    if kind is IdKind.fault and value.startswith("fx"):
        # A token contains both digits and letters: it cannot be a dictionary word
        # or a count disguised by the prefix. The composer rejects either rare case.
        return any(c.isdigit() for c in value[2:]) and any(c.isalpha() for c in value[2:])
    return value == neutral_id(kind, int(value[1:]))


def check_neutral(kind: IdKind | str, value: object) -> str:
    """Return `value` if it is a neutral identifier of `kind`; raise ValueError otherwise."""
    if not is_neutral(kind, value):
        example = neutral_id(kind, 7)
        raise ValueError(
            f"{value!r} is not a neutral {IdKind(kind).value} identifier (expected the form "
            f"{example!r}: a letter and a number, nothing that describes the item)"
        )
    assert isinstance(value, str)
    return value


def which_kind(value: str) -> IdKind | None:
    """The kind of a neutral identifier, or None if `value` is not one."""
    for kind in IdKind:
        if is_neutral(kind, value):
            return kind
    return None


def _checker(kind: IdKind) -> AfterValidator:
    return AfterValidator(lambda value: check_neutral(kind, value))


# Pydantic field types. The JSON schema carries the pattern; validation also
# checks the canonical form.
WorldId = Annotated[
    str, StringConstraints(pattern=ID_PATTERNS[IdKind.world]), _checker(IdKind.world)
]
EpisodeId = Annotated[
    str, StringConstraints(pattern=ID_PATTERNS[IdKind.episode]), _checker(IdKind.episode)
]
RobotId = Annotated[
    str, StringConstraints(pattern=ID_PATTERNS[IdKind.robot]), _checker(IdKind.robot)
]
SiteId = Annotated[str, StringConstraints(pattern=ID_PATTERNS[IdKind.site]), _checker(IdKind.site)]
FaultId = Annotated[
    str, StringConstraints(pattern=ID_PATTERNS[IdKind.fault]), _checker(IdKind.fault)
]
ExperimentId = Annotated[
    str, StringConstraints(pattern=ID_PATTERNS[IdKind.experiment]), _checker(IdKind.experiment)
]


class ScopeKind(StrEnum):
    """How wide a scope is: the whole fleet, one site, or one robot."""

    fleet = "fleet"
    site = "site"
    robot = "robot"


# Narrow scopes override wide ones: a robot's configuration value wins over its site's.
SCOPE_RANK: dict[ScopeKind, int] = {ScopeKind.fleet: 0, ScopeKind.site: 1, ScopeKind.robot: 2}


def scope_kind(scope: str) -> ScopeKind:
    """The kind of a scope: "fleet", a site id or a robot id. Raises ValueError otherwise."""
    if scope == FLEET_SCOPE:
        return ScopeKind.fleet
    if is_neutral(IdKind.site, scope):
        return ScopeKind.site
    if is_neutral(IdKind.robot, scope):
        return ScopeKind.robot
    raise ValueError(f"{scope!r} is not a scope: expected 'fleet', a site id or a robot id")


def check_scope(scope: object) -> str:
    """Return `scope` if it is "fleet", a site id or a robot id; raise ValueError otherwise."""
    if not isinstance(scope, str):
        raise ValueError(f"a scope is a string, not {scope!r}")
    scope_kind(scope)
    return scope


Scope = Annotated[str, StringConstraints(pattern=SCOPE_PATTERN), AfterValidator(check_scope)]


def scope_covers(scope: str, robot_id: str, site_id: str) -> bool:
    """Whether `scope` includes the robot `robot_id`, which stands at site `site_id`."""
    kind = scope_kind(scope)
    if kind is ScopeKind.fleet:
        return True
    if kind is ScopeKind.site:
        return scope == site_id
    return scope == robot_id


def scope_rank(scope: str) -> int:
    """0 for the fleet, 1 for a site, 2 for a robot: the higher, the narrower."""
    return SCOPE_RANK[scope_kind(scope)]
