"""The host's start allocation: development starts 0 to 29 and release starts 30 to 49.

A start is a `contracts.jobs.StartIdentity`: the base task, LIBERO's stored initial state,
the scene seed and the policy's noise seed. The base task supplies the pool of 50 stored
starts of the shared scene; the requested goals, not the base task, are the work.

How the seeds derive is the request's allocation profile (`contracts.jobs.
ALLOCATION_PROFILES`), host configuration the agent never sees:

- null, the legacy profile: `seeds_for(JOB_SEED, "r01", 0, init_state)`, the derivation of
  lane S's spike runner (`spikes/jobs/runner.py`, `start_identity`), so a start of the
  runtime is the spike's start and its runs compare bit for bit. The seeds do not depend
  on the base task: the same initial state of every pool shares them.
- `pool-v1`: `seeds_for(JOB_SEED + 1 + base_task, "r01", 0, init_state)`, so that a pool's
  starts share no seeds with another pool's or with the legacy starts.

Every derivation names its profile (`start_for`, `development_starts_for`,
`release_starts_for`, keyword `profile`). `start_identity`, `development_starts` and
`release_starts` are the legacy profile's, by name, for the records and tests written
before the profiles. Host side only: `contracts.identity`, which derives the seeds, is
kept out of the tool server's bundle.
"""

from __future__ import annotations

from collections.abc import Iterable

from fleetlab.contracts.identity import seeds_for
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.jobs import (
    ALLOCATION_PROFILES,
    DEVELOPMENT_INIT_STATES,
    POOL_V1,
    RELEASE_INIT_STATES,
    StartIdentity,
)

__all__ = [
    "JOB_SEED",
    "check_profile",
    "development_starts",
    "development_starts_for",
    "release_starts",
    "release_starts_for",
    "seed_origin",
    "start_for",
    "start_identity",
    "starts_for",
]

# The seed every start's scene and noise seeds derive from: the one the spike's runner used.
JOB_SEED = 20261020


def check_profile(profile: str | None) -> str | None:
    """`profile` when it is one the host knows (null, the legacy profile, included)."""
    if profile is not None and profile not in ALLOCATION_PROFILES:
        raise ValueError(f"unknown allocation profile {profile!r}: null or {ALLOCATION_PROFILES}")
    return profile


def seed_origin(base_task: int, profile: str | None) -> int:
    """The seed a pool's starts derive from under a profile."""
    if check_profile(profile) == POOL_V1:
        return JOB_SEED + 1 + base_task
    return JOB_SEED


def start_for(base_task: int, init_state: int, *, profile: str | None) -> StartIdentity:
    """The host's start for a base task and a stored initial state under `profile`."""
    scene, noise = seeds_for(
        seed_origin(base_task, profile), neutral_id(IdKind.robot, 1), 0, init_state
    )
    return StartIdentity(
        base_task=base_task, init_state=init_state, scene_seed=scene, noise_seed=noise
    )


def starts_for(
    base_task: int, init_states: Iterable[int], *, profile: str | None
) -> list[StartIdentity]:
    """The starts of these stored initial states, in their order."""
    return [start_for(base_task, k, profile=profile) for k in init_states]


def development_starts_for(
    base_task: int, n: int, first: int = 0, *, profile: str | None
) -> list[StartIdentity]:
    """`n` development starts from initial state `first` on, within 0 to 29."""
    states = range(first, first + n)
    if n < 1 or states.start < 0 or states.stop > DEVELOPMENT_INIT_STATES.stop:
        raise ValueError(f"development starts are initial states 0 to 29, not {states}")
    return starts_for(base_task, states, profile=profile)


def release_starts_for(base_task: int, n: int, *, profile: str | None) -> list[StartIdentity]:
    """The first `n` of the reserved release starts, initial states 30 to 49."""
    if not 1 <= n <= len(RELEASE_INIT_STATES):
        raise ValueError(f"there are {len(RELEASE_INIT_STATES)} release starts, not {n}")
    return starts_for(base_task, RELEASE_INIT_STATES[:n], profile=profile)


def start_identity(base_task: int, init_state: int) -> StartIdentity:
    """The legacy profile's start (`start_for` with profile null)."""
    return start_for(base_task, init_state, profile=None)


def development_starts(base_task: int, n: int, first: int = 0) -> list[StartIdentity]:
    """The legacy profile's development starts (`development_starts_for`, profile null)."""
    return development_starts_for(base_task, n, first, profile=None)


def release_starts(base_task: int, n: int) -> list[StartIdentity]:
    """The legacy profile's release starts (`release_starts_for`, profile null)."""
    return release_starts_for(base_task, n, profile=None)
