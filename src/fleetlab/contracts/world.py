"""World specifications and world files (docs/design/contracts.md, section 9).

A world is one generated fleet with its own structure, faults and history,
made from a seed. Generation is a pure function of the `WorldSpec`; the world
files are sealed by a hash that is recorded for the test worlds. Seeds of every
episode are derived from the world seed and the identity fields
(`contracts.identity.seeds_for`), never drawn from a global generator.

Choices this module makes where the note is silent:

- `WorldSpec` carries the world's neutral `world_id`, which every identity
  repeats.
- Windows are inclusive day ranges and follow each other in the order
  observed, experiment, scoring, without overlap.
- Patterns refer to faults of the schedule by id: a confounder is a pair
  (effective, inert), an interaction a pair, decoys and drifts lists.
- The world hash is the full SHA-256 (not the 8-byte chain hash) over the four
  files in a fixed order, each as its name, its length and its bytes; it is
  stored in the file `hash` next to them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import WORLD_SEED_LIMIT, PolicySpec, Task
from fleetlab.contracts.ids import FaultId, WorldId
from fleetlab.contracts.vocabulary import LATER_CONFIG_KEYS, ConfigKey

# The world files, in the order they are hashed, and the file holding the hash.
WORLD_FILES: tuple[str, ...] = ("spec.json", "model.json", "description.json", "manifest.jsonl")
HASH_FILE = "hash"


class DayRange(ContractModel):
    """An inclusive range of simulated days."""

    first: int = Field(ge=0)
    last: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> DayRange:
        if self.last < self.first:
            raise ValueError(f"a day range ends ({self.last}) before it starts ({self.first})")
        return self

    def __contains__(self, day: object) -> bool:
        return isinstance(day, int) and self.first <= day <= self.last


class Windows(ContractModel):
    """The observed window, the experiment window and the scoring window, in that order."""

    observed: DayRange
    experiment: DayRange
    scoring: DayRange

    @model_validator(mode="after")
    def _in_order(self) -> Windows:
        if not (
            self.observed.last < self.experiment.first <= self.experiment.last < self.scoring.first
        ):
            raise ValueError("windows follow each other: observed, experiment, scoring")
        return self


class Patterns(ContractModel):
    """The world-level patterns the generator composed from the fault schedule."""

    confounders: list[tuple[FaultId, FaultId]] = Field(default_factory=list)  # (effective, inert)
    decoys: list[FaultId] = Field(default_factory=list)
    interactions: list[tuple[FaultId, FaultId]] = Field(default_factory=list)
    additive_pairs: list[tuple[FaultId, FaultId]] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    drifts: list[FaultId] = Field(default_factory=list)

    def fault_ids(self) -> set[str]:
        pairs = [*self.confounders, *self.interactions, *self.additive_pairs]
        return {*self.decoys, *self.drifts, *(f for pair in pairs for f in pair)}

    def excluded_from_replay(self) -> frozenset[str]:
        """Faults left out of the oracle's power set: decoys and inert confounder entries."""
        return frozenset({*self.decoys, *(inert for _, inert in self.confounders)})


class WorldSpec(ContractModel):
    """Everything the world generator needs; generation is a pure function of it."""

    world_id: WorldId
    seed: int = Field(ge=0, lt=WORLD_SEED_LIMIT)
    size: Literal["small", "medium"]
    sites: int = Field(ge=1)
    robots_per_site: int = Field(ge=1)
    days: int = Field(ge=1)
    windows: Windows
    fault_schedule: list[FaultSpec]
    patterns: Patterns
    policy: PolicySpec
    tasks: list[Task]
    episodes_per_robot_day: int = Field(ge=1)
    # Configuration keys added later that this fleet has, at their nominal values
    # (`vocabulary.LATER_CONFIG_NOMINAL`), recorded at fleet scope. Left out of the
    # serialized spec when empty, so that every world composed before them keeps its hash.
    later_settings: list[ConfigKey] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def _consistent(self) -> WorldSpec:
        if not set(self.later_settings) <= LATER_CONFIG_KEYS:
            raise ValueError("later_settings names a key the vocabulary did not add later")
        if len(set(self.later_settings)) != len(self.later_settings):
            raise ValueError("later_settings names a key twice")
        if self.windows.scoring.last >= self.days:
            raise ValueError(f"the scoring window ends after day {self.days - 1}")
        ids = [f.fault_id for f in self.fault_schedule]
        if len(ids) != len(set(ids)):
            raise ValueError("fault ids repeat in the schedule")
        unknown = sorted(self.patterns.fault_ids() - set(ids))
        if unknown:
            raise ValueError(f"patterns name faults not in the schedule: {unknown}")
        by_id = {f.fault_id: f for f in self.fault_schedule}
        for effective, inert in self.patterns.confounders:
            if by_id[effective].schedule.onset_day != by_id[inert].schedule.onset_day:
                raise ValueError(f"confounders {effective}, {inert} have different onsets")
        for fault_id in self.patterns.drifts:
            if by_id[fault_id].schedule.drift is None:
                raise ValueError(f"drift {fault_id} has no drift in its schedule")
        if not self.tasks:
            raise ValueError("a world runs at least one task")
        return self


@dataclass(frozen=True)
class WorldFiles:
    """The files of one world in a directory: spec, model (truth), description, manifest, hash."""

    root: Path

    @property
    def spec(self) -> Path:
        return self.root / "spec.json"

    @property
    def model(self) -> Path:
        return self.root / "model.json"

    @property
    def description(self) -> Path:
        return self.root / "description.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def hash_file(self) -> Path:
        return self.root / HASH_FILE


def hash_world_files(directory: Path | str) -> str:
    """The SHA-256 of a world's four files, as 64 hexadecimal digits.

    Each file enters as its name, a NUL, its length in bytes, a NUL and its
    bytes, in the order of `WORLD_FILES`. Raises FileNotFoundError if one is
    missing.
    """
    root = Path(directory)
    digest = hashlib.sha256()
    for name in WORLD_FILES:
        data = (root / name).read_bytes()
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def seal_world(directory: Path | str) -> str:
    """Write the hash of the world files to `hash` and return it."""
    value = hash_world_files(directory)
    (Path(directory) / HASH_FILE).write_text(value + "\n")
    return value


def verify_world(directory: Path | str) -> bool:
    """Whether the world files still match the hash stored with them."""
    stored = (Path(directory) / HASH_FILE).read_text().strip()
    return stored == hash_world_files(directory)
