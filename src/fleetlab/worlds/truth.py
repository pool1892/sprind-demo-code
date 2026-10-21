"""The hidden fault table of a world: every fault of the schedule with its role and
detectability, as the world generator computes
it and the scorer reads it.

Truth side only: the file `fault_table.json` lies in the world directory beside the
four hashed world files and is never mounted into a sandbox.

A fault is a **detectable cause** when removing it alone changes fleet success by at
least `DETECT_POINTS` points over the observed window or over the scoring window
(computed from the calibrated dose-response curves at generation: the observed window
is not replayed), its target or scope is identifiable in the visible data, and at
least `DETECT_EPISODES` episodes of the observed window carry it. An effective fault
that misses the rule is a **small cause**, scored separately; a decoy or the inert
entry of a confounder pair is neither.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.ids import FaultId, WorldId
from fleetlab.contracts.vocabulary import FaultClass, Verb
from fleetlab.contracts.world import Windows

DETECT_POINTS = 2.0
DETECT_EPISODES = 20
FAULT_TABLE_FILE = "fault_table.json"

WindowName = Literal["observed", "experiment", "scoring"]
Role = Literal["effective", "decoy", "inert"]


class FaultTruth(ContractModel):
    """One fault of the schedule as the scorer sees it.

    `role`: "effective" for a fault meant to act, "decoy" for a fault whose
    dose-response is flat, "inert" for the inert entry of a confounder pair.
    `directions` are the diagnosis directions claim 1 accepts for it (its type's
    truth mapping, plus "drifting" when it drifts). `expected_loss_points` is the
    fleet success lost to it over each window, from the calibrated curve (positive
    is harm). `episodes` counts the as-run episodes of each window that carry it.
    `qualified_by` lists the windows whose loss reaches `DETECT_POINTS`.
    `calibration` names the curve point the expectation used.
    """

    fault_id: FaultId
    kind: str
    fault_class: FaultClass
    target: Target
    role: Role
    onset_day: int = Field(ge=0)
    level: float
    directions: list[Direction]
    fixing_verbs: list[Verb]
    episodes: dict[WindowName, int]
    expected_loss_points: dict[WindowName, float]
    identifiable: bool
    qualified_by: list[WindowName]
    detectable: bool
    small: bool
    calibration: str
    drift_crossing_day: int | None = Field(default=None, ge=0)


class FaultTable(ContractModel):
    """The hidden fault table of one world (`fault_table.json`)."""

    world_id: WorldId
    windows: Windows
    episodes_per_window: dict[WindowName, int]
    rule: str
    faults: list[FaultTruth]

    def by_id(self) -> dict[str, FaultTruth]:
        return {f.fault_id: f for f in self.faults}

    def detectable(self) -> list[FaultTruth]:
        return [f for f in self.faults if f.detectable]

    def small(self) -> list[FaultTruth]:
        return [f for f in self.faults if f.small]

    def decoys(self) -> list[FaultTruth]:
        return [f for f in self.faults if f.role == "decoy"]

    def inert(self) -> list[FaultTruth]:
        return [f for f in self.faults if f.role == "inert"]
