"""The hash chain that proves a replay exact (contracts, section 4 and the `chain` table).

Row 0 holds the hashes after the reset; row t the hashes after step t, with the
action that was executed to get there (after the action hook). Hashes are the
contracts' `hash_array`: the state as float64 from `mj_getState(INTEGRATION)`,
each camera image as rendered (rotated, uint8, before the observation hook).

Stored as `.npz`, one array per column of `contracts.log.SCHEMA["chain"]`, plus
`fixture_hash` (one string: the body poses the reset placed) when this module
writes it; section 4 names it and the table has no column for it, so the
log's writer (`fleetlab.log`, which writes the run's chains) leaves it out
and the engine keeps it in the job's record in `run.json`. A chain that
continues from a snapshot starts at the snapshot's step (`first_step`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.policy import ACTION_DIM

HASH_DTYPE = "<U16"


@dataclass
class Chain:
    """One job's chain while it is recorded, and as it is read back."""

    fixture_hash: str = ""
    steps: list[int] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    wrist: list[str] = field(default_factory=list)
    executed: list[npt.NDArray[np.float32]] = field(default_factory=list)

    def add(
        self,
        step: int,
        state_hash: str,
        scene_hash: str,
        wrist_hash: str,
        executed: npt.NDArray[np.float32] | None,
    ) -> None:
        if self.steps and step != self.steps[-1] + 1:
            raise ValueError(f"chain step {step} does not follow {self.steps[-1]}")
        self.steps.append(step)
        self.state.append(state_hash)
        self.scene.append(scene_hash)
        self.wrist.append(wrist_hash)
        if executed is None:
            self.executed.append(np.full(ACTION_DIM, np.nan, np.float32))
        else:
            self.executed.append(np.asarray(executed, np.float32).reshape(ACTION_DIM).copy())

    @property
    def first_step(self) -> int:
        return self.steps[0] if self.steps else 0

    def __len__(self) -> int:
        return len(self.steps)

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {
            "step": np.asarray(self.steps, np.int32),
            "state_hash": np.asarray(self.state, HASH_DTYPE),
            "scene_image_hash": np.asarray(self.scene, HASH_DTYPE),
            "wrist_image_hash": np.asarray(self.wrist, HASH_DTYPE),
            "executed_action": (
                np.stack(self.executed) if self.executed else np.zeros((0, ACTION_DIM), np.float32)
            ),
            "fixture_hash": np.asarray(self.fixture_hash, HASH_DTYPE),
        }

    def save(self, path: Path) -> None:
        """Write atomically: a chain file that exists is complete (resumability)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(temporary, "wb") as fh:
            np.savez_compressed(fh, **self.arrays())
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> Chain:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                fixture_hash=str(data["fixture_hash"]) if "fixture_hash" in data.files else "",
                steps=[int(s) for s in data["step"]],
                state=[str(s) for s in data["state_hash"]],
                scene=[str(s) for s in data["scene_image_hash"]],
                wrist=[str(s) for s in data["wrist_image_hash"]],
                executed=list(np.asarray(data["executed_action"], np.float32)),
            )


@dataclass(frozen=True)
class ChainComparison:
    """How a chain compares with a reference over the steps they share."""

    identical: bool
    compared_steps: int
    first_difference: int | None  # the first step that differs, or None
    what: str  # which column differs first, or why the comparison failed


def compare(chain: Chain, reference: Chain, *, check_fixture: bool = True) -> ChainComparison:
    """Compare `chain` with `reference` from `chain.first_step` to the end of both.

    A chain that continues from a snapshot is compared with the reference's rows
    from the snapshot's step on; it must end where the reference ends.
    """
    if not len(chain):
        return ChainComparison(False, 0, None, "empty chain")
    both = bool(chain.fixture_hash and reference.fixture_hash)
    if check_fixture and both and chain.fixture_hash != reference.fixture_hash:
        return ChainComparison(False, 0, chain.first_step, "fixture_hash")
    offset = chain.first_step - reference.first_step
    if offset < 0 or offset >= len(reference):
        return ChainComparison(False, 0, chain.first_step, "chain starts outside the reference")
    ref_rows = range(offset, len(reference))
    shorter = min(len(ref_rows), len(chain))
    for i in range(shorter):
        j = offset + i
        step = chain.steps[i]
        for name, mine, theirs in (
            ("state_hash", chain.state[i], reference.state[j]),
            ("scene_image_hash", chain.scene[i], reference.scene[j]),
            ("wrist_image_hash", chain.wrist[i], reference.wrist[j]),
        ):
            if mine != theirs:
                return ChainComparison(False, i, step, name)
        if i > 0 or chain.first_step == reference.first_step:
            a, b = chain.executed[i], reference.executed[j]
            if a.tobytes() != b.tobytes():
                return ChainComparison(False, i, step, "executed_action")
    if len(ref_rows) != len(chain):
        return ChainComparison(
            False, shorter, chain.first_step + shorter, f"length {len(chain)} != {len(ref_rows)}"
        )
    return ChainComparison(True, len(chain), None, "")


# ---- the chain sample's three verdicts (docs/design/contracts.md section 4) -------------------

IDENTICAL = "identical"
RENDER_TRANSIENT = "render_transient"
BROKEN = "broken"
CAMERAS = ("scene", "wrist")


@dataclass(frozen=True)
class SampleCheck:
    """A sample replay against its run: identical, a render transient, or broken.

    `classify` gives "identical", "broken", or a candidate transient
    (`verdict` "render_transient" with `odd` None): every state and executed
    action equal, one frame's image different, a frame nobody consumed. The
    candidate becomes a transient only when a model-free rendering names the
    odd one (`settled`); otherwise it is broken.
    """

    verdict: str
    what: str = ""  # why it is broken
    frames: tuple[tuple[int, str], ...] = ()  # (step, camera) whose image hashes differ
    compared_frames: int = 0  # image pairs compared (rows times cameras)
    odd: str | None = None  # "run" or "replay": the rendering the re-rendering found odd

    def settled(self, odd: str | None, why: str = "") -> SampleCheck:
        """The candidate after the re-rendering: a transient if it named the odd one."""
        if self.verdict != RENDER_TRANSIENT:
            return self
        if odd in ("run", "replay"):
            return replace(self, odd=odd)
        return replace(self, verdict=BROKEN, what=why or "the re-rendering named no odd frame")

    def as_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "what": self.what,
            "frames": [list(f) for f in self.frames],
            "compared_frames": self.compared_frames,
            "odd": self.odd,
        }


def classify(chain: Chain, reference: Chain, consumed: Callable[[int, str], bool]) -> SampleCheck:
    """Classify a sample replay's `chain` against its run's `reference`.

    Broken: a fixture or length difference, a state hash or executed action
    that differs at any step, image hashes that differ in more than one
    frame, or one differing frame that `consumed(step, camera)` says a policy
    call, a supervisor or a video used. Identical: nothing differs. Otherwise
    a candidate render transient (to be `settled`).
    """
    if not len(chain):
        return SampleCheck(BROKEN, "empty chain")
    if (
        chain.fixture_hash
        and reference.fixture_hash
        and chain.fixture_hash != reference.fixture_hash
    ):
        return SampleCheck(BROKEN, "fixture_hash")
    offset = chain.first_step - reference.first_step
    if offset < 0 or offset >= len(reference):
        return SampleCheck(BROKEN, "chain starts outside the reference")
    rows = len(reference) - offset
    if rows != len(chain):
        return SampleCheck(BROKEN, f"length {len(chain)} != {rows}")
    frames: list[tuple[int, str]] = []
    for i in range(len(chain)):
        j = offset + i
        step = chain.steps[i]
        if chain.state[i] != reference.state[j]:
            return SampleCheck(BROKEN, f"state_hash at step {step}")
        if (i > 0 or offset == 0) and chain.executed[i].tobytes() != reference.executed[
            j
        ].tobytes():
            return SampleCheck(BROKEN, f"executed_action at step {step}")
        if chain.scene[i] != reference.scene[j]:
            frames.append((step, "scene"))
        if chain.wrist[i] != reference.wrist[j]:
            frames.append((step, "wrist"))
    compared = len(chain) * len(CAMERAS)
    if not frames:
        return SampleCheck(IDENTICAL, compared_frames=compared)
    if len(frames) > 1:
        what = f"image hashes differ in {len(frames)} frames"
        return SampleCheck(BROKEN, what, tuple(frames), compared)
    step, camera = frames[0]
    if consumed(step, camera):
        what = f"the {camera} image of step {step} differs and was consumed"
        return SampleCheck(BROKEN, what, tuple(frames), compared)
    return SampleCheck(RENDER_TRANSIENT, "", tuple(frames), compared)
