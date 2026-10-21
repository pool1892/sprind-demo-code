"""The policy adapter (docs/design/contracts.md, section 6).

A policy sits behind one interface, so a second policy (the scripted
teleoperator today, pi0.5 tomorrow) runs through the same rollout unchanged.
The flags that carry the replay rules:

- `needs_noise`: the policy samples (X-VLA, pi0.5). Its noise comes from
  `contracts.identity.noise_for` with the policy's `noise_shape` and
  `noise_dtype`, drawn by the rollout and handed in (replay rule 2); a policy
  never draws its own. False for deterministic policies (ACT, the
  teleoperator).
- `row_independent`: a row's chunk does not depend on its position in the
  batch or on the other rows. Verified by the rows check (`make check-model`),
  never assumed (rule 3). When False, the row is part of the identity.
- `in_worker`: the policy runs inside the worker, per step, with the
  environment handle (`act_one`), and never goes through the inference server.
  Only the teleoperator has it; the flag is logged on the truth side.

Server policies implement `act` and `convert`; in-worker policies implement
`act_one`. The other method of each pair raises NotImplementedError.

A server policy's `act` receives the batch group's effective spec. An
in-worker policy (`InWorkerPolicy`) is constructed per episode or
continuation, learns its identity and starting step from `begin_episode`, and
exposes `done` and a `status` in words. The engine ends a continuation on
`done`; after `done` the policy keeps returning a hold with its last gripper
command (`contracts.live.hold_action`), never the adapter's open no-op, which
would drop a held object. `check_policy_flags` refuses an in-worker policy
without `done`, `status` or `begin_episode`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from fleetlab.contracts.env import Observation, SimEnv
    from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec

# The 7-vector LIBERO expects: position (3), rotation (3), gripper (1), in the
# convention of the episode's control mode. The gripper command is -1 (open),
# +1 (close) or 0 (hold the opening where it is). X-VLA sends only -1 and +1;
# the teleoperator also sends 0.
Action7 = npt.NDArray[np.float32]
ACTION_DIM = 7

# The flags every policy declares (section 15: replay-rule flags on every policy).
REPLAY_FLAGS: tuple[str, ...] = ("needs_noise", "row_independent", "in_worker")


class ControlMode(StrEnum):
    """How the 7-vector action is read: target poses (absolute) or changes of pose (relative).

    A per-episode setting with one home, `PolicySpec.control_mode`; the
    environment takes it from the identity at reset.
    """

    absolute = "absolute"
    relative = "relative"


@dataclass(frozen=True)
class NoiseSample:
    """One row's sampling noise for one policy call, from `noise_for`.

    `values` has the policy's `noise_shape` and `noise_dtype`. `noise_seed`
    and `chunk_index` say where it came from, so a continuation from a
    snapshot draws the same noise.
    """

    noise_seed: int
    chunk_index: int
    values: npt.NDArray[np.generic]


@dataclass(frozen=True)
class ActionChunk:
    """What one policy call returns for one row: `chunk_length` actions in the policy's own form.

    `actions` has shape (chunk_length, action width) in the policy's native
    output; `Policy.convert` turns it into `Action7`s for a control mode. How
    many of them the rollout executes is `policy.actions_executed_per_chunk`.
    """

    chunk_index: int
    actions: npt.NDArray[np.float32]


@runtime_checkable
class Policy(Protocol):
    """A policy as the rollout uses it. Load once, then call per batch or per step."""

    spec: PolicySpec
    chunk_length: int  # actions per call (X-VLA 30, SmolVLA 50, ACT 100, teleoperator 1)
    needs_noise: bool
    noise_shape: tuple[int, ...] | None  # one row's noise per call; None without noise
    noise_dtype: str | None  # numpy dtype name of that noise; None without noise
    row_independent: bool
    in_worker: bool
    image_views: int

    def load(self, device: str, precision: str) -> None: ...

    def act(
        self,
        batch: Sequence[Observation],
        instructions: Sequence[str],
        noise: Sequence[NoiseSample | None],
        spec: PolicySpec,
    ) -> list[ActionChunk]:
        """Server policies: one chunk per row of the batch.

        `spec` is the batch group's effective spec: the server groups requests
        by effective spec (a configuration fault may change the sampling steps
        or the image views for one robot), and the sampling loop reads its
        sampling steps and image views from it, not from `self.spec`.
        """
        ...

    def act_one(self, observation: Observation, env: SimEnv) -> Action7:
        """In-worker policies: the next action, with the environment's handle at hand."""
        ...

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        """The chunk as `Action7`s in the given control mode's convention."""
        ...


@runtime_checkable
class InWorkerPolicy(Policy, Protocol):
    """A policy that runs inside the worker, one step at a time (the teleoperator).

    Constructed per episode or continuation, so it carries no state across
    them. `done` turns true when it has finished its task; the engine then ends
    the continuation, and any further `act_one` returns a hold with the last
    gripper command. `status` says in words where it stands (for the log and
    for debugging; never shown to the agents).
    """

    done: bool
    status: str

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        """Called by the rollout before the first `act_one` of an episode or continuation:
        the identity (whose noise the human model draws from, `noise_for`) and the
        step it starts at (0, or a snapshot's step)."""
        ...


def check_policy_flags(policy: object) -> None:
    """Raise ValueError unless `policy` declares the replay-rule flags consistently.

    Every flag is present and a bool; a policy that needs noise names the shape
    and dtype of its noise and one that does not names neither; an in-worker
    policy has a chunk length of one and exposes `done` (a bool) and `status`
    (a string).
    """
    problems: list[str] = []
    for flag in REPLAY_FLAGS:
        value = getattr(policy, flag, None)
        if not isinstance(value, bool):
            problems.append(f"{flag} must be a bool, found {value!r}")
    needs_noise = getattr(policy, "needs_noise", None)
    shape = getattr(policy, "noise_shape", None)
    dtype = getattr(policy, "noise_dtype", None)
    if needs_noise is True:
        if not (isinstance(shape, tuple) and shape and all(isinstance(n, int) for n in shape)):
            problems.append("a policy that needs noise states its noise_shape")
        try:
            np.dtype(dtype)
        except TypeError:
            problems.append(f"noise_dtype {dtype!r} is not a numpy dtype")
        if dtype is None:
            problems.append("a policy that needs noise states its noise_dtype")
    if needs_noise is False and (shape is not None or dtype is not None):
        problems.append("a policy without noise has noise_shape and noise_dtype None")
    chunk_length = getattr(policy, "chunk_length", None)
    if not isinstance(chunk_length, int) or chunk_length < 1:
        problems.append(f"chunk_length must be a positive integer, found {chunk_length!r}")
    if getattr(policy, "in_worker", None) is True:
        if chunk_length != 1:
            problems.append("an in-worker policy acts one step at a time (chunk_length 1)")
        if not isinstance(getattr(policy, "done", None), bool):
            problems.append("an in-worker policy exposes done, a bool")
        if not isinstance(getattr(policy, "status", None), str):
            problems.append("an in-worker policy exposes status, a string")
        if not callable(getattr(policy, "begin_episode", None)):
            problems.append("an in-worker policy has begin_episode(identity, step)")
    if problems:
        raise ValueError(f"{type(policy).__name__}: " + "; ".join(problems))
