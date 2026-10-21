"""X-VLA episodes recorded by the walking skeleton, and the state hash they were recorded with.

The skeleton (spikes/skeleton) stored, for every episode, the
actions the simulator executed and a hash of MuJoCo's complete integration
state after the reset and after every step (`truth/chain/<episode>.npz`), with
the episode's identity in `truth/episodes.jsonl`. Those runs are read here, not
changed. The hash is the skeleton's: the first eight bytes of the SHA-256 of
`mj_getState(..., mjSTATE_INTEGRATION)` as an unsigned 64-bit number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RUNS = Path.home() / ".cache" / "robots-probes" / "skeleton" / "runs"


@dataclass(frozen=True)
class RecordedEpisode:
    run: str
    episode_id: str
    suite: str
    task: int
    init: int
    scene_seed: int
    success: bool
    faults: list[dict[str, Any]]
    actions: np.ndarray  # (steps, 7) float32, as executed
    state_hashes: np.ndarray  # (steps + 1,) uint64: after the reset, then after every step


def load_index(run_dir: Path) -> list[dict[str, Any]]:
    with open(run_dir / "truth" / "episodes.jsonl") as f:
        return [json.loads(line) for line in f]


def load_episode(run_dir: Path, episode_id: str, index: list[dict[str, Any]] | None = None) -> RecordedEpisode:
    rows = index if index is not None else load_index(run_dir)
    row = next(r for r in rows if r["episode_id"] == episode_id)
    chain = np.load(run_dir / "truth" / "chain" / f"{episode_id}.npz")
    return RecordedEpisode(
        run=run_dir.name,
        episode_id=episode_id,
        suite=row["suite"],
        task=int(row["task"]),
        init=int(row["init"]),
        scene_seed=int(row["scene_seed"]),
        success=bool(row["success"]),
        faults=list(row.get("faults") or []),
        actions=np.asarray(chain["exe"], np.float32),
        state_hashes=np.asarray(chain["state_hash"], np.uint64),
    )


def mj_state(env: Any) -> np.ndarray:
    """MuJoCo's complete integration state, warm start included. `env` is fleetlab's LiberoEnv."""
    import mujoco

    sim = env._env.sim
    m, d = sim.model._model, sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    out = np.empty(mujoco.mj_stateSize(m, spec), np.float64)
    mujoco.mj_getState(m, d, out, spec)
    return out


def h64(array: Any) -> np.uint64:
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()[:8]
    return np.frombuffer(digest, np.uint64)[0]


def state_hash(env: Any) -> np.uint64:
    return h64(mj_state(env))
