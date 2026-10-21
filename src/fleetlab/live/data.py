"""Offline development data only: visible telemetry as X, verdicts solely as labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from fleetlab.contracts.live import StepTelemetry
from fleetlab.live.features import FEATURE_NAMES, FeatureWindow, FloatArray

FIT_ROBOTS = ("r01", "r02", "r05", "r06", "r09", "r10")
CALIBRATION_ROBOTS = ("r03", "r07", "r11")
TEST_ROBOTS = ("r04", "r08", "r12")
HORIZONS = (20, 40, 100)
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class Dataset:
    x: FloatArray
    steps: IntArray
    episode: IntArray
    episodes: list[dict[str, Any]]
    provenance: list[dict[str, Any]]

    @property
    def terminal(self) -> IntArray:
        return np.asarray([e["terminal"] for e in self.episodes], dtype=np.int64)[self.episode]

    @property
    def failures(self) -> BoolArray:
        return np.asarray([e["failure"] for e in self.episodes], dtype=np.bool_)[self.episode]

    def labels(self, horizon: int) -> BoolArray:
        return self.failures & (self.terminal - self.steps <= horizon)

    def robot_mask(self, robots: tuple[str, ...]) -> BoolArray:
        return np.asarray([e["robot"] in robots for e in self.episodes], dtype=np.bool_)[
            self.episode
        ]

    def subset(self, mask: BoolArray) -> Dataset:
        indices = np.unique(self.episode[mask])
        mapping = np.full(len(self.episodes), -1, dtype=np.int64)
        mapping[indices] = np.arange(len(indices))
        return Dataset(
            self.x[mask],
            self.steps[mask],
            mapping[self.episode[mask]],
            [self.episodes[i] for i in indices],
            self.provenance,
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            x=self.x,
            steps=self.steps,
            episode=self.episode,
            metadata=json.dumps(
                {
                    "version": 1,
                    "features": FEATURE_NAMES,
                    "episodes": self.episodes,
                    "provenance": self.provenance,
                }
            ),
        )

    @classmethod
    def load(cls, path: Path) -> Dataset:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["version"] != 1 or meta["features"] != list(FEATURE_NAMES):
                raise ValueError("incompatible cached feature schema")
            return cls(
                data["x"], data["steps"], data["episode"], meta["episodes"], meta["provenance"]
            )


def concatenate(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("no datasets")
    episodes: list[dict[str, Any]] = []
    index: list[IntArray] = []
    for data in datasets:
        index.append(data.episode + len(episodes))
        episodes.extend(data.episodes)
    keys = [(e["world"], e["run"], e["id"]) for e in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episodes across datasets")
    return Dataset(
        np.concatenate([d.x for d in datasets]),
        np.concatenate([d.steps for d in datasets]),
        np.concatenate(index),
        episodes,
        [p for d in datasets for p in d.provenance],
    )


def telemetry_rows(path: Path) -> list[StepTelemetry]:
    """Read command rows only. The terminal null command is never a model input."""
    rows = pq.read_table(path).to_pylist()
    if not rows or rows[-1]["commanded_action"] is not None:
        raise ValueError(f"missing terminal row: {path}")
    if [r["step"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"nonconsecutive step log: {path}")
    output = []
    for r in rows[:-1]:
        if r["commanded_action"] is None:
            raise ValueError(f"null nonterminal command: {path}")
        output.append(
            StepTelemetry(
                r["step"],
                r["time_s"],
                np.asarray(r["joint_pos"], np.float64),
                np.asarray(r["joint_vel"], np.float64),
                np.asarray(r["gripper_qpos"], np.float64),
                np.asarray(r["eef_pos"], np.float64),
                np.asarray(r["eef_quat"], np.float64),
                np.asarray(r["commanded_action"], np.float32),
                -1 if r["chunk_index"] is None else r["chunk_index"],
                -1 if r["chunk_step"] is None else r["chunk_step"],
            )
        )
    return output


def build_dataset(world: str, runs: list[Path]) -> Dataset:
    features: list[FloatArray] = []
    steps: list[int] = []
    indices: list[int] = []
    episodes: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for run in sorted(runs):
        digest = hashlib.sha256()
        episode_file = run / "log/episodes.parquet"
        verdict_file = run / "truth/verdicts.parquet"
        for path in (episode_file, verdict_file):
            digest.update(path.relative_to(run).as_posix().encode())
            digest.update(path.read_bytes())
        verdicts = {v["episode_id"]: v for v in pq.read_table(verdict_file).to_pylist()}
        rows = sorted(pq.read_table(episode_file).to_pylist(), key=lambda e: e["episode_id"])
        for e in rows:
            v = verdicts[e["episode_id"]]
            if v["true_reason"] not in ("placed", "step limit", "LIBERO only"):
                raise ValueError(f"not an unassisted placed-check episode: {v['true_reason']}")
            path = run / "log/steps" / f"{e['episode_id']}.parquet"
            digest.update(path.relative_to(run).as_posix().encode())
            digest.update(path.read_bytes())
            telemetry = telemetry_rows(path)
            if len(telemetry) != e["steps"] or not telemetry:
                raise ValueError(f"episode/steps terminal mismatch: {path}")
            if bool(v["true_success"]) != (v["success_step"] is not None):
                raise ValueError(f"inconsistent success label: {path}")
            window = FeatureWindow()
            for t in telemetry:
                features.append(window.update(t))
                steps.append(t.step)
                indices.append(len(episodes))
            episodes.append(
                {
                    "world": world,
                    "run": run.name,
                    "id": e["episode_id"],
                    "robot": e["robot_id"],
                    "site": e["site_id"],
                    "day": e["day"],
                    "terminal": e["steps"],
                    "failure": not v["true_success"],
                    "reason": v["true_reason"],
                }
            )
        provenance.append(
            {
                "world": world,
                "run": str(run.resolve()),
                "episodes": len(rows),
                "sha256": digest.hexdigest(),
            }
        )
    if not episodes:
        raise ValueError("no episodes in dataset")
    return Dataset(
        np.stack(features),
        np.asarray(steps, np.int64),
        np.asarray(indices, np.int64),
        episodes,
        provenance,
    )


def development_runs(root: Path) -> list[Path]:
    """Refuse an unfinished world; do not race its runner's files."""
    from fleetlab.worlds.layout import WorldPaths

    paths = WorldPaths.from_root(root)
    problem = paths.readiness_problem
    if problem:
        raise ValueError(f"development world is not ready: {root}: {problem}")
    runs = [paths.observed, paths.later]
    if not all((r / "truth/verdicts.parquet").is_file() for r in runs):
        raise ValueError(f"ready world lacks expected runs: {root}")
    return runs
