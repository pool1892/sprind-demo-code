"""Executed horizons with deterministic observations/actions; no model or simulator.

The fake scene advances one observation per command. Actions identify the policy call,
the observation it received and the offset within its 30-action prediction. This tests
the real jobs loop, recorder and replay files, not robot performance or inference cost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from fleetlab.contracts.env import Camera, Observation
from fleetlab.contracts.identity import EpisodeIdentity, SoftwareVersion
from fleetlab.contracts.jobs import JobProgram, RequestState
from fleetlab.jobs import runner, tables
from fleetlab.jobs.replay import record_problems
from fleetlab.jobs.starts import start_identity


class FakeScene:
    def __init__(self, delivery_at: int = 62) -> None:
        self.delivery_at = delivery_at
        self.t = 0
        self.actions: list[npt.NDArray[np.float32]] = []
        self._rs = self

    def observation(self) -> Observation:
        image = np.full((2, 2, 3), self.t, dtype=np.uint8)
        return {
            "pixels": {"image": image, "image2": image},
            "robot_state": {
                "eef": {
                    "pos": np.array([self.t, 0, 0], np.float64),
                    "quat": np.array([0, 0, 0, 1], np.float64),
                    "mat": np.eye(3),
                },
                "gripper": {"qpos": np.zeros(2), "qvel": np.zeros(2)},
                "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
            },
            "task_description": "fixture",
            "step": self.t,
        }

    def reset(self, identity: EpisodeIdentity, faults: list[Any]) -> Observation:
        self.t = 0
        self.actions = []
        return self.observation()

    def step(self, action: npt.NDArray[np.float32]) -> Any:
        self.actions.append(action.copy())
        self.t += 1
        return SimpleNamespace(observation=self.observation())

    def _index(self) -> Any:
        return SimpleNamespace(object_dofs={})

    def privileged_state(self) -> Any:
        return SimpleNamespace(
            gripper_opening=0.08 if self.t >= self.delivery_at else 0.0,
            object_poses={},
            object_velocities={},
            contacts=[],
        )

    def state_hash(self) -> bytes:
        return hashlib.blake2b(
            np.asarray(self.actions, np.float32).tobytes(), digest_size=8
        ).digest()


class FakeGoals:
    named = ("flat_stove_1",)

    def __init__(self, scene: FakeScene, goals: Any) -> None:
        self.scene = scene

    def holds(self) -> tuple[bool, ...]:
        return (self.scene.t >= 3,)


class ThirtyActions:
    """Always predicts 30, even when the runner requests a shorter executed horizon."""

    def __init__(self) -> None:
        self.spec = runner.HoldSource().spec
        self.seen: list[tuple[int, str, int, int]] = []
        self.calls = 0

    def chunk(
        self, observation: Observation, instruction: str, identity: EpisodeIdentity, index: int
    ) -> npt.NDArray[np.float32]:
        observed = int(observation["robot_state"]["eef"]["pos"][0])
        self.seen.append((index, instruction, observed, identity.policy.actions_executed_per_chunk))
        self.calls += 1
        actions = np.zeros((30, 7), np.float32)
        actions[:, 0] = index
        actions[:, 1] = observed
        actions[:, 2] = np.arange(30)
        actions[:, 6] = -1
        return actions


def program(**settings: Any) -> JobProgram:
    return JobProgram.model_validate(
        {
            "name": "cadence",
            "purpose": "exercise step and tail boundaries",
            "steps": [
                {
                    "kind": "policy",
                    "instruction": "first",
                    "until": {"goal": 0, "steps": None},
                    "max_steps": 20,
                    "segment": None,
                    "steps": None,
                },
                {
                    "kind": "policy",
                    "instruction": "second",
                    "until": {"goal": None, "steps": 7},
                    "max_steps": 20,
                    "segment": None,
                    "steps": None,
                },
            ],
            "notes": "",
            **settings,
        }
    )


def state() -> RequestState:
    return RequestState.model_validate(
        {
            "request_text": "turn on stove",
            "requested_goals": [{"predicate": "Turnon", "objects": ["flat_stove_1"]}],
            "base_task": 6,
            "order_rules": [],
            "budget": {
                "starts_per_request": 20,
                "requests_per_session": 4,
                "sessions": 6,
                "starts_per_investigation": 400,
            },
            "acceptance": None,
            "message_log": [],
        }
    )


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> tuple[runner.JobRunner, ThirtyActions, FakeScene]:
    scene = FakeScene()
    source = ThirtyActions()
    monkeypatch.setattr(runner, "SceneGoals", FakeGoals)
    monkeypatch.setattr(runner, "make_env", lambda bench, size: scene)
    software = SoftwareVersion(
        lock_hash="fixture",
        libero_patch_set="fixture",
        mujoco_version="fixture",
        miopen_db_hash="fixture",
    )
    suite: Any = SimpleNamespace(bench=tuple(range(10)))
    job_runner = runner.JobRunner(
        source,
        software,
        suite=suite,
        log=lambda line: None,
    )
    return job_runner, source, scene


def play(job_runner: runner.JobRunner, candidate: JobProgram) -> runner.Played:
    return job_runner.play(
        candidate,
        state(),
        start_identity(6, 0),
        "j0001",
        "e000101",
        record_video=True,
    )


@pytest.mark.parametrize(
    ("horizon", "observed_at"),
    [
        (5, [0, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58]),
        (10, [0, 3, 13, 23, 33, 43, 53]),
        (30, [0, 3, 33]),
    ],
)
def test_prefix_cadence_step_discard_tail_continuity_and_recording(
    setup: tuple[runner.JobRunner, ThirtyActions, FakeScene],
    horizon: int,
    observed_at: list[int],
) -> None:
    job_runner, source, scene = setup
    result = play(job_runner, program(actions_per_chunk=horizon))
    assert source.seen == [
        (i, "first" if i == 0 else "second", t, horizon) for i, t in enumerate(observed_at)
    ]
    expected = []
    for i, (begin, end) in enumerate(zip(observed_at, [*observed_at[1:], 62], strict=True)):
        expected.extend([[i, begin, k, 0, 0, 0, -1] for k in range(end - begin)])
    np.testing.assert_array_equal(result.executed, expected)
    np.testing.assert_array_equal(scene.actions, expected)
    assert result.stats["policy_calls"] == len(observed_at)
    assert result.identity.policy.actions_executed_per_chunk == horizon
    assert source.spec.actions_executed_per_chunk == 30  # shared source stays unchanged
    assert [s.end_control_step for s in result.episode.program_steps] == [3, 10]
    assert result.episode.control_steps == 62 and result.episode.tail_steps == 52
    assert result.episode.delivered
    rows = result.recorder.robot
    assert len(rows) == len(result.hashes) == 63
    assert [row["step"] for row in rows] == list(range(63))
    np.testing.assert_array_equal([row["commanded_action"] for row in rows[:-1]], expected)
    assert [row["phase"] for row in rows] == ["program"] * 10 + ["tail"] * 52 + ["end"]
    framed = [(row["step"], row["frame_index"]) for row in rows if row["frame_index"] is not None]
    assert framed == [(t, i) for i, t in enumerate([*range(0, 61, 4), 62])]
    for camera in Camera:
        assert [int(f[0, 0, 0]) for f in result.frames[camera]] == [*range(0, 61, 4), 62]


@pytest.mark.parametrize("delivery_at", [10, 1000])
def test_tail_stops_at_delivery_or_its_fixed_budget(
    setup: tuple[runner.JobRunner, ThirtyActions, FakeScene],
    delivery_at: int,
) -> None:
    job_runner, source, scene = setup
    scene.delivery_at = delivery_at
    result = play(job_runner, program(actions_per_chunk=5))
    end = 10 if delivery_at == 10 else 70
    assert result.episode.control_steps == end
    assert result.episode.tail_steps == end - 10
    assert [s[2] for s in source.seen] == [0, *range(3, end, 5)]


def test_absent_null_and_explicit_thirty_keep_the_legacy_execution(
    setup: tuple[runner.JobRunner, ThirtyActions, FakeScene],
) -> None:
    job_runner, source, _ = setup
    results = []
    for settings in ({}, {"actions_per_chunk": None}, {"actions_per_chunk": 30}):
        source.seen.clear()
        result = play(job_runner, program(**settings))
        assert source.seen == [(0, "first", 0, 30), (1, "second", 3, 30), (2, "second", 33, 30)]
        results.append(result)
    for result in results[1:]:
        np.testing.assert_array_equal(result.executed, results[0].executed)
        assert result.hashes == results[0].hashes
        assert result.identity == results[0].identity
        assert result.recorder.robot == results[0].recorder.robot
        assert result.episode.model_dump(exclude={"program_digest"}) == results[
            0
        ].episode.model_dump(exclude={"program_digest"})


@pytest.mark.parametrize("invalid", [4, 31, 5.5])
def test_invalid_horizons_are_refused(invalid: Any) -> None:
    with pytest.raises(ValidationError, match="actions_per_chunk"):
        program(actions_per_chunk=invalid)


def test_saved_episode_provenance_and_replay_actions_use_the_effective_horizon(
    setup: tuple[runner.JobRunner, ThirtyActions, FakeScene],
    tmp_path: Path,
) -> None:
    job_runner, _, scene = setup
    for i, horizon in enumerate([5, 30, 10]):
        [episode] = job_runner.run_starts(
            tmp_path,
            f"j{i + 1:04d}",
            program(actions_per_chunk=horizon),
            state(),
            [start_identity(6, 0)],
            first_episode=101 + i,
            record_video=False,
        )
        actions, hashes, identity_json = tables.load_replay_record(tmp_path, episode.episode_id)
        identity = EpisodeIdentity.model_validate_json(identity_json)
        assert identity.policy.actions_executed_per_chunk == horizon
        np.testing.assert_array_equal(actions, scene.actions)
        assert record_problems(episode.episode_id, actions, hashes, identity, episode) == []
        record = json.loads((tmp_path / tables.RUN_RECORD).read_text())
        assert record["episodes"][episode.episode_id]["policy"] == identity.policy.model_dump(
            mode="json"
        )


@pytest.mark.parametrize("horizon", [5, 10, 30])
def test_xvla_source_passes_effective_spec_to_policy_without_changing_prediction_size(
    setup: tuple[runner.JobRunner, ThirtyActions, FakeScene],
    horizon: int,
) -> None:
    job_runner, source, scene = setup
    identity = play(job_runner, program(actions_per_chunk=horizon)).identity
    seen: list[Any] = []

    def predict(observations: Any, instructions: Any, noise: Any, spec: Any) -> Any:
        seen.append(spec)
        return np.arange(210, dtype=np.float32).reshape(1, 30, 7)

    policy = SimpleNamespace(
        spec=source.spec,
        noise_shape=(30, 7),
        noise_dtype=np.float32,
        act=predict,
        convert=lambda actions, mode: actions,
    )
    xvla = runner.XVLASource(policy)
    actions = xvla.chunk(scene.observation(), "first", identity, 0)
    assert seen == [identity.policy]
    assert actions.shape == (horizon, 7)
    np.testing.assert_array_equal(
        actions, np.arange(210, dtype=np.float32).reshape(30, 7)[:horizon]
    )
    assert xvla.calls == 1 and policy.spec.actions_executed_per_chunk == 30


def test_batch_accepts_and_executes_a_shorter_horizon(
    setup: tuple[runner.JobRunner, ThirtyActions, FakeScene],
    tmp_path: Path,
) -> None:
    job_runner, source, _ = setup
    candidate = program(actions_per_chunk=10)
    batch = {
        "format": runner.BATCH_FORMAT,
        "state": state().model_dump(mode="json"),
        "jobs": [
            {
                "job_id": "j0001",
                "request": {
                    "program": candidate.model_dump(mode="json"),
                    "starts": [start_identity(6, 0).model_dump(mode="json")],
                    "record_video": False,
                },
            }
        ],
    }
    [result] = runner.run_batch(batch, tmp_path, job_runner)["jobs"]
    assert result["status"] == "done" and result["problems"] == []
    assert result["episode_ids"] == ["e000101"]
    assert [s[2] for s in source.seen] == [0, 3, 13, 23, 33, 43, 53]
    [episode] = tables.read_episodes(tmp_path)
    assert episode.program_digest == candidate.digest and episode.delivered
