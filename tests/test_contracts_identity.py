"""Episode identity: seeds and noise are pure functions, the identity checks itself, and the
one hash is SHA-256 cut to 8 bytes.

docs/design/contracts.md, section 4.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import random
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.hashing import hash_array, hash_bytes, hash_json
from fleetlab.contracts.identity import (
    NOISE_SEED_LIMIT,
    REPLAY_RULES,
    SCENE_SEED_LIMIT,
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    noise_for,
    seeds_for,
    started_at_for,
)
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.version import CONTRACTS_VERSION
from fleetlab.contracts.vocabulary import Component

WORLD_SEED = 20261007


def identity(**changes: Any) -> EpisodeIdentity:
    scene, noise = seeds_for(WORLD_SEED, "r07", 3, 5)
    fields: dict[str, Any] = {
        "world_id": "w01",
        "episode_id": "e000123",
        "robot_id": "r07",
        "site_id": "s2",
        "day": 3,
        "episode_index": 5,
        "suite": "libero_spatial",
        "task_index": 4,
        "start_state": 17,
        "scene_seed": scene,
        "noise_seed": noise,
        "policy": PolicySpec(
            name="xvla-libero",
            revision="12e8783e",
            precision="float32",
            control_mode=ControlMode.absolute,
            actions_executed_per_chunk=30,
            sampling_steps=10,
            image_views=3,
        ),
        "batch_size": 4,
        "row": None,
        "image_size": 360,
        "max_steps": 280,
        "faults": [],
        "software": SoftwareVersion(
            lock_hash="abc", libero_patch_set="p1", mujoco_version="3.3.7", miopen_db_hash="def"
        ),
    }
    fields.update(changes)
    return EpisodeIdentity(**fields)


def camera_fault(fault_id: str = "f01", scope: str = "r07") -> FaultSpec:
    return FaultSpec(
        fault_id=fault_id,
        kind="camera_pose_offset",
        params={"degrees": 6.0},
        level=6.0,
        target=Target(scope=scope, component=Component.scene_camera),
        schedule=Schedule(onset_day=1),
    )


def test_seeds_are_a_pure_function_of_the_fields() -> None:
    first = seeds_for(WORLD_SEED, "r07", 3, 5)
    random.seed(1)
    np.random.seed(2)
    np.random.standard_normal(10)
    assert seeds_for(WORLD_SEED, "r07", 3, 5) == first
    # In another process, too: no state of this one enters.
    code = (
        "from fleetlab.contracts.identity import seeds_for; "
        f"print(*seeds_for({WORLD_SEED}, 'r07', 3, 5))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert tuple(int(x) for x in out.stdout.split()) == first


def test_seeds_are_distinct_over_a_grid_and_in_range() -> None:
    grid = itertools.product((1, 2), (f"r{n:02d}" for n in range(10)), range(20), range(10))
    scene_seeds, noise_seeds = [], []
    for world_seed, robot, day, index in grid:
        scene, noise = seeds_for(world_seed, robot, day, index)
        assert 0 <= scene < SCENE_SEED_LIMIT and 0 <= noise < NOISE_SEED_LIMIT
        scene_seeds.append(scene)
        noise_seeds.append(noise)
    assert len(set(scene_seeds)) == len(scene_seeds) == 4000
    assert len(set(noise_seeds)) == len(noise_seeds)
    assert not set(scene_seeds) & set(noise_seeds)


def test_every_field_moves_the_seeds() -> None:
    base = seeds_for(1, "r07", 3, 5)
    for other in (
        seeds_for(2, "r07", 3, 5),
        seeds_for(1, "r08", 3, 5),
        seeds_for(1, "r07", 4, 5),
        seeds_for(1, "r07", 3, 6),
    ):
        assert other[0] != base[0] and other[1] != base[1]


@pytest.mark.parametrize(
    "args",
    [
        (-1, "r07", 0, 0),
        (1, "robot_7", 0, 0),
        (1, "r07", -1, 0),
        (1, "r07", 0, True),
        (1.5, "r07", 0, 0),
    ],
)
def test_seeds_refuse_bad_fields(args: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        seeds_for(*args)


def test_noise_is_deterministic_float32_and_cast_afterwards() -> None:
    ep = identity()
    a = noise_for(ep, 2, (30, 20))
    assert a.dtype == np.float32 and a.shape == (30, 20)
    assert np.array_equal(a, noise_for(ep, 2, (30, 20)))
    assert not np.array_equal(a, noise_for(ep, 3, (30, 20)))
    assert not np.array_equal(a, noise_for(ep, 2, (30, 20), sample=1))
    half = noise_for(ep, 2, (30, 20), "float16")
    assert half.dtype == np.float16
    assert np.array_equal(half, a.astype(np.float16))
    other_episode = identity(episode_index=6, noise_seed=seeds_for(WORLD_SEED, "r07", 3, 6)[1])
    assert not np.array_equal(a, noise_for(other_episode, 2, (30, 20)))


def test_noise_is_the_same_in_another_process() -> None:
    ep = identity()
    code = (
        "import sys, numpy as np\n"
        "from fleetlab.contracts.identity import EpisodeIdentity, noise_for\n"
        "ep = EpisodeIdentity.model_validate_json(sys.stdin.read())\n"
        "sys.stdout.write(noise_for(ep, 7, (50, 32)).tobytes().hex())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=ep.model_dump_json(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert bytes.fromhex(out.stdout) == noise_for(ep, 7, (50, 32)).tobytes()


def test_the_seven_replay_rules_are_listed_once() -> None:
    names = [rule.name for rule in REPLAY_RULES]
    assert len(names) == 7 == len(set(names))
    seventh = REPLAY_RULES[6].description
    assert "per images per call (batch size times real image views)" in seventh
    assert all(rule.description and rule.source for rule in REPLAY_RULES)


def test_identity_checks_itself() -> None:
    ep = identity(faults=[camera_fault()])
    ep.check_seeds(WORLD_SEED)
    with pytest.raises(ValueError, match="not derived"):
        ep.check_seeds(WORLD_SEED + 1)
    assert ep.fault_ids == {"f01"}
    assert ep.software.contracts_version == CONTRACTS_VERSION
    assert EpisodeIdentity.model_validate_json(ep.model_dump_json()) == ep
    with pytest.raises(ValidationError, match="outside a batch"):
        identity(row=4)
    with pytest.raises(ValidationError, match="do not cover"):
        identity(faults=[camera_fault(scope="r08")])
    with pytest.raises(ValidationError, match="each fault once"):
        identity(faults=[camera_fault(), camera_fault()])
    with pytest.raises(ValidationError, match="level zero"):
        identity(faults=[camera_fault().model_copy(update={"level": 0.0})])
    with pytest.raises(ValidationError, match="not scheduled on day 3"):
        identity(faults=[camera_fault().model_copy(update={"schedule": Schedule(onset_day=4)})])
    ended = Schedule(onset_day=0, duration_days=3)  # days 0, 1, 2
    with pytest.raises(ValidationError, match="not scheduled"):
        identity(faults=[camera_fault().model_copy(update={"schedule": ended})])
    with pytest.raises(ValidationError):
        identity(episode_id="e12")
    with pytest.raises(ValidationError):
        identity(scene_seed=SCENE_SEED_LIMIT)
    assert identity().experiment_id is None
    assert identity(experiment_id="x0001").experiment_id == "x0001"
    with pytest.raises(ValidationError):
        identity(experiment_id="ab-test-camera")  # neutral ids only


def test_identity_needs_the_row_field_stated() -> None:
    assert identity(row=2).row == 2
    fields = identity().model_dump()
    del fields["row"]
    with pytest.raises(ValidationError):
        EpisodeIdentity.model_validate(fields)


def test_the_hash_is_the_first_eight_bytes_of_sha256() -> None:
    data = b"fleetlab"
    assert hash_bytes(data) == hashlib.sha256(data).digest()[:8].hex()
    assert len(hash_bytes(data)) == 16
    state = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert hash_array(state) == hash_bytes(state.tobytes())
    view = state[:, ::2]  # not contiguous: hashed as its contiguous bytes
    assert hash_array(view) == hash_bytes(np.ascontiguousarray(view).tobytes())
    assert hash_json({"b": 1, "a": [1, 2]}) == hash_json({"a": [1, 2], "b": 1})


def test_the_start_time_is_a_pure_function_of_the_identity() -> None:
    times = [started_at_for(identity(episode_index=i)) for i in range(14)]
    assert times[0] == "08:00:00" and times[13] == "15:25:42"
    assert times == sorted(times) and len(set(times)) == 14
    seconds = [
        (t.hour * 60 + t.minute) * 60 + t.second
        for t in (datetime.time.fromisoformat(x) for x in times)
    ]
    gaps = {b - a for a, b in itertools.pairwise(seconds)}
    assert gaps <= {2057, 2058}  # equal intervals over eight hours, whole seconds
    assert started_at_for(identity(episode_index=3)) == started_at_for(identity(episode_index=3))
    assert started_at_for(identity(episode_index=1), episodes_per_robot_day=2) == "12:00:00"
    other_day = identity(day=4, episode_index=5, faults=[])
    assert started_at_for(other_day) == times[5]  # the shift is the same every day


def test_arm_episodes_start_at_four_every_three_minutes_until_midnight() -> None:
    # Experiment arms take the indices after the as-run ones: from 16:00, every
    # three minutes whatever the as-run spacing, 160 a robot and day.
    assert started_at_for(identity(episode_index=14)) == "16:00:00"
    assert started_at_for(identity(episode_index=15)) == "16:03:00"
    assert started_at_for(identity(episode_index=14 + 159)) == "23:57:00"
    with pytest.raises(ValueError, match="after 23:59:59"):
        started_at_for(identity(episode_index=14 + 160))
    assert started_at_for(identity(episode_index=7), episodes_per_robot_day=4) == "16:09:00"
    assert started_at_for(identity(episode_index=3), episodes_per_robot_day=4) == "14:00:00"
    with pytest.raises(ValueError, match="after 23:59:59"):
        started_at_for(identity(episode_index=4 + 160), episodes_per_robot_day=4)
    with pytest.raises(ValueError, match="positive integer"):
        started_at_for(identity(), episodes_per_robot_day=0)
