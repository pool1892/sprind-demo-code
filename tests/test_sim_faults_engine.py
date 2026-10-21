"""The stage-3 fault types through the engine's own LIBERO adapter and loop (`make check-sim`).

Every new kind at its strongest calibration point, plus the
scene camera's tilt and the release that carries two faults, runs through
`engine.episode.run_job` on `LiberoSimEnv` with the scripted sway policy (a
server policy without a model: it reads the end effector, not the images,
and closes and opens the gripper in every chunk). Each faulted episode run a
second time in the same process gives the identical chain (state, both
images and the executed action at every step, and the fixture hash); the
types whose mechanism the sway policy can feel change the chain, the image
types and the inert calibration record leave it as it is healthy (the chain
holds the images as rendered, before the observation hook); the
false-positive detector logs a success on the failed attempt and nothing
else. A healthy episode run after all of them has the chain of the same
episode before them and in a fresh process: every cell hook was undone.
"""

from __future__ import annotations

from typing import Any

import pytest
from fault_toys import spec_for
from sim_helpers import in_fresh_process

pytestmark = pytest.mark.sim

SIZE = 128
TASK = 4  # a cabinet with a drawer, a stove, two bowls, a plate

# (kind, calibration point index, whether the sway policy's chain must change)
CASES: tuple[tuple[str, int, bool], ...] = (
    ("lens_contamination", 3, False),
    ("camera_latency", 3, False),
    ("gripper_force_loss", 2, True),
    ("gripper_sticky_release", 3, True),
    ("arm_torque_loss", 3, True),
    ("controller_overload", 3, True),
    ("calibration_update", 0, False),
    ("image_pipeline_release", 3, False),
    ("table_height_change", 2, True),
    ("light_color_shift", 1, True),
    ("success_false_positive", 2, False),
    ("camera_pose_offset", 7, True),  # 6 degrees tilted down
    ("software_release", 1, True),  # a 14-step delay and 10 actions per chunk
)


def _identity(faults: Any = (), number: int = 1) -> Any:
    from fleetlab.engine import manifests, software
    from fleetlab.engine.scripted import SWAY_SPEC

    return manifests.identity(
        world_seed=5,
        episode_number=number,
        robot=1,
        site=1,
        day=0,
        episode_index=2,
        suite="libero_spatial",
        task_index=TASK,
        start_state=2,
        policy=SWAY_SPEC,
        batch_size=1,
        software=software.current("none"),
        image_size=SIZE,
        max_steps=60,
        faults=list(faults),
    )


def _fault(kind: str, index: int) -> Any:
    """The calibration point's fault, at r01 or s1: the robot and site of `_identity`."""
    import fleetlab.faults as faults

    return spec_for(kind, faults.get(kind).calibration[index])  # type: ignore[attr-defined]


def _context() -> Any:
    from fleetlab.engine.episode import DirectSource, EpisodeContext, MemoryStore
    from fleetlab.engine.scripted import SwayPolicy
    from fleetlab.faults import REGISTRY
    from fleetlab.sim.libero_sim import LiberoCheck

    return EpisodeContext(
        fault_types=REGISTRY.get,
        make_check=LiberoCheck,
        source=DirectSource(SwayPolicy()),
        store=MemoryStore(),
    )


def healthy_chain() -> tuple[str, ...]:
    from fleetlab.contracts.rollout import Job
    from fleetlab.engine.episode import run_job
    from fleetlab.sim.libero_sim import LiberoSimEnv

    env = LiberoSimEnv("libero_spatial", TASK, SIZE)
    record = run_job(Job(identity=_identity()), env, _context())
    env.close()
    c = record.chain
    return (c.fixture_hash, *c.state, *c.scene, *c.wrist)


def every_case() -> dict[str, Any]:
    from fleetlab.contracts.rollout import Job
    from fleetlab.engine.chain import compare
    from fleetlab.engine.episode import run_job
    from fleetlab.sim.libero_sim import LiberoSimEnv

    env = LiberoSimEnv("libero_spatial", TASK, SIZE)
    ctx = _context()
    healthy = run_job(Job(identity=_identity()), env, ctx)
    out: dict[str, Any] = {}
    for number, (kind, index, _) in enumerate(CASES, start=2):
        job = Job(identity=_identity([_fault(kind, index)], number))
        first = run_job(job, env, ctx)
        second = run_job(job, env, ctx)
        out[kind] = {
            "repeat": compare(second.chain, first.chain).identical,
            "as_healthy": compare(first.chain, healthy.chain).identical,
            "true": first.true_success,
            "visible": first.visible_success,
            "steps": first.n_steps,
        }
    after = run_job(Job(identity=_identity()), env, ctx)
    env.close()
    c = after.chain
    out["healthy_after"] = compare(after.chain, healthy.chain).identical
    out["healthy"] = (c.fixture_hash, *c.state, *c.scene, *c.wrist)
    return out


def test_every_new_kind_runs_through_the_engine_and_leaves_nothing_behind() -> None:
    result = in_fresh_process(every_case)
    for kind, _, changes in CASES:
        case = result[kind]
        assert case["repeat"], kind
        assert case["as_healthy"] is not changes, kind
        assert not case["true"], kind  # the sway policy never succeeds
    assert result["success_false_positive"]["visible"]  # a failed placement logged a success
    assert not any(result[k]["visible"] for k, _, _ in CASES if k != "success_false_positive")
    assert result["healthy_after"]
    assert tuple(result["healthy"]) == in_fresh_process(healthy_chain)
