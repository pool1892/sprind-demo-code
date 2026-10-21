"""The engine's LIBERO adapter without a model (`make check-sim`).

Reset determinism through the adapter, snapshot and restore bit-identical in
absolute control, the cell hook undone (the start of a healthy episode after
a faulted one equals a healthy start in a fresh process), and one rendering
context at a time with two adapters in one process. The policy is a scripted
in-worker policy, so no model is loaded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sim_helpers import in_fresh_process

from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.contracts.vocabulary import ConditionKey
from fleetlab.engine import manifests, software
from fleetlab.engine.episode import EngineError, EpisodeContext, MemoryStore, run_job
from fleetlab.engine.scripted import LIFT_SPEC, LiftPolicy
from fleetlab.engine.standin_faults import STANDIN
from fleetlab.sim.libero_sim import LiberoCheck, LiberoSimEnv

pytestmark = pytest.mark.sim

SIZE = 128


def identity(start: int = 0, task: int = 0, faults: Any = (), number: int = 1) -> Any:
    return manifests.identity(
        world_seed=5,
        episode_number=number,
        robot=2,
        site=1,
        day=0,
        episode_index=start,
        suite="libero_spatial",
        task_index=task,
        start_state=start,
        policy=LIFT_SPEC,
        batch_size=1,
        software=software.current("none"),
        image_size=SIZE,
        max_steps=60,
        faults=list(faults),
    )


def context(store: MemoryStore | None = None, steps: tuple[int, ...] = ()) -> EpisodeContext:
    return EpisodeContext(
        fault_types=STANDIN.get,
        make_check=LiberoCheck,
        teleoperator=LiftPolicy,
        store=store or MemoryStore(),
        snapshot_steps=steps,
    )


def start_hashes(faulted_first: bool) -> tuple[str, str, str, str]:
    env = LiberoSimEnv("libero_spatial", 0, SIZE)
    if faulted_first:
        cam = manifests.camera_fault(1, "r02", 9.0)
        record = run_job(Job(identity=identity(faults=[cam])), env, context())
        assert record.chain.scene[0]  # the faulted episode ran
    record = run_job(Job(identity=identity(start=3, number=2)), env, context())
    env.close()
    c = record.chain
    return c.fixture_hash, c.state[0], c.scene[0], c.wrist[0]


def test_a_cell_hook_is_undone() -> None:
    fresh = in_fresh_process(start_hashes, False)
    after_fault = in_fresh_process(start_hashes, True)
    assert after_fault == fresh


def test_the_camera_hook_changes_the_scene_image() -> None:
    env = LiberoSimEnv("libero_spatial", 0, SIZE)
    healthy = run_job(Job(identity=identity()), env, context())
    cam = manifests.camera_fault(1, "r02", 6.0)
    faulted = run_job(Job(identity=identity(faults=[cam])), env, context())
    env.close()
    assert healthy.chain.fixture_hash != faulted.chain.fixture_hash  # the camera's pose
    assert healthy.chain.scene[0] != faulted.chain.scene[0]
    assert healthy.chain.wrist[0] == faulted.chain.wrist[0]


def run_twice_and_restore(step: int) -> dict[str, Any]:
    """With a server policy (the observation after a restore is the saved one, replay rule 5).

    An in-worker policy reads the live simulator, which a restore recomputes
    (`mj_forward`): its continuation replays itself, but is not the
    uninterrupted episode (the take-over tests check the former).
    """
    from fleetlab.engine.episode import DirectSource
    from fleetlab.engine.scripted import SWAY_SPEC, SwayPolicy

    store = MemoryStore()
    ctx = context(store, (step,))
    ctx.source, ctx.teleoperator = DirectSource(SwayPolicy()), None
    env = LiberoSimEnv("libero_spatial", 1, SIZE)
    delay = manifests.delay_fault(2, "r02", 5)
    sway = {"policy": SWAY_SPEC, "batch_size": 1}
    job = Job(identity=identity(task=1, faults=[delay]).model_copy(update=sway))
    original = run_job(job, env, ctx)
    # something else on the same simulator in between: another start state, another fault
    cam = manifests.camera_fault(1, "r02", 9.0)
    other = identity(start=4, task=1, faults=[cam], number=3).model_copy(update=sway)
    ctx.snapshot_steps = ()
    run_job(Job(identity=other), env, ctx)
    ref = original.snapshots[0]
    continued = run_job(
        Job(identity=job.identity, mode=JobMode.replay, from_snapshot=ref), env, ctx
    )
    again = run_job(job, env, ctx)
    env.close()
    from fleetlab.engine.chain import compare

    return {
        "again": compare(again.chain, original.chain).identical,
        "restore": compare(continued.chain, original.chain).identical,
        "restore_first": continued.chain.first_step,
        "steps": original.n_steps,
    }


def test_reset_and_restore_are_bit_identical_in_absolute_control() -> None:
    result = in_fresh_process(run_twice_and_restore, 23)
    assert result["again"], "the same identity twice on one simulator"
    assert result["restore_first"] == 23
    assert result["restore"], "a restore from step 23 continues as the original did"


def two_adapters() -> tuple[list[str], list[str]]:
    a = LiberoSimEnv("libero_spatial", 0, SIZE)
    first = run_job(Job(identity=identity()), a, context()).chain.scene[:10]
    b = LiberoSimEnv("libero_spatial", 2, SIZE)  # a second context, created last
    run_job(Job(identity=identity(task=2, number=4)), b, context())
    second = run_job(Job(identity=identity()), a, context()).chain.scene[:10]
    a.close()
    b.close()
    return first, second


def test_one_rendering_context_at_a_time() -> None:
    first, second = in_fresh_process(two_adapters)
    assert first == second


def test_the_lean_snapshot_holds_no_cell_state() -> None:
    env = LiberoSimEnv("libero_spatial", 4, SIZE)  # a scene with furniture
    env.reset(identity(task=4), [])
    for _ in range(5):
        env.step(np.array([0.0, 0.0, 1.1, 3.14, 0.0, 0.0, -1.0], np.float32))
    snap = env.snapshot()
    import pickle

    state = pickle.loads(snap.state)
    nbody = env._sim.model._model.nbody
    env.close()
    assert 0 < len(state["fixtures"]) < nbody  # the fixture rows only, not every body
    assert state["goal_qvel"] and state["current_action"]
    flat = repr(state.keys())
    assert "actuator" not in flat and "groups" not in flat


def furniture_and_back() -> dict[str, Any]:
    from fleetlab.faults import REGISTRY

    store = MemoryStore()
    shift = FaultSpec(
        fault_id="f03",
        kind="furniture_shift",
        params={},
        level=4.0,
        target=Target(scope="s1", condition_key=ConditionKey.placement),
        schedule=Schedule(onset_day=0),
        nominal=[0.0, 0.0],
    )
    REGISTRY.check(shift)
    from fleetlab.engine.episode import DirectSource
    from fleetlab.engine.scripted import SWAY_SPEC, SwayPolicy

    ctx = EpisodeContext(
        fault_types=REGISTRY.get,
        make_check=LiberoCheck,
        source=DirectSource(SwayPolicy()),
        store=store,
        snapshot_steps=(12,),
    )
    env = LiberoSimEnv("libero_spatial", 4, SIZE)
    sway = {"policy": SWAY_SPEC, "batch_size": 1}
    faulted = run_job(
        Job(identity=identity(task=4, faults=[shift]).model_copy(update=sway)), env, ctx
    )
    ref = faulted.snapshots[0]
    refused = ""
    try:
        run_job(
            Job(identity=faulted.identity, mode=JobMode.replay, remove=["f03"], from_snapshot=ref),
            env,
            ctx,
        )
    except EngineError as error:
        refused = str(error)
    kept = run_job(Job(identity=faulted.identity, mode=JobMode.replay, from_snapshot=ref), env, ctx)
    from fleetlab.engine.chain import compare

    healthy = run_job(Job(identity=identity(start=2, task=4, number=5)), env, ctx)
    env.close()
    return {
        "refused": refused,
        "kept": compare(kept.chain, faulted.chain).identical,
        "healthy": (healthy.chain.fixture_hash, healthy.chain.state[0], healthy.chain.scene[0]),
    }


def healthy_start() -> tuple[str, str, str]:
    from fleetlab.faults import REGISTRY

    ctx = EpisodeContext(fault_types=REGISTRY.get, make_check=LiberoCheck, teleoperator=LiftPolicy)
    env = LiberoSimEnv("libero_spatial", 4, SIZE)
    healthy = run_job(Job(identity=identity(start=2, task=4, number=5)), env, ctx)
    env.close()
    return healthy.chain.fixture_hash, healthy.chain.state[0], healthy.chain.scene[0]


def test_a_snapshot_under_a_cell_fault_does_not_leak_into_later_episodes() -> None:
    result = in_fresh_process(furniture_and_back)
    assert "cell hook" in result["refused"]  # version 1 refuses removing it from a snapshot
    assert result["kept"]  # replaying it with the fault kept continues bit for bit
    assert tuple(result["healthy"]) == in_fresh_process(healthy_start)


@pytest.mark.parametrize(("suite", "task"), [("libero_spatial", 0), ("libero_object", 3)])
def test_the_goal_and_the_privileged_state_name_the_same_bodies(suite: str, task: int) -> None:
    env = LiberoSimEnv(suite, task, SIZE)
    env.reset(identity(task=task).model_copy(update={"suite": suite}), [])
    goal = env.goal
    state = env.privileged_state()
    # held: close the gripper around nothing and look at contacts later; here only the names
    env.close()
    assert goal.predicate == ("On" if suite == "libero_spatial" else "In")
    assert goal.object_body in state.object_poses and goal.target_body in state.object_poses
    assert goal.object_body in state.object_velocities
    assert np.linalg.norm(state.object_velocities[goal.object_body][:3]) < 0.05  # settled
    assert not state.libero_success
    touching = [pair for pair in state.contacts if goal.object_body in pair]
    assert touching, "the object rests on something"
    assert not any(name.startswith(("robot0_", "gripper0_")) for p in touching for name in p)


def hand_over_on_libero() -> dict[str, Any]:
    from fleetlab.contracts.rollout import SupervisorSpec
    from fleetlab.engine.supervisors import make

    store = MemoryStore()
    env = LiberoSimEnv("libero_spatial", 0, SIZE)
    supervisor = SupervisorSpec(name="timeout", params={"after": 12, "decision": "hand_over"})
    ctx = context(store)
    ctx.supervisors = make
    job = Job(identity=identity(), supervisor=supervisor)
    first = run_job(job, env, ctx)
    follow = first.spawned[0]
    taken = run_job(follow, env, context(store))
    again = run_job(follow, LiberoSimEnv("libero_spatial", 0, SIZE), context(store))
    env.close()
    from fleetlab.engine.chain import compare

    return {
        "ended_by": first.ended_by,
        "handed_at": first.n_steps,
        "continued_from": taken.chain.first_step,
        "same_state": taken.chain.state[0] == first.chain.state[-1],
        "in_worker": taken.in_worker,
        "repeats": compare(again.chain, taken.chain).identical,
    }


def test_a_hand_over_continues_in_the_worker_on_libero() -> None:
    result = in_fresh_process(hand_over_on_libero)
    assert result["ended_by"] == "hand_over" and result["in_worker"]
    assert result["continued_from"] == result["handed_at"] == 23  # decided at 13, held 10
    assert result["same_state"] and result["repeats"]


def teleoperator_takes_over() -> dict[str, Any]:
    from fleetlab.contracts.rollout import SupervisorSpec
    from fleetlab.engine.chain import compare
    from fleetlab.engine.supervisors import make
    from fleetlab.policies.teleoperator import Teleoperator

    store = MemoryStore()
    env = LiberoSimEnv("libero_spatial", 0, SIZE)
    supervisor = SupervisorSpec(name="timeout", params={"after": 8, "decision": "hand_over"})
    ctx = context(store)
    ctx.supervisors = make
    first = run_job(Job(identity=identity(start=1), supervisor=supervisor), env, ctx)
    follow = first.spawned[0].model_copy(update={"max_steps": 700})
    teleop_ctx = context(store)
    teleop_ctx.teleoperator = Teleoperator
    taken = run_job(follow, env, teleop_ctx)
    again = run_job(follow, LiberoSimEnv("libero_spatial", 0, SIZE), teleop_ctx)
    env.close()
    return {
        "from": taken.chain.first_step,
        "steps": taken.n_steps,
        "success": taken.true_success,
        "reason": taken.true_reason,
        "ended_by": taken.ended_by,
        "repeats": compare(again.chain, taken.chain).identical,
    }


def test_the_scripted_teleoperator_takes_over_through_the_same_loop() -> None:
    result = in_fresh_process(teleoperator_takes_over)
    assert result["from"] == 19  # handed over at 9, held 10 steps
    assert result["success"], result  # the expert places the bowl (999 of 1,000 in its spike)
    assert result["repeats"], "the take-over replays bit for bit"
