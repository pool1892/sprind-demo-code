"""The CPU mechanics fixtures' faults on LIBERO (`make check-sim`, no model): every fault kind
and level the three fixtures use runs through the engine's loop on a `libero_object` task
with the scripted server policy, and a
healthy episode afterwards on the same simulator is the healthy episode of a fresh one."""

from __future__ import annotations

from typing import Any

import pytest

from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.contracts.rollout import Job

pytestmark = pytest.mark.sim

SIZE = 128
STEPS = 20
SOFT = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)


def _faults() -> list[tuple[str, Any]]:
    from test_worlds_v1 import example_curves, recipes

    from fleetlab.worlds.recipe import TASKS
    from fleetlab.worlds.world import generate

    texts = {Task(t.suite, t.task_index): "x" for t in TASKS}
    out: dict[tuple[str, str], Any] = {}
    for name in ("D1", "D2", "D3"):
        world = generate(recipes()[name], SOFT, texts, example_curves())
        for fault in world.spec.fault_schedule:
            key = (fault.kind, repr(sorted(fault.params.items())) + str(fault.level))
            still = fault.schedule.model_copy(update={"onset_day": 0, "drift": None})
            out.setdefault(key, fault.model_copy(update={"schedule": still}))
    return sorted(((f"{k[0]} {k[1]}", f) for k, f in out.items()), key=lambda kv: kv[0])


def test_every_fault_of_the_development_worlds_runs_and_undoes_itself() -> None:
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.engine import manifests, software
    from fleetlab.engine.episode import DirectSource, EpisodeContext, MemoryStore, run_job
    from fleetlab.engine.scripted import SWAY_SPEC, SwayPolicy
    from fleetlab.faults import REGISTRY
    from fleetlab.sim.libero_sim import LiberoSimEnv
    from fleetlab.sim.libero_success import FleetCheck

    def ctx() -> EpisodeContext:
        return EpisodeContext(
            fault_types=REGISTRY.get,
            make_check=FleetCheck,
            source=DirectSource(SwayPolicy()),
            store=MemoryStore(),
        )

    def ident(fault: Any, n: int) -> Any:
        scope = fault.target.scope if fault is not None else "r01"
        if scope.startswith("r"):
            robot = int(scope[1:])
        elif scope.startswith("s"):
            robot = 1 + 4 * (int(scope[1:]) - 1)
        else:
            robot = 1
        site = (robot - 1) // 4 + 1
        return manifests.identity(
            world_seed=7,
            episode_number=n,
            robot=robot,
            site=site,
            day=0,
            episode_index=n,
            suite="libero_object",
            task_index=5,
            start_state=0,
            policy=SWAY_SPEC,
            batch_size=1,
            software=software.current("none"),
            image_size=SIZE,
            max_steps=STEPS,
            faults=[] if fault is None else [fault],
        )

    env = LiberoSimEnv("libero_object", 5, SIZE)
    try:
        healthy = run_job(Job(identity=ident(None, 900)), env, ctx()).chain
        failures = []
        moves_cell = ("camera_pose_offset", "furniture_shift", "object_batch", "gripper_wear")
        for label, fault in _faults():
            faulted_job = Job(identity=ident(fault, 900))
            faulted = run_job(faulted_job, env, ctx()).chain
            twin = faulted_job.identity.model_copy(update={"faults": []})
            same_robot = run_job(Job(identity=twin), env, ctx()).chain
            after = run_job(Job(identity=ident(None, 900)), env, ctx()).chain
            if after.state != healthy.state or after.fixture_hash != healthy.fixture_hash:
                failures.append(f"{label}: not undone")
            changed = faulted.fixture_hash != same_robot.fixture_hash
            if label.startswith(moves_cell) and not changed:
                failures.append(f"{label}: the cell did not change")
        assert failures == []
    finally:
        env.close()
