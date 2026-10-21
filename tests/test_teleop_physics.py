"""Human takeover through real LIBERO physics, without a model."""

from __future__ import annotations

from typing import Any

import pytest
from sim_helpers import in_fresh_process

from fleetlab.contracts.rollout import Job, SupervisorSpec
from fleetlab.engine import manifests, software
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import DirectSource, EpisodeContext, MemoryStore, run_job
from fleetlab.engine.scripted import SWAY_SPEC, SwayPolicy
from fleetlab.engine.supervisors import make
from fleetlab.faults import REGISTRY
from fleetlab.policies.teleop.human import HumanSettings, HumanTeleoperator, admit
from fleetlab.sim.libero_sim import LiberoSimEnv
from fleetlab.sim.libero_success import FleetCheck

pytestmark = pytest.mark.sim


def forced_fumble() -> dict[str, Any]:
    identity = manifests.identity(
        world_seed=604,
        episode_number=1,
        robot=1,
        site=1,
        day=0,
        episode_index=0,
        suite="libero_spatial",
        task_index=0,
        start_state=2,
        policy=SWAY_SPEC,
        batch_size=1,
        software=software.current("none"),
        image_size=64,
        max_steps=280,
    )
    settings = HumanSettings(fumble_probability=1)
    store = MemoryStore()
    ctx = EpisodeContext(
        source=DirectSource(SwayPolicy()),
        make_check=FleetCheck,
        fault_types=REGISTRY.get,
        teleoperator=HumanTeleoperator,
        supervisors=make,
        store=store,
        continuation_steps=settings.active_budget,
    )
    env = LiberoSimEnv("libero_spatial", 0, 64)
    first = run_job(
        Job(identity=identity, supervisor=SupervisorSpec(name="timeout", params={"after": 20})),
        env,
        ctx,
    )
    follow = admit(first.spawned, settings, 14)[0]
    # Wait longer than the original attempt's complete budget. It still gets active time.
    params = dict(follow.teleoperator_params or {})
    params["queue_steps"] = 320
    follow = follow.model_copy(update={"teleoperator_params": params})
    taken = run_job(follow, env, ctx)
    repeated = run_job(follow, env, ctx)
    env.close()
    return {
        "success": taken.true_success,
        "human": taken.stats["human"],
        "identical": compare(repeated.chain, taken.chain).identical,
        "steps": taken.n_steps,
        "snapshot_step": first.n_steps,
    }


def test_the_fumble_is_physical_retries_and_replays_after_a_long_wait() -> None:
    result = in_fresh_process(forced_fumble)
    assert result["human"]["fumble_draw"] and result["human"]["fumble_applied"]
    assert result["human"]["idle_steps"] > 280
    assert result["human"]["active_steps"] > 0
    assert result["success"] and result["identical"], result
