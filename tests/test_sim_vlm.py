"""Physical recovery and scripted replay on LIBERO without a GPU policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fleetlab.contracts.live import Decision, SupervisorContext, SupervisorDecision
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import manifests, software
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import DirectSource, EpisodeContext, SupervisorParts, run_job
from fleetlab.engine.scripted import SWAY_SPEC, SwayPolicy
from fleetlab.engine.standin_faults import STANDIN
from fleetlab.engine.supervisors import AfterSteps
from fleetlab.sim.libero_sim import LiberoCheck, LiberoSimEnv

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("decision", [Decision.recover, Decision.degrade])
def test_recovery_physically_acts_and_replays(decision: Decision) -> None:
    class Once:
        def decide(self, context: SupervisorContext) -> SupervisorDecision:
            return SupervisorDecision(decision, "physical primitive test", 500, 400)

    def factory(name: str, params: Any) -> SupervisorParts:
        return SupervisorParts(
            AfterSteps(9), Once(), should_call=lambda c: not c.previous_decisions
        )

    identity = manifests.identity(
        world_seed=602,
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
        max_steps=100,
    )
    env = LiberoSimEnv("libero_spatial", 0, 64)
    ctx = EpisodeContext(
        fault_types=STANDIN.get,
        make_check=LiberoCheck,
        source=DirectSource(SwayPolicy()),
        supervisors=factory,
    )
    try:
        original = run_job(Job(identity=identity, supervisor=SupervisorSpec(name="once")), env, ctx)
        replay = run_job(
            Job(identity=identity, mode=JobMode.scripted, decision_trace=original.decisions),
            env,
            ctx,
        )
        assert compare(original.chain, replay.chain).identical
        assert original.stats["decision_hold_steps"] == 10
        if decision in (Decision.retry, Decision.recover):
            z = np.array(original.steps.eef_pos)[:, 2]
            assert z[45] > z[20] + 0.03
            assert original.steps.gripper_qpos[45][0] > 0.02
    finally:
        env.close()


def test_retry_returns_to_actual_first_grasp_pose_and_replays() -> None:
    from scipy.spatial.transform import Rotation

    from fleetlab.policies.teleoperator import TELEOPERATOR_SPEC, Teleoperator

    class Once:
        def decide(self, observation: SupervisorContext) -> SupervisorDecision:
            return SupervisorDecision(Decision.retry, "CPU expert fixture", 500, 400)

    def factory(name: str, params: Any) -> SupervisorParts:
        return SupervisorParts(
            AfterSteps(0),
            Once(),
            should_call=lambda c: not c.previous_decisions and Decision.retry in c.options,
        )

    identity = manifests.identity(
        world_seed=602,
        episode_number=1,
        robot=1,
        site=1,
        day=0,
        episode_index=0,
        suite="libero_spatial",
        task_index=0,
        start_state=2,
        policy=TELEOPERATOR_SPEC,
        batch_size=1,
        software=software.current("none"),
        image_size=64,
        max_steps=280,
    )
    env = LiberoSimEnv("libero_spatial", 0, 64)
    ctx = EpisodeContext(
        fault_types=STANDIN.get,
        make_check=LiberoCheck,
        teleoperator=Teleoperator,
        supervisors=factory,
    )
    try:
        original = run_job(
            Job(identity=identity, supervisor=SupervisorSpec(name="fixture")), env, ctx
        )
        assert len(original.decisions) == 1
        assert original.stats["robot_active_steps"] == (
            original.n_steps - original.stats["decision_hold_steps"]
        )
        assert original.stats["retry_active_steps"] > 70
        decision = original.decisions[0]
        first_close = next(
            i for i, a in enumerate(original.steps.commanded) if a is not None and a[6] > 0
        )
        anchor = original.steps.commanded[first_close - 1]
        assert anchor is not None
        start = decision.step + decision.hold_steps
        # This fixture approaches the actual bowl, using the CPU expert only to
        # provide realistic policy commands; the recovery gets no object geometry.
        np.testing.assert_allclose(original.steps.eef_pos[start + 70], anchor[:3], atol=0.03)
        actual = Rotation.from_quat(original.steps.eef_quat[start + 70])
        # Observed quaternion and commanded gripper frame have a fixed frame offset.
        desired = Rotation.from_quat(original.steps.eef_quat[first_close])
        assert (actual.inv() * desired).magnitude() < 0.1
        replay = run_job(
            Job(identity=identity, mode=JobMode.scripted, decision_trace=original.decisions),
            env,
            ctx,
        )
        assert compare(original.chain, replay.chain).identical
    finally:
        env.close()
