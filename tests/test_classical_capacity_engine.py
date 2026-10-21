"""Capacity selection against real engine minting on an in-memory synthetic fleet.

Only plan_queue is called: no simulation, export, or stored world is read.
Distinct, public-valid treatment arms exercise preflight and real action resolution.
"""

from __future__ import annotations

from collections import Counter

import pytest
from engine_helpers import SOFTWARE

from fleetlab.agents.sessions import ExperimentQueue, QueuedExperiment
from fleetlab.baselines.classical.capacity import select_feasible
from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.identity import ARM_SLOTS, Task
from fleetlab.contracts.ids import scope_covers
from fleetlab.contracts.model import ComponentState, ConfigVersion, Fleet, Robot, Site
from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.contracts.world import DayRange, Windows
from fleetlab.engine.experiments import WorldTruth, plan_queue
from fleetlab.engine.toy import TOY_SPEC
from fleetlab.toolserver.public_results import COMPLETED, OVER_CAPACITY


def test_capacity_report_matches_real_minted_jobs_and_skips_an_engine_refusal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fleet = Fleet(
        id="w01",
        config=ConfigVersion(version=0, values={ConfigKey.control_action_delay_steps: 0}),
        sites=[
            Site(
                id=f"s{site}",
                conditions={
                    ConditionKey.lighting: 0,
                    ConditionKey.placement: [0, 0],
                    ConditionKey.object_batch: [0, 0],
                    ConditionKey.clutter: 0,
                },
                robots=[
                    Robot(
                        id=f"r{number:02d}",
                        site_id=f"s{site}",
                        components={
                            component: ComponentState(component=component, installed_day=0)
                            for component in Component
                        },
                    )
                    for number in range(4 * (site - 1) + 1, 4 * site + 1)
                ],
            )
            for site in range(1, 4)
        ],
    )
    world = WorldTruth(
        world_id=fleet.id,
        world_seed=73,
        fleet=fleet,
        windows=Windows(
            observed=DayRange(first=0, last=1),
            experiment=DayRange(first=2, last=3),
            scoring=DayRange(first=4, last=5),
        ),
        fault_schedule=[],
        episodes_per_robot_day=4,
        tasks=[Task("toy", 0)],
        policy=TOY_SPEC,
        image_size=32,
        max_steps={"toy": 40},
        base_runs=[],
        first_free_episode=1,
    )
    wide = Action(
        verb=Verb.set,
        target=Target(scope="fleet", config_key=ConfigKey.control_action_delay_steps),
        value=0,
    )
    conditions = [
        Action(
            verb=Verb.change_condition,
            target=Target(scope="s1", condition_key=key),
            value=fleet.sites[0].conditions[key],
        )
        for key in ConditionKey
    ]

    def repair(robot: str, verb: Verb = Verb.recalibrate) -> Action:
        return Action(verb=verb, target=Target(scope=robot, component=Component.scene_camera))

    first = repair("r01")
    overloaded = repair("r01", Verb.clean)
    alternatives = [repair("r02"), repair("r05")]
    candidates = [wide, *conditions, first, overloaded, *alternatives]
    rows = [{"robot_id": robot.id, "site_id": robot.site_id} for robot in fleet.robots()]
    chosen, report = select_feasible(candidates, list(reversed(rows)))
    assert chosen == [wide, *conditions, first, *alternatives]
    assert len(chosen) == 8 and report["shortfall"] == 0
    assert report["rejected"][0]["action"] == overloaded.model_dump(mode="json")
    assert report["rejected"][0]["reason"] == "capacity"
    assert [row["projected"] for row in report["rejected"][0]["overloaded"]] == [190, 180]

    def queue(actions: list[Action]) -> ExperimentQueue:
        return ExperimentQueue(
            world_id=fleet.id,
            analysis_id="w01-synthetic-capacity",
            session=1,
            created_at="2026-10-09T00:00:00",
            export="/synthetic-export-never-read",
            experiments=[
                QueuedExperiment(
                    experiment_id=f"x{index:04d}",
                    session=1,
                    request=ExperimentRequest(
                        robots=sorted(
                            robot.id
                            for robot in fleet.robots()
                            if scope_covers(action.target.scope, robot.id, robot.site_id)
                        ),
                        arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[action])],
                        episodes_per_arm=60,
                        paired_starts=True,
                        power_note="synthetic allocation test; neither arm is simulated",
                    ),
                )
                for index, action in enumerate(actions, start=1)
            ],
        )

    planned = plan_queue(
        queue(chosen), world, {}, software=SOFTWARE, batch_size=4, truth={}, record_video=False
    )
    assert len(planned.results) == 8 and all(result.status == "done" for result in planned.results)
    assert all(result.message == COMPLETED for result in planned.results)
    assert all(
        [len(arm.episode_ids) for arm in result.arms] == [60, 60] for result in planned.results
    )
    assert len(planned.jobs) == report["total_arm_episodes"] == 960
    actual = Counter((job.identity.robot_id, job.identity.day) for job in planned.jobs)
    expected = {
        (row["robot_id"], world.windows.experiment.first + row["day_index"]): row["episodes"]
        for row in report["usage"]
    }
    assert dict(actual) == expected
    assert all(count <= ARM_SLOTS for count in actual.values())
    per_request = Counter(
        (job.identity.experiment_id, job.identity.robot_id, job.identity.day)
        for job in planned.jobs
    )
    for index, selection in enumerate(report["selected"], start=1):
        for row in selection["allocation"]:
            day = world.windows.experiment.first + row["day_index"]
            assert per_request[f"x{index:04d}", row["robot_id"], day] == row["episodes"]

    # Independently mint the rejected candidate after its six predecessors.
    # The real engine refuses it because its starts would pass midnight, with the public reason.
    unfiltered = plan_queue(
        queue(candidates[:7]),
        world,
        {},
        software=SOFTWARE,
        batch_size=4,
        truth={},
        record_video=False,
    )
    assert all(result.status == "done" for result in unfiltered.results[:-1])
    assert unfiltered.results[-1].status == "refused"
    assert unfiltered.results[-1].message == OVER_CAPACITY
    assert unfiltered.results[-1].arms == []
    assert "arm episodes on a robot-day" in caplog.text  # checked before minting
    # The public message is the fixed capacity text, never the host exception. The unchanged
    # allocation assertions above and this retained-job count isolate capacity.
    assert len(unfiltered.jobs) == 6 * 2 * 60
