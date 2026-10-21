"""The experiment queue on a toy world (`make check`): minting the arms, running them,
extending the export, answering the runtime. No simulator, no GPU."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest
from engine_helpers import SOFTWARE, config, fault

from fleetlab.agents.sessions import ExperimentQueue, QueuedExperiment, arm_problems
from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.identity import Task
from fleetlab.contracts.model import (
    ComponentState,
    ConfigVersion,
    Fleet,
    Robot,
    Site,
)
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.contracts.world import DayRange, Windows
from fleetlab.engine import manifests, toy
from fleetlab.engine.experiments import QueueEngine, WorldTruth, run_queue
from fleetlab.engine.fleet import run
from fleetlab.log import LogReader, export_runs

PER_DAY = 4


def toy_fleet() -> Fleet:
    def robot(robot_id: str, site_id: str) -> Robot:
        return Robot(
            id=robot_id,
            site_id=site_id,
            components={c: ComponentState(component=c, installed_day=0) for c in Component},
        )

    return Fleet(
        id="w01",
        sites=[
            Site(id="s1", robots=[robot("r01", "s1"), robot("r02", "s1")]),
            Site(id="s2", robots=[robot("r03", "s2")]),
        ],
        config=ConfigVersion(version=1, values={ConfigKey.control_action_delay_steps: 0}),
    )


def toy_world(base_runs: list[str], first_free: int) -> WorldTruth:
    return WorldTruth(
        world_id="w01",
        world_seed=3,
        fleet=toy_fleet(),
        windows=Windows(
            observed=DayRange(first=0, last=1),
            experiment=DayRange(first=2, last=3),
            scoring=DayRange(first=4, last=5),
        ),
        fault_schedule=[
            fault("toy_camera_shift", 1, robot=2, pixels=3),
            fault("toy_delay", 2, robot=3, steps=3),
        ],
        episodes_per_robot_day=PER_DAY,
        tasks=[Task("toy", 0), Task("toy", 1)],
        task_classes=["carry"],
        start_states=7,
        policy=toy.TOY_SPEC,
        image_size=32,
        max_steps={"toy": 60},
        base_runs=base_runs,
        first_free_episode=first_free,
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[WorldTruth, Path, Path]:
    """The as-run observed window of the toy world, its run and its export."""
    root = tmp_path_factory.mktemp("world")
    draft = toy_world([], 1)
    jobs = []
    n = 1
    for day in (0, 1):
        for robot in (1, 2, 3):
            robot_id = f"r0{robot}"
            for index in range(PER_DAY):
                ident = manifests.identity(
                    world_seed=3,
                    episode_number=n,
                    robot=robot,
                    site=1 if robot < 3 else 2,
                    day=day,
                    episode_index=index,
                    suite="toy",
                    task_index=index % 2,
                    start_state=index,
                    policy=toy.TOY_SPEC,
                    batch_size=4,
                    software=SOFTWARE,
                    image_size=32,
                    max_steps=60,
                    faults=draft.faults_on(robot_id, day),
                )
                jobs.append(Job(identity=ident))
                n += 1
    base = root / "runs" / "as-run"
    run(config(base, episodes_per_robot_day=PER_DAY, chain_sample=False), jobs)
    export = root / "export"
    export_runs([base], export, fleet=draft.fleet, task_classes=draft.task_classes)
    return toy_world([str(base)], n), root, export


def recalibrate(robot_id: str) -> Action:
    return Action(
        verb=Verb.recalibrate, target=Target(scope=robot_id, component=Component.scene_camera)
    )


def set_delay(robot_id: str, value: int) -> Action:
    target = Target(scope=robot_id, config_key=ConfigKey.control_action_delay_steps)
    return Action(verb=Verb.set, target=target, value=value)


def queue(
    export: Path, session: int, *experiments: tuple[str, ExperimentRequest]
) -> ExperimentQueue:
    return ExperimentQueue(
        world_id="w01",
        analysis_id="w01-test",
        session=session,
        created_at="2026-10-08T00:00:00",
        export=str(export),
        experiments=[
            QueuedExperiment(experiment_id=x, session=session, request=r) for x, r in experiments
        ],
    )


def request(robots: list[str], *arms: list[Action], n: int = 3, paired: bool = True) -> Any:
    return ExperimentRequest(
        arms=[ExperimentArm(actions=a) for a in arms],
        robots=robots,
        episodes_per_arm=n,
        paired_starts=paired,
        power_note="a toy",
    )


def identities(run_dir: Path) -> dict[str, Any]:
    from fleetlab.log.runfiles import read_identities

    return {i.episode_id: i for i in read_identities(run_dir)}


def test_a_queue_is_minted_run_exported_and_answered(
    world: tuple[WorldTruth, Path, Path], tmp_path: Path
) -> None:
    truth, root, export = world
    q = queue(
        export,
        1,
        ("x0001", request(["r02"], [], [recalibrate("r02")])),
        ("x0002", request(["r03", "r01"], [set_delay("r03", 0)], [], n=4, paired=False)),
        ("x0003", request(["r09"], [], [])),
    )
    path = tmp_path / "queue.json"
    path.write_text(q.model_dump_json())
    cfg = config(root / "unused")
    reply = run_queue(path, truth, root, config=cfg)
    by_id = {r.experiment_id: r for r in reply.results}
    assert by_id["x0003"].status == "refused"
    assert by_id["x0003"].message == "invalid_public_request"
    assert by_id["x0001"].status == by_id["x0002"].status == "done"
    assert [len(a.episode_ids) for a in by_id["x0001"].arms] == [3, 3]
    assert arm_problems(reply) == []  # every arm episode in the export, with its experiment id

    run_dir = root / "experiments" / "w01-test" / "session-01"
    ids = identities(run_dir)
    control, treated = (by_id["x0001"].arms[a].episode_ids for a in (0, 1))
    for c, t in zip(control, treated, strict=True):
        a, b = ids[c], ids[t]
        assert (a.task, a.start_state) == (b.task, b.start_state)  # paired starts
        assert a.experiment_id == b.experiment_id == "x0001"
        assert a.day in (2, 3) and a.episode_index >= PER_DAY and a.scene_seed != b.scene_seed
        assert [f.kind for f in a.faults] == ["toy_camera_shift"]
        assert b.faults == []  # recalibrated: the camera fault removed
        a.check_seeds(3)
    delay_arm = [ids[e] for e in by_id["x0002"].arms[0].episode_ids]
    assert {i.robot_id for i in delay_arm} == {"r01", "r03"}
    assert all(i.faults == [] for i in delay_arm)  # set to the nominal 0 removes the delay
    second = [ids[e] for e in by_id["x0002"].arms[1].episode_ids]
    assert [f.kind for i in second if i.robot_id == "r03" for f in i.faults] == ["toy_delay"] * 2
    assert all(int(e[1:]) >= truth.first_free_episode for e in ids)
    with LogReader(reply.export) as reader:
        rows = reader.query("select count(*) as n, count(experiment_id) as arms from episodes")
    assert rows.to_pylist() == [{"n": 24 + 14, "arms": 14}]
    # The allocations belong to the analysis, not to the world.
    assert (root / "experiments" / "w01-test" / "allocations.json").is_file()
    assert not (root / "experiments" / "allocations.json").exists()
    import json

    record = json.loads((run_dir / "run.json").read_text())
    assert record["chain_ok"] is True
    assert record["extra"]["episodes_per_robot_day"] == PER_DAY  # the world's own day length
    with LogReader(reply.export) as reader:  # arm episodes start after the day's regular work
        times = reader.query(
            "select experiment_id is not null as arm, min(started_at) as first, "
            "max(started_at) as last from episodes group by arm order by arm"
        ).to_pylist()
    assert times[0]["last"] < datetime.time(16) <= times[1]["first"]

    # Answered again: the plan is read back, nothing new is minted or run.
    again = run_queue(path, truth, root, config=cfg)
    assert again.results == reply.results

    # The next session's queue gets fresh episodes and indices, and the export holds both.
    q2 = queue(reply.export, 2, ("x0004", request(["r02"], [], [recalibrate("r02")], n=2)))
    path2 = tmp_path / "queue2.json"
    path2.write_text(q2.model_dump_json())
    engine = QueueEngine(truth, root, cfg)
    later = engine.run_queue(path2, q2)
    new_ids = identities(root / "experiments" / "w01-test" / "session-02")
    assert not set(new_ids) & set(ids)
    indices = {
        (i.robot_id, i.day, i.episode_index) for i in list(ids.values()) + list(new_ids.values())
    }
    assert len(indices) == len(ids) + len(new_ids)
    assert arm_problems(later) == []
    with LogReader(later.export) as reader:
        n = reader.query("select count(*) as n from episodes").to_pylist()[0]["n"]
    assert n == 24 + 14 + 4


def test_the_runtime_reads_the_engine_s_reply(
    world: tuple[WorldTruth, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents.toolenv import project_tool_env
    from fleetlab.engine.experiments import scripted_analysis

    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    truth, root, export = world
    engine = QueueEngine(truth, root, config(root / "unused"))
    request = {
        "arms": [{"actions": []}, {"actions": [recalibrate("r02").model_dump(mode="json")]}],
        "robots": ["r02"],
        "episodes_per_arm": 2,
        "paired_starts": True,
        "power_note": "a toy",
    }
    env = project_tool_env(tmp_path / "env")
    record, seen = scripted_analysis(tmp_path / "analyses", truth, export, engine, request, env)
    assert [s.state for s in record.sessions] == ["waiting", "ended"]
    assert seen[1]["results"][0]["status"] == "done"
    assert [len(a["episode_ids"]) for a in seen[1]["results"][0]["arms"]] == [2, 2]
    answer = seen[1]["rows"]  # the query tool's answer: columns and rows
    table = [dict(zip(answer["columns"], row, strict=True)) for row in answer["rows"]]
    assert [(r["n"]) for r in table if r["experiment_id"] == "x0001"] == [4]


def test_arm_episodes_start_in_the_evening_and_never_after_midnight(tmp_path: Path) -> None:
    from fleetlab.contracts.identity import started_at_for
    from fleetlab.engine.experiments import plan_queue

    world = toy_world([], 1)
    fits = request(
        ["r01"], [], [recalibrate("r01")], n=4
    )  # 2 arms x 4 over two days: 4 a day on r01
    # 60 a day on r02 per request (30 slots, both arms): the third would pass the 160 slots.
    full = [
        (
            f"x000{i}",
            request(
                ["r02"],
                [],
                [
                    Action(
                        verb=Verb.service, target=Target(scope="r02", component=Component.gripper)
                    )
                ],
                n=60,
            ),
        )
        for i in (2, 3, 4)
    ]
    planned = plan_queue(
        queue(tmp_path, 1, ("x0001", fits), *full),
        world,
        {},
        software=SOFTWARE,
        batch_size=4,
        truth={},
    )
    assert [r.status for r in planned.results] == ["done", "done", "done", "refused"]
    assert planned.results[-1].message == "exceeds_robot_day_capacity"
    times = sorted(started_at_for(j.identity, PER_DAY) for j in planned.jobs)
    # After the day's regular work, every three minutes: r02's 120 a day end at 21:57.
    assert times[0] == "16:00:00" and times[-1] == "21:57:00"
    r01 = sorted(
        started_at_for(j.identity, PER_DAY) for j in planned.jobs if j.identity.robot_id == "r01"
    )
    assert r01 == [t for t in ("16:00:00", "16:03:00", "16:06:00", "16:09:00") for _ in (2, 3)]


def test_a_request_past_a_robot_days_capacity_is_refused_with_the_public_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from fleetlab.contracts.identity import ARM_SLOTS
    from fleetlab.engine.experiments import plan_queue
    from fleetlab.toolserver.public_results import OVER_CAPACITY, PUBLIC_REFUSALS

    world = toy_world([], 1)
    service = Action(verb=Verb.service, target=Target(scope="r02", component=Component.gripper))
    # Two arms over the window's two days: n slots put n arm episodes on each day.
    planned = {
        n: plan_queue(
            queue(tmp_path, 1, ("x0001", request(["r02"], [], [service], n=n))),
            world,
            {},
            software=SOFTWARE,
            batch_size=4,
            truth={},
        )
        for n in (ARM_SLOTS, ARM_SLOTS + 1)
    }
    fits, past = planned[ARM_SLOTS].results[0], planned[ARM_SLOTS + 1].results[0]
    assert fits.status == "done" and len(planned[ARM_SLOTS].jobs) == 2 * ARM_SLOTS
    assert past.status == "refused" and past.message == OVER_CAPACITY and not past.arms
    assert not planned[ARM_SLOTS + 1].jobs
    assert "23:59:59" not in caplog.text  # refused before minting, not by the exception
    # An earlier probe takes a slot through the same allocations (loop.probes), with no
    # earlier experiment: two 80-episode arms on that one day no longer fit.
    day = world.windows.experiment.first
    probed = {"next_index": {f"r02/{day}": PER_DAY + 1}}

    def on_one_day(allocations: dict[str, Any]) -> Any:
        two_arms = request(["r02"], [], [service], n=ARM_SLOTS // 2)
        return plan_queue(
            queue(tmp_path, 1, ("x0001", two_arms)),
            world,
            allocations,
            software=SOFTWARE,
            batch_size=4,
            truth={},
            days=[day],
        ).results[0]

    assert on_one_day({}).status == "done"
    after_probe = on_one_day(probed)
    assert after_probe.status == "refused" and after_probe.message == OVER_CAPACITY
    assert PUBLIC_REFUSALS[OVER_CAPACITY] == (
        "A robot-day holds at most 160 experiment-arm episodes, counting every arm and "
        "repetition this request and the earlier experiments and probes of this analysis "
        "put on that robot and day; spread the arms over robots or days."
    )


def test_each_analysis_keeps_its_own_allocations(tmp_path: Path) -> None:
    from fleetlab.engine.experiments import allocations_file, plan_queue

    world = toy_world([], 1)
    first = allocations_file(tmp_path, "w01-a")
    assert first == tmp_path / "experiments" / "w01-a" / "allocations.json"
    assert first != allocations_file(tmp_path, "w01-b")
    # Two analyses are alternative histories: each restarts after the as-run episodes.
    plans = [
        plan_queue(
            queue(tmp_path, 1, ("x0001", request(["r01"], [], [recalibrate("r01")], n=2))),
            world,
            {},
            software=SOFTWARE,
            batch_size=4,
            truth={},
        )
        for _ in range(2)
    ]
    keys = [[(j.identity.episode_id, j.identity.episode_index) for j in p.jobs] for p in plans]
    assert keys[0] == keys[1]
    assert min(i for _, i in keys[0]) == PER_DAY


def test_refusal_keeps_truth_details_only_in_host_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from fleetlab.engine.experiments import plan_queue

    # The real resolver raises with an active fault kind when its truth mapping
    # is absent. Its detailed traceback must not become the public reply.
    world = toy_world([], 1)
    q = queue(tmp_path, 1, ("x0001", request(["r02"], [], [recalibrate("r02")], n=1)))
    planned = plan_queue(q, world, {}, software=SOFTWARE, batch_size=4, truth={})
    assert not planned.jobs
    assert planned.results[0].status == "refused"
    assert planned.results[0].message == "request_not_runnable"
    assert "toy_camera_shift" in caplog.text
    assert "toy_camera_shift" not in planned.results[0].model_dump_json()


def test_cached_plan_cannot_omit_replies_or_answer_another_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from fleetlab.contracts.hashing import hash_json
    from fleetlab.engine.experiments import queue_dirs

    world = toy_world([], 1)
    q = queue(tmp_path, 1, ("x0001", request(["r01"], [], [recalibrate("r01")], n=1)))
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(q.model_dump_json())
    run_dir, _ = queue_dirs(tmp_path, q.analysis_id, q.session)
    plan = run_dir.parent / (run_dir.name + ".plan.json")
    plan.parent.mkdir(parents=True)
    monkeypatch.setattr("fleetlab.engine.fleet.prepare", lambda cfg: SOFTWARE)
    monkeypatch.setattr("fleetlab.engine.fleet.run", lambda *args: pytest.fail("must not execute"))
    saved = {"queue_sha256": hash_json(q.model_dump(mode="json")), "jobs": [], "results": []}
    plan.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="requests"):
        run_queue(queue_file, world, tmp_path, config=config(tmp_path / "unused"), export=False)
    saved["queue_sha256"] = "stale"
    plan.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="does not match this queue"):
        run_queue(queue_file, world, tmp_path, config=config(tmp_path / "unused"), export=False)
