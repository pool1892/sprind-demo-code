"""The jobs runtime on the LIBERO simulator, without a model (`make check-sim`).

LIBERO's own goal predicates on the loaded scene (`On` needs contact; a settled item
counts), a scene with a unary goal built, a program of every step kind played from a
development start with a policy that holds still, its five tables read back through
`LogReader` as the tool server reads them, the runner's refusal of release starts, and a
re-render that is accepted only when every state hash and the delivery verdict match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fleetlab.contracts.jobs import JobProgram, RequestState
from fleetlab.contracts.log import JOB_TABLE_NAMES, log_tree_problems
from fleetlab.engine import software
from fleetlab.jobs import tables
from fleetlab.jobs.check import SceneGoals
from fleetlab.jobs.runner import (
    GoalSuite,
    HoldSource,
    JobRunner,
    episode_identity,
    hold_here,
    make_env,
)
from fleetlab.jobs.starts import development_starts, release_starts, start_identity
from fleetlab.log import LogReader

pytestmark = pytest.mark.sim

SIZE = 128


def state(goals: list[dict[str, Any]], base_task: int, rules: Any = ()) -> RequestState:
    return RequestState.model_validate(
        {
            "request_text": "test",
            "requested_goals": goals,
            "base_task": base_task,
            "order_rules": list(rules),
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


CHEESE_THEN_PLATE = state(
    [
        {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
        {"predicate": "On", "objects": ["akita_black_bowl_1", "plate_1"]},
    ],
    6,
    [{"first": 0, "then": 1}],
)


def step(kind: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": kind,
        "instruction": None,
        "until": None,
        "max_steps": None,
        "segment": None,
        "steps": None,
    }
    return {**base, **fields}


def segment(primitive: str, steps: int, axis: str | None = None, amp: float | None = None) -> Any:
    return {
        "primitive": primitive,
        "axis": axis,
        "amplitude_m": amp,
        "duration_steps": steps,
        "period_steps": None,
    }


EVERY_KIND = JobProgram.model_validate(
    {
        "name": "every-step-kind",
        "purpose": "the runtime's machinery",
        "notes": "",
        "steps": [
            step(
                "policy",
                instruction="put the cream cheese in the bowl",
                until={"goal": 0, "steps": None},
                max_steps=8,
                on_timeout="continue",
            ),
            step("motion", segment=segment("ramp", 10, "z", 0.03)),
            step("motion", segment=segment("close_gripper", 4)),
            step("wait", steps=5),
            step("motion", segment=segment("open_gripper", 4)),
            step(
                "policy",
                instruction="put the bowl on the plate",
                until={"goal": None, "steps": 6},
                max_steps=6,
            ),
        ],
    }
)
PROGRAM_STEPS = 8 + 10 + 4 + 5 + 4 + 6


@pytest.fixture(scope="module")
def suite() -> GoalSuite:
    return GoalSuite()


@pytest.fixture(scope="module")
def played(suite: GoalSuite, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("jobs") / "run"
    runner = JobRunner(HoldSource(), software.current("none"), image_size=SIZE, suite=suite)
    try:
        runner.run_starts(
            root,
            "j0001",
            EVERY_KIND,
            CHEESE_THEN_PLATE,
            development_starts(6, 2),
            record_video=True,
            first_episode=101,
        )
    finally:
        runner.close()
    return root


def _identity(suite: GoalSuite, task: int, steps: int) -> Any:
    return episode_identity(
        start_identity(task, 0),
        "e000001",
        HoldSource().spec,
        software.current("none"),
        suite.bench[task],
        SIZE,
        steps,
    )


def _move(env: Any, name: str, xyz: Any) -> None:
    import mujoco

    rs = env._rs
    model, data = env._sim.model._model, env._sim.data._data
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rs.objects_dict[name].joints[-1])
    q, v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    data.qpos[q : q + 3] = xyz
    data.qvel[v : v + 6] = 0.0
    mujoco.mj_forward(model, data)


def test_libero_s_on_needs_contact_and_holds_once_the_item_rests(suite: GoalSuite) -> None:
    env = make_env(suite.bench[6], SIZE)
    try:
        obs = env.reset(_identity(suite, 6, 100), [])
        goals = SceneGoals(env._rs, CHEESE_THEN_PLATE.requested_goals)
        assert goals.holds() == (False, False)
        assert goals.named == ("cream_cheese_1", "akita_black_bowl_1", "plate_1")
        bowl = env.privileged_state().object_poses["akita_black_bowl_1"][:3]
        _move(env, "cream_cheese_1", bowl + np.array([0.0, 0.0, 0.06]))
        assert goals.holds() == (False, False)  # above the bowl, not touching it
        hold = hold_here(obs)
        for _ in range(40):
            env.step(hold)
        assert goals.holds() == (True, False)
        assert bool(env._inner._env.check_success())  # the base task's own check agrees
    finally:
        env.close()


def test_a_scene_with_a_unary_goal_builds_and_evaluates(suite: GoalSuite) -> None:
    unary = state(
        [
            {"predicate": "Turnon", "objects": ["flat_stove_1"]},
            {"predicate": "Open", "objects": ["wooden_cabinet_1_middle_region"]},
        ],
        2,
    )
    env = make_env(suite.bench[2], SIZE)  # build_index alone refuses this task's goal
    try:
        env.reset(_identity(suite, 2, 10), [])
        goals = SceneGoals(env._rs, unary.requested_goals)
        assert goals.holds() == (False, False)
        # The base task's own check is LIBERO's Turnon of the stove: the two agree.
        assert goals.holds()[0] == bool(env._inner._env.check_success())
        assert goals.named == ("flat_stove_1", "wooden_cabinet_1")
    finally:
        env.close()
    assert suite.instructions()["Turnon flat_stove_1"] == "turn on the stove"
    assert suite.instructions()["On akita_black_bowl_1 plate_1"] == "put the bowl on the plate"


def test_a_program_of_every_kind_runs_and_fills_the_five_tables(played: Path) -> None:
    episodes = tables.read_episodes(played)
    assert [e.episode_id for e in episodes] == ["e000101", "e000102"]
    first = episodes[0]
    outcomes = [(r.kind.value, r.outcome.value, r.steps_used) for r in first.program_steps]
    assert outcomes == [
        ("policy", "timeout", 8),  # continue: the program goes on
        ("motion", "reached", 10),
        ("motion", "reached", 4),
        ("wait", "reached", 5),
        ("motion", "reached", 4),
        ("policy", "reached", 6),
    ]
    assert first.tail_steps == 60 and first.control_steps == PROGRAM_STEPS + 60
    assert not first.delivered and first.order_kept
    assert len(first.end_state_hash) == 16
    assert log_tree_problems(played) == []
    with LogReader(played, materialize=True, lock=True) as reader:
        for name in JOB_TABLE_NAMES:
            n = reader.query(f"SELECT count(*) AS n FROM {name}").to_pylist()[0]["n"]
            assert n > 0, name
        rows = reader.query(
            "SELECT step, phase, program_step, gripper_command, instruction FROM robot_steps "
            "WHERE episode_id = 'e000101' ORDER BY step"
        ).to_pylist()
        assert len(rows) == first.control_steps + 1
        assert rows[0]["instruction"] == "put the cream cheese in the bowl"
        closed = [r["gripper_command"] for r in rows if r["program_step"] in (2, 3)]
        assert closed == [1.0] * 9  # close_gripper, then the wait keeps it closed
        assert rows[-1]["phase"] == "end" and rows[-2]["phase"] == "tail"
        ramp = reader.query(
            "SELECT eef_pos[3] AS z FROM robot_steps WHERE episode_id = 'e000101' "
            "AND step IN (8, 18) ORDER BY step"
        ).to_pylist()
        assert ramp[1]["z"] - ramp[0]["z"] > 0.015  # the 3 cm ramp lifted the hand
        frames = reader.query(
            "SELECT count(*) AS n FROM robot_steps WHERE episode_id = 'e000101' "
            "AND frame_index IS NOT NULL"
        ).to_pylist()[0]["n"]
        video = reader.query(
            "SELECT video_scene FROM job_episodes WHERE episode_id = 'e000101'"
        ).to_pylist()[0]["video_scene"]
    assert video == "video/e000101.scene.mp4" and (played / "log" / video).is_file()
    assert frames == first.control_steps // 4 + 1 + (first.control_steps % 4 != 0)
    record = json.loads((played / "run.json").read_text())
    assert record["state"] == CHEESE_THEN_PLATE.model_dump(mode="json")
    assert set(record["episodes"]) == {"e000101", "e000102"}


def test_the_runner_refuses_release_starts_on_a_development_run(suite: GoalSuite) -> None:
    runner = JobRunner(HoldSource(), software.current("none"), image_size=SIZE, suite=suite)
    try:
        with pytest.raises(ValueError, match="release start"):
            runner.play(
                EVERY_KIND,
                CHEESE_THEN_PLATE,
                release_starts(6, 1)[0],
                "j0001",
                "e000001",
                record_video=False,
            )
    finally:
        runner.close()


def test_a_re_render_is_kept_only_when_every_hash_and_the_verdict_match(
    played: Path, tmp_path: Path
) -> None:
    from fleetlab.jobs.replay import rerender

    report = rerender(
        played, "e000101", tmp_path, image_size=160, camera="frontview", frame_every=2
    )
    assert report["accepted"] and report["label"] == "re-render"
    assert report["hash_mismatches"] == 0 and report["verdict_matches"]
    assert report["hashes_compared"] == PROGRAM_STEPS + 60 + 1
    assert report["video"] == "e000101.rerender.frontview.160px.mp4"
    assert (tmp_path / report["video"]).is_file()
    # A record that the re-execution does not reproduce keeps no video.
    executed, hashes, identity = tables.load_replay_record(played, "e000102")
    tampered = tmp_path / "tampered"
    for relative in ("run.json", "episodes.jsonl", "truth/jobs/e000102.json"):
        target = tampered / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((played / relative).read_text())
    (tampered / "log").mkdir()
    executed[20, 2] += 0.02
    tables.save_replay_record(tampered, "e000102", executed, hashes, identity)
    rejected = rerender(tampered, "e000102", tmp_path / "out")
    assert not rejected["accepted"] and rejected["video"] is None
    assert rejected["first_mismatch_step"] == 21
    assert not list((tmp_path / "out").glob("*.mp4"))


HOMEWARD = JobProgram.model_validate(
    {
        "name": "away-and-home",
        "purpose": "the home step",
        "notes": "",
        "steps": [
            step("motion", segment=segment("ramp", 20, "z", 0.05)),
            step("motion", segment=segment("ramp", 20, "x", -0.04)),
            step("home", duration_steps=2, on_timeout="continue"),  # too short for the ramp
            step("home", duration_steps=150),
        ],
    }
)


def test_a_home_step_brings_the_hand_back_to_the_start_pose(
    suite: GoalSuite, tmp_path: Path
) -> None:
    runner = JobRunner(HoldSource(), software.current("none"), image_size=SIZE, suite=suite)
    try:
        [episode] = runner.run_starts(
            tmp_path / "run",
            "j0001",
            HOMEWARD,
            CHEESE_THEN_PLATE,
            development_starts(6, 1),
            record_video=False,
            first_episode=101,
        )
    finally:
        runner.close()
    outcomes = [(r.kind.value, r.outcome.value) for r in episode.program_steps]
    assert outcomes == [
        ("motion", "reached"),
        ("motion", "reached"),
        ("home", "timeout"),
        ("home", "reached"),
    ]
    assert episode.program_steps[2].steps_used == 2
    with LogReader(tmp_path / "run", materialize=True, lock=True) as reader:
        start = reader.query("SELECT start_eef_pos FROM job_episodes").to_pylist()[0]
        rows = reader.query(
            "SELECT step, eef_pos, gripper_command, program_step FROM robot_steps ORDER BY step"
        ).to_pylist()
        # A check reads how far each step began from the start pose.
        began = reader.query(
            "SELECT s.program_step, sqrt((r.eef_pos[1] - j.start_eef_pos[1]) ^ 2 + "
            "(r.eef_pos[2] - j.start_eef_pos[2]) ^ 2 + (r.eef_pos[3] - j.start_eef_pos[3]) ^ 2) "
            "AS distance FROM job_step_events s JOIN job_episodes j USING (episode_id) "
            "JOIN robot_steps r ON r.episode_id = s.episode_id AND r.step = s.start_step "
            "ORDER BY s.program_step"
        ).to_pylist()
    origin = np.asarray(start["start_eef_pos"])
    assert np.allclose(origin, rows[0]["eef_pos"])
    distance = [float(np.linalg.norm(np.asarray(r["eef_pos"]) - origin)) for r in rows]
    assert max(distance) > 0.05  # the motions took the hand away
    assert distance[-1] < 0.01  # and the home step brought it back
    assert all(r["gripper_command"] == -1.0 for r in rows if r["program_step"] in (2, 3))
    assert began[0]["distance"] < 0.005
    assert began[2]["distance"] > 0.05 and began[3]["distance"] < began[2]["distance"]


def test_plan_task_numbers_reach_libero_s_benchmark_by_name(suite: GoalSuite) -> None:
    # Lane S: the plan's tasks_info.txt order is not LIBERO's benchmark order.
    from fleetlab.contracts.jobs import GOAL_SUITE_TASKS

    assert suite.bench == (2, 3, 7, 4, 8, 9, 6, 0, 5, 1)
    for task in (2, 4):  # plan 2 is benchmark 7, plan 4 is benchmark 8
        env = make_env(suite.bench[task], SIZE)
        try:
            assert env.task_description == GOAL_SUITE_TASKS[task].replace("_", " ")
            assert suite.language[task] == env.task_description
        finally:
            env.close()


def test_job_one_s_goals_name_the_cabinet_that_owns_its_top(suite: GoalSuite) -> None:
    # Gate 1's job one: the cheese into the bowl, then the bowl onto the cabinet top.
    job_one = state(
        [
            {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
            {"predicate": "On", "objects": ["akita_black_bowl_1", "wooden_cabinet_1_top_side"]},
        ],
        6,
        [{"first": 0, "then": 1}],
    )
    env = make_env(suite.bench[6], SIZE)
    try:
        env.reset(_identity(suite, 6, 10), [])
        goals = SceneGoals(env._rs, job_one.requested_goals)
        assert goals.holds() == (False, False)
        assert goals.named == ("cream_cheese_1", "akita_black_bowl_1", "wooden_cabinet_1")
    finally:
        env.close()
    instructions = suite.instructions()
    assert instructions["On akita_black_bowl_1 wooden_cabinet_1_top_side"] == (
        "put the bowl on top of the cabinet"
    )


def test_a_re_render_report_carries_the_video_s_hash(played: Path, tmp_path: Path) -> None:
    # Review finding 6: bind the verified MP4 to its report by its bytes.
    import hashlib

    from fleetlab.jobs.replay import rerender

    report = rerender(played, "e000102", tmp_path, frame_every=4)
    assert report["accepted"]
    video = (tmp_path / report["video"]).read_bytes()
    assert report["video_sha256"] == hashlib.sha256(video).hexdigest()
    original = (played / "log" / "video" / "e000102.scene.mp4").read_bytes()
    assert report["original_video_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["hashes_compared"] == report["recorded_hashes"] == report["steps"] + 1


def test_a_resumed_batch_keeps_only_the_same_run(suite: GoalSuite, tmp_path: Path) -> None:
    # Review finding 4: resume attributed old episodes to a changed program or start.
    from fleetlab.contracts.jobs import JobRequest
    from fleetlab.jobs.runner import run_batch

    short = JobProgram.model_validate(
        {
            "name": "short-run",
            "purpose": "resume",
            "notes": "",
            "steps": [step("motion", segment=segment("ramp", 6, "z", 0.02)), step("wait", steps=3)],
        }
    )

    def batch(program: JobProgram, first: int = 0) -> dict[str, Any]:
        request = JobRequest(
            program=program, starts=development_starts(6, 2, first), record_video=True
        )
        return {
            "format": "fleetlab.job-batch/1",
            "state": CHEESE_THEN_PLATE.model_dump(mode="json"),
            "jobs": [{"job_id": "j0001", "request": request.model_dump(mode="json")}],
        }

    out = tmp_path / "run"
    runner = JobRunner(HoldSource(), software.current("none"), image_size=SIZE, suite=suite)
    try:
        assert run_batch(batch(short), out, runner)["jobs"][0]["status"] == "done"
        lines = (out / "episodes.jsonl").read_text()
        record = (out / "run.json").read_text()
        again = run_batch(batch(short), out, runner, resume=True)
        assert again["jobs"][0]["status"] == "done"
        assert (out / "episodes.jsonl").read_text() == lines  # nothing ran again
        changed = short.model_copy(update={"notes": "another program"})
        refused = run_batch(batch(changed), out, runner, resume=True)
        assert refused["jobs"][0]["status"] == "refused"
        assert "another request" in " ".join(refused["jobs"][0]["problems"])
        moved = run_batch(batch(short, first=20), out, runner, resume=True)
        assert moved["jobs"][0]["status"] == "refused"
        assert (out / "episodes.jsonl").read_text() == lines
        assert (
            json.loads((out / "run.json").read_text())["requests"] == json.loads(record)["requests"]
        )
        (out / "log" / "video" / "e000102.wrist.mp4").unlink()
        missing = run_batch(batch(short), out, runner, resume=True)
        assert missing["jobs"][0]["status"] == "refused"
        assert "no wrist video" in " ".join(missing["jobs"][0]["problems"])
        with pytest.raises(ValueError, match="pass --resume"):
            run_batch(batch(short), out, runner)
    finally:
        runner.close()
