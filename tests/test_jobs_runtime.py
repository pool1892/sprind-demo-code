"""The jobs runtime without the simulator (`fleetlab.jobs`, docs/design/jobs.md).

The host-fixed delivery check on constructed states, the job tables from a synthetic
episode (written, validated as a visible tree, read through `LogReader` as the tool
server's `query` reads them, merged), the library and its checks, the naive chain, the
batch's refusals and the release receipt. The simulator tests are
tests/test_jobs_runtime_sim.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import numpy as np
import pydantic
import pytest

from fleetlab.contracts.env import Camera
from fleetlab.contracts.jobs import (
    DELIVERY,
    Acceptance,
    CheckSpec,
    DeliveryVerdict,
    GoalTrace,
    JobEpisode,
    JobProgram,
    JobStepKind,
    LibraryKind,
    OrderVerdict,
    ProgramStepResult,
    ReleaseReceipt,
    RequestedGoal,
    RequestState,
    SavedJob,
    StepOutcome,
)
from fleetlab.contracts.log import JOB_TABLE_NAMES, log_tree_problems
from fleetlab.jobs import library, tables
from fleetlab.jobs.__main__ import main as cli
from fleetlab.jobs.__main__ import release_receipt, saved_job_from_run
from fleetlab.jobs.check import (
    DeliveryState,
    at_rest,
    delivery_verdict,
    gripper_state,
    named_objects,
    robot_touching,
)
from fleetlab.jobs.runner import HoldSource, JobRunner, naive_program, run_batch
from fleetlab.jobs.starts import development_starts, release_starts, start_identity
from fleetlab.log import LogReader

CHEESE_IN_BOWL = RequestedGoal.model_validate(
    {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]}
)
BOWL_ON_PLATE = RequestedGoal.model_validate(
    {"predicate": "On", "objects": ["akita_black_bowl_1", "plate_1"]}
)
GOALS = [CHEESE_IN_BOWL, BOWL_ON_PLATE]
NAMED = ("cream_cheese_1", "akita_black_bowl_1", "plate_1")
STILL = np.zeros(6)


def scene(**update: Any) -> DeliveryState:
    base: dict[str, Any] = {
        "goals_hold": (True, True),
        "contacts": (("akita_black_bowl_1", "cream_cheese_1"), ("akita_black_bowl_1", "plate_1")),
        "velocities": {n: STILL for n in NAMED},
        "gripper_opening": 0.075,
        "gripper_command": -1.0,
    }
    return DeliveryState(**{**base, **update})


# ---- the delivery check ----------------------------------------------------------------------


def test_delivered_when_every_part_holds() -> None:
    verdict = delivery_verdict(scene(), GOALS, NAMED)
    assert verdict.delivered and verdict.reason == "delivered"
    assert verdict.goals_hold == [True, True] and verdict.robot_touches == []


def test_a_goal_that_does_not_hold_is_not_delivered() -> None:
    verdict = delivery_verdict(scene(goals_hold=(True, False)), GOALS, NAMED)
    assert not verdict.delivered
    assert "goal 1 (On akita_black_bowl_1 plate_1) does not hold" in verdict.reason


def test_the_robot_may_not_touch_a_named_object() -> None:
    held = scene(contacts=(("akita_black_bowl_1", "gripper0_finger1_pad"),))
    verdict = delivery_verdict(held, GOALS, NAMED)
    assert not verdict.delivered and verdict.robot_touches == ["akita_black_bowl_1"]
    # A robot body touching something no goal names does not matter.
    other = scene(contacts=(("wine_bottle_1", "robot0_link7"),))
    assert delivery_verdict(other, GOALS, NAMED).delivered
    assert robot_touching((("plate_1", "robot0_link3"),), "plate_1")
    assert not robot_touching((("plate_1", "cream_cheese_1"),), "plate_1")


def test_rest_is_below_one_centimetre_per_second_and_point_one_five_radians() -> None:
    assert DELIVERY.linear_m_s == 0.01 and DELIVERY.angular_rad_s == 0.15
    assert at_rest([0.0099, 0, 0, 0, 0, 0])
    assert not at_rest([0.0101, 0, 0, 0, 0, 0])
    assert not at_rest([0.006, 0.006, 0.006, 0, 0, 0])  # 1.04 cm/s over three axes
    assert at_rest([0, 0, 0, 0.149, 0, 0])
    assert not at_rest([0, 0, 0, 0.1, 0.1, 0.1])  # 0.17 rad/s
    moving = scene(
        velocities={**{n: STILL for n in NAMED}, "plate_1": np.array([0.02, 0, 0, 0, 0, 0])}
    )
    verdict = delivery_verdict(moving, GOALS, NAMED)
    assert not verdict.delivered and verdict.not_at_rest == ["plate_1"]


def test_an_object_without_a_free_joint_is_checked_for_contact_only() -> None:
    stove = RequestedGoal.model_validate({"predicate": "Turnon", "objects": ["flat_stove_1"]})
    state = DeliveryState(goals_hold=(True,), contacts=(), velocities={}, gripper_opening=0.08)
    assert delivery_verdict(state, [stove], ("flat_stove_1",)).delivered
    touching = DeliveryState(
        goals_hold=(True,), contacts=(("flat_stove_1", "gripper0_finger2"),), gripper_opening=0.08
    )
    assert not delivery_verdict(touching, [stove], ("flat_stove_1",)).delivered


def test_the_gripper_is_open_when_commanded_open_and_six_centimetres_apart() -> None:
    assert gripper_state(0.061, -1.0) == (True, True)
    assert gripper_state(0.059, -1.0) == (True, False)
    assert gripper_state(0.075, 1.0) == (False, False)
    assert gripper_state(0.075, None) == (True, True)  # never commanded: open as it starts
    closed = delivery_verdict(scene(gripper_command=1.0), GOALS, NAMED)
    assert not closed.delivered and "commanded closed" in closed.reason
    narrow = delivery_verdict(scene(gripper_opening=0.04), GOALS, NAMED)
    assert not narrow.delivered and narrow.gripper_commanded_open
    assert "commanded open, 0.040 m apart" in narrow.reason


def test_the_goals_name_their_objects_and_regions_owners_never_the_table() -> None:
    owners = {
        "flat_stove_1_cook_region": "flat_stove_1",
        "main_table_stove_front_region": None,
        "wooden_cabinet_1_top_region": "wooden_cabinet_1",
    }

    def owner(name: str) -> str | None:
        return owners.get(name, name)

    goals = [
        RequestedGoal.model_validate(
            {"predicate": "On", "objects": ["plate_1", "main_table_stove_front_region"]}
        ),
        RequestedGoal.model_validate(
            {"predicate": "On", "objects": ["akita_black_bowl_1", "flat_stove_1_cook_region"]}
        ),
        RequestedGoal.model_validate(
            {"predicate": "In", "objects": ["akita_black_bowl_1", "wooden_cabinet_1_top_region"]}
        ),
    ]
    assert named_objects(goals, owner) == (
        "plate_1",
        "akita_black_bowl_1",
        "flat_stove_1",
        "wooden_cabinet_1",
    )


# ---- the tables ------------------------------------------------------------------------------


def program() -> JobProgram:
    policy = {
        "kind": "policy",
        "instruction": "put the cream cheese in the bowl",
        "until": {"goal": 0, "steps": None},
        "max_steps": 300,
        "segment": None,
        "steps": None,
    }
    wait = {
        **policy,
        "kind": "wait",
        "instruction": None,
        "until": None,
        "max_steps": None,
        "steps": 2,
    }
    return JobProgram.model_validate(
        {"name": "cheese-then-wait", "purpose": "test", "steps": [policy, wait], "notes": ""}
    )


def request_state() -> RequestState:
    return RequestState.model_validate(
        {
            "request_text": "the cheese in the bowl, then the bowl on the plate",
            "requested_goals": [g.model_dump() for g in GOALS],
            "base_task": 6,
            "order_rules": [{"first": 0, "then": 1}],
            "budget": {
                "starts_per_request": 20,
                "requests_per_session": 4,
                "sessions": 6,
                "starts_per_investigation": 400,
            },
            "acceptance": {"N": 20, "K": 16},
            "message_log": [],
        }
    )


def synthetic_episode(
    root: Path,
    episode_id: str,
    job_id: str = "j0001",
    *,
    delivered: bool = True,
    order_kept: bool = True,
    init_state: int = 3,
) -> JobEpisode:
    """Five control steps of a made-up episode, written as the runner writes one."""
    objects = ["akita_black_bowl_1", "cream_cheese_1"]
    rec = tables.EpisodeRecorder(episode_id, objects, 2)
    for t in range(6):
        contacts = [("akita_black_bowl_1", "gripper0_finger1_pad")] if t == 2 else []
        rec.observe(
            t,
            eef_pos=[0.1, 0.2, 1.0 + 0.01 * t],
            eef_quat=[1, 0, 0, 0],
            gripper_opening=0.07,
            poses={o: [0.0, 0.1 * i, 0.9, 1, 0, 0, 0] for i, o in enumerate(objects)},
            velocities={o: np.full(6, 0.001 * t) for o in objects},
            contacts=contacts,
            goals_hold=(t >= 3, False),
            frame_index=t // 4 if t % 4 == 0 else None,
        )
        if t < 5:
            rec.command(
                np.arange(7, dtype=np.float32) * 0.1,
                instruction="put the cream cheese in the bowl" if t < 3 else None,
                program_step=0 if t < 3 else 1,
                phase="program",
            )
    rec.finish()
    verdict = DeliveryVerdict(
        delivered=delivered,
        goals_hold=[True, delivered],
        robot_touches=[],
        not_at_rest=[],
        gripper_commanded_open=True,
        gripper_opening_m=0.07,
        reason="delivered" if delivered else "goal 1 (On akita_black_bowl_1 plate_1) does not hold",
    )
    start = start_identity(6, init_state)
    episode = JobEpisode(
        job_id=job_id,
        episode_id=episode_id,
        start=start,
        program_digest=program().digest,
        delivered=delivered,
        goals=[
            GoalTrace(goal=0, first_held_step=3, held_at_end=True),
            GoalTrace(goal=1, first_held_step=None, held_at_end=delivered),
        ],
        program_steps=[
            ProgramStepResult(
                step=0,
                kind=JobStepKind.policy,
                outcome=StepOutcome.reached,
                steps_used=3,
                start_control_step=0,
                end_control_step=3,
            ),
            ProgramStepResult(
                step=1,
                kind=JobStepKind.wait,
                outcome=StepOutcome.reached,
                steps_used=2,
                start_control_step=3,
                end_control_step=5,
            ),
        ],
        tail_steps=0,
        control_steps=5,
        order_verdicts=[OrderVerdict(rule=0, first=0, then=1, kept=order_kept, detail="x")],
        order_kept=order_kept,
        delivery=verdict,
        end_state_hash="00" * 8,
    )
    tables.write_episode_tables(root, episode_id, rec.tables())
    tables.append_episode(root, episode)
    episodes, events = _ROWS.setdefault(root, ([], []))
    episodes.append(tables.episode_row(episode, {Camera.scene: None, Camera.wrist: None}, "x"))
    events.extend(tables.step_event_rows(episode, program(), None))
    tables.write_run_tables(root, episodes, events)
    record_path = root / tables.RUN_RECORD
    if record_path.is_file():
        record = json.loads(record_path.read_text())
    else:
        state = request_state().model_dump(mode="json")
        record = {"format": "fleetlab.job-run/1", "state": state, "jobs": []}
    if not any(j["job_id"] == job_id for j in record["jobs"]):
        record["jobs"].append({"job_id": job_id, "program": program().model_dump(mode="json")})
    record_path.write_text(json.dumps(record))
    return episode


# The run-level rows written so far, by run directory, as the runner keeps them.
_ROWS: dict[Path, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    synthetic_episode(root, "e000101", delivered=True, init_state=3)
    synthetic_episode(root, "e000102", delivered=False, init_state=4)
    return root


def test_a_run_s_tree_is_a_clean_visible_log_with_all_five_tables(run: Path) -> None:
    assert log_tree_problems(run) == []
    with LogReader(run, materialize=True, lock=True) as reader:
        assert set(JOB_TABLE_NAMES) <= set(reader.tables)
        counts = {
            t: reader.query(f"SELECT count(*) AS n FROM {t}").to_pylist()[0]["n"]
            for t in JOB_TABLE_NAMES
        }
        assert counts == {
            "job_episodes": 2,
            "job_step_events": 4,
            "object_steps": 2 * 6 * 2,
            "goal_steps": 2 * 6 * 2,
            "robot_steps": 2 * 6,
        }
        last = reader.query(
            "SELECT phase, commanded_action, gripper_command FROM robot_steps "
            "WHERE episode_id = 'e000101' ORDER BY step DESC LIMIT 1"
        ).to_pylist()[0]
        assert last == {"phase": "end", "commanded_action": None, "gripper_command": None}
        touch = reader.query(
            "SELECT step, in_contact_with FROM object_steps WHERE in_contact_with_robot "
            "AND episode_id = 'e000101'"
        ).to_pylist()
        assert touch == [{"step": 2, "in_contact_with": ["gripper0_finger1_pad"]}]
        first = reader.query(
            "SELECT goal, min(step) AS first FROM goal_steps WHERE holds AND "
            "episode_id = 'e000101' GROUP BY goal"
        ).to_pylist()
        assert first == [{"goal": 0, "first": 3}]
        delivered = reader.query(
            "SELECT episode_id, delivered, goals_held_at_end FROM job_episodes ORDER BY episode_id"
        ).to_pylist()
        assert [r["delivered"] for r in delivered] == [True, False]


def test_the_recorder_refuses_a_row_without_its_command() -> None:
    rec = tables.EpisodeRecorder("e000001", [], 1)
    values: dict[str, Any] = {
        "eef_pos": [0, 0, 0],
        "eef_quat": [1, 0, 0, 0],
        "gripper_opening": 0.0,
        "poses": {},
        "velocities": {},
        "contacts": [],
        "goals_hold": (False,),
        "frame_index": None,
    }
    rec.observe(0, **values)
    with pytest.raises(ValueError, match="no command"):
        rec.observe(1, **values)
    with pytest.raises(ValueError, match="finish"):
        rec.tables()


def test_runs_merge_into_one_tree_and_refuse_a_repeated_episode(run: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    synthetic_episode(other, "e000201", "j0002")
    stage = tmp_path / "stage"
    assert tables.write_job_tables(stage, [run, other]) == ["e000101", "e000102", "e000201"]
    assert log_tree_problems(stage) == []
    with LogReader(stage) as reader:
        rows = reader.query("SELECT job_id, episode_id FROM job_episodes").to_pylist()
        assert [r["episode_id"] for r in rows] == ["e000101", "e000102", "e000201"]
        n = reader.query("SELECT count(DISTINCT episode_id) AS n FROM robot_steps").to_pylist()
        assert n == [{"n": 3}]
    with pytest.raises(ValueError, match="e000101"):
        tables.write_job_tables(tmp_path / "again", [run, run])


def test_the_replay_record_needs_one_hash_more_than_actions(tmp_path: Path) -> None:
    executed = np.zeros((3, 7), np.float32)
    with pytest.raises(ValueError):
        tables.save_replay_record(tmp_path, "e000001", executed, ["a"] * 3, "{}")
    (tmp_path / "log").mkdir()
    tables.save_replay_record(tmp_path, "e000001", executed, ["a", "b", "c", "d"], '{"x": 1}')
    back, hashes, ident = tables.load_replay_record(tmp_path, "e000001")
    assert back.shape == (3, 7) and hashes == ["a", "b", "c", "d"] and ident == '{"x": 1}'


# ---- the library and its checks --------------------------------------------------------------


def saved(name: str = "cheese-then-plate", **update: Any) -> SavedJob:
    base: dict[str, Any] = {
        "name": name,
        "program": program().model_dump(),
        "requested_goals": [g.model_dump() for g in GOALS],
        "base_task": 6,
        "order_rules": [{"first": 0, "then": 1}],
        "acceptance": {"N": 20, "K": 16},
        "saved_from": ["j0001"],
        "receipt": None,
    }
    return SavedJob.model_validate({**base, **update})


def check_spec(**update: Any) -> CheckSpec:
    base: dict[str, Any] = {
        "name": "undelivered",
        "purpose": "the runs of a job that were not delivered, with why",
        "parameters": [{"name": "job", "type": "str"}, {"name": "first", "type": "int"}],
        "sql": "SELECT episode_id, delivery_detail FROM job_episodes WHERE job_id = $job "
        "AND NOT delivered AND init_state >= $first ORDER BY episode_id",
        "expects": "no rows once the job is fixed",
    }
    return CheckSpec.model_validate({**base, **update})


def test_the_library_saves_lists_and_refuses_a_silent_replace(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    path = library.save_job(lib, saved())
    assert path == lib / "jobs" / "cheese-then-plate.json"
    assert library.load_job(lib, "cheese-then-plate") == saved()
    assert library.save_job(lib, saved()) == path  # the same content again: nothing changes
    changed = saved(program={**program().model_dump(), "notes": "a second version"})
    with pytest.raises(library.LibraryError, match="another job"):
        library.save_job(lib, changed)
    library.save_job(lib, changed, replace=True)
    library.save_check(lib, check_spec())
    library.save_recipe(lib, "step-response", "a probe recipe", {"request": {"name": "x"}})
    entries = library.list_entries(lib)
    assert [(e.kind, e.name) for e in entries] == [
        (LibraryKind.job, "cheese-then-plate"),
        (LibraryKind.check, "undelivered"),
        (LibraryKind.recipe, "step-response"),
    ]
    assert entries[0].digest == library.digest_of(changed.model_dump(mode="json"))
    assert library.load_index(lib).entries == entries
    assert library.load_recipe(lib, "step-response")["purpose"] == "a probe recipe"
    with pytest.raises(library.LibraryError):
        library.load_check(lib, "absent")
    with pytest.raises(library.LibraryError, match="not a library name"):
        library.load_job(lib, "../escape")


def test_a_check_runs_read_only_with_typed_parameters(run: Path, tmp_path: Path) -> None:
    spec = check_spec()
    rows = library.run_check(spec, run, {"job": "j0001", "first": 0})
    assert rows == [
        {
            "episode_id": "e000102",
            "delivery_detail": "goal 1 (On akita_black_bowl_1 plate_1) does not hold",
        }
    ]
    assert library.run_check(spec, run, library.parse_params(spec, ["job=j0001", "first=4"])) == [
        rows[0]
    ]
    assert library.run_check(spec, run, {"job": "j0001", "first": 5}) == []
    with pytest.raises(library.CheckError, match="needs first"):
        library.run_check(spec, run, {"job": "j0001"})
    with pytest.raises(library.CheckError, match="not a int"):
        library.parse_params(spec, ["job=j0001", "first=three"])
    with pytest.raises(library.CheckError, match="takes no parameter"):
        library.run_check(spec, run, {"job": "j0001", "first": 0, "other": 1})
    library.save_check(tmp_path / "lib", spec)
    by_name = library.run_saved_check(
        tmp_path / "lib", "undelivered", run, {"job": "j0001", "first": 0}
    )
    assert by_name == rows
    writes = check_spec(sql="CREATE TABLE x AS SELECT * FROM job_episodes", parameters=[])
    with pytest.raises(library.CheckError, match="one SELECT"):
        library.run_check(writes, run, {})
    escape = check_spec(sql="SELECT * FROM read_csv('/etc/passwd')", parameters=[])
    with pytest.raises(duckdb.Error):  # file access is off: DuckDB refuses
        library.run_check(escape, run, {})


def test_the_command_line_runs_a_saved_check(run: Path, tmp_path: Path, capsys: Any) -> None:
    lib = tmp_path / "lib"
    library.save_check(lib, check_spec())
    assert (
        cli(
            [
                "check",
                "undelivered",
                "--library",
                str(lib),
                "--run",
                str(run),
                "--param",
                "job=j0001",
                "--param",
                "first=0",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["check"] == "undelivered" and len(printed["rows"]) == 1
    assert cli(["list", "--library", str(lib)]) == 0
    assert "undelivered" in capsys.readouterr().out


def test_a_job_is_saved_from_its_runs_unless_they_broke_the_order(
    run: Path, tmp_path: Path
) -> None:
    job = saved_job_from_run(run, "j0001", "cheese-then-plate")
    assert job.program == program() and job.acceptance == Acceptance(N=20, K=16)
    assert job.requested_goals == GOALS and job.saved_from == ["j0001"]
    broken = tmp_path / "broken"
    broken.mkdir()
    synthetic_episode(broken, "e000301", "j0003", order_kept=False)
    with pytest.raises(SystemExit, match="broke an order rule"):
        saved_job_from_run(broken, "j0003", "not-saved")


# ---- the naive chain, the batch, the receipt -------------------------------------------------


def test_the_naive_chain_switches_on_each_goal_under_its_task_s_instruction() -> None:
    instructions = {
        CHEESE_IN_BOWL.text: "put the cream cheese in the bowl",
        BOWL_ON_PLATE.text: "put the bowl on the plate",
    }
    naive = naive_program(GOALS, instructions)
    assert [
        (s.instruction, s.until.goal if s.until else None, s.max_steps) for s in naive.steps
    ] == [
        ("put the cream cheese in the bowl", 0, 300),
        ("put the bowl on the plate", 1, 300),
    ]
    assert all(s.on_timeout is not None and s.on_timeout.value == "stop" for s in naive.steps)
    with pytest.raises(ValueError, match="no goal-suite skill"):
        naive_program(
            [RequestedGoal.model_validate({"predicate": "Up", "objects": ["plate_1"]})],
            instructions,
        )


def test_the_batch_refuses_what_it_cannot_run(tmp_path: Path) -> None:
    fake_suite: Any = SimpleNamespace(bench=tuple(range(10)))
    from fleetlab.engine import software

    runner = JobRunner(
        HoldSource(), software.current("none"), suite=fake_suite, log=lambda line: None
    )
    good: dict[str, Any] = {"program": program().model_dump(mode="json"), "record_video": False}
    dev = [s.model_dump() for s in development_starts(6, 2)]
    batch = {
        "format": "fleetlab.job-batch/1",
        "state": request_state().model_dump(mode="json"),
        "jobs": [
            {
                "job_id": "j0001",
                "request": {**good, "starts": [release_starts(6, 1)[0].model_dump()]},
            },
            {"job_id": "j0002", "request": {**good, "starts": [s | {"base_task": 4} for s in dev]}},
            {"job_id": "jx", "request": {**good, "starts": dev}},
            {
                "job_id": "j0004",
                "request": {
                    **good,
                    "starts": dev,
                    "program": {
                        **good["program"],
                        "steps": [
                            {**good["program"]["steps"][0], "until": {"goal": 5, "steps": None}}
                        ],
                    },
                },
            },
        ],
    }
    result = run_batch(batch, tmp_path / "out", runner)
    assert [j["status"] for j in result["jobs"]] == ["refused"] * 4
    assert "reserved for release runs" in " ".join(result["jobs"][0]["problems"])
    assert "the request's is 6" in " ".join(result["jobs"][1]["problems"])
    assert "not a job id" in " ".join(result["jobs"][2]["problems"])
    assert "the request has 2 goals" in " ".join(result["jobs"][3]["problems"])
    assert json.loads((tmp_path / "out" / "batch_result.json").read_text()) == result
    with pytest.raises(ValueError, match="not a job batch"):
        run_batch({**batch, "format": "x"}, tmp_path / "out2", runner)


@pytest.mark.parametrize("invalid", [4, 31])
def test_an_invalid_action_horizon_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: int
) -> None:
    from fleetlab.engine import software
    from fleetlab.jobs import runner as runner_module

    def no_scene(*args: Any) -> Any:
        raise AssertionError("a scene was built")

    monkeypatch.setattr(runner_module, "make_env", no_scene)
    fake_suite: Any = SimpleNamespace(bench=tuple(range(10)))
    runner = JobRunner(
        HoldSource(), software.current("none"), suite=fake_suite, log=lambda line: None
    )
    starts = development_starts(6, 2)
    batch = {
        "format": "fleetlab.job-batch/1",
        "state": request_state().model_dump(mode="json"),
        "jobs": [
            {
                "job_id": "j0001",
                "request": {
                    "program": {**program().model_dump(mode="json"), "actions_per_chunk": invalid},
                    "starts": [s.model_dump() for s in starts],
                    "record_video": False,
                },
            }
        ],
    }
    [job] = run_batch(batch, tmp_path / "out", runner)["jobs"]
    assert job["status"] == "refused" and job["episode_ids"] == []  # free: nothing ran
    assert "actions_per_chunk" in " ".join(job["problems"])
    assert tables.read_episodes(tmp_path / "out") == []


NAIVE_INSTRUCTIONS = {CHEESE_IN_BOWL.text: "a", BOWL_ON_PLATE.text: "b"}


def _paired(
    delivered: list[bool],
    naive: list[bool],
    kept: list[bool] | None = None,
    digests: tuple[str, str] | None = None,
) -> Any:
    starts = release_starts(6, len(delivered))
    kept = kept or [True] * len(delivered)
    ours_digest, naive_digest = digests or (
        saved().program.digest,
        naive_program(GOALS, NAIVE_INSTRUCTIONS).digest,
    )

    def ep(eid: str, d: bool, k: bool, s: Any, digest: str) -> JobEpisode:
        verdict = DeliveryVerdict(
            delivered=d,
            goals_hold=[d, d],
            robot_touches=[],
            not_at_rest=[],
            gripper_commanded_open=True,
            gripper_opening_m=0.07,
            reason="delivered" if d else "no",
        )
        return JobEpisode(
            job_id="j0001",
            episode_id=eid,
            start=s,
            program_digest=digest,
            delivered=d,
            goals=[],
            program_steps=[],
            tail_steps=0,
            control_steps=1,
            order_verdicts=[],
            order_kept=k,
            delivery=verdict,
            end_state_hash="h",
        )

    ours = [
        ep(f"e{101 + i:06d}", d, k, s, ours_digest)
        for i, (d, k, s) in enumerate(zip(delivered, kept, starts, strict=True))
    ]
    theirs = [
        ep(f"e{201 + i:06d}", d, True, s, naive_digest)
        for i, (d, s) in enumerate(zip(naive, starts, strict=True))
    ]
    return ours, theirs


def test_the_receipt_pairs_the_program_with_the_naive_chain(tmp_path: Path) -> None:
    job = saved()
    naive = naive_program(GOALS, NAIVE_INSTRUCTIONS)
    delivered = [True] * 17 + [False] * 3
    naive_delivered = [False] * 12 + [True] * 6 + [False, True]
    ours, theirs = _paired(delivered, naive_delivered)
    receipt = release_receipt(job, naive, ours, theirs, tmp_path, fresh=True)
    assert (receipt.N, receipt.K, receipt.delivered, receipt.naive_delivered) == (20, 16, 17, 7)
    assert (receipt.rescued, receipt.broken) == (12, 2)
    assert receipt.passed and receipt.released and receipt.order_violations == 0
    assert [s.init_state for s in receipt.release_starts] == list(range(30, 50))
    assert receipt.starts_fresh is True
    # A broken order rule, a short run, or starts examined before: not released.
    ours_broken, _ = _paired(delivered, naive_delivered, [False] + [True] * 19)
    assert not release_receipt(job, naive, ours_broken, theirs, tmp_path, fresh=True).released
    few_ours, few_theirs = _paired(delivered[:5], naive_delivered[:5])
    short = release_receipt(job, naive, few_ours, few_theirs, tmp_path, fresh=True)
    assert short.passed is None and not short.released
    again = release_receipt(job, naive, ours, theirs, tmp_path, fresh=False)
    assert again.passed and not again.released and again.starts_fresh is False
    failing_ours, _ = _paired([True] * 15 + [False] * 5, naive_delivered)
    assert not release_receipt(job, naive, failing_ours, theirs, tmp_path, fresh=True).passed


def test_the_receipt_certifies_only_episodes_of_its_programs(tmp_path: Path) -> None:
    job = saved()
    naive = naive_program(GOALS, NAIVE_INSTRUCTIONS)
    other, theirs = _paired([True] * 20, [False] * 20, digests=("e" * 64, naive.digest))
    with pytest.raises(ValueError, match="another program than cheese-then-plate"):
        release_receipt(job, naive, other, theirs, tmp_path, fresh=True)
    ours, not_naive = _paired([True] * 20, [False] * 20, digests=(job.program.digest, "f" * 64))
    with pytest.raises(ValueError, match="another program than the naive chain"):
        release_receipt(job, naive, ours, not_naive, tmp_path, fresh=True)
    # One delivered pair repeated twenty times is not twenty deliveries.
    with pytest.raises(ValueError):
        release_receipt(job, naive, [ours[0]] * 20, [theirs[0]] * 20, tmp_path, fresh=True)


def test_the_tool_server_s_query_sees_the_job_tables(run: Path, tmp_path: Path) -> None:
    from fleetlab.toolserver.server import Paths, ToolBox

    work = tmp_path / "session"
    for name in ("memory", "outbox", "results"):
        (work / name).mkdir(parents=True)
    box = ToolBox(
        Paths(export=run, memory=work / "memory", outbox=work / "outbox", results=work / "results"),
        enabled=["query"],
    )
    try:
        for name, expected in (
            ("job_episodes", 2),
            ("job_step_events", 4),
            ("object_steps", 24),
            ("goal_steps", 24),
            ("robot_steps", 12),
        ):
            result = box.call("query", {"sql": f"SELECT count(*) AS n FROM {name}"})
            text = json.loads(result.content[0]["text"])
            assert text["rows"] == [[expected]], name
        refused = box.call("query", {"sql": "DELETE FROM job_episodes"})
        assert refused.is_error
    finally:
        box.close()


# ---- the review of 48e7288 (19 October): one regression per finding -------------------------


def test_saving_keeps_the_approved_acceptance_and_refuses_another(run: Path) -> None:
    # Finding 1: an override could replace the approved N=20, K=16 by N=1, K=0.
    with pytest.raises(SystemExit, match="keeps the approved condition"):
        saved_job_from_run(run, "j0001", "cheese-then-plate", Acceptance(N=1, K=0))
    same = saved_job_from_run(run, "j0001", "cheese-then-plate", Acceptance(N=20, K=16))
    assert same.acceptance == Acceptance(N=20, K=16)
    record = json.loads((run / "run.json").read_text())
    record["state"]["acceptance"] = None
    (run / "run.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="not an acceptance the lead approves"):
        saved_job_from_run(run, "j0001", "cheese-then-plate", Acceptance(N=1, K=0))
    with pytest.raises(pydantic.ValidationError, match="not one the lead approves"):
        saved(acceptance={"N": 1, "K": 0})
    with pytest.raises(SystemExit):
        cli(
            [
                "save-job",
                "cheese-then-plate",
                "--library",
                str(run.parent / "lib"),
                "--run",
                str(run),
                "--job-id",
                "j0001",
                "--acceptance",
                "1,0",
            ]
        )


class _Interrupted(RuntimeError):
    pass


def _fake_runner(fail: list[bool], delivered: bool = True) -> Any:
    """A runner that plays nothing: it returns episodes of the programs it is given."""

    class FakeRunner:
        def __init__(self, source: Any, soft: Any, **kw: Any) -> None:
            self.suite = SimpleNamespace(instructions=lambda: NAIVE_INSTRUCTIONS)

        def run_starts(
            self, root: Path, job_id: str, program: JobProgram, state: Any, starts: Any, **kw: Any
        ) -> list[JobEpisode]:
            if fail[0]:
                raise _Interrupted("the attempt was killed")
            ours, _ = _paired(
                [delivered] * len(starts),
                [False] * len(starts),
                digests=(program.digest, program.digest),
            )
            first = kw["first_episode"]
            return [
                e.model_copy(update={"job_id": job_id, "episode_id": f"e{first + i:06d}"})
                for i, e in enumerate(ours)
            ]

        def close(self) -> None:
            pass

    return FakeRunner


def test_an_interrupted_or_partial_release_attempt_uses_up_its_starts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Finding 2: a partial or killed attempt left no receipt, so a later one counted as fresh.
    lib = tmp_path / "lib"
    library.save_job(lib, saved())
    fail = [True]
    monkeypatch.setattr("fleetlab.jobs.runner.JobRunner", _fake_runner(fail))
    monkeypatch.setattr("fleetlab.jobs.runner.make_source", lambda name: (None, None))
    run = ["run", "cheese-then-plate", "--library", str(lib), "--starts", "release"]
    with pytest.raises(_Interrupted):
        cli([*run, "--n", "5", "--out", str(tmp_path / "killed")])
    assert library.exposed_release_starts(lib, saved()) == set(range(30, 35))
    library.save_job(lib, saved(), replace=True)  # saving again forgets nothing
    fail[0] = False
    assert cli([*run, "--n", "20", "--out", str(tmp_path / "full")]) == 0
    receipt = ReleaseReceipt.model_validate_json((tmp_path / "full" / "receipt.json").read_text())
    assert receipt.passed is True and receipt.starts_fresh is False and not receipt.released
    events = [(e["event"], len(e.get("init_states", []))) for e in library.release_exposure(lib)]
    assert events == [("started", 5), ("started", 20), ("finished", 0)]
    # Another program for the same request meets the same exposure.
    renamed = saved(name="cheese-plate-two", program={**program().model_dump(), "notes": "v2"})
    assert library.exposed_release_starts(lib, renamed) == set(range(30, 50))
    # A library whose release starts were never examined releases on its first full attempt.
    fresh_lib = tmp_path / "fresh"
    library.save_job(fresh_lib, saved())
    assert (
        cli(
            [
                "run",
                "cheese-then-plate",
                "--library",
                str(fresh_lib),
                "--starts",
                "release",
                "--n",
                "20",
                "--out",
                str(tmp_path / "first"),
            ]
        )
        == 0
    )
    first = ReleaseReceipt.model_validate_json((tmp_path / "first" / "receipt.json").read_text())
    assert first.released and first.starts_fresh is True


def _replay_record(root: Path, episode: JobEpisode, hashes: list[str] | None = None) -> str:
    from fleetlab.engine import software
    from fleetlab.jobs.runner import episode_identity

    identity = episode_identity(
        episode.start, episode.episode_id, HoldSource().spec, software.current("none"), 6, 128, 100
    )
    executed = np.zeros((episode.control_steps, 7), np.float32)
    recorded = hashes or ["ab" * 8] * episode.control_steps + [episode.end_state_hash]
    tables.save_replay_record(
        root, episode.episode_id, executed, recorded, identity.model_dump_json()
    )
    return identity.model_dump_json()


def test_a_replay_record_must_hold_exactly_the_episode_s_hashes(run: Path, tmp_path: Path) -> None:
    # Finding 3: one bogus hash appended was "compared" and accepted.
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.jobs.replay import record_problems, rerender

    episode = tables.read_episodes(run)[0]
    identity = EpisodeIdentity.model_validate_json(_replay_record(run, episode))
    executed, hashes, _ = tables.load_replay_record(run, episode.episode_id)
    assert record_problems(episode.episode_id, executed, hashes, identity, episode) == []
    assert record_problems(episode.episode_id, executed, [*hashes, "ff" * 8], identity, episode)
    assert record_problems(episode.episode_id, executed[:-1], hashes[:-1], identity, episode)
    assert record_problems(
        episode.episode_id, executed, [*hashes[:-1], "ff" * 8], identity, episode
    )
    elsewhere = identity.model_copy(update={"start_state": 29})
    assert record_problems(episode.episode_id, executed, hashes, elsewhere, episode)
    assert record_problems("e000777", executed, hashes, identity, None)
    npz, _ = tables.truth_paths(run, episode.episode_id)  # tampered behind the writer's back
    np.savez_compressed(npz, executed=executed, state_hash=np.asarray([*hashes, "ff" * 8]))
    report = rerender(run, episode.episode_id, tmp_path / "out")
    assert not report["accepted"] and report["hashes_compared"] == 0
    assert report["recorded_hashes"] == len(hashes) + 1 and report["video"] is None
    assert not list((tmp_path / "out").glob("*.mp4"))


def test_a_rerender_report_names_the_original_of_its_own_camera(run: Path, tmp_path: Path) -> None:
    # A wrist re-render's report named the scene clip as its original, so the film refused it.
    from fleetlab.jobs.replay import rerender

    episode = tables.read_episodes(run)[0]
    _replay_record(run, episode)
    executed, hashes, _ = tables.load_replay_record(run, episode.episode_id)
    npz, _ = tables.truth_paths(run, episode.episode_id)  # refused before any simulation
    np.savez_compressed(npz, executed=executed, state_hash=np.asarray([*hashes, "ff" * 8]))
    originals = {}
    for camera in (Camera.scene, Camera.wrist):
        path = run / "log" / tables.video_relative(episode.episode_id, camera)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(camera.value.encode())
        originals[camera.value] = path
    for name, path in originals.items():
        report = rerender(run, episode.episode_id, tmp_path / name, camera=name)
        assert report["original_camera"] == name
        assert report["original_video"] == str(path)
        assert report["original_video_sha256"] == hashlib.sha256(name.encode()).hexdigest()
    other = rerender(run, episode.episode_id, tmp_path / "frontview", camera="frontview")
    assert other["original_camera"] is None
    assert other["original_video"] is None and other["original_video_sha256"] is None
    originals["wrist"].unlink()  # a recorded camera whose clip is missing has no original
    missing = rerender(run, episode.episode_id, tmp_path / "missing", camera="wrist")
    assert missing["original_camera"] == "wrist" and missing["original_video"] is None


def test_a_resume_keeps_only_this_job_s_runs_from_these_starts(run: Path) -> None:
    # Finding 4: resume skipped episodes by id, whatever program or start they ran.
    from fleetlab.jobs.runner import resume_problems

    for episode in tables.read_episodes(run):
        _replay_record(run, episode)
    starts = [start_identity(6, 3), start_identity(6, 4)]
    record = json.loads((run / "run.json").read_text())
    record["jobs"] = [
        {
            "job_id": "j0001",
            "program_digest": program().digest,
            "first_episode": 101,
            "starts": [s.model_dump(mode="json") for s in starts],
        }
    ]

    def problems(prog: JobProgram | None = None, at: Any = None) -> list[str]:
        return resume_problems(
            run,
            record,
            "j0001",
            prog or program(),
            at or starts,
            first_episode=101,
            record_video=False,
        )

    assert problems() == []
    changed = program().model_copy(update={"notes": "another program"})
    assert any("another program" in p for p in problems(changed))
    moved = [start_identity(6, 29), starts[1]]
    assert any("other starts" in p for p in problems(at=moved))
    record["jobs"] = []  # no saved entry: the episodes themselves must match
    assert any("another start" in p for p in problems(at=moved))
    assert any("another program" in p for p in problems(changed))
    tables.table_path(run, "object_steps", "e000102").unlink()
    assert any("no object_steps file" in p for p in problems())
