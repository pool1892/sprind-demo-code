"""Event detection beyond grasp and release: src/fleetlab/worlds/events.py.

Synthetic steps tables check the rules (what must and must not fire); a toy engine run
checks the plumbing into a finalized run (idempotent, still finalized, still exportable).
The toy cell's telemetry is not LIBERO's, so the toy run says nothing about the rules.
"""

from __future__ import annotations

import multiprocessing
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import pytest
from engine_helpers import config, jobs
from log_toy import toy_fleet

from fleetlab.contracts.log import EventKind, log_tree_problems
from fleetlab.engine.fleet import run
from fleetlab.log import export_runs, run_status
from fleetlab.log._files import MERGED_FILE
from fleetlab.log.leaks import audit_export
from fleetlab.worlds import events as ev
from fleetlab.worlds.events import DERIVED_KINDS, AddEventsError, add_events, episode_events

OPEN_M = 0.079
HOLD_M = 0.004
EMPTY_M = 0.001
MAX_STEPS = 280

type Array = npt.NDArray[np.float64]


def table(
    eef: Array, opening: Array, gripper: Array, target: Array | None = None
) -> dict[str, list[Any]]:
    """A steps table as pyarrow's to_pydict() gives it; no commanded action in the last row."""
    n = len(eef)
    target = eef + np.array([0.0, 0.0, -0.04]) if target is None else target
    commanded: list[Any] = [
        [float(x) for x in (*target[t], 0.0, 0.0, 0.0, gripper[t])] for t in range(n - 1)
    ]
    return {
        "step": list(range(n)),
        "time_s": [0.05 * t for t in range(n)],
        "joint_pos": [[0.0] * 7 for _ in range(n)],
        "joint_vel": [[0.0] * 7 for _ in range(n)],
        "gripper_qpos": [[o / 2, -o / 2] for o in opening],
        "eef_pos": [list(map(float, p)) for p in eef],
        "eef_quat": [[1.0, 0.0, 0.0, 0.0] for _ in range(n)],
        "commanded_action": [*commanded, None],
        "chunk_index": [t // 30 for t in range(n - 1)] + [None],
        "chunk_step": [t % 30 for t in range(n - 1)] + [None],
        "frame_index": [None] * n,
    }


def drifting(n: int, speed: float = 0.005) -> Array:
    """An arm that keeps moving along x: no stall, no abrupt stop."""
    eef = np.zeros((n, 3))
    eef[:, 0] = speed * np.arange(n)
    eef[:, 2] = 1.0
    return eef


def closing(n: int, close_at: int, hold: float, open_at: int | None = None) -> Array:
    """Finger opening: open, then closing at 7.5 mm per step onto `hold` (the command at row
    t shapes row t + 1), open again after `open_at`."""
    opening = np.full(n, OPEN_M)
    for t in range(close_at + 1, n):
        opening[t] = max(hold, OPEN_M - 0.0075 * (t - close_at))
    if open_at is not None:
        opening[open_at + 1 :] = OPEN_M
    return opening


def gripper(n: int, *closed: tuple[int, int]) -> Array:
    """Commanded gripper: -1 open, +1 closed over the given [first, last] rows."""
    g = np.full(n, -1.0)
    for first, last in closed:
        g[first : last + 1] = 1.0
    return g


def kinds(found: list[tuple[int, EventKind, str]]) -> list[EventKind]:
    return [k for _, k, _ in found]


def of(found: list[tuple[int, EventKind, str]], kind: EventKind) -> list[tuple[int, str]]:
    return [(s, d) for s, k, d in found if k is kind]


# ---- drop ------------------------------------------------------------------------------------


def lifted(n: int, grasp_at: int) -> Array:
    eef = drifting(n)
    eef[grasp_at:, 2] += 0.004 * np.arange(n - grasp_at)  # the arm lifts after the grasp
    return eef


def test_an_object_that_slips_out_of_the_closed_gripper_is_a_drop() -> None:
    n = 100
    opening = closing(n, 25, HOLD_M)
    slip = 60
    opening[slip:80] = np.maximum(EMPTY_M, HOLD_M - 0.0003 * np.arange(1, 21))
    opening[80:] = OPEN_M
    found = episode_events(table(lifted(n, 30), opening, gripper(n, (25, 79))), "failure", 280)
    drops = of(found, EventKind.drop)
    first_below = next(t for t in range(slip, n) if opening[t] < ev.HOLD_MIN_M)
    assert [s for s, _ in drops] == [first_below]
    assert "slipped out of the closed gripper" in drops[0][1]
    assert "above where it was grasped" in drops[0][1]


def test_a_steady_hold_and_a_grasp_on_nothing_are_no_drop() -> None:
    n = 100
    held = closing(n, 25, HOLD_M, open_at=79)
    missed = closing(n, 25, EMPTY_M, open_at=79)
    for opening in (held, missed):
        found = episode_events(table(lifted(n, 30), opening, gripper(n, (25, 79))), "failure", 280)
        assert EventKind.drop not in kinds(found)


def test_a_late_close_on_nothing_is_a_miss_not_a_drop() -> None:
    """Under a command delay the fingers stay open for a while after the close command,
    steady near their widest; that is no hold, and closing on nothing then is no drop."""
    n = 100
    late = closing(n, 37, EMPTY_M, open_at=79)  # commanded closed at row 25, moving at 37
    found = episode_events(table(lifted(n, 30), late, gripper(n, (25, 79))), "success", 280)
    assert EventKind.drop not in kinds(found)
    assert "closed on nothing" in found[-1][2]


def test_a_slip_after_the_open_command_is_no_drop() -> None:
    n = 100
    opening = closing(n, 25, HOLD_M)
    opening[61:] = OPEN_M  # opened by command at row 60, so the fingers part
    found = episode_events(table(lifted(n, 30), opening, gripper(n, (25, 60))), "failure", 280)
    assert EventKind.drop not in kinds(found)


# ---- stall -----------------------------------------------------------------------------------


def test_an_arm_that_stays_put_for_two_seconds_is_one_stall() -> None:
    n = 140
    eef = drifting(n)
    eef[30:90] = eef[30]
    eef[90:, 0] = eef[30, 0] + 0.005 * np.arange(1, n - 89)
    target = eef + np.array([0.0, 0.0, -0.04])
    found = episode_events(table(eef, np.full(n, OPEN_M), gripper(n), target), "failure", 280)
    stalls = of(found, EventKind.stall)
    assert len(stalls) == 1
    step, detail = stalls[0]
    assert 27 <= step <= 30  # the window reaches back while the arm is within 1.5 cm
    assert re.search(r"for 3\.\d s", detail) and "stayed within 3 cm" in detail


def test_a_stall_says_when_the_commanded_target_moved_on() -> None:
    n = 140
    eef = drifting(n)
    eef[30:90] = eef[30]
    eef[90:, 0] = eef[30, 0] + 0.005 * np.arange(1, n - 89)
    target = drifting(n) + np.array([0.05, 0.0, 0.0])  # it keeps going
    found = episode_events(table(eef, np.full(n, OPEN_M), gripper(n), target), "failure", 280)
    (stall,) = of(found, EventKind.stall)
    assert "the commanded target moved up to" in stall[1]


def test_merged_stalls_describe_local_windows_when_the_arm_keeps_drifting() -> None:
    n = 300
    eef = drifting(n, speed=0.0003)
    (stall,) = of(
        episode_events(table(eef, np.full(n, OPEN_M), gripper(n)), "failure", 280), EventKind.stall
    )
    step, detail = stall
    assert np.linalg.norm(eef[-1] - eef[step]) > 0.08
    assert "2.0 s windows" in detail
    assert "overlapping" not in detail
    assert f"within 1.5 cm of each window's start for {(n - 1 - step) * 0.05:.1f} s" in detail
    assert "no end-effector motion beyond" not in detail


def test_slow_motion_short_pauses_and_the_first_steps_are_no_stall() -> None:
    n = 140
    slow = drifting(n, speed=0.001)  # 4 cm in two seconds
    pause = drifting(n)
    pause[40:65] = pause[40]  # 25 steps
    pause[65:, 0] = pause[40, 0] + 0.005 * np.arange(1, n - 64)
    start = drifting(n, speed=0.02)
    start[:43] = start[0]  # still at the start, then away fast
    start[43:, 0] = 0.02 * np.arange(1, n - 42)
    for eef in (slow, pause, start):
        found = episode_events(table(eef, np.full(n, OPEN_M), gripper(n)), "failure", 280)
        assert EventKind.stall not in kinds(found)


# ---- collision -------------------------------------------------------------------------------


def moving_then(n: int, stop_at: int, after: Array) -> Array:
    """Along x at 10 mm per step until `stop_at`, then the given per-step motion."""
    eef = np.zeros((n, 3))
    eef[:, 2] = 1.0
    for t in range(1, n):
        speed = 0.010 if t <= stop_at else after[min(t - stop_at - 1, len(after) - 1)]
        eef[t, 0] = eef[t - 1, 0] + speed
    return eef


def test_an_abrupt_stop_while_the_command_moves_on_is_a_collision() -> None:
    n = 80
    eef = moving_then(n, 40, np.array([0.0] * 8 + [0.005]))
    target = np.zeros((n, 3))
    target[:, 0] = 0.05 + 0.010 * np.arange(n)  # leads by 5 cm and keeps going
    target[:, 2] = 1.0
    found = episode_events(table(eef, np.full(n, OPEN_M), gripper(n), target), "failure", 280)
    collisions = of(found, EventKind.collision)
    assert [s for s, _ in collisions] == [40]
    assert "stopped abruptly from 0.20 m/s" in collisions[0][1]
    assert "commanded target moved on 3.0 cm" in collisions[0][1]


def test_a_commanded_stop_a_smooth_stop_and_a_rebound_are_no_collision() -> None:
    n = 80
    held_target = np.zeros((n, 3))
    held_target[:, 0] = np.minimum(0.05 + 0.010 * np.arange(n), 0.41)  # stops at the arm
    held_target[:, 2] = 1.0
    going = np.zeros((n, 3))
    going[:, 0] = 0.05 + 0.010 * np.arange(n)
    going[:, 2] = 1.0
    abrupt = moving_then(n, 40, np.array([0.0] * 8 + [0.005]))
    smooth = moving_then(n, 40, np.array([0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.005]))
    rebound = moving_then(n, 40, np.array([0.0, -0.008, -0.008, 0.005]))
    for eef, target in ((abrupt, held_target), (smooth, going), (rebound, going)):
        found = episode_events(table(eef, np.full(n, OPEN_M), gripper(n), target), "failure", 280)
        assert EventKind.collision not in kinds(found)


# ---- near miss -------------------------------------------------------------------------------


def two_grasps(n: int, first_hold: float) -> tuple[Array, Array]:
    """A first grasp (rows 20 to 34) onto `first_hold`, then one onto a held object."""
    opening = closing(n, 20, first_hold, open_at=34)
    second = closing(n, 50, HOLD_M, open_at=n - 5)
    opening[51:] = second[51:]
    return opening, gripper(n, (20, 34), (50, n - 5))


def test_a_success_after_a_grasp_on_nothing_is_a_near_miss() -> None:
    n = 110
    opening, g = two_grasps(n, EMPTY_M)
    found = episode_events(table(drifting(n), opening, g), "success", MAX_STEPS)
    (near,) = of(found, EventKind.near_miss)
    assert near[0] == n - 1
    assert "only after a grasp closed on nothing at step" in near[1]


def test_a_success_late_in_the_step_limit_is_a_near_miss() -> None:
    n = 241  # ends at step 240 of 280
    opening, g = two_grasps(n, HOLD_M)
    found = episode_events(table(drifting(n), opening, g), "success", MAX_STEPS)
    (near,) = of(found, EventKind.near_miss)
    assert near == (n - 1, "succeeded late, with 2.0 s left")


def test_a_clean_success_and_a_failure_are_no_near_miss() -> None:
    n = 110
    clean, g = two_grasps(n, HOLD_M)
    missed, g_missed = two_grasps(n, EMPTY_M)
    assert not of(episode_events(table(drifting(n), clean, g), "success", 280), EventKind.near_miss)
    for outcome in ("failure", "handed_over"):
        found = episode_events(table(drifting(n), missed, g_missed), outcome, 280)
        assert not of(found, EventKind.near_miss)


# ---- the function as a whole -----------------------------------------------------------------


def busy_episode() -> dict[str, list[Any]]:
    """One table that fires all four kinds: a slip at row 43, a stall from row 99, an abrupt
    stop at row 169 while the target goes on, a success late in the step limit."""
    n = 240
    step = np.full(n, 0.007)  # x motion into each row
    step[0] = 0.0
    step[100:150] = 0.0  # the stall
    step[150:170] = 0.010
    step[170:178] = 0.0  # the stop
    eef = np.zeros((n, 3))
    eef[:, 0] = np.cumsum(step)
    eef[:, 2] = 1.0
    target = eef + np.array([0.0, 0.0, -0.04])
    target[169:180, 0] = eef[169, 0] + 0.010 * np.arange(11)
    opening = closing(n, 20, HOLD_M)
    opening[40:60] = np.maximum(EMPTY_M, HOLD_M - 0.0003 * np.arange(1, 21))  # a slip
    later = closing(n, 70, HOLD_M, open_at=n - 5)
    opening[60:71] = OPEN_M
    opening[71:] = later[71:]
    return table(eef, opening, gripper(n, (20, 59), (70, n - 5)), target)


def test_events_are_sorted_and_carry_only_the_derived_kinds() -> None:
    found = episode_events(busy_episode(), "success", MAX_STEPS)
    assert set(kinds(found)) == DERIVED_KINDS
    assert found == sorted(found, key=lambda e: (e[0], e[1].value, e[2]))
    assert episode_events({"step": []}, "failure", 280) == []


def test_the_same_table_gives_the_same_events_in_another_process() -> None:
    steps = busy_episode()
    here = episode_events(steps, "success", MAX_STEPS)
    assert episode_events({k: list(v) for k, v in steps.items()}, "success", MAX_STEPS) == here
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn) as pool:
        there = pool.submit(episode_events, steps, "success", MAX_STEPS).result()
    assert there == here


def test_details_speak_the_fleet_s_language(tmp_path: Path) -> None:
    found = episode_events(busy_episode(), "success", MAX_STEPS)
    report = audit_export(
        tmp_path,
        [],
        texts={str(i): detail for i, (_, _, detail) in enumerate(found)},
    )
    assert report.clean, report.summary()


def test_abutting_stall_windows_make_no_overlap_claim() -> None:
    n = 120
    eef = drifting(n)
    eef[25:66] = eef[25]
    eef[66:107] = eef[25] + np.array([0.05, 0.0, 0.0])
    eef[107:, 0] = eef[106, 0] + 0.005 * np.arange(1, n - 106)
    stalls = of(
        episode_events(table(eef, np.full(n, OPEN_M), gripper(n)), "failure", 280), EventKind.stall
    )
    assert len(stalls) == 1
    assert "2.0 s windows" in stalls[0][1] and "overlapping" not in stalls[0][1]


# ---- into a run ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finalized toy engine run, as tests/test_engine_fleet.py makes them."""
    base = tmp_path_factory.mktemp("events")
    run(config(base / "run", workers=2), jobs(6))
    assert run_status(base / "run").finalized
    return base / "run"


def copy_run(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(source, target)
    return target


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def event_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(root / "log" / "events.parquet").to_pylist()
    return rows


def test_add_events_is_idempotent_and_the_run_stays_exportable(
    toy_run: Path, tmp_path: Path
) -> None:
    root = copy_run(toy_run, tmp_path)
    engine = event_rows(root)
    episode = engine[0]["episode_id"]
    contact = (3, EventKind.collision, "the gripper touched the table edge")
    counts = add_events(root, {episode: [contact]})
    assert set(counts) == {k.value for k in DERIVED_KINDS} and counts["collision"] >= 1
    rows = event_rows(root)
    assert [r for r in rows if r["kind"] not in {k.value for k in DERIVED_KINDS}] == engine
    assert {"episode_id": episode, "step": 3, "kind": "collision", "detail": contact[2]} in rows
    once = snapshot(root)
    assert add_events(root, {episode: [contact]}) == counts
    assert snapshot(root) == once
    assert run_status(root).finalized
    assert log_tree_problems(root) == []
    report = export_runs([root], tmp_path / "export", fleet=toy_fleet())
    assert report is not None
    exported = pq.read_table(tmp_path / "export" / "log" / "events.parquet").to_pylist()
    assert any(r["kind"] == "collision" and r["detail"] == contact[2] for r in exported)


def test_a_later_call_replaces_the_events_of_an_earlier_one(toy_run: Path, tmp_path: Path) -> None:
    root = copy_run(toy_run, tmp_path / "a")
    plain = copy_run(toy_run, tmp_path / "b")
    episode = event_rows(root)[0]["episode_id"]
    add_events(root, {episode: [(2, EventKind.stall, "held still by an operator")]})
    add_events(root)
    add_events(plain)
    assert snapshot(root / "log") == snapshot(plain / "log")
    assert not any(r["detail"] == "held still by an operator" for r in event_rows(root))


def test_add_events_refuses_what_does_not_fit_and_writes_nothing(
    toy_run: Path, tmp_path: Path
) -> None:
    root = copy_run(toy_run, tmp_path)
    before = snapshot(root)
    episode = event_rows(root)[0]["episode_id"]
    with pytest.raises(AddEventsError, match="not in the run"):
        add_events(root, {"e999999": [(1, EventKind.collision, "x")]})
    with pytest.raises(AddEventsError, match="not one this module adds"):
        add_events(root, {episode: [(1, EventKind.grasp, "x")]})
    with pytest.raises(AddEventsError, match="outside"):
        add_events(root, {episode: [(10_000, EventKind.collision, "x")]})
    assert snapshot(root) == before
    (root / MERGED_FILE).unlink()
    with pytest.raises(AddEventsError, match="not finalized"):
        add_events(root)


# ---- contacts --------------------------------------------------------------------------------


def scene(*steps: set[tuple[str, str]], moved: float = 0.0) -> ev.SceneContacts:
    """Contacts per step; the ramekin moves `moved` metres over steps 3 to 6."""
    n = len(steps)
    ramekin = np.zeros((n, 3))
    ramekin[3:7, 0] = np.linspace(0, moved, 4)
    ramekin[7:, 0] = moved
    return ev.SceneContacts(
        contacts=[frozenset(s) for s in steps],
        positions={"ramekin_1": ramekin, "bowl_1": np.zeros((n, 3))},
        target="bowl_1",
        movable=frozenset({"ramekin_1", "bowl_1"}),
    )


def test_contact_onsets_of_the_arm_and_the_gripper_body_are_collisions() -> None:
    arm = ("arm", "wooden_cabinet_1")
    palm_on_table = ("gripper", "table")
    palm_on_target = ("gripper", "bowl_1")
    steps = [set(), {arm}, {arm}, set(), {arm, palm_on_table, palm_on_target}]
    steps += [set()] * 6 + [{arm}]
    found = ev.scene_contact_events(scene(*steps))
    assert found == [
        (1, EventKind.collision, "the arm touched the wooden cabinet"),
        (4, EventKind.collision, "the gripper body struck the table"),
        (11, EventKind.collision, "the arm touched the wooden cabinet"),
    ]


def test_fingers_count_only_when_they_push_another_object() -> None:
    touch = ("finger", "ramekin_1")
    held = ("finger", "bowl_1")
    steps = [set(), set(), set(), {touch, held}, {touch, held}, {held}, set(), set(), set()]
    assert ev.scene_contact_events(scene(*steps, moved=0.004)) == []
    pushed = ev.scene_contact_events(scene(*steps, moved=0.03))
    assert pushed == [(3, EventKind.collision, "the gripper pushed the ramekin 3 cm")]
    assert all(
        robot_part == part
        for robot_part, part in [
            (ev.robot_part("robot0_link4"), "arm"),
            (ev.robot_part("robot0_right_hand"), "wrist"),
            (ev.robot_part("gripper0_right_right_gripper"), "gripper"),
            (ev.robot_part("gripper0_right_finger_joint1_tip"), "finger"),
            (ev.robot_part("gripper0_right_leftfinger"), "finger"),
        ]
    )
    assert ev.robot_part("robot0_base") is None and ev.robot_part("table") is None


def record_one_episode(run_dir: Path) -> str:
    """One LIBERO episode with a scripted in-worker policy and a worn gripper (a cell hook),
    written in the layout `contact_pass` reads: identity, episode row, chain."""
    import pyarrow as pa

    from fleetlab.contracts.faults import FaultSpec, Schedule, Target
    from fleetlab.contracts.rollout import Job
    from fleetlab.contracts.vocabulary import Component
    from fleetlab.engine import manifests, software
    from fleetlab.engine.episode import EpisodeContext, MemoryStore, run_job
    from fleetlab.engine.scripted import LIFT_SPEC, LiftPolicy
    from fleetlab.faults import REGISTRY, load_all
    from fleetlab.sim.libero_sim import LiberoCheck, LiberoSimEnv

    load_all()
    wear = FaultSpec(
        fault_id="f01",
        kind="gripper_wear",
        level=0.5,
        target=Target(scope="r01", component=Component.gripper),
        schedule=Schedule(onset_day=0),
    )
    identity = manifests.identity(
        world_seed=5, episode_number=1, robot=1, site=1, day=0, episode_index=0,
        suite="libero_spatial", task_index=0, start_state=0, policy=LIFT_SPEC, batch_size=1,
        software=software.current("none"), image_size=128, max_steps=40, faults=[wear],
    )  # fmt: skip
    env = LiberoSimEnv("libero_spatial", 0, 128)
    ctx = EpisodeContext(
        fault_types=REGISTRY.get,
        make_check=LiberoCheck,
        teleoperator=LiftPolicy,
        store=MemoryStore(),
    )
    record = run_job(Job(identity=identity), env, ctx)
    env.close()
    episode = identity.episode_id
    (run_dir / "truth" / "chain").mkdir(parents=True)
    (run_dir / "log").mkdir()
    ids = pa.table({"episode_id": [episode], "identity_json": [identity.model_dump_json()]})
    pq.write_table(ids, run_dir / "truth" / "identities.parquet")
    pq.write_table(pa.table({"episode_id": [episode]}), run_dir / "log" / "episodes.parquet")
    np.savez(
        run_dir / "truth" / "chain" / f"{episode}.npz",
        state_hash=np.array(record.chain.state),
        executed_action=np.array(record.chain.executed),
    )
    return episode


def replay_recorded(run_dir: Path) -> tuple[str, Any, Any]:
    episode = record_one_episode(run_dir)
    good = ev.contact_pass(run_dir, workers=1)
    chain = run_dir / "truth" / "chain" / f"{episode}.npz"
    data = dict(np.load(chain))
    data["state_hash"][12] = "0" * 16
    np.savez(chain, **data)
    bad = ev.contact_pass(run_dir, workers=1)
    return episode, good, bad


@pytest.mark.sim
def test_a_replay_matches_the_chain_at_every_step(tmp_path: Path) -> None:
    from sim_helpers import in_fresh_process

    episode, good, bad = in_fresh_process(replay_recorded, tmp_path / "run")
    assert good.mismatched == {} and episode in good.events
    assert all(k is EventKind.collision for _, k, _ in good.events[episode])
    assert bad.mismatched == {episode: 12} and bad.events == {}
