"""The stage-3 calibration plan and its analysis on toy runs (fleetlab.dose_response_stage3).

The plan pairs every faulted episode with a healthy twin on its own suite,
never repeats an episode id between its runs, runs the joints' two faults in
the same episodes and the false-positive detector on top of its camera; the
interaction is the joint effect minus the sum of the single effects; the
analysis of toy runs gives the roles and verdicts of the rules and merges
into a copy of curves.json without duplicating what it already holds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fault_toys import POLICY, SOFTWARE

from fleetlab import dose_response as dr
from fleetlab import dose_response_stage3 as s3
from fleetlab.contracts.rollout import Job, RunLayout

PLAN = s3.plan()
JOBS = {run.name: s3.jobs_for(run, POLICY, SOFTWARE) for run in PLAN}


def test_the_libero_object_runs_are_made_only_when_named() -> None:
    optional = [run.name for run in PLAN if run.optional]
    assert optional and all(name.startswith("s3-object-") for name in optional)
    assert all(run.suite == "libero_object" for run in PLAN if run.optional)


def test_the_plan_never_repeats_an_episode_between_its_runs() -> None:
    seen: dict[str, str] = {}
    for run in PLAN:
        if run.films:
            continue  # the films are episodes of the other runs, filmed
        for job in JOBS[run.name]:
            eid = job.identity.episode_id
            assert eid not in seen, (eid, run.name, seen.get(eid))
            seen[eid] = run.name
    films = JOBS["s3-films"]
    assert all(j.record_video for j in films) and len(films) == 2 * (1 + len(s3.FILMED))


def test_every_faulted_episode_has_a_healthy_twin_on_its_suite() -> None:
    twins: dict[tuple[str, int], Any] = {}
    for name in ("s3-healthy", "s3-healthy-1000", "s3-object-healthy"):
        for job in JOBS[name]:
            twins[(job.identity.suite, job.identity.episode_index)] = job.identity
    for run in PLAN:
        if run.kind is None and run.joint_name is None:
            continue
        for job in JOBS[run.name]:
            ident = job.identity
            twin = twins[(ident.suite, ident.episode_index)]
            drop = {"episode_id", "faults", "record_video"}
            assert ident.model_dump(exclude=drop) == twin.model_dump(exclude=drop), run.name
    blocks = {dr.split_episode_id(j.identity.episode_id)[0] for j in JOBS["s3-object-healthy"]}
    assert blocks == {500}


def test_joints_and_companions_put_their_faults_in_the_same_episodes() -> None:
    joint = JOBS["s3-joint-wear-heavier"]
    assert len(joint) == 300
    assert {(f.fault_id, f.kind, f.level) for f in joint[0].identity.faults} == {
        ("f01", "gripper_wear", 0.5),
        ("f02", "object_batch_variant", 4.0),
    }
    assert {dr.split_episode_id(j.identity.episode_id)[0] for j in joint} == {dr.JOINT_BLOCK}
    detector = JOBS["s3-success_false_positive"][0].identity.faults
    assert [(f.fault_id, f.kind, f.level) for f in detector] == [
        ("f01", "success_false_positive", 0.25),
        ("f02", "camera_pose_offset", 6.0),
    ]
    extension = JOBS["s3-light_color_shift-1000"]
    assert [j.identity.episode_index for j in extension] == list(range(300, 1000))
    assert {j.identity.faults[0].level for j in extension} == {1.0}
    assert not extension[0].identity.faults[0].params  # warmer, the level a world would use


def outcomes(values: list[int], block: int = 0) -> dict[int, dr.Outcome]:
    return {
        i: dr.Outcome(dr.episode_id(block, i), bool(v), bool(v), bool(v), 100)
        for i, v in enumerate(values)
    }


def test_the_interaction_is_the_joint_effect_minus_the_sum_of_the_singles() -> None:
    n = 100
    healthy = [1] * n
    a = [0] * 5 + [1] * 95  # -5 points alone
    b = [1] * 5 + [0] * 5 + [1] * 90  # -5 points alone
    joint = [0] * 40 + [1] * 60  # -40 together
    result = s3.interaction(outcomes(healthy), outcomes(a), outcomes(b), outcomes(joint))
    assert result["n"] == n and (result["a"], result["b"], result["joint"]) == (95, 95, 60)
    assert result["effect_a"][0] == -0.05 and result["effect_joint"][0] == -0.40
    assert result["interaction"][0] == pytest.approx(-0.30)
    low, high = result["interaction"][1:]
    assert low < -0.30 < high < 0 and result["role"] == "interaction"
    additive = s3.interaction(
        outcomes(healthy), outcomes(a), outcomes(b), outcomes([0] * 10 + [1] * 90)
    )
    assert additive["interaction"][0] == 0.0 and additive["role"] == "additive"


# ---------------------------------------------------------------------------- toy runs


def toy_outcome(job: Job) -> tuple[bool, bool]:
    """(true, logged): healthy 95%; wear and heavier 90% each, 50% together; a camera at 6
    degrees 60%, on which the detector logs every other failure as a success."""
    ident = job.identity
    digest = hashlib.sha256(f"{ident.suite}{ident.episode_index}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    kinds = {f.kind for f in ident.faults}
    rate = 0.95
    if {"gripper_wear", "object_batch_variant"} <= kinds:
        rate = 0.5
    elif kinds & {"gripper_wear", "object_batch_variant"}:
        rate = 0.9
    if "camera_pose_offset" in kinds:
        rate = 0.6
    true = u < rate
    logged = true
    if "success_false_positive" in kinds and not true:
        logged = ident.episode_index % 2 == 0
    return true, logged


def toy_run(jobs: list[Job], run_dir: Path, outcome: Any = None) -> None:
    layout = RunLayout(run_dir)
    rows, verdicts = [], []
    for job in jobs:
        true, logged = (outcome or toy_outcome)(job)
        eid = job.identity.episode_id
        rows.append({"episode_id": eid, "outcome": "success" if logged else "failure", "steps": 9})
        verdicts.append(
            {
                "episode_id": eid,
                "true_success": true,
                "true_reason": "",
                "libero_first_success_step": 5 if true else None,
            }
        )
    for name, table in (("episodes", rows), ("verdicts", verdicts)):
        path = layout.table(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(table), path)
    layout.run_record.write_text(json.dumps({"chain_ok": True}))


def test_the_analysis_of_toy_runs_and_its_merge_into_curves(tmp_path: Path) -> None:
    for name in (
        "s3-healthy",
        "s3-calibration_update",
        "s3-success_false_positive",
        "s3-gripper_wear-300",
        "s3-object_batch_variant-300",
        "s3-joint-wear-heavier",
        "s3-object-healthy",
        "s3-object-gripper_wear",
        "s3-object-object_batch_variant",
        "s3-object-joint-wear-heavier",
    ):
        toy_run(JOBS[name], s3.run_dir(name, tmp_path))
    assert s3.complete(s3.run_dir("s3-healthy", tmp_path), 300)
    assert not s3.complete(s3.run_dir("s3-healthy", tmp_path), 301)
    result = s3.analyse(tmp_path)
    assert result["healthy"]["n"] == 300 and result["healthy"]["note"] == "stage 3's own only"
    inert = result["types"]["calibration_update"]
    assert inert["verdict"] == "inert" and inert["points"][0]["role"] == "inert"
    detector = result["types"]["success_false_positive"]
    assert detector["verdict"] == "kept"
    for point in detector["points"]:  # logged, against its own episodes' true outcome
        assert point["companions"] == [["camera_pose_offset", "6 deg scene pan"]]
        assert point["successes"] == point["logged_successes"] > point["healthy_successes"]
        assert point["broken"] == 0 and point["rescued"] > 0 and point["role"] == "effect"
    joints = {(j["name"], j["suite"]): j for j in result["joints"]}
    spatial = joints[("wear+heavier", "libero_spatial")]
    assert spatial["n"] == 300 and spatial["interaction_role"] == "interaction"
    assert spatial["role"] == "effect"  # the joint against healthy, as the worlds write it
    assert spatial["interaction"][0] < -0.2
    assert ("wear+heavier", "libero_object") in joints
    assert [p["label"] for p in result["object"]["gripper_wear"]] == [
        "0.25 wear",
        "0.5 wear",
        "0.75 wear",
        "1 wear",
    ]
    tables = s3.tables(result)
    assert "#### calibration_update: inert" in tables and "#### Interactions" in tables
    assert "| Logged successes | Physically placed (same attempts) |" in tables

    curves = tmp_path / "curves.json"
    shutil.copy(Path(dr.__file__).parent / "faults" / "curves.json", curves)
    before = json.loads(curves.read_text())
    s3.write_curves(result, curves)
    noted = json.loads(curves.read_text())
    noted["types"]["calibration_update"]["note"] = "written by hand"
    curves.write_text(json.dumps(noted))
    s3.write_curves(result, curves)  # twice: nothing is duplicated, a note stays
    after = json.loads(curves.read_text())
    assert after["types"]["calibration_update"]["note"] == "written by hand"
    assert after["types"]["camera_pose_offset"] == before["types"]["camera_pose_offset"]
    assert after["healthy"] == before["healthy"]
    assert after["types"]["calibration_update"]["verdicts"][dr.SUITE] == "inert"
    sup = after["supplement"]
    assert sup["healthy"]["libero_object"]["n"] == 300
    keys = [(j["name"], j["suite"]) for j in sup["joints"]]
    assert len(keys) == len(set(keys))  # one entry per pair and suite
    assert {("wear+heavier", "libero_spatial"), ("wear+heavier", "libero_object")} <= set(keys)
    assert len([p for p in sup["points"] if p["kind"] == "gripper_wear"]) == 4


# ---------------------------------------------------------------------------- the runner


def runner_outcome(job: Job) -> tuple[bool, bool]:
    """Healthy 95%; lamps of any colour harmless, except cooler ones, which fail every time."""
    ident = job.identity
    digest = hashlib.sha256(f"{ident.episode_index}".encode()).digest()
    true = int.from_bytes(digest[:8], "big") / 2**64 < 0.95
    if any(f.kind == "light_color_shift" and f.params.get("cooler") for f in ident.faults):
        true = False
    return true, true


class StubFleet:
    """The engine's `fleet` module as the runner uses it: writes toy runs, records the runs."""

    @dataclass
    class RunConfig:
        out: str
        workers: int
        episodes_per_robot_day: int
        mem_floor_gib: float = 8.0
        worker_gib: float = 2.1
        server_gib: float = 4.0

    def __init__(self, available: list[float] | None = None, fail_first: int = 0) -> None:
        self.ran: list[str] = []
        self.workers: list[int] = []
        self.available = available or []
        self.fail_first = fail_first

    def mem_available_gib(self) -> float:
        return self.available.pop(0) if len(self.available) > 1 else (self.available or [64.0])[0]

    def run(self, config: Any, jobs: list[Job]) -> Any:
        self.workers.append(config.workers)
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("GPU memory 22.0 GiB, memory available 5.0 GiB")
        path = Path(config.out)
        toy_run(jobs, path, outcome=runner_outcome)
        self.ran.append(path.name)
        # A resumed engine run counts only what it ran in this call: report a fraction.
        return SimpleNamespace(record=SimpleNamespace(chain_ok=True, extra={"successes": 3}))


def test_the_runner_skips_what_is_done_and_what_its_conditions_rule_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubFleet()
    monkeypatch.setattr(dr, "_engine", lambda module: stub)
    held = {"compared": 300, "identical": 300, "differ": [], "missing": []}
    monkeypatch.setattr(s3, "healthy_identity", lambda root: held)
    runs = (
        s3.PlanRun("s3-healthy", None, 30),
        s3.PlanRun("s3-healthy-1000", None, 100, first=30, when="healthy_differs"),
        s3.PlanRun("s3-light_color_shift", "light_color_shift", 30),
        s3.PlanRun(
            "s3-light_color_shift-1000",
            "light_color_shift",
            100,
            (1,),
            first=30,
            when="decoy_holds",
        ),
        s3.PlanRun("s3-object-healthy", None, 30, suite="libero_object", optional=True),
    )
    setup = s3.EngineSetup(policy=POLICY, software=SOFTWARE, config={}, min_healthy=0.5)
    assert s3.run(root=tmp_path, setup=setup, runs=runs) == 0
    # the stage-2 healthy run holds, and cooler lamps lose: no extension; nothing optional
    assert stub.ran == ["s3-healthy", "s3-light_color_shift"]
    assert json.loads((tmp_path / "s3-healthy-identity.json").read_text()) == held
    assert s3.run(root=tmp_path, setup=setup, runs=runs) == 0
    assert stub.ran == ["s3-healthy", "s3-light_color_shift"]  # all done: nothing again
    progress = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [p["state"] for p in progress if p["run"] == "s3-healthy"] == ["ran", "complete before"]
    held["identical"] = 299
    held["differ"] = ["e000007"]
    assert s3.run(root=tmp_path, setup=setup, runs=runs) == 0
    assert stub.ran[-1] == "s3-healthy-1000"  # the chains differ: its own 700 twins
    assert s3.run(["s3-object-healthy"], root=tmp_path, setup=setup, runs=runs) == 0
    assert stub.ran[-1] == "s3-object-healthy"  # an optional run when it is named
    strict = s3.EngineSetup(policy=POLICY, software=SOFTWARE, config={}, min_healthy=0.99)
    assert s3.run(root=tmp_path / "strict", setup=strict, runs=runs) == 1  # the floor stops it
    assert stub.ran[-1] == "s3-healthy"
    assert s3.run(root=tmp_path / "strict", setup=strict, runs=runs) == 1
    assert s3.run(root=tmp_path / "strict", setup=setup, runs=runs) == 1  # rejection persists


def test_the_decoy_extension_runs_when_no_level_of_the_type_loses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubFleet()
    monkeypatch.setattr(dr, "_engine", lambda module: stub)
    harmless = s3.PlanRun("s3-light_color_shift", "light_color_shift", 30, points=(0, 1))
    extension = s3.PlanRun(
        "s3-light_color_shift-1000", "light_color_shift", 100, (1,), first=30, when="decoy_holds"
    )
    runs = (s3.PlanRun("s3-healthy", None, 30), harmless, extension)
    setup = s3.EngineSetup(policy=POLICY, software=SOFTWARE, config={}, min_healthy=0.5)
    assert s3.run(root=tmp_path, setup=setup, runs=runs) == 0
    assert stub.ran == ["s3-healthy", "s3-light_color_shift", "s3-light_color_shift-1000"]


def test_the_runner_waits_out_a_shortage_of_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 18 GiB: only 2 workers fit (wait); 23 GiB: 5 fit, the run starts and is stopped for
    # memory once (wait); then 40 GiB: 8 fit and it resumes.
    stub = StubFleet(available=[18.0, 23.0, 40.0], fail_first=1)
    waits: list[float] = []
    monkeypatch.setattr(dr, "_engine", lambda module: stub)
    monkeypatch.setattr(s3, "_sleep", waits.append)
    runs = (s3.PlanRun("s3-healthy", None, 30),)
    setup = s3.EngineSetup(policy=POLICY, software=SOFTWARE, config={}, min_healthy=0.5)
    assert s3.run(root=tmp_path, workers=8, setup=setup, runs=runs) == 0
    assert stub.workers == [5, 8] and len(waits) == 2 and stub.ran == ["s3-healthy"]
    last = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert (last["episodes"], last["successes"]) == (30, 29)  # the directory's, not the call's
    states = [
        json.loads(line)["state"] for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert states[0].startswith("waiting for memory: only 2 workers fit")
    assert states[1].startswith("waiting for memory: GPU memory") and states[2] == "ran"
    other = StubFleet(fail_first=1)
    monkeypatch.setattr(dr, "_engine", lambda module: other)
    monkeypatch.setattr(other, "run", lambda config, jobs: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(ValueError):
        s3.run(root=tmp_path / "other", workers=8, setup=setup, runs=runs)


def test_the_films_bundle_puts_each_filmed_fault_beside_its_twin(tmp_path: Path) -> None:
    import numpy as np

    from fleetlab.video import write_mp4

    films = JOBS["s3-films"]
    run = s3.run_dir("s3-films", tmp_path)
    toy_run(films, run)
    frames = [np.full((16, 16, 3), shade, np.uint8) for shade in range(0, 200, 20)]
    for job in films[:4]:  # two filmed episodes and their twins have videos
        for camera in ("scene", "wrist"):
            write_mp4(run / "log" / "video" / f"{job.identity.episode_id}.{camera}.mp4", frames, 10)
    index = s3.films(run, tmp_path / "page")
    text = index.read_text()
    assert text.count("### `") == len(s3.FILMED)
    assert "- healthy `e000000` (" in text and "](media/e000000.jpg)" in text
    assert "no video" in text  # the episodes without a video say so
    assert not (tmp_path / "page" / "media" / "e000000.scene.mp4").exists()
    s3.films(run, tmp_path / "bundle", copy_videos=True)
    assert (tmp_path / "bundle" / "media" / "e000000.scene.mp4").is_file()
    assert "[scene](media/e000000.scene.mp4)" in (tmp_path / "bundle" / "index.md").read_text()


def test_a_re_measured_point_of_a_stage_2_type_replaces_the_stored_one(tmp_path: Path) -> None:
    curves = tmp_path / "curves.json"
    shutil.copy(Path(dr.__file__).parent / "faults" / "curves.json", curves)
    stored = json.loads(curves.read_text())["types"]["camera_pose_offset"]["points"]
    first = {**stored[0], "n": 300, "run_id": "s3-again", "chain_ok": True}
    extra = {**stored[0], "label": "a new point", "chain_ok": True}
    result = {"types": {}, "appended": {"camera_pose_offset": [first, extra]}, "joints": []}
    s3.write_curves(result, curves)
    points = json.loads(curves.read_text())["types"]["camera_pose_offset"]["points"]
    assert points[0]["run_id"] == "s3-again" and points[1:-1] == stored[1:]
    assert points[-1]["label"] == "a new point" and len(points) == len(stored) + 1


@pytest.mark.parametrize("chain_ok", [False, None])
def test_failed_replay_is_rejected_on_resume_analysis_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chain_ok: bool | None
) -> None:
    stub = StubFleet()
    monkeypatch.setattr(dr, "_engine", lambda module: stub)
    item = s3.PlanRun("s3-lens_contamination", "lens_contamination", 30, (0,))
    path = s3.run_dir(item.name, tmp_path)
    toy_run(s3.jobs_for(item, POLICY, SOFTWARE), path)
    RunLayout(path).run_record.write_text(json.dumps({"chain_ok": chain_ok}))
    setup = s3.EngineSetup(POLICY, SOFTWARE, {}, min_healthy=0.5)
    assert s3.run(root=tmp_path, setup=setup, runs=[item]) == 1
    assert not stub.ran
    with pytest.raises(ValueError, match="rejected calibration"):
        s3._runs(tmp_path, item.name)
    bad = {"types": {}, "appended": {}, "joints": [{"chain_ok": chain_ok}]}
    curves = tmp_path / "curves.json"
    curves.write_text("{}")
    with pytest.raises(ValueError, match="replay validation"):
        s3.write_curves(bad, curves)
    assert curves.read_text() == "{}"


def test_a_fresh_fault_run_with_failed_replay_stops_before_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubFleet()
    original = stub.run

    def failed(config: Any, jobs: Any) -> Any:
        result = original(config, jobs)
        RunLayout(Path(config.out)).run_record.write_text(json.dumps({"chain_ok": False}))
        result.record.chain_ok = False
        return result

    monkeypatch.setattr(stub, "run", failed)
    monkeypatch.setattr(dr, "_engine", lambda module: stub)
    runs = [
        s3.PlanRun("bad", "lens_contamination", 30, (0,)),
        s3.PlanRun("next", "arm_torque_loss", 30, (0,)),
    ]
    setup = s3.EngineSetup(POLICY, SOFTWARE, {}, min_healthy=0.5)
    assert s3.run(root=tmp_path, setup=setup, runs=runs) == 1
    assert stub.ran == ["bad"]
    assert s3.run(root=tmp_path, setup=setup, runs=runs) == 1
    with pytest.raises(ValueError, match="rejected calibration"):
        s3._runs(tmp_path, "bad")


def test_exact_episode_set_rejects_a_complete_run_of_the_wrong_episodes(tmp_path: Path) -> None:
    item = s3.PlanRun("wrong-ids", "lens_contamination", 10, (0,))
    jobs = s3.jobs_for(item, POLICY, SOFTWARE)
    wrong = [
        jobs[0].model_copy(
            update={"identity": jobs[0].identity.model_copy(update={"episode_id": "e999999"})}
        ),
        *jobs[1:],
    ]
    path = s3.run_dir(item.name, tmp_path)
    toy_run(wrong, path)
    assert s3.complete(path, len(jobs))
    with pytest.raises(ValueError, match="planned episodes"):
        s3.validate_run(path, item, jobs, 0.85)
    receipt = json.loads((path / "calibration-validation.json").read_text())
    assert receipt["accepted"] is False
    # chain_ok=True is insufficient after a persisted rejection.
    with pytest.raises(ValueError, match="rejected calibration"):
        s3._runs(tmp_path, item.name)


@pytest.mark.parametrize("flag", [None, False])
@pytest.mark.parametrize("arm", ["healthy", "fault"])
def test_direct_analysis_requires_validation_of_both_arms(
    tmp_path: Path, flag: bool | None, arm: str
) -> None:
    from dataclasses import replace

    healthy = s3.PlanRun("healthy", count=100)
    faulted = s3.PlanRun("fault", "lens_contamination", 100, (0,))
    for item in (healthy, faulted):
        toy_run(s3.jobs_for(item, POLICY, SOFTWARE), s3.run_dir(item.name, tmp_path))
    h = dr.read_run(s3.run_dir(healthy.name, tmp_path))
    f = dr.read_run(s3.run_dir(faulted.name, tmp_path))
    point = dr.analyse(faulted.kind or "", h, f)[0]
    with pytest.raises(ValueError, match="replay validation"):
        dr.analyse(
            faulted.kind or "",
            replace(h, chain_ok=flag) if arm == "healthy" else h,
            replace(f, chain_ok=flag) if arm == "fault" else f,
        )
    with pytest.raises(ValueError, match="replay validation"):
        s3.point_dict(replace(point, chain_ok=flag), "small_cause")


def test_object_analysis_merges_extensions_and_pairs_joint_on_object_twins(tmp_path: Path) -> None:
    plans = [
        s3.PlanRun("object-healthy", count=300, suite=s3.OBJECT),
        s3.PlanRun("object-healthy-600", count=600, first=300, suite=s3.OBJECT),
        s3.PlanRun("spatial-healthy", count=300),
        s3.PlanRun(
            "dim-300",
            "illumination_change",
            300,
            s3.labels("illumination_change", "2 stops dimmer"),
            suite=s3.OBJECT,
        ),
        s3.PlanRun(
            "dim-600",
            "illumination_change",
            600,
            s3.labels("illumination_change", "2 stops dimmer"),
            first=300,
            suite=s3.OBJECT,
        ),
        s3.PlanRun(
            "delay", "action_latency", 300, s3.labels("action_latency", "14 steps"), suite=s3.OBJECT
        ),
        s3.PlanRun(
            "chunks",
            "policy_knob",
            300,
            s3.labels("policy_knob", "10 actions per chunk"),
            suite=s3.OBJECT,
        ),
        s3.PlanRun("joint", count=300, suite=s3.OBJECT, joint_name="delay+short-chunks"),
        s3.PlanRun("detector", "success_false_positive", 100, (0,), suite=s3.OBJECT),
    ]

    def outcome(job: Job) -> tuple[bool, bool]:
        identity = job.identity
        kinds = {f.kind for f in identity.faults}
        i = identity.episode_index
        if identity.suite != s3.OBJECT:
            return False, False  # the other suite must not affect the healthy floor or joint
        true = i % 20 != 0
        if {"action_latency", "policy_knob"} <= kinds:
            true = False
        elif "action_latency" in kinds:
            true &= i % 5 != 1
        elif "policy_knob" in kinds or "illumination_change" in kinds:
            true &= i % 7 != 3
        if "success_false_positive" in kinds:
            return False, i % 2 == 0
        return true, true

    for item in plans:
        toy_run(s3.jobs_for(item, POLICY, SOFTWARE), s3.run_dir(item.name, tmp_path), outcome)
    result = s3.analyse_object(tmp_path, plans)
    assert result["object"]["healthy"]["n"] == 600
    dim = result["object"]["illumination_change"][0]
    assert dim["n"] == 600 and dim["healthy_successes"] == 570
    expected_broken = sum(i % 20 != 0 and i % 7 == 3 for i in range(600))
    assert dim["broken"] == expected_broken and dim["rescued"] == 0
    assert dim["run_dirs"] == [str(s3.run_dir(name, tmp_path)) for name in ("dim-300", "dim-600")]
    joint = result["joints"][0]
    assert joint["suite"] == s3.OBJECT and joint["n"] == 300
    assert joint["successes"] == 0 and joint["healthy_successes"] == 285
    assert joint["single_successes"] == [
        sum(i % 20 != 0 and i % 5 != 1 for i in range(300)),
        sum(i % 20 != 0 and i % 7 != 3 for i in range(300)),
    ]
    assert joint["single_run_dirs"] == [
        [str(s3.run_dir(name, tmp_path))] for name in ("delay", "chunks")
    ]
    detector = result["object"]["success_false_positive"][0]
    assert detector["healthy_successes"] == 0 and detector["successes"] == 50
    assert detector["compared_with"] == "the same episodes' true outcome"
    assert detector["healthy_run_dirs"] == detector["run_dirs"]
    curves = tmp_path / "curves.json"
    shutil.copy(Path(dr.__file__).parent / "faults" / "curves.json", curves)
    before = json.loads(curves.read_text())
    s3.write_curves(result, curves)
    after = json.loads(curves.read_text())
    assert after["stage3"] == before["stage3"] and after["healthy"] == before["healthy"]
    assert after["types"]["illumination_change"]["verdicts"] == {
        dr.SUITE: "decoy",
        s3.OBJECT: "kept",
    }
    assert (
        after["types"]["illumination_change"]["points"]
        == before["types"]["illumination_change"]["points"]
    )
    assert next(p for p in after["supplement"]["points"] if p["kind"] == "illumination_change") == {
        "kind": "illumination_change",
        "suite": s3.OBJECT,
        **dim,
    }


def test_object_analysis_refuses_a_low_healthy_baseline(tmp_path: Path) -> None:
    item = s3.PlanRun("object-healthy", count=100, suite=s3.OBJECT)
    toy_run(
        s3.jobs_for(item, POLICY, SOFTWARE),
        s3.run_dir(item.name, tmp_path),
        lambda job: (False, False),
    )
    with pytest.raises(ValueError, match="object healthy calibration"):
        s3.analyse_object(tmp_path, [item])
