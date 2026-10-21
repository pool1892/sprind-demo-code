"""The ladder of worlds without a simulator (`make check`): the worlds, the oracle's power set,
the digests, a toy world run end to end and the comparison of two nights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fleetlab.contracts.identity import Task
from fleetlab.engine import ladder, software
from fleetlab.engine.chain import Chain
from fleetlab.policies.xvla import nominal_spec

SOFT = software.current("none")
SPEC = nominal_spec("fp16_res32")
TEXTS = {Task(t.suite, t.task_index): f"task {t.task_index}" for t in ladder.NIGHTLY_TASKS}


def test_the_smoke_world_is_one_robot_and_eight_episodes() -> None:
    world = ladder.smoke_world()
    ids = world.identities(SOFT, SPEC)
    assert len(ids) == 8 and {i.robot_id for i in ids} == {"r01"}
    assert [len(i.faults) for i in ids] == [0] * 4 + [1] * 4  # the camera from day 1
    assert ids == world.identities(SOFT, SPEC)
    for i in ids:
        i.check_seeds(world.seed)
    plan = world.plan(SOFT, SPEC)
    assert plan.scoring_days == (2,) and plan.window_of(1) == "experiment"


@pytest.fixture(scope="module")
def fixed() -> ladder.WorldPlan:
    return ladder.generated_plan(ladder.nightly_spec(), SOFT, TEXTS)


def test_the_fixed_nightly_world_repeats(fixed: ladder.WorldPlan) -> None:
    assert fixed.world_id == "w20" and len(fixed.identities) == 480
    assert len({i.episode_id for i in fixed.identities}) == 480
    again = ladder.generated_plan(ladder.nightly_spec(), SOFT, TEXTS)
    assert again.identities == fixed.identities
    faulted = {i.robot_id for i in fixed.identities if i.faults}
    assert faulted == {"r02", "r03", "r04"}
    assert {i.robot_id for i in fixed.identities if i.day == 0 and i.faults} == {"r02", "r03"}
    for i in fixed.identities:
        i.check_seeds(fixed.seed)


def test_the_drawn_nightly_world_moves_with_its_seed() -> None:
    a, b = ladder.nightly_spec(20261008), ladder.nightly_spec(20261009)
    assert a.world_id == b.world_id == "w21" and a.seed != b.seed
    assert a == ladder.nightly_spec(20261008)
    draws = set()
    for s in range(20261001, 20261011):
        faults = ladder.nightly_spec(s).fault_schedule
        draws.add(tuple((f.target.scope, f.level, f.schedule.onset_day) for f in faults))
    assert len(draws) > 1
    small = ladder.generated_plan(ladder.nightly_spec(20261008, per_day=2), SOFT, TEXTS)
    assert len(small.identities) == 24 and small.per_day == 2


def test_the_oracle_replays_every_subset_of_a_scoring_episode_s_faults(
    fixed: ladder.WorldPlan,
) -> None:
    from fleetlab.integration.oracle import oracle_jobs as integration_oracle

    jobs = ladder.oracle_jobs(fixed.identities, fixed.scoring_days)
    scoring_faulted = [i for i in fixed.identities if i.day == 2 and i.faults]
    expected = sum(2 ** len(i.faults) - 1 for i in scoring_faulted)
    assert len(jobs) == expected == 120 and len({j.key for j in jobs}) == expected
    theirs = integration_oracle(fixed.generated).jobs
    assert sorted(j.key for j in jobs) == sorted(j.key for j in theirs)
    live = ladder.live_jobs(fixed.identities, fixed.scoring_days)
    assert len(live) == 160 and all(j.supervisor is not None for j in live)


def test_the_registry_accepts_every_fault_of_the_ladder() -> None:
    import fleetlab.faults as faults

    specs = [*ladder.smoke_world().faults, *ladder.nightly_spec().fault_schedule]
    for seed in range(20261001, 20261011):
        specs += ladder.nightly_spec(seed).fault_schedule
    for spec in specs:
        faults.REGISTRY.check(spec)
        levels = {p.level for p in getattr(faults.get(spec.kind), "calibration", ())}
        assert spec.level in levels, (spec.kind, spec.level)


def test_a_chain_digest_sees_every_bit(tmp_path: Path) -> None:
    chain = Chain(fixture_hash="00" * 8)
    for t in range(4):
        chain.add(t, f"{t:016x}", "a" * 16, "b" * 16, None if t == 0 else np.ones(7, np.float32))
    path = tmp_path / "e000001.npz"
    chain.save(path)
    first = ladder.chain_digest(path)
    assert first == ladder.chain_digest(path)
    chain.executed[2] = chain.executed[2].copy()
    chain.executed[2][0] = np.nextafter(np.float32(1), np.float32(2))
    chain.save(path)
    assert ladder.chain_digest(path) != first


def test_a_toy_world_is_run_replayed_handed_over_and_scored(tmp_path: Path) -> None:
    from engine_helpers import config, fault

    from fleetlab.contracts.rollout import Job
    from fleetlab.engine import fleet, manifests, toy

    delay = fault("toy_delay", 1, robot=1, steps=3)
    ids = [
        manifests.identity(
            world_seed=3, episode_number=n, robot=1, site=1, day=0 if n <= 2 else 2,
            episode_index=n, suite="toy", task_index=n % 2, start_state=n % 7,
            policy=toy.TOY_SPEC, batch_size=4, software=SOFT, image_size=32, max_steps=60,
            faults=[delay] if n % 2 else [],
        )
        for n in range(1, 7)
    ]  # fmt: skip
    windows: dict[str, tuple[int, ...]] = {"observed": (0,), "experiment": (1,), "scoring": (2,)}
    plan = ladder.WorldPlan("w01", 3, (delay,), windows, 6, ids)
    oracle = ladder.oracle_jobs(ids, plan.scoring_days)
    assert len(oracle) == 2  # e000003 and e000005 carry the fault on the scoring day
    dirs = ladder.run_dirs(tmp_path, "w01-toy")
    assert dirs["live"].name == "w01-toy-live"
    fleet.run(config(dirs["world"]), [Job(identity=i) for i in ids] + oracle)
    live = ladder.live_jobs(ids, plan.scoring_days, after=10)
    fleet.run(config(dirs["live"], continuation_steps=30), live)
    result = ladder.scores(plan, dirs)
    assert result["success"]["observed"]["n"] == 2 and result["success"]["scoring"]["n"] == 4
    assert result["success"]["experiment"]["n"] == 0
    assert set(result["fault_effects"]) == {"f01"}
    assert set(result["achievable_net"]) == {"1", "2", "3"}
    assert result["runs"]["world"]["chain_ok"] and result["runs"]["world"]["jobs_done"] >= 8
    assert result["live"]["episodes"] == 4
    assert result["live"]["handed_over"] + result["live"]["succeeded_alone"] <= 4
    found = ladder.digests(dirs["world"])
    assert {"e000001", "e000006"} <= set(found)


def night(
    digest: str, world_hash: str = "h1", files: dict[str, str] | None = None
) -> dict[str, Any]:
    world: dict[str, Any] = {
        "seed": 20,
        "world_hash": world_hash,
        "world_files": files,
        "generator_source": "g1" if files is None else "g2",
        "success": {w: {"ok": 30, "n": 40} for w in ("observed", "experiment", "scoring")},
        "by_robot": {"r01": {"ok": 40, "n": 40}},
        "fault_effects": {"f01": {"net": 5}},
        "achievable_net": {"1": 5, "2": 8, "3": 9},
        "faults": [{"kind": "camera_pose_offset", "level": 6.0, "robot": "r02", "onset": 0}],
        "runs": {"world": {"jobs_done": 600, "chain_ok": True, "total_s": 1200.0}},
        "digests": {"world": {"e000001": digest}, "live": {}},
    }
    return {
        "date": "2026-10-08",
        "started": 0.0,
        "finished": 3600.0,
        "deterministic": world,
        "randomized": {**world, "seed": 20261008},
    }


def test_two_nights_are_compared_chain_by_chain() -> None:
    same = ladder.compare_nights(night("aa"), night("aa"), Path("2026-10-07"))
    assert same["json"]["identity"]["world"]["differ"] == []
    assert "identical" in same["markdown"] and "**different**" not in same["markdown"]
    moved = ladder.compare_nights(night("bb"), night("aa"), Path("2026-10-07"))
    assert moved["json"]["identity"]["world"]["differ"] == ["e000001"]
    assert "**different**" in moved["markdown"]
    regenerated = ladder.compare_nights(night("aa", "h2"), night("aa"), Path("2026-10-07"))
    assert "world files **changed**" in regenerated["markdown"]
    assert "not known" in regenerated["markdown"]
    old = {"spec.json": "1", "model.json": "2", "description.json": "3", "manifest.jsonl": "4"}
    new = {**old, "model.json": "5"}
    told = ladder.compare_nights(night("aa", "h2", new), night("aa", "h1", old), Path("x"))
    assert told["json"]["identity"]["world_files_changed"] == ["model.json"]
    assert "World files changed: model.json" in told["markdown"]
    assert "**different**" not in told["markdown"]
    both = ladder.compare_nights(night("bb", "h2", new), night("aa", "h1", old), Path("x"))
    assert "**different**" in both["markdown"]
    first = ladder.compare_nights(night("aa"), None, None)
    assert "first reference" in first["markdown"]


def test_the_soak_summary_reads_the_memory_series(tmp_path: Path) -> None:
    import json

    series = [[t * 30.0, 9.0, 40.0 - t * 0.01, 4.0 + t * 0.001, 20.0, 10, 2.1] for t in range(240)]
    lives = [{"worker": w, "life": 1, "jobs": 16, "pss_mib": 2000.0 + w} for w in range(12)]
    run = {
        "chain_ok": True,
        "extra": {"total_s": 7200.0, "jobs_done": 900, "memory_series": series,
                  "worker_lives": lives, "incidents": []},
    }  # fmt: skip
    (tmp_path / "run.json").write_text(json.dumps(run))
    out = ladder.soak_summary(tmp_path)
    assert out["hours"] == 2.0 and out["samples"] == 240 and out["worker_lives"] == 12
    assert abs(out["server_slope_gib_per_h"] - 0.12) < 1e-3  # 0.001 GiB per 30 s
    assert out["mem_available_min_gib"] < 40.0
    assert out["worker_pss_at_end_gib"]["last_quarter_median"] > 1.95


def test_the_soak_runs_worlds_until_its_budget_and_resumes(tmp_path: Path) -> None:
    import json

    from engine_helpers import config, fault

    from fleetlab.engine import manifests, toy

    def plans(k: int) -> Any:
        def make(soft: Any) -> ladder.WorldPlan:
            delay = fault("toy_delay", 1, robot=1, steps=3)
            ids = [
                manifests.identity(
                    world_seed=100 + k, episode_number=n, robot=1, site=1, day=n % 3,
                    episode_index=n, suite="toy", task_index=n % 2, start_state=n % 7,
                    policy=toy.TOY_SPEC, batch_size=4, software=SOFT, image_size=32,
                    max_steps=40, faults=[delay] if n % 2 else [],
                )
                for n in range(1, 13)
            ]  # fmt: skip
            windows: dict[str, tuple[int, ...]] = {
                "observed": (0,), "experiment": (1,), "scoring": (2,)
            }  # fmt: skip
            return ladder.WorldPlan("w01", 100 + k, (delay,), windows, 13, ids)

        return make

    cfg = config(tmp_path / "unused", continuation_steps=20)
    root = tmp_path / "soak"
    record = ladder.soak_run(
        root, cfg, hours=1.0, base_seed=7, max_worlds=2, plans=plans, soft=SOFT
    )
    assert record["worlds"] == 2 and record["sessions"] == 1
    assert record["canaries"] == {"worlds": 2, "compared": 20, "identical": 20}
    assert record["incidents"] == 0 and record["chain_samples_broken"] == []
    assert not record["passed"]  # two toy worlds are not an hour
    assert len(record["runs"]) == 6 and (root / "soak-summary.md").is_file()
    state = json.loads((root / "soak.json").read_text())
    assert [w["seed"] for w in state["worlds"]] == [700, 701]
    # A rerun after the end runs nothing and keeps one running session.
    again = ladder.soak_run(root, cfg, hours=1.0, base_seed=7, max_worlds=2, plans=plans, soft=SOFT)
    assert again["worlds"] == 2 and again["sessions"] == 1
    assert len(json.loads((root / "soak.json").read_text())["sessions"]) == 2
    # A budget already spent stops before a new world.
    state["hours"] = 0.0
    (root / "soak.json").write_text(json.dumps(state))
    assert (
        ladder.soak_run(root, cfg, base_seed=7, max_worlds=5, plans=plans, soft=SOFT)["worlds"] == 2
    )


def soak_state(
    root: Path,
    *,
    incidents: int = 1,
    recovered: bool = True,
    chain_ok: bool | None = True,
    hours: float = 6.0,
    identical: int = 10,
    sessions: int = 1,
) -> dict[str, Any]:
    """Stored CPU fixtures for the pass rule, without running six hours of toy jobs."""
    dirs = {}
    for part in ("world", "live", "canary"):
        folder = root / part
        folder.mkdir()
        dirs[part] = str(folder)
        losses = [
            {"worker": 0, "why": "traceback line\nexit code -6", "jobs": ["e000001"],
             "attempts": {"e000001": 1}, "recovered": recovered,
             "replay_checks": {"e000001": {"checked": True, "verdict": "identical"}}}
            for _ in range(incidents if part == "world" else 0)
        ]  # fmt: skip
        run = {
            "chain_ok": chain_ok if part == "world" else (True if part == "live" else None),
            "extra": {"incidents": losses, "jobs_done": 1, "total_s": hours * 3600},
        }
        (folder / "run.json").write_text(json.dumps(run))
    world = {"index": 0, "seed": 7, "faults": [], "dirs": dirs, "started": 1.0,
             "finished": 1.0 + hours * 3600, "seconds": hours * 3600,
             "timing": {"world": hours * 3600, "live": 0, "canary": 0},
             "canary": {"compared": 10, "identical": identical}}  # fmt: skip
    return {"hours": 6.0, "worlds": [world], "sessions": [{"worlds": [0]}] * sessions}


@pytest.mark.parametrize(
    ("changes", "passed"),
    [
        ({}, True),
        ({"incidents": 6}, True),
        ({"incidents": 7}, False),
        ({"recovered": False}, False),
        ({"chain_ok": False}, False),
        ({"identical": 9}, False),
        ({"sessions": 2}, False),
        ({"hours": 5.9}, False),
        ({"hours": 5.995}, False),
        ({"hours": 5.9999}, False),
    ],
)
def test_the_soak_pass_rule(tmp_path: Path, changes: dict[str, Any], passed: bool) -> None:
    state = soak_state(tmp_path, **changes)
    record = ladder.soak_record(state)
    assert record["passed"] is passed
    assert record["incidents_per_hour"] == pytest.approx(
        record["incidents"] / (state["worlds"][0]["seconds"] / 3600)
    )
    assert record["incidents_recovered"] == (
        record["incidents"] if changes.get("recovered", True) else 0
    )
    markdown = ladder.soak_markdown(record)
    assert f"incidents {record['incidents']} ({record['incidents_recovered']} recovered" in markdown
    assert "per running hour (limit 1)" in markdown
    assert markdown.count("cause: exit code -6") == record["incidents"]
    assert (
        "lost assignment 1; execution outcome not recorded; replay checked, verdict identical"
        in markdown
    )
    assert bool(record["failed_conditions"]) is not passed
    assert all(why in markdown for why in record["failed_conditions"])
    assert "{'" not in markdown


def test_the_soak_rate_uses_unrounded_hours(tmp_path: Path) -> None:
    state = soak_state(tmp_path, incidents=6, hours=5.9999)
    state["hours"] = 5.0  # hours gate holds; only the rate exceeds its bound
    record = ladder.soak_record(state)
    assert record["incidents_unrecovered"] == 0 and record["chain_samples_broken"] == []
    assert record["sessions"] == 1 and record["canaries"]["identical"] == 10
    assert record["incidents_per_hour"] > 1.0 and not record["passed"]
    markdown = ladder.soak_markdown(record)
    assert ">1.00 per running hour" in markdown
    assert "Incident rate exceeds 1 per running hour." in markdown


def test_the_soak_hours_do_not_look_within_their_bound_when_they_fail(tmp_path: Path) -> None:
    record = ladder.soak_record(soak_state(tmp_path, hours=5.9999))
    assert record["hours_running"] == 6.0
    assert record["hours_running_exact"] == 5.9999
    markdown = ladder.soak_markdown(record)
    assert "<6 hours running" in markdown and "6.0 hours running" not in markdown
    assert "Hours requirement not reached (6 required)." in markdown


@pytest.mark.parametrize(
    ("chain_ok", "present", "held"), [(True, 2, 2), (False, 2, 1), (None, 1, 1)]
)
def test_the_soak_counts_only_runs_with_a_chain_sample(
    tmp_path: Path, chain_ok: bool | None, present: int, held: int
) -> None:
    record = ladder.soak_record(soak_state(tmp_path, chain_ok=chain_ok))
    assert len(record["runs"]) == 3
    assert record["chain_samples_present"] == present
    assert record["chain_samples_held"] == held
    assert f"{held} of {present} runs with a sample" in ladder.soak_markdown(record)


def test_the_soak_markdown_rounds_rate_and_describes_unchecked_replays(tmp_path: Path) -> None:
    state = soak_state(tmp_path)
    path = tmp_path / "world" / "run.json"
    run = json.loads(path.read_text())
    run["extra"]["incidents"][0]["replay_checks"]["e000001"] = {"checked": False, "verdict": None}
    path.write_text(json.dumps(run))
    record = ladder.soak_record(state)
    assert record["incidents_per_hour"] == 1.0 / 6
    markdown = ladder.soak_markdown(record)
    assert "0.17 per running hour" in markdown and "0.166666" not in markdown
    assert (
        "Job e000001: lost assignment 1; execution outcome not recorded; not checked by replay."
        in markdown
    )
    assert "{'" not in markdown


def test_an_old_incident_is_recovered_only_with_completion_evidence(tmp_path: Path) -> None:
    from fleetlab.contracts.rollout import RunLayout

    state = soak_state(tmp_path)
    path = tmp_path / "world" / "run.json"
    run = json.loads(path.read_text())
    del run["extra"]["incidents"][0]["recovered"]
    path.write_text(json.dumps(run))
    assert not ladder.soak_record(state)["passed"]
    run["extra"]["jobs_detail"] = [{"key": "e000001", "mode": "run"}]
    path.write_text(json.dumps(run))
    assert not ladder.soak_record(state)["passed"]
    chain = RunLayout(tmp_path / "world").table("chain", "e000001")
    chain.parent.mkdir(parents=True)
    chain.write_bytes(b"stored completion")
    original = path.read_bytes()
    record = ladder.soak_record(state)
    assert record["passed"] and record["incidents_recovered"] == 1
    assert record["incidents_detail"][0]["recovery_inferred"]
    assert path.read_bytes() == original
