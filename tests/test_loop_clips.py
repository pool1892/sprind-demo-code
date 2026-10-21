"""Video for a subset of the reserved panel, and the clip rule (docs/design/m3-recipe.md, 3).

CPU only. The panel comes from the lead loop fleet written to a temporary root; the
engine is never started (its calls are stubbed where `evaluate` is exercised), and the
clip rule runs on synthetic evaluation directories.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fleetlab.contracts.identity import SoftwareVersion
from fleetlab.loop.clips import final_state, select_clips
from fleetlab.loop.evaluate import panel_jobs
from fleetlab.loop.operate import EngineSetup

SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)


def config() -> Any:
    from fleetlab.engine.fleet import RunConfig

    return RunConfig(out="", batch_size=4, workers=1, precision="fp16_res32")


@cache
def _lead_root(base: str) -> Path:
    from fleetlab.worlds import loop_fleets as lf

    texts = {t: f"task {t.suite} {t.task_index}" for t in lf.TASKS}
    world = lf.build(lf.RECIPES["LEAD"], SOFTWARE, texts)
    root = Path(base) / "w70"
    lf.write_world(world, root / "files", [str(root / "runs" / "w70-observed")])
    lf.write_beside(world, root)
    return root


@pytest.fixture(scope="module")
def lead(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from fleetlab.loop.world import LoopWorld

    return LoopWorld.load(_lead_root(str(tmp_path_factory.mktemp("lead"))))


# ---- the video subset ----------------------------------------------------------------------


def test_only_the_chosen_indices_record_video(lead: Any) -> None:
    from fleetlab.loop.state import Ledger

    panel = lead.panel_identities([14, 15], SOFTWARE)
    empty = Ledger.empty(lead.world_id)
    jobs = panel_jobs(lead, panel, empty, SOFTWARE, frozenset({0}))
    assert len(jobs) == len(panel) == 2 * 12 * 12
    filmed = [j for j in jobs if j.record_video]
    assert len(filmed) == 2 * 12  # one attempt per robot and day
    assert {j.identity.episode_index for j in filmed} == {0}
    assert {(j.identity.robot_id, j.identity.day) for j in filmed} == {
        (i.robot_id, i.day) for i in panel
    }
    assert all(j.record_video for j in panel_jobs(lead, panel, empty, SOFTWARE, True))
    assert not any(j.record_video for j in panel_jobs(lead, panel, empty, SOFTWARE, False))
    assert not any(j.record_video for j in panel_jobs(lead, panel, empty, SOFTWARE, ()))


def test_the_same_attempts_are_filmed_in_every_state(lead: Any) -> None:
    from fleetlab.contracts.actions import Action
    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.vocabulary import ConfigKey, Verb
    from fleetlab.loop.state import Ledger

    panel = lead.panel_identities([14], SOFTWARE)
    empty = Ledger.empty(lead.world_id)
    compensate = Action(
        verb=Verb.set,
        target=Target(scope="s1", config_key=ConfigKey.control_latency_compensation_steps),
        value=17,
    )
    adopted = empty.adopt(1, 8, adopt=[compensate])
    before = panel_jobs(lead, panel, empty, SOFTWARE, frozenset({0}))
    after = panel_jobs(lead, panel, adopted, SOFTWARE, frozenset({0}))
    assert [j.identity.episode_id for j in before if j.record_video] == [
        j.identity.episode_id for j in after if j.record_video
    ]


def test_evaluate_films_the_setups_indices(
    lead: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine import fleet
    from fleetlab.loop import evaluate as evaluate_module
    from fleetlab.loop import operate
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.state import Ledger

    runs: list[list[Any]] = []
    monkeypatch.setattr(fleet, "prepare", lambda config: SOFTWARE)
    monkeypatch.setattr(fleet, "run", lambda config, jobs: runs.append(list(jobs)))
    monkeypatch.setattr(operate, "_complete", lambda run_dir, jobs: False)
    monkeypatch.setattr(evaluate_module, "summarize", lambda *args, **kwargs: None)
    layout = layout_for_world(lead.spec)
    paths = LoopPaths(tmp_path / "loop")
    states = {"original": Ledger.empty(lead.world_id)}

    setup = EngineSetup(config=config(), panel_video_indices=(0,))
    evaluate_module.evaluate(lead, layout, states, paths, setup, per_robot_day=2)
    (jobs,) = runs
    assert len(jobs) == 4 * 12 * 2
    assert [j.identity.episode_index for j in jobs if j.record_video] == [0] * (4 * 12)

    runs.clear()  # an explicit argument wins over the setup; `video` films everything
    evaluate_module.evaluate(
        lead, layout, states, LoopPaths(tmp_path / "b"), setup, per_robot_day=2, video_indices=()
    )
    assert not any(j.record_video for j in runs[0])
    runs.clear()
    evaluate_module.evaluate(
        lead, layout, states, LoopPaths(tmp_path / "c"), setup, per_robot_day=2, video=True
    )
    assert all(j.record_video for j in runs[0])


def test_the_setup_carries_its_indices_through_json() -> None:
    from fleetlab.loop.world_loop import setup_from_json, setup_to_json

    setup = EngineSetup(config=config(), panel_video_indices=(3, 0, 0))
    assert setup.panel_video_indices == (0, 3)
    again = setup_from_json(json.loads(json.dumps(setup_to_json(setup))))
    assert again == setup
    assert again.panel_video_indices == (0, 3)
    assert EngineSetup(config=config()).panel_video_indices == ()
    with pytest.raises(ValueError):
        EngineSetup(config=config(), panel_video_indices=(-1,))


# ---- the clip rule -------------------------------------------------------------------------


def _panel(*rows: tuple[str, str, str, int, int]) -> list[dict[str, Any]]:
    return [
        {"episode_id": e, "robot_id": r, "site_id": s, "day": d, "episode_index": i}
        for e, r, s, d, i in rows
    ]


def _state(root: Path, name: str, outcomes: dict[str, str], filmed: set[str]) -> None:
    """A state's run as the clip rule reads it: the episode log and the filmed videos."""
    log = root / name / "log"
    (log / "video").mkdir(parents=True)
    rows = []
    for eid, outcome in outcomes.items():
        videos = {}
        for camera in ("scene", "wrist"):
            relative = f"video/{eid}.{camera}.mp4" if eid in filmed else None
            if relative:
                (log / relative).write_bytes(b"mp4")
            videos[f"video_{camera}"] = relative
        rows.append({"episode_id": eid, "outcome": outcome, **videos})
    pq.write_table(pa.Table.from_pylist(rows), log / "episodes.parquet")


def _evaluation(root: Path) -> Path:
    evaluation = root / "evaluation"
    evaluation.mkdir()
    panel = _panel(
        ("e1", "r01", "s1", 14, 0),  # rescued at the site, not filmed
        ("e2", "r01", "s1", 14, 1),  # not rescued, filmed
        ("e3", "r02", "s1", 14, 0),  # rescued at the site, filmed: the clip at the site
        ("e4", "r05", "s2", 14, 0),  # rescued elsewhere, filmed under one state only
        ("e5", "r06", "s2", 15, 0),  # rescued elsewhere, filmed: the clip elsewhere
        ("e6", "r07", "s2", 15, 0),  # broken (success, then failure), filmed
    )
    (evaluation / "panel.json").write_text(json.dumps(panel))
    before = {"e1": "failure", "e2": "failure", "e3": "failure", "e4": "failure"}
    before |= {"e5": "failure", "e6": "success"}
    after = {"e1": "success", "e2": "failure", "e3": "success", "e4": "success"}
    after |= {"e5": "success", "e6": "failure"}
    _state(evaluation, "original", before, {"e2", "e3", "e4", "e5", "e6"})
    _state(evaluation, "round-2", after, {"e2", "e3", "e5", "e6"})
    return evaluation


def test_the_clips_are_the_first_filmed_rescues(tmp_path: Path) -> None:
    evaluation = _evaluation(tmp_path)
    out = select_clips(evaluation, "s1")
    assert out["after"] == "round-2"
    assert out["recorded_candidates"] == 4  # e2, e3, e5, e6 (e1 unfilmed, e4 half filmed)
    assert out["rescued_among_recorded"] == 2
    assert out["at_site"]["episode_id"] == "e3"
    assert out["at_site"]["before"]["outcome"] == "failure"
    assert out["at_site"]["after"]["outcome"] == "success"
    assert all(Path(p).is_file() for p in out["at_site"]["after"]["videos"].values())
    assert out["elsewhere"]["episode_id"] == "e5"


def test_no_unfilmed_attempt_is_ever_chosen(tmp_path: Path) -> None:
    evaluation = _evaluation(tmp_path)
    assert select_clips(evaluation, "s2")["at_site"]["episode_id"] == "e5"
    # Without e5's videos under the final state, s2 is left with e4, rescued but filmed
    # under one state only: nothing is chosen there, and the record says why.
    for video in (evaluation / "round-2" / "log" / "video").glob("e5.*"):
        video.unlink()
    out = select_clips(evaluation, "s2")
    assert out["at_site"] == {
        "none": "no recorded attempt at s2 failed under original and succeeded under round-2"
    }
    assert out["elsewhere"]["episode_id"] == "e3"  # not e1, rescued first but never filmed
    assert out["recorded_candidates"] == 3


def test_the_final_state_is_round_two_when_it_was_evaluated(tmp_path: Path) -> None:
    (tmp_path / "round-1").mkdir()
    assert final_state(tmp_path) == "round-1"
    (tmp_path / "round-2").mkdir()
    assert final_state(tmp_path) == "round-2"
