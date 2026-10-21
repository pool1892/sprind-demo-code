"""Same-attempt evidence must survive analysis, not just a derived flag."""

from pathlib import Path

import pytest

from fleetlab.contracts.identity import SoftwareVersion
from fleetlab.contracts.rollout import Job
from fleetlab.dose_response import Outcome, RunOutcomes
from fleetlab.worlds import supplement as s
from fleetlab.worlds.compose import ADDITIVE_PAIR
from fleetlab.worlds.curves import SPATIAL
from fleetlab.worlds.generate import policy_spec_nominal


def setup_run(monkeypatch: pytest.MonkeyPatch, root: Path, *, singles: bool) -> list[Job]:
    points = [s.Point(SPATIAL, ADDITIVE_PAIR)]
    if singles:
        points += [s.Point(SPATIAL, (member,)) for member in ADDITIVE_PAIR]
    monkeypatch.setattr(s, "POINTS", tuple(points))
    monkeypatch.setattr(s, "EPISODES", 4)
    software = SoftwareVersion(
        lock_hash="0" * 16,
        libero_patch_set="1" * 16,
        mujoco_version="3.3.7",
        miopen_db_hash="2" * 16,
    )
    jobs = s.plan(policy_spec_nominal(), software)
    (root / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    outcomes = {}
    for j in jobs:
        i = j.identity
        if len(i.faults) == 2:
            success = i.episode_index == 3
        elif i.faults:
            failure_index = 0 if i.faults[0].kind == "gripper_wear" else 1
            success = i.episode_index != failure_index
        else:
            success = True
        outcomes[i.episode_id] = Outcome(i.episode_id, success, success, success, 10, "fixture")
    monkeypatch.setattr(
        "fleetlab.dose_response.read_run", lambda _: RunOutcomes("fixture", True, outcomes)
    )
    return jobs


def test_missing_singles_are_never_imported_from_other_seed_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup_run(monkeypatch, tmp_path, singles=False)
    joint = s.analyse(tmp_path, tmp_path / "result.json")["joints"][0]
    assert joint["singles"] is None
    assert "single_successes" not in joint and "interaction" not in joint


def test_joint_analysis_uses_four_matched_outcome_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup_run(monkeypatch, tmp_path, singles=True)
    joint = s.analyse(tmp_path, tmp_path / "result.json")["joints"][0]
    assert joint["n"] == 4
    assert joint["healthy_successes"] == 4 and joint["successes"] == 1
    assert joint["single_successes"] == [3, 3]
    assert joint["interaction"][0] == -0.25  # (1 - 3 - 3 + 4) / 4


def test_joint_analysis_rejects_single_with_different_scene_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = setup_run(monkeypatch, tmp_path, singles=True)
    single = next(j for j in jobs if len(j.identity.faults) == 1)
    index = jobs.index(single)
    jobs[index] = single.model_copy(
        update={
            "identity": single.identity.model_copy(
                update={"scene_seed": single.identity.scene_seed + 1}
            )
        }
    )
    (tmp_path / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    with pytest.raises(ValueError, match="not the same attempts"):
        s.analyse(tmp_path, tmp_path / "result.json")
