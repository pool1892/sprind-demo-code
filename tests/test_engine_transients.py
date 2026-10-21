"""The chain sample's three verdicts (docs/design/contracts.md section 4): identical,
render transient, broken; on synthetic chains and on toy runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from engine_helpers import config, jobs

from fleetlab.engine import fleet, rerender
from fleetlab.engine.chain import BROKEN, IDENTICAL, RENDER_TRANSIENT, Chain, classify


def chain(n: int = 12) -> Chain:
    c = Chain(fixture_hash="ab" * 8)
    for t in range(n):
        executed = None if t == 0 else np.full(7, t, np.float32)
        c.add(t, f"{t:016x}", f"s{t:015x}", f"w{t:015x}", executed)
    return c


def never(step: int, camera: str) -> bool:
    return False


def test_identical_chains_are_identical() -> None:
    check = classify(chain(), chain(), never)
    assert check.verdict == IDENTICAL and check.compared_frames == 24


def test_a_state_or_action_difference_breaks_the_sample() -> None:
    other = chain()
    other.state[4] = "f" * 16
    assert classify(other, chain(), never).verdict == BROKEN
    other = chain()
    other.executed[3] = np.zeros(7, np.float32)
    check = classify(other, chain(), never)
    assert check.verdict == BROKEN and "executed_action" in check.what


def test_one_unconsumed_frame_is_a_candidate_two_frames_or_a_consumed_one_are_not() -> None:
    one = chain()
    one.wrist[5] = "0" * 16
    candidate = classify(one, chain(), never)
    assert candidate.verdict == RENDER_TRANSIENT and candidate.frames == ((5, "wrist"),)
    assert candidate.odd is None
    assert candidate.settled("replay").verdict == RENDER_TRANSIENT
    assert candidate.settled("replay").odd == "replay"
    assert candidate.settled(None, "no").verdict == BROKEN
    consumed = classify(one, chain(), lambda step, camera: step == 5)
    assert consumed.verdict == BROKEN and "consumed" in consumed.what
    two = chain()
    two.wrist[5] = "0" * 16
    two.scene[5] = "0" * 16
    assert classify(two, chain(), never).verdict == BROKEN


def passes(run_bad: list[int], replay_bad: list[int], n: int = 2) -> dict[str, object]:
    clean: dict[str, list[int]] = {"state": [], "scene": []}
    return {
        "key": "e1",
        "against": ["e1.r"],
        "passes": [
            {"e1": {**clean, "wrist": run_bad}, "e1.r": {**clean, "wrist": replay_bad}}
            for _ in range(n)
        ],
    }


def test_the_re_rendering_names_the_odd_one_or_nobody() -> None:
    assert rerender.odd_rendering(passes([], [5]), "e1", "e1.r", (5, "wrist"))[0] == "replay"
    assert rerender.odd_rendering(passes([5], []), "e1", "e1.r", (5, "wrist"))[0] == "run"
    assert rerender.odd_rendering(passes([5], [5]), "e1", "e1.r", (5, "wrist"))[0] is None
    assert rerender.odd_rendering(passes([], [5, 7]), "e1", "e1.r", (5, "wrist"))[0] is None


def run_glitched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *targets: str) -> Any:
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv("FLEETLAB_TEST_GLITCH", ";".join([str(marks), *targets]))
    out = tmp_path / "run"
    cfg = config(out, workers=2, env_factory="engine_helpers:GlitchToyEnv")
    result = fleet.run(cfg, jobs(4))
    record = json.loads((out / "run.json").read_text())
    assert record["chain_ok"] == result.record.chain_ok
    return record


def test_a_toy_run_without_glitches_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = run_glitched(tmp_path, monkeypatch)
    checks = record["extra"]["chain_checks"]
    assert record["chain_ok"] is True and checks["identical"] == 4
    assert checks["render_transient"] == 0 and checks["compared_frames"] > 0


def test_one_glitched_frame_of_a_replay_is_a_render_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # e000002 carries the toy camera fault (a cell hook); the toy policy is called every 5th
    # step, so step 6 is no policy call
    record = run_glitched(tmp_path, monkeypatch, "e000002:6:wrist")
    checks = record["extra"]["chain_checks"]
    assert record["chain_ok"] is True
    assert checks["render_transient"] == 1 and checks["broken"] == 0
    assert checks["transients"] == [
        {"episode_id": "e000002", "key": checks["transients"][0]["key"], "step": 6,
         "camera": "wrist", "odd": "replay"}
    ]  # fmt: skip


def test_a_glitched_frame_a_policy_call_consumed_breaks_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = run_glitched(tmp_path, monkeypatch, "e000002:10:wrist")
    assert record["chain_ok"] is False
    broken = [c for c in record["extra"]["comparisons"] if c["verdict"] == BROKEN]
    assert len(broken) == 1 and "consumed" in broken[0]["what"]


def test_a_second_transient_breaks_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # e000003 carries the toy delay (an action hook): the re-rendering feeds the executed actions
    record = run_glitched(tmp_path, monkeypatch, "e000002:6:wrist", "e000003:3:scene")
    checks = record["extra"]["chain_checks"]
    assert checks["render_transient"] == 2 and checks["broken"] == 0
    assert record["chain_ok"] is False and "2 render transients" in checks["broken_by"]
