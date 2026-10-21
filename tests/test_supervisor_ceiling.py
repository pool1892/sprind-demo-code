"""The ceiling trigger cannot drift later or bypass physical eligibility."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_vlm_supervisor import observation

from fleetlab.contracts.live import Decision
from fleetlab.live.vlm.ceiling import factory


@pytest.mark.parametrize("step", [120, 160, 200])
def test_ceiling_is_one_exact_eligible_retry(step: int) -> None:
    parts = factory("retry-ceiling", {"step": step})
    assert parts.failure_fallback is None
    assert parts.should_call is not None
    c = replace(observation(), active_step=step, options=[Decision.retry, Decision.continue_])
    assert not parts.should_call(replace(c, active_step=step - 1))
    assert not parts.should_call(replace(c, active_step=step + 1))
    assert parts.should_call(c)
    answer = parts.supervisor.decide(c)
    assert answer.decision is Decision.retry
    assert answer.decision_ms == answer.model_ms == 0
    assert not parts.should_call(replace(c, previous_decisions=(Decision.retry,)))
    assert (
        parts.supervisor.decide(replace(c, options=[Decision.continue_])).decision
        is Decision.continue_
    )


def test_ceiling_union_does_not_double_count_or_credit_an_unretried_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace

    from fleetlab.live.vlm import ceiling

    successes = {120: {"a", "b"}, 160: {"a", "b"}, 200: {"c"}}
    retries = {120: {"a"}, 160: {"a", "b"}, 200: {"c"}}
    for step in ceiling.STEPS:
        root = tmp_path / str(step)
        root.mkdir()
        (root / "run.json").write_text(
            json.dumps(
                {
                    "chain_ok": True,
                    "extra": {"total_s": 1, "jobs_detail": [{"mode": "run", "robot_time_s": 2}]},
                }
            )
        )

    def table(root: Path, name: str) -> Any:
        step = int(root.name)
        if name == "job_outcomes":
            rows = [
                {"episode_id": e, "mode": "run", "success": e in successes[step]}
                for e in ["a", "b", "c"]
            ]
        else:
            rows = [{"episode_id": e, "decision": "retry"} for e in retries[step]]
        return SimpleNamespace(to_pylist=lambda: rows)

    monkeypatch.setattr(ceiling, "read_table", table)
    result = ceiling.summarize(tmp_path, {"a", "b", "c"})
    assert [arm["rescued"] for arm in result["arms"].values()] == [1, 2, 1]
    assert result["best_of_three_rescued"] == 3
    assert result["arms"]["120"]["success_without_retry_ids"] == ["b"]
    assert result["arms"]["120"]["ineligible_ids"] == ["b", "c"]


def test_ceiling_refuses_an_arm_with_a_failed_replay_proof(tmp_path: Path) -> None:
    import json

    from fleetlab.live.vlm import ceiling

    root = tmp_path / str(ceiling.STEPS[0])
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"chain_ok": False}))
    with pytest.raises(RuntimeError, match="replay proof failed"):
        ceiling.summarize(tmp_path, {"a"})
    assert not (tmp_path / "ceiling.json").exists()


def test_ceiling_runs_only_the_60_original_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from engine_helpers import ident

    from fleetlab.contracts.rollout import Job
    from fleetlab.engine import manifests
    from fleetlab.live.vlm import ceiling
    from fleetlab.live.vlm.measure import dump

    jobs = [Job(identity=ident(i)) for i in range(1, 81)]
    sample = tmp_path / "sample"
    manifests.write(sample / "sample.jsonl", jobs)
    dump(
        sample / "selection.json",
        {
            "episodes_per_robot_day": 100,
            "episodes": {j.identity.episode_id: {"success": i >= 60} for i, j in enumerate(jobs)},
        },
    )
    expected = {j.identity.episode_id for j in jobs[:60]}
    runs = []
    monkeypatch.setattr(ceiling, "check_sample", lambda _: None)

    def run(cfg: Any, watched: Any) -> Any:
        assert len(watched) == 60
        assert {j.identity.episode_id for j in watched} == expected
        assert not any(j.record_video for j in watched)
        runs.append([j.supervisor.params["step"] for j in watched])
        return SimpleNamespace(record=SimpleNamespace(chain_ok=True))

    def summarize(out: Path, ids: set[str]) -> Any:
        assert ids == expected
        return {}

    monkeypatch.setattr(ceiling, "run", run)
    monkeypatch.setattr(ceiling, "summarize", summarize)
    ceiling.measure(sample, tmp_path / "out")
    assert runs == [[step] * 60 for step in ceiling.STEPS]
