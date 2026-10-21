"""An experiment reply cannot carry arbitrary host text or partial arm records."""

import json
from pathlib import Path

import pytest

from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.tools import ArmResult, ExperimentResult
from fleetlab.toolserver.files import EXPERIMENTS_FILE, RESULTS_FILE, append_line
from fleetlab.toolserver.public_results import result_problems, results_problems


def refused(experiment: str = "x0001", message: str = "request_not_runnable") -> ExperimentResult:
    return ExperimentResult(experiment_id=experiment, status="refused", arms=[], message=message)


def test_only_fixed_refusal_codes_are_public() -> None:
    assert not result_problems(refused())
    assert result_problems(refused(message="missing truth mapping for hidden_kind"))
    assert result_problems(refused().model_copy(update={"status": "queued"}))
    assert result_problems(
        refused().model_copy(update={"arms": [ArmResult(arm_index=0, episode_ids=["e000001"])]})
    )
    assert results_problems([refused(), refused()])


def test_result_recording_is_atomic_before_any_mount(tmp_path: Path) -> None:
    analysis = Analysis.create("w01", base=tmp_path)
    request = {
        "arms": [{"actions": []}, {"actions": []}],
        "robots": ["r01"],
        "episodes_per_arm": 1,
        "paired_starts": True,
        "power_note": "test",
    }
    for experiment in ("x0001", "x0002"):
        append_line(
            analysis.outbox / EXPERIMENTS_FILE,
            {"experiment_id": experiment, "session": 1, "request": request},
        )
    with pytest.raises(ValueError, match="public status/message"):
        analysis.record_results([refused(), refused("x0002", "hidden detail")])
    assert not (analysis.results / RESULTS_FILE).exists()
    analysis.record_results([refused(), refused("x0002")])
    assert len((analysis.results / RESULTS_FILE).read_text().splitlines()) == 2
    assert "hidden detail" not in json.dumps(analysis.results_by_id(), default=str)


@pytest.mark.parametrize("stored", ["null\n[]\n", '{"x": 1}\n'])
def test_nonobject_or_malformed_results_fail_closed(tmp_path: Path, stored: str) -> None:
    from fleetlab.toolserver.public_results import load_results

    result_file = tmp_path / "results.jsonl"
    result_file.write_text(stored)
    with pytest.raises(ValueError, match="public validation"):
        load_results(result_file, tmp_path / "requests.jsonl", 1, {})


def test_persisted_result_matches_request_and_export(tmp_path: Path) -> None:
    from fleetlab.toolserver.public_results import load_results

    request_file, result_file = tmp_path / "requests.jsonl", tmp_path / "results.jsonl"
    request = {
        "arms": [{"actions": []}, {"actions": []}],
        "robots": ["r01"],
        "episodes_per_arm": 2,
        "paired_starts": True,
        "power_note": "test",
    }
    append_line(request_file, {"experiment_id": "x0001", "session": 1, "request": request})
    with pytest.raises(ValueError, match="public validation"):
        load_results(result_file, request_file, 2, {})
    assert not load_results(result_file, request_file, 1, {})
    result = ExperimentResult(
        experiment_id="x0001",
        status="done",
        message="done",
        arms=[
            ArmResult(arm_index=0, episode_ids=["e000001"]),
            ArmResult(arm_index=1, episode_ids=["e000002"]),
        ],
    )
    append_line(result_file, result.model_dump(mode="json"))
    with pytest.raises(ValueError, match="public validation"):
        load_results(
            result_file, request_file, 2, {"e000001": ("x0001", "r01"), "e000002": ("x0001", "r01")}
        )
    request["episodes_per_arm"] = 1
    request_file.write_text(
        json.dumps({"experiment_id": "x0001", "session": 1, "request": request}) + "\n"
    )
    with pytest.raises(ValueError, match="public validation"):
        load_results(result_file, request_file, 2, {})
    with pytest.raises(ValueError, match="public validation"):
        load_results(
            result_file, request_file, 2, {"e000001": ("x0001", "r02"), "e000002": ("x0001", "r02")}
        )
    assert load_results(
        result_file, request_file, 2, {"e000001": ("x0001", "r01"), "e000002": ("x0001", "r01")}
    ) == [result]


def test_runtime_measurement_uses_valid_refusal_fixture(tmp_path: Path) -> None:
    from fleetlab.agents.measure import ToyEngine
    from fleetlab.agents.sessions import ExperimentQueue, QueuedExperiment
    from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
    from fleetlab.toolserver.public_results import request_results_problems

    request = ExperimentRequest(
        arms=[ExperimentArm(actions=[]), ExperimentArm(actions=[])],
        robots=["r01"],
        episodes_per_arm=4,
        paired_starts=True,
        power_note="test",
    )
    queue = ExperimentQueue(
        world_id="w01",
        analysis_id="test",
        session=1,
        created_at="now",
        export=str(tmp_path),
        experiments=[QueuedExperiment(experiment_id="x0001", session=1, request=request)],
    )
    reply = ToyEngine(tmp_path).run_queue(tmp_path / "queue.json", queue)
    assert reply.results[0].status == "refused"
    assert not request_results_problems(reply.results, {"x0001": request})
