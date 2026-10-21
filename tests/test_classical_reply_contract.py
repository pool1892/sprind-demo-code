"""A request refusal is evidence; an unfinished machine run is not an answer."""

from pathlib import Path

import pytest
from test_classical_baseline import write_export
from test_classical_statistics import noisy

from fleetlab.agents.sessions import Analysis
from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.experiments import plan, revise, update
from fleetlab.contracts.tools import ExperimentResult
from fleetlab.toolserver.files import RESULTS_FILE, append_line
from fleetlab.toolserver.public_results import (
    COMPLETED,
    EXECUTION_FAILED,
    PUBLIC_REFUSALS,
)


@pytest.mark.parametrize("message", sorted(PUBLIC_REFUSALS))
def test_public_refusal_keeps_observational_answer_and_spends_request(
    tmp_path: Path, message: str
) -> None:
    data = noisy()
    answer = analyse(data, bootstraps=20)
    export, root = tmp_path / "export", tmp_path / "analysis"
    write_export(export, data)
    plan(export, answer, root, requests=1, episodes_per_arm=1)
    analysis = Analysis(root)
    request = analysis.requests()[0]
    analysis.record_results(
        [
            ExperimentResult(
                experiment_id=request.experiment_id, status="refused", arms=[], message=message
            )
        ]
    )
    if message == EXECUTION_FAILED:
        with pytest.raises(ValueError, match=r"resume experiments.*execution failed"):
            update(export, answer, root)
        return
    updated = update(export, answer, root)
    expected = revise(answer, [], data.costs, requests=1)
    assert updated.diagnosis == expected.diagnosis
    assert updated.fix_set == expected.fix_set
    assert updated.fix_sets == expected.fix_sets
    assert updated.evidence["completed_requests"] == 0
    assert updated.evidence["experiments"] == []
    assert len(analysis.requests()) == 1  # Refusal does not silently erase the request.


@pytest.mark.parametrize("state", ["missing", "queued", "free-text", "empty-done"])
def test_incomplete_or_invalid_reply_cannot_produce_an_answer(tmp_path: Path, state: str) -> None:
    data = noisy()
    answer = analyse(data, bootstraps=20)
    export, root = tmp_path / "export", tmp_path / "analysis"
    write_export(export, data)
    plan(export, answer, root, requests=1, episodes_per_arm=1)
    analysis = Analysis(root)
    if state != "missing":
        result = ExperimentResult(
            experiment_id=analysis.requests()[0].experiment_id,
            status="queued" if state == "queued" else "done",
            arms=[],
            message="untrusted engine detail" if state == "free-text" else COMPLETED,
        )
        # Bypass the writer to test validation of a malformed persisted reply.
        append_line(analysis.results / RESULTS_FILE, result.model_dump(mode="json"))
    with pytest.raises(ValueError, match="resume experiments before updating"):
        update(export, answer, root)
