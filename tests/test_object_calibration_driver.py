"""Budget and precision orchestration with real paired analysis, without inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fault_toys import POLICY, SOFTWARE
from test_dose_response_stage3 import toy_run

from fleetlab import dose_response_stage3 as s3
from fleetlab.contracts.rollout import Job
from fleetlab.faults import object_calibration as driver


class Calibration:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.time = 0.0
        self.calls: list[s3.PlanRun] = []
        self.fail = ""
        self.small_backdrop_effect = False
        monkeypatch.setattr("fleetlab.faults.object_calibration.time.monotonic", lambda: self.time)
        monkeypatch.setattr(s3, "xvla_setup", lambda: s3.EngineSetup(POLICY, SOFTWARE, {}))
        monkeypatch.setattr(s3, "run", self.run)
        monkeypatch.setattr(
            driver, "LEVELS", (("lens_contamination", ("0.75 scene film", "1 scene film")),)
        )

    def outcome(self, job: Job) -> tuple[bool, bool]:
        true = True
        for fault in job.identity.faults:
            if fault.kind == "backdrop_change" and self.small_backdrop_effect:
                true &= job.identity.episode_index % 60 != 0
            if fault.kind == "lens_contamination":
                # The primary level needs 300 pairs; the other is outside 2--15 points.
                divisor = 10 if fault.level == 1 else 5
                true &= job.identity.episode_index % divisor != 0
        return true, true

    def run(self, *, root: Path, workers: int, setup: Any, runs: list[s3.PlanRun]) -> int:
        assert workers <= 3
        assert len(runs) == 1
        item = runs[0]
        self.calls.append(item)
        if item.name == self.fail:
            return 1
        jobs = s3.jobs_for(item, setup.policy, setup.software)
        self.time += len(jobs)  # one second per attempt, entirely virtual
        toy_run(jobs, s3.run_dir(item.name, root), self.outcome)
        return 0


def test_precision_extensions_keep_exact_nonoverlapping_twins_and_moderate_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = Calibration(monkeypatch)
    stub.small_backdrop_effect = True
    assert driver.run(tmp_path, 100000, workers=1) == 0
    by_name = {item.name: item for item in stub.calls}
    assert "object-backdrop_change-600" in by_name
    assert "object-backdrop_change-1000" in by_name
    assert not any(
        "illumination_change-600" in name or "light_color_shift-600" in name for name in by_name
    )
    for kind in (None, "backdrop_change"):
        cohort = [item for item in stub.calls if item.kind == kind]
        assert [(item.first, item.count) for item in cohort] == [(0, 300), (300, 600), (600, 1000)]
        seen: set[str] = set()
        for item in cohort:
            ids = {job.identity.episode_id for job in s3.jobs_for(item, POLICY, SOFTWARE)}
            assert not seen & ids
            seen |= ids
        assert len(seen) == 1000
    primary = s3.labels("lens_contamination", "1 scene film")
    primary_runs = [
        item for item in stub.calls if item.kind == "lens_contamination" and item.points == primary
    ]
    assert [(item.first, item.count) for item in primary_runs] == [(0, 100), (100, 300)]
    second = [
        item for item in stub.calls if item.kind == "lens_contamination" and item.points != primary
    ]
    assert [(item.first, item.count) for item in second] == [(0, 100)]
    result = json.loads((tmp_path / "analysis.json").read_text())
    points = {p["label"]: p for p in result["object"]["lens_contamination"]}
    assert points["1 scene film"]["n"] == 300
    assert points["0.75 scene film"]["n"] == 100
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert [p["name"] for p in receipt["completed"]] == [p.name for p in stub.calls]


@pytest.mark.parametrize(
    "budget, expected", [(479, []), (700, ["object-healthy", "object-illumination_change-300"])]
)
def test_driver_refuses_a_run_whose_forecast_exceeds_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int, expected: list[str]
) -> None:
    stub = Calibration(monkeypatch)
    assert driver.run(tmp_path, budget, workers=1) == 0
    assert [item.name for item in stub.calls] == expected
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert [item["name"] for item in receipt["completed"]] == expected
    assert receipt["elapsed_seconds"] <= budget


def test_failed_run_never_enters_analysis_or_advances_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = Calibration(monkeypatch)
    stub.fail = "object-illumination_change-300"
    with pytest.raises(ValueError, match="failed validation"):
        driver.run(tmp_path, 100000, workers=1)
    assert [p.name for p in stub.calls] == ["object-healthy", stub.fail]
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert set(result["object"]) == {"healthy"}
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert [p["name"] for p in receipt["completed"]] == ["object-healthy"]
