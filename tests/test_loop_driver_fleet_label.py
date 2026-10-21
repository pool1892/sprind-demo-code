"""The driver checks `--fleet-label` when it reads its arguments, before anything runs.

The learned freeze after the rounds refuses a provenance label that holds an identifier
(`fleetlab.loop.learned.freeze`). Story B's first engagement was launched with the fleet's
public id as its label, ran both rounds and then stopped at that freeze, before its
evaluation (20 October 2026). No model, no simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fleetlab.loop import driver


def test_a_fleet_label_with_an_identifier_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def run_rounds(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"outcome": "completed"}

    monkeypatch.setattr(driver, "run_rounds", run_rounds)
    out = tmp_path / "engagement"
    base = ["rounds", "--world-root", str(tmp_path / "world"), "--out", str(out)]
    with pytest.raises(SystemExit) as refused:
        driver.main([*base, "--fleet-label", "w08"])
    assert refused.value.code == 2
    assert "--fleet-label: 'w08' holds identifiers ['w08']" in capsys.readouterr().err
    assert not calls and not out.exists()
    # A label in words reaches the engagement unchanged, and so does the default.
    assert driver.main([*base, "--fleet-label", "the first fleet"]) == 0
    assert driver.main(base) == 0
    assert [c["fleet_label"] for c in calls] == ["the first fleet", "an earlier fleet"]
