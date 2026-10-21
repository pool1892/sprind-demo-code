"""The real POSIX timer stops work at the calibration queue budget."""

import json
import signal
import sys
import time
from pathlib import Path

import pytest

from fleetlab.faults import object_calibration


def test_budget_alarm_preserves_previous_completed_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = '{"previous": "validated"}\n'

    def stalled(root: Path, budget_seconds: float, workers: int) -> int:
        (root / "analysis.json").write_text(sentinel)
        time.sleep(1)
        raise AssertionError("budget timer did not stop stalled work")

    prior = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(object_calibration, "run", stalled)
    monkeypatch.setattr(sys, "argv", ["calibration", str(tmp_path), "--budget-seconds", "0.02"])
    assert object_calibration.main() == 0
    assert (tmp_path / "analysis.json").read_text() == sentinel
    receipt = json.loads((tmp_path / "budget-stop.json").read_text())
    assert receipt["elapsed_seconds"] >= 0.02
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == prior
