"""A worker imports the engine before the fault registry in a fresh interpreter."""

import os
import subprocess
import sys
from pathlib import Path


def test_engine_import_does_not_reenter_a_partially_initialized_episode() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import fleetlab.engine.fleet"],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
