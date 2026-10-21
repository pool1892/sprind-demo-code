"""torch is loaded before mujoco, by `fleetlab/__init__.py`, without anyone remembering to.

Each case runs in its own interpreter, since the order of imports in a process
cannot be undone. The first tests use an empty stand-in for mujoco, so they
need neither simulator nor GPU; the last uses the real one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPORT_ORDER = (
    "; import sys; loaded = list(sys.modules); "
    "print(loaded.index('torch') < loaded.index('mujoco'))"
)


def run(code: str, stand_in_for_mujoco: Path | None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    if stand_in_for_mujoco is not None:
        (stand_in_for_mujoco / "mujoco.py").write_text("")
        environment["PYTHONPATH"] = str(stand_in_for_mujoco)  # ahead of the installed mujoco
    return subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True
    )


def test_importing_mujoco_after_fleetlab_loads_torch_first(tmp_path: Path) -> None:
    result = run("import fleetlab, mujoco" + REPORT_ORDER, tmp_path)
    assert result.stdout.strip() == "True", result.stderr


def test_fleetlab_alone_does_not_load_torch(tmp_path: Path) -> None:
    # The import costs a second and 0.6 GiB; a process without a simulator is spared it.
    result = run("import fleetlab, sys; print('torch' in sys.modules)", tmp_path)
    assert result.stdout.strip() == "False", result.stderr


def test_mujoco_before_torch_is_refused(tmp_path: Path) -> None:
    for too_late in ("import mujoco, fleetlab", "import mujoco, torch, fleetlab"):
        result = run(too_late, tmp_path)
        assert result.returncode != 0
        assert "mujoco was imported before torch" in result.stderr


def test_torch_before_mujoco_is_accepted_from_anyone(tmp_path: Path) -> None:
    result = run("import torch, mujoco, fleetlab", tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.sim
def test_the_real_simulator_stack_comes_after_torch() -> None:
    # robosuite imports mujoco, which sets up EGL and with it Mesa's driver.
    code = (
        "from fleetlab.runtime import configure_process; configure_process(); "
        "import robosuite" + REPORT_ORDER
    )
    result = run(code, None)
    assert result.stdout.strip().endswith("True"), result.stderr
