"""The uv settings that keep the build on its own interpreter are in force."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from fleetlab.runtime import repo_root


def test_uv_settings_are_not_shadowed_by_a_uv_toml() -> None:
    # uv ignores [tool.uv] in pyproject.toml as soon as a uv.toml sits beside it.
    assert not (repo_root() / "uv.toml").exists()


def test_uv_is_restricted_to_managed_interpreters() -> None:
    with (repo_root() / "pyproject.toml").open("rb") as file:
        settings = tomllib.load(file)["tool"]["uv"]
    assert settings["python-preference"] == "only-managed"


def test_this_environment_runs_on_a_uv_managed_interpreter() -> None:
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv, "uv is not on PATH"
    managed = subprocess.run(
        [uv, "python", "dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert Path(sys.base_prefix).resolve().is_relative_to(Path(managed).resolve())
