from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fleetlab import runtime


def test_repo_root_is_the_checkout() -> None:
    root = runtime.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "fleetlab" / "runtime.py").is_file()


def test_data_dir_defaults_to_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLEETLAB_DATA_DIR", raising=False)
    assert runtime.data_dir() == runtime.repo_root() / "data"


def test_data_dir_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FLEETLAB_DATA_DIR", str(tmp_path))
    assert runtime.data_dir() == tmp_path.resolve()


def test_libero_config_points_at_the_vendored_package(tmp_path: Path) -> None:
    config_dir = runtime.write_libero_config(tmp_path / "libero")

    lines = (config_dir / "config.yaml").read_text().splitlines()
    paths = {key: Path(value.strip('"')) for key, value in (line.split(": ") for line in lines)}
    assert set(paths) == {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
    vendored = runtime.repo_root() / "third_party" / "libero" / "libero" / "libero"
    assert paths["benchmark_root"] == vendored
    assert (paths["bddl_files"] / "libero_spatial").is_dir()
    assert (paths["init_states"] / "libero_spatial").is_dir()


def test_writing_the_libero_config_does_not_import_libero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Importing LIBERO is what prompts, so the config has to be written without it.
    # Another test may have imported LIBERO already: forget that for the duration.
    monkeypatch.delitem(sys.modules, "libero.libero", raising=False)

    runtime.write_libero_config(tmp_path)

    assert "libero.libero" not in sys.modules


def test_libero_config_is_rewritten_only_when_it_changes(tmp_path: Path) -> None:
    config_file = runtime.write_libero_config(tmp_path) / "config.yaml"
    first_written = config_file.stat().st_mtime_ns
    runtime.write_libero_config(tmp_path)
    assert config_file.stat().st_mtime_ns == first_written

    config_file.write_text("benchmark_root: /somewhere/else\n")
    runtime.write_libero_config(tmp_path)
    assert "third_party" in config_file.read_text()
    assert list(tmp_path.glob("*.tmp")) == []


def test_configure_process_sets_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FLEETLAB_DATA_DIR", str(tmp_path))
    for name in (
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MIOPEN_FIND_MODE",
        "LIBERO_CONFIG_PATH",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)

    runtime.configure_process()

    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert os.environ["MIOPEN_FIND_MODE"] == "2"
    assert Path(os.environ["LIBERO_CONFIG_PATH"], "config.yaml").is_file()
    # The ROCm wheels carry kernels for this chip; pretending to be another one breaks them.
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


def test_gpu_memory_cap_is_a_share_of_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    fractions: list[float] = []
    total = 32 * 2**30
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=total)
    )
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fractions.append)

    monkeypatch.delenv("FLEETLAB_GPU_GIB", raising=False)
    runtime.cap_gpu_memory()
    monkeypatch.setenv("FLEETLAB_GPU_GIB", "12")
    runtime.cap_gpu_memory()
    monkeypatch.setenv("FLEETLAB_GPU_GIB", "64")  # more than there is
    runtime.cap_gpu_memory()

    assert fractions == [8 / 32, 12 / 32, 1.0]


def test_gpu_memory_cap_does_nothing_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", pytest.fail)

    runtime.cap_gpu_memory()
