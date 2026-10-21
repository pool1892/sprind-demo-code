"""Process setup: where data lives and which environment variables the stack needs.

Call `configure_process()` at the start of every entry point, before torch,
mujoco, robosuite or libero are imported. They read these variables on import.
The order of those imports matters as well; `fleetlab/__init__.py` takes care
of it.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


def repo_root() -> Path:
    """The checkout this package runs from. The package is always installed editable."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "fleetlab").is_dir():
            return parent
    raise RuntimeError("fleetlab is not running from a checkout of its repository")


def data_dir() -> Path:
    """Generated data: `data/` in the checkout, ignored by git. `FLEETLAB_DATA_DIR` overrides."""
    override = os.environ.get("FLEETLAB_DATA_DIR")
    return Path(override).resolve() if override else repo_root() / "data"


def write_libero_config(config_dir: Path | None = None) -> Path:
    """Write LIBERO's `config.yaml` and return the directory that holds it.

    LIBERO asks for input on standard input when it is imported and finds no
    config file. Writing the file first avoids the prompt. The paths point into
    the installed package, so the file is rewritten whenever they change.
    """
    spec = importlib.util.find_spec("libero.libero")  # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("the libero package is not installed; run `make setup`")
    package = Path(next(iter(spec.submodule_search_locations))).resolve()

    config_dir = config_dir or data_dir() / "libero"
    paths = {
        "benchmark_root": package,
        "bddl_files": package / "bddl_files",
        "init_states": package / "init_files",
        "datasets": config_dir / "datasets",
        # Where LIBERO itself downloads meshes and textures on first use. Nothing reads this key.
        "assets": Path.home() / ".cache" / "libero" / "assets",
    }
    # A JSON string is a valid YAML scalar.
    content = "".join(f"{key}: {json.dumps(str(path))}\n" for key, path in paths.items())

    config_file = config_dir / "config.yaml"
    if not config_file.is_file() or config_file.read_text() != content:
        config_dir.mkdir(parents=True, exist_ok=True)
        temporary = config_file.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(content)
        temporary.replace(config_file)  # atomic: parallel processes write the same content
    return config_dir


def configure_process() -> None:
    """Set the environment variables for headless simulation and GPU inference."""
    # Render off-screen through EGL. There is no display to open a window on.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # MIOpen's default search for the fastest convolution takes minutes on first use.
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    os.environ["LIBERO_CONFIG_PATH"] = str(write_libero_config())


def cap_gpu_memory(gib: float = 8.0) -> None:
    """Keep PyTorch in this process from taking more than `gib` of GPU memory.

    The GPU shares the system's memory, and the ceiling `tools/heavy-run` puts
    on a job does not count what the job holds on the GPU. Every process that
    loads a model therefore limits itself. The default suits X-VLA and
    SmolVLA, which stay below 5 GiB with up to ten environments;
    `FLEETLAB_GPU_GIB` overrides it. The simulator's renderers are not
    PyTorch's memory and come on top, about 0.6 GiB each.
    """
    import torch

    if torch.cuda.is_available():  # nothing to limit when a run is on the CPU
        gib = float(os.environ.get("FLEETLAB_GPU_GIB", gib))
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2**30 / total))
