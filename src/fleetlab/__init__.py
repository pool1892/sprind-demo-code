"""Simulated robot fleets with injected faults, and agents that diagnose them."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

# torch has to be loaded before mujoco.
#
# Both bring a copy of LLVM into the process: torch's ROCm libraries bundle
# their own (23), and mujoco, when it sets up EGL on import, loads Mesa's
# graphics driver with the system's (22). The dynamic linker binds a symbol
# that both copies export to whichever was loaded first. With Mesa first,
# about 300 symbols of ROCm's compiler libraries resolve into Mesa's LLVM, and
# the first operation on the GPU fails ("CUDA error: invalid argument") or
# crashes the process. With torch first the mix-up goes the other way, and
# Mesa has tolerated that in every measurement so far.
#
# Importing torch costs a second and 0.6 GiB, so this package does not do it
# for every process that imports it, only at the moment mujoco is asked for.


class _TorchBeforeMujoco(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname == "mujoco":
            import torch  # noqa: F401
        return None  # not found here: the regular finders load mujoco itself


def _torch_was_first() -> bool:
    loaded = list(sys.modules)  # in the order the imports began
    return "torch" in loaded and loaded.index("torch") < loaded.index("mujoco")


if "mujoco" in sys.modules and not _torch_was_first():
    raise ImportError(
        "mujoco was imported before torch, which breaks PyTorch on the GPU in this process. "
        "Import fleetlab (or torch) before mujoco, robosuite, libero or lerobot's environments."
    )
sys.meta_path.insert(0, _TorchBeforeMujoco())
