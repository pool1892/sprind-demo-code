"""Name the parts of a run by import path, so that a worker process can build them.

A run is configured with strings such as `"fleetlab.sim.libero_sim:LiberoSimEnv"`.
Worker processes are spawned, not forked, and build their environment, their
in-worker policies and their success checks from these names. The same engine
then runs the toy parts of the CPU tests and the real ones.
"""

from __future__ import annotations

import importlib
from typing import Any


def resolve(path: str) -> Any:
    """The object named by `module.path:attribute.path`."""
    module_name, sep, attribute = path.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(f"expected 'module:attribute', got {path!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def build(path: str, *args: Any, **kwargs: Any) -> Any:
    """Call the factory named by `path` with the given arguments."""
    factory = resolve(path)
    if not callable(factory):
        raise TypeError(f"{path} is not callable")
    return factory(*args, **kwargs)
