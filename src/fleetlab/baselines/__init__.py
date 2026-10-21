"""Claim-3 simple baselines, all selected from visible exports only."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fleetlab.baselines.common import BaselineAnswer
    from fleetlab.baselines.models import no_data_ranker
    from fleetlab.baselines.random import random_fixes
    from fleetlab.baselines.simple import fixed_prior, worst_health
    from fleetlab.baselines.video import dramatic_videos

_LAZY = {
    "BaselineAnswer": "common",
    "dramatic_videos": "video",
    "fixed_prior": "simple",
    "no_data_ranker": "models",
    "random_fixes": "random",
    "worst_health": "simple",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


__all__ = [
    "BaselineAnswer",
    "dramatic_videos",
    "fixed_prior",
    "no_data_ranker",
    "random_fixes",
    "worst_health",
]
