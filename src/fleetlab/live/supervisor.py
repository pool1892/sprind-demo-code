"""Fixed-decision telemetry adapter; subscription model calls are in ``live.vlm``.

Engine SupervisorSpec(name="vlm", params={...}) uses that adapter through
the same contracts.live.Supervisor protocol and engine hook.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts.live import Decision, SupervisorContext, SupervisorDecision
from fleetlab.engine.episode import SupervisorParts
from fleetlab.live.model import RiskModel, TelemetryRiskEstimator


@dataclass(frozen=True)
class RiskRule:
    decision: Decision
    decision_ms: float

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        return SupervisorDecision(
            self.decision, f"telemetry risk {context.risk:.6f}", self.decision_ms, 0.0
        )


def make_risk_rule(params: Mapping[str, Any]) -> SupervisorParts:
    """Embed the model in SupervisorSpec or pin a local JSON file by SHA-256.

    `decision` is required: selecting an estimator does not silently choose
    hand-over versus stop. A real supervisor can reuse the estimator directly.
    """
    if "model" in params:
        if "model_path" in params:
            raise ValueError("supply an embedded model or a pinned path, not both")
        model = RiskModel.from_dict(params["model"])
    else:
        if not params.get("model_sha256"):
            raise ValueError("a risk model path requires model_sha256")
        model = RiskModel.load(Path(params["model_path"]), str(params["model_sha256"]))
    horizon = int(params.get("horizon_steps", 100))
    estimator = TelemetryRiskEstimator(model, horizon)
    threshold = float(params.get("threshold", model.heads[estimator.index].threshold))
    delay = float(params.get("decision_ms", 500))
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("decision_ms must be finite and nonnegative")
    return SupervisorParts(
        estimator,
        RiskRule(Decision(params["decision"]), delay),
        threshold,
        supports_snapshot_start=False,
    )
