"""Persist each completed human trajectory's accounting inside its worker.

Unlike the fleet's end-of-phase summary, this survives interruption between
completed jobs and the final run.json write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleetlab.contracts.policy import Action7
from fleetlab.policies.teleop.human import HumanTeleoperator


class RecordedHuman(HumanTeleoperator):
    def configure_takeover(
        self, params: dict[str, Any] | None, last_commanded: Action7 | None
    ) -> None:
        data = dict(params or {})
        self.metrics_path = Path(data.pop("metrics_path"))
        super().configure_takeover(data, last_commanded)

    def metrics(self) -> dict[str, Any]:
        value = super().metrics()
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.metrics_path)
        return value
