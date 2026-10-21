"""The second and third round-7 jobs are fixed before admission."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fleetlab.worlds.calibration_round7 import PAIR_STAGES, SITE_STAGES


@pytest.mark.parametrize(
    "job,definitions,maximum", [("b", PAIR_STAGES, 3000), ("c", SITE_STAGES, 3500)]
)
def test_round7_following_plans(
    job: str, definitions: tuple[tuple[Any, ...], ...], maximum: int
) -> None:
    root = Path.home() / f".cache/robots-probes/worlds-v4/20261011-task-cells-r7{job}"
    if not (root / "plan.json").is_file():
        pytest.skip("archived round-7 manifests are not installed on this host")
    sealed = json.loads(Path(__file__).with_name(f"worlds_r7{job}_plan.json").read_text())
    assert json.loads((root / "plan.json").read_text()) == sealed
    assert sealed["budget_seconds"] == 10800
    assert sealed["order"] == [s for row in definitions for s in (row[0], f"{row[0]}-refine")]
    assert sum(row["jobs"] for row in sealed["stages"].values()) == maximum
    for name, row in sealed["stages"].items():
        payload = (root / f"{name}.jsonl").read_bytes()
        text = payload.decode()
        assert len(text.splitlines()) == row["jobs"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
