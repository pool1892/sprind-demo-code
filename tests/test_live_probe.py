"""The queue deadline must prevent model loading even with invalid input paths."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fleetlab.live.probe import run_probe


def test_expired_gpu_deadline_is_checked_before_inputs_or_model_loading() -> None:
    with pytest.raises(RuntimeError, match="latest-start deadline"):
        run_probe(
            Path("missing"), Path("missing"), Path("unused"), datetime(2000, 1, 1, tzinfo=UTC)
        )
