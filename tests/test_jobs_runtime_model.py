"""The jobs runtime with X-VLA on the GPU (`make check-model`, heavy).

The naive chain of job J1 (the cream cheese into the bowl, then the bowl onto the plate),
played by the runtime from development start 0, reproduces lane S's spike run of the same
chain on the same start bit for bit: the same end state hash, step ends and verdict
(`spikes/jobs/runner.py`, batch naive10, 19 October 2026, J1 naive s00; the starts'
seeds are the spike's, `fleetlab.jobs.starts`). Needs the pinned convolution database
(`FLEETLAB_MIOPEN_DIR`, `python -m fleetlab.engine.miopen pin`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleetlab.contracts.jobs import RequestState

pytestmark = pytest.mark.model

J1 = RequestState.model_validate(
    {
        "request_text": "the cream cheese in the bowl, then the bowl on the plate",
        "requested_goals": [
            {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
            {"predicate": "On", "objects": ["akita_black_bowl_1", "plate_1"]},
        ],
        "base_task": 6,
        "order_rules": [{"first": 0, "then": 1}],
        "budget": {
            "starts_per_request": 20,
            "requests_per_session": 4,
            "sessions": 6,
            "starts_per_investigation": 400,
        },
        "acceptance": None,
        "message_log": [],
    }
)
# What the spike recorded for J1 naive on initial state 0 (summary.json of s00).
SPIKE_S00 = {"end_hash": "4628b30cfad97684", "steps": 208, "ends": [92, 167], "tail": 41}


def test_the_naive_chain_reproduces_the_spike_s_run_bit_for_bit(tmp_path: Path) -> None:
    from fleetlab.jobs.runner import JobRunner, load_xvla, naive_program
    from fleetlab.jobs.starts import development_starts

    source, soft = load_xvla()
    runner = JobRunner(source, soft)
    try:
        naive = naive_program(J1.requested_goals, runner.suite.instructions())
        [episode] = runner.run_starts(
            tmp_path / "run",
            "j0001",
            naive,
            J1,
            development_starts(6, 1),
            record_video=False,
            first_episode=101,
        )
    finally:
        runner.close()
    assert episode.end_state_hash == SPIKE_S00["end_hash"]
    assert episode.control_steps == SPIKE_S00["steps"]
    assert [r.end_control_step for r in episode.program_steps] == SPIKE_S00["ends"]
    assert episode.tail_steps == SPIKE_S00["tail"] and episode.delivered and episode.order_kept
