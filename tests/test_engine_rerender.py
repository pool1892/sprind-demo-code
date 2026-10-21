"""Rendering a recorded episode again from its executed actions, on the toy cell."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from engine_helpers import config, jobs

from fleetlab.contracts.rollout import RunLayout
from fleetlab.engine import fleet, rerender
from fleetlab.engine.chain import Chain

TOY = {
    "env_factory": "fleetlab.engine.toy:ToyEnv",
    "check_factory": "fleetlab.engine.toy:ToyCheck",
    "fault_registry": "fleetlab.engine.toy:TOY_FAULTS",
}


def test_a_toy_episode_renders_again_to_the_same_frames(tmp_path: Path) -> None:
    out = tmp_path / "run"
    fleet.run(config(out), jobs(6))
    for key in ("e000001", "e000002", "e000003"):  # healthy, a camera fault, a delay fault
        result = rerender.rerender(out, key, passes=2, **TOY)
        for p in result["passes"]:
            assert p[key] == {"state": [], "scene": [], "wrist": []}, (key, p)


def test_an_odd_frame_is_named(tmp_path: Path) -> None:
    out = tmp_path / "run"
    fleet.run(config(out), jobs(2))
    layout = RunLayout(out)
    chain = Chain.load(layout.table("chain", "e000001"))
    chain.wrist[5] = "0" * 16  # a frame rendered differently once
    chain.save(layout.table("chain", "e000001.odd"))
    result = rerender.rerender(out, "e000001", against=["e000001.odd"], passes=2, **TOY)
    odd = rerender.odd_one(result)
    assert odd["e000001"]["wrist"]["in_every_pass"] == []
    assert odd["e000001.odd"]["wrist"]["in_every_pass"] == [5]
    assert odd["e000001.odd"]["state"]["in_some_pass"] == []
    assert np.asarray(chain.executed[1]).shape == (7,)
