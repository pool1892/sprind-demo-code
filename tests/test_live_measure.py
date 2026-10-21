"""A complete offline experiment on labelled synthetic episodes, including its receipts."""

from __future__ import annotations

import json
from pathlib import Path

from test_live_model import dataset

from fleetlab.live.data import concatenate
from fleetlab.live.measure import measure, tables
from fleetlab.live.model import RiskModel


def test_measure_freezes_selection_and_emits_finite_reproducible_metrics(tmp_path: Path) -> None:
    samples = []
    for robot in range(1, 13):
        data = dataset(f"r{robot:02d}")
        for episode in data.episodes:
            episode["id"] = f"{episode['robot']}-{episode['id']}"
            episode["reason"] = "step limit" if episode["failure"] else "placed"
        samples.append(data)
    development = concatenate(samples)
    path = tmp_path / "development.npz"
    mini = tmp_path / "transfer.npz"
    development.save(path)
    transfer = samples[0]
    for episode in transfer.episodes:
        episode["world"] = "another_world"
    transfer.save(mini)
    out = tmp_path / "results"
    models = tmp_path / "runtime-models"
    results = measure([path], mini, out, models_out=models)
    json.dumps(results, allow_nan=False)
    assert results["populations"]["fit"]["episodes"] == 12
    assert results["populations"]["calibration"]["episodes"] == 6
    assert results["populations"]["test"]["episodes"] == 6
    assert (out / "selection.json").is_file()
    assert not (out / "telemetry.json").exists()
    model = RiskModel.load(models / "telemetry.json")
    assert [h.horizon_steps for h in model.heads] == [20, 40, 100]
    for group in results["scores"].values():
        for method in group["methods"].values():
            for score in method.values():
                assert sum(r["rows"] for r in score["reliability"]) == score["rows"]
                if score["alarms"]["caught"] == 0:
                    assert score["robot_bootstrap_95"]["intervals"]["failure_recall"] == [0, 0]
    assert "miniature_transfer" in tables(results)
