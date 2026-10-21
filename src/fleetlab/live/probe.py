"""Small X-VLA hook/replay proof, admitted by heavy-run and a latest-start deadline.

This is a plumbing probe, not rescue scoring. It stops on the first call and
checks all recorded decisions via the engine's scripted chain sample. It
also checks non-intervened episodes against their original as-run chains.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts.rollout import Job, SupervisorSpec
from fleetlab.live.data import CALIBRATION_ROBOTS, Dataset
from fleetlab.live.evaluation import first_calls
from fleetlab.live.model import RiskModel


def run_probe(
    data_path: Path, model_path: Path, out: Path, latest_start: datetime
) -> dict[str, Any]:
    if latest_start.tzinfo is None:
        raise ValueError("latest_start needs an explicit timezone")
    if datetime.now(UTC) >= latest_start:
        raise RuntimeError("GPU latest-start deadline passed; no model loaded")
    from fleetlab.runtime import configure_process

    configure_process()
    from fleetlab.engine.chain import Chain, compare
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.log.runfiles import read_identities, read_table

    data = Dataset.load(data_path)
    data = data.subset(data.robot_mask(CALIBRATION_ROBOTS))
    model = RiskModel.load(model_path)
    index = [h.horizon_steps for h in model.heads].index(100)
    scores = model.probabilities(data.x)[:, index]
    threshold = model.heads[index].threshold
    calls = first_calls(data, scores >= threshold)
    alarm = np.flatnonzero(calls >= 0).tolist()
    quiet = np.flatnonzero(calls < 0).tolist()
    chosen = alarm[:2] + quiet[:2]
    if not chosen:
        raise ValueError("no calibration episodes available for the probe")
    runs = {(p["world"], Path(p["run"]).name): Path(p["run"]) for p in data.provenance}
    jobs = []
    sources: list[dict[str, Any]] = []
    keys: set[str] = set()
    day_lengths: set[int] = set()
    for i in chosen:
        e = data.episodes[i]
        if e["id"] in keys:
            raise ValueError("probe episode IDs collide; use one development world at a time")
        keys.add(e["id"])
        source_path = runs[(e["world"], e["run"])]
        record = json.loads((source_path / "run.json").read_text())
        day_lengths.add(int(record["extra"]["episodes_per_robot_day"]))
        identity = next(x for x in read_identities(source_path) if x.episode_id == e["id"])
        jobs.append(
            Job(
                identity=identity,
                supervisor=SupervisorSpec(
                    name="risk",
                    params={
                        "model": model.to_dict(),
                        "horizon_steps": 100,
                        "decision": "stop",
                        "decision_ms": 500,
                    },
                ),
            )
        )
        sources.append(
            {"run": str(source_path), "episode": e["id"], "predicted_first_call": int(calls[i])}
        )
    if len(day_lengths) != 1:
        raise ValueError("probe source runs have different day lengths")
    if len({j.identity.batch_size for j in jobs}) != 1:
        raise ValueError("probe requires a single recorded batch size")
    # Check again immediately before fleet.run loads the model, after input preparation.
    if datetime.now(UTC) >= latest_start:
        raise RuntimeError("GPU latest-start deadline passed; no model loaded")
    result = run(
        RunConfig(
            out=str(out / "run"),
            workers=2,
            batch_size=jobs[0].identity.batch_size,
            precision=jobs[0].identity.policy.precision,
            episodes_per_robot_day=day_lengths.pop(),
            mem_floor_gib=8,
            chain_sample=True,
        ),
        jobs,
    )
    decisions = read_table(out / "run", "decisions").to_pylist()
    for source in sources:
        episode = source["episode"]
        trace = [d for d in decisions if d["episode_id"] == episode]
        actual = trace[0]["step"] if trace else -1
        source["actual_first_call"] = actual
        source["score_matches_hook"] = actual == source["predicted_first_call"]
        if not trace:
            original = Chain.load(Path(source["run"]) / "truth/chain" / f"{episode}.npz")
            watched = Chain.load(out / "run/truth/chain" / f"{episode}.npz")
            source["inactive_chain_identical"] = compare(original, watched).identical
    report = {
        "episodes": sources,
        "chain_ok": result.record.chain_ok,
        "threshold": threshold,
        "supervisor_decision_ms": 500,
        "mode": "fixed stop, no rescue claim",
        "run_record": result.record.model_dump(mode="json"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["chain_ok"] or not all(s["score_matches_hook"] for s in sources):
        raise RuntimeError("probe replay or offline/live score correspondence failed")
    if any(s.get("inactive_chain_identical") is False for s in sources):
        raise RuntimeError("inactive estimator changed an original chain")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latest-start", required=True)
    args = parser.parse_args()
    report = run_probe(args.data, args.model, args.out, datetime.fromisoformat(args.latest_start))
    print(json.dumps({k: v for k, v in report.items() if k != "run_record"}, indent=2))


if __name__ == "__main__":
    main()
