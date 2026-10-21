"""Reproducible planted-fleet study. No stored exports or oracle are read.

Run: PYTHONPATH=tests uv run python tests/classical_planted.py NEW_RESULT.json
300 independent panels per contrast; 30 full analyses per onset/effect case.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from test_classical_statistics import noisy

from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.analysis import _onset_evidence, fleet_diagnostic
from fleetlab.baselines.classical.data import Row, VisibleData
from fleetlab.baselines.classical.experiments import choose
from fleetlab.baselines.classical.signals import stratified_contrast
from fleetlab.contracts.actions import catalogue
from fleetlab.contracts.model import Site
from fleetlab.contracts.vocabulary import ConfigKey


def planted(drop: float, onset: int, seed: int, local: bool = False) -> VisibleData:
    data = noisy()
    template = data.fleet.robots()[0]
    robots = [
        template.model_copy(update={"id": f"r{i + 1:02}", "site_id": f"s{i // 4 + 1}"})
        for i in range(12)
    ]
    sites = [Site(id=f"s{i + 1}", robots=robots[4 * i : 4 * i + 4]) for i in range(3)]
    fleet = data.fleet.model_copy(update={"sites": sites, "change_log": []})
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    health: list[Row] = []
    for robot in robots:
        for day in range(6):
            for i in range(14):
                probability = (0.85 if i % 2 else 0.65) - drop * (day >= onset) * (
                    not local or robot.id in {"r01", "r05", "r09"}
                )
                success = rng.random() < probability
                rows.append(
                    dict(
                        episode_id=f"e{len(rows) + 1:06}",
                        robot_id=robot.id,
                        site_id=robot.site_id,
                        day=day,
                        task_text="easy" if i % 2 else "hard",
                        outcome="success" if success else "failure",
                        steps=int(rng.integers(90, 230)) if success else 280,
                        experiment_id=None,
                    )
                )
            health.append(
                dict(
                    robot_id=robot.id,
                    component="scene_camera",
                    metric="calibration_residual",
                    day=day,
                    value=float(20 + rng.normal(0, 2)),
                )
            )
    changes = [
        dict(
            day=onset,
            scope="fleet",
            key=ConfigKey.software_version,
            old="v0",
            new="v1",
            author="ops",
        )
    ]
    if local:
        changes += [
            dict(
                day=onset,
                scope=r,
                key=ConfigKey.control_action_delay_steps,
                old=0,
                new=20,
                author="ops",
            )
            for r in ("r01", "r05", "r09")
        ]
    return replace(
        data,
        fleet=fleet,
        episodes=rows,
        health=health,
        changes=changes,
        events=[],
        actions=catalogue(fleet),
    )


def study(onsets: tuple[int, ...] = (1, 3, 5)) -> list[dict[str, object]]:
    records = []
    for mode in ("fleet", "logged-local", "unlogged-local"):
        for onset in onsets:
            for drop in (0.4,) if mode != "fleet" else (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                significant, retained, heterogeneous = 0, 0, 0
                fits = []
                for seed in range(300):
                    data = planted(drop, onset, seed + 1000, mode != "fleet")
                    if mode == "unlogged-local":
                        data.changes[:] = data.changes[:1]
                    rows = [
                        {**r, "task_text": r["site_id"] + ":" + r["task_text"]}
                        for r in data.episodes
                    ]
                    z, _ = stratified_contrast(
                        rows,
                        [r["day"] >= onset for r in rows],
                        [float(r["outcome"] == "success") for r in rows],
                    )
                    significant += z < -1.6448536269514722
                    retained += _onset_evidence(data, data.changes[:1], [])[0]
                    heterogeneous += fleet_diagnostic(data, onset)["heterogeneity_p"] < 0.05
                    if seed < 30:
                        answer = analyse(data, bootstraps=200)
                        estimate = next(
                            r
                            for r in answer.evidence["actions"]
                            if r["action"]["target"]["scope"] == "fleet"
                        )
                        fits.append(
                            dict(
                                seed=seed + 1000,
                                gain_points=estimate["gain_points"],
                                interval=[estimate["low"], estimate["high"]],
                                diagnosed=any(
                                    c.target.scope == "fleet" for c in answer.diagnosis.causes
                                ),
                                informative=estimate["informative_interval"],
                                experiment_reserved=any(
                                    a.target.scope == "fleet" for a in choose(answer)
                                ),
                            )
                        )
                row = dict(
                    mode=mode,
                    onset=onset,
                    drop_points=drop * 100,
                    draws=300,
                    pooled_one_sided_significant=significant,
                    retained=retained,
                    heterogeneous=heterogeneous,
                    full_fits=len(fits),
                    bootstrap_draws=200,
                    diagnosed=sum(f["diagnosed"] for f in fits),
                    positive_estimates=sum(f["gain_points"] > 0 for f in fits),
                    reserved=sum(f["experiment_reserved"] for f in fits),
                    fits=fits,
                )
                assert row["retained"] == 300 and row["reserved"] == len(fits)
                records.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "fits"}), flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--onset", type=int, choices=(1, 3, 5))
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(study((args.onset,) if args.onset else (1, 3, 5)), indent=2) + "\n"
    )
