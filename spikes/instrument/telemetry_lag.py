"""Does ordinary task telemetry separate the lookalike causes? (lane P, step 4)

    uv run --locked python spikes/instrument/telemetry_lag.py OUT.json

Reads the recorded fault-calibration runs of docs/design/faults.md (sections 6 and 8,
`~/.cache/robots-probes/faults/runs/`): X-VLA on libero_spatial, robot r01, 100 episodes
per level with healthy twins, the visible `steps` and `episodes` tables. Light: no
simulation, no policy. Each episode is labelled with its condition from the run's
manifest (host side: the planted fault), and per condition it computes

- the success share (`episodes.outcome`, the fleet's placed check);
- `spikes/adaptation/lag.sql` itself, unchanged, run by DuckDB over the condition's
  episodes (all of them are r01, so its one row per robot is the condition's estimate);
- the per-episode lag of `spikes/adaptation/lag.py` (lags 0 to 40, steps from 40 on,
  episodes of at least 100 commanded steps), its median and quartiles;
- the same lag in an early window (steps 41 to 100) and a late one (steps 181 to 280) of
  the long episodes (at least 280 steps), to see a delay that grows within an episode;
- the residual at the best lag: mean 3-D distance and mean vertical offset (measured z
  minus commanded z), in millimetres, where a sagging arm would show.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

FAULTS = Path.home() / ".cache" / "robots-probes" / "faults"
LAG_SQL = Path(__file__).resolve().parents[1] / "adaptation" / "lag.sql"
RUNS = (
    "s3-healthy",
    "action_latency",
    "s3-controller_overload",
    "s3-controller_overload-300",
    "s3-arm_torque_loss",
    "s3-camera_latency",
    "camera_pose_offset",
    "s3-image_pipeline_release",
    "s3-calibration_update",
)
MAX_LAG = 40
MIN_STEPS = 100
EARLY = (41, 100)
LATE = (181, 280)


def label_of(faults: list[dict[str, Any]]) -> str:
    if not faults:
        return "healthy"
    (f,) = faults
    kind, level, target, params = f["kind"], f["level"], f["target"], f["params"]
    where = target["component"] or target["config_key"]
    extra = ",".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "value")
    return f"{kind} {level:g} {where}" + (f" {extra}" if extra else "")


def labels(run: str) -> dict[str, str]:
    out = {}
    for line in (FAULTS / "manifests" / f"{run}.jsonl").read_text().splitlines():
        if line.strip():
            identity = json.loads(line)["identity"]
            out[identity["episode_id"]] = label_of(identity["faults"])
    return out


def curve(cmd: np.ndarray, eef: np.ndarray, first: int, last: int) -> np.ndarray:
    """Mean distance per lag between the command `lag` steps before the step that produced
    each measurement and the measurement, scored on steps first..last (lag.py's pairing)."""
    steps = np.arange(first, min(last, len(cmd) - 1) + 1)
    return np.asarray(
        [np.linalg.norm(cmd[steps - lag] - eef[steps], axis=1).mean() for lag in range(MAX_LAG + 1)]
    )


def episode(path: Path) -> dict[str, Any] | None:
    table = pq.read_table(path, columns=["step", "eef_pos", "commanded_action"]).to_pydict()
    commanded = [c for c in table["commanded_action"] if c is not None]
    if len(commanded) < MIN_STEPS:
        return None
    cmd = np.asarray([c[:3] for c in commanded], np.float64)
    eef = np.asarray(table["eef_pos"][1 : len(commanded) + 1], np.float64)  # after each step
    whole = curve(cmd, eef, MAX_LAG, len(cmd) - 1)
    lag = int(np.argmin(whole))
    steps = np.arange(MAX_LAG, len(cmd))
    dz = (eef[steps, 2] - cmd[steps - lag, 2]).mean()
    out: dict[str, Any] = {
        "steps": len(cmd),
        "lag": lag,
        "distance_mm": float(whole[lag] * 1000),
        "dz_mm": float(dz * 1000),
    }
    if len(cmd) >= LATE[1]:
        out["early_lag"] = int(np.argmin(curve(cmd, eef, *EARLY)))
        out["late_lag"] = int(np.argmin(curve(cmd, eef, *LATE)))
    return out


def lag_sql(con: Any, steps: list[str], ids: list[str], episodes_file: Path) -> dict[str, Any]:
    """lag.sql, unchanged, over the given episodes."""
    files = ", ".join(f"'{p}'" for p in steps)
    id_list = ", ".join(f"'{i}'" for i in ids)
    con.execute(f"CREATE OR REPLACE VIEW steps AS SELECT * FROM read_parquet([{files}])")
    con.execute(
        f"CREATE OR REPLACE VIEW episodes AS SELECT * FROM read_parquet('{episodes_file}') "
        f"WHERE episode_id IN ({id_list})"
    )
    rows = con.execute(LAG_SQL.read_text()).fetchall()
    return {"rows": [list(r) for r in rows]}


def quartiles(values: list[float]) -> list[float]:
    return [float(q) for q in np.percentile(values, [25, 50, 75])] if values else []


def main() -> int:
    import duckdb

    out_file = Path(sys.argv[1])
    con = duckdb.connect()
    con.execute("SET threads = 2")  # a builder's share of the machine
    result: dict[str, Any] = {}
    for run in RUNS:
        run_dir = FAULTS / "runs" / run
        by_label: dict[str, list[str]] = {}
        for eid, label in labels(run).items():
            by_label.setdefault(label, []).append(eid)
        outcomes = {
            r["episode_id"]: r["outcome"]
            for r in pq.read_table(run_dir / "log" / "episodes.parquet").to_pylist()
        }
        for label, ids in sorted(by_label.items()):
            ids = [i for i in ids if (run_dir / "log" / "steps" / f"{i}.parquet").is_file()]
            per = [episode(run_dir / "log" / "steps" / f"{i}.parquet") for i in ids]
            scored = [p for p in per if p is not None]
            long = [p for p in scored if "late_lag" in p]
            entry = result.setdefault(
                label, {"runs": [], "episodes": 0, "successes": 0, "per_episode": []}
            )
            entry["runs"].append(run)
            entry["episodes"] += len(ids)
            entry["successes"] += sum(outcomes.get(i) == "success" for i in ids)
            entry["per_episode"].extend(scored)
            entry.setdefault("lag_sql", []).append(
                lag_sql(
                    con,
                    [str(run_dir / "log" / "steps" / f"{i}.parquet") for i in ids],
                    ids,
                    run_dir / "log" / "episodes.parquet",
                )
            )
            print(
                f"{run} {label}: {len(ids)} episodes, {len(scored)} scored, {len(long)} long",
                flush=True,
            )
    summary: dict[str, Any] = {}
    for label, entry in result.items():
        per = entry["per_episode"]
        long = [p for p in per if "late_lag" in p]
        lags = [p["lag"] for p in per]
        summary[label] = {
            "runs": entry["runs"],
            "episodes": entry["episodes"],
            "successes": entry["successes"],
            "lag_sql": entry["lag_sql"],
            "scored": len(per),
            "lag_quartiles": quartiles(lags),
            "lag_mode_share": (max(lags.count(v) for v in set(lags)) / len(lags)) if lags else None,
            "distance_mm_median": statistics.median(p["distance_mm"] for p in per) if per else None,
            "dz_mm_median": statistics.median(p["dz_mm"] for p in per) if per else None,
            "long": len(long),
            "early_lag_quartiles": quartiles([p["early_lag"] for p in long]),
            "late_lag_quartiles": quartiles([p["late_lag"] for p in long]),
        }
    out_file.write_text(
        json.dumps(
            {"summary": summary, "per_episode": {k: v["per_episode"] for k, v in result.items()}},
            indent=1,
        )
        + "\n"
    )
    for label, s in summary.items():
        print(label, json.dumps({k: v for k, v in s.items() if k not in ("runs",)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
