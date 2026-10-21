"""Reproducible offline development experiment with a frozen robot split."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.live.data import (
    CALIBRATION_ROBOTS,
    FIT_ROBOTS,
    HORIZONS,
    TEST_ROBOTS,
    Dataset,
    concatenate,
)
from fleetlab.live.evaluation import (
    alarm_metrics,
    baseline_fired,
    evaluate,
    forecast_metrics,
    select_baseline,
    select_threshold,
)
from fleetlab.live.features import FEATURE_NAMES
from fleetlab.live.training import fit, with_thresholds


def _population(data: Dataset) -> dict[str, Any]:
    return {
        "episodes": len(data.episodes),
        "command_rows": len(data.steps),
        "failures": sum(e["failure"] for e in data.episodes),
        "robots": sorted({f"{e['world']}/{e['robot']}" for e in data.episodes}),
        "reasons": {
            reason: sum(e["reason"] == reason for e in data.episodes)
            for reason in sorted({e["reason"] for e in data.episodes})
        },
    }


def measure(
    paths: list[Path], mini: Path, out: Path, *, models_out: Path | None = None
) -> dict[str, Any]:
    """Write frozen model/selection before computing any test-set scores."""
    out.mkdir(parents=True, exist_ok=True)
    models_out = out if models_out is None else models_out
    models_out.mkdir(parents=True, exist_ok=True)
    development = concatenate([Dataset.load(path) for path in paths])
    train = development.subset(development.robot_mask(FIT_ROBOTS))
    calibration = development.subset(development.robot_mask(CALIBRATION_ROBOTS))
    test = development.subset(development.robot_mask(TEST_ROBOTS))
    if len(train.steps) + len(calibration.steps) + len(test.steps) != len(development.steps):
        raise ValueError("unassigned development robots")
    if any(not data.episodes for data in (train, calibration, test)):
        raise ValueError("empty robot partition")
    populations = {
        name: _population(data)
        for name, data in (("fit", train), ("calibration", calibration), ("test", test))
    }
    results: dict[str, Any] = {
        "version": 1,
        "label": "terminal failure in (t, t+h]",
        "features": FEATURE_NAMES,
        "populations": populations,
        "provenance": development.provenance,
        "selection": {},
        "scores": {},
    }
    models = {}
    baseline_parameters: dict[str, dict[str, Any]] = {}
    for name in ("telemetry", "time_only"):
        started = time.perf_counter()
        model, fitting = fit(train, calibration, time_only=name == "time_only")
        elapsed = time.perf_counter() - started
        cal_p = model.probabilities(calibration.x)
        selections = [select_threshold(calibration, cal_p[:, i], h) for i, h in enumerate(HORIZONS)]
        model = with_thresholds(model, [s[0] for s in selections])
        path = models_out / f"{name}.json"
        model.save(path)
        models[name] = model
        results["selection"][name] = {
            "fit_seconds": elapsed,
            "fitting": fitting,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "horizons": {
                str(h): {"threshold": s[0], "calibration_alarms": s[1]}
                for h, s in zip(HORIZONS, selections, strict=True)
            },
        }
        diagnostics: dict[str, Any] = {}
        standardized = np.clip((calibration.x - model.mean) / model.scale, -20, 20)
        for i, head in enumerate(model.heads):
            raw = 1 / (
                1 + np.exp(-np.clip(standardized @ head.coefficients + head.intercept, -40, 40))
            )
            diagnostics[str(head.horizon_steps)] = {
                "sigmoid_slope": head.calibration_slope,
                "sigmoid_intercept": head.calibration_intercept,
                "per_robot": [],
            }
            for robot in CALIBRATION_ROBOTS:
                mask = calibration.robot_mask((robot,))
                y = calibration.labels(head.horizon_steps)[mask]
                diagnostics[str(head.horizon_steps)]["per_robot"].append(
                    {
                        "robot": robot,
                        "rows": int(mask.sum()),
                        "observed": float(y.mean()),
                        "raw_mean": float(raw[mask].mean()),
                        "final_mean": float(cal_p[mask, i].mean()),
                        "raw_brier": forecast_metrics(y, raw[mask])["brier"],
                        "final_brier": forecast_metrics(y, cal_p[mask, i])["brier"],
                    }
                )
        results["selection"][name]["calibration_diagnostics"] = diagnostics
    for kind in ("timeout", "stall"):
        parameters: dict[str, Any] = {}
        for h in HORIZONS:
            parameter, alarms = select_baseline(calibration, kind, h)
            fired = baseline_fired(train, kind, parameter)
            y = train.labels(h)
            # Give a rule a probability forecast too: its two rates fitted on training only.
            rates = [
                float((y[fired == flag].sum() + 0.5) / ((fired == flag).sum() + 1))
                for flag in (False, True)
            ]
            parameters[str(h)] = {
                "parameter": parameter,
                "training_rates": rates,
                "calibration_alarms": alarms,
            }
        baseline_parameters[kind] = parameters
        results["selection"][kind] = parameters
    # This receipt is written before test outcomes are scored; no selection below this line.
    (out / "selection.json").write_text(
        json.dumps(
            {k: results[k] for k in ("version", "populations", "selection", "provenance")}, indent=2
        )
        + "\n"
    )
    sets = {"development_held_out": test, "miniature_transfer": Dataset.load(mini)}
    for world in sorted({e["world"] for e in test.episodes}):
        mask = np.asarray([e["world"] == world for e in test.episodes], np.bool_)[test.episode]
        sets[world] = test.subset(mask)
    for name, data in sets.items():
        result: dict[str, Any] = {"population": _population(data), "methods": {}}
        for method, model in models.items():
            p = model.probabilities(data.x)
            result["methods"][method] = {
                str(h): evaluate(data, p[:, i], model.heads[i].threshold, h)
                for i, h in enumerate(HORIZONS)
            }
            if method == "telemetry":
                # Costs are hypothetical ratios, not measured operator/rescue costs.
                result["cost_ratio_sweep"] = {
                    str(threshold): {
                        str(h): alarm_metrics(data, p[:, i] >= threshold, h)
                        for i, h in enumerate(HORIZONS)
                    }
                    for threshold in (0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.75)
                }
        for method, parameters in baseline_parameters.items():
            method_scores = {}
            for h in HORIZONS:
                spec = parameters[str(h)]
                fired = baseline_fired(data, method, spec["parameter"])
                p = np.asarray(spec["training_rates"])[fired.astype(int)]
                metrics = evaluate(data, p, 0.5, h, fired=fired)
                method_scores[str(h)] = metrics
            result["methods"][method] = method_scores
        results["scores"][name] = result
    (out / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def tables(results: dict[str, Any]) -> str:
    """Measured tables for the package README/bundle; all values from the saved JSON."""
    lines = [
        "# Measured forecasts and first-call opportunities",
        "",
        "Generated from [metrics.json](metrics.json); the [selection receipt](selection.json) "
        "records fitting, calibration and provenance. NA means the denominator or both "
        "outcome classes were absent. Lead is seconds from the first possible supervisor call "
        "to the unassisted failure, among caught failures. It is not rescue time.",
        "",
    ]

    def num(value: Any, digits: int = 3) -> str:
        return "NA" if value is None else f"{value:.{digits}f}"

    for name, dataset in results["scores"].items():
        population = dataset["population"]
        lines += [
            f"## {name}",
            "",
            f"{population['episodes']} episodes, {population['failures']} failures, "
            f"{population['command_rows']} active command rows; robots "
            + ", ".join(population["robots"])
            + ".",
            "",
            "| Horizon (s) | Method | Brier | AUROC | Caught / failures | "
            "False calls / episodes | False calls / successes | Median lead (s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for h in HORIZONS:
            for method, scores in dataset["methods"].items():
                score = scores[str(h)]
                a = score["alarms"]
                lines.append(
                    f"| {h / 20:g} | {method} | {num(score['brier'])} | "
                    f"{num(score['auroc'])} | {a['caught']}/{a['failures']} | "
                    f"{a['false_alarms']}/{a['episodes']} | "
                    f"{a['false_alarms']}/{a['successes']} | "
                    f"{num(a['lead_seconds_median'], 2)} |"
                )
        lines += [
            "",
            "### Five-second discrimination within elapsed-time strata",
            "",
            "Aggregate AUROC above includes the predictable terminal timeout. These "
            "strata compare telemetry with elapsed time on the same command rows.",
            "",
            "| Steps | Positive / rows | Telemetry AUROC | Time-only AUROC | "
            "Telemetry Brier | Time-only Brier |",
            "|---|---|---|---|---|---|",
        ]
        telemetry = dataset["methods"]["telemetry"]["100"]["time_strata"]
        clock = dataset["methods"]["time_only"]["100"]["time_strata"]
        for a, b in zip(telemetry, clock, strict=True):
            lines.append(
                f"| {a['first_step']}-{a['last_step_exclusive']} | "
                f"{a['positives']}/{a['rows']} | {num(a['auroc'])} | {num(b['auroc'])} | "
                f"{num(a['brier'])} | {num(b['brier'])} |"
            )
        lines += [
            "",
            "### Telemetry reliability",
            "",
            "Each cell is rows; mean predicted probability; observed failure frequency. "
            "The row populations differ by horizon.",
            "",
            "| Probability bin | 1 s | 2 s | 5 s |",
            "|---|---|---|---|",
        ]
        for i in range(10):
            cells = []
            for h in HORIZONS:
                row = dataset["methods"]["telemetry"][str(h)]["reliability"][i]
                cells.append(f"{row['rows']}; {num(row['predicted'])}; {num(row['observed'])}")
            lines.append(f"| {i / 10:.1f}-{(i + 1) / 10:.1f} | " + " | ".join(cells) + " |")
        lines += [""]
    lines += [
        "## Selected settings",
        "",
        "Calibration robots only; at most 5% of successful episodes called. "
        "Within that constraint, maximize total failure lead divided by all failures. "
        "This selection does not price operator time or prove claim 4.",
        "",
        "| Horizon (s) | Telemetry threshold | Time-only threshold | "
        "Timeout telemetry step | Stall consecutive steps |",
        "|---|---|---|---|---|",
    ]
    selection = results["selection"]
    for h in HORIZONS:
        lines.append(
            f"| {h / 20:g} | {selection['telemetry']['horizons'][str(h)]['threshold']:.2f} | "
            f"{selection['time_only']['horizons'][str(h)]['threshold']:.2f} | "
            f"{selection['timeout'][str(h)]['parameter']} | "
            f"{selection['stall'][str(h)]['parameter']} |"
        )
    return "\n".join(lines) + "\n"
