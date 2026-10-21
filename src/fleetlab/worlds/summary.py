"""What a world measured, against what its generator expected, and gate 3's marks.

Read from the world's finished runs (visible outcomes and the truth side's verdicts) and,
for a version-1 world, from its detectability list. The gate-3 marks:

- the faulted fleet's success over the observed window, on the outcome the fleet logs,
  between 60 and 75% (measured);
- at least six detectable causes of two points each in the observed window
  (generation, `worlds.detect`;
  counted only when their calibrated points are effects beyond noise);
- the four patterns present (confounder, decoy, interaction, drift; their confirmation
  is the oracle's);
- the catalogue of actions at least ten times the number of active faults;
- the export's leak audit clean, and both runs' chain samples held.

`rate comparison` per cause: the logged success of the observed episodes that carry it,
against the success the generator expected for those same episodes. It is a check of
the curves on this world's episodes, not an effect estimate (the episodes differ from
the others in robot, site and task mix).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fleetlab.worlds.generate import World
from fleetlab.worlds.layout import WorldPaths

BAND = (0.60, 0.75)
MIN_DETECTABLE = 6
CATALOGUE_RATIO = 10


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {
        "ok": k,
        "n": n,
        "rate": round(k / n, 4) if n else None,
        "interval": [round(lo, 4), round(hi, 4)],
    }


def outcomes(paths: WorldPaths) -> dict[str, dict[str, Any]]:
    """Per as-run episode: identity fields, logged and true outcome, steps."""
    from fleetlab.log.runfiles import read_identities, read_table

    out: dict[str, dict[str, Any]] = {}
    for run_dir in paths.runs():
        idents = {i.episode_id: i for i in read_identities(run_dir)}
        verdicts = {r["episode_id"]: r for r in read_table(run_dir, "verdicts").to_pylist()}
        for row in read_table(run_dir, "episodes").to_pylist():
            ident = idents.get(row["episode_id"])
            if ident is None:
                continue
            out[row["episode_id"]] = {
                "identity": ident,
                "logged": row["outcome"] == "success",
                "true": bool(verdicts.get(row["episode_id"], {}).get("true_success", False)),
                "steps": row["steps"],
            }
    return out


def summary(world: World, paths: WorldPaths) -> dict[str, Any]:
    """The world's measured numbers, the generator's expectations, and the gate-3 marks."""
    from fleetlab.contracts.actions import catalogue

    spec = world.spec
    seen = outcomes(paths)
    ranges = {
        "observed": spec.windows.observed,
        "experiment": spec.windows.experiment,
        "scoring": spec.windows.scoring,
    }
    measured: dict[str, Any] = {}
    for name, days in ranges.items():
        rows = [o for o in seen.values() if o["identity"].day in days]
        measured[name] = {
            "logged": _rate(sum(o["logged"] for o in rows), len(rows)),
            "true": _rate(sum(o["true"] for o in rows), len(rows)),
        }
    by: dict[str, dict[str, list[int]]] = {"site": {}, "robot": {}, "day": {}}
    for o in seen.values():
        ident = o["identity"]
        for key, value in (("site", ident.site_id), ("robot", ident.robot_id), ("day", ident.day)):
            cell = by[key].setdefault(str(value), [0, 0])
            cell[0] += o["logged"]
            cell[1] += 1
    clean = [o for o in seen.values() if not o["identity"].faults]
    result: dict[str, Any] = {
        "episodes": len(seen),
        "measured": measured,
        "by": {k: {v: _rate(*c) for v, c in sorted(cells.items())} for k, cells in by.items()},
        "without_any_fault": {
            "logged": _rate(sum(o["logged"] for o in clean), len(clean)),
            "true": _rate(sum(o["true"] for o in clean), len(clean)),
        },
    }
    detect = getattr(world, "detect", None)
    if detect is not None:
        result["expected"] = detect.window_success
        result["predictions"] = getattr(world, "predictions", {})
        result["causes"] = _causes(world, seen)
    observed = measured["observed"]["logged"]
    faults = len(spec.fault_schedule)
    entries = len(catalogue(world.fleet, world.task_classes))
    record = paths.read_record().get("steps", {})
    chain = {
        name: record.get(name, {}).get("chain_ok")
        for name in ("observed", "later")
        if record.get(name, {}).get("chain_ok") is not None
    }
    marks: dict[str, Any] = {
        "observed_success_in_band": {
            "value": observed["rate"],
            "band": list(BAND),
            "met": observed["rate"] is not None and BAND[0] <= observed["rate"] <= BAND[1],
        },
        "patterns_present": {
            "confounders": len(spec.patterns.confounders),
            "decoys": len(spec.patterns.decoys),
            "interactions": len(spec.patterns.interactions),
            "additive_pairs": len(spec.patterns.additive_pairs),
            "drifts": len(spec.patterns.drifts),
            "met": all(
                (
                    spec.patterns.confounders,
                    spec.patterns.decoys,
                    spec.patterns.interactions,
                    spec.patterns.drifts,
                )
            ),
            "confirmed_by_the_oracle": "pending (robots-75s.4)",
        },
        "catalogue_ratio": {
            "entries": entries,
            "faults": faults,
            "ratio": round(entries / max(1, faults), 1),
            "met": entries >= CATALOGUE_RATIO * faults,
        },
        "leak_audit_clean": record.get("export", {}).get("leaks_clean"),
        "chain_ok": chain,
    }
    if detect is not None:
        counted = detect.counted()
        observed_counted = detect.observed_counted()
        marks["detectable_causes"] = {
            "counted": len(counted),
            "by_rule": sum(1 for u in detect.units if u.detectable),
            "qualified_by_observed": sum(1 for u in counted if "observed" in u.qualified_by),
            "wanted": MIN_DETECTABLE,
            "observed_counted": len(observed_counted),
            "met": len(observed_counted) >= MIN_DETECTABLE,
        }
    composed = getattr(getattr(world, "composition", None), "expected", {})
    if "window_forecasts" in composed:
        # The forecast the composition was accepted and sealed on: every window with its
        # interval under the retained error the recipe names. `predictions` above are the
        # world module's and always carry the third generation's error.
        result["forecast"] = {
            "windows": composed["window_forecasts"],
            "seal": composed["forecast_seal"],
            "retained_error": getattr(getattr(world, "recipe", None), "retained_error", None),
        }
    forecast = composed.get("measured_cause_forecast")
    if forecast is not None:
        result["measured_cause_forecast"] = forecast
        marks["detectable_causes"] = {
            "forecast": forecast,
            "met": None,
            "confirmation": "pending measured scoring-window oracle removals",
        }
    result["gate3"] = marks
    return result


def _causes(world: World, seen: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per cause: the observed carriers' logged success against the expected for them."""
    from fleetlab.worlds import estimate
    from fleetlab.worlds.world import WorldV1, context

    if not isinstance(world, WorldV1):
        return []
    ctx = context(world.composition, _curves(world))
    days = world.spec.windows.observed
    out = []
    for unit in world.detect.units:
        ids = set(unit.fault_ids)
        rows = [
            o for o in seen.values() if o["identity"].day in days and ids & o["identity"].fault_ids
        ]
        expected = estimate.mean_success(
            [estimate.Episode.of(o["identity"]) for o in rows], ctx, logged=True
        )
        out.append(
            {
                "unit": unit.unit,
                "kinds": list(unit.kinds),
                "scopes": list(unit.scopes),
                "observed_carriers": _rate(sum(o["logged"] for o in rows), len(rows)),
                "expected_for_them": round(expected, 4) if rows else None,
                "expected_effect_points": {k: round(v, 2) for k, v in unit.effect.items()},
                "counts_for_gate": unit.counts_observed,
            }
        )
    return out


def _curves(world: Any) -> Any:
    from fleetlab.worlds import curves

    return getattr(world, "calibration", curves.load())


def table(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> str:
    """A Markdown table of rows, for the README and the report."""
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def composition_report(world: Any) -> dict[str, Any]:
    """Truth-side handoff: placement, visibility, observed causes and valuable fixes."""
    from fleetlab.faults import get
    from fleetlab.faults.records import health_shift
    from fleetlab.worlds import base, estimate
    from fleetlab.worlds.world import context, episode_slots

    spec = world.spec
    table = world.fault_table.by_id()
    placements: list[dict[str, Any]] = []
    for fault in spec.fault_schedule:
        row = table[fault.fault_id]
        visible = []
        if get(fault.kind).truth.writes_change:
            visible.append("change_log")
        if any(
            health_shift(fault.model_copy(update={"level": base.day_level(fault, day)}))
            for day in range(spec.windows.observed.first, spec.windows.observed.last + 1)
        ):
            visible.append("health")
        if not visible:
            visible.append("behaviour")
        placements.append(
            {
                "fault_id": fault.fault_id,
                "kind": fault.kind,
                "class": row.fault_class.value,
                "scope": fault.target.scope,
                "visible_in": visible,
                "observed_loss_points": row.expected_loss_points["observed"],
                "scoring_gain_points": row.expected_loss_points["scoring"],
                "drift_crossing_day": row.drift_crossing_day,
                "fixing_verbs": [verb.value for verb in row.fixing_verbs],
            }
        )
    ctx = context(world.composition, world.calibration)
    slots = episode_slots(world.jobs)
    robot_days = estimate.robot_day_success(spec.fault_schedule, slots, ctx)
    lowest = min(robot_days, key=lambda row: min(row["true"], row["logged"]))
    return {
        "faults": placements,
        "observed_causes": [cause.as_json() for cause in world.detect.observed_counted()],
        "lowest_robot_day": lowest,
        "robot_success_floor": world.recipe.robot_success_floor,
        "largest_three_single_fault_fixes": sorted(
            placements, key=lambda row: (-row["scoring_gain_points"], row["fault_id"])
        )[:3],
        "fix_gain_basis": "removing each fault alone over scoring; other faults remain",
    }
