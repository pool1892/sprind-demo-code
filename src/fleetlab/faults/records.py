"""What a fault writes into the fleet's records, under the one rule (contracts, section 3),
and how it moves the health checks.

A configuration or software fault is a `Change` of its own key with the true
old and new values and an author drawn like a benign change's; every other
fault writes nothing. Nothing in a `Change` marks it as a fault: its fields
are a day, a scope, a key, two values and an author, and the values are the
fleet's own (a number of steps, a version string, a mode).

`condition_value` gives the true value of the condition a site fault changed,
in the units the condition's fault type defines (`condition_change`). The
fleet's description keeps showing the believed value, the fault's `nominal`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.model import Change, ChangeAuthor
from fleetlab.faults import get


def change_entry(spec: FaultSpec, day: int, author: ChangeAuthor) -> Change | None:
    """The change-log entry a fault writes on its onset day; None for a fault that writes none."""
    fault_type = get(spec.kind)
    if not fault_type.truth.writes_change:
        return None
    key = spec.target.config_key
    if key is None:  # the truth mapping's rules make this impossible; keep mypy honest
        raise ValueError(f"fault {spec.fault_id} writes a change but names no key")
    writer: Any = fault_type
    if not hasattr(writer, "new_value"):
        raise TypeError(f"fault kind {spec.kind} writes a change but defines no new_value")
    new = writer.new_value(spec)
    return Change(
        day=day, scope=spec.target.scope, key=key, old=spec.nominal, new=new, author=author
    )


def condition_value(spec: FaultSpec) -> ConfigValue:
    """The true value of the condition a site fault changed."""
    fault_type = get(spec.kind)
    if spec.target.condition_key is None or not hasattr(fault_type, "condition_value"):
        raise TypeError(f"fault kind {spec.kind} does not change a condition")
    changer: Any = fault_type
    value: ConfigValue = changer.condition_value(spec)
    return value


def health_shift(spec: FaultSpec) -> dict[str, float]:
    """How a fault moves its component's health readings, before noise; empty for most.

    The health model: a camera pose fault raises
    the camera's calibration residual, gripper wear lowers its grip force.
    Configuration, software, site and data faults never appear in health.
    """
    reader: Any = get(spec.kind)
    if not hasattr(reader, "health_shift"):
        return {}
    shift: dict[str, float] = reader.health_shift(spec)
    return shift


def site_notes(schedule: Sequence[FaultSpec], site_id: str, robot_ids: Sequence[str]) -> list[str]:
    """What an operator knows about a site's lasting conditions, for its public description.

    The `site_note` of every fault type in the schedule that defines one and
    has a fault covering a robot of the site, once each, sorted; empty for
    most sites. A note says that a condition exists, never its size or effect
    (`remote_link_latency`). It is written from day 0 whatever the onset: an
    operator knows how the site was installed.
    """
    notes: set[str] = set()
    for spec in schedule:
        note = getattr(get(spec.kind), "site_note", None)
        if note and any(spec.target.covers(robot, site_id) for robot in robot_ids):
            notes.add(str(note))
    return sorted(notes)


def measured(kind: str, suite: str = "libero_spatial") -> dict[str, Any]:
    """The measured dose-response of a fault type and its verdict (faults.md, sections 6 and 8).

    From `curves.json` beside this module: per calibration point, in the
    order of the type's `calibration`, the episodes, the successes by the
    fleet's check and by LIBERO's, the logged successes, the twins' successes
    (healthy twins; for a point with `companions`, a data type measured on top
    of other faults, the same episodes' true outcome, as `compared_with`
    says), the episodes broken and rescued, the exact McNemar p, the
    difference with its 95% interval, the run, and the point's role: "effect"
    (outcomes change beyond noise: fewer successes, or for a data fault either
    way), "decoy" (no change beyond noise and a loss of two points excluded,
    on 1,000 pairs), "inert" (a type with no mechanism whose episodes all kept
    their outcome) or "small_cause" (none of these). Points appended to a type
    in a later stage come after the others, so the points are a prefix of the
    type's `calibration` until every point is measured. The type's verdict is
    "kept" (some point is an effect), "decoy" (none is, and the decoy points
    are the levels to plant as decoys), "inert" or "dropped", on this suite
    only. The file stores these verdicts under `types[kind].verdicts[suite]`.
    A decoy on spatial tasks can be a cause on object tasks; callers must
    query every task family their carriers run.

    Spatial points live under the type; object points live in the supplement.
    Both queries return `suite`, `verdict`, `note` and `points`. Every point
    names its source directories (`run_dirs`) and twins (`healthy_run_dirs`);
    merged extensions retain all contributing directories. Raises KeyError
    for a kind or suite without a measured curve. The object coverage is
    partial; it includes the delay/chunks pair, three small lighting/backdrop
    causes and the inert calibration update.

    A type whose mechanism is another type's, bit for bit, names that type as
    its `curve_kind` and is read off its curve (`remote_link_latency`, the
    delay line of `action_latency`).
    """
    kind = getattr(get(kind), "curve_kind", kind)
    curves = _curves()
    entry = curves["types"][kind]
    points = (
        entry["points"]
        if suite == "libero_spatial"
        else [
            p
            for p in curves.get("supplement", {}).get("points", [])
            if p["kind"] == kind and p["suite"] == suite
        ]
    )
    if not points:
        raise KeyError(f"no curve of {kind} on {suite}")
    return {
        "suite": suite,
        "verdict": entry["verdicts"][suite],
        "note": entry.get("note", "")
        if suite == "libero_spatial"
        else entry.get("notes", {}).get(suite, ""),
        "points": points,
    }


def joints(suite: str | None = None) -> list[dict[str, Any]]:
    """The interaction pairs measured together (faults.md, section 8), on `suite` or all.

    Each: `name`, `members` [{kind, label}, ...], `suite`, `n`, the joint's
    `successes`, the `healthy_successes` and `single_successes` on the same
    episodes, `singles` (each member's successes over the healthy ones, on the
    same episodes as the joint), the joint's paired `difference` to healthy
    and its `role` against healthy (effect, decoy or small_cause, as the
    world generator's supplement writes it), the per-episode `effects` of
    each member and of the pair, the `interaction` (joint effect minus the
    sum of the single effects, with its bootstrap interval) and the
    `interaction_role`: "interaction" (the pair costs more than the sum),
    "additive" or "sub-additive".
    """
    found: list[dict[str, Any]] = _curves().get("supplement", {}).get("joints", [])
    return [j for j in found if suite is None or j["suite"] == suite]


def healthy_rate(suite: str = "libero_spatial") -> dict[str, Any]:
    """The healthy run a suite's curves were measured against: run id, n, successes."""
    curves = _curves()
    if suite == "libero_spatial":
        entry: dict[str, Any] = curves["healthy"]
        return entry
    entry = curves.get("supplement", {}).get("healthy", {})[suite]
    return entry


def _curves() -> dict[str, Any]:
    import json
    from importlib import resources

    text = resources.files("fleetlab.faults").joinpath("curves.json").read_text()
    curves: dict[str, Any] = json.loads(text)
    return curves
