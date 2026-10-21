"""The generator's detectability list against the oracle (experimental design, section 5).

The generator calls a cause detectable from the calibrated curves (at least two points of
fleet success over the observed or the scoring window, an identifiable target, at least
20 observed episodes); its list is `detectability.json` (a cause is a fault, or the
faults an interaction joins), else each fault of `fault_table.json`. The oracle measures
the scoring window by replay, so every cause the generator calls detectable must show at
least `MIN_POINTS` there: removing the fault (for an interaction, all its faults
together) must gain at least two points net. Verdicts per cause:

- `ok`: detectable and at least two points in the oracle;
- `miss`: detectable by the scoring window, below two points in the oracle;
- `observed-only`: detectable by the observed window alone and below two points over the
  scoring window, which the rule allows (experimental design, section 5: either window); listed,
  not a miss;
- `data`: a data fault (the success detector's own error): it changes the logged
  outcome, not what the robot does, so its gain by replay is zero by construction and
  its detectability is the generator's, on the logged outcome; listed, not a miss;
- `absent`: detectable, but no scoring episode carries it;
- `missing`: the oracle could not value it (replays missing);
- `stronger`: not detectable in the generator's list (an effective fault called small),
  yet at least two points in the oracle;
- `small`: an effective fault called small, and small in the oracle;
- `decoy` and `inert`: not causes; listed with their measured effect, which should be
  zero within noise (the bootstrap interval holds zero).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from fleetlab.oracle.gains import Estimate
from fleetlab.oracle.world import Cause
from fleetlab.worlds.truth import FaultTable

MIN_POINTS = 2.0


class Detectability(BaseModel):
    """One fault (or marked pair) of the generator's table against the oracle."""

    model_config = ConfigDict(extra="forbid")

    item: str
    role: str  # effective, decoy, inert, pair
    detectable: bool
    qualified_by: list[str]
    expected_points: float | None  # the generator's expected loss over the scoring window
    oracle_points: float | None
    low: float | None
    high: float | None
    verdict: str


def _verdict(detectable: bool, points: float | None) -> str:
    if points is None:
        return "missing"
    if detectable:
        return "ok" if points >= MIN_POINTS else "miss"
    return "stronger" if points >= MIN_POINTS else "small"


def check_detectability(
    table: FaultTable | None,
    effects: Mapping[str, Estimate],
    causes: Sequence[Cause] = (),
    joint: Mapping[str, Estimate] | None = None,
) -> list[Detectability]:
    """The generator's list against the oracle (module docstring).

    With the generator's causes (`detectability.json`), every cause is checked: a single
    fault by its `fault_effect` (`effects`), the faults an interaction joins by their
    joint removal (`joint`, by the cause's unit); the decoys and inert entries of the
    fault table follow. Without causes, every fault of the fault table is checked alone.
    """
    out: list[Detectability] = []
    joint = joint or {}
    verdict: str
    if causes:
        for cause in causes:
            single = len(cause.fault_ids) == 1
            effect = effects.get(cause.fault_ids[0]) if single else joint.get(cause.unit)
            points = None if effect is None else effect.points
            if cause.data:
                verdict = "data"
            elif effect is None:
                verdict = "absent"
            else:
                verdict = _verdict(cause.detectable, points)
                if verdict == "miss" and "scoring" not in cause.qualified_by:
                    verdict = "observed-only"
            out.append(
                Detectability(
                    item=cause.unit,
                    role=",".join(sorted(set(cause.roles))) or "effective",
                    detectable=cause.detectable,
                    qualified_by=list(cause.qualified_by),
                    expected_points=cause.expected_points,
                    oracle_points=points,
                    low=None if effect is None else effect.low,
                    high=None if effect is None else effect.high,
                    verdict=verdict,
                )
            )
    for fault in [] if table is None else table.faults:
        if causes and fault.role == "effective":
            continue  # the causes above stand for the effective faults
        effect = effects.get(fault.fault_id)
        expected = fault.expected_loss_points.get("scoring")
        points = None if effect is None else effect.points
        if effect is None:
            verdict = "absent" if fault.role == "effective" else fault.role
        elif points is None:
            verdict = "missing"
        elif fault.role in ("decoy", "inert"):
            verdict = fault.role
        else:
            verdict = _verdict(fault.detectable, points)
            if verdict == "miss" and "scoring" not in fault.qualified_by:
                verdict = "observed-only"
        out.append(
            Detectability(
                item=fault.fault_id,
                role=fault.role,
                detectable=fault.detectable,
                qualified_by=list(fault.qualified_by),
                expected_points=expected,
                oracle_points=points,
                low=None if effect is None else effect.low,
                high=None if effect is None else effect.high,
                verdict=verdict,
            )
        )
    return out


LISTED = frozenset({"ok", "observed-only", "data"})


def misses(rows: Sequence[Detectability]) -> list[Detectability]:
    """The causes the generator calls detectable that the oracle does not confirm (a cause
    qualified by the observed window alone, and a data fault, are listed, not missed)."""
    return [r for r in rows if r.detectable and r.verdict not in LISTED]
