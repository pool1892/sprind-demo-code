"""Generated feedback for the procedure builder, directly from scoring verdicts."""

from typing import Any

from fleetlab.scoring.diagnosis import FALSE_POSITIVES
from fleetlab.scoring.models import ScoreReport


def error_account(report: ScoreReport) -> dict[str, Any]:
    missed = [
        f.model_dump(mode="json") for f in report.claim1.faults if f.detectable and not f.found
    ]
    positives = [
        c.model_dump(mode="json")
        for c in report.claim1.causes
        if c.named and (c.verdict in FALSE_POSITIVES or c.verdict == "vague")
    ]
    if report.claim1.units:
        missed = [u for u in report.claim1.units if u["detectable"] and not u["found"]]
    for positive in positives:
        positive["matched"] = {
            "decoy": "a decoy",
            "inert": "the harmless companion of a confounder",
            "no fault": "nothing",
            "vague": "no target given; an effective fault sits at this scope",
        }[positive["verdict"]]
    actions = [a for a in report.claim2.actions if a.true_points is not None]
    actions.sort(key=lambda a: (-abs(a.expected_gain - float(a.true_points or 0)), a.rank))
    return {
        "missed_causes": missed,
        "false_positives": positives,
        "largest_gain_errors": [
            {
                **a.model_dump(mode="json"),
                "absolute_error_points": abs(a.expected_gain - float(a.true_points or 0)),
            }
            for a in actions[:10]
        ],
        "wrong_directions": [
            c.model_dump(mode="json")
            for c in report.claim1.causes
            if c.named and c.verdict == "wrong direction"
        ],
    }
