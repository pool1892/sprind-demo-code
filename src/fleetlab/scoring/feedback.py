"""Readable generated feedback; the JSON retains every scoring detail."""

from typing import Any

from fleetlab.contracts.actions import Action
from fleetlab.scoring.markdown import _table, action_text


def feedback_markdown(errors: dict[str, Any]) -> list[str]:
    lines = ["## What the analysis got wrong", ""]
    lines += _table(
        ["Missed cause", "Truth"],
        [(m.get("unit", m.get("fault_id")), m) for m in errors.get("missed_causes", [])],
    )
    lines += _table(
        ["Named cause index", "Target", "Matched", "Faults"],
        [
            (c["index"], c["target"], c["matched"], c.get("pair", c.get("faults")))
            for c in errors.get("false_positives", [])
        ],
    )
    lines += _table(
        ["Action rank", "Action", "Expected points", "True points", "Absolute error points"],
        [
            (
                a["rank"],
                action_text(Action.model_validate(a["action"])),
                a["expected_gain"],
                a["true_points"],
                a["absolute_error_points"],
            )
            for a in errors.get("largest_gain_errors", [])
        ],
    )
    return lines
