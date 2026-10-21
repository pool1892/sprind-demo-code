"""Public experiment preflight, shared by the tool and host queue planner.

Only the visible fleet model and request are read. This is an applicability
check, never a prediction of a repair's benefit or a test for hidden faults.
"""

from __future__ import annotations

import math
from typing import Any

from fleetlab.contracts.actions import Action, catalogue
from fleetlab.contracts.model import Fleet, effective_config
from fleetlab.contracts.tools import ExperimentRequest
from fleetlab.contracts.vocabulary import Verb

# Executable setting domains, not measured response curves. The loaded policy
# cannot change precision mid-run (engine.episode); pose and software changes
# need an adapter not exposed by the experiment runner. Recorded reverts remain
# available because they do not introduce a new setting implementation.
ACTION_DOMAINS: dict[str, str] = {
    "policy.sampling_steps": "positive integer",
    "policy.actions_executed_per_chunk": "positive integer",
    "policy.image_views": "integer from 1 to 4",
    "policy.precision": "fp32, fp16, fp16_res32; fixed for a loaded policy",
    "control.mode": "absolute or relative",
    "control.action_delay_steps": "nonnegative integer",
    "control.latency_compensation_steps": "nonnegative integer",
    "camera.scene.calibration": "recorded revert only",
    "camera.wrist.calibration": "recorded revert only",
    "software.version": "recorded revert only",
    "lighting": "finite signed stops, [stops], or [stops, warmth] (positive warmth is warmer)",
    "placement": "[x, y] or [x, y, z] in centimetres; finite signed numbers, z upward",
    "object_batch": "[mass doublings, friction halvings]; two finite signed numbers",
    "clutter": "blend fraction from 0 to 1, as a number or [number]",
}


def value_problem(action: Action) -> str | None:
    """A public, actionable domain error, independent of the world's truth."""
    if action.verb not in (Verb.set, Verb.change_condition):
        return None
    key, value = str(action.target.thing), action.value
    domain = ACTION_DOMAINS[key]
    integer = type(value) is int
    valid = False
    if key in ("policy.sampling_steps", "policy.actions_executed_per_chunk"):
        valid = integer and value >= 1  # type: ignore[operator]
    elif key == "policy.image_views":
        valid = integer and 1 <= value <= 4  # type: ignore[operator]
    elif key in ("control.action_delay_steps", "control.latency_compensation_steps"):
        valid = integer and value >= 0  # type: ignore[operator]
    elif key == "control.mode":
        valid = value in ("absolute", "relative")
    elif key in ("lighting", "placement", "object_batch", "clutter"):
        values = value if isinstance(value, list) else [value]
        counts = {"lighting": (1, 2), "placement": (2, 3), "object_batch": (2,), "clutter": (1,)}
        numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        valid = (
            len(values) in counts[key]
            and len(numbers) == len(values)
            and all(math.isfinite(v) for v in numbers)
        )
        if key == "clutter" and valid:
            valid = 0 <= numbers[0] <= 1
    return None if valid else f"{key}: allowed experiment values: {domain}"


def request_problems(request: ExperimentRequest, fleet: Fleet) -> list[str]:
    """Check every arm/action before allocating a request or any simulation work."""
    problems: list[str] = []
    known = {robot.id: robot for robot in fleet.robots()}
    if unknown := set(request.robots) - set(known):
        problems.extend(f"no robot {name!r} in this fleet" for name in sorted(unknown))
        return problems
    if len(request.robots) != len(set(request.robots)):
        problems.append("robots must be distinct")
    arms = [tuple(sorted(a.model_dump_json() for a in arm.actions)) for arm in request.arms]
    if len(arms) != len(set(arms)):
        problems.append("experiment arms must have distinct action sets")
    scopes = {"fleet", *known, *(site.id for site in fleet.sites)}
    listed = {catalogue_key(entry) for entry in catalogue(fleet)}
    for index, arm in enumerate(request.arms):
        changed: set[tuple[str, str]] = set()
        for action in arm.actions:
            prefix = f"arm {index}: "
            if action.target.scope not in scopes:
                problems.append(prefix + "action scope is absent from the public fleet")
                continue
            if catalogue_key(action) not in listed:
                problems.append(prefix + "action is not in the current catalogue")
                continue
            if not action.has_replay:
                problems.append(prefix + f"{action.verb.value} has no simulated effect")
                continue
            covered = [
                known[r] for r in request.robots if action.target.covers(r, known[r].site_id)
            ]
            if not covered:
                problems.append(prefix + "action covers none of the requested robots")
            if problem := value_problem(action):
                problems.append(prefix + problem)
            if action.verb == Verb.revert:
                changes = [
                    c
                    for c in fleet.change_log
                    if c.scope == action.target.scope and c.key == action.target.config_key
                ]
                if not changes:
                    problems.append(
                        prefix
                        + "key has no recorded change at this scope in the visible change log"
                    )
                elif all(c.old == c.new for c in changes):
                    problems.append(
                        prefix
                        + "visible change log records no value change for this key at this scope"
                    )
            for robot in covered:
                if action.target.component is not None:
                    if action.target.component not in robot.components:
                        problems.append(prefix + "component is absent from a requested robot")
                    continue
                key = action.target.config_key or action.target.condition_key
                assert key is not None
                current: dict[Any, Any] = {
                    **effective_config(fleet, robot.id),
                    **fleet.site(robot.site_id).conditions,
                }
                if action.verb in (Verb.set, Verb.change_condition) and key not in current:
                    problems.append(prefix + "key has no current value in the public description")
                entry = (robot.id, str(key))
                if entry in changed:
                    problems.append(prefix + "multiple actions change the same key on one robot")
                changed.add(entry)
    return list(dict.fromkeys(problems))


def catalogue_key(action: Action) -> tuple[str, ...]:
    """What a catalogue entry fixes: the verb and the target (the value is the proposer's)."""
    target = action.target
    return (
        action.verb.value,
        target.scope,
        "" if target.component is None else target.component.value,
        "" if target.config_key is None else target.config_key.value,
        "" if target.condition_key is None else target.condition_key.value,
    )


def public_fleet(description: dict[str, Any]) -> Fleet:
    """The public fleet schema has only omitted optional hidden fields."""
    return Fleet.model_validate(description["fleet"])


def request_refusal_code(problems: list[str]) -> str:
    """A fixed, actionable reason; classification uses public preflight only."""
    from fleetlab.toolserver.public_results import INVALID_REQUEST

    categories = (
        ("not in the current catalogue", "not_in_catalogue"),
        ("allowed experiment values:", "invalid_action_value"),
        ("no recorded change", "no_recorded_change"),
        ("no value change", "no_recorded_value_change"),
        ("distinct action sets", "identical_arms"),
        ("covers none", "uncovered_action"),
        ("multiple actions change", "conflicting_actions"),
        ("no simulated effect", "unsupported_action"),
        ("no current value", "missing_public_value"),
    )
    return next(
        (code for text, code in categories if any(text in problem for problem in problems)),
        INVALID_REQUEST,
    )
