"""Static public description additions, served for every export revision."""

from copy import deepcopy
from typing import Any

from fleetlab.toolserver.action_validation import ACTION_DOMAINS
from fleetlab.toolserver.public_results import PUBLIC_REFUSALS


def public_description(description: dict[str, Any]) -> dict[str, Any]:
    """Keep stored exports immutable while serving the current system glossary."""
    public = deepcopy(description)
    vocabulary = public["vocabulary"]
    for component in vocabulary["components"]:
        if component["name"] == "controller":
            component["description"] = (
                "the computer that runs the policy and control stack, including the robot's "
                "outcome detector and its outcome recording"
            )
    vocabulary["experiment_value_domains"] = dict(ACTION_DOMAINS)
    vocabulary["experiment_refusal_codes"] = dict(PUBLIC_REFUSALS)
    return public


# System semantics, independent of any world or measured fault response. Keep
# these in describe(dictionary), where the episode-card procedure points.
COMPLETION_CRITERION = """## Completion and outcome recording

The fleet's completion criterion is placed: the task's LIBERO goal check holds,
no body of the robot (including the arm and gripper) touches the object, and
the object's linear speed is below 0.01 m/s and angular speed below 0.15 rad/s.
An object still held by the robot is not placed. The attempt ends successfully
when this criterion first holds. If the goal check holds without placement,
the fleet allows 60 further control steps before ending unsuccessfully; the
attempt can also end at its step limit or by a supervisor intervention.

Outcome recording is part of the controller. A recording error can change the
reported outcome without changing the object's motion or whether the task
really completed. Repairing the record alone does not change physical success.
"""


def public_dictionary(dictionary: str) -> str:
    return dictionary.rstrip() + "\n\n" + COMPLETION_CRITERION
