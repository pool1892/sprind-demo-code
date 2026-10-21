"""Few subscription calls through the existing runtime; no tools or skills mounted."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fleetlab.agents.run import AgentResult, attempt_record, lost_runtime_answer, run_agent
from fleetlab.baselines.common import (
    BaselineAnswer,
    export_catalogue,
    ranking,
    static_description,
)
from fleetlab.contracts.actions import Action
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.tools import RankedFixSet
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey, Verb

MODEL = "gpt-6-astra"
SYSTEM = (
    "Rank robot fleet interventions. Follow the stated information restriction. "
    "Rank only the supplied catalogue entries. No tools, no skills, no external information. "
    "Return the requested JSON. Do not ask questions."
)
SCORING_RULE = (
    "route_to_teleop and request_training_data: "
    "Neither changes what the robots do in the scoring window, so their expected gain "
    "in fleet success is zero; they belong at the bottom of the list, and in the "
    "diagnosis as a retrain indicator. "
)
VALUE_INSTRUCTIONS = (
    "For each choice return only index and rationale; do not choose or return a value. "
    "A set writes the recorded standard value of that key at that scope, so it "
    "changes something only where the current value at that scope differs from "
    "the standard, and at fleet scope it changes nothing on a robot or site that "
    "carries a value of its own. The recorded standard for a configuration key is "
    "the fleet's recorded value. A change_condition writes the recorded standard "
    "value of its condition at that site. These values are not shown. "
    "Configuration precedence is robot over site over fleet. A site set leaves robot "
    "overrides in place. Setting an unchanged value changes nothing. Revert undoes "
    "the recorded change at exactly its named scope; other scopes' changes and "
    "narrower overrides remain. route_to_teleop uses its catalogue task class; "
    "request_training_data uses the recorded operator default, any failure mode. " + SCORING_RULE
)


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    rationale: str


class IndexRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choices: list[Choice] = Field(min_length=10)


class BaselineModelError(RuntimeError):
    """Failed calls are missing; authentication/rate-limit errors stop further calls."""

    def __init__(self, result: AgentResult, attempts: Sequence[dict[str, Any]] = ()) -> None:
        self.result = result
        self.attempts = list(attempts)
        super().__init__(
            f"model call failed ({result.error}): {result.error_detail}; {result.run_dir}"
        )


def catalogue_prompt(entries: Sequence[Action], *, values: bool) -> list[dict[str, Any]]:
    """No-data sees verb/target/cost catalogue structure, never recorded parameter values."""
    return [
        {
            "index": i,
            **a.model_dump(
                mode="json",
                exclude=set() if values or a.verb.value == "route_to_teleop" else {"value"},
            ),
        }
        for i, a in enumerate(entries)
    ]


def model_call(
    prompt: str,
    *,
    images: Sequence[Path] = (),
    run_root: Path | None = None,
    caller: Callable[..., AgentResult] | None = None,
    label: str,
    schema: type[BaseModel] = IndexRanking,
    previous_attempt: AgentResult | None = None,
) -> tuple[Any, dict[str, Any]]:
    if previous_attempt is not None and not lost_runtime_answer(previous_attempt):
        raise ValueError("a recorded rerun requires a lost runtime answer")
    attempts = [] if previous_attempt is None else [attempt_record(previous_attempt)]
    while len(attempts) < 2:
        result = (caller or run_agent)(
            "codex",
            MODEL,
            SYSTEM,
            prompt,
            images=images,
            tools=None,
            schema=schema,
            effort="xhigh",
            service_tier="fast",
            auth="subscription",
            timeout=600,
            run_root=run_root,
            label=label,
        )
        attempts.append(attempt_record(result))
        if result.ok:
            break
        if not lost_runtime_answer(result) or len(attempts) == 2:
            raise BaselineModelError(result, attempts)
    else:
        raise AssertionError("a model repeat must have at most two starts")
    return result.result, {
        "backend": "codex",
        "requested_model": MODEL,
        "auth": "subscription",
        "effort": "xhigh",
        "service_tier": "fast",
        "skills": False,
        "tools": False,
        "seconds": sum(a["seconds"] for a in attempts),
        "attempts": attempts,
        "rerun": len(attempts) == 2,
        "usage": result.usage,
        "run_dir": result.run_dir,
        "runtime_meta": result.meta,
    }


def validate_actions(actions: Sequence[Action], entries: Sequence[Action]) -> None:
    """Check catalogue membership and simulator value domains, without hidden world reads.

    This checks whether an action is applicable; existing replay coverage is a separate
    post-selection check. Inert camera/software keys have no parameter mechanism.
    """
    allowed = {a.model_dump_json() for a in entries}
    positive_ints = {
        ConfigKey.policy_sampling_steps,
        ConfigKey.policy_actions_executed_per_chunk,
        ConfigKey.policy_image_views,
    }
    counts = {
        ConditionKey.lighting: (1, 2),
        ConditionKey.placement: (2, 3),
        ConditionKey.object_batch: (2,),
        ConditionKey.clutter: (1,),
    }
    for action in actions:
        if action.model_dump_json() not in allowed:
            raise ValueError("action or teleoperation task class is outside the filled catalogue")
        if not action.is_complete:
            raise ValueError("incomplete recorded action value")
        value, key = action.value, action.target.config_key
        valid = True
        if action.verb == Verb.set:
            if key in positive_ints:
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 1
            elif key == ConfigKey.control_action_delay_steps:
                valid = (
                    isinstance(value, int | float) and not isinstance(value, bool) and value >= 0
                )
            elif key == ConfigKey.control_latency_compensation_steps:
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
            elif key == ConfigKey.control_mode:
                valid = value in {m.value for m in ControlMode} if isinstance(value, str) else False
            elif key == ConfigKey.policy_precision:
                from fleetlab.policies.xvla import PRECISIONS

                valid = isinstance(value, str) and value in PRECISIONS
        elif action.verb == Verb.change_condition:
            condition = action.target.condition_key
            assert condition is not None
            values = value if isinstance(value, list) else [value]
            valid = len(values) in counts[condition] and all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in values
            )
            if valid and condition == ConditionKey.clutter:
                blend = values[0]
                assert isinstance(blend, int | float)
                valid = 0 <= blend <= 1
        if not valid:
            raise ValueError(f"recorded value cannot be applied by the oracle: {action}")
        # Serialization with allow_nan=False also refuses nonfinite scalar/vector values.
        try:
            json.dumps(action.model_dump(mode="json"), allow_nan=False)
        except ValueError as exc:
            raise ValueError("recorded action value is not finite") from exc


def model_ranking(
    entries: Sequence[Action],
    prompt: str,
    *,
    run_root: Path | None = None,
    caller: Callable[..., AgentResult] | None = None,
    label: str,
) -> tuple[RankedFixSet, dict[str, Any]]:
    raw, metadata = model_call(prompt, run_root=run_root, caller=caller, label=label)
    parsed = IndexRanking.model_validate(raw)
    indices = [c.index for c in parsed.choices]
    if len(indices) != len(set(indices)) or any(i >= len(entries) for i in indices):
        raise ValueError("model named repeated or out-of-catalogue indices; no automatic retry")
    chosen = [entries[c.index] for c in parsed.choices]
    validate_actions(chosen, entries)
    return ranking(
        chosen,
        [c.rationale for c in parsed.choices],
    ), metadata


def no_data_ranker(
    export: Path,
    *,
    run_root: Path | None = None,
    caller: Callable[..., AgentResult] | None = None,
) -> BaselineAnswer:
    entries = export_catalogue(export)
    prompt = (
        "See only static sites, robots, component names, keys, costs and action catalogue. "
        "No outcomes, health readings, ages, service records, actual configuration values, "
        "logs or video are available. Rank actions by prior plausibility of improving fleet "
        "success. Return 32 distinct action indices (or all if fewer), best first, with a "
        "brief rationale for each. The headline budgets are 3 and 6 actions; the secondary "
        "budgets are 4 and 8 operator hours, cut greedily from this same ranking. Catalogue "
        "entries specify the available interventions. "
        + VALUE_INSTRUCTIONS
        + "Do not infer a fault from an identifier.\n"
        + json.dumps(
            {
                "system": static_description(export),
                "catalogue": catalogue_prompt(entries, values=False),
            },
            separators=(",", ":"),
        )
    )
    ranked, metadata = model_ranking(
        entries, prompt, run_root=run_root, caller=caller, label="baseline no-data"
    )
    return BaselineAnswer(
        "no_data", ranked, metadata={"values": "recorded-standard; hidden from model", **metadata}
    )
