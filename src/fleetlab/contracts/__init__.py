"""The contracts of the test bed: the interfaces of docs/design/contracts.md.

Implementation agents import from here (or from the submodules, one per
section of the note) and propose changes to the note.
A change that breaks a caller bumps `CONTRACTS_VERSION`.

| module | section | what |
|---|---|---|
| `vocabulary` | 1, 2 | components, keys, conditions, verbs, fault classes |
| `ids` | 1 | neutral identifiers and scopes |
| `model` | 3 | fleet, sites, robots, configuration, logs, system description |
| `identity` | 4 | episode identity, seeds, noise, replay rules |
| `hashing` | 4 | the one hash |
| `env` | 5 | the environment adapter |
| `policy` | 6 | the policy adapter |
| `faults` | 7 | fault specs, truth mappings, hooks |
| `actions` | 8 | actions, budgets, fix sets, resolution |
| `world`, `oracle` | 9 | world specs and files; the oracle lookup |
| `rollout` | 10 | jobs, snapshots, run layout |
| `log` | 11 | the episode log schema and its validator |
| `tools` | 12 | agent tools and answer schemas |
| `skills` | 13 | the skill format |
| `live` | 14 | risk estimator, supervisor, decisions |

This package never imports `fleetlab.faults`, where the fault kinds are
registered, and names no fault kind but the two generic kinds of the changes
an action adds (`actions.ADDED_CHANGE_KINDS`).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fleetlab.contracts.actions import (
        COSTS,
        SCORING_BUDGETS,
        Action,
        Budget,
        Cost,
        FixSet,
        Resolution,
        catalogue,
        cost_of,
        cut,
        resolve,
        resolve_all,
    )
    from fleetlab.contracts.env import (
        PLACED_THRESHOLDS,
        VERDICT_REASONS,
        Camera,
        CellHandle,
        CellHook,
        EnvSnapshot,
        EnvSpec,
        Observation,
        PrivilegedState,
        SimEnv,
        StepResult,
        SuccessCheck,
        SuccessVerdict,
        TaskGoal,
    )
    from fleetlab.contracts.faults import (
        Direction,
        FaultSpec,
        FaultState,
        FaultType,
        FaultTypeBase,
        Schedule,
        Target,
        TruthMapping,
        TruthTable,
        check_data_hook,
    )
    from fleetlab.contracts.hashing import hash_array, hash_bytes, hash_digest, hash_json
    from fleetlab.contracts.identity import (
        REPLAY_RULES,
        EpisodeIdentity,
        PolicySpec,
        SoftwareVersion,
        Task,
        neutral_id,
        noise_for,
        seeds_for,
        started_at_for,
    )
    from fleetlab.contracts.live import (
        Decision,
        DecisionRecord,
        RiskEstimator,
        StepTelemetry,
        Supervisor,
        SupervisorContext,
        SupervisorDecision,
    )
    from fleetlab.contracts.log import SCHEMA, Visibility, allowed_columns, visible_schema
    from fleetlab.contracts.model import Fleet, Robot, Site, describe_for_agents, effective_config
    from fleetlab.contracts.oracle import (
        Gain,
        OracleRow,
        OracleTable,
        achievable_gain,
        fault_effect,
        interaction,
        value_fix_set,
    )
    from fleetlab.contracts.policy import (
        Action7,
        ActionChunk,
        ControlMode,
        InWorkerPolicy,
        NoiseSample,
        Policy,
    )
    from fleetlab.contracts.rollout import Job, JobMode, RunLayout, Snapshot
    from fleetlab.contracts.skills import SkillManifest, load_skill, validate_skill_general
    from fleetlab.contracts.tools import (
        EXPERIMENT_BUDGET,
        Cause,
        Diagnosis,
        ExperimentRequest,
        ExperimentResult,
        FleetWarning,
        RankedAction,
        RankedFixSet,
        Receipt,
        RetrainIndicator,
        json_schema_strict,
    )
    from fleetlab.contracts.version import CONTRACTS_VERSION
    from fleetlab.contracts.vocabulary import (
        Component,
        ConditionKey,
        ConfigKey,
        FaultClass,
        Verb,
        vocabulary_json,
    )
    from fleetlab.contracts.world import WorldSpec, hash_world_files

# Keep truth-side contracts outside the tool server import closure.
_EXPORTS = {
    "COSTS": ("fleetlab.contracts.actions", "COSTS"),
    "SCORING_BUDGETS": ("fleetlab.contracts.actions", "SCORING_BUDGETS"),
    "Action": ("fleetlab.contracts.actions", "Action"),
    "Budget": ("fleetlab.contracts.actions", "Budget"),
    "Cost": ("fleetlab.contracts.actions", "Cost"),
    "FixSet": ("fleetlab.contracts.actions", "FixSet"),
    "Resolution": ("fleetlab.contracts.actions", "Resolution"),
    "catalogue": ("fleetlab.contracts.actions", "catalogue"),
    "cost_of": ("fleetlab.contracts.actions", "cost_of"),
    "cut": ("fleetlab.contracts.actions", "cut"),
    "resolve": ("fleetlab.contracts.actions", "resolve"),
    "resolve_all": ("fleetlab.contracts.actions", "resolve_all"),
    "PLACED_THRESHOLDS": ("fleetlab.contracts.env", "PLACED_THRESHOLDS"),
    "VERDICT_REASONS": ("fleetlab.contracts.env", "VERDICT_REASONS"),
    "Camera": ("fleetlab.contracts.env", "Camera"),
    "CellHandle": ("fleetlab.contracts.env", "CellHandle"),
    "CellHook": ("fleetlab.contracts.env", "CellHook"),
    "EnvSnapshot": ("fleetlab.contracts.env", "EnvSnapshot"),
    "EnvSpec": ("fleetlab.contracts.env", "EnvSpec"),
    "Observation": ("fleetlab.contracts.env", "Observation"),
    "PrivilegedState": ("fleetlab.contracts.env", "PrivilegedState"),
    "SimEnv": ("fleetlab.contracts.env", "SimEnv"),
    "StepResult": ("fleetlab.contracts.env", "StepResult"),
    "SuccessCheck": ("fleetlab.contracts.env", "SuccessCheck"),
    "SuccessVerdict": ("fleetlab.contracts.env", "SuccessVerdict"),
    "TaskGoal": ("fleetlab.contracts.env", "TaskGoal"),
    "Direction": ("fleetlab.contracts.faults", "Direction"),
    "FaultSpec": ("fleetlab.contracts.faults", "FaultSpec"),
    "FaultState": ("fleetlab.contracts.faults", "FaultState"),
    "FaultType": ("fleetlab.contracts.faults", "FaultType"),
    "FaultTypeBase": ("fleetlab.contracts.faults", "FaultTypeBase"),
    "Schedule": ("fleetlab.contracts.faults", "Schedule"),
    "Target": ("fleetlab.contracts.faults", "Target"),
    "TruthMapping": ("fleetlab.contracts.faults", "TruthMapping"),
    "TruthTable": ("fleetlab.contracts.faults", "TruthTable"),
    "check_data_hook": ("fleetlab.contracts.faults", "check_data_hook"),
    "hash_array": ("fleetlab.contracts.hashing", "hash_array"),
    "hash_bytes": ("fleetlab.contracts.hashing", "hash_bytes"),
    "hash_digest": ("fleetlab.contracts.hashing", "hash_digest"),
    "hash_json": ("fleetlab.contracts.hashing", "hash_json"),
    "REPLAY_RULES": ("fleetlab.contracts.identity", "REPLAY_RULES"),
    "EpisodeIdentity": ("fleetlab.contracts.identity", "EpisodeIdentity"),
    "PolicySpec": ("fleetlab.contracts.identity", "PolicySpec"),
    "SoftwareVersion": ("fleetlab.contracts.identity", "SoftwareVersion"),
    "Task": ("fleetlab.contracts.identity", "Task"),
    "neutral_id": ("fleetlab.contracts.identity", "neutral_id"),
    "noise_for": ("fleetlab.contracts.identity", "noise_for"),
    "seeds_for": ("fleetlab.contracts.identity", "seeds_for"),
    "started_at_for": ("fleetlab.contracts.identity", "started_at_for"),
    "Decision": ("fleetlab.contracts.live", "Decision"),
    "DecisionRecord": ("fleetlab.contracts.live", "DecisionRecord"),
    "RiskEstimator": ("fleetlab.contracts.live", "RiskEstimator"),
    "StepTelemetry": ("fleetlab.contracts.live", "StepTelemetry"),
    "Supervisor": ("fleetlab.contracts.live", "Supervisor"),
    "SupervisorContext": ("fleetlab.contracts.live", "SupervisorContext"),
    "SupervisorDecision": ("fleetlab.contracts.live", "SupervisorDecision"),
    "SCHEMA": ("fleetlab.contracts.log", "SCHEMA"),
    "Visibility": ("fleetlab.contracts.log", "Visibility"),
    "allowed_columns": ("fleetlab.contracts.log", "allowed_columns"),
    "visible_schema": ("fleetlab.contracts.log", "visible_schema"),
    "Fleet": ("fleetlab.contracts.model", "Fleet"),
    "Robot": ("fleetlab.contracts.model", "Robot"),
    "Site": ("fleetlab.contracts.model", "Site"),
    "describe_for_agents": ("fleetlab.contracts.model", "describe_for_agents"),
    "effective_config": ("fleetlab.contracts.model", "effective_config"),
    "Gain": ("fleetlab.contracts.oracle", "Gain"),
    "OracleRow": ("fleetlab.contracts.oracle", "OracleRow"),
    "OracleTable": ("fleetlab.contracts.oracle", "OracleTable"),
    "achievable_gain": ("fleetlab.contracts.oracle", "achievable_gain"),
    "fault_effect": ("fleetlab.contracts.oracle", "fault_effect"),
    "interaction": ("fleetlab.contracts.oracle", "interaction"),
    "value_fix_set": ("fleetlab.contracts.oracle", "value_fix_set"),
    "Action7": ("fleetlab.contracts.policy", "Action7"),
    "ActionChunk": ("fleetlab.contracts.policy", "ActionChunk"),
    "ControlMode": ("fleetlab.contracts.policy", "ControlMode"),
    "InWorkerPolicy": ("fleetlab.contracts.policy", "InWorkerPolicy"),
    "NoiseSample": ("fleetlab.contracts.policy", "NoiseSample"),
    "Policy": ("fleetlab.contracts.policy", "Policy"),
    "Job": ("fleetlab.contracts.rollout", "Job"),
    "JobMode": ("fleetlab.contracts.rollout", "JobMode"),
    "RunLayout": ("fleetlab.contracts.rollout", "RunLayout"),
    "Snapshot": ("fleetlab.contracts.rollout", "Snapshot"),
    "SkillManifest": ("fleetlab.contracts.skills", "SkillManifest"),
    "load_skill": ("fleetlab.contracts.skills", "load_skill"),
    "validate_skill_general": ("fleetlab.contracts.skills", "validate_skill_general"),
    "EXPERIMENT_BUDGET": ("fleetlab.contracts.tools", "EXPERIMENT_BUDGET"),
    "Cause": ("fleetlab.contracts.tools", "Cause"),
    "Diagnosis": ("fleetlab.contracts.tools", "Diagnosis"),
    "ExperimentRequest": ("fleetlab.contracts.tools", "ExperimentRequest"),
    "ExperimentResult": ("fleetlab.contracts.tools", "ExperimentResult"),
    "FleetWarning": ("fleetlab.contracts.tools", "FleetWarning"),
    "RankedAction": ("fleetlab.contracts.tools", "RankedAction"),
    "RankedFixSet": ("fleetlab.contracts.tools", "RankedFixSet"),
    "Receipt": ("fleetlab.contracts.tools", "Receipt"),
    "RetrainIndicator": ("fleetlab.contracts.tools", "RetrainIndicator"),
    "json_schema_strict": ("fleetlab.contracts.tools", "json_schema_strict"),
    "CONTRACTS_VERSION": ("fleetlab.contracts.version", "CONTRACTS_VERSION"),
    "Component": ("fleetlab.contracts.vocabulary", "Component"),
    "ConditionKey": ("fleetlab.contracts.vocabulary", "ConditionKey"),
    "ConfigKey": ("fleetlab.contracts.vocabulary", "ConfigKey"),
    "FaultClass": ("fleetlab.contracts.vocabulary", "FaultClass"),
    "Verb": ("fleetlab.contracts.vocabulary", "Verb"),
    "vocabulary_json": ("fleetlab.contracts.vocabulary", "vocabulary_json"),
    "WorldSpec": ("fleetlab.contracts.world", "WorldSpec"),
    "hash_world_files": ("fleetlab.contracts.world", "hash_world_files"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = [
    "CONTRACTS_VERSION",
    "COSTS",
    "EXPERIMENT_BUDGET",
    "PLACED_THRESHOLDS",
    "REPLAY_RULES",
    "SCHEMA",
    "SCORING_BUDGETS",
    "VERDICT_REASONS",
    "Action",
    "Action7",
    "ActionChunk",
    "Budget",
    "Camera",
    "Cause",
    "CellHandle",
    "CellHook",
    "Component",
    "ConditionKey",
    "ConfigKey",
    "ControlMode",
    "Cost",
    "Decision",
    "DecisionRecord",
    "Diagnosis",
    "Direction",
    "EnvSnapshot",
    "EnvSpec",
    "EpisodeIdentity",
    "ExperimentRequest",
    "ExperimentResult",
    "FaultClass",
    "FaultSpec",
    "FaultState",
    "FaultType",
    "FaultTypeBase",
    "FixSet",
    "Fleet",
    "FleetWarning",
    "Gain",
    "InWorkerPolicy",
    "Job",
    "JobMode",
    "NoiseSample",
    "Observation",
    "OracleRow",
    "OracleTable",
    "Policy",
    "PolicySpec",
    "PrivilegedState",
    "RankedAction",
    "RankedFixSet",
    "Receipt",
    "Resolution",
    "RetrainIndicator",
    "RiskEstimator",
    "Robot",
    "RunLayout",
    "Schedule",
    "SimEnv",
    "Site",
    "SkillManifest",
    "Snapshot",
    "SoftwareVersion",
    "StepResult",
    "StepTelemetry",
    "SuccessCheck",
    "SuccessVerdict",
    "Supervisor",
    "SupervisorContext",
    "SupervisorDecision",
    "Target",
    "Task",
    "TaskGoal",
    "TruthMapping",
    "TruthTable",
    "Verb",
    "Visibility",
    "WorldSpec",
    "achievable_gain",
    "allowed_columns",
    "catalogue",
    "check_data_hook",
    "cost_of",
    "cut",
    "describe_for_agents",
    "effective_config",
    "fault_effect",
    "hash_array",
    "hash_bytes",
    "hash_digest",
    "hash_json",
    "hash_world_files",
    "interaction",
    "json_schema_strict",
    "load_skill",
    "neutral_id",
    "noise_for",
    "resolve",
    "resolve_all",
    "seeds_for",
    "started_at_for",
    "validate_skill_general",
    "value_fix_set",
    "visible_schema",
    "vocabulary_json",
]
