"""The agents' tools and answer schemas (docs/design/contracts.md, section 12).

The tool server runs inside the inner sandbox with the export mounted read
only and no network. Its tools are listed in `TOOLS` with their argument
models; the answers the agents submit are the pydantic models below, and the
scorer reads the same models. `claude -p` and `codex exec` both get the JSON
schema of an answer from `json_schema_strict`, which OpenAI's strict mode
accepts: every object lists all its properties as required and forbids others,
an optional field is a union with null, no reference is left unresolved, and
nothing is untyped (agent-runtime spike, "Schemas"). `strict_schema_problems`
checks a schema against those rules.

Analysis runs in sessions: `request_experiment` queues an experiment and the
session ends "waiting"; the runtime runs the queued requests in the experiment
window, and the next session reads `experiment_results` and its `memory_read`
notes. The server enforces `EXPERIMENT_BUDGET` (8 requests per analysis, 60
episodes per arm); arm episodes carry the experiment's id in the visible
column `episodes.experiment_id`.

`describe(part, include_logs)` returns one part of the data dictionary and
the system description as compact JSON (`describe_part`): "dictionary",
"fleet", "vocabulary", "costs", or "all" for a small world, each refused above
`DESCRIBE_SIZE_CAP` (30,000 characters). The action catalogue is the visible
table `catalogue`, queried like any table.

Choices this module makes where the note is silent:

- `uncertainty` is an `Interval` object (`low`, `high`), not a pair, because
  strict mode has no tuples.
- `Warning.scope` holds the robot id or site id the note writes as
  "robot_id | site_id". The class shadows Python's built-in `Warning` inside
  this module only; the package re-exports it as `FleetWarning` as well.
- `EvidenceRef` is one object with nullable fields: an episode (and
  optionally a step), or a visible table and a query.
- `ExperimentResult` reports the episode ids each arm produced and a status;
  the outcomes are in the visible tables.
- `Receipt` says whether the submission was accepted, with its problems.
"""

from __future__ import annotations

import copy
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.actions import COSTS, Action, Budget, CostTable, FixSet, cut
from fleetlab.contracts.env import Camera
from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.ids import (
    ID_PATTERNS,
    EpisodeId,
    ExperimentId,
    IdKind,
    RobotId,
    ScopeKind,
    scope_kind,
)
from fleetlab.contracts.jobs import JOB_NAME_PATTERN, CheckSpec, JobId, JobProgram
from fleetlab.contracts.learned import CheckValue, LearnedSkillDraft, LearnedSkillReceipt
from fleetlab.contracts.log import visible_table_names
from fleetlab.contracts.probes import ProbeReceipt, ProbeRequest, ProbeResult
from fleetlab.contracts.skills import SKILL_NAME_PATTERN, SkillManifest

__all__ = [
    "DESCRIBE_SIZE_CAP",
    "EXPERIMENT_BUDGET",
    "JOB_TOOLS",
    "LEDGER_ENTRY_PATTERN",
    "MAX_ADOPTED_PER_ROUND",
    "MIN_RANKED_ACTIONS",
    "OPT_IN_TOOLS",
    "TOOLS",
    "AcceptanceProposal",
    "AcceptanceReceipt",
    "AdoptedChange",
    "Cause",
    "CheckReceipt",
    "DescribePart",
    "DescribeTooLarge",
    "Diagnosis",
    "EvidenceRef",
    "ExperimentArm",
    "ExperimentBudgetStatus",
    "ExperimentRequest",
    "ExperimentResult",
    "FixSet",
    "FleetWarning",
    "Interval",
    "JobAccount",
    "JobReceipt",
    "JobSubmission",
    "LedgerEntryId",
    "RankedAction",
    "RankedFixSet",
    "Receipt",
    "Replacement",
    "RetrainIndicator",
    "RoundDecision",
    "RunCheckArgs",
    "SaveJobArgs",
    "SaveJobReceipt",
    "SetGain",
    "ToolSpec",
    "Undo",
    "Warning",
    "WriteCheckArgs",
    "describe_part",
    "json_schema_strict",
    "strict_schema_problems",
    "tool_schemas",
]


def _check_table(name: str) -> str:
    if name not in visible_table_names():
        raise ValueError(f"{name!r} is not a visible table")
    return name


VisibleTable = Annotated[
    str,
    AfterValidator(_check_table),
    WithJsonSchema({"type": "string", "enum": list(visible_table_names())}),
]


def _check_warning_scope(scope: str) -> str:
    if scope_kind(scope) is ScopeKind.fleet:
        raise ValueError("a warning names a robot or a site")
    return scope


WarningScope = Annotated[
    str,
    Field(pattern=f"^({ID_PATTERNS[IdKind.site][1:-1]}|{ID_PATTERNS[IdKind.robot][1:-1]})$"),
    AfterValidator(_check_warning_scope),
]


class EvidenceRef(ContractModel):
    """Where a claim comes from: an episode (and a step), or a table and the query run on it."""

    episode_id: EpisodeId | None
    step: int | None = Field(ge=0)
    table: VisibleTable | None
    query: str | None

    @model_validator(mode="after")
    def _one_kind(self) -> EvidenceRef:
        episode = self.episode_id is not None
        table = self.table is not None and self.query is not None
        if episode == table:
            raise ValueError("evidence names an episode, or a table with a query, not both")
        if self.step is not None and not episode:
            raise ValueError("a step belongs to an episode")
        if (self.table is None) != (self.query is None):
            raise ValueError("a table and its query come together")
        return self


class Cause(ContractModel):
    """One diagnosed cause: what is wrong where (the target carries the scope), and how sure."""

    target: Target
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef]
    failure_mode: str


class Warning(ContractModel):
    """A prediction that a metric of a robot or site will fail by a day."""

    scope: WarningScope
    metric: str
    expected_day: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


FleetWarning = Warning


class RetrainIndicator(ContractModel):
    """A failure mode no action in the catalogue fixes, with examples and the data wanted."""

    failure_mode: str
    episode_ids: list[EpisodeId]
    why_no_action_fixes_it: str
    data_wanted: str


class Diagnosis(ContractModel):
    """The structured diagnosis of a world (`submit_diagnosis`)."""

    causes: list[Cause]
    mode_summaries: list[str]
    warnings: list[Warning]
    retrain: list[RetrainIndicator]


class Interval(ContractModel):
    """A range of values, low to high."""

    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        if self.high < self.low:
            raise ValueError("an interval's high is not below its low")
        return self


class RankedAction(ContractModel):
    """A proposed action, its expected gain in points of fleet success over the scoring window."""

    action: Action
    expected_gain: float
    uncertainty: Interval
    rationale: str


# The fewest actions a ranked list holds, so that the scorer can cut it at every budget.
MIN_RANKED_ACTIONS = 10


class RankedFixSet(ContractModel):
    """The ranked list an agent submits (`submit_fix_set`), best action first.

    At least ten actions, each with its value filled in and none twice, ordered
    by value. It carries no budget: the scorer cuts it at every budget of
    `contracts.actions.SCORING_BUDGETS` (`as_fix_set`).
    """

    actions: list[RankedAction] = Field(min_length=MIN_RANKED_ACTIONS)

    @model_validator(mode="after")
    def _complete_and_distinct(self) -> RankedFixSet:
        incomplete = [r.action.verb.value for r in self.actions if not r.action.is_complete]
        if incomplete:
            raise ValueError(f"actions without a value: {incomplete}")
        dumped = [r.action.model_dump_json() for r in self.actions]
        if len(dumped) != len(set(dumped)):
            raise ValueError("a ranked list names each action once")
        return self

    def as_fix_set(self, budget: Budget, costs: CostTable = COSTS) -> FixSet:
        """The fix set of section 8 at a budget: `contracts.actions.cut` of the ranked list."""
        return cut([ranked.action for ranked in self.actions], budget, costs)


class ExperimentArm(ContractModel):
    """One arm of an experiment: the actions in force (none for the control arm)."""

    actions: list[Action]


# The historical experiment budget of the design (stage 4), per analysis: the tool
# server's default limits (`toolserver.server.Limits`). The round driver sets its own,
# in attempts per round, with a limit per arm only where a run asks for one
# (`fleetlab.loop.driver.AttemptBudget`).
EXPERIMENT_BUDGET: dict[str, int] = {"requests": 8, "episodes_per_arm": 60}


class ExperimentRequest(ContractModel):
    """An A/B test in the experiment window (`request_experiment`). The budget in force,
    the episodes per arm included, is the tool server's (`Limits`), not the schema's."""

    arms: list[ExperimentArm] = Field(min_length=2)
    robots: list[RobotId] = Field(min_length=1)
    episodes_per_arm: int = Field(ge=1)
    paired_starts: bool  # the arms run the same tasks and start states
    power_note: str


class ArmResult(ContractModel):
    """The episodes one arm produced; their outcomes are in the visible tables."""

    arm_index: int = Field(ge=0)
    episode_ids: list[EpisodeId]


class ExperimentResult(ContractModel):
    """What became of a requested experiment (`experiment_results`)."""

    experiment_id: ExperimentId
    status: Literal["queued", "done", "refused"]
    arms: list[ArmResult]
    message: str


class Receipt(ContractModel):
    """The answer to a submission or a request: accepted or not, and why."""

    accepted: bool
    receipt_id: str
    problems: list[str]


# The round decision and the experiment budget's status (docs/design/loop.md, section 8;
# documented and re-exported by `contracts.decisions`).

# At most three new changes in one round ("at most three changes, or none").
MAX_ADOPTED_PER_ROUND = 3
# Ledger entries are numbered in adoption order over the whole engagement.
LEDGER_ENTRY_PATTERN = r"^L[1-9][0-9]*$"
LedgerEntryId = Annotated[str, StringConstraints(pattern=LEDGER_ENTRY_PATTERN)]


class AdoptedChange(ContractModel):
    """A new change the agent adopts this round, with its reason (public, kept in the ledger)."""

    action: Action
    rationale: str


class Replacement(ContractModel):
    """An earlier change (its ledger entry) replaced by another action."""

    entry_id: LedgerEntryId
    action: Action
    rationale: str


class Undo(ContractModel):
    """An earlier change (its ledger entry) taken back."""

    entry_id: LedgerEntryId
    rationale: str


class SetGain(ContractModel):
    """The predicted gain of the decision's whole set of changes, in points of fleet
    success over the fleet's current state (not a sum of single gains)."""

    low: float
    mid: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> SetGain:
        if not self.low <= self.mid <= self.high:
            raise ValueError("a set gain is ordered low <= mid <= high")
        return self


class RoundDecision(ContractModel):
    """What the agent decides at the end of a round (`submit_decision`).

    `ranked_candidates` are the further actions the agent considered and did not adopt,
    best first; they are optional, and a decision to change nothing, or to stop, needs
    none. The plan fixed after one look is round 1's adopted actions in order, then
    these alternatives in order, cut at the final number of changes
    (`fleetlab.loop.rounds.one_look_actions`, lane state's `one_look_ledger`). The tool
    server checks the scopes, the values, that every change has a simulated effect, and
    that replaced or undone entries are active in the fleet's ledger.
    """

    adopt: list[AdoptedChange] = Field(max_length=MAX_ADOPTED_PER_ROUND)
    replace: list[Replacement]
    undo: list[Undo]
    predicted_set_gain: SetGain
    expectations: list[str]  # what the next round's records should show if the set works
    stop: bool  # True: nothing further is worth changing after this decision
    ranked_candidates: list[RankedAction]  # alternatives, best first; may be empty

    @model_validator(mode="after")
    def _distinct(self) -> RoundDecision:
        targets = [e.entry_id for e in self.replace] + [e.entry_id for e in self.undo]
        if len(targets) != len(set(targets)):
            raise ValueError("an entry is replaced or undone once per decision")
        ranked = [r.action.model_dump_json() for r in self.ranked_candidates]
        if len(ranked) != len(set(ranked)):
            raise ValueError("a ranked list names each action once")
        changes = [c.action.model_dump_json() for c in self.adopt]
        if len(changes) != len(set(changes)):
            raise ValueError("an action is adopted once")
        incomplete = [
            a.verb.value
            for a in [c.action for c in self.adopt] + [r.action for r in self.replace]
            if not a.is_complete
        ]
        if incomplete:
            raise ValueError(f"actions without a value: {incomplete}")
        return self


class ExperimentBudgetStatus(ContractModel):
    """The experiment budget in force and what is left of it (`experiment_budget`).

    Attempts are episodes over every arm of every charged request; a refused request
    is not charged. Requests of sessions before `counted_from_session` (earlier
    rounds) do not count. Probe episodes (`request_probe`, `contracts.probes`) are
    charged to the same attempts: `attempts_used` is `task_attempts_used` (experiment
    arms, the policy doing a task) plus `probe_attempts_used` (scripted probe motions).
    """

    attempts: int = Field(ge=0)
    attempts_used: int = Field(ge=0)
    attempts_left: int = Field(ge=0)
    max_per_arm: int = Field(ge=1)
    requests: int = Field(ge=0)
    requests_used: int = Field(ge=0)
    counted_from_session: int = Field(ge=1)
    task_attempts_used: int = Field(default=0, ge=0)
    probe_attempts_used: int = Field(default=0, ge=0)
    probes: int = Field(default=0, ge=0)  # probe requests allowed; 0: probes are not offered
    probes_used: int = Field(default=0, ge=0)


# Tool arguments.


class QueryArgs(ContractModel):
    sql: str


class GetFramesArgs(ContractModel):
    episode_id: EpisodeId
    steps: list[int] = Field(min_length=1)
    camera: Camera


class DescribePart(StrEnum):
    """The parts `describe` returns, each small enough for one tool result."""

    dictionary = "dictionary"  # the data dictionary: every visible table and column
    fleet = "fleet"  # sites, robots, components, configuration; logs with include_logs
    vocabulary = "vocabulary"  # components, keys, conditions, verbs, fault classes
    costs = "costs"  # what each action costs in operator hours
    all = "all"  # everything at once, for a small world


class DescribeArgs(ContractModel):
    part: DescribePart
    include_logs: bool  # False: the static structure only (the no-data baseline)


# The largest text `describe` returns: a larger part is refused with a pointer to the
# smaller parts and the tables. Claude Code refused a 93 kB result.
DESCRIBE_SIZE_CAP = 30_000


class DescribeTooLarge(ValueError):
    """A part of the description is larger than `DESCRIBE_SIZE_CAP`."""


def describe_part(
    part: DescribePart | str,
    description: dict[str, Any],
    dictionary: str,
    include_logs: bool,
) -> str:
    """One part of the data dictionary and the system description, as compact JSON.

    `description` is the export's `description.json` (`describe_for_agents`
    with the logs), `dictionary` its data dictionary. "fleet" leaves out the
    change and maintenance logs unless `include_logs`; "all" is every part in
    one object. The action catalogue is never part of it: it is the table
    `catalogue`. Raises `DescribeTooLarge` for a text over `DESCRIBE_SIZE_CAP`.
    """
    part = DescribePart(part)
    fleet = description["fleet"]
    if not include_logs:
        fleet = {k: v for k, v in fleet.items() if k not in ("change_log", "maintenance_log")}
    parts: dict[DescribePart, dict[str, Any]] = {
        DescribePart.dictionary: {"dictionary": dictionary},
        DescribePart.fleet: {
            "contracts_version": description.get("contracts_version"),
            "fleet": fleet,
            "task_classes": description.get("task_classes", []),
            "catalogue_entries": description.get("catalogue_entries"),
        },
        DescribePart.vocabulary: {"vocabulary": description["vocabulary"]},
        DescribePart.costs: {"costs": description.get("costs", [])},
    }
    if part is DescribePart.all:
        chosen: dict[str, Any] = {}
        for value in parts.values():
            chosen.update(value)
    else:
        chosen = parts[part]
    text = json.dumps(chosen, separators=(",", ":"), ensure_ascii=False)
    if len(text) > DESCRIBE_SIZE_CAP:
        raise DescribeTooLarge(
            f"describe(part={part.value!r}) is {len(text):,} characters, more than "
            f"{DESCRIBE_SIZE_CAP:,}: ask for one part (dictionary, fleet, vocabulary, costs), "
            "leave the logs out, or query the tables change_log, maintenance and catalogue"
        )
    return text


class NoArgs(ContractModel):
    pass


class ReadSkillArgs(ContractModel):
    name: Annotated[str, Field(pattern=SKILL_NAME_PATTERN)]


class MemoryWriteArgs(ContractModel):
    notes: str


class ToolSpec(NamedTuple):
    """One tool of the server: name, what it does, its argument model, what it returns."""

    name: str
    description: str
    arguments: type[BaseModel]
    returns: str


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "query",
        "run SQL (DuckDB) over the visible tables; no file access",
        QueryArgs,
        "rows",
    ),
    ToolSpec(
        "get_frames",
        "camera images of an episode at the given steps, from its video",
        GetFramesArgs,
        "images",
    ),
    ToolSpec(
        "describe",
        "one part of the data dictionary and the system description, as compact JSON: "
        "dictionary, fleet, vocabulary, costs, or all for a small world; the action "
        "catalogue is the table catalogue",
        DescribeArgs,
        "JSON",
    ),
    ToolSpec(
        "list_skills", "the skills available, by their front matter", NoArgs, "[SkillManifest]"
    ),
    ToolSpec("read_skill", "the text of one skill", ReadSkillArgs, "text"),
    ToolSpec(
        "request_experiment",
        "queue an A/B test; it runs after this session ends, results in the next",
        ExperimentRequest,
        "Receipt",
    ),
    ToolSpec(
        "experiment_results",
        "the experiments requested so far and their episodes",
        NoArgs,
        "[ExperimentResult]",
    ),
    ToolSpec("submit_diagnosis", "submit the diagnosis", Diagnosis, "Receipt"),
    ToolSpec("submit_fix_set", "submit the ranked fix set", RankedFixSet, "Receipt"),
    ToolSpec("memory_read", "your notes for this world", NoArgs, "text"),
    ToolSpec("memory_write", "replace your notes for this world", MemoryWriteArgs, "Receipt"),
    ToolSpec(
        "write_learned_skill",
        "write, or update under the same name, a learned diagnostic skill that a later "
        "investigation of a similar fleet can read: when it applies, the queries that showed "
        "it with what they returned here, the competing explanations and how to tell them "
        "apart, and the experiment that settles the question",
        LearnedSkillDraft,
        "LearnedSkillReceipt",
    ),
)

# The round driver's two tools (docs/design/loop.md, section 8).
TOOLS += (
    ToolSpec(
        "experiment_budget",
        "the experiment budget in force and what is left of it (refused requests are free)",
        NoArgs,
        "ExperimentBudgetStatus",
    ),
    ToolSpec(
        "earlier_decisions",
        "the round decisions you submitted so far, oldest first, each with the session it "
        "came from: what you adopted, replaced and undid, your reasons, your predicted gain "
        "and your expectations",
        NoArgs,
        "[decision with session]",
    ),
    ToolSpec(
        "submit_decision",
        "submit the round's decision (changes to adopt, replace or undo, the predicted gain "
        "of the set, expectations, whether to stop, and any ranked alternatives)",
        RoundDecision,
        "Receipt",
    ),
)

# Probes the agent composes (docs/design/loop.md, section 10; `contracts.probes`). They
# are offered only where the runtime enables them (`OPT_IN_TOOLS`).
TOOLS += (
    ToolSpec(
        "request_probe",
        "queue a probe you compose: a short scripted motion of the end effector and the "
        "gripper (hold, step, ramp, sine, open_gripper, close_gripper), played without the "
        "policy on the robots you name; every control step's commanded target and measured "
        "response is recorded. It runs after this session ends, like an experiment, and each "
        "probe episode is one attempt of the experiment budget",
        ProbeRequest,
        "ProbeReceipt",
    ),
    ToolSpec(
        "probe_results",
        "the probes requested so far, with your own name, purpose and expectation for each, "
        "and their probe episodes (tables probe_episodes, probe_segments and probe_steps)",
        NoArgs,
        "[ProbeResult]",
    ),
)
# Jobs: the installation takes on a new job (docs/design/jobs.md; the job models are
# `fleetlab.contracts.jobs`). The agent proposes the acceptance condition, composes
# job programs that run after the session on development starts it names, reads their
# records in the job tables, saves a job and the checks it wrote, and gives the shift
# lead its account. Offered only where the job driver names them
# (`fleetlab.jobs.driver`); the tool server's side is `fleetlab.toolserver.jobs`.

JobName = Annotated[str, StringConstraints(pattern=JOB_NAME_PATTERN)]


class AcceptanceProposal(ContractModel):
    """The agent's proposed acceptance condition (`submit_acceptance_proposal`): the job
    must deliver on at least `K` of `N` reserved release starts it was never run on
    (`contracts.jobs.Acceptance`, with the agent's reason)."""

    N: int = Field(ge=1, le=1000)  # release starts
    K: int = Field(ge=0, le=1000)  # deliveries required of the N
    reason: str = Field(max_length=1200)


class AcceptanceReceipt(ContractModel):
    """The shift lead's answer to a proposal: approved, or what is missing."""

    accepted: bool  # the proposal was taken (refused: problems say why)
    receipt_id: str
    problems: list[str]
    approved: bool  # the shift lead approved it; it then stands for the whole request
    reply: str  # the shift lead's words


class JobSubmission(ContractModel):
    """A job program to run (`submit_job_program`) after this session, on the development
    starts named by their stored initial-state index."""

    program: JobProgram
    starts: list[int] = Field(min_length=1, max_length=20)
    record_video: bool


class JobReceipt(ContractModel):
    """The answer to `submit_job_program`: accepted or not, why, and the budget after it."""

    accepted: bool
    job_id: str  # empty when refused
    problems: list[str]
    program_digest: str  # sha256 of the program's canonical JSON
    starts: int = Field(ge=0)  # starts this request is charged; 0 when refused
    starts_used: int = Field(ge=0)  # of the investigation's starts, this request included
    starts_left: int = Field(ge=0)
    requests_this_session: int = Field(ge=0)


class SaveJobArgs(ContractModel):
    """Save the program of a job that ran into the installation's job list, with the
    check that goes with it; the job saved last in a session is released after it."""

    name: JobName
    job_id: JobId
    purpose: str = Field(max_length=1200)
    check: JobName | None  # the name of a check written in this investigation or saved


class SaveJobReceipt(ContractModel):
    accepted: bool
    receipt_id: str
    problems: list[str]
    name: str
    program_digest: str


class WriteCheckArgs(ContractModel):
    """A check to keep (`write_check`): its SQL runs once here with `check_with`'s values,
    which are not stored."""

    check: CheckSpec
    check_with: list[CheckValue] = Field(max_length=8)


class CheckReceipt(ContractModel):
    accepted: bool
    receipt_id: str
    problems: list[str]
    name: str


class RunCheckArgs(ContractModel):
    """Run a saved check (`run_check`): over every job run so far, or over one job's."""

    name: JobName
    job_id: JobId | None
    params: list[CheckValue] = Field(max_length=8)


class JobAccount(ContractModel):
    """The agent's account to the shift lead (`submit_job_account`), which ends the
    investigation: `job` names the released job, or is null when nothing was released."""

    job: JobName | None
    account: str = Field(min_length=1, max_length=6000)


TOOLS += (
    ToolSpec(
        "operator_state",
        "the shift lead's request: its text, the requested goals (what must hold at the "
        "end), the base-task identifier, the process rules on the order of goals, "
        "the budget and what is left of it, the development starts, the acceptance condition "
        "once approved, and the message log",
        NoArgs,
        "JSON",
    ),
    ToolSpec(
        "submit_acceptance_proposal",
        "propose to the shift lead the acceptance condition the job must meet before any "
        "program is tested: N reserved release starts and K of them that must deliver; the "
        "shift lead approves or says what is missing",
        AcceptanceProposal,
        "AcceptanceReceipt",
    ),
    ToolSpec(
        "submit_job_program",
        "queue a job program you compose (policy steps with an instruction and a stop "
        "condition, motion steps of a small motion vocabulary, waits, and returns of the hand "
        "to its start pose) to run on the development "
        "starts you name, in one continuous simulator state per start; it runs after this "
        "session ends, results in the next",
        JobSubmission,
        "JobReceipt",
    ),
    ToolSpec(
        "job_results",
        "the job programs requested so far, with your own name and purpose for each, their "
        "status, episodes, deliveries and order-rule violations (tables job_episodes, "
        "job_step_events, object_steps, goal_steps and robot_steps)",
        NoArgs,
        "[JobResult]",
    ),
    ToolSpec(
        "save_job",
        "save the program of a job that ran under a name, with its check, into the "
        "installation's job list; the job saved last in a session is released after that "
        "session on the reserved release starts, once per investigation",
        SaveJobArgs,
        "SaveJobReceipt",
    ),
    ToolSpec(
        "write_check",
        "write, or update under the same name, a check the next shift can run: one SQL "
        "SELECT over the job tables with named parameters, its purpose and what it expects",
        WriteCheckArgs,
        "CheckReceipt",
    ),
    ToolSpec(
        "run_check",
        "run a saved check (yours or the installation's) with values for its parameters, "
        "over every job run so far or over one job's episodes",
        RunCheckArgs,
        "rows",
    ),
    ToolSpec(
        "job_library",
        "the installation's job list: saved jobs, checks and probe recipes, and those you "
        "saved in this investigation",
        NoArgs,
        "JSON",
    ),
    ToolSpec(
        "submit_job_account",
        "submit your account to the shift lead, which ends the investigation: name the "
        "released job, or null when nothing was released",
        JobAccount,
        "Receipt",
    ),
)
JOB_TOOLS: tuple[str, ...] = (
    "operator_state",
    "submit_acceptance_proposal",
    "submit_job_program",
    "job_results",
    "save_job",
    "write_check",
    "run_check",
    "job_library",
    "submit_job_account",
)

# Tools a session gets only when its runtime names them: a session that names no tools
# gets every other tool. The job tools are opt-in as the probes are, so that no session
# that names no tools changes.
OPT_IN_TOOLS: frozenset[str] = frozenset({"request_probe", "probe_results", *JOB_TOOLS})

# The models a tool can return as JSON, for the server's own validation.
RESULT_MODELS: dict[str, type[BaseModel]] = {
    "list_skills": SkillManifest,
    "request_experiment": Receipt,
    "experiment_results": ExperimentResult,
    "experiment_budget": ExperimentBudgetStatus,
    "submit_diagnosis": Receipt,
    "submit_fix_set": Receipt,
    "submit_decision": Receipt,
    "memory_write": Receipt,
    "write_learned_skill": LearnedSkillReceipt,
    "request_probe": ProbeReceipt,
    "probe_results": ProbeResult,
    "submit_acceptance_proposal": AcceptanceReceipt,
    "submit_job_program": JobReceipt,
    "save_job": SaveJobReceipt,
    "write_check": CheckReceipt,
    "submit_job_account": Receipt,
}


# Strict JSON schemas.

# Keywords OpenAI's strict mode accepts. Everything else is either dropped
# (annotations and constraints that pydantic checks again on the answer) or
# refused (structure strict mode cannot express).
_STRICT_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "description",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
    }
)
_DROPPED = frozenset(
    {"title", "default", "examples", "uniqueItems", "minLength", "maxLength", "discriminator"}
)
MAX_DEPTH = 10


def _resolve_refs(node: Any, defs: dict[str, Any], seen: tuple[str, ...] = ()) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            if name in seen:
                raise ValueError(f"recursive schema through {name}; strict schemas are inlined")
            target = copy.deepcopy(defs[name])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return _resolve_refs({**target, **extra}, defs, (*seen, name))
        return {k: _resolve_refs(v, defs, seen) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_resolve_refs(v, defs, seen) for v in node]
    return node


def _strict(node: dict[str, Any], where: str) -> dict[str, Any]:
    node = dict(node)
    if "allOf" in node:
        parts = node.pop("allOf")
        if len(parts) != 1:
            raise ValueError(f"{where}: allOf with several parts cannot be made strict")
        node = {**parts[0], **node}
    if "oneOf" in node:
        node["anyOf"] = node.pop("oneOf")
    if "const" in node:
        node["enum"] = [node.pop("const")]
    for key in _DROPPED:
        node.pop(key, None)
    unknown = sorted(set(node) - _STRICT_KEYWORDS)
    if unknown:
        raise ValueError(f"{where}: keywords {unknown} have no strict form")
    if not node:
        raise ValueError(f"{where}: an untyped value has no strict form")
    if "anyOf" in node:
        node["anyOf"] = [_strict(s, f"{where}|{i}") for i, s in enumerate(node["anyOf"])]
    if node.get("type") == "object":
        extra = node.get("additionalProperties")
        if isinstance(extra, dict) or extra is True:
            raise ValueError(f"{where}: a free-form mapping has no strict form")
        properties = {
            name: _strict(schema, f"{where}.{name}")
            for name, schema in node.get("properties", {}).items()
        }
        required = set(node.get("required", []))
        for name, schema in properties.items():
            if name not in required and not _allows_null(schema):
                raise ValueError(
                    f"{where}.{name}: optional without null; make it required or nullable"
                )
        node["properties"] = properties
        node["required"] = list(properties)
        node["additionalProperties"] = False
    if "items" in node:
        node["items"] = _strict(node["items"], f"{where}[]")
    return node


def _allows_null(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "null" or (
        isinstance(schema.get("type"), list) and "null" in schema["type"]
    ):
        return True
    return any(_allows_null(s) for s in schema.get("anyOf", []))


def json_schema_strict(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema in the form OpenAI's strict mode accepts.

    References are inlined, every object lists all its properties as required
    and sets `additionalProperties: false`, `const` becomes a one-value enum,
    annotations without a strict form (titles, defaults, string lengths) are
    dropped (pydantic checks them again on the answer). Raises ValueError for a
    model strict mode cannot express: an optional field that is not nullable,
    a free-form mapping, an untyped value, a recursive model.
    """
    raw = model.model_json_schema(mode="validation")
    inlined = _resolve_refs(raw, raw.get("$defs", {}))
    schema = _strict(inlined, model.__name__)
    problems = strict_schema_problems(schema)
    if problems:
        raise ValueError(f"{model.__name__}: " + "; ".join(problems))
    return schema


def strict_schema_problems(schema: dict[str, Any]) -> list[str]:
    """What keeps a JSON schema from OpenAI's strict mode; empty when it complies."""
    problems: list[str] = []
    if schema.get("type") != "object":
        problems.append("the root is not an object")

    def walk(node: Any, where: str, depth: int) -> None:
        if not isinstance(node, dict):
            problems.append(f"{where}: not a schema")
            return
        if depth > MAX_DEPTH:
            problems.append(f"{where}: nested deeper than {MAX_DEPTH}")
        unknown = sorted(set(node) - _STRICT_KEYWORDS)
        if unknown:
            problems.append(f"{where}: unsupported keywords {unknown}")
        if "type" not in node and "anyOf" not in node and "enum" not in node:
            problems.append(f"{where}: untyped")
        for i, sub in enumerate(node.get("anyOf", [])):
            walk(sub, f"{where}|{i}", depth)
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False:
                problems.append(f"{where}: additionalProperties is not false")
            properties = node.get("properties", {})
            if sorted(node.get("required", [])) != sorted(properties):
                problems.append(f"{where}: not every property is required")
            for name, sub in properties.items():
                walk(sub, f"{where}.{name}", depth + 1)
        if "items" in node:
            walk(node["items"], f"{where}[]", depth + 1)

    walk(schema, "$", 0)
    return problems


def tool_schemas() -> list[dict[str, Any]]:
    """Every tool with its name, description and strict argument schema."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": json_schema_strict(t.arguments),
        }
        for t in TOOLS
    ]
