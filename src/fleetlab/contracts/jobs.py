"""Jobs: goal-suite skills chained into one job in one continuous simulator state.

The interfaces of the jobs note (docs/design/jobs.md).
A *skill* is one LIBERO goal-suite task run by the fixed policy under an instruction. A
*job* is a program of steps run from one start without a reset between them:

- `policy`: the policy acts under `instruction` (free text) until `until` holds, for at
  most `max_steps` control steps. `until.goal` is an index into the request's requested
  goals and holds at the first control step at which LIBERO's own predicate of that goal
  holds on the loaded scene (`reached`; `timeout` when `max_steps` run out first).
  `until.steps` lets the policy act for exactly that many steps (`reached`). After a
  timeout, `on_timeout` decides: `stop` (the default when absent) ends the program and
  every later step is `skipped`; `continue` goes on with the next step, so a retry is a
  repeated policy step with the same instruction. The chunk left from an earlier step is
  dropped at every step boundary, so a step's first action comes from a fresh policy call.
- `motion`: one probe segment (`contracts.probes.ProbeSegment`, with the probe bounds),
  played like a one-segment probe from the end effector's position and orientation as
  measured at the step's start. Unlike a probe, the gripper keeps its last command
  through `hold`, `step`, `ramp` and `sine`; `open_gripper` and `close_gripper` change it.
- `wait`: the robot holds its last command for `steps` control steps.
- `home`: the hand goes back to the pose the robot reported when the job started, position
  and orientation, the gripper commanded open: the target ramps there no faster than the
  probes' speed bound (0.5 m/s), then holds until the hand is within 1 cm of that position
  and at rest (`reached`) or the step's `duration_steps` (at most 200) run out (`timeout`,
  then `on_timeout`, as for a policy step). `HOME` holds the numbers.

The host owns what the job must achieve (`RequestState`): the requested goals, the base
task whose scene is loaded, the order rules and the budget. Programs, checks and the
acceptance proposal are the agent's; the delivery check, the order-rule verdicts and the
start allocation are the host's (`fleetlab.jobs`). Development starts use LIBERO's stored
initial states 0 to 29 of the base task, release starts 30 to 49 (`DEVELOPMENT_INIT_STATES`,
`RELEASE_INIT_STATES`); a `JobRequest` refuses release starts. The base task supplies the
pool of stored starts of the shared scene, not the work: the requested goals are. How the
starts' seeds derive is the request's host-only `allocation_profile` (`ALLOCATION_PROFILES`;
null, the legacy rule).

Every model is a `ContractModel` (frozen, unknown fields refused). Every field is required
and nullable where a kind does not use it, so the argument models compile to OpenAI's
strict mode (`contracts.tools.json_schema_strict`). Task numbers (`base_task`) are the
goal suite's `tasks_info.txt` order, the plan's numbering, not LIBERO's benchmark index.

This module imports only `fleetlab.contracts` modules the tool server's bundle allows, so
the tool server may import it: not `contracts.identity`, which holds truth
(`fleetlab.agents.toolenv.FORBIDDEN_MODULES`). The host's start allocation, which derives
seeds there, is `fleetlab.jobs.starts`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import Field, StringConstraints, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.env import PLACED_THRESHOLDS
from fleetlab.contracts.hashing import canonical_json
from fleetlab.contracts.ids import EpisodeId
from fleetlab.contracts.learned import PARAMETER_PATTERN, sql_parameters
from fleetlab.contracts.probes import PROBE_BOUNDS, ProbeProgram, ProbeSegment

__all__ = [
    "ALLOCATION_PROFILES",
    "DEFAULT_LEAD_FRACTION",
    "DELIVERY",
    "DEVELOPMENT_INIT_STATES",
    "GOAL_ARITY",
    "GOAL_SUITE",
    "GOAL_SUITE_TASKS",
    "HOME",
    "JOB_ID_PATTERN",
    "JOB_LIMITS",
    "JOB_NAME_PATTERN",
    "NOISE_SEED_LIMIT",
    "POOL_V1",
    "RELEASE_INIT_STATES",
    "SCENE_SEED_LIMIT",
    "Acceptance",
    "AllocationProfile",
    "CheckParameter",
    "CheckParameterType",
    "CheckSpec",
    "DeliveryThresholds",
    "DeliveryVerdict",
    "GoalPredicate",
    "GoalTrace",
    "HomeLimits",
    "JobBudget",
    "JobEpisode",
    "JobId",
    "JobLimits",
    "JobProgram",
    "JobRequest",
    "JobStep",
    "JobStepKind",
    "LibraryEntry",
    "LibraryIndex",
    "LibraryKind",
    "OnTimeout",
    "OrderRule",
    "OrderVerdict",
    "ProgramStepResult",
    "ReleasePair",
    "ReleaseReceipt",
    "RequestMessage",
    "RequestState",
    "RequestedGoal",
    "SavedJob",
    "StartIdentity",
    "StepOutcome",
    "StepUntil",
    "acceptance_problems",
    "job_id",
    "lead_fraction_problems",
    "order_verdicts",
    "program_problems",
]

GOAL_SUITE = "libero_goal"
# The goal suite's tasks in `tasks_info.txt` order: the plan's task numbers (`base_task`).
GOAL_SUITE_TASKS: tuple[str, ...] = (
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "turn_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
    "put_the_cream_cheese_in_the_bowl",
    "open_the_middle_drawer_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_stove",
)

# LIBERO stores 50 initial states per goal-suite task; the host allocates them disjointly.
DEVELOPMENT_INIT_STATES = range(0, 30)
RELEASE_INIT_STATES = range(30, 50)
# The seeds' ranges, as `contracts.identity` fixes them (LIBERO's 32 bits for the scene, a
# signed 64-bit integer for the noise); repeated here because the tool server may not import
# that module. tests/test_jobs_runtime_contracts.py checks they agree.
SCENE_SEED_LIMIT = 2**32
NOISE_SEED_LIMIT = 2**63

JOB_NAME_PATTERN = r"^[a-z][a-z0-9-]{2,39}$"
# A job as the runtime numbered it (`j0001`); not one of the fleet's neutral id kinds.
JOB_ID_PATTERN = r"^j[0-9]{4,}$"
JobId = Annotated[str, StringConstraints(pattern=JOB_ID_PATTERN)]
_OBJECT_PATTERN = r"^[a-z][a-z0-9_]{0,79}$"


def job_id(number: int) -> str:
    """The id of the runtime's job `number`: `job_id(3) == "j0003"`."""
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        raise ValueError(f"a job number is a non-negative integer, not {number!r}")
    return f"j{number:04d}"


class JobLimits(NamedTuple):
    """The bounds of programs, requests and checks (docs/design/jobs.md, section 1)."""

    max_program_steps: int = 8
    max_policy_steps: int = 300  # control steps of one policy step
    max_wait_steps: int = 100
    max_home_steps: int = 200  # a home step's budget, its ramp and its settling together
    max_goals: int = 4
    max_instruction: int = 300  # characters
    max_text: int = 600  # characters of a purpose, an expectation, a message
    max_notes: int = 2000
    max_request_text: int = 2000
    starts_per_request: int = 20
    requests_per_session: int = 4
    # Twelve since 20 October; the requests written for six stay valid.
    sessions: int = 12
    starts_per_investigation: int = 400
    max_check_parameters: int = 8
    max_check_sql: int = 4000
    min_release_starts: int = 20  # the scripted lead approves N >= 20 ...
    min_delivery_fraction: float = 0.8  # ... and K >= 4/5 of N
    # A program's `actions_per_chunk`: the actions of each fresh policy chunk the robot
    # executes before the policy is called again. Null: the whole chunk, X-VLA's 30.
    min_actions_per_chunk: int = 5
    max_actions_per_chunk: int = 30


JOB_LIMITS = JobLimits()


class DeliveryThresholds(NamedTuple):
    """The host-fixed delivery check's numbers, in one place (`fleetlab.jobs.check`).

    At rest: the success spike's thresholds (`contracts.env.PLACED_THRESHOLDS`). Open: the
    last gripper command is open and the fingers are at least `gripper_open_m` apart (the
    widest opening is about 0.08 m). `tail_steps`: after a program whose last step is a
    policy step that reached its goal, the policy goes on under that step's instruction
    until the delivery check holds or this many control steps have passed, the same for
    every program (the fleet's placed check gives LIBERO's check the same tail).
    """

    linear_m_s: float = PLACED_THRESHOLDS["linear_m_s"]
    angular_rad_s: float = PLACED_THRESHOLDS["angular_rad_s"]
    gripper_open_m: float = 0.06
    tail_steps: int = int(PLACED_THRESHOLDS["tail_steps"])


DELIVERY = DeliveryThresholds()


class HomeLimits(NamedTuple):
    """The numbers of a `home` step (coordinator's decision, 19 October).

    The target moves from where the hand is to the start pose at `max_speed_m_s`, the
    probes' bound, and the orientation turns along with it; the step is `reached` once the
    hand is within `reach_m` of the start position and slower than `rest_m_s`, measured
    between consecutive control steps.
    """

    max_speed_m_s: float = PROBE_BOUNDS.max_speed_m_s
    reach_m: float = 0.01
    rest_m_s: float = 0.01


HOME = HomeLimits()


# ---- what the host asks for ------------------------------------------------------------------


class GoalPredicate(StrEnum):
    """LIBERO's goal predicates as the BDDL files spell them."""

    on = "On"
    in_ = "In"
    open_ = "Open"
    close = "Close"
    turnon = "Turnon"
    turnoff = "Turnoff"
    up = "Up"


# How many objects each predicate takes (LIBERO's `envs/predicates`).
GOAL_ARITY: dict[GoalPredicate, int] = {
    GoalPredicate.on: 2,
    GoalPredicate.in_: 2,
    GoalPredicate.open_: 1,
    GoalPredicate.close: 1,
    GoalPredicate.turnon: 1,
    GoalPredicate.turnoff: 1,
    GoalPredicate.up: 1,
}


class RequestedGoal(ContractModel):
    """One goal the job must leave true: a LIBERO predicate over scene names, for example
    `On cream_cheese_1 akita_black_bowl_1` or `Turnon flat_stove_1`."""

    predicate: GoalPredicate
    objects: list[Annotated[str, StringConstraints(pattern=_OBJECT_PATTERN)]] = Field(
        min_length=1, max_length=2
    )

    @model_validator(mode="after")
    def _arity(self) -> RequestedGoal:
        need = GOAL_ARITY[self.predicate]
        if len(self.objects) != need:
            raise ValueError(f"{self.predicate.value} takes {need} object(s), not {self.objects}")
        return self

    @property
    def text(self) -> str:
        return " ".join([self.predicate.value, *self.objects])


class OrderRule(ContractModel):
    """The operator's process rule on goals: goal `then` may not first hold before goal
    `first` first holds (indices into the requested goals)."""

    first: int = Field(ge=0)
    then: int = Field(ge=0)

    @model_validator(mode="after")
    def _distinct(self) -> OrderRule:
        if self.first == self.then:
            raise ValueError("an order rule relates two different goals")
        return self


class JobBudget(ContractModel):
    """What one investigation may spend, within the plan's ceilings."""

    starts_per_request: int = Field(ge=1, le=JOB_LIMITS.starts_per_request)
    requests_per_session: int = Field(ge=1, le=JOB_LIMITS.requests_per_session)
    sessions: int = Field(ge=1, le=JOB_LIMITS.sessions)
    starts_per_investigation: int = Field(ge=1, le=JOB_LIMITS.starts_per_investigation)


class Acceptance(ContractModel):
    """The acceptance condition: delivered on at least K of the N reserved release starts.

    The agent proposes it; the scripted shift lead approves it (`acceptance_problems`)
    before any program is tested.
    """

    N: int = Field(ge=1, le=len(RELEASE_INIT_STATES))
    K: int = Field(ge=0)

    @model_validator(mode="after")
    def _k_within_n(self) -> Acceptance:
        if self.K > self.N:
            raise ValueError(f"K ({self.K}) cannot exceed N ({self.N})")
        return self


# The share of N the scripted lead asks to be delivered, as (numerator, denominator): four
# fifths unless the host launched the investigation with another bar (the driver's
# `--lead-fraction`). A saved job and a release receipt carry the bar their acceptance was
# approved under, so that they validate against it rather than against the default.
DEFAULT_LEAD_FRACTION: tuple[int, int] = (4, 5)
# As a model field: [A, B] (a list, which strict mode can state; a tuple it cannot).
LeadFraction = Annotated[list[int], Field(min_length=2, max_length=2)]


def bar(fraction: Sequence[int] | None) -> tuple[int, int]:
    """A recorded bar as (A, B): the default when none was recorded."""
    if fraction is None:
        return DEFAULT_LEAD_FRACTION
    a, b = fraction
    return int(a), int(b)


def lead_fraction_problems(fraction: Sequence[int]) -> list[str]:
    a, b = fraction
    return [] if 0 < a <= b else [f"the lead's bar {a}/{b} is not a fraction 0 < A/B <= 1"]


def acceptance_problems(
    acceptance: Acceptance, fraction: Sequence[int] = DEFAULT_LEAD_FRACTION
) -> list[str]:
    """Why the scripted lead would not approve a proposal under the bar `fraction` (A of
    every B starts delivered); empty when it would."""
    problems: list[str] = lead_fraction_problems(fraction)
    if problems:
        return problems
    n, k = acceptance.N, acceptance.K
    least = JOB_LIMITS.min_release_starts
    if n < least:
        problems.append(f"N is {n}; the lead asks for at least {least} release starts")
    a, b = fraction
    need = -(-a * n // b)
    if k < need:
        problems.append(f"K is {k}; the lead asks for at least {a}/{b} of N ({need} of {n})")
    return problems


# The host's start allocation profiles (`fleetlab.jobs.starts`). Null, the legacy profile:
# both seeds of a start derive from its initial-state index alone, whatever the base task.
# `pool-v1`: they derive from the base task too, so that another base task's stored starts
# share no seeds with the ones used so far. Host configuration: the agent never chooses it
# and the mounted request leaves it out (`toolserver.jobs.public_request`).
POOL_V1 = "pool-v1"
AllocationProfile = Literal["pool-v1"]
ALLOCATION_PROFILES: tuple[str, ...] = (POOL_V1,)


def _null(value: Any) -> bool:
    """`exclude_if` of a field added later: left out of the dump when null, so that every
    file written before the field existed keeps its bytes and its digest."""
    return value is None


class RequestMessage(ContractModel):
    """One message between the shift lead, the agent and the host."""

    author: Literal["lead", "agent", "host", "operator"]
    text: str = Field(max_length=JOB_LIMITS.max_request_text)
    session: int | None = Field(ge=0)


class RequestState(ContractModel):
    """The host-owned state of one request: what must be true at the end and under which
    rules, the budget, the approved acceptance and the message log. The agent reads it and
    cannot change it."""

    request_text: str = Field(max_length=JOB_LIMITS.max_request_text)
    requested_goals: list[RequestedGoal] = Field(min_length=1, max_length=JOB_LIMITS.max_goals)
    base_task: int = Field(ge=0, lt=len(GOAL_SUITE_TASKS))
    order_rules: list[OrderRule]
    budget: JobBudget
    acceptance: Acceptance | None  # None until the lead has approved a proposal
    message_log: list[RequestMessage]
    # How the host derives the starts' seeds (`ALLOCATION_PROFILES`); null: the legacy rule.
    allocation_profile: AllocationProfile | None = Field(default=None, exclude_if=_null)

    @model_validator(mode="after")
    def _rules_name_goals(self) -> RequestState:
        n = len(self.requested_goals)
        for i, rule in enumerate(self.order_rules):
            if rule.first >= n or rule.then >= n:
                raise ValueError(f"order rule {i} names a goal beyond the {n} requested")
        if len({g.text for g in self.requested_goals}) != n:
            raise ValueError("each requested goal is named once")
        return self


# ---- what the agent writes -------------------------------------------------------------------


class JobStepKind(StrEnum):
    policy = "policy"
    motion = "motion"
    wait = "wait"
    home = "home"


class OnTimeout(StrEnum):
    """What a policy step's timeout does: end the program, or go on with the next step."""

    stop = "stop"
    continue_ = "continue"


class StepUntil(ContractModel):
    """When a policy step ends: when a requested goal holds, or after a number of steps.
    Exactly one is set; the other is null."""

    goal: int | None = Field(ge=0)
    steps: int | None = Field(ge=1, le=JOB_LIMITS.max_policy_steps)

    @model_validator(mode="after")
    def _one(self) -> StepUntil:
        if (self.goal is None) == (self.steps is None):
            raise ValueError("until names a goal or a number of steps (one; the other null)")
        return self


class JobStep(ContractModel):
    """One step of a program; the fields a kind does not use are null.

    policy: `instruction`, `until`, `max_steps`, `on_timeout` (`stop` when absent or null);
    motion: `segment`; wait: `steps`; home: `duration_steps`, `on_timeout`.
    """

    kind: JobStepKind
    instruction: str | None = Field(max_length=JOB_LIMITS.max_instruction)
    until: StepUntil | None
    max_steps: int | None = Field(ge=1, le=JOB_LIMITS.max_policy_steps)
    segment: ProbeSegment | None
    steps: int | None = Field(ge=1, le=JOB_LIMITS.max_wait_steps)
    # Nullable with a default so that strict mode lists it as required-or-null and a stored
    # program without it still loads; a policy step always carries it explicitly (below), so
    # an absent value and `stop` give the same digest.
    on_timeout: OnTimeout | None = None
    # A home step's budget. Left out of the dump when null, so that every program written
    # before the home step existed keeps its digest; strict mode lists it as required-or-null.
    duration_steps: int | None = Field(
        default=None,
        ge=1,
        le=JOB_LIMITS.max_home_steps,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="before")
    @classmethod
    def _stop_by_default(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("kind") in (JobStepKind.policy, JobStepKind.home)
            and data.get("on_timeout") is None
        ):
            return {**data, "on_timeout": OnTimeout.stop}
        return data

    @model_validator(mode="after")
    def _fields_fit(self) -> JobStep:
        used = {
            JobStepKind.policy: ("instruction", "until", "max_steps", "on_timeout"),
            JobStepKind.motion: ("segment",),
            JobStepKind.wait: ("steps",),
            JobStepKind.home: ("duration_steps", "on_timeout"),
        }[self.kind]
        names = ("instruction", "until", "max_steps", "segment", "steps", "on_timeout")
        for name in (*names, "duration_steps"):
            value = getattr(self, name)
            if name in used and value is None:
                raise ValueError(f"a {self.kind.value} step needs {name}")
            if name not in used and value is not None:
                raise ValueError(f"a {self.kind.value} step takes no {name} (null)")
        if self.kind is JobStepKind.policy:
            assert self.instruction is not None and self.until is not None
            assert self.max_steps is not None
            if not self.instruction.strip():
                raise ValueError("a policy step needs a non-empty instruction")
            if self.until.steps is not None and self.until.steps > self.max_steps:
                raise ValueError(
                    f"until.steps ({self.until.steps}) exceeds max_steps ({self.max_steps})"
                )
        if self.segment is not None:
            ProbeProgram(segments=[self.segment])  # the probe bounds, ramp speed included
        return self

    @property
    def budget(self) -> int:
        """The most control steps this step can take."""
        if self.kind is JobStepKind.policy:
            assert self.max_steps is not None
            return self.max_steps
        if self.kind is JobStepKind.motion:
            assert self.segment is not None
            return self.segment.duration_steps
        if self.kind is JobStepKind.home:
            assert self.duration_steps is not None
            return self.duration_steps
        assert self.steps is not None
        return self.steps


class JobProgram(ContractModel):
    """A job: its name, what it is for, its steps in order, and notes. The digest is the
    SHA-256 of the canonical JSON dump, so a frozen program is named by its text.

    `actions_per_chunk`: how many actions of each fresh policy chunk the robot executes
    before the policy is called again, in every policy step of the program; null executes
    the whole chunk (X-VLA's 30), as every program did before the field existed. Left out
    of the dump when null, so those programs keep their digests; a program that sets it,
    to 30 as well, is another program with another digest.
    """

    name: Annotated[str, StringConstraints(pattern=JOB_NAME_PATTERN)]
    purpose: str = Field(max_length=JOB_LIMITS.max_text)
    steps: list[JobStep] = Field(min_length=1, max_length=JOB_LIMITS.max_program_steps)
    notes: str = Field(max_length=JOB_LIMITS.max_notes)
    actions_per_chunk: int | None = Field(
        default=None,
        ge=JOB_LIMITS.min_actions_per_chunk,
        le=JOB_LIMITS.max_actions_per_chunk,
        exclude_if=_null,
    )

    @property
    def digest(self) -> str:
        """SHA-256 (hex) over the canonical JSON of the program (sorted keys, no spaces)."""
        text = canonical_json(self.model_dump(mode="json"))
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def max_control_steps(self) -> int:
        """The most control steps the program can take, the delivery tail excluded."""
        return sum(step.budget for step in self.steps)


def program_problems(program: JobProgram, state: RequestState) -> list[str]:
    """What keeps a program from running against a request; empty when it can."""
    n = len(state.requested_goals)
    problems = []
    for i, step in enumerate(program.steps):
        if step.until is not None and step.until.goal is not None and step.until.goal >= n:
            problems.append(f"step {i}: until.goal {step.until.goal}; the request has {n} goals")
    return problems


class StartIdentity(ContractModel):
    """One start: the base task whose scene is loaded (the plan's task number), LIBERO's
    stored initial-state index, the scene seed (set before the reset; it places the
    furniture) and the seed of the policy's sampling noise."""

    base_task: int = Field(ge=0, lt=len(GOAL_SUITE_TASKS))
    init_state: int = Field(ge=0, lt=RELEASE_INIT_STATES.stop)
    scene_seed: int = Field(ge=0, lt=SCENE_SEED_LIMIT)
    noise_seed: int = Field(ge=0, lt=NOISE_SEED_LIMIT)

    @property
    def is_release(self) -> bool:
        return self.init_state in RELEASE_INIT_STATES


class JobRequest(ContractModel):
    """A program and the development starts it runs on, with or without video."""

    program: JobProgram
    starts: list[StartIdentity] = Field(min_length=1, max_length=JOB_LIMITS.starts_per_request)
    record_video: bool

    @model_validator(mode="after")
    def _development_only(self) -> JobRequest:
        release = sorted({s.init_state for s in self.starts if s.is_release})
        if release:
            raise ValueError(
                f"initial states {release} are reserved for release runs; a request runs on "
                f"development starts ({DEVELOPMENT_INIT_STATES.start} to "
                f"{DEVELOPMENT_INIT_STATES.stop - 1}) only"
            )
        if len({s.model_dump_json() for s in self.starts}) != len(self.starts):
            raise ValueError("a request names each start once")
        if len({s.base_task for s in self.starts}) != 1:
            raise ValueError("a request's starts share one base task")
        return self


class CheckParameterType(StrEnum):
    int_ = "int"
    float_ = "float"
    str_ = "str"
    bool_ = "bool"


class CheckParameter(ContractModel):
    """A named parameter of a check, written `$name` in its SQL."""

    name: Annotated[str, StringConstraints(pattern=PARAMETER_PATTERN)]
    type: CheckParameterType


class CheckSpec(ContractModel):
    """A check the agent writes: one SELECT over the job tables with named parameters,
    saved under a name with its purpose and what it expects."""

    name: Annotated[str, StringConstraints(pattern=JOB_NAME_PATTERN)]
    purpose: str = Field(max_length=JOB_LIMITS.max_text)
    parameters: list[CheckParameter] = Field(max_length=JOB_LIMITS.max_check_parameters)
    sql: str = Field(min_length=1, max_length=JOB_LIMITS.max_check_sql)
    expects: str = Field(max_length=JOB_LIMITS.max_text)

    @model_validator(mode="after")
    def _parameters_declared(self) -> CheckSpec:
        declared = [p.name for p in self.parameters]
        if len(set(declared)) != len(declared):
            raise ValueError("each parameter is declared once")
        used = sql_parameters(self.sql)
        missing = [p for p in used if p not in declared]
        unused = [p for p in declared if p not in used]
        if missing:
            raise ValueError(f"the SQL uses undeclared parameters {missing}")
        if unused:
            raise ValueError(f"declared parameters the SQL never uses: {unused}")
        return self


# ---- what a run records ----------------------------------------------------------------------


class StepOutcome(StrEnum):
    reached = "reached"
    timeout = "timeout"
    skipped = "skipped"


class GoalTrace(ContractModel):
    """One requested goal over a run: the first control step it held and whether it held
    at the end."""

    goal: int = Field(ge=0)
    first_held_step: int | None = Field(ge=0)
    held_at_end: bool


class ProgramStepResult(ContractModel):
    """One program step as it ran: its outcome, the control steps it used and where."""

    step: int = Field(ge=0)
    kind: JobStepKind
    outcome: StepOutcome
    steps_used: int = Field(ge=0)
    start_control_step: int | None = Field(ge=0)  # null when skipped
    end_control_step: int | None = Field(ge=0)


class OrderVerdict(ContractModel):
    """One order rule on one run's recorded trace."""

    rule: int = Field(ge=0)
    first: int = Field(ge=0)
    then: int = Field(ge=0)
    kept: bool
    detail: str


def order_verdicts(rules: list[OrderRule], first_held: list[int | None]) -> list[OrderVerdict]:
    """The order rules on a run's goal times: rule (A, B) is broken when B first held at a
    control step before A first held, or B held while A never did."""
    out = []
    for i, rule in enumerate(rules):
        a, b = first_held[rule.first], first_held[rule.then]
        if b is None:
            kept, detail = True, f"goal {rule.then} never held"
        elif a is None:
            kept, detail = False, f"goal {rule.then} held at step {b}, goal {rule.first} never"
        elif b < a:
            kept, detail = False, f"goal {rule.then} held at step {b}, goal {rule.first} at {a}"
        else:
            kept, detail = True, f"goal {rule.first} held at step {a}, goal {rule.then} at {b}"
        out.append(OrderVerdict(rule=i, first=rule.first, then=rule.then, kept=kept, detail=detail))
    return out


class DeliveryVerdict(ContractModel):
    """The host-fixed delivery check at the end of a run, with what failed."""

    delivered: bool
    goals_hold: list[bool]  # per requested goal, at the end
    robot_touches: list[str]  # named objects a robot body touches at the end
    not_at_rest: list[str]  # named free objects moving faster than the thresholds
    gripper_commanded_open: bool
    gripper_opening_m: float
    reason: str


class JobEpisode(ContractModel):
    """One run of a program from one start: the delivery verdict, the goals' times, each
    program step's outcome, the order-rule verdicts and the end state's hash."""

    job_id: JobId
    episode_id: EpisodeId
    start: StartIdentity
    program_digest: str
    delivered: bool
    goals: list[GoalTrace]
    program_steps: list[ProgramStepResult]
    tail_steps: int = Field(ge=0)  # the delivery tail after the program (DELIVERY.tail_steps)
    control_steps: int = Field(ge=0)  # control steps from the reset to the end
    order_verdicts: list[OrderVerdict]
    order_kept: bool
    delivery: DeliveryVerdict
    end_state_hash: str
    # The pose the robot reported when the job started (after the reset): a home step's
    # target, and what a check measures how far each step began from.
    start_eef_pos: list[float] | None = Field(default=None, min_length=3, max_length=3)
    start_eef_quat: list[float] | None = Field(default=None, min_length=4, max_length=4)


class ReleasePair(ContractModel):
    """One release start: the program's run and the naive chain's run on it."""

    start: StartIdentity
    episode_id: EpisodeId
    delivered: bool
    order_kept: bool
    naive_episode_id: EpisodeId
    naive_delivered: bool


class ReleaseReceipt(ContractModel):
    """A frozen program run on the reserved release starts beside the naive chain."""

    job_name: str
    program_digest: str
    naive_digest: str
    release_starts: list[StartIdentity]
    N: int = Field(ge=1)
    K: int | None  # the approved K; null when the operator ran without an acceptance
    delivered: int = Field(ge=0)  # of N
    naive_delivered: int = Field(ge=0)  # of N, on the same starts
    rescued: int = Field(ge=0)  # delivered by the program, not by the naive chain
    broken: int = Field(ge=0)  # delivered by the naive chain, not by the program
    order_violations: int = Field(ge=0)  # the program's runs that broke an order rule
    passed: bool | None  # delivered >= K; null without an approved acceptance of size N
    released: bool  # passed, no order violation, and every start fresh to the request
    pairs: list[ReleasePair]
    run_dir: str  # where the runs are, for the record
    # Whether none of these release starts had been examined for the request (its base task
    # and requested goals) by an earlier release attempt of the library, recorded before the
    # attempt ran (`fleetlab.jobs.library`, the release exposure log); null when unknown.
    starts_fresh: bool | None = None
    # The lead's bar the acceptance was approved under (the saved job's `lead_fraction`).
    lead_fraction: LeadFraction | None = None  # null: the default, four fifths

    @model_validator(mode="after")
    def _counts(self) -> ReleaseReceipt:
        n = self.N
        if len(self.release_starts) != n or len(self.pairs) != n:
            raise ValueError("N is the number of release starts and of pairs")
        if [p.start for p in self.pairs] != self.release_starts:
            raise ValueError("the pairs ran from the release starts, one each, in their order")
        states = [s.init_state for s in self.release_starts]
        if len(set(states)) != n:
            raise ValueError("each release start appears once")
        if not all(s.is_release for s in self.release_starts):
            raise ValueError("a release receipt runs on the reserved release starts only")
        if len({s.base_task for s in self.release_starts}) != 1:
            raise ValueError("the release starts share one base task")
        ids = [p.episode_id for p in self.pairs] + [p.naive_episode_id for p in self.pairs]
        if len(set(ids)) != len(ids):
            raise ValueError("every episode of the program and of the naive chain is its own")
        if self.delivered != sum(p.delivered for p in self.pairs):
            raise ValueError("delivered counts the pairs the program delivered")
        if self.naive_delivered != sum(p.naive_delivered for p in self.pairs):
            raise ValueError("naive_delivered counts the pairs the naive chain delivered")
        if self.rescued != sum(p.delivered and not p.naive_delivered for p in self.pairs):
            raise ValueError("rescued counts program-only deliveries")
        if self.broken != sum(p.naive_delivered and not p.delivered for p in self.pairs):
            raise ValueError("broken counts naive-only deliveries")
        if self.order_violations != sum(not p.order_kept for p in self.pairs):
            raise ValueError("order_violations counts the program's runs that broke a rule")
        k = self.K
        if k is None and self.passed is not None:
            raise ValueError("passed needs the approved K")
        if self.passed is not None:
            assert k is not None
            if acceptance_problems(Acceptance(N=n, K=k), bar(self.lead_fraction)):
                raise ValueError(f"N {n} and K {k} are not an acceptance the lead approves")
            if self.passed != (self.delivered >= k):
                raise ValueError("passed is whether at least K of the N were delivered")
        if self.released and not (
            self.passed is True and self.order_violations == 0 and self.starts_fresh is True
        ):
            raise ValueError("released needs passed, no order violation and fresh starts")
        return self


# ---- the job library -------------------------------------------------------------------------


class SavedJob(ContractModel):
    """A job in the installation's library (`jobs/<name>.json`): the frozen program with
    what it must achieve and the acceptance condition approved for it, the job ids of the
    recorded runs it was saved from, and its release receipt once it has one."""

    name: Annotated[str, StringConstraints(pattern=JOB_NAME_PATTERN)]
    program: JobProgram
    requested_goals: list[RequestedGoal] = Field(min_length=1, max_length=JOB_LIMITS.max_goals)
    base_task: int = Field(ge=0, lt=len(GOAL_SUITE_TASKS))
    order_rules: list[OrderRule]
    acceptance: Acceptance | None  # the request's approved condition; a release run reads K
    saved_from: list[JobId]
    receipt: ReleaseReceipt | None
    # The lead's bar the host launched the request with and approved `acceptance` under.
    lead_fraction: LeadFraction | None = None  # null: the default, four fifths
    # The check the job was saved with, and the SHA-256 of the version that was run on the
    # recorded runs it was saved from (`toolserver.jobs.check_digest`); null without one.
    check: str | None = None
    check_digest: str | None = None
    # The host's allocation profile of the request the job was saved for, so that an
    # operator's run derives its starts as the request did; null: the legacy rule.
    allocation_profile: AllocationProfile | None = Field(default=None, exclude_if=_null)

    @model_validator(mode="after")
    def _approvable(self) -> SavedJob:
        if lead_fraction_problems(bar(self.lead_fraction)):
            raise ValueError("; ".join(lead_fraction_problems(bar(self.lead_fraction))))
        if self.receipt is not None and bar(self.receipt.lead_fraction) != bar(self.lead_fraction):
            raise ValueError("a saved job's receipt was judged under the job's own bar")
        if self.acceptance is not None:
            problems = acceptance_problems(self.acceptance, bar(self.lead_fraction))
            if problems:
                raise ValueError(
                    "the acceptance is not one the lead approves: " + "; ".join(problems)
                )
        if (self.check is None) != (self.check_digest is None):
            raise ValueError("a saved job's check comes with the digest of its tested version")
        if self.receipt is not None and self.receipt.job_name != self.name:
            raise ValueError("a saved job's receipt is its own")
        return self


class LibraryKind(StrEnum):
    job = "job"
    check = "check"
    recipe = "recipe"


class LibraryEntry(ContractModel):
    """One line of the library's `index.json`."""

    kind: LibraryKind
    name: Annotated[str, StringConstraints(pattern=JOB_NAME_PATTERN)]
    purpose: str
    digest: str  # SHA-256 of the saved file's canonical JSON


class LibraryIndex(ContractModel):
    entries: list[LibraryEntry]
