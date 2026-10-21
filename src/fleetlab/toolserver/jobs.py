"""The job tools of the tool server (docs/design/jobs.md).

The installation takes on a new job: the shift lead's request names the goals that must
hold at the end and a process rule on their order; the agent proposes the acceptance
condition, composes job programs that run after the session on development starts it
names, reads their records in the job tables, writes checks, saves the job and gives the
shift lead its account. The tools are `contracts.tools.JOB_TOOLS`, opt-in like the
probes; the job models are `contracts.jobs`.

Files besides those of `files` (host means the runtime, `fleetlab.jobs.driver`):

    results/operator_state.json   the RequestState, written by the host       read-only
    results/jobs.jsonl            one JobOutcome per line, by the host        read-only
    results/library/              the installation's job list, if mounted     read-only
                                  (jobs/, checks/, recipes/, index.json)
    results/history.json          earlier investigations' public records,     read-only
                                  imported by the host (`PublicHistory`)
    outbox/acceptance.jsonl       every proposal with the shift lead's reply  read-write
    outbox/job_requests.jsonl     one line per accepted submit_job_program    read-write
    outbox/saved_jobs.jsonl       one line per accepted save_job              read-write
    outbox/checks.jsonl           one line per accepted write_check           read-write
    outbox/job_account.json       the accepted account (also submissions.jsonl, kind
                                  "job_account", which makes the session "done")

What is host-owned stays outside the agent's reach: the requested goals, the order rules,
the budget and the start allocation are in the RequestState, which only the host writes;
the delivery check and the order-rule verdicts are columns the host's runner writes into
the job tables. The scripted shift lead's rule (`lead_reply`) is one function, used here
for the immediate reply and again by the host after the session (`fleetlab.jobs.operator`)
before an approval enters the RequestState.

Budgets (`RequestState.budget`) are enforced from the first session: starts per request,
requests per session, starts per investigation. A refused request costs nothing, here or
at the runner; a request whose run broke off stays charged. Outcomes are visible only in
a later session and only when bound to their request and to the export's job_episodes
rows; any contradiction refuses them as a whole, as for experiment and probe results.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import duckdb
from pydantic import Field, ValidationError, model_validator

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.ids import EpisodeId
from fleetlab.contracts.jobs import (
    DELIVERY,
    DEVELOPMENT_INIT_STATES,
    GOAL_SUITE_TASKS,
    HOME,
    JOB_LIMITS,
    RELEASE_INIT_STATES,
    CheckSpec,
    JobBudget,
    JobId,
    OrderRule,
    RequestedGoal,
    RequestState,
    job_id,
    program_problems,
)
from fleetlab.contracts.learned import CheckValue, sql_parameters
from fleetlab.contracts.log import JOB_TABLE_NAMES
from fleetlab.contracts.probes import PROBE_BOUNDS
from fleetlab.contracts.tools import (
    JOB_TOOLS,
    AcceptanceProposal,
    AcceptanceReceipt,
    CheckReceipt,
    JobAccount,
    JobReceipt,
    JobSubmission,
    Receipt,
    RunCheckArgs,
    SaveJobArgs,
    SaveJobReceipt,
    ToolSpec,
    WriteCheckArgs,
)
from fleetlab.toolserver.files import SUBMISSIONS_FILE, append_line, read_lines, replace_text
from fleetlab.toolserver.mcp import ToolResult, image_part, text_part

if TYPE_CHECKING:
    from fleetlab.toolserver.server import ToolBox

__all__ = [
    "ACCEPTANCE_FILE",
    "ACCOUNT_FILE",
    "ACCOUNT_KIND",
    "CHECKS_FILE",
    "CHECK_RUNS_FILE",
    "FIRST_IMPORTED_JOB",
    "HISTORY_FILE",
    "HISTORY_FORMAT",
    "IMPORTED_PARTS",
    "JOB_FRAMES_NOTE",
    "JOB_MESSAGES",
    "JOB_REQUESTS_FILE",
    "JOB_RESULTS_FILE",
    "JOB_TABLES",
    "LEAD_FRACTION",
    "LIBRARY_DIR",
    "LIBRARY_FILES",
    "MIN_RELEASE_STARTS",
    "OPERATOR_FILE",
    "RELEASE_FILE",
    "SAVED_JOBS_FILE",
    "ImportedAccount",
    "ImportedCheck",
    "ImportedEpisode",
    "ImportedEvaluation",
    "ImportedInvestigation",
    "ImportedJob",
    "ImportedProgram",
    "JobBudgetStatus",
    "JobOutcome",
    "JobResult",
    "JobTools",
    "PublicHistory",
    "PublicRule",
    "check_digest",
    "default_budget",
    "goal_text",
    "history_problems",
    "job_outcome_problems",
    "job_tool_extra",
    "lead_reply",
    "proposal_messages",
    "public_release",
    "public_request",
    "read_lead_fraction",
    "read_policy_skills",
    "read_request_state",
    "request_from_public",
    "required_deliveries",
    "rule_text",
]

OPERATOR_FILE = "operator_state.json"  # in results/
JOB_RESULTS_FILE = "job_results.jsonl"  # in results/
LIBRARY_DIR = "library"  # in results/: the mounted library's public form (LIBRARY_FILES)
# The mounted library, one file per kind, each a list of entries (`public_library`).
LIBRARY_FILES = {"jobs": "saved_jobs.json", "checks": "saved_checks.json",
                 "recipes": "saved_recipes.json"}  # fmt: skip
ACCEPTANCE_FILE = "acceptance.jsonl"  # in outbox/
JOB_REQUESTS_FILE = "job_requests.jsonl"  # in outbox/
SAVED_JOBS_FILE = "saved_jobs.jsonl"  # in outbox/
CHECKS_FILE = "checks.jsonl"  # in outbox/
CHECK_RUNS_FILE = "check_runs.jsonl"  # in outbox/: run_check's runs on one job
ACCOUNT_FILE = "job_account.json"  # in outbox/
ACCOUNT_KIND = "job_account"  # its kind in outbox/submissions.jsonl
RELEASE_FILE = "release.json"  # in results/: the release's counts (`public_release`)

# The job tables (contracts.log.SCHEMA, source "export"), keyed by episode_id; the
# episodes table also carries job_id, the delivery verdict, the order-rule verdict and the
# videos by camera (`video_scene`, `video_wrist`), the per-step table the frame index.
JOB_TABLES = JOB_TABLE_NAMES
EPISODES_TABLE = "job_episodes"
DELIVERED_COLUMN = "delivered"
ORDER_COLUMN = "order_kept"
STEPS_TABLE = "robot_steps"

# The scripted shift lead: at least this many release starts, and at least four fifths
# of them delivered (contracts.jobs).
MIN_RELEASE_STARTS = JOB_LIMITS.min_release_starts
MAX_CHECKS = 6
MAX_SAVED_JOBS = 4
MAX_CHECK_SQL = 4000


class _Refused(Exception):
    """Raised through `_refuse` as the server's ToolError (imported late: no cycle)."""


def _refuse(text: str) -> NoReturn:
    from fleetlab.toolserver.server import ToolError

    raise ToolError(text)


def default_budget() -> JobBudget:
    """The plan's ceilings, for a description outside a job session."""
    return JobBudget(
        starts_per_request=JOB_LIMITS.starts_per_request,
        requests_per_session=JOB_LIMITS.requests_per_session,
        sessions=JOB_LIMITS.sessions,
        starts_per_investigation=JOB_LIMITS.starts_per_investigation,
    )


# The share of N the scripted lead asks to be delivered, as (numerator, denominator): four
# fifths by default (`contracts.jobs.JOB_LIMITS`), set at launch by the driver
# (`--lead-fraction`, for example 2/3) and mounted beside the request (`public_request`).
LEAD_FRACTION: tuple[int, int] = (4, 5)


def required_deliveries(n: int, fraction: tuple[int, int] = LEAD_FRACTION) -> int:
    """The fewest deliveries the shift lead accepts of `n` starts: the fraction, rounded up."""
    a, b = fraction
    return -(-a * n // b)


def lead_reply(
    n: int, k: int, release_available: int, fraction: tuple[int, int] = LEAD_FRACTION
) -> tuple[bool, str]:
    """The scripted shift lead's answer to a proposal of `k` deliveries on `n` starts: at
    least `MIN_RELEASE_STARTS` starts and at least `fraction` of them delivered (the rule of
    `contracts.jobs.acceptance_problems` at the default fraction), in the lead's words, which
    name the deliveries needed and never the fraction itself."""
    missing = []
    if n > release_available:
        missing.append(
            f"N is {n}: the installation reserves {release_available} release starts, so N "
            f"can be at most {release_available}"
        )
    if k > n:
        missing.append(f"K is {k}: it cannot exceed N ({n})")
    if not missing:
        if n < MIN_RELEASE_STARTS:
            missing.append(
                f"N is {n}; I need the job shown on at least {MIN_RELEASE_STARTS} release starts"
            )
        elif k < (need := required_deliveries(n, fraction)):
            missing.append(f"K is {k}; I need at least {need} deliveries of {n}")
    if missing:
        return False, "Not yet: " + "; ".join(missing) + ". Send me a new proposal."
    return True, (
        f"Approved: the job is released when it delivers on at least {k} of {n} reserved "
        "release starts it was never run on. This condition stands for the whole request."
    )


def goal_text(goal: RequestedGoal) -> str:
    """A requested goal as LIBERO writes it: the predicate and its objects."""
    return goal.text


def rule_text(state: RequestState, rule: OrderRule) -> str:
    """The shift lead's process rule in words, from the goals it names."""
    own = getattr(rule, "text", None)
    if own:
        return str(own)
    goals = state.requested_goals
    return (
        f"goal {rule.then} ({goal_text(goals[rule.then])}) may not first hold before goal "
        f"{rule.first} ({goal_text(goals[rule.first])}) first holds"
    )


# The mounted files are pipeline state, which the launch audit scans strictly: every key
# and file name must stay clear of the truth side's names (`agents.audit.audit_state`),
# and `first` and `text` are among them. The mounted request therefore writes the order
# rules' goals as `earlier_goal` and `later_goal` and a message's text as `words`; the
# integrator's request file and `contracts.jobs` keep their own names.


def public_request(
    state: RequestState,
    skills: Sequence[Mapping[str, Any]] = (),
    fraction: tuple[int, int] = LEAD_FRACTION,
) -> dict[str, Any]:
    """The RequestState as it is mounted (results/operator_state.json), with the policy's
    skills (`policy_skills`: each goal-suite task's number, instruction and goal) and the
    scripted lead's bar (`lead_bar`), which operator_state does not show. The host's
    allocation profile stays out: how the host derives seeds is not the agent's."""
    data = state.model_dump(mode="json", exclude={"allocation_profile"})
    data["policy_skills"] = [dict(s) for s in skills]
    data["lead_bar"] = {"numerator": fraction[0], "denominator": fraction[1]}
    data["order_rules"] = [
        {"earlier_goal": r["first"], "later_goal": r["then"]} for r in data["order_rules"]
    ]
    data["message_log"] = [
        {"author": m["author"], "words": m["text"], "session": m["session"]}
        for m in data["message_log"]
    ]
    return data


def request_from_public(data: Mapping[str, Any]) -> RequestState:
    """The RequestState back from its mounted form."""
    plain = {k: v for k, v in data.items() if k not in ("policy_skills", "lead_bar")}
    plain["order_rules"] = [
        {"first": r["earlier_goal"], "then": r["later_goal"]} for r in data["order_rules"]
    ]
    plain["message_log"] = [
        {"author": m["author"], "text": m["words"], "session": m["session"]}
        for m in data["message_log"]
    ]
    return RequestState.model_validate(plain)


def read_policy_skills(results: Path) -> list[dict[str, Any]]:
    """The policy's skills as the host mounted them beside the request."""
    path = results / OPERATOR_FILE
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("policy_skills", []))


def read_lead_fraction(results: Path) -> tuple[int, int]:
    """The scripted lead's bar as the host mounted it, or the default."""
    path = results / OPERATOR_FILE
    if not path.is_file():
        return LEAD_FRACTION
    bar = json.loads(path.read_text(encoding="utf-8")).get("lead_bar")
    if not bar:
        return LEAD_FRACTION
    return int(bar["numerator"]), int(bar["denominator"])


def public_release(receipt: Any, session: int) -> dict[str, Any]:
    """A release receipt (`contracts.jobs.ReleaseReceipt`) as it is mounted: the counts, in
    names the launch audit accepts (the receipt's own `pairs`, `rescued`, `broken` and
    `run_dir` are truth-side names)."""
    return {
        "job": receipt.job_name,
        "program_digest": receipt.program_digest,
        "after_session": session,
        "N": receipt.N,
        "K": receipt.K,
        "init_states": [s.init_state for s in receipt.release_starts],
        "delivered": receipt.delivered,
        "naive_delivered": receipt.naive_delivered,
        "program_only": receipt.rescued,
        "naive_only": receipt.broken,
        "order_violations": receipt.order_violations,
        "condition_met": receipt.passed,
        "released": receipt.released,
    }


def read_request_state(results: Path) -> RequestState | None:
    """The host's RequestState in the results, or None outside a job session."""
    path = results / OPERATOR_FILE
    if not path.is_file():
        return None
    return request_from_public(json.loads(path.read_text(encoding="utf-8")))


class JobBudgetStatus(ContractModel):
    """The investigation's job budget and what is left of it."""

    starts_per_request: int
    requests_per_session: int
    sessions: int
    starts_per_investigation: int
    starts_used: int  # charged: every accepted request but those the runner refused
    starts_left: int
    session: int
    requests_this_session: int


# The fixed public messages of a job's outcome, by status; nothing else of the runner's
# reaches the agent. "done": every start ran and its episode is in the tables, charged.
# "refused": the runner turned the program down before anything ran; free. "failed": the
# run broke off; the whole request stays charged and its partial records are withheld.
JOB_MESSAGES: dict[str, str] = {
    "done": "The job ran on every start; its episodes are in job_episodes and the job tables.",
    "refused": "The runner could not run this program; its starts were returned.",
    "failed": (
        "The runner could not complete the job. Its starts stay charged; the episodes it "
        "left are withheld."
    ),
}


class JobOutcome(ContractModel):
    """What the host records of a job request it ran, refused or could not finish
    (`results/jobs.jsonl`): ids, the runner's program digest, and a fixed message."""

    job_id: JobId
    status: Literal["done", "refused", "failed"]
    episode_ids: list[EpisodeId]
    message: str
    program_digest: str = ""  # the runner's digest of the program it ran (lane J's)
    # Why the runner refused the program (contracts.jobs' program checks); only when refused.
    problems: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _public(self) -> JobOutcome:
        if self.message != JOB_MESSAGES[self.status]:
            raise ValueError("a job outcome carries its status's fixed public message")
        if self.problems and self.status != "refused":
            raise ValueError("only a refused job carries the runner's problems")
        if any(len(p) > 300 for p in self.problems):
            raise ValueError("a problem is at most 300 characters")
        if (self.status == "done") != bool(self.episode_ids):
            raise ValueError("a job that ran has episodes; a refused or failed one none")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("a job's episodes are distinct")
        return self


def _null(value: Any) -> bool:
    return value is None


class JobResult(ContractModel):
    """What became of a requested job (`job_results`), or an imported job of an earlier
    investigation (status `imported`, session 0, never charged), with where it came from;
    the fields of an import are left out of a current job's result."""

    job_id: str
    session: int
    status: Literal["queued", "done", "refused", "failed", "imported"]
    name: str
    purpose: str
    program_digest: str
    starts: list[int]
    episode_ids: list[str]
    delivered: int | None  # episodes that delivered, from job_episodes; null unless done
    order_rule_violations: int | None  # episodes whose trace broke a process rule
    starts_charged: int
    message: str
    problems: list[str]  # why the runner refused it, when it did
    investigation: int | None = Field(default=None, exclude_if=_null)
    part: str | None = Field(default=None, exclude_if=_null)
    source_job_id: str | None = Field(default=None, exclude_if=_null)
    source_status: str | None = Field(default=None, exclude_if=_null)
    source_session: int | None = Field(default=None, exclude_if=_null)
    source_episode_ids: list[str] | None = Field(default=None, exclude_if=_null)
    base_task: int | None = Field(default=None, exclude_if=_null)
    requested_goals: list[str] | None = Field(default=None, exclude_if=_null)
    order_rules: list[dict[str, int]] | None = Field(default=None, exclude_if=_null)


# ---- the history of a numbered investigation (results/history.json) --------------------------
#
# The host (`fleetlab.jobs.prior`) imports the public records of earlier investigations of
# the job line: their completed development and final-evaluation episodes under new job and
# episode ids (in the export's job tables, with their videos), their programs, checks, notes
# and accounts, and the names they gave. The file is written once, before the first session;
# the tools validate it as a whole against the export's job_episodes rows before they show
# or count anything (`history_problems`).

HISTORY_FILE = "history.json"  # in results/
HISTORY_FORMAT = "fleetlab.job-history/1"
IMPORTED_PARTS = ("development", "release-candidate", "release-naive")
# Imported job numbers: investigation i's jobs from 1000 + 100 (i - 1) + 1, a development
# job jNNNN keeping NN, the final evaluation's program 98 and its naive chain 99; episode
# numbers keep their last two digits (the start's place in its job). All of them lie beyond
# the job numbers this investigation's requests can reach.
FIRST_IMPORTED_JOB = 1001
IMPORTED_MESSAGE = (
    "Imported from an earlier investigation: its episodes are in the job tables under these "
    "ids; it costs nothing and cannot be saved."
)


class PublicRule(ContractModel):
    """An order rule as mounted: goal `later_goal` may not first hold before `earlier_goal`."""

    earlier_goal: int = Field(ge=0)
    later_goal: int = Field(ge=0)


class ImportedEpisode(ContractModel):
    episode_id: EpisodeId
    source_episode_id: EpisodeId
    init_state: int = Field(ge=0, lt=RELEASE_INIT_STATES.stop)


class ImportedJob(ContractModel):
    """One job of an earlier investigation as imported: its new id and its own, which part
    of that investigation it was, how its run ended there, the program, and the request it
    ran for (its goal indices are that request's own)."""

    job_id: JobId
    investigation: int = Field(ge=1)
    source_job_id: JobId
    part: Literal["development", "release-candidate", "release-naive"]
    source_status: Literal["done", "failed", "interrupted"]
    source_session: int | None
    name: str
    purpose: str
    program_digest: str
    base_task: int = Field(ge=0, lt=len(GOAL_SUITE_TASKS))
    requested_goals: list[str] = Field(min_length=1)
    order_rules: list[PublicRule]
    episodes: list[ImportedEpisode] = Field(min_length=1)


class ImportedEvaluation(ContractModel):
    """An earlier investigation's final evaluation: the saved job's run on its reserved
    starts and the naive chain beside it, with the receipt's counts once it finished."""

    job: str
    status: Literal["done", "interrupted"]
    N: int | None
    K: int | None
    delivered: int | None
    naive_delivered: int | None
    condition_met: bool | None
    released: bool | None
    program_job_id: JobId | None
    naive_job_id: JobId | None


class ImportedAccount(ContractModel):
    job: str | None
    words: str


class ImportedInvestigation(ContractModel):
    investigation: int = Field(ge=1)
    requested_goals: list[str] = Field(min_length=1)
    order_rules: list[PublicRule]
    base_task: int = Field(ge=0, lt=len(GOAL_SUITE_TASKS))
    outcome: str
    sessions: int = Field(ge=0)
    final_evaluation: ImportedEvaluation | None
    notes: str | None
    account: ImportedAccount | None


class ImportedProgram(ContractModel):
    program_digest: str
    program: dict[str, Any]


class ImportedCheck(ContractModel):
    investigation: int = Field(ge=1)
    check: CheckSpec


class PublicHistory(ContractModel):
    """results/history.json: what a numbered investigation imports, in public names."""

    format: Literal["fleetlab.job-history/1"]
    investigations: list[ImportedInvestigation]
    imported: list[ImportedJob]
    programs: list[ImportedProgram]
    checks: list[ImportedCheck]
    job_names: list[str]
    check_names: list[str]


def history_problems(
    history: PublicHistory,
    rows: Mapping[str, Mapping[str, Any]],
    current: Sequence[str] = (),
) -> list[str]:
    """The history against the export's job_episodes rows (episode id -> job_id,
    program_digest, base_task, init_state) and the current requests' job ids: every
    imported job's episodes are exactly its rows, with its digest and its starts; no
    imported id is reachable by a current request or used twice; every program is the text
    of its digest; every job and evaluation names an investigation of the file."""
    from fleetlab.contracts.jobs import JobProgram

    problems = []
    ceiling = JOB_LIMITS.sessions * JOB_LIMITS.requests_per_session
    numbers = [i.investigation for i in history.investigations]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append("the investigations are numbered from 1 in order")
    programs = {}
    for entry in history.programs:
        try:
            digest = JobProgram.model_validate(entry.program).digest
        except ValidationError:
            digest = ""
        if digest != entry.program_digest or entry.program_digest in programs:
            problems.append(f"program {entry.program_digest[:12]} is not its digest's text")
        programs[entry.program_digest] = entry
    ids = [job.job_id for job in history.imported]
    if len(set(ids)) != len(ids):
        problems.append("an imported job id is used twice")
    for both in sorted(set(ids) & set(current)):
        problems.append(f"{both} is imported and requested in this investigation")
    claimed: set[str] = set()
    for job in history.imported:
        if int(job.job_id[1:]) < max(FIRST_IMPORTED_JOB, ceiling + 1):
            problems.append(f"{job.job_id} lies within the ids this investigation's requests use")
        if job.investigation not in numbers:
            problems.append(f"{job.job_id} names no imported investigation")
        if job.program_digest not in programs:
            problems.append(f"{job.job_id}'s program is not in the history")
        mine = [e.episode_id for e in job.episodes]
        if len(set(mine)) != len(mine) or set(mine) & claimed:
            problems.append(f"{job.job_id}: an episode id is used twice")
        claimed.update(mine)
        rows_of_job = {e for e, r in rows.items() if r.get("job_id") == job.job_id}
        if rows_of_job != set(mine):
            problems.append(f"{job.job_id}: its episodes differ from job_episodes")
            continue
        for episode in job.episodes:
            row = rows[episode.episode_id]
            if row.get("program_digest") != job.program_digest:
                problems.append(f"{episode.episode_id} ran another program than {job.job_id}'s")
            if (row.get("base_task"), row.get("init_state")) != (job.base_task, episode.init_state):
                problems.append(f"{episode.episode_id} ran from another start than the history's")
    for investigation in history.investigations:
        evaluation = investigation.final_evaluation
        for named in (evaluation.program_job_id, evaluation.naive_job_id) if evaluation else ():
            if named is not None and named not in ids:
                problems.append(f"investigation {investigation.investigation} names {named}")
    return problems


def job_outcome_problems(
    outcome: JobOutcome,
    request: Mapping[str, Any],
    rows: Mapping[str, str] | None = None,
    digests: Mapping[str, str] | None = None,
) -> list[str]:
    """An outcome against its request and, given, the export's job_episodes rows
    (episode id -> job id, and episode id -> program digest): a done job has one episode
    per start, all of them its own, and ran the program that was queued, by the runner's
    digest and by every episode's."""
    problems = []
    starts = list(request.get("request", {}).get("starts", []))
    if outcome.status == "done" and len(outcome.episode_ids) != len(starts):
        problems.append(
            f"{outcome.job_id}: {len(outcome.episode_ids)} episodes for {len(starts)} starts"
        )
    queued = str(request.get("program_digest", ""))
    if outcome.status == "done" and outcome.program_digest != queued:
        problems.append(f"{outcome.job_id}: the runner ran another program than was queued")
    if (
        digests is not None
        and outcome.status == "done"
        and any(digests.get(e) != queued for e in outcome.episode_ids)
    ):
        problems.append(f"{outcome.job_id}: an episode ran another program than was queued")
    if rows is not None:
        own = {e for e, j in rows.items() if j == outcome.job_id}
        if outcome.status == "done" and own != set(outcome.episode_ids):
            problems.append(f"{outcome.job_id}: its episodes differ from job_episodes")
        if outcome.status != "done" and own:
            problems.append(f"{outcome.job_id}: a job that did not run has episodes")
    return problems


def check_digest(check: CheckSpec) -> str:
    """SHA-256 (hex) over the canonical JSON of a check, so a saved job names the version
    of its check that was run on it."""
    import hashlib

    from fleetlab.contracts.hashing import canonical_json

    return hashlib.sha256(canonical_json(check.model_dump(mode="json")).encode()).hexdigest()


def _scalar(value: str, kind: str) -> Any:
    text = value.strip()
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "bool":
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{value!r} is not true or false")
        return text.lower() == "true"
    return value


_EXTRA = {
    "operator_state": " Read it first and again whenever you need the request's words.",
    "submit_acceptance_proposal": (
        " Before any program is tested. n is the number of reserved release starts, k the "
        "deliveries required of them; the shift lead answers at once. Once approved, the "
        "condition does not change."
    ),
    "submit_job_program": (
        " Steps, of four kinds: policy (instruction: free text the policy reads; until: goal, "
        "an index into the requested goals, or steps; max_steps at most {policy}; on_timeout: "
        "stop ends the program, continue goes on to the next step, null is stop); motion (one "
        "segment of the probe vocabulary: hold, step, ramp, sine, open_gripper, close_gripper; "
        "targets are offsets on x, y or z from the hand's position when the step starts, at "
        "most {offset} m, at most {speed} m/s in a ramp or sine, sine periods of at least "
        "{period} steps, at most {segment} steps; hold, open_gripper and close_gripper take "
        "null axis and amplitude_m; the gripper keeps its last command otherwise); wait "
        "(steps at most {wait}); home (the hand moves in a straight line, "
        "at most {home_speed} m/s with the gripper open, to the pose it had when the job "
        "started and holds until within {home_reach} m of it and at rest; "
        "duration_steps at most {home}; on_timeout as for policy). Fields a kind does not use "
        "are null; at most {program} steps. actions_per_chunk (the program's, for every "
        "policy step): how many actions of each policy chunk the robot executes before the "
        "policy observes again and is called anew, {chunk_min} to {chunk_max}; null executes "
        "the whole chunk ({chunk_max}); a shorter one calls the policy more often, which "
        "costs more inference time. starts: stored initial-state indices from "
        "{dev_first} to {dev_last} (the development starts); the same index is the same start "
        "every time. "
        "Budget: at most {per_request} starts a request, {per_session} requests a session, "
        "{total} starts in the investigation; a refused request costs nothing. Only after the "
        "acceptance condition is approved. End the session after requesting."
    ),
    "job_results": (
        " Per-episode records are in the job tables; query them. Jobs of earlier "
        "investigations of this work follow with status imported and session 0: their part "
        "(development; release-candidate, the saved program on that investigation's final "
        "evaluation starts; release-naive, the naive chain beside it), their own ids there "
        "(source_job_id, source_episode_ids), how their run ended there (source_status), and "
        "the request they ran for: their goal indices are that request's own goals. They cost "
        "nothing and cannot be saved."
    ),
    "save_job": (
        " Only a job that ran, in a session that queued no programs; refused when a recorded "
        "run of the same program broke a process rule. check: the name of the check that goes "
        "with the job, or null; run it on this job with run_check (its job_id) first, and it "
        "is not rewritten once a saved job names it. A name the installation or an earlier "
        "investigation used is taken. The job saved last in a session is "
        "released after the session "
        "on the reserved release starts, against the approved condition, beside the naive "
        "chain on the same starts; the next session reads the counts with operator_state. One "
        "release per investigation: after it, no programs and no saves."
    ),
    "write_check": (
        " One SELECT; parameters as $name, each declared with its type (int, float, str or "
        "bool). check_with gives a value for each parameter, used once to run the "
        "check here and not stored. At most {checks} checks."
    ),
    "run_check": " At most {rows} rows.",
    "job_library": (
        " With history: the earlier investigations of this work (their requests, final "
        "evaluations, notes and accounts), the programs their jobs ran by digest, their checks "
        "(run their SQL with query; run_check runs the installation's and yours), and the job "
        "and check names they used, which are taken."
    ),
    "submit_job_account": (
        " Required to finish, and it ends the investigation: after the release, naming the "
        "released job, with its counts; or with job null when you stop without a release. "
        "Never in a session that queued programs or saved a job. Plain words for the shift "
        "lead: what was asked, the condition, what you tried with its counts, what was "
        "released or why nothing was, the check and how to run it, what is not known."
    ),
}


def job_tool_extra(name: str, budget: JobBudget, max_rows: int) -> str:
    """What a job tool's description adds, with the request's budget."""
    return _EXTRA.get(name, "").format(
        policy=JOB_LIMITS.max_policy_steps,
        wait=JOB_LIMITS.max_wait_steps,
        home=JOB_LIMITS.max_home_steps,
        offset=PROBE_BOUNDS.max_offset_m,
        speed=PROBE_BOUNDS.max_speed_m_s,
        period=PROBE_BOUNDS.min_period_steps,
        segment=PROBE_BOUNDS.max_segment_steps,
        home_speed=HOME.max_speed_m_s,
        home_reach=HOME.reach_m,
        program=JOB_LIMITS.max_program_steps,
        chunk_min=JOB_LIMITS.min_actions_per_chunk,
        chunk_max=JOB_LIMITS.max_actions_per_chunk,
        dev_first=DEVELOPMENT_INIT_STATES.start,
        dev_last=DEVELOPMENT_INIT_STATES.stop - 1,
        per_request=budget.starts_per_request,
        per_session=budget.requests_per_session,
        total=budget.starts_per_investigation,
        checks=MAX_CHECKS,
        rows=max_rows,
    )


# What get_frames adds to its description in a job session (the job tools offered).
JOB_FRAMES_NOTE = (
    " For a job episode: its scene or wrist video by control step; a step without a frame "
    "shows the nearest earlier frame, and the text names the step the frame shows."
)


def job_tool_texts(budget: JobBudget | None = None) -> dict[str, str]:
    """The job tools' descriptions as the agent reads them, and get_frames' as it reads in
    a job session, for the launch gate's texts."""
    from fleetlab.contracts.tools import TOOLS
    from fleetlab.toolserver.server import Limits, tool_description

    texts = {
        f"tool {spec.name}": tool_description(spec, Limits(), job_budget=budget)
        for spec in TOOLS
        if spec.name in JOB_TOOLS
    }
    frames = next(spec for spec in TOOLS if spec.name == "get_frames")
    texts["tool get_frames (job session)"] = tool_description(frames, Limits()) + JOB_FRAMES_NOTE
    return texts


class JobTools:
    """The job tools of one session, over the ToolBox's export, outbox and results."""

    def __init__(self, box: ToolBox) -> None:
        self.box = box

    @property
    def outbox(self) -> Path:
        return self.box.paths.outbox

    @property
    def results(self) -> Path:
        return self.box.paths.results

    def handlers(self) -> dict[str, Any]:
        return {
            "operator_state": self.operator_state,
            "submit_acceptance_proposal": self.submit_acceptance_proposal,
            "submit_job_program": self.submit_job_program,
            "job_results": self.job_results,
            "save_job": self.save_job,
            "write_check": self.write_check,
            "run_check": self.run_check,
            "job_library": self.job_library,
            "submit_job_account": self.submit_job_account,
        }

    # State.

    def state(self) -> RequestState:
        try:
            state = read_request_state(self.results)
        except (ValidationError, ValueError):
            _refuse("the operator state failed validation")
        if state is None:
            _refuse("no job request in this session")
        return state

    def budget(self) -> JobBudget:
        """The request's budget, or the plan's ceilings outside a job session."""
        try:
            state = read_request_state(self.results)
        except (ValidationError, ValueError):
            state = None
        return state.budget if state is not None else default_budget()

    def requests(self) -> list[dict[str, Any]]:
        return list(read_lines(self.outbox / JOB_REQUESTS_FILE))

    def proposals(self) -> list[dict[str, Any]]:
        return list(read_lines(self.outbox / ACCEPTANCE_FILE))

    def acceptance(self, state: RequestState) -> tuple[int, int] | None:
        """The approved condition: the host's, else one approved in this session (an earlier
        session's approval stands only as the host recorded it in the request)."""
        if state.acceptance is not None:
            return state.acceptance.N, state.acceptance.K
        for record in self.proposals():
            if record["approved"] and int(record["session"]) == self.box.session:
                return int(record["proposal"]["N"]), int(record["proposal"]["K"])
        return None

    def release(self) -> dict[str, Any] | None:
        """The release's counts, once the host has run it."""
        path = self.results / RELEASE_FILE
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def saved_here(self) -> list[dict[str, Any]]:
        """The jobs saved in this session; the last of them is released after it."""
        return [
            r for r in read_lines(self.outbox / SAVED_JOBS_FILE)
            if int(r["session"]) == self.box.session
        ]  # fmt: skip

    def queued_here(self) -> list[str]:
        return [r["job_id"] for r in self.requests() if int(r["session"]) == self.box.session]

    def account(self) -> dict[str, Any] | None:
        found = [
            r for r in read_lines(self.outbox / SUBMISSIONS_FILE) if r.get("kind") == ACCOUNT_KIND
        ]
        return found[-1] if found else None

    def episode_rows(self) -> dict[str, str]:
        """job_episodes: episode id -> job id."""
        rows = self.box.reader().query(f"SELECT episode_id, job_id FROM {EPISODES_TABLE}")
        return dict(zip(rows["episode_id"].to_pylist(), rows["job_id"].to_pylist(), strict=True))

    def episode_table(self) -> dict[str, dict[str, Any]]:
        """job_episodes: episode id -> job id, program digest, base task, initial state."""
        columns = "episode_id, job_id, program_digest, base_task, init_state"
        rows = self.box.reader().query(f"SELECT {columns} FROM {EPISODES_TABLE}").to_pylist()
        return {str(r["episode_id"]): r for r in rows}

    def history(self) -> PublicHistory | None:
        """The imported history (results/history.json), validated as a whole against the
        export's rows and this investigation's requests; None without one."""
        path = self.results / HISTORY_FILE
        if not path.is_file():
            return None
        try:
            history = PublicHistory.model_validate_json(path.read_text(encoding="utf-8"))
            requested = [str(r["job_id"]) for r in self.requests()]
            problems = history_problems(history, self.episode_table(), requested)
        except (ValueError, ValidationError, duckdb.Error):
            problems = ["unreadable"]
        if problems:
            _refuse("the imported history failed public validation")
        return history

    def imported(self) -> dict[str, ImportedJob]:
        history = self.history()
        return {} if history is None else {j.job_id: j for j in history.imported}

    def episode_digests(self) -> dict[str, str]:
        """job_episodes: episode id -> the digest of the program the episode ran."""
        rows = self.box.reader().query(f"SELECT episode_id, program_digest FROM {EPISODES_TABLE}")
        ids, digests = rows["episode_id"].to_pylist(), rows["program_digest"].to_pylist()
        return dict(zip(ids, digests, strict=True))

    def outcomes(self) -> dict[str, JobOutcome]:
        """The host's outcomes of earlier sessions' requests, bound to their requests and to
        the export's rows; any problem refuses them as a whole."""
        requests = {r["job_id"]: r for r in self.requests()}
        out: dict[str, JobOutcome] = {}
        imported = {e.episode_id for j in self.imported().values() for e in j.episodes}
        try:
            rows = self.episode_rows()
            digests = self.episode_digests()
            for record in read_lines(self.results / JOB_RESULTS_FILE, strict=True):
                outcome = JobOutcome.model_validate(record)
                request = requests.get(outcome.job_id)
                if request is None or int(request["session"]) >= self.box.session:
                    raise ValueError("an outcome without an earlier request")
                if outcome.job_id in out:
                    raise ValueError("two outcomes for one job")
                if job_outcome_problems(outcome, request, rows, digests):
                    raise ValueError("an outcome that does not match its request")
                out[outcome.job_id] = outcome
            done = {e for o in out.values() for e in o.episode_ids}
            if set(rows) - done - imported:
                raise ValueError("job episodes without an outcome or an import")
        except (ValueError, ValidationError):
            _refuse("job results failed public validation")
        return out

    def budget_status(self, state: RequestState | None = None) -> JobBudgetStatus:
        state = state or self.state()
        budget = state.budget
        refused = {
            job_id for job_id, outcome in self.outcomes().items() if outcome.status == "refused"
        }
        used = sum(
            len(r["request"]["starts"]) for r in self.requests() if r["job_id"] not in refused
        )
        here = sum(1 for r in self.requests() if int(r["session"]) == self.box.session)
        return JobBudgetStatus(
            starts_per_request=budget.starts_per_request,
            requests_per_session=budget.requests_per_session,
            sessions=budget.sessions,
            starts_per_investigation=budget.starts_per_investigation,
            starts_used=used,
            starts_left=max(0, budget.starts_per_investigation - used),
            session=self.box.session,
            requests_this_session=here,
        )

    # The request and the acceptance condition.

    def operator_state(self, args: Any) -> ToolResult:
        state = self.state()
        accepted = self.acceptance(state)
        messages = public_request(state)["message_log"]
        # This investigation's exchanges the host has not copied into its state yet.
        known = {(m["author"], m["words"]) for m in messages}
        for record in self.proposals():
            for author, text in proposal_messages(record):
                if (author, text) not in known:
                    messages.append({"author": author, "words": text, "session": record["session"]})
        out = {
            "request_text": state.request_text,
            "requested_goals": [
                {"index": i, "goal": goal_text(g), **g.model_dump(mode="json")}
                for i, g in enumerate(state.requested_goals)
            ],
            "base_task": {
                "number": state.base_task,
                "task": GOAL_SUITE_TASKS[state.base_task],
                "means": (
                    "the pool of stored starting states of the shared kitchen scene this "
                    "request's starts come from, named after the goal-suite task it was "
                    "recorded for; it is not an instruction, and the work is the requested goals"
                ),
            },
            "order_rules": [
                {"earlier_goal": r.first, "later_goal": r.then, "in_words": rule_text(state, r)}
                for r in state.order_rules
            ],
            "policy_skills": read_policy_skills(self.results),
            "delivered_means": (
                "every requested goal holds at the end, by LIBERO's own predicates on the scene; "
                "nothing of the robot touches the objects the goals name; those objects are at "
                "rest; the gripper is open"
            ),
            "delivery_thresholds": {
                "linear_speed_below_m_s": DELIVERY.linear_m_s,
                "angular_speed_below_rad_s": DELIVERY.angular_rad_s,
                "gripper_open_at_least_m": DELIVERY.gripper_open_m,
                "tail_steps": DELIVERY.tail_steps,
            },
            "development_starts": {
                "from_index": DEVELOPMENT_INIT_STATES.start,
                "to_index": DEVELOPMENT_INIT_STATES.stop - 1,
            },
            "release_starts_reserved": len(RELEASE_INIT_STATES),
            "acceptance": None if accepted is None else {"N": accepted[0], "K": accepted[1]},
            "release": self.release(),
            "release_pending": (
                self.saved_here()[-1]["name"] if self.saved_here() and not self.release() else None
            ),
            "budget": self.budget_status(state).model_dump(mode="json"),
            "message_log": messages,
            "history": self._history_summary(),
        }
        return ToolResult.text(json.dumps(out, ensure_ascii=False))

    def _history_summary(self) -> dict[str, Any] | None:
        history = self.history()
        if history is None:
            return None
        return {
            "investigations": len(history.investigations),
            "imported_jobs": len(history.imported),
            "imported_episodes": sum(len(j.episodes) for j in history.imported),
            "where": (
                "job_results lists the imported jobs with their parts and own ids; job_library "
                "gives the investigations' final evaluations, notes, accounts, programs and checks"
            ),
        }

    def submit_acceptance_proposal(self, proposal: AcceptanceProposal) -> ToolResult:
        state = self.state()
        accepted = self.acceptance(state)
        if accepted is not None:
            receipt = AcceptanceReceipt(
                accepted=False,
                receipt_id="",
                problems=[
                    f"the shift lead approved N={accepted[0]}, K={accepted[1]} already; the "
                    "condition does not change"
                ],
                approved=False,
                reply="",
            )
            return ToolResult.text(receipt.model_dump_json())
        approved, reply = lead_reply(
            proposal.N, proposal.K, len(RELEASE_INIT_STATES), read_lead_fraction(self.results)
        )
        receipt_id = f"acceptance-{len(self.proposals()) + 1}"
        append_line(
            self.outbox / ACCEPTANCE_FILE,
            {
                "receipt_id": receipt_id,
                "session": self.box.session,
                "proposal": proposal.model_dump(mode="json"),
                "approved": approved,
                "reply": reply,
            },
        )
        receipt = AcceptanceReceipt(
            accepted=True, receipt_id=receipt_id, problems=[], approved=approved, reply=reply
        )
        return ToolResult.text(receipt.model_dump_json())

    # Job programs.

    def submit_job_program(self, submission: JobSubmission) -> ToolResult:
        state = self.state()
        status = self.budget_status(state)
        program = submission.program
        problems: list[str] = []
        if self.acceptance(state) is None:
            problems.append(
                "the shift lead has not approved an acceptance condition yet: propose one with "
                "submit_acceptance_proposal before testing any program"
            )
        if self.account() is not None:
            problems.append("your account is submitted; the investigation takes no more jobs")
        if self.release() is not None:
            problems.append("the release has run and its result is final; no more programs")
        elif self.saved_here():
            problems.append(
                "a job is saved in this session and is released after it; no more programs"
            )
        dev = DEVELOPMENT_INIT_STATES
        outside = sorted({s for s in submission.starts if s not in dev})
        if outside:
            problems.append(
                f"starts {outside} are not development starts ({dev.start} to {dev.stop - 1}); "
                "the release starts are reserved for the release run"
            )
        if len(set(submission.starts)) != len(submission.starts):
            problems.append("a request names each start once")
        problems += program_problems(program, state)
        cost = len(submission.starts)
        if cost > status.starts_per_request:
            problems.append(f"at most {status.starts_per_request} starts a request, not {cost}")
        if status.requests_this_session >= status.requests_per_session:
            problems.append(
                f"at most {status.requests_per_session} requests a session; "
                f"{status.requests_this_session} are requested already"
            )
        if status.starts_used + cost > status.starts_per_investigation:
            problems.append(
                f"this request needs {cost} starts and {status.starts_left} of the "
                f"investigation's {status.starts_per_investigation} are left"
            )
        digest = program.digest
        if problems:
            receipt = JobReceipt(
                accepted=False,
                job_id="",
                problems=problems,
                program_digest=digest,
                starts=0,
                starts_used=status.starts_used,
                starts_left=status.starts_left,
                requests_this_session=status.requests_this_session,
            )
            return ToolResult.text(receipt.model_dump_json())
        number = job_id(len(self.requests()) + 1)
        append_line(
            self.outbox / JOB_REQUESTS_FILE,
            {
                "job_id": number,
                "session": self.box.session,
                "program_digest": digest,
                "request": submission.model_dump(mode="json"),
            },
        )
        receipt = JobReceipt(
            accepted=True,
            job_id=number,
            problems=[],
            program_digest=digest,
            starts=cost,
            starts_used=status.starts_used + cost,
            starts_left=max(0, status.starts_left - cost),
            requests_this_session=status.requests_this_session + 1,
        )
        return ToolResult.text(receipt.model_dump_json())

    def _verdicts(self) -> dict[str, tuple[int, int]]:
        """Per job id: episodes delivered and episodes whose trace broke a process rule."""
        rows = (
            self.box.reader()
            .query(
                f"SELECT job_id, count(*) FILTER (WHERE {DELIVERED_COLUMN}) AS delivered, "
                f"count(*) FILTER (WHERE NOT {ORDER_COLUMN}) AS violations "
                f"FROM {EPISODES_TABLE} GROUP BY job_id"
            )
            .to_pylist()
        )
        return {r["job_id"]: (int(r["delivered"]), int(r["violations"])) for r in rows}

    def job_results(self, args: Any) -> ToolResult:
        done = self.outcomes()
        verdicts = self._verdicts()
        out = []
        for record in self.requests():
            request = record["request"]
            outcome = done.get(record["job_id"])
            status = outcome.status if outcome is not None else "queued"
            delivered, violations = verdicts.get(record["job_id"], (None, None))
            result = JobResult(
                job_id=record["job_id"],
                session=int(record["session"]),
                status=status,
                name=request["program"]["name"],
                purpose=request["program"]["purpose"],
                program_digest=record["program_digest"],
                starts=list(request["starts"]),
                episode_ids=list(outcome.episode_ids) if outcome is not None else [],
                delivered=delivered if status == "done" else None,
                order_rule_violations=violations if status == "done" else None,
                starts_charged=0 if status == "refused" else len(request["starts"]),
                message=(
                    JOB_MESSAGES[outcome.status]
                    if outcome is not None
                    else "runs after this session ends; results in the next session"
                ),
                problems=list(outcome.problems) if outcome is not None else [],
            )
            out.append(result.model_dump(mode="json"))
        for job in self.imported().values():
            delivered, violations = verdicts.get(job.job_id, (0, 0))
            imported = JobResult(
                job_id=job.job_id,
                session=0,
                status="imported",
                name=job.name,
                purpose=job.purpose,
                program_digest=job.program_digest,
                starts=[e.init_state for e in job.episodes],
                episode_ids=[e.episode_id for e in job.episodes],
                delivered=delivered,
                order_rule_violations=violations,
                starts_charged=0,
                message=IMPORTED_MESSAGE,
                problems=[],
                investigation=job.investigation,
                part=job.part,
                source_job_id=job.source_job_id,
                source_status=job.source_status,
                source_session=job.source_session,
                source_episode_ids=[e.source_episode_id for e in job.episodes],
                base_task=job.base_task,
                requested_goals=list(job.requested_goals),
                order_rules=[r.model_dump(mode="json") for r in job.order_rules],
            )
            out.append(imported.model_dump(mode="json"))
        return ToolResult.text(json.dumps(out, ensure_ascii=False))

    # The installation's job list.

    def library_root(self) -> Path:
        return self.results / LIBRARY_DIR

    def _library(self, part: str) -> dict[str, dict[str, Any]]:
        """The mounted library's entries of one kind, by name (`LIBRARY_FILES`)."""
        path = self.library_root() / LIBRARY_FILES[part]
        if not path.is_file():
            return {}
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {str(e["name"]): e for e in entries if isinstance(e, dict) and "name" in e}

    def _check_spec(self, name: str) -> CheckSpec | None:
        """The check of that name, this investigation's latest or the library's."""
        own = self.checks()
        if name in own:
            return CheckSpec.model_validate(own[name]["check"])
        library = self._library("checks")
        if name in library:
            return CheckSpec.model_validate(library[name])
        return None

    def saved_jobs(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in read_lines(self.outbox / SAVED_JOBS_FILE):
            latest[record["name"]] = record
        return latest

    def checks(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in read_lines(self.outbox / CHECKS_FILE):
            latest[record["check"]["name"]] = record
        return latest

    def save_job(self, args: SaveJobArgs) -> ToolResult:
        state = self.state()
        problems: list[str] = []
        requests = {r["job_id"]: r for r in self.requests()}
        done = self.outcomes()
        imported = self.imported()
        history = self.history()
        record = requests.get(args.job_id)
        digest = ""
        if record is None and args.job_id in imported:
            problems.append(
                f"{args.job_id} is imported from investigation "
                f"{imported[args.job_id].investigation}: save_job takes a job of this "
                "investigation that ran; run the program here first"
            )
        elif record is None:
            problems.append(f"no job {args.job_id}; job_results lists them")
        elif args.job_id not in done or done[args.job_id].status != "done":
            problems.append(f"{args.job_id} has not run; only a job that ran can be saved")
        else:
            digest = str(record["program_digest"])
            same = [j for j, r in requests.items() if r["program_digest"] == digest and j in done]
            # An imported run of the same program counts when it ran for the same goals and
            # order rules; under another request's its verdicts say nothing about this one.
            contract = (
                [goal_text(g) for g in state.requested_goals],
                [(r.first, r.then) for r in state.order_rules],
            )
            same += [
                j.job_id
                for j in imported.values()
                if j.program_digest == digest
                and (
                    list(j.requested_goals),
                    [(r.earlier_goal, r.later_goal) for r in j.order_rules],
                )
                == contract
            ]
            broken = self._violations(same)
            if broken:
                rules = "; ".join(rule_text(state, rule) for rule in state.order_rules)
                problems.append(
                    f"recorded runs of this program broke the process rule ({rules}): "
                    f"{', '.join(broken)}; the shift lead does not release it"
                )
        if args.name in self._library("jobs"):
            problems.append(f"the installation has a job named {args.name!r}; choose another")
        elif history is not None and args.name in history.job_names:
            problems.append(
                f"an earlier investigation used the job name {args.name!r}; a new candidate "
                "takes a new name"
            )
        bound = ""
        if args.check is not None:
            spec = self._check_spec(args.check)
            if spec is None:
                problems.append(f"no check named {args.check!r}; write it with write_check first")
            else:
                bound = check_digest(spec)
                ran = any(
                    r["check"] == args.check
                    and r["check_digest"] == bound
                    and r["job_id"] == args.job_id
                    for r in read_lines(self.outbox / CHECK_RUNS_FILE)
                )
                if not ran:
                    problems.append(
                        f"{args.check} has not been run on {args.job_id}: run it with run_check "
                        f"and job_id {args.job_id} first, so that the saved job keeps a check "
                        "tested on its runs"
                    )
        if self.account() is not None:
            problems.append("your account is submitted; the investigation saves no more jobs")
        if self.release() is not None:
            problems.append("the release has run and its result is final; nothing more is saved")
        if queued := self.queued_here():
            problems.append(
                f"this session queued {', '.join(queued)}; save a job in a session after reading "
                "the results of its runs"
            )
        saved = self.saved_jobs()
        if args.name not in saved and len(saved) >= MAX_SAVED_JOBS:
            problems.append(f"at most {MAX_SAVED_JOBS} saved jobs ({sorted(saved)} are saved)")
        if problems:
            receipt = SaveJobReceipt(
                accepted=False,
                receipt_id="",
                problems=problems,
                name=args.name,
                program_digest=digest,
            )
            return ToolResult.text(receipt.model_dump_json())
        assert record is not None
        count = sum(1 for _ in read_lines(self.outbox / SAVED_JOBS_FILE))
        receipt_id = f"job-{count + 1}"
        append_line(
            self.outbox / SAVED_JOBS_FILE,
            {
                "receipt_id": receipt_id,
                "session": self.box.session,
                "name": args.name,
                "job_id": args.job_id,
                "purpose": args.purpose,
                "check": args.check,
                "check_digest": bound or None,
                "program_digest": digest,
                "program": record["request"]["program"],
            },
        )
        receipt = SaveJobReceipt(
            accepted=True,
            receipt_id=receipt_id,
            problems=[],
            name=args.name,
            program_digest=digest,
        )
        return ToolResult.text(receipt.model_dump_json())

    def _violations(self, job_ids: Sequence[str]) -> list[str]:
        """Episodes of these jobs whose recorded trace broke a process rule."""
        if not job_ids:
            return []
        rows = self.box.reader().query(
            f"SELECT episode_id FROM {EPISODES_TABLE} WHERE NOT {ORDER_COLUMN} "
            f"AND job_id IN ({', '.join('?' for _ in job_ids)}) ORDER BY episode_id",
            list(job_ids),
        )
        return [str(e) for e in rows["episode_id"].to_pylist()]

    # Checks.

    def _check_problems(self, check: CheckSpec, values: Sequence[CheckValue]) -> list[str]:
        problems: list[str] = []
        if len(check.sql) > MAX_CHECK_SQL:
            return [f"sql: at most {MAX_CHECK_SQL} characters"]
        declared = {p.name: p.type for p in check.parameters}
        if len(declared) != len(check.parameters):
            problems.append("parameters: each name once")
        used = sql_parameters(check.sql)
        if missing := [n for n in used if n not in declared]:
            problems.append(f"sql uses {missing}, not declared under parameters")
        if unused := [n for n in declared if n not in used]:
            problems.append(f"parameters {unused} are not used in the sql")
        given = {v.name: v.value for v in values}
        if lacking := [n for n in used if n not in given]:
            problems.append(f"give a value for {lacking} in check_with, so that it runs once here")
        if problems:
            return problems
        try:
            bound = {n: _scalar(given[n], declared[n]) for n in used}
        except ValueError as error:
            return [f"check_with: {error}"]
        try:
            self._execute(check.sql, bound, None)
        except _Refused as error:
            return [str(error)]
        except duckdb.Error as error:
            return [f"sql failed on the job tables: {error}"]
        return []

    def _execute(
        self, sql: str, params: Mapping[str, Any], job_id: str | None
    ) -> tuple[list[str], list[list[Any]], bool]:
        """One SELECT over the job tables (all of them, or one job's episodes only)."""
        from fleetlab.toolserver.server import _READ_STATEMENTS

        limits = self.box.limits
        main = self.box.reader().con
        try:
            statements = main.extract_statements(sql)
        except duckdb.Error as error:
            raise _Refused(f"sql does not parse: {error}") from None
        if len(statements) != 1 or statements[0].type not in _READ_STATEMENTS:
            raise _Refused("one SELECT, as query takes it")
        con = main
        scoped = None
        if job_id is not None:
            scoped = duckdb.connect(":memory:")
            scoped.execute(f"SET threads = {int(limits.duckdb_threads)}")
            scoped.execute(f"SET memory_limit = '{limits.duckdb_memory}'")
            for table in JOB_TABLES:
                data = main.execute(
                    f"SELECT * FROM {table} WHERE episode_id IN "
                    f"(SELECT episode_id FROM {EPISODES_TABLE} WHERE job_id = ?)",
                    [job_id],
                ).to_arrow_table()
                scoped.register("_part", data)
                scoped.execute(f"CREATE TABLE {table} AS SELECT * FROM _part")
                scoped.unregister("_part")
            scoped.execute("SET enable_external_access = false")
            scoped.execute("SET lock_configuration = true")
            con = scoped
        timer = threading.Timer(limits.query_seconds, con.interrupt)
        timer.start()
        try:
            cursor = con.execute(sql, dict(params) if params else None)
            columns = [c[0] for c in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limits.max_rows + 1) if cursor.description else []
        finally:
            timer.cancel()
            if scoped is not None:
                scoped.close()
        return columns, [list(r) for r in rows[: limits.max_rows]], len(rows) > limits.max_rows

    def write_check(self, args: WriteCheckArgs) -> ToolResult:
        check = args.check
        problems = self._check_problems(check, args.check_with)
        if check.name in self._library("checks"):
            problems.append(f"the installation has a check named {check.name!r}; choose another")
        history = self.history()
        if history is not None and check.name in history.check_names:
            problems.append(
                f"an earlier investigation used the check name {check.name!r}; choose another"
            )
        if any(r.get("check") == check.name for r in self.saved_jobs().values()):
            problems.append(
                f"a saved job names {check.name!r} as its check; write a new one under another name"
            )
        checks = self.checks()
        if check.name not in checks and len(checks) >= MAX_CHECKS:
            problems.append(f"at most {MAX_CHECKS} checks ({sorted(checks)} are written)")
        if problems:
            receipt = CheckReceipt(
                accepted=False, receipt_id="", problems=problems, name=check.name
            )
            return ToolResult.text(receipt.model_dump_json())
        count = sum(1 for _ in read_lines(self.outbox / CHECKS_FILE))
        receipt_id = f"check-{count + 1}"
        append_line(
            self.outbox / CHECKS_FILE,
            {
                "receipt_id": receipt_id,
                "session": self.box.session,
                "check": check.model_dump(mode="json"),
            },
        )
        receipt = CheckReceipt(accepted=True, receipt_id=receipt_id, problems=[], name=check.name)
        return ToolResult.text(receipt.model_dump_json())

    def run_check(self, args: RunCheckArgs) -> ToolResult:
        own = self.checks()
        library = self._library("checks")
        if args.name in own:
            check = CheckSpec.model_validate(own[args.name]["check"])
        elif args.name in library:
            check = CheckSpec.model_validate(library[args.name])
        else:
            _refuse(f"no check named {args.name!r}; job_library lists them")
        declared = {p.name: p.type for p in check.parameters}
        given = {v.name: v.value for v in args.params}
        if set(given) != set(declared):
            _refuse(f"{args.name} takes the parameters {sorted(declared)}; given {sorted(given)}")
        if args.job_id is not None and args.job_id not in self.episode_rows().values():
            _refuse(f"no episodes of {args.job_id} in the job tables")
        try:
            bound = {n: _scalar(given[n], declared[n]) for n in declared}
        except ValueError as error:
            _refuse(f"params: {error}")
        try:
            columns, rows, truncated = self._execute(check.sql, bound, args.job_id)
        except _Refused as error:
            _refuse(str(error))
        if args.job_id is not None:
            append_line(
                self.outbox / CHECK_RUNS_FILE,
                {
                    "session": self.box.session,
                    "check": check.name,
                    "check_digest": check_digest(check),
                    "job_id": args.job_id,
                },
            )
        out = {
            "check": check.name,
            "expects": check.expects,
            "job_id": args.job_id,
            "columns": columns,
            "rows": rows,
            "truncated": truncated,
        }
        return ToolResult.text(json.dumps(out, default=str, ensure_ascii=False))

    def job_library(self, args: Any) -> ToolResult:
        out = {
            "installation": {
                "jobs": list(self._library("jobs").values()),
                "checks": list(self._library("checks").values()),
                "recipes": list(self._library("recipes").values()),
            },
            "this_investigation": {
                "jobs": [
                    {
                        k: r.get(k)
                        for k in ("name", "job_id", "purpose", "check", "program_digest", "program")
                    }
                    for r in self.saved_jobs().values()
                ],
                "checks": [r["check"] for r in self.checks().values()],
            },
        }
        history = self.history()
        if history is not None:
            out["history"] = {
                "investigations": [i.model_dump(mode="json") for i in history.investigations],
                "programs": [p.model_dump(mode="json") for p in history.programs],
                "checks": [c.model_dump(mode="json") for c in history.checks],
                "job_names": list(history.job_names),
                "check_names": list(history.check_names),
            }
        return ToolResult.text(json.dumps(out, ensure_ascii=False))

    # The account.

    def submit_job_account(self, account: JobAccount) -> ToolResult:
        state = self.state()
        problems: list[str] = []
        release = self.release()
        if account.job is not None and self.acceptance(state) is None:
            problems.append("no acceptance condition is approved, so nothing can be released")
        if release is None and self.saved_here():
            problems.append(
                f"{self.saved_here()[-1]['name']} is released after this session; give your "
                "account in the next session, with the release's counts from operator_state"
            )
        elif release is not None and account.job != release["job"]:
            problems.append(f"the released job is {release['job']!r}; name it as job")
        elif release is None and account.job is not None:
            problems.append(f"nothing was released, so job is null; {account.job!r} has no release")
        if here := self.queued_here():
            problems.append(
                f"this session queued {', '.join(here)}; read their results in the next session "
                "and give your account then"
            )
        if problems:
            return ToolResult.text(
                Receipt(accepted=False, receipt_id="", problems=problems).model_dump_json()
            )
        count = sum(
            1 for r in read_lines(self.outbox / SUBMISSIONS_FILE) if r.get("kind") == ACCOUNT_KIND
        )
        receipt_id = f"{ACCOUNT_KIND}-{count + 1}"
        data = account.model_dump(mode="json")
        append_line(
            self.outbox / SUBMISSIONS_FILE,
            {
                "kind": ACCOUNT_KIND,
                "receipt_id": receipt_id,
                "session": self.box.session,
                "answer": data,
            },
        )
        replace_text(self.outbox / ACCOUNT_FILE, json.dumps(data, indent=1, ensure_ascii=False))
        return ToolResult.text(
            Receipt(accepted=True, receipt_id=receipt_id, problems=[]).model_dump_json()
        )

    # Frames of job episodes (get_frames).

    def is_job_episode(self, episode_id: str) -> bool:
        try:
            rows = self.box.reader().query(
                f"SELECT 1 FROM {EPISODES_TABLE} WHERE episode_id = ?", [episode_id]
            )
        except duckdb.Error:
            return False
        return bool(rows.num_rows > 0)

    def frames(self, episode_id: str, steps: Sequence[int], camera: Any) -> ToolResult:
        """The scene video's frames of a job episode at the given control steps."""
        from fleetlab.log.reader import _decode

        reader = self.box.reader()
        name = str(getattr(camera, "value", camera))
        if name not in ("scene", "wrist"):
            _refuse(f"job episodes record the scene and wrist cameras, not {name}")
        row = reader.query(
            f"SELECT video_{name} AS video FROM {EPISODES_TABLE} WHERE episode_id = ?",
            [episode_id],
        ).to_pylist()[0]
        video = row["video"]
        if not video:
            _refuse(f"episode {episode_id} has no {name} video (record_video was false)")
        path = (reader.log_dir / video).resolve()
        if not path.is_relative_to(reader.log_dir) or not path.is_file():
            _refuse(f"episode {episode_id}: no {name} video file")
        logged = reader.query(
            f"SELECT step, frame_index FROM {STEPS_TABLE} WHERE episode_id = ? "
            "AND frame_index IS NOT NULL ORDER BY step",
            [episode_id],
        ).to_pylist()
        if not logged:
            _refuse(f"episode {episode_id} has no frames")
        import bisect

        # A control step without a frame shows the nearest earlier one (a step before the
        # first frame, the first), and the text says which step the frame shows.
        frame_steps = [r["step"] for r in logged]
        chosen = []
        for step in steps:
            j = max(0, bisect.bisect_right(frame_steps, step) - 1)
            chosen.append((int(step), logged[j]["step"], logged[j]["frame_index"]))
        try:
            images = _decode(path, {i for _, _, i in chosen})
        except ValueError as error:
            _refuse(str(error))
        from PIL import Image

        parts: list[dict[str, Any]] = []
        for step, frame_step, frame_index in chosen:
            if frame_step == step:
                which = f"its frame (video frame {frame_index})"
            elif frame_step < step:
                which = (
                    f"no frame at this step; the nearest earlier frame, at step {frame_step} "
                    f"(video frame {frame_index})"
                )
            else:
                which = (
                    f"no frame at or before this step; the first frame, at step {frame_step} "
                    f"(video frame {frame_index})"
                )
            parts.append(text_part(f"{episode_id}, {name} camera, control step {step}: {which}"))
            buffer = io.BytesIO()
            Image.fromarray(images[frame_index]).save(buffer, format="PNG")
            parts.append(image_part(base64.b64encode(buffer.getvalue()).decode()))
        return ToolResult(parts)


def proposal_messages(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """An acceptance record of the outbox as the message log writes it: (author, text)."""
    proposal = record["proposal"]
    return [
        ("agent", f"Proposal: N={proposal['N']}, K={proposal['K']}. {proposal['reason']}"),
        ("lead", record["reply"]),
    ]


def job_description(spec: ToolSpec, budget: JobBudget | None, max_rows: int) -> str:
    """A job tool's description with the request's budget (the plan's ceilings if none)."""
    extra = job_tool_extra(spec.name, budget or default_budget(), max_rows)
    return f"{spec.description[0].upper()}{spec.description[1:]}. Returns {spec.returns}.{extra}"
