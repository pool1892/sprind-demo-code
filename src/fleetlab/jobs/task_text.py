"""The task text the jobs driver gives its agent, one per session (plan lane K).

`task_text(state, session, library_mounted)` returns the text of one session of a job
investigation. It is pipeline text in the style of the loop's round task
(`fleetlab.loop.rounds.round_task`): the shift lead's request, the requested goals and
the order rules, the installation's definition of delivered, the order of the work
(the acceptance proposal first; programs on development starts; the check; release or
stop; the account), the tools, the budget and the skills. It names no job, no
condition and no difficulty, and nothing the agent wrote is pasted into it: the shift
lead's replies, the budget left and a release receipt are read with `operator_state`.

`state` is the host's `RequestState` (`fleetlab.contracts.jobs`; docs/design/jobs.md,
section 4), read by duck typing: attributes or mapping keys alike, nested items too. Read
are `request_text`, `requested_goals` (each with `predicate` and `objects`, rendered as
LIBERO spells them, `On <object> <object>`), `base_task` (the goal-suite
task number, named from `GOAL_SUITE_TASKS` when `fleetlab.contracts.jobs` is importable),
`order_rules` (each with `first` and `then`, or as mounted `earlier_goal` and
`later_goal`: goal indices, a goal itself accepted too), `budget` (`starts_per_request`,
`requests_per_session`, `sessions`, `starts_per_investigation`) and `acceptance` (`N` and
`K`, or `n` and `k`, once approved; empty before). A missing attribute raises `KeyError`
naming it.

Sessions count from 1. A later session's text starts with what to read first and
then repeats the whole task, as the round driver's continuations do. With
`library_mounted` the text says that the job library holds an earlier request's jobs
and checks, which may be used or ignored. With `history` (the imported history's
`investigations`, `jobs` and `episodes`, `fleetlab.jobs.prior`) it says where the
earlier investigations of this job line are and what of them may be used; without it the
text is as it was before the import existed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["JOB_SKILLS", "goal_text", "task_text"]

# The base skills of a job investigation, in the order the task text names them.
JOB_SKILLS: tuple[str, ...] = (
    "jobs-traces",
    "jobs-installation",
    "jobs-release",
    "jobs-decompose",
    "jobs-practice",
    "jobs-preserve",
    "jobs-transitions",
    "jobs-counts",
    "jobs-check",
)

_MISSING = object()


def _get(obj: Any, name: str, default: Any = _MISSING) -> Any:
    """`obj[name]` for a mapping, `obj.name` otherwise; `default` when absent if given."""
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)
    if default is _MISSING:
        raise KeyError(f"the request state has no {name!r}")
    return default


def goal_text(goal: Any) -> str:
    """A requested goal as LIBERO spells it: the predicate, then its objects (`On a b`)."""
    objects = _get(goal, "objects")
    if isinstance(objects, str):
        objects = [objects]
    predicate = _get(goal, "predicate")
    predicate = getattr(predicate, "value", predicate)  # an enum member, or its text
    return " ".join([str(predicate), *(str(o) for o in objects)])


def _goal_ref(ref: Any, goals: Sequence[Any]) -> str:
    """An order rule's goal, an index or a goal, as `goal 1, On(a, b)`."""
    if isinstance(ref, int) and not isinstance(ref, bool):
        if not 0 <= ref < len(goals):
            raise ValueError(f"an order rule names goal {ref}, and the request has {len(goals)}")
        return f"goal {ref} ({goal_text(goals[ref])})"
    text = goal_text(ref)
    for index, goal in enumerate(goals):
        if goal_text(goal) == text:
            return f"goal {index} ({text})"
    return text


def _rule_goal(rule: Any, side: str) -> Any:
    """An order rule's earlier or later goal, in the contracts' names (`first`, `then`) or
    the mounted state's (`earlier_goal`, `later_goal`)."""
    mounted = _get(rule, f"{side}_goal", None)
    if mounted is not None:
        return mounted
    return _get(rule, "first" if side == "earlier" else "then")


def _acceptance(state: Any) -> tuple[int, int] | None:
    """The approved (N, K), or None before approval."""
    found = _get(state, "acceptance", None)
    if not found:
        return None
    n = _get(found, "N", None)
    k = _get(found, "K", None)
    n = _get(found, "n", None) if n is None else n
    k = _get(found, "k", None) if k is None else k
    if n is None or k is None:
        raise KeyError("the request state's acceptance has no N and K")
    return int(n), int(k)


def _request(state: Any) -> str:
    goals = list(_get(state, "requested_goals"))
    if not goals:
        raise ValueError("the request state names no requested goal")
    rules = list(_get(state, "order_rules"))
    lines = [
        f'The request. In the shift lead\'s words: "{str(_get(state, "request_text")).strip()}"',
        "",
        "The requested goals, all of which must hold at the end, numbered as the job tables "
        "number them:",
        *(f"- goal {i}: {goal_text(g)}" for i, g in enumerate(goals)),
    ]
    if rules:
        lines += ["", "The order rules, checked on the recorded trace of every run:"]
        lines += [
            f"- {_goal_ref(_rule_goal(r, 'later'), goals)} may not hold before "
            f"{_goal_ref(_rule_goal(r, 'earlier'), goals)} has held."
            for r in rules
        ]
    else:
        lines += ["", "The request has no order rule."]
    lines += [
        "",
        "Delivered "
        "means, as the installation checks it at the end of every run: every requested goal "
        "holds, nothing of the robot touches the objects the goals name, those objects are at "
        "rest, and the gripper is open; operator_state gives the thresholds. The requested "
        "goals, the order rules and the delivery check are the installation's, and you cannot "
        "change them.",
    ]
    return "\n".join(lines)


def _budget(state: Any) -> str:
    budget = _get(state, "budget")
    return (
        f"The budget of this investigation: at most {_get(budget, 'starts_per_request')} starts "
        f"a request, {_get(budget, 'requests_per_session')} requests a session, "
        f"{_get(budget, 'sessions')} sessions, and {_get(budget, 'starts_per_investigation')} "
        "starts in all. A refused request costs nothing. operator_state shows what is left."
    )


def _condition(state: Any) -> str:
    approved = _acceptance(state)
    if approved is None:
        return (
            "No acceptance condition has been approved yet: the proposal comes before "
            "anything else."
        )
    n, k = approved
    return (
        f"The approved acceptance condition: the released program is delivered on at least "
        f"{k} of {n} reserved release starts."
    )


_WORK = """How the work goes.

1. The acceptance condition. Before any program runs, propose with
submit_acceptance_proposal on how many reserved release starts the job is to run (N) and
on how many of them it must be delivered (K), with your reason. The shift lead approves
the proposal or says what is missing; operator_state shows the reply. Nothing runs before
a condition is approved.

2. Programs. A program is a list of steps the installation runs in one continuous scene
from one start: policy steps (an instruction, an until condition, a step budget), motion
steps (one segment of the motion vocabulary of the hand and the gripper), waits, and home
steps (the hand back to the pose it started the job in, gripper open).
operator_state lists the skills the policy has (each with its goal-suite task, its
instruction and the goal it achieves on its own), the development starts, how many
starts are reserved for releases, and the delivery check's thresholds. Submit programs
with submit_job_program on development starts, named by their stored initial states;
release starts are refused. Requests run after this session ends. The next session finds
them with job_results, their records in the tables job_episodes, job_step_events,
goal_steps, object_steps and robot_steps, queried like any table with query, and their
frames with get_frames.

3. The check. Write a check with write_check: a named SQL query with parameters over the
job tables that answers a question about a run's episodes, with its purpose and what a
healthy run returns, so that the next shift can run it on any run. Test it with
run_check.

4. Release or stop. When a program is ready, save it with save_job: a name, the job id of
a run of that program, a purpose, and the name of its check. The job saved last in a
session is released after that session: the installation runs its program once on the
reserved release starts, which it has never run on, against the approved condition, and
the release receipt appears in operator_state in the next session. There is one release
per investigation, and no programs and no saves after it. A program whose recorded runs
broke an order rule cannot be saved. If no program is ready within the budget, save none:
stopping without a release is a valid outcome and is recorded as such.

5. The account. Submit your account to the shift lead with submit_job_account: after the
release, naming the released job; when you stop without a release, with job null. It
ends the investigation, and it is refused while a release is pending. Write it in plain
words: what was asked, the condition, what you tried with its counts, what was released
or why nothing was, the release result, the check and how to run it, and what is not
known."""

_SKILLS = """The skills. Read them with read_skill and follow them: jobs-traces first (the job
tables and how to run the SQL blocks of the job skills), jobs-installation for the arm, its
control loop, its gripper and the numbers the checks use, jobs-release for the acceptance
proposal, jobs-decompose for the first program, jobs-practice for making the program
deliver, with jobs-preserve, jobs-transitions and jobs-counts, jobs-check for the check,
and jobs-release again at the end. Save evidence, the programs you tried and your plans
with memory_write: your notes are the only thing that carries from session to session.
A method a later job could use goes into a learned skill with write_learned_skill, with
this request's objects, goals and steps as parameters, never named."""

_LIBRARY = """An earlier job. The installation's job library holds what was saved for an
earlier request: its jobs, its checks and any probe recipes, listed by job_library and
runnable with run_check; list_skills shows any skill written in that investigation, with
origin learned. They were made for another request. You may use them, adapt them or
ignore them. Test anything you rely on against this request's records, and say in your
account what you used."""

_HISTORY = """Earlier investigations. This investigation continues the work of {investigations}
earlier {word} of this job: their {jobs} jobs and {episodes} episodes are in the job
tables under job ids from j1001 up, with their videos. job_results lists them after your
own, with status imported: the part each was (a development batch, or the saved program
and the naive chain beside it on that investigation's final evaluation starts), its own ids
there, and the request it ran for, whose goal indices are that request's own. job_library
gives each investigation's final evaluation, notes and account, the programs its jobs ran
and its checks. Read them before you plan: what was tried, what the records show of the
failures that remained, what the final evaluation found. Their starts are examined, and
this investigation's final evaluation runs on starts that no investigation has run on.
Imported jobs cost nothing and cannot be saved: a program is saved from a run of this
investigation. The names they used are taken."""

_CLOSING = """Report counts as counts; give a 95% interval only where a count rests on at
least twenty starts. Compute numbers with query; every number you report comes from a tool
result. Complete the work autonomously; report uncertainty rather than asking."""


def task_text(
    state: Any, session: int, library_mounted: bool, history: Mapping[str, int] | None = None
) -> str:
    """The text of session `session` (from 1) of a job investigation on `state`; `history`
    the size of the imported history, when there is one."""
    if isinstance(session, bool) or not isinstance(session, int) or session < 1:
        raise ValueError(f"sessions count from 1, not {session!r}")
    parts = [
        "You are the engineering agent of a robotic installation. Its policy is fixed and "
        "performs single tasks, each under its own instruction. The shift lead has asked for "
        "a job: several of those tasks done in one continuous scene, so that every requested "
        "goal holds at the end. Your work is to make the job, qualify it against a condition "
        "agreed in advance, and hand it over to the next shift.",
        _request(state),
        _condition(state),
        _WORK,
        _budget(state),
        _SKILLS,
    ]
    if library_mounted:
        parts.append(_LIBRARY)
    if history:
        count = int(history["investigations"])
        parts.append(
            _HISTORY.format(
                investigations=count,
                word="investigation" if count == 1 else "investigations",
                jobs=int(history["jobs"]),
                episodes=int(history["episodes"]),
            )
        )
    parts.append(_CLOSING)
    task = "\n\n".join(parts)
    if session == 1:
        return task
    return (
        f"This is session {session} of the investigation. Read your notes with memory_read, "
        "then operator_state (the shift lead's replies, the acceptance condition, the budget "
        "left, and the release receipt once a release has run) and job_results (the runs of "
        "your requests). If the release has run, submit your account to the shift lead with "
        f"submit_job_account. Otherwise continue with the task.\n\nThe task:\n{task}"
    )
