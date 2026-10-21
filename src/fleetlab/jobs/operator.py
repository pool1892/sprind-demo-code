"""The operator side of a job request: the host-owned RequestState and the scripted shift lead
(docs/design/jobs.md).

The integrator writes the request as a JSON file (`RequestState`: the request's text, the
requested goals, the base task, the order rules, the budget). The job driver
(`fleetlab.jobs.driver`) publishes it read-only into the analysis' results before every
session (`publish`), where the tool server reads it (`fleetlab.toolserver.jobs`). Only
the host writes it: after each session it copies the agent's acceptance proposals and the
shift lead's replies into the message log (`sync_session`), and the approved condition
into `acceptance`, after deciding the proposal again with the same rule (`lead_reply`),
so a reply the agent might have edited in its outbox never stands.

The scripted shift lead approves a proposal of k deliveries on n release starts when n is
at least 20 (and no more than the installation reserves) and k at least four fifths of
n; otherwise it says what is missing. The order rule is checked by the runner on every
recorded run's trace (`order_rule_ok` in job_episodes); the tool server refuses
`save_job` for a program whose recorded runs broke it, with the rule's text.

    python -m fleetlab.jobs.operator check REQUEST.json        # validate a request file
    python -m fleetlab.jobs.operator reply N K                  # the shift lead's answer
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetlab.contracts.jobs import (
    GOAL_SUITE_TASKS,
    JOB_LIMITS,
    RELEASE_INIT_STATES,
    Acceptance,
    RequestMessage,
    RequestState,
)
from fleetlab.toolserver.files import SUBMISSIONS_FILE, read_lines
from fleetlab.toolserver.jobs import (
    ACCEPTANCE_FILE,
    ACCOUNT_KIND,
    LEAD_FRACTION,
    OPERATOR_FILE,
    goal_text,
    lead_reply,
    proposal_messages,
    public_request,
    rule_text,
)

__all__ = [
    "ShiftLead",
    "account_of",
    "describe",
    "load_request",
    "publish",
    "request_problems",
    "sync_session",
]


class ShiftLead:
    """The scripted shift lead: one rule for every proposal (`toolserver.jobs.lead_reply`)."""

    def __init__(
        self, release_available: int | None = None, fraction: tuple[int, int] = LEAD_FRACTION
    ) -> None:
        self.release_available = (
            release_available if release_available is not None else len(RELEASE_INIT_STATES)
        )
        self.fraction = fraction

    def reply(self, n: int, k: int) -> tuple[bool, str]:
        return lead_reply(n, k, self.release_available, self.fraction)


def request_problems(state: RequestState) -> list[str]:
    """What is wrong with a request the integrator wrote beyond what the model refuses: an
    acceptance condition set before the agent proposed one, agent messages in the log."""
    problems = []
    if state.acceptance is not None:
        problems.append("acceptance is the agent's proposal, approved during the investigation")
    if any(m.author == "agent" for m in state.message_log):
        problems.append("the message log of a new request holds no agent message")
    return problems


def load_request(path: Path) -> RequestState:
    """The integrator's request file, checked, with the shift lead's request as the first
    message of the log when the file carries none."""
    state = RequestState.model_validate_json(path.read_text(encoding="utf-8"))
    if problems := request_problems(state):
        raise ValueError(f"{path}: " + "; ".join(problems))
    if not state.message_log:
        first = RequestMessage(author="lead", text=state.request_text, session=None)
        state = state.model_copy(update={"message_log": [first]})
    return state


def publish(
    results: Path,
    state: RequestState,
    skills: Sequence[Mapping[str, Any]] = (),
    fraction: tuple[int, int] = LEAD_FRACTION,
) -> Path:
    """The RequestState as the tool server reads it (results/operator_state.json), in
    its mounted form (`toolserver.jobs.public_request`)."""
    path = results / OPERATOR_FILE
    temporary = path.with_name(f".{path.name}.tmp")
    data = public_request(state, skills, fraction)
    temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
    temporary.replace(path)
    return path


def _message(author: str, text: str, session: int) -> RequestMessage:
    return RequestMessage.model_validate({"author": author, "text": text, "session": session})


def account_of(outbox: Path) -> dict[str, Any] | None:
    """The agent's last accepted account (`submit_job_account`), with its session."""
    found = [r for r in read_lines(outbox / SUBMISSIONS_FILE) if r.get("kind") == ACCOUNT_KIND]
    return found[-1] if found else None


def account_message(head: str, account: str) -> str:
    """The account as the message log keeps it: whole when it fits a message, otherwise its
    beginning with a pointer to the full account, which its own record keeps."""
    whole = f"{head}: {account}"
    limit = JOB_LIMITS.max_request_text
    if len(whole) <= limit:
        return whole
    tail = " [...] (shortened for the message log; the full account is in its record)"
    return whole[: limit - len(tail)].rstrip() + tail


def sync_session(
    state: RequestState, outbox: Path, session: int, lead: ShiftLead | None = None
) -> tuple[RequestState, list[str]]:
    """After session `session`: the host's copy of its acceptance exchange and account.

    Every proposal of the session enters the message log with the shift lead's reply as
    the host decides it again; the first approved one becomes `acceptance`. Returns the
    new state and any disagreement between the outbox and the host's decision."""
    lead = lead or ShiftLead()
    messages = list(state.message_log)
    acceptance = state.acceptance
    problems: list[str] = []
    for record in read_lines(outbox / ACCEPTANCE_FILE):
        if int(record["session"]) != session:
            continue
        proposal = record["proposal"]
        n, k = int(proposal["N"]), int(proposal["K"])
        (_, asked), _ = proposal_messages(record)
        messages.append(_message("agent", asked, session))
        if acceptance is not None:
            problems.append(f"{record['receipt_id']}: a proposal after the approval")
            continue
        approved, reply = lead.reply(n, k)
        if (approved, reply) != (record["approved"], record["reply"]):
            problems.append(f"{record['receipt_id']}: the outbox's reply differs from the lead's")
        messages.append(_message("lead", reply, session))
        if approved:
            acceptance = Acceptance(N=n, K=k)
    account = account_of(outbox)
    if account is not None and int(account["session"]) == session:
        answer = account["answer"]
        head = f"Account, job {answer['job']}" if answer["job"] else "Account, nothing released"
        messages.append(_message("agent", account_message(head, answer["account"]), session))
    update = {"message_log": messages, "acceptance": acceptance}
    return state.model_copy(update=update), problems


def describe(state: RequestState) -> dict[str, Any]:
    """The request in words, for a record."""
    return {
        "request_text": state.request_text,
        "goals": [goal_text(g) for g in state.requested_goals],
        "base_task": {"number": state.base_task, "task": GOAL_SUITE_TASKS[state.base_task]},
        "order_rules": [rule_text(state, r) for r in state.order_rules],
        "budget": state.budget.model_dump(mode="json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.jobs.operator")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate a request file")
    check.add_argument("request", type=Path)
    reply = commands.add_parser("reply", help="the scripted shift lead's answer to n and k")
    reply.add_argument("n", type=int)
    reply.add_argument("k", type=int)
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps(describe(load_request(args.request)), indent=1))
        return 0
    approved, text = ShiftLead().reply(args.n, args.k)
    print(json.dumps({"approved": approved, "reply": text}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
