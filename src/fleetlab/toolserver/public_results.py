"""Public experiment replies: fixed messages, never engine exception text.

The engine logs refusal details on the host. This module is safe to mount and
checks stored replies again before the tool server returns them.
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from fleetlab.contracts.day_layout import ARM_SLOTS
from fleetlab.contracts.probes import ProbeOutcome, ProbeRequest
from fleetlab.contracts.tools import ExperimentRequest, ExperimentResult
from fleetlab.toolserver.files import read_lines

REFUSED = "request_not_runnable"
COMPLETED = "done"
INVALID_REQUEST = "invalid_public_request"
EXECUTION_FAILED = "execution_failed"
# The layout's room for experiment arms on one robot and day (contracts.identity), public.
OVER_CAPACITY = "exceeds_robot_day_capacity"
PUBLIC_REFUSALS = {
    INVALID_REQUEST: "Check requested robots, scopes and components against describe fleet.",
    "invalid_action_value": "Check describe vocabulary experiment_value_domains.",
    "no_recorded_change": "The visible change log has no change for this key at this scope.",
    "no_recorded_value_change": "The visible change log records only unchanged values here.",
    "identical_arms": "The arms must have different action sets.",
    "uncovered_action": "Every action must cover at least one requested robot.",
    "conflicting_actions": "An arm cannot change one robot's key more than once.",
    "unsupported_action": "Only actions with a simulated effect can be tested.",
    "missing_public_value": "The public description has no current value for this key.",
    "not_in_catalogue": "Every action must be an entry of the current catalogue table.",
    OVER_CAPACITY: (
        f"A robot-day holds at most {ARM_SLOTS} experiment-arm episodes, counting every arm "
        "and repetition this request and the earlier experiments and probes of this analysis "
        "put on that robot and day; spread the arms over robots or days."
    ),
    REFUSED: "The queue could not resolve this request; choose a supported catalogue action.",
    EXECUTION_FAILED: "The runner could not complete the request; its budget was returned.",
}
REFUSAL_CODES = frozenset(PUBLIC_REFUSALS)


def result_problems(
    result: ExperimentResult, request: ExperimentRequest | None = None
) -> list[str]:
    """Return public structural errors without interpolating untrusted reply text."""
    problems = []
    expected = {"done": {COMPLETED}, "refused": REFUSAL_CODES}
    if result.message not in expected.get(result.status, set()):
        problems.append("result has no allow-listed public status/message")
    if result.status == "refused":
        if result.arms:
            problems.append("a refused result has arms")
        return problems
    if result.status != "done":
        return problems
    indices = [arm.arm_index for arm in result.arms]
    if indices != list(range(len(result.arms))) or len(indices) < 2:
        problems.append("completed arms must have consecutive indices starting at zero")
    episodes = [episode for arm in result.arms for episode in arm.episode_ids]
    if len(episodes) != len(set(episodes)):
        problems.append("an episode appears in more than one arm")
    if any(not arm.episode_ids for arm in result.arms):
        problems.append("a completed arm is empty")
    if request is not None:
        if len(result.arms) != len(request.arms):
            problems.append("result arm count differs from the request")
        if any(len(arm.episode_ids) != request.episodes_per_arm for arm in result.arms):
            problems.append("result episode count differs from the request")
    return problems


def results_problems(results: Sequence[ExperimentResult]) -> list[str]:
    problems = [problem for result in results for problem in result_problems(result)]
    ids = [result.experiment_id for result in results]
    if len(ids) != len(set(ids)):
        problems.append("duplicate experiment result")
    episodes = [e for r in results for arm in r.arms for e in arm.episode_ids]
    if len(episodes) != len(set(episodes)):
        problems.append("an episode belongs to more than one experiment result")
    return problems


def request_results_problems(
    results: Sequence[ExperimentResult], requests: Mapping[str, ExperimentRequest]
) -> list[str]:
    """A queue has exactly one structurally valid reply per requested experiment."""
    problems = results_problems(results)
    if {r.experiment_id for r in results} != set(requests):
        problems.append("reply results differ from the queue's requests")
    for result in results:
        if result.experiment_id in requests:
            problems.extend(result_problems(result, requests[result.experiment_id]))
    return problems


def load_results(
    results_file: Path,
    requests_file: Path,
    session: int,
    episodes: Mapping[str, tuple[str | None, str]],
) -> list[ExperimentResult]:
    """Validate every persisted result against its request and visible episode rows.

    Pending requests from this session are allowed; older sessions must have
    replies before a continuation starts. No exception includes stored text.
    """
    try:
        records = list(read_lines(requests_file, strict=True))
        requests = {
            r["experiment_id"]: ExperimentRequest.model_validate(r["request"]) for r in records
        }
        if len(requests) != len(records):
            raise ValueError("duplicate request")
        results = [
            ExperimentResult.model_validate(r) for r in read_lines(results_file, strict=True)
        ]
        problems = results_problems(results)
        done = {result.experiment_id for result in results}
        if any(r["session"] < session and r["experiment_id"] not in done for r in records):
            problems.append("earlier request has no result")
        for result in results:
            request = requests.get(result.experiment_id)
            if request is None:
                problems.append("result has no request")
                continue
            problems.extend(result_problems(result, request))
            for arm in result.arms:
                for episode in arm.episode_ids:
                    found = episodes.get(episode)
                    if (
                        found is None
                        or found[0] != result.experiment_id
                        or found[1] not in request.robots
                    ):
                        problems.append("result episode does not match its request and export")
        if problems:
            raise ValueError("; ".join(problems))
        return results
    except (ValueError, KeyError, TypeError):
        raise ValueError("experiment results failed public validation") from None


# Probes (contracts.probes): the same discipline as experiment results. A done outcome is
# bound to its request and to the export's probe rows; refused and failed outcomes show
# nothing; one outcome per probe; an outcome becomes visible only in a later session.

ProbeRows = Mapping[str, tuple[str, str, int]]  # episode -> (probe_id, robot_id, repeat)


def probe_outcome_problems(
    outcome: ProbeOutcome,
    request: ProbeRequest | None = None,
    rows: ProbeRows | None = None,
    with_steps: Collection[str] | None = None,
) -> list[str]:
    """Public structural errors of one probe outcome, against its request and, when
    given, the export's `probe_episodes` rows and the episodes `probe_steps` holds."""
    problems: list[str] = []
    if request is not None and outcome.attempts_executed > request.attempts:
        problems.append("a probe executed more attempts than robots x repeats")
    if request is not None and outcome.status == "failed" and outcome.charged != request.attempts:
        problems.append("a failed probe keeps its whole request charged, executed or reserved")
    if outcome.status != "done":
        if rows is not None and any(p == outcome.probe_id for p, _, _ in rows.values()):
            problems.append("a probe without episodes has rows in the export")
        return problems
    if request is not None and len(outcome.episode_ids) != request.attempts:
        problems.append("probe outcome episode count differs from robots x repeats")
    if rows is not None:
        own = {e for e, (p, _, _) in rows.items() if p == outcome.probe_id}
        if own != set(outcome.episode_ids):
            problems.append("probe outcome episodes differ from the export's probe rows")
        if request is not None:
            seen = sorted((rows[e][1], rows[e][2]) for e in outcome.episode_ids if e in rows)
            wanted = sorted(
                (robot, repeat) for robot in request.robots for repeat in range(request.repeats)
            )
            if seen != wanted:
                problems.append("probe outcome episodes do not cover its robots and repeats")
    if with_steps is not None and any(e not in with_steps for e in outcome.episode_ids):
        problems.append("a probe episode has no probe_steps rows")
    return problems


def probe_outcomes_problems(outcomes: Sequence[ProbeOutcome]) -> list[str]:
    """One outcome per probe, and no episode in two of them."""
    problems = [p for outcome in outcomes for p in probe_outcome_problems(outcome)]
    ids = [o.probe_id for o in outcomes]
    if len(ids) != len(set(ids)):
        problems.append("duplicate probe outcome")
    episodes = [e for o in outcomes for e in o.episode_ids]
    if len(episodes) != len(set(episodes)):
        problems.append("an episode belongs to more than one probe outcome")
    return problems


def load_probe_outcomes(
    results_file: Path,
    requests_file: Path,
    session: int,
    rows: ProbeRows,
    with_steps: Collection[str],
) -> list[ProbeOutcome]:
    """Validate every persisted probe outcome against its request and the export's probe
    rows. Requests of this session have none yet; earlier sessions' must all have one
    before a continuation starts. No exception includes stored text; any problem refuses
    the outcomes as a whole, so a contradictory record never refunds a probe it shows."""
    try:
        records = list(read_lines(requests_file, strict=True))
        requests = {r["probe_id"]: ProbeRequest.model_validate(r["request"]) for r in records}
        sessions = {r["probe_id"]: int(r["session"]) for r in records}
        if len(requests) != len(records):
            raise ValueError("duplicate probe request")
        outcomes = [ProbeOutcome.model_validate(r) for r in read_lines(results_file, strict=True)]
        problems = probe_outcomes_problems(outcomes)
        done = {o.probe_id for o in outcomes}
        if any(sessions[p] < session and p not in done for p in requests):
            problems.append("earlier probe has no outcome")
        listed = {e for o in outcomes for e in o.episode_ids}
        if any(e not in listed for e in rows):
            problems.append("the export has probe rows no outcome names")
        for outcome in outcomes:
            request = requests.get(outcome.probe_id)
            if request is None:
                problems.append("probe outcome has no request")
                continue
            if sessions[outcome.probe_id] >= session:
                problems.append("a probe outcome is visible only in a later session")
            problems.extend(probe_outcome_problems(outcome, request, rows, with_steps))
        if problems:
            raise ValueError("; ".join(problems))
        return outcomes
    except (ValueError, KeyError, TypeError):
        raise ValueError("probe results failed public validation") from None
