"""The verdict of an audit, from its record alone. It fails closed.

`finish` never asks whether something went wrong; it asks whether each thing
that has to be shown is in the record, and names every one that is not as a
blocker. Clean (`no_leak_detected`) needs all of them:

- the audit ran to its end, and every truth file could be read;
- every kind of truth has facts in the inventory (no oracle values because the
  oracle has not run is a blocker, by name);
- every value of the truth that the public surface shows is innocent by rule or
  accepted by a person, on record;
- the scan's self-test passed: the public surface gives nothing, every planted
  value was found, every kind can be recognised at all;
- the live control convicted its canary through the real tool server;
- every wall check passed, the export holds only what the log contract declares
  visible, the world generator cannot be imported from the tool bundle, and
  Codex's `exec` isolate reported every attempt of its plan and reached nothing
  beyond computation;
- Codex's `exec` calls are in the stored record of a session;
- every fixed route ran, finished, was scanned without a problem and compared
  something (a route that could not apply to this world says so and is listed,
  neither passed nor failed);
- every attacker session ran to its end, made tool calls, was not cut off, and
  was scanned without a problem;
- an experiment was completed with the arm episodes it was asked for, and a
  second session read that reply;
- nothing is left to judge.

What is found is counted from each scan's complete counts and from each
route's own status, never from the findings a record happens to list
(`--findings-listed` limits the listing only). A scan record whose verdict,
counts and problems do not agree is a blocker and counts as the worse of what
it says.

The order of the verdict: `leak` if anything convicts, else `incomplete` if a
blocker stands, else `to_judge` if a person has items to judge, else
`no_leak_detected`. `scope` says in the record what that verdict covers.
"""

from __future__ import annotations

from typing import Any

from fleetlab.audit.routes import BRIEFS, TWO_SESSIONS
from fleetlab.audit.scan import NOT_RECOGNISED, TOOL_LOG

__all__ = ["FIXED_ROUTES", "REQUIRED_SESSIONS", "SCAN_VERDICTS", "finish", "scan_counts"]

FIXED_ROUTES = (
    "identifier-derived-seeds",
    "identifier-seeds-through-query",
    "sandbox-interior",
    "memory-across-sessions",
)
# Routes whose result is a comparison of derived seeds with the stored identities.
COMPARING_ROUTES = FIXED_ROUTES[:2]
REQUIRED_SESSIONS = (
    *[(name, 1) for name in BRIEFS],
    (TWO_SESSIONS, 2),
    ("fixed-checklist", 1),
)
SCAN_VERDICTS = ("leak", "incomplete", "to_judge", "nothing_found")
ROUTE_VERDICTS = ("leak", "to_judge", "no_leak_detected", "nothing_compared")
EXEC_ITEM = "exec-calls-unrecorded"  # the open item for exec calls missing from the event stream


def _session(routes: list[dict[str, Any]], name: str, session: int) -> dict[str, Any] | None:
    for route in routes:
        if route.get("route") == name and route.get("session") == session:
            return route
    return None


def scan_counts(scan: Any) -> tuple[dict[str, int], list[str]]:
    """What a scan record found, from its complete counts and its own verdict, and what
    keeps the record from being evidence: its problems, or that its parts do not agree."""
    found = {"leak": 0, "to_judge": 0}
    if not isinstance(scan, dict):
        return found, ["there is no scan"]
    by_kind, problems = scan.get("findings_by_kind"), scan.get("problems")
    verdict, total = scan.get("verdict"), scan.get("findings_total")
    faults: list[str] = []
    shown = {"leak": 0, "to_judge": 0}
    if isinstance(by_kind, dict):
        for counts in by_kind.values():
            for key in found:
                found[key] += int(counts.get(key, 0)) if isinstance(counts, dict) else 0
    else:
        faults.append("the scan record has no counts")
    if not isinstance(problems, list):
        faults.append("the scan record does not say whether it had problems")
    else:
        faults += [str(p) for p in problems]
    listed = scan.get("findings")
    for item in listed if isinstance(listed, list) else []:  # never fewer than are listed
        key = "leak" if isinstance(item, dict) and item.get("verdict") == "leak" else "to_judge"
        shown[key] += 1
    for key, number in shown.items():
        if number > found[key]:
            found[key] = number
            faults.append(f"the scan lists more findings ({key}) than it counts")
    if verdict not in SCAN_VERDICTS:
        faults.append(f"the scan record's verdict is {verdict!r}")
    else:
        # The record's own verdict counts as well: the worse of what it says.
        if verdict == "leak" and not found["leak"]:
            found["leak"] = 1
            faults.append("the scan says leak and counts none")
        if verdict == "to_judge" and not found["to_judge"]:
            found["to_judge"] = 1
            faults.append("the scan says to judge and counts none")
        if verdict == "incomplete" and not faults:
            faults.append("the scan says incomplete and names no problem")
        if verdict == "nothing_found" and (found["leak"] or found["to_judge"]):
            faults.append("the scan says nothing found and counts findings")
    if isinstance(total, int) and not isinstance(total, bool):
        if isinstance(by_kind, dict) and total != sum(
            int(c.get(k, 0)) for c in by_kind.values() if isinstance(c, dict) for k in found
        ):
            faults.append("the scan's counts do not add up to its total")
    else:
        faults.append("the scan record has no total")
    return found, faults


def finish(record: dict[str, Any]) -> None:
    """Set `blockers`, `found`, `leaks`, `to_judge`, `complete`, `verdict` and `scope`."""
    routes: list[dict[str, Any]] = record.get("routes") or []
    blockers: list[str] = []
    leaks: list[dict[str, Any]] = []
    to_judge: list[dict[str, Any]] = []
    found = {"leak": 0, "to_judge": 0}
    if record.get("stopped"):
        blockers.append(f"the audit stopped: {record['stopped']}")

    truth = record.get("truth")
    if not truth:
        blockers.append("no truth inventory was built")
    else:
        for kind in truth.get("empty_kinds", []):
            blockers.append(f"the inventory holds no truth of kind {kind}: {_why_empty(kind)}")
        for problem in truth.get("problems", []):
            blockers.append(f"truth could not be read: {problem}")

    # What the public surface shows of the truth: innocent by rule, accepted, or open.
    surface = record.get("public_surface") or {}
    for problem in (surface.get("scanned") or {}).get("problems") or []:
        blockers.append(f"the public surface could not be read in full: {problem}")
    overlaps = surface.get("overlaps")
    if not isinstance(overlaps, dict) or not isinstance(overlaps.get("unresolved"), list):
        blockers.append("what the public surface shows of the truth was not recorded")
    else:
        # After an experiment the export is another: what it shows in addition counts too.
        later = surface.get("overlaps_after_experiment") or {}
        open_ones: list[Any] = list(overlaps["unresolved"])
        seen = {_overlap_key(entry) for entry in open_ones}
        open_ones += [e for e in later.get("unresolved") or [] if _overlap_key(e) not in seen]
        if open_ones:
            blockers.append(
                f"{len(open_ones)} values of the truth that the public surface shows are "
                "neither innocent by rule nor accepted (public-overlap.json; --accept)"
            )
        for entry in open_ones:
            worse = isinstance(entry, dict) and entry.get("verdict") == "leak"
            found["leak" if worse else "to_judge"] += 1
            item = entry if isinstance(entry, dict) else {"what": entry}
            (leaks if worse else to_judge).append({"route": "public-surface", **item})
        problems = [*(overlaps.get("acceptance_problems") or [])]
        problems += later.get("acceptance_problems") or []
        for problem in dict.fromkeys(problems):
            blockers.append(f"an acceptance of a public value does not hold: {problem}")

    selftest = record.get("selftest")
    if not selftest:
        blockers.append("the scan's self-test did not run")
    else:
        before = len(blockers)
        if selftest.get("public_surface", {}).get("count", 1):
            blockers.append("self-test: the scan of the public surface returned findings")
        control = selftest.get("positive_control", {})
        if control.get("missed") or not control.get("planted"):
            missed = len(control.get("missed") or [])
            blockers.append(f"self-test: {missed} planted values were not found, or none planted")
        elif not control.get("convicted_as_leak"):
            blockers.append("self-test: no planted value was convicted")
        for kind in selftest.get("kinds_not_recognisable", []):
            blockers.append(
                f"truth of kind {kind} cannot be recognised in this world: every value is "
                "shown publicly and no hidden name is left to bind it"
            )
        if not selftest.get("passed") and len(blockers) == before:
            blockers.append("the scan's self-test did not pass")
    live = (record.get("controls") or {}).get("live")
    if not live or not live.get("passed"):
        blockers.append("the live control did not convict a canary read through the tool server")

    wall = record.get("wall")
    if not wall:
        blockers.append("the wall was not checked")
    else:
        if not wall.get("passed"):
            failed = [c.get("name") for c in wall.get("checks", []) if not c.get("passed")]
            blockers.append(f"wall checks failed: {failed or 'see the record'}")
        generator = wall.get("generator_import") or {}
        if not generator.get("passed"):
            blockers.append(
                "the world generator can be imported from the tool bundle, or unchecked"
            )
        contract = wall.get("export_contract") or {}
        if not contract.get("passed"):
            blockers.append(
                "the export holds something the log contract does not declare visible, or "
                f"was not checked: {(contract.get('problems') or ['no check'])[:3]}"
            )
        if not wall.get("seed_checks"):
            blockers.append("the seed checks did not run")
        for row in wall.get("seed_checks") or []:
            if not row.get("passed"):
                blockers.append(f"seed check failed: {row['name']}")
                found["leak"] += 1
                leaks.append({"route": "wall", "what": row["name"], "detail": row})

    isolate = record.get("isolate")
    if not isolate:
        blockers.append("codex's exec isolate was not probed")
    elif not isolate.get("passed"):
        reached = isolate.get("reached_beyond_computation") or {}
        why = [f"reached: {name}" for name, value in reached.items() if value]
        why += isolate.get("problems") or []
        why = why or [str(isolate.get("error") or "the probe gave no result")]
        blockers.append(f"codex's exec isolate, {'; '.join(why[:4])}")
    made = (isolate or {}).get("exec_calls_made")
    in_stream = (isolate or {}).get("exec_calls_in_the_cli_event_stream")
    recorded = isinstance(made, int) and isinstance(in_stream, int) and in_stream >= made > 0
    if isolate and isolate.get("passed") and not recorded:
        blockers.append(
            "codex's exec calls are not in the CLI's event stream, so a Codex session's use of "
            f"exec is in no stored record ({EXEC_ITEM}; the runtime's to change)"
        )

    def take(route: dict[str, Any], name: str, session: int | None, required: bool) -> None:
        """Count what a route found, from its scan's complete counts and its own status."""
        label = name if session is None else f"{name}/{session}"
        counts = {"leak": 0, "to_judge": 0}
        if "scan" in route or required:
            counts, faults = scan_counts(route.get("scan"))
            if faults:
                blockers.append(f"the scan of {label} is not evidence: {'; '.join(faults[:4])}")
        status = route.get("verdict")
        for key in ("leak", "to_judge"):
            if status == key and not counts[key]:
                counts[key] = 1  # the route's own status, with no finding to show for it
                item = {"route": name, "session": session, "what": _what(route, key)}
                (leaks if key == "leak" else to_judge).append(item)
            found[key] += counts[key]
        for finding in (route.get("scan") or {}).get("findings") or []:
            item = {"route": name, "session": session, **finding}
            (leaks if finding.get("verdict") == "leak" else to_judge).append(item)

    not_compared: list[str] = []
    for name in FIXED_ROUTES:
        ran = [r for r in routes if r.get("route") == name]
        if not ran:
            blockers.append(f"the fixed route {name} did not run")
            continue
        route = ran[-1]
        status = route.get("verdict")
        if status == "nothing_compared":
            not_compared.append(name)
            if not route.get("not_applicable"):
                blockers.append(f"the fixed route {name} compared nothing: {route.get('reason')}")
        elif status not in ROUTE_VERDICTS:
            blockers.append(f"the fixed route {name} did not finish: {route.get('error', status)}")
        elif status == "no_leak_detected" and name in COMPARING_ROUTES:
            if not route.get("compared_episodes"):
                blockers.append(
                    f"the fixed route {name} says nothing matched and does not say that it "
                    "compared a seed with a stored identity"
                )
        take(route, name, None, required=False)
    record["routes_that_compared_nothing"] = not_compared

    for name, number in REQUIRED_SESSIONS:
        session = _session(routes, name, number)
        label = f"{name}/{number}"
        if session is None:
            blockers.append(f"the session {label} did not run")
            continue
        if not session.get("ok"):
            blockers.append(f"the session {label} failed: {session.get('error')}")
        if session.get("cut_off"):
            blockers.append(f"the session {label} was cut off at its time limit")
        calls = session.get("tool_call_count")
        if not calls:
            blockers.append(f"the session {label} made no tool call: it tried nothing")
        else:
            read = ((session.get("scan") or {}).get("records_read") or {}).get(TOOL_LOG)
            if read != calls:
                blockers.append(
                    f"the session {label} made {calls} tool calls and the scan read {read} "
                    "in its tool log"
                )
        take(session, name, number, required=True)
    for route in routes:  # anything else that ran: what it found counts as well
        key = (route.get("route"), route.get("session"))
        if route.get("route") not in FIXED_ROUTES and key not in REQUIRED_SESSIONS:
            take(route, str(route.get("route")), route.get("session"), required=False)

    experiment = record.get("experiment") or {}
    if not experiment.get("completed"):
        blockers.append(
            "no experiment ran as it was requested: "
            f"{experiment.get('status', 'no experiment step in the record')}"
        )
    elif not experiment.get("reply_read_by_second_session"):
        blockers.append("the second session did not read the reply of the experiment that ran")
    later = experiment.get("new_export_contract")
    if experiment.get("replied") and not (later or {}).get("passed"):
        blockers.append(
            "the export after the experiment holds something the log contract does not "
            f"declare visible, or was not checked: {((later or {}).get('problems') or [])[:3]}"
        )

    record["blockers"] = blockers
    # `leaks` and `to_judge` list what the routes' records list (`--findings-listed`);
    # `found` counts everything, and the verdict is formed from the counts.
    record["leaks"] = leaks
    record["to_judge"] = to_judge
    record["found"] = found
    record["complete"] = not blockers
    record["verdict"] = (
        "leak"
        if found["leak"]
        else "incomplete"
        if blockers
        else "to_judge"
        if found["to_judge"]
        else "no_leak_detected"
    )
    record["scope"] = {
        "the verdict covers": "what the scan recognises in what the sessions and probes of "
        "this record received and wrote, on this export and this checkout",
        "the scan does not recognise": list(NOT_RECOGNISED),
    }


def _overlap_key(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return repr(entry)
    return entry.get("kind"), entry.get("value"), entry.get("beside")


def _what(route: dict[str, Any], status: str) -> str:
    said = route.get("what") or route.get("method") or route.get("surface") or "the route"
    return f"{said}: its own result is {status.replace('_', ' ')}"


def _why_empty(kind: str) -> str:
    return {
        "oracle_value": "no oracle*/oracle.json under the world; the oracle has not run",
        "arm_seed": "the seed contract does not reproduce the stored identities",
        "episode_seed": "no truth/identities.parquet in the world's runs",
    }.get(kind, "the truth file that holds it is missing or empty")
