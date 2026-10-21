"""Host-only, current-content audit required at the common agent launch boundary."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fleetlab.contracts.model import Fleet

if TYPE_CHECKING:
    from fleetlab.agents.run import ToolAccess
    from fleetlab.log.leaks import LeakReport


@dataclass(frozen=True)
class AuditContext:
    """Truth-side audit inputs. These paths and this fleet are never mounted."""

    run_dirs: tuple[Path, ...]
    fleet: Fleet | None = None
    experiment_root: Path | None = None
    experiment_days: tuple[int, int] | None = None

    def runs(self) -> tuple[Path, ...]:
        extra = () if self.experiment_root is None else completed_runs(self.experiment_root)
        return tuple(dict.fromkeys((*self.run_dirs, *extra)))


def completed_runs(root: Path) -> tuple[Path, ...]:
    """Only finalized queues can contribute to the engine's exported log."""
    from fleetlab.log import run_status

    return tuple(sorted(p.parent for p in root.rglob("run.json") if run_status(p.parent).finalized))


def tree_digest(root: Path) -> str:
    """One sha256 over a plain directory's `tree_manifest`: the identity of what is mounted.

    The same value as `integration.accounting.library_revision` for a skill library.
    """
    return hashlib.sha256(json.dumps(tree_manifest(root), sort_keys=True).encode()).hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    """Hash every mounted data/source file; links and special files fail closed."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"audit root is not a plain directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(mode):
            files[relative + "/"] = "directory"
        elif stat.S_ISREG(mode):
            with path.open("rb") as stream:
                files[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            raise ValueError(f"mounted audit input is a link or special file: {relative}")
    return files


# Only these schema locations contain agent-authored prose. Keys, envelopes,
# config/condition values and identifiers stay ours even in an outbox.
_DIAGNOSIS_PROSE: set[tuple[str | None, ...]] = {
    ("causes", None, "failure_mode"),
    ("causes", None, "evidence", None, "query"),
    ("mode_summaries", None),
    ("warnings", None, "metric"),
    ("retrain", None, "failure_mode"),
    ("retrain", None, "why_no_action_fixes_it"),
    ("retrain", None, "data_wanted"),
}
_FIX_PROSE: set[tuple[str | None, ...]] = {
    ("actions", None, "rationale"),
    ("actions", None, "action", "value"),
}
# A learned skill's draft in the outbox (contracts.learned; docs/design/loop.md, section 9).
_LEARNED_PROSE: set[tuple[str | None, ...]] = {
    ("draft", "name"),
    ("draft", "description"),
    ("draft", "applies_when"),
    ("draft", "next_experiment"),
    ("draft", "parameters", None, "name"),
    ("draft", "parameters", None, "meaning"),
    ("draft", "evidence", None, "claim"),
    ("draft", "evidence", None, "sql"),
    ("draft", "evidence", None, "result"),
    ("draft", "alternatives", None, "explanation"),
    ("draft", "alternatives", None, "how_to_tell"),
}
# A probe the agent composed (outbox/probes.jsonl; contracts.probes): its own words.
_PROBE_PROSE: set[tuple[str | None, ...]] = {
    ("request", "name"),
    ("request", "purpose"),
    ("request", "expectation"),
}
# The round decision (contracts.decisions): rationales, expectations, and the ranked
# candidates' rationales and their no-replay values, as for a fix set.
_DECISION_PROSE: set[tuple[str | None, ...]] = {
    ("adopt", None, "rationale"),
    ("replace", None, "rationale"),
    ("undo", None, "rationale"),
    ("expectations", None),
    ("ranked_candidates", None, "rationale"),
    ("ranked_candidates", None, "action", "value"),
}

# A job session's own writings (fleetlab.toolserver.jobs), by outbox file: its checks' names,
# purposes, SQL and expectations, its programs' names, purposes, notes and instructions, its
# saved jobs' names and purposes, its proposals' reasons and its account. The agent writes
# SQL and prose with ordinary words that are also truth-side keys (`verdict`, `rate`), so
# these are scanned with the agent-text rules (canaries, hidden identifiers, seeds) only.
_PROGRAM_PROSE: set[tuple[str | None, ...]] = {
    ("name",),
    ("purpose",),
    ("notes",),
    ("steps", None, "instruction"),
}
_JOB_PROSE: dict[str, set[tuple[str | None, ...]]] = {
    "checks.jsonl": {
        ("check", "name"),
        ("check", "purpose"),
        ("check", "sql"),
        ("check", "expects"),
        ("check", "parameters", None, "name"),
    },
    "job_requests.jsonl": {("request", "program", *path) for path in _PROGRAM_PROSE},
    "saved_jobs.jsonl": {
        ("name",),
        ("purpose",),
        ("check",),
        *{("program", *path) for path in _PROGRAM_PROSE},
    },
    "acceptance.jsonl": {("proposal", "reason")},
    "check_runs.jsonl": {("check",)},
    "job_account.json": {("job",), ("account",)},
}
_JOB_ACCOUNT_PROSE: set[tuple[str | None, ...]] = {("job",), ("account",)}
# A mounted job library (results/library, `fleetlab.jobs.driver.public_library`): the
# earlier investigation's own words in its saved jobs, checks and recipes. Its names, keys,
# digests, goals and release counts stay strict.
_LIBRARY_PROSE: dict[str, set[tuple[str | None, ...]]] = {
    "library/saved_checks.json": {
        (None, "purpose"),
        (None, "sql"),
        (None, "expects"),
    },
    "library/saved_jobs.json": {
        (None, "purpose"),
        (None, "program", "purpose"),
        (None, "program", "notes"),
        (None, "program", "steps", None, "instruction"),
    },
    "library/saved_recipes.json": {(None, "purpose")},
}
# The public history of a numbered investigation (results/history.json,
# `fleetlab.jobs.prior`): the earlier investigations' own words in their programs, checks,
# notes and accounts, and the names they gave their jobs and checks. Every other key and
# value of the file (ids, parts, statuses, digests, goals, counts) stays strict.
_HISTORY_PROSE: dict[str, set[tuple[str | None, ...]]] = {
    "history.json": {
        ("imported", None, "name"),
        ("imported", None, "purpose"),
        ("programs", None, "program", "name"),
        ("programs", None, "program", "purpose"),
        ("programs", None, "program", "notes"),
        ("programs", None, "program", "steps", None, "instruction"),
        ("checks", None, "check", "name"),
        ("checks", None, "check", "purpose"),
        ("checks", None, "check", "sql"),
        ("checks", None, "check", "expects"),
        ("checks", None, "check", "parameters", None, "name"),
        ("investigations", None, "notes"),
        ("investigations", None, "account", "words"),
        ("investigations", None, "account", "job"),
        ("investigations", None, "final_evaluation", "job"),
        ("job_names", None),
        ("check_names", None),
    },
}


def _agent_prose(root: Path, state: str) -> dict[str, str]:
    from fleetlab.contracts.vocabulary import NO_REPLAY_VERBS
    from fleetlab.toolserver.decisions import DECISION_FILE
    from fleetlab.toolserver.files import (
        DIAGNOSIS_FILE,
        EXPERIMENTS_FILE,
        FIX_SET_FILE,
        LEARNED_FILE,
        NOTES_FILE,
        PROBES_FILE,
        SUBMISSIONS_FILE,
    )

    values: dict[str, str] = {}

    def unambiguous(value: Any) -> bool:
        # The scanner's file/field labels are not escaped JSON pointers. Keep
        # malformed keys that could impersonate another location fully strict.
        if isinstance(value, dict):
            return all(
                not any(token in key for token in ("/", " at ", " key ", " (decoded JSON)"))
                and unambiguous(item)
                for key, item in value.items()
            )
        return all(map(unambiguous, value)) if isinstance(value, list) else True

    def collect(
        value: Any,
        where: str,
        allowed: set[tuple[str | None, ...]],
        path: str = "",
        shape: tuple[str | None, ...] = (),
    ) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                # The public contract defines these values as agent-written
                # task/data descriptions. Config/condition settings stay ours.
                if key == "value" and value.get("verb") not in tuple(NO_REPLAY_VERBS):
                    continue
                collect(item, where, allowed, path + "/" + key, (*shape, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, where, allowed, path + "/" + str(index), (*shape, None))
        elif isinstance(value, str) and shape in allowed:
            values[f"{where} at {path}"] = value

    if state == "memory":
        path = root / NOTES_FILE
        if path.is_file() and not path.is_symlink():
            values[NOTES_FILE] = path.read_text()
    elif state == "results":
        for name, fields in (*_LIBRARY_PROSE.items(), *_HISTORY_PROSE.items()):
            path = root / name
            if not path.is_file() or path.is_symlink():
                continue
            try:
                record = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                continue
            if unambiguous(record):
                collect(record, name, fields)
        # The request's message log as mounted (`toolserver.jobs.public_request`): the
        # agent's own messages (its proposals' reasons, its account); the lead's stay strict.
        path = root / "operator_state.json"
        if path.is_file() and not path.is_symlink():
            try:
                request = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                request = None
            log = request.get("message_log") if isinstance(request, dict) else None
            for index, message in enumerate(log if isinstance(log, list) else []):
                if (
                    isinstance(message, dict)
                    and message.get("author") == "agent"
                    and isinstance(message.get("words"), str)
                ):
                    values[f"operator_state.json at /message_log/{index}/words"] = message["words"]
    elif state == "outbox":
        names = (
            EXPERIMENTS_FILE,
            DIAGNOSIS_FILE,
            FIX_SET_FILE,
            DECISION_FILE,
            SUBMISSIONS_FILE,
            LEARNED_FILE,
            PROBES_FILE,
            *_JOB_PROSE,
        )
        for name in names:
            path = root / name
            if not path.is_file() or path.is_symlink():
                continue
            try:
                content = path.read_text()
                if path.suffix == ".jsonl":
                    records = [
                        (f"{name} line {number}", json.loads(line))
                        for number, line in enumerate(content.splitlines(), 1)
                        if line.strip()
                    ]
                else:
                    records = [(name, json.loads(content))]
            except (ValueError, UnicodeError):
                continue  # malformed files keep the original strict scanner findings
            for where, record in records:
                if not unambiguous(record):
                    continue
                allowed: set[tuple[str | None, ...]]
                if name == EXPERIMENTS_FILE:
                    allowed = {("request", "power_note")}
                elif name == DIAGNOSIS_FILE:
                    allowed = _DIAGNOSIS_PROSE
                elif name == FIX_SET_FILE:
                    allowed = _FIX_PROSE
                elif name == LEARNED_FILE:
                    allowed = _LEARNED_PROSE
                elif name == PROBES_FILE:
                    allowed = _PROBE_PROSE
                elif name == DECISION_FILE:
                    allowed = _DECISION_PROSE
                elif name in _JOB_PROSE:
                    allowed = _JOB_PROSE[name]
                else:
                    kind = record.get("kind") if isinstance(record, dict) else None
                    prose = {
                        "diagnosis": _DIAGNOSIS_PROSE,
                        "fix_set": _FIX_PROSE,
                        "decision": _DECISION_PROSE,
                        "job_account": _JOB_ACCOUNT_PROSE,
                    }.get(str(kind), set())
                    allowed = {("answer", *item) for item in prose}
                collect(record, where, allowed)
    return values


def audit_state(root: Path, runs: tuple[Path, ...], fleet: Fleet, state: str) -> LeakReport:
    """Keep pipeline state strict; rescan only recognized agent prose as agent.

    The first scan strictly covers every byte, key and filename. Replace
    findings only at known prose locations with the agent-text scan, keeping
    their original file/field locations. JSON strings inside prose are decoded
    by that scan too. Unknown files/fields and all results/box contents stay ours.
    """
    from fleetlab.log import audit_export

    report = audit_export(root, runs, fleet=fleet)
    values = _agent_prose(root, state)
    if not values:
        return report
    # Source labels retain the original file/field. The .json representation
    # wraps each prose value as one string so nested JSON/seed checks still run;
    # no pipeline-authored fields or code enter agent_texts, and its tree is empty.
    locations = {where + " (agent prose).json": where for where in values}
    agent_texts = {label: json.dumps(values[where]) for label, where in locations.items()}
    with tempfile.TemporaryDirectory(prefix="fleetlab-state-audit-") as empty:
        rescanned = audit_export(empty, runs, fleet=fleet, agent_texts=agent_texts)

    def agent_location(where: str) -> bool:
        parent, decoded, _ = where.partition(" (decoded JSON)")
        return where in values or bool(decoded and parent.replace(" key ", " at ", 1) in values)

    report.hits[:] = [hit for hit in report.hits if not agent_location(hit.where)]
    for hit in rescanned.hits:
        where = hit.where
        for prefix, original in locations.items():
            if where.startswith(prefix + " at "):
                where = original + where.removeprefix(prefix + " at ")
                break
            if where.startswith(prefix + " key "):
                where = original.replace(" at ", " key ", 1) + where.removeprefix(prefix + " key ")
                break
        report.hits.append(replace(hit, where=where))
    report.searched.update({"agent " + key: value for key, value in rescanned.searched.items()})
    return report


def audit_learned_skill(skill: Path, runs: tuple[Path, ...], fleet: Fleet) -> list[str]:
    """Why one learned skill directory may not be mounted; empty when it may.

    The skill is agent-written prose in a fixed template, checked against the
    truth side of the fleet in `runs` and `fleet`: the agent-text leak rules
    (`audit_export` with `agent_texts`: canaries, fault identifiers, planted
    names in their identifier forms, seeds), the rules of a mounted skill
    (`validate_skill_general`: a valid manifest, no neutral identifier, no
    world-specific term of `code_terms`), the learned skill's own front matter
    (`origin: learned`, `applies_when`, `provenance`) and its sections, and
    nothing in the directory but its `SKILL.md`.
    """
    from fleetlab.agents.checks import CODE_EXEMPT_TERMS
    from fleetlab.contracts.learned import LEARNED_SECTIONS
    from fleetlab.contracts.skills import SKILL_FILE, load_skill, validate_skill_general
    from fleetlab.log import audit_export
    from fleetlab.log.leaks import code_terms

    name = skill.name
    if skill.is_symlink() or not skill.is_dir():
        return [f"{name}: not a plain directory"]
    extra = sorted(p.name for p in skill.iterdir() if p.name != SKILL_FILE)
    problems = (
        [f"{name}: only {SKILL_FILE} belongs in a learned skill, not {extra}"] if extra else []
    )
    path = skill / SKILL_FILE
    if path.is_symlink() or not path.is_file():
        return [*problems, f"{name}: no plain {SKILL_FILE}"]
    terms = [t for t in code_terms(runs, fleet=fleet) if t not in CODE_EXEMPT_TERMS]
    problems += validate_skill_general(skill, terms)
    text = path.read_text(encoding="utf-8")
    try:
        manifest = load_skill(skill).manifest
    except (OSError, ValueError):
        manifest = None  # validate_skill_general reported it
    if manifest is not None and (
        manifest.origin != "learned"
        or not (manifest.applies_when or "").strip()
        or not (manifest.provenance or "").strip()
    ):
        problems.append(f"{name}: front matter lacks origin: learned, applies_when or provenance")
    missing = [s for s in LEARNED_SECTIONS if f"\n## {s}\n" not in text]
    if missing:
        problems.append(f"{name}: sections {missing} are missing")
    with tempfile.TemporaryDirectory(prefix="fleetlab-learned-audit-") as empty:
        report = audit_export(
            empty, runs, fleet=fleet, agent_texts={f"{name}/{SKILL_FILE}.json": json.dumps(text)}
        )
    problems += [f"{name}: {hit.term!r} in {hit.context!r}" for hit in report.hits]
    return problems


def audit_learned_library(
    root: Path,
    runs: tuple[Path, ...],
    fleet: Fleet,
    base_names: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Why a library of learned skills may not be mounted beside the base skills; [] if it may."""
    if root.is_symlink() or not root.is_dir():
        return [f"{root}: the learned library is not a plain directory"]
    problems: list[str] = []
    for skill in sorted(root.iterdir()):
        if skill.name in base_names:
            problems.append(f"{skill.name}: a learned skill may not take a base skill's name")
        problems += audit_learned_skill(skill, runs, fleet)
    return problems


def launch_audit(
    access: ToolAccess,
    box: Path,
    texts: Mapping[str, str],
) -> dict[str, Any]:
    """Recheck exactly what this session will see; no reusable data-audit approval.

    The manifest brackets the scan to detect changes during it. The receipt
    belongs beside the CLI box, where neither sandbox can read it.
    """
    from fleetlab.agents.checks import GateError, source_audit
    from fleetlab.log import LogReader
    from fleetlab.log.export import export_problems
    from fleetlab.toolserver.files import EXPERIMENTS_FILE, PROBES_FILE, RESULTS_FILE
    from fleetlab.toolserver.public_results import load_probe_outcomes, load_results

    started = time.perf_counter()
    context = access.audit
    if context is None or not context.run_dirs or context.fleet is None:
        raise GateError("agent launch requires truth-side audit inputs for this export")
    runs = context.runs()
    roots = {
        "export": access.export,
        "source": access.env.src,
        "memory": access.memory,
        "outbox": access.outbox,
        "results": access.results,
        "box": box,
    }
    if access.env.skills is not None:
        roots["skills"] = access.env.skills
    if access.learned is not None:
        roots["learned"] = access.learned
    before = {name: tree_manifest(root) for name, root in roots.items()}
    if export_problems(access.export):
        raise GateError("mounted export fails structural validation")
    description = json.loads((access.export / "log" / "description.json").read_text())
    if description.get("fleet", {}).get("id") != context.fleet.id:
        raise GateError("export and audit fleet ids differ")
    try:
        with LogReader(access.export) as reader:
            rows = reader.query(
                "SELECT episode_id, experiment_id, robot_id, day FROM episodes"
            ).to_pylist()
        arm_days = [r["day"] for r in rows if r["experiment_id"] is not None]
        if arm_days and (
            context.experiment_days is None
            or any(
                not context.experiment_days[0] <= day <= context.experiment_days[1]
                for day in arm_days
            )
        ):
            raise ValueError("experiment episodes lie outside the authorized experiment window")
        load_results(
            access.results / RESULTS_FILE,
            access.outbox / EXPERIMENTS_FILE,
            access.session,
            {r["episode_id"]: (r["experiment_id"], r["robot_id"]) for r in rows},
        )
    except ValueError:
        raise GateError("mounted experiment results fail public validation") from None
    try:
        # Probes the agent composed (contracts.probes): the same binding the tool server
        # checks on every read, before anything is mounted.
        with LogReader(access.export) as reader:
            probe_rows = {
                r["episode_id"]: (r["probe_id"], r["robot_id"], r["repeat"])
                for r in reader.query(
                    "SELECT episode_id, probe_id, robot_id, repeat FROM probe_episodes"
                ).to_pylist()
            }
            with_steps = set(
                reader.query("SELECT DISTINCT episode_id FROM probe_steps")
                .column("episode_id")
                .to_pylist()
            )
        load_probe_outcomes(
            access.results / PROBES_FILE,
            access.outbox / PROBES_FILE,
            access.session,
            probe_rows,
            with_steps,
        )
    except ValueError:
        raise GateError("mounted probe results fail public validation") from None
    report = source_audit(access.export, runs, access.env, fleet=context.fleet, texts=texts)
    if not report.passed:
        raise GateError("launch source audit: " + "; ".join(report.failures()))
    for name in ("memory", "outbox", "results", "box"):
        scan = audit_state(roots[name], runs, context.fleet, name)
        if not scan.clean:
            raise GateError(f"launch {name} audit: {scan.summary()}")
    if access.learned is not None:
        base = (
            {p.name for p in access.env.skills.iterdir()}
            if access.env.skills is not None
            else set()
        )
        learned = audit_learned_library(access.learned, runs, context.fleet, base)
        if learned:
            raise GateError("launch learned-library audit: " + "; ".join(learned))
    after = {name: tree_manifest(root) for name, root in roots.items()}
    if before != after:
        raise GateError("mounted input changed during the launch audit")
    inputs = {
        "files": before,
        "texts": dict(texts),
        "tools": access.enabled,
        "session": access.session,
        "max_experiment_episodes": access.max_experiment_episodes,
        "learned": None if access.learned is None else tree_digest(access.learned),
        **{
            name: value
            for name in ("max_experiment_requests", "max_episodes_per_arm", "budget_first_session")
            if (value := getattr(access, name)) is not None
        },
        "fleet_sha256": hashlib.sha256(context.fleet.model_dump_json().encode()).hexdigest(),
        "run_dirs": [str(p.resolve()) for p in runs],
        "experiment_days": context.experiment_days,
    }
    return {
        "passed": True,
        "sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "manifest": before,
        "source_audit": report.as_json(),
        "seconds": time.perf_counter() - started,
    }
