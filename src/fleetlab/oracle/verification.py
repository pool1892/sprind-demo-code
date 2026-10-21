"""Recorded-chain majority rule, shared by proofs and conflicting oracle rows.

The engine's chain row is AFTER an executed action. At the first divergent row,
compare the action before its resulting state/images. No tolerance is applied.
Ceilings require per-job policy request totals; missing totals refuse.
Today's records do not retain per-call hooked observations and returned chunks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import Job, JobMode, RunLayout, RunRecord
from fleetlab.engine.chain import Chain
from fleetlab.oracle.run import RunRefused

# Garwood upper 95% endpoint: chi2.ppf(.975, 2 * (3 + 1)) / 2.
MEASURED_EVENTS_UPPER_95 = 8.767273069742323
MEASURED_CALLS = 26928
POLICY_CALL_RATE_BOUND = MEASURED_EVENTS_UPPER_95 / MEASURED_CALLS
EVENT_CEILING_COVERAGE = 0.999
RULE = "recorded-action-four-executions-v2"
LEGACY_RULE = "recorded-action-majority-v1"


def exact_interval(events: int, steps: int) -> list[float | None]:
    """Clopper-Pearson 95% interval per million comparison units."""
    from scipy.stats import beta

    if not steps:
        return [None, None]
    return [
        0.0 if not events else float(beta.ppf(0.025, events, steps - events + 1)) * 1e6,
        1e6 if events == steps else float(beta.ppf(0.975, events + 1, steps - events)) * 1e6,
    ]


def event_ceiling(steps: int, rate: float = POLICY_CALL_RATE_BOUND) -> int:
    from scipy.stats import poisson

    if not steps:
        return 0
    ceiling = int(poisson.ppf(EVENT_CEILING_COVERAGE, steps * rate))
    # The note requires a strict tail probability below .001.
    while poisson.sf(ceiling, steps * rate) >= 1 - EVENT_CEILING_COVERAGE:
        ceiling += 1
    return ceiling


def full_equal(a: Chain, b: Chain) -> bool:
    return all(a.arrays()[k].tobytes() == b.arrays()[k].tobytes() for k in a.arrays())


def first_difference(a: Chain, b: Chain) -> tuple[int | None, str]:
    """Causal order: reset, then action, then resulting state and camera images."""
    if not len(a) or not len(b) or a.first_step != 0 or b.first_step != 0:
        return None, "empty chain or not from reset"
    if a.fixture_hash != b.fixture_hash:
        return 0, "fixture_hash"
    for row in range(min(len(a), len(b))):
        step = a.steps[row]
        if step != b.steps[row] or step != row:
            return step, "step"
        if a.executed[row].tobytes() != b.executed[row].tobytes():
            return step, "executed_action" if row else "reset action"
        for field in ("state", "scene", "wrist"):
            if getattr(a, field)[row] != getattr(b, field)[row]:
                return step, field
    if len(a) != len(b):
        return min(len(a), len(b)), "length"
    return None, ""


@dataclass(frozen=True)
class Execution:
    run: Path
    job: Job
    success: bool

    @property
    def chain_path(self) -> Path:
        return RunLayout(self.run).chain(self.job)

    def evidence(self) -> dict[str, Any]:
        return {
            "run": str(self.run.resolve()),
            "job": self.job.model_dump(mode="json"),
            "success": self.success,
            "chain_sha256": hashlib.sha256(self.chain_path.read_bytes()).hexdigest(),
        }


def outcome(run: Path, job: Job) -> bool:
    from fleetlab.log.runfiles import read_table

    rows = [r for r in read_table(run, "job_outcomes").to_pylist() if r["job_key"] == job.key]
    if len(rows) != 1:
        raise RunRefused(f"{run}/{job.key}: missing or ambiguous outcome")
    return bool(rows[0]["success"])


Executor = Callable[[Path, Job], Execution]


def recorded_call_totals(run: Path) -> dict[str, int]:
    """Exact robot-request counts, never inferred from duration or action chunks.

    Finished replay chains contain no call metadata. If the engine was killed
    before saving its job summaries, their missing totals cannot be recovered
    from the chain, a nominal chunk length, or the run's server-batch count.
    """
    path = run / "run.json"
    record = RunRecord.model_validate_json(path.read_text()) if path.is_file() else None
    details: Any = record.extra.get("jobs_detail", []) if record else []
    return {
        str(r["key"]): r["policy_calls"]
        for r in details
        if isinstance(r, dict) and type(r.get("policy_calls")) is int and r["policy_calls"] >= 0
    }


class ReplayVerifier:
    """Persist immutable evidence; replay it on reads, including the rate ceiling.

    `execute` is available only inside the run/verify job. Report never simulates.
    Further executions have separate run directories; resumption reuses their files.
    """

    def __init__(
        self,
        folder: Path,
        windows: dict[str, str],
        world_hash: str | None,
        execute: Executor | None = None,
    ) -> None:
        self.folder, self.windows, self.world_hash, self.execute = (
            folder,
            windows,
            world_hash,
            execute,
        )
        self.records: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[Path, dict[str, bool]] = {}
        self._calls: dict[Path, dict[str, int]] = {}
        self._canonical: dict[tuple[str, str], set[bool]] = {}
        self.required: dict[str, list[str]] = {}
        self.path = folder / "comparisons.json"
        if self.path.is_file():
            stored = json.loads(self.path.read_text())
            if stored["world_hash"] != world_hash or stored["rule"] not in (RULE, LEGACY_RULE):
                raise RunRefused(f"{self.path}: verification belongs to another world or rule")
            self.records = stored["comparisons"]
            self.required = stored.get("required_windows", {})
            for record in self.records.values():
                self.index(record)

    def index(self, record: dict[str, Any]) -> None:
        for e in record["evidence"]:
            key = (e["run"], Job.model_validate(e["job"]).key)
            self._canonical.setdefault(key, set()).add(record["success"])

    def recorded_outcome(self, run: Path, job: Job) -> bool:
        from fleetlab.log.runfiles import read_table

        if run not in self._outcomes:
            self._outcomes[run] = {
                r["job_key"]: bool(r["success"])
                for r in read_table(run, "job_outcomes").to_pylist()
            }
        if job.key not in self._outcomes[run]:
            raise RunRefused(f"{run}/{job.key}: outcome missing")
        return self._outcomes[run][job.key]

    def calls(self, execution: Execution) -> int | None:
        if execution.job.key not in self._calls.get(execution.run, {}):
            self._calls[execution.run] = recorded_call_totals(execution.run)
        return self._calls[execution.run].get(execution.job.key)

    def refresh(self, run: Path) -> None:
        """A resumed engine run has appended outcomes and per-job call totals."""
        for path in {run, run.resolve()}:
            self._outcomes.pop(path, None)
            self._calls.pop(path, None)

    def settle(self, left: Execution, right: Execution) -> dict[str, Any]:
        if (
            left.job.identity != right.job.identity
            or left.job.effective_faults != right.job.effective_faults
        ):
            raise RunRefused("a comparison needs the same identity and fault list")
        a, b = Chain.load(left.chain_path), Chain.load(right.chain_path)
        step, field = first_difference(a, b)
        evidence = [left.evidence(), right.evidence()]
        left_calls, right_calls = self.calls(left), self.calls(right)
        key = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        stored = self.records.get(key)
        record: dict[str, Any] = {
            "episode_id": left.job.identity.episode_id,
            "run_dir": str(left.run.resolve()),
            "window": self.windows.get(left.job.identity.episode_id, "proof outside named windows"),
            "compared_steps": len(b) - 1,
            "compared_calls": right_calls,
            "execution_calls": [left_calls, right_calls],
            "execution_steps": [len(a) - 1, len(b) - 1],
            "first_difference": step,
            "evidence": evidence,
            "identical": not field,
            "odd": None,
            "outcome_differs": left.success != right.success,
            "success": left.success,
        }
        if not field:
            if left.success != right.success:
                raise RunRefused("identical chains have disagreeing outcomes")
        else:
            if field != "executed_action":
                raise RunRefused(f"{left.job.key}: first difference is {field} at {step}")
            kinds = [left.job.mode, right.job.mode]
            if any(mode not in (JobMode.run, JobMode.replay) for mode in kinds):
                raise RunRefused("four-execution settlement needs run and replay jobs from reset")
            further: list[Execution] = []
            needed = [
                mode for mode in (JobMode.run, JobMode.replay) for _ in range(2 - kinds.count(mode))
            ]
            if stored and stored.get("third"):
                old_mode = Job.model_validate(stored["third"]["job"]).mode
                if old_mode in needed:
                    needed.remove(old_mode)
                    needed.insert(0, old_mode)
            for ordinal, mode in enumerate(needed):
                label = "third" if ordinal == 0 else "fourth"
                data = stored.get(label) if stored else None
                # Round-six thirds were replay jobs; preserve them only in that slot.
                if data and Job.model_validate(data["job"]).mode is not mode:
                    data = None
                if data:
                    job = Job.model_validate(data["job"])
                    execution = Execution(
                        Path(data["run"]), job, self.recorded_outcome(Path(data["run"]), job)
                    )
                    if execution.evidence() != data:
                        raise RunRefused(f"the {label} execution's evidence changed")
                elif self.execute is not None:
                    if mode is JobMode.run:
                        # The real engine run path: run jobs cannot carry remove/add.
                        identity = left.job.identity.model_copy(
                            update={"faults": left.job.effective_faults}
                        )
                        job = Job(identity=identity)
                    else:
                        job = left.job.model_copy(update={"mode": JobMode.replay})
                    try:
                        execution = self.execute(self.folder / label / key, job)
                    except Exception as exc:
                        raise RunRefused(
                            f"{left.job.key}: {label} execution could not be made: {exc}"
                        ) from exc
                else:
                    raise RunRefused(
                        f"{left.job.key}: {label} execution required; run verify first"
                    )
                if (
                    execution.job.mode is not mode
                    or execution.job.identity.model_dump(exclude={"faults"})
                    != left.job.identity.model_dump(exclude={"faults"})
                    or execution.job.effective_faults != left.job.effective_faults
                    or execution.job.from_snapshot is not None
                ):
                    raise RunRefused("further execution changed the kind, identity or fault list")
                further.append(execution)
            all_executions = [left, right, *further]
            chains = [a, b, *(Chain.load(e.chain_path) for e in further)]
            classes = [
                [
                    j
                    for j, other in enumerate(chains)
                    if full_equal(chain, other)
                    and all_executions[i].success == all_executions[j].success
                ]
                for i, chain in enumerate(chains)
            ]
            majority = next((members for members in classes if len(members) == 3), None)
            if majority is None:
                raise RunRefused(
                    f"{left.job.key}: four executions have no three-to-one "
                    "agreement (replay defect or matches neither side in full)"
                )
            odd_index = next(i for i in range(4) if i not in majority)
            record.update(
                odd="left" if odd_index == 0 else "right",
                success=all_executions[majority[0]].success,
                third=further[0].evidence(),
                fourth=further[1].evidence(),
            )
        self.records[key] = record
        self.index(record)
        return record

    def proof(self, job: Job, original: Path, replay: Path) -> dict[str, Any]:
        original_run = original.parents[2]
        replay_run = replay.parents[2]
        as_run = Job(identity=job.identity)
        record = self.settle(
            Execution(original_run, as_run, self.recorded_outcome(original_run, as_run)),
            Execution(replay_run, job, self.recorded_outcome(replay_run, job)),
        )
        return {**record, "accepted": True, "what": "" if record["identical"] else RULE}

    def canonical(self, execution: Execution) -> bool:
        values = self._canonical.get((str(execution.run.resolve()), execution.job.key), set())
        if len(values) > 1:
            raise RunRefused("settlements disagree about the reproducible outcome")
        return next(iter(values)) if values else execution.success

    def validate(self, *, ceiling: bool = True) -> None:
        """Recompute every saved classification against its complete recorded evidence."""
        for saved in list(self.records.values()):
            executions = []
            for e in saved["evidence"]:
                job = Job.model_validate(e["job"])
                execution = Execution(
                    Path(e["run"]), job, self.recorded_outcome(Path(e["run"]), job)
                )
                if execution.evidence() != e:
                    raise RunRefused("a compared execution's evidence changed")
                executions.append(execution)
            self.settle(*executions)
        if ceiling:
            self.summary()  # the ceiling is checked even on read-only report

    def summary(self) -> dict[str, Any]:
        covered = {
            r["episode_id"]
            for r in self.records.values()
            if r["evidence"][0]["job"]["mode"] == JobMode.run.value
            and r["evidence"][1]["job"]["mode"] == JobMode.replay.value
            and not (r["evidence"][1]["job"]["remove"] or r["evidence"][1]["job"]["add"])
        }
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self.records.values():
            left, right = r["evidence"]
            group = groups.setdefault(
                (left["run"], r["window"]),
                {
                    "run_dir": left["run"],
                    "window": r["window"],
                    "attempts_executed_twice": set(),
                    "events": {},
                    "executions": {},
                },
            )
            key = (right["run"], Job.model_validate(right["job"]).key)
            if r["execution_calls"][1] is None:
                # A legacy ledger may omit a total which the resumed engine
                # subsequently saved. Recover only that exact attempt's count.
                execution = Execution(
                    Path(right["run"]), Job.model_validate(right["job"]), right["success"]
                )
                r["execution_calls"][1] = r["compared_calls"] = self.calls(execution)
            if r["execution_calls"][1] is None:
                raise RunRefused(
                    f"missing policy call total for second execution {right['run']}/{key[1]}; "
                    "the engine did not retain this attempt's request count; "
                    "executed steps cannot supply a policy-call ceiling"
                )
            group["executions"][key] = (r["execution_steps"][1], r["execution_calls"][1])
            if not (left["job"]["remove"] or left["job"]["add"]):
                group["attempts_executed_twice"].add(r["episode_id"])
            if r["odd"]:
                odd = left if r["odd"] == "left" else right
                event_key = (odd["run"], Job.model_validate(odd["job"]).key)
                group["events"][event_key] = {
                    "episode_id": r["episode_id"],
                    "step": r["first_difference"],
                    "odd_side": "as_run" if odd["job"]["mode"] == "run" else "replay",
                    "odd_run": odd["run"],
                    "outcome_differs": r["outcome_differs"],
                    "reproducible_success": r["success"],
                    "third_run": r["third"]["run"],
                    "fourth_run": r["fourth"]["run"],
                }
        result = []
        for g in groups.values():
            exposures = list(g.pop("executions").values())
            g["compared_steps"] = sum(n for n, _ in exposures)
            g["compared_calls"] = (
                sum(n for _, n in exposures) if all(n is not None for _, n in exposures) else None
            )
            steps, events = g["compared_steps"], list(g["events"].values())
            calls = g["compared_calls"]
            units = calls
            ceiling = event_ceiling(units, POLICY_CALL_RATE_BOUND)
            coverage_pending = bool(set(self.required.get(g["window"], [])) - covered)
            if not coverage_pending and len(events) > ceiling:
                raise RunRefused(
                    f"{g['run_dir']} {g['window']}: {len(events)} events exceed ceiling {ceiling}"
                )
            result.append(
                {
                    **g,
                    "attempts_executed_twice": len(g["attempts_executed_twice"]),
                    "events": events,
                    "event_ceiling": ceiling,
                    "ceiling_pending": coverage_pending,
                    "rate_per_million_steps": len(events) / steps * 1e6 if steps else None,
                    "rate_interval_95_per_million_steps": exact_interval(len(events), steps),
                    "rate_per_million_calls": len(events) / calls * 1e6 if calls else None,
                    "rate_interval_95_per_million_calls": exact_interval(len(events), calls)
                    if calls is not None
                    else [None, None],
                    "ceiling_unit": "policy request executions",
                }
            )
        coverage = [
            {
                "window": name,
                "attempts": len(episodes),
                "attempts_executed_twice": sum(e in covered for e in episodes),
                "missing": [e for e in episodes if e not in covered],
            }
            for name, episodes in self.required.items()
        ]
        return {
            "rule": RULE,
            "unit": "policy requests of second executions only",
            "windows": result,
            "coverage": coverage,
            "verified_episodes": sorted(covered),
            "policy_call_rate_bound": POLICY_CALL_RATE_BOUND,
            "step_rate_bound": POLICY_CALL_RATE_BOUND,
            "ceiling_coverage": EVENT_CEILING_COVERAGE,
        }

    def save(self) -> None:
        if any(r.get("odd") and "fourth" not in r for r in self.records.values()):
            # Upgrade a saved v1 event before the v2 summary can name its fourth.
            self.validate(ceiling=False)
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "world_hash": self.world_hash,
                    "rule": RULE,
                    "comparisons": self.records,
                    "required_windows": self.required,
                },
                indent=1,
            )
            + "\n"
        )
        temporary.replace(self.path)
        # Preserve all classified evidence even if the completed window refuses.
        self.summary()


def once_replay_risk(executions: Sequence[Execution], scoring: int) -> dict[str, Any]:
    """Union bound over recorded requests; missing totals use a labelled budget.

    A duplicated changed job supplies one outcome to the table. Use the largest
    exposure of its executions, including the budget of any missing summary.
    """
    changed: dict[str, list[Execution]] = {}
    totals: dict[Path, dict[str, int]] = {}
    for execution in executions:
        if execution.job.remove or execution.job.add:
            changed.setdefault(execution.job.key, []).append(execution)
            if execution.run not in totals:
                totals[execution.run] = recorded_call_totals(execution.run)
    exposures = []
    recorded_expected = 0.0
    fallback: list[str] = []
    recorded = budget = 0
    for key, copies in changed.items():
        calls = [totals[e.run].get(key) for e in copies]
        if any(n is None for n in calls):
            fallback.append(key)
            units = max(
                e.job.identity.max_steps if n is None else n
                for e, n in zip(copies, calls, strict=True)
            )
            budget += units
        else:
            units = max(n for n in calls if n is not None)
            recorded += units
            recorded_expected += min(1.0, units * POLICY_CALL_RATE_BOUND)
        exposures.append(units)
    expected = sum(min(1.0, n * POLICY_CALL_RATE_BOUND) for n in exposures)
    points = min(100.0, 100 * expected / scoring) if scoring else None
    return {
        "changed_replays": len(changed),
        "recorded_policy_calls": recorded,
        "recorded_expected_affected_replays_upper": recorded_expected,
        "recorded_expected_wrong_outcome_points_upper_per_table": 100 * recorded_expected / scoring
        if scoring
        else None,
        "step_budget_fallback_replays": len(fallback),
        "step_budget_fallback_jobs": fallback,
        "fallback_step_budget": budget,
        "expected_affected_replays_upper": expected,
        "expected_wrong_outcome_points_upper_per_table": points,
        "wrong_outcome_points_95_upper_per_table": min(100.0, 20 * points)
        if points is not None
        else None,
        "text": f"Recorded policy calls supply {len(changed) - len(fallback)} changed replay "
        f"exposures; {len(fallback)} changed replay exposures use the maximum step budget "
        "because their call totals are missing. "
        "A replay that changes something is executed once, so a disturbance there "
        "is not detected. Bounds assume every disturbance changes an outcome; "
        "the 95% bound uses Markov's inequality. "
        "Interaction contrasts can use four outcomes: multiply these point bounds by four.",
    }
