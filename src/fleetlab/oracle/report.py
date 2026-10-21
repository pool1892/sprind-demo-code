"""The oracle's report of one world: every gain with its interval, the patterns confirmed,
the detectability check, the chance level, the per-episode verdicts, and the files the
dashboard's truth view and the scorer read.

Files (`OracleLayout`): `oracle.json` (`OracleReport`), `summary.md` (the same as
Markdown tables), `episodes.parquet` (per scoring episode: faults, toggled faults, the
reproducible as-run outcome and outcome with every fault removed), and
`truth/oracle.parquet` in a run
directory (the contract's table: item, item_kind, rescued, broken, net), written
through `fleetlab.log.RunWriter`. The report holds no wall-clock time, so the same table
gives the same files; timings go to `steps.json`.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fleetlab.contracts.actions import COSTS, SCORING_BUDGETS
from fleetlab.contracts.oracle import POWER_SET_CAP
from fleetlab.oracle.causes import MeasuredCauses
from fleetlab.oracle.chance import DRAWS, Chance, budget_label, chance_intervals
from fleetlab.oracle.detect import Detectability, check_detectability, misses
from fleetlab.oracle.exact import ExactChance
from fleetlab.oracle.gains import BOOT, Estimate, Gains, fault_specs
from fleetlab.oracle.plan import OraclePlan
from fleetlab.oracle.power import ASSUMPTIONS, METHOD, POWER, sufficient_power_bound
from fleetlab.oracle.table import TableBuild
from fleetlab.oracle.world import OracleWorld

# A drift is confirmed when the net gain per carrying episode rises by at least this many
# points from the first to the last scoring day; a smaller rise is "weak" (one or two
# episodes of the fourteen a robot runs a day are within what a decoy flips).
DRIFT_RISE = 10.0
# Excess is compared with this fraction of nonnegative summed standalone harm.
ADDITIVE_RELATIVE_TOLERANCE = 0.10
ADDITIVE_DETECTION_POWER = POWER  # Retained target-power constant; no observed-SE estimate.


class PatternCheck(BaseModel):
    """Whether the oracle confirms one pattern of the world (gate 3)."""

    model_config = ConfigDict(extra="forbid")

    pattern: str  # decoy, inert, confounder, interaction, additive_pair, drift
    items: list[str]
    verdict: str  # additive: contradicted, below the sum, not contradicted
    text: str


class OracleReport(BaseModel):
    """`oracle.json`: what the oracle measured on one world's scoring window."""

    model_config = ConfigDict(extra="forbid")

    world_id: str
    world_hash: str | None
    scoring_episodes: int
    rows: int
    plan: dict[str, Any] | None
    runs: list[dict[str, Any]]
    over_cap: list[str]
    excluded: list[str]
    marked_interactions: list[list[str]]
    marked_additive_pairs: list[list[str]] = []
    drifts: list[str]
    boot: int
    seed: int
    draws: int
    fault_effects: list[Estimate]
    by_day: list[Estimate]
    interactions: list[Estimate]
    causes: list[Estimate]  # the joint removal of each cause of several faults
    achievable: list[Estimate]
    all_faults: Estimate
    still_failing: int | None
    policy_failures_on_healthy_robot: int | None = None  # alias of still_failing
    all_removed_known_failures: int | None = None
    all_removed_missing: int = 0
    chance: list[Chance]
    detectability: list[Detectability]
    patterns: list[PatternCheck]
    problems: list[str]
    notes: list[str]
    replay_verification: dict[str, Any] = Field(default_factory=dict)
    measured_causes: dict[str, Any] = Field(default_factory=dict)

    def effect(self, fault_id: str) -> Estimate | None:
        return next((e for e in self.fault_effects if e.item == fault_id), None)

    def achievable_at(self, label: str) -> Estimate | None:
        return next((e for e in self.achievable if e.item == label), None)


def _label_of(estimate: Estimate) -> str:
    detail = estimate.detail
    if detail.get("budget") == "count":
        return f"k = {detail['amount']}"
    return f"{float(detail['amount']):g} h"


def interaction_confirmed(term: Estimate) -> bool:
    """Strict excess-harm test, including agreement of the point and interval."""
    return term.net is not None and term.net < 0 and term.high is not None and term.high < 0


def additive_verdict(term: Estimate, scoring_episodes: int) -> tuple[str, float | None]:
    """Only intervals outside the tolerance band contradict it; no confirmation test."""
    summed = term.detail.get("summed_harm_net")
    bound = (
        100.0 * ADDITIVE_RELATIVE_TOLERANCE * max(0, summed) / scoring_episodes
        if summed is not None and scoring_episodes
        else None
    )
    if bound is not None and term.low is not None and term.high is not None:
        if term.high < -bound:  # excess = -term
            return "contradicted", bound
        if term.low > bound:
            return "below the sum", bound
    return "not contradicted", bound


def additive_numbers(term: Estimate, scoring_episodes: int, boot: int = BOOT) -> dict[str, Any]:
    """Positive excess harm and a design-only sufficient power bound."""
    _, band = additive_verdict(term, scoring_episodes)
    power = sufficient_power_bound(
        scoring_episodes, term.episodes, boot, ADDITIVE_RELATIVE_TOLERANCE
    )
    return {
        "excess_net": None if term.net is None else -term.net,
        "excess_points": None if term.points is None else -term.points,
        "excess_low": None if term.high is None else -term.high,
        "excess_high": None if term.low is None else -term.low,
        "summed_harm_points": (
            round(100 * term.detail["summed_harm_net"] / scoring_episodes, 3)
            if scoring_episodes and "summed_harm_net" in term.detail
            else None
        ),
        "max_excess_not_excluded_points": None if term.low is None else -term.low,
        # Preserve old keys for readers, but withdraw the misleading advisory values.
        "min_reliable_excess_points": None,
        "reliable_excess_power": None,
        "reliable_excess_method": "withdrawn: observed-SE estimate did not guarantee power",
        "sufficient_excess_power_bound_points": power.points,
        "excess_power_guarantee": POWER if power.points is not None else None,
        "excess_power_method": METHOD,
        "excess_power_assumptions": ASSUMPTIONS,
        "excess_power_unavailable": power.unavailable,
        "additive_tolerance_points": band,
    }


def pattern_checks(world: OracleWorld, report: dict[str, Any]) -> list[PatternCheck]:
    effects: dict[str, Estimate] = {e.item: e for e in report["fault_effects"]}
    out: list[PatternCheck] = []

    def zero(kind: str, fault_id: str) -> PatternCheck:
        effect = effects.get(fault_id)
        if effect is None:
            return PatternCheck(
                pattern=kind, items=[fault_id], verdict="missing", text="no scoring episode"
            )
        if effect.points is None:
            return PatternCheck(
                pattern=kind, items=[fault_id], verdict="missing", text=effect.missing or ""
            )
        held = effect.zero_within_noise
        return PatternCheck(
            pattern=kind,
            items=[fault_id],
            verdict="missing" if held is None else "confirmed" if held else "not confirmed",
            text=(
                f"removing it: {effect.points:+.2f} points ({effect.rescued} rescued, "
                f"{effect.broken} broken), interval {_interval(effect)}"
                + ("" if held is None else ", zero within noise" if held else ", not zero")
            ),
        )

    for fault_id in world.spec.patterns.decoys:
        out.append(zero("decoy", fault_id))
    for effective, inert in world.spec.patterns.confounders:
        check = zero("inert", inert)
        main = effects.get(effective)
        strong = main is not None and main.points is not None and main.points >= 2.0
        verdict = check.verdict if strong or check.verdict == "missing" else "not confirmed"
        text = check.text + (
            f"; the effective {effective}: "
            + ("missing" if main is None or main.points is None else f"{main.points:+.2f} points")
        )
        out.append(
            PatternCheck(pattern="confounder", items=[effective, inert], verdict=verdict, text=text)
        )
    interactions = {e.item: e for e in report["interactions"]}
    pairs = [("interaction", p) for p in world.spec.patterns.interactions] + [
        ("additive_pair", p) for p in marked_additive_pairs(world)
    ]
    for kind, (a, b) in pairs:
        term = interactions.get(f"{a}+{b}") or interactions.get(f"{b}+{a}")
        if term is None or term.net is None:
            out.append(
                PatternCheck(
                    pattern=kind,
                    items=[a, b],
                    verdict="missing",
                    text="no episode carries both" if term is None else term.missing or "",
                )
            )
            continue
        # The stored term is joint removal minus single removals. Super-additive
        # fault harm has the opposite sign, so its interval must lie BELOW zero.
        excess = interaction_confirmed(term)
        verdict = "confirmed" if excess else "not confirmed"
        alone = term.detail.get("alone_net", [None, None])
        extra = ""
        if kind == "additive_pair":
            summed = term.detail.get("summed_harm_net")
            verdict, bound = additive_verdict(term, report.get("scoring_episodes", 0))
            harm_low = term.detail.get("summed_harm_low")
            harm_high = term.detail.get("summed_harm_high")
            harm_interval = (
                "unavailable"
                if harm_low is None or harm_high is None
                else f"{harm_low:+.3f} to {harm_high:+.3f} points"
            )
            extra = (
                f"; summed standalone harm {summed} net, additive tolerance "
                f"{ADDITIVE_RELATIVE_TOLERANCE:.0%} (band {-bound:+.3f} to {bound:+.3f} points); "
                f"summed harm interval {harm_interval}"
                if bound is not None
                else "; additive interval or tolerance unavailable"
            )
            numbers = additive_numbers(
                term, report.get("scoring_episodes", 0), report.get("boot", BOOT)
            )
            extra += (
                f"; excess {numbers['excess_net']} net ({_pts(numbers['excess_points'])} points), "
                f"interval {numbers['excess_low']} to {numbers['excess_high']} points; "
                f"largest excess not excluded {numbers['max_excess_not_excluded_points']} points; "
                f"conservative sufficient 80% power bound "
                f"{numbers['sufficient_excess_power_bound_points']} points "
                f"(independent carrier samples and bootstrap draws; not a minimum)"
            ).replace("None", "unavailable")
        out.append(
            PatternCheck(
                pattern=kind,
                items=[a, b],
                verdict=verdict,
                text=(
                    f"joint removal {term.detail.get('joint_net')} net against {alone[0]} and "
                    f"{alone[1]} alone over {term.episodes} episodes: interaction {term.net:+d} "
                    f"({term.points:+.2f} points, interval {_interval(term)})" + extra
                ),
            )
        )
    by_day: dict[str, list[Estimate]] = {}
    for e in report["by_day"]:
        by_day.setdefault(str(e.detail["fault_id"]), []).append(e)
    for fault_id in world.spec.patterns.drifts:
        days = by_day.get(fault_id, [])
        valued = [d for d in days if d.net is not None]
        if len(valued) < 2:
            out.append(
                PatternCheck(
                    pattern="drift",
                    items=[fault_id],
                    verdict="missing",
                    text=f"{len(valued)} scoring days valued; a rise needs two",
                )
            )
            continue
        rates = [
            (int(d.detail["day"]), 100.0 * (d.net or 0) / d.episodes if d.episodes else 0.0)
            for d in valued
        ]
        rise = rates[-1][1] - rates[0][1]
        verdict = "confirmed" if rise >= DRIFT_RISE else "weak" if rise > 0 else "not confirmed"
        counts = ", ".join(
            f"day {d.detail['day']} {d.net:+d} of {d.episodes}" for d in valued if d.net is not None
        )
        out.append(
            PatternCheck(
                pattern="drift",
                items=[fault_id],
                verdict=verdict,
                text=f"net gain over the carrying episodes by day: {counts} "
                f"(a rise of {rise:+.0f} points of them)",
            )
        )
    return out


def marked_additive_pairs(world: OracleWorld) -> list[tuple[str, str]]:
    """Truth-only pairs, including the shared contract once its parallel change lands."""
    return sorted(
        {
            tuple(sorted(p))
            for p in [*world.additive_pairs, *getattr(world.spec.patterns, "additive_pairs", [])]
        }
    )


def compute_report(
    world: OracleWorld,
    build: TableBuild,
    plan: OraclePlan | dict[str, Any] | None = None,
    *,
    boot: int = BOOT,
    draws: int = DRAWS,
    seed: int | None = None,
    cap: int = POWER_SET_CAP,
    measured: MeasuredCauses | None = None,
) -> tuple[OracleReport, list[dict[str, Any]]]:
    """Everything the oracle measures over the table (module docstring), and the rows of
    `episodes.parquet`."""
    table = build.table
    seed = world.spec.seed if seed is None else seed
    gains = Gains(table, boot=boot, seed=seed)
    truth = world.truth()
    notes = list(world.notes)
    effects = [gains.fault_effect(f) for f in table.fault_ids()]
    drifting = {
        f.fault_id for i in table.identities for f in i.faults if f.schedule.drift is not None
    } | set(world.spec.patterns.drifts)
    by_day = [
        e for f in sorted(drifting) if f in table.fault_ids() for e in gains.fault_effect_by_day(f)
    ]
    marked = {tuple(sorted(p)) for p in world.spec.patterns.interactions}
    additive = set(marked_additive_pairs(world))
    together = sorted(
        {pair for i in table.identities for pair in itertools.combinations(sorted(i.fault_ids), 2)}
        | marked
        | additive
    )
    interactions = [gains.interaction(a, b, (a, b) in marked | additive) for a, b in together]
    for (a, b), term in zip(together, interactions, strict=True):
        if (a, b) in additive:
            term.detail.update(additive_numbers(term, len(table.identities), boot))
    if additive:
        notes.append(
            "Additivity rests on the world's calibration, not on this scoring window; "
            "not contradicted is not confirmation."
        )
    causes = []
    for cause in world.causes:
        if len(cause.fault_ids) > 1:
            vector, needs = gains.removal(cause.fault_ids)
            touched = len({i for f in cause.fault_ids for i in gains.carrying(f)})
            causes.append(gains.estimate(cause.unit, "cause", vector, needs, touched))
    specs = fault_specs(table.identities)
    achievable = gains.achievable(specs, truth, COSTS)
    all_faults, episode_rows = gains.all_faults()
    still = sum(1 for r in episode_rows if r["all_removed_success"] is False)
    unknown = sum(1 for r in episode_rows if r["all_removed_success"] is None)
    if unknown:
        notes.append(f"{unknown} scoring episodes have no all-faults-removed row")
    best = {_label_of(e): e.net for e in achievable if e.detail.get("budget") in ("count", "cost")}
    entries = world.catalogue()
    if not entries:
        notes.append("no fleet to build the catalogue from: the chance level is missing")
    chance = chance_intervals(
        table,
        truth,
        world.current(),
        entries,
        seed=world.spec.seed,
        achievable={budget_label(b): best.get(budget_label(b)) for b in SCORING_BUDGETS},
        draws=draws,
    )
    detectability: list[Detectability] = []
    if world.fault_table is not None or world.causes:
        detectability = check_detectability(
            world.fault_table,
            {e.item: e for e in effects},
            world.causes,
            {e.item: e for e in causes},
        )
    problems = build.problems()
    if all_faults.missing:
        problems.append(f"incomplete all-faults-removed result: {all_faults.missing}")
    if unknown:
        problems.append(f"incomplete healthy-policy failure count: {unknown} missing outcomes")
    for row in misses(detectability) if measured is None else []:
        problems.append(
            ("incomplete " if row.verdict == "missing" else "")
            + f"detectability: {row.item} is detectable in the generator's list "
            f"(qualified by {', '.join(row.qualified_by) or 'nothing'}) but the oracle says "
            f"{row.verdict}"
            + ("" if row.oracle_points is None else f" ({row.oracle_points:+.2f} points)")
        )
    if measured is not None:
        problems.extend(f"incomplete measured causes: {p}" for p in measured.problems)
        if measured.no_decoy_left:
            notes.append("No measured decoy remains: decoy rejection is unavailable in this world.")
    parts: dict[str, Any] = {
        "fault_effects": effects,
        "by_day": by_day,
        "interactions": interactions,
        "scoring_episodes": len(table.identities),
        "boot": boot,
    }
    return OracleReport(
        world_id=world.world_id,
        world_hash=world.world_hash,
        scoring_episodes=len(table.identities),
        rows=build.rows_held,
        plan=plan.counts() if isinstance(plan, OraclePlan) else plan,
        runs=[
            {
                "run_dir": str(r.run_dir),
                "used": r.used,
                "rows": r.rows,
                "as_run": r.as_run,
                "replays": r.replays,
                "unmodified": r.unmodified,
                "refused": r.refused,
            }
            for r in build.runs
        ],
        over_cap=table.over_cap(cap),
        excluded=sorted(table.excluded),
        marked_interactions=[list(p) for p in sorted(marked)],
        marked_additive_pairs=[list(p) for p in sorted(additive)],
        drifts=sorted(drifting),
        boot=boot,
        seed=seed,
        draws=draws,
        fault_effects=effects,
        by_day=by_day,
        interactions=interactions,
        causes=causes,
        achievable=achievable,
        all_faults=all_faults,
        still_failing=None if unknown else still,
        policy_failures_on_healthy_robot=None if unknown else still,
        all_removed_known_failures=still,
        all_removed_missing=unknown,
        chance=chance,
        detectability=detectability,
        patterns=pattern_checks(world, parts),
        problems=problems,
        notes=notes,
        measured_causes={} if measured is None else measured.model_dump(mode="json"),
    ), episode_rows


# ---- files ---------------------------------------------------------------------------------------


def report_failed(report: OracleReport) -> bool:
    """A refused row or incomplete measured result makes the CLI unsuccessful.

    Disabling bootstrap or draws intentionally does not itself fail the command.
    Pattern uncertainty is a measured verdict, not a missing replay.
    """
    return report_status(report) != 0


def report_status(report: OracleReport) -> int:
    """0 complete, 1 refused/invalid, 2 incomplete replay outcomes.

    Refusals take precedence; missing replay data can be completed by fresh.
    """
    if (
        any(r.get("refused") for r in report.runs)
        or any(not p.startswith("incomplete ") for p in report.problems)
        or any(c.skipped_error for c in report.chance)
        or (
            report.draws > 0
            and any(
                c.budget.startswith("k = ") and (c.exact is None or c.exact.missing)
                for c in report.chance
            )
        )
    ):
        return 1
    incomplete = bool(
        report.problems
        or report.all_removed_missing
        or any(r.get("refused") for r in report.runs)
        or any(
            e.missing
            for e in [
                *report.fault_effects,
                *report.by_day,
                *report.interactions,
                *report.causes,
                *report.achievable,
                report.all_faults,
            ]
        )
        or (
            report.draws > 0
            and any(
                c.missing or (c.budget.startswith("k = ") and (c.exact is None or c.exact.missing))
                for c in report.chance
            )
        )
    )
    return 2 if incomplete else 0


class ReportRefused(RuntimeError):
    """A report would replace a complete record with less measured information."""


def parquet_rows(report: OracleReport) -> list[dict[str, Any]]:
    """The rows of `truth/oracle.parquet` (contracts section 11): every valued gain."""
    rows: list[dict[str, Any]] = []
    for e in [*report.fault_effects, *report.interactions, *report.achievable]:
        if e.net is None:
            continue
        rows.append(
            {
                "item": e.item,
                "item_kind": e.item_kind,
                "rescued": e.rescued,
                "broken": e.broken,
                "net": e.net,
            }
        )
    e = report.all_faults
    if e.net is not None:
        rows.append(
            {
                "item": "all faults",
                "item_kind": "fix_set",
                "rescued": e.rescued,
                "broken": e.broken,
                "net": e.net,
            }
        )
    return rows


def write_truth_table(run_dir: Path, report: OracleReport, episodes_per_robot_day: int) -> None:
    """`truth/oracle.parquet` into a run directory, through the run's writer."""
    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.log import RunWriter

    record = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    with RunWriter(run_dir, record, episodes_per_robot_day=episodes_per_robot_day) as writer:
        writer.write_table("oracle", parquet_rows(report))


def write_episodes(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("episode_id", pa.string()),
            ("robot_id", pa.string()),
            ("site_id", pa.string()),
            ("day", pa.int32()),
            ("faults", pa.list_(pa.string())),
            ("toggled", pa.list_(pa.string())),
            ("as_run_success", pa.bool_()),
            ("all_removed_success", pa.bool_()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), temporary, compression="zstd")
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_report(
    oracle_dir: Path, report: OracleReport, episodes: Sequence[dict[str, Any]]
) -> None:
    previous = oracle_dir / "oracle.json"
    if previous.is_file():
        old = report_json(previous)
        if old.world_hash != report.world_hash or old.world_id != report.world_id:
            raise ReportRefused("report directory belongs to another world")
        if report_status(old) == 0 and (
            report_status(report) != 0
            or report.rows < old.rows
            or report.draws < old.draws
            or report.boot < old.boot
        ):
            raise ReportRefused(
                "complete report preserved; use a new --oracle-dir for a less complete result"
            )
    write_text(oracle_dir / "oracle.json", report.model_dump_json(indent=1) + "\n")
    write_text(oracle_dir / "summary.md", summary_markdown(report))
    write_episodes(oracle_dir / "episodes.parquet", episodes)


# ---- Markdown ------------------------------------------------------------------------------------


def _pts(value: float | None) -> str:
    return "" if value is None else f"{value:+.2f}"


def _interval(e: Estimate) -> str:
    if e.low is None or e.high is None:
        return "unavailable"
    return f"{e.low:+.2f} to {e.high:+.2f}"


def _population_stat(population: ExactChance, key: str) -> str:
    value = getattr(population, key)
    if value is not None:
        return f"{value:+.3f}"
    if population.bounds and key in population.bounds:
        lo, hi = population.bounds[key]
        return f"{lo:+.3f} to {hi:+.3f}"
    return "unavailable"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join("" if c is None else str(c) for c in cells) + " |"


def summary_markdown(report: OracleReport) -> str:
    """The report as Markdown tables, for the dashboard's truth view and people."""
    n = report.scoring_episodes
    lines = [
        f"# Oracle of world {report.world_id}",
        "",
        f"Scoring window: {n} episodes; {report.rows} distinct rows (episode, fault list) "
        f"in the table. Points are 100 times the net over the {n} scoring episodes; intervals "
        + (
            f"are the 2.5th to 97.5th percentiles of {report.boot} bootstrap resamples of the "
            f"scoring episodes (seed {report.seed})."
            if report.boot > 0
            else "are unavailable (bootstrap disabled)."
        ),
        "",
    ]
    if report.world_hash:
        lines += [f"World hash `{report.world_hash}`.", ""]
    if report.plan:
        p = report.plan
        lines += [
            "## Replays",
            "",
            _row(["Planned", "Count"]),
            _row(["---", "---:"]),
            _row(["contract power set", p["contract"]]),
            _row(["confirmation (decoys and inert entries removed too)", p["confirmation"]]),
            _row(["proof (unmodified, against the as-run chains)", p["proof"]]),
            _row(["jobs", p["jobs"]]),
            _row(["per scoring episode (contract and confirmation)", p["per_scoring_episode"]]),
            _row(["episodes above the cap of four", len(p["over_cap"])]),
            "",
        ]
    lines += ["Runs read:", ""]
    for r in report.runs:
        state = (
            f"refused: {r['refused']}"
            if r["refused"]
            else (
                f"{r['rows']} rows ({r['as_run']} as run, {r['replays']} replays, "
                f"{r.get('unmodified', 0)} unmodified replays)"
            )
        )
        lines.append(f"- `{r['run_dir']}`: {state}")
    lines += ["", "## Fault effects (removing one fault wherever it is active)", ""]
    verification = report.replay_verification
    verification_lines = ["## Replay verification", ""]
    if not verification.get("coverage"):
        verification_lines += ["Claim-bearing windows were not executed twice (legacy run).", ""]
    else:
        for coverage in verification["coverage"]:
            verification_lines.append(
                f"{coverage['window']}: {coverage['attempts_executed_twice']} of "
                f"{coverage['attempts']} attempts executed twice."
            )
        for window in verification.get("windows", []):
            verification_lines.append(
                f"{window['run_dir']} / {window['window']}: "
                f"{window['compared_steps']} compared executed steps (second executions only); "
                f"{len(window['events'])} events "
                f"(ceiling {window['event_ceiling']} in {window['ceiling_unit']}); "
                f"{window['compared_calls']} compared policy request executions; "
                f"rate per million calls {window['rate_per_million_calls']}, exact 95% interval "
                f"{window['rate_interval_95_per_million_calls']}; rate per million executed steps "
                f"{window['rate_per_million_steps']}, exact 95% interval "
                f"{window['rate_interval_95_per_million_steps']}."
            )
            for event in window["events"]:
                verification_lines.append(
                    f"- {event['episode_id']} step {event['step']}: "
                    f"{event['odd_side']} odd; outcome differs: {event['outcome_differs']}."
                )
        verification_lines += [
            "",
            "Call totals count requests of second executions only, from per-job statistics; "
            "where absent, only executed steps can be counted.",
            "",
        ]
    if residual := verification.get("residual"):
        recorded_replays = residual.get("recorded_expected_affected_replays_upper", "unavailable")
        recorded_points = residual.get(
            "recorded_expected_wrong_outcome_points_upper_per_table", "unavailable"
        )
        verification_lines += [
            residual["text"],
            f"Recorded-call component: expected affected replays ≤ "
            f"{recorded_replays}; expected wrong-outcome points per table ≤ {recorded_points}.",
            "At the fixed upper rate bound: expected affected replays ≤ "
            f"{residual['expected_affected_replays_upper']}; expected wrong-outcome points "
            f"per table ≤ {residual['expected_wrong_outcome_points_upper_per_table']}; "
            f"95% upper bound ≤ {residual['wrong_outcome_points_95_upper_per_table']} points.",
            "",
        ]
    lines[-3:-3] = verification_lines
    roles = {d.item: d for d in report.detectability}
    lines += [
        _row(
            [
                "Fault",
                "Role",
                "Episodes",
                "Rescued",
                "Broken",
                "Net",
                "Points",
                "Interval",
                "Detectability",
            ]
        ),
        _row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---", "---"]),
    ]
    for row in report.detectability:  # a fault inside a cause of several faults
        for fault_id in row.item.split("+") if "+" in row.item else ():
            roles.setdefault(fault_id, row)
    for e in report.fault_effects:
        d = roles.get(e.item)
        role = "" if d is None else d.role
        verdict = "" if d is None else d.verdict if d.item == e.item else f"in {d.item}"
        if e.missing:
            lines.append(_row([e.item, role, e.episodes, "", "", "", "", e.missing, verdict]))
        else:
            lines.append(
                _row(
                    [
                        e.item,
                        role,
                        e.episodes,
                        e.rescued,
                        e.broken,
                        e.net,
                        _pts(e.points),
                        _interval(e),
                        verdict,
                    ]
                )
            )
    if report.by_day:
        lines += ["", "By day (drifting faults):", ""]
        lines += [_row(["Fault", "Day", "Episodes", "Net", "Points", "Interval"])]
        lines += [_row(["---", "---:", "---:", "---:", "---:", "---"])]
        for e in report.by_day:
            lines.append(
                _row(
                    [
                        e.detail.get("fault_id"),
                        e.detail.get("day"),
                        e.episodes,
                        e.net,
                        _pts(e.points),
                        _interval(e),
                    ]
                )
            )
    if report.interactions:
        lines += ["", "## Interactions (joint removal minus the two single removals)", ""]
        lines += [
            _row(["Pair", "Marked", "Episodes", "Joint net", "Alone", "Interaction", "Interval"])
        ]
        lines += [_row(["---", "---", "---:", "---:", "---", "---:", "---"])]
        for e in report.interactions:
            alone = e.detail.get("alone_net")
            lines.append(
                _row(
                    [
                        e.item,
                        "yes" if e.detail.get("marked") else "",
                        e.episodes,
                        e.detail.get("joint_net"),
                        "" if alone is None else f"{alone[0]}, {alone[1]}",
                        e.missing or e.net,
                        _interval(e),
                    ]
                )
            )
    lines += ["", "## Achievable gain (the oracle's best set of resolved fixes)", ""]
    lines += [_row(["Budget", "Faults", "Rescued", "Broken", "Net", "Points", "Interval", "Hours"])]
    lines += [_row(["---", "---", "---:", "---:", "---:", "---:", "---", "---:"])]
    for e in report.achievable:
        lines.append(
            _row(
                [
                    _label_of(e),
                    ", ".join(e.detail.get("faults", [])),
                    e.rescued,
                    e.broken,
                    e.missing or e.net,
                    _pts(e.points),
                    _interval(e),
                    e.detail.get("hours", ""),
                ]
            )
        )
    a = report.all_faults
    lines += [
        "",
        "## Every fault removed (claim 5)",
        "",
        (
            "Incomplete gain: " + a.missing
            if a.missing
            else f"Removing every fault, decoys and inert entries included: {a.rescued} rescued, "
            f"{a.broken} broken, net {a.net} ({_pts(a.points)} points, interval {_interval(a)}). "
        ),
        (
            f"Incomplete policy-failure count: {report.all_removed_known_failures} known failures "
            f"and {report.all_removed_missing} missing healthy outcomes among {n} scoring episodes "
            f"(count bounded by {report.all_removed_known_failures} to "
            f"{(report.all_removed_known_failures or 0) + report.all_removed_missing})."
            if report.all_removed_missing
            else f"{report.still_failing} of {n} scoring episodes still fail with every fault "
            "removed: "
            "failures of the policy itself on a healthy robot, the measure behind "
            '"needs new training data".'
        ),
        "",
        "## Random fixes (chance level)",
        "",
        _row(
            [
                "Budget",
                "Draws valued",
                "Missing (fresh, error)",
                "Mean points",
                "5th to 95th",
                "Share of achievable, 5th to 95th",
            ]
        ),
        _row(["---", "---:", "---", "---:", "---", "---"]),
    ]
    for c in report.chance:
        share = "" if c.p5_share is None else f"{c.p5_share:.2f} to {c.p95_share:.2f}"
        span = "" if c.p5_points is None else f"{c.p5_points:+.2f} to {c.p95_points:+.2f}"
        lines.append(
            _row(
                [
                    c.budget,
                    f"{c.valued} of {c.draws}",
                    f"{c.skipped_fresh_replay}, {c.skipped_error}",
                    c.missing or _pts(c.mean_points),
                    span,
                    share,
                ]
            )
        )
        if c.bounds:
            lines.append(
                _row(
                    [
                        c.budget + " bounds",
                        "all draws",
                        "",
                        f"{c.bounds['mean_points'][0]:+.2f} to {c.bounds['mean_points'][1]:+.2f}",
                        f"5th: {c.bounds['p5_points'][0]:+.2f} to {c.bounds['p5_points'][1]:+.2f}; "
                        f"95th: {c.bounds['p95_points'][0]:+.2f} to "
                        f"{c.bounds['p95_points'][1]:+.2f}",
                        (
                            f"5th: {c.bounds['p5_share'][0]:.2f} to {c.bounds['p5_share'][1]:.2f}; "
                            f"95th: {c.bounds['p95_share'][0]:.2f} to "
                            f"{c.bounds['p95_share'][1]:.2f}"
                            if "p5_share" in c.bounds
                            else ""
                        ),
                    ]
                )
            )
    if any(c.bounds for c in report.chance):
        lines += [
            "",
            "Bounds include every draw: each missing draw may break any baseline "
            "success or rescue any baseline failure. These are worst-case bounds on "
            "the mean and each percentile, not a chance interval from selected draws.",
        ]
    if any(c.exact is not None for c in report.chance):
        lines += [
            "",
            "Exact population beside the seeded draws (the floor where available):",
            "",
            _row(["Budget", "Population", "Sets / permutations", "Mean points", "5th to 95th"]),
            _row(["---", "---", "---:", "---:", "---"]),
        ]
        for c in report.chance:
            x = c.exact
            if x is None:
                continue
            lines.append(
                _row(
                    [
                        c.budget,
                        x.method,
                        x.sets,
                        x.missing or _pts(x.mean_points),
                        f"5th: {_population_stat(x, 'p5_points')}; "
                        f"95th: {_population_stat(x, 'p95_points')}",
                    ]
                )
            )
            if x.bounds:
                lines.append(
                    _row(
                        [
                            c.budget + " bounds",
                            f"{x.unvalued_sets} sets unvalued ({x.uncomputed_sets} uncomputed)",
                            x.sets,
                            _population_stat(x, "mean_points"),
                            f"5th: {_population_stat(x, 'p5_points')}; "
                            f"95th: {_population_stat(x, 'p95_points')}",
                        ]
                    )
                )
            if x.counting_limited:
                lines += ["", x.counting_limited + "; unvisited sets retain worst/best bounds."]
    for c in report.chance:
        if c.p95_points_interval is not None:
            low, high = c.p95_points_interval
            lines += [
                "",
                f"{c.budget}: population 95th percentile {low:+.3f} to {high:+.3f} "
                f"points; distribution-free order-statistic coverage "
                f"{c.p95_interval_coverage:.4%} (ranks {c.p95_interval_order_stats}).",
            ]
    if report.patterns:
        lines += ["", "## Patterns", ""]
        lines += [
            _row(["Pattern", "Faults", "Verdict", "Measured"]),
            _row(["---", "---", "---", "---"]),
        ]
        for pattern in report.patterns:
            lines.append(
                _row([pattern.pattern, ", ".join(pattern.items), pattern.verdict, pattern.text])
            )
    if report.detectability:
        lines += ["", "## Detectability: the generator's list against the oracle", ""]
        lines += [
            _row(
                [
                    "Item",
                    "Role",
                    "Detectable",
                    "Qualified by",
                    "Expected (scoring)",
                    "Oracle",
                    "Verdict",
                ]
            )
        ]
        lines += [_row(["---", "---", "---", "---", "---:", "---:", "---"])]
        for d in report.detectability:
            lines.append(
                _row(
                    [
                        d.item,
                        d.role,
                        "yes" if d.detectable else "no",
                        ", ".join(d.qualified_by),
                        _pts(d.expected_points),
                        _pts(d.oracle_points),
                        d.verdict,
                    ]
                )
            )
    if report.measured_causes:
        m = report.measured_causes
        lines += [
            "",
            "## Measured and curve-based causes",
            "",
            _row(["List", "Causes", "Count"]),
            _row(["---", "---", "---:"]),
            _row(["Measured", ", ".join(str(c["unit"]) for c in m["causes"]), m["cause_count"]]),
            _row(["Curves", ", ".join(m["curve_causes"]), len(m["curve_causes"])]),
            "",
            "Measured classes use realised removal gains in either window, with the "
            "identifiable-target and 20-observed-carrier conditions. Pending rows "
            "make the measured list provisional; data faults use misreported outcomes.",
            "",
            _row(
                [
                    "Item",
                    "Curve class",
                    "Measured class",
                    "Scoring points",
                    "Observed points",
                    "Qualified by",
                ]
            ),
            _row(["---", "---", "---", "---:", "---:", "---"]),
        ]
        for item in [*m["faults"], *m["pairs"]]:
            lines.append(
                _row(
                    [
                        item["item"],
                        item["curve_class"],
                        item["measured_class"],
                        _pts(item["windows"]["scoring"]["points"]),
                        _pts(item["windows"]["observed"]["points"]),
                        ", ".join(item["qualified_by"]),
                    ]
                )
            )
    if report.over_cap:
        lines += ["", f"Episodes above the cap: {', '.join(report.over_cap)}."]
    if report.problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in report.problems]
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {p}" for p in report.notes]
    return "\n".join(lines) + "\n"


def report_json(path: Path) -> OracleReport:
    return OracleReport.model_validate(json.loads(path.read_text()))
