"""Claim 3: the fix sets of the agent and the baselines at every scoring budget, against the
achievable gain; and the headline.
"""

from __future__ import annotations

from collections.abc import Sequence

from fleetlab.contracts.actions import SCORING_BUDGETS, Action, Budget, cut, fits_budget, total_cost
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.oracle import Gain, MissingReplays, ReplayNeed, achievable_gain
from fleetlab.oracle import valuation as shared
from fleetlab.scoring.comparisons import (
    EXPERIMENTAL_BASELINE,
    MODEL_BASELINES,
    REQUIRED_BASELINES,
    chance_net,
    chance_share,
    exact_problem,
    required_baselines,
    required_values,
)
from fleetlab.scoring.models import (
    HEADLINE_K,
    Achievable,
    BudgetScore,
    DecisionScore,
    FixSetValue,
    Headline,
    HeadlineRow,
    Metric,
    RandomBaseline,
    ScoringInputs,
)
from fleetlab.scoring.valuation import (
    DrawResult,
    best_removal,
    cheapest_fix,
    complete_actions,
    missing_text,
    points,
    removal_options,
    value_or_missing,
)

NeedsLog = list[tuple[str, list[ReplayNeed]]]

RANDOM = "random fixes"
NO_DATA = "no-data ranker"
AGENT = "agent"


def budget_label(budget: Budget) -> str:
    """ "k = 3" for a count budget, "4 h" for a cost budget."""
    if budget.kind == "count":
        return f"k = {int(budget.amount)}"
    return f"{budget.amount:g} h"


def fault_specs(inputs: ScoringInputs) -> dict[str, FaultSpec]:
    """Every fault of the scoring episodes by id, as the first episode carrying it lists it."""
    out: dict[str, FaultSpec] = {}
    for identity in inputs.oracle.identities:
        for fault in identity.faults:
            out.setdefault(fault.fault_id, fault)
    return out


def achievable_at(
    budget: Budget,
    inputs: ScoringInputs,
    options_missing: list[ReplayNeed],
    options: dict[frozenset[str], dict[frozenset[str], Gain]],
    fix_costs: dict[str, tuple[Action, float]],
) -> Achievable:
    """The oracle's best at a budget: `achievable_gain(k)` for a count budget; for a cost
    budget the best set of resolved fixes whose cheapest actions fit the hours."""
    episodes = len(inputs.oracle.identities)
    if episodes == 0:
        return Achievable(missing="the oracle holds no scoring episode")
    if options_missing:
        return Achievable(missing=missing_text(options_missing))
    if budget.kind == "count":
        try:
            gain = achievable_gain(int(budget.amount), inputs.oracle)
        except MissingReplays as exc:
            return Achievable(missing=missing_text(exc.needs))
        return Achievable(gain=gain, points=points(gain.net, episodes))
    costs = {fault_id: hours for fault_id, (_, hours) in fix_costs.items()}
    gain, chosen = best_removal(options, costs, budget.amount)
    return Achievable(
        gain=gain,
        points=points(gain.net, episodes),
        faults=list(chosen),
        fixes=[fix_costs[f][0] for f in chosen],
        hours=sum(fix_costs[f][1] for f in chosen),
    )


def share_of(net: int | None, achievable: Achievable, why_net: str | None) -> Metric:
    if net is None:
        return Metric.absent(why_net or "the fix set is not valued")
    if achievable.gain is None:
        return Metric.absent(f"the achievable gain is missing: {achievable.missing}")
    if achievable.gain.net == 0:
        return Metric.absent("the achievable gain at this budget is zero")
    return Metric.of(net / achievable.gain.net)


def fix_set_value(
    ranked: Sequence[Action] | None,
    budget: Budget,
    inputs: ScoringInputs,
    achievable: Achievable,
    why_none: str,
    label: str,
    needs: NeedsLog,
) -> FixSetValue:
    """A ranked list cut at a budget and valued, with its share of the achievable gain."""
    if ranked is None:
        return FixSetValue(missing=why_none, share=Metric.absent(why_none))
    try:
        fix_set = cut(ranked, budget, inputs.costs)
    except (ValueError, KeyError) as exc:
        why = f"the list cannot be cut: {exc}"
        return FixSetValue(missing=why, share=Metric.absent(why))
    hours = total_cost(fix_set.actions, inputs.costs).hours
    episodes = len(inputs.oracle.identities)
    if episodes == 0:
        why = "the oracle holds no scoring episode"
        return FixSetValue(
            actions=fix_set.actions, hours=hours, missing=why, share=Metric.absent(why)
        )
    valued = value_or_missing(fix_set, inputs.oracle, inputs.truth, inputs.current)
    if valued.gain is None:
        if valued.needs:
            needs.append((label, valued.needs))
        return FixSetValue(
            actions=fix_set.actions,
            hours=hours,
            missing=valued.missing,
            share=Metric.absent(valued.missing or "not valued"),
        )
    return FixSetValue(
        actions=fix_set.actions,
        hours=hours,
        gain=valued.gain,
        points=points(valued.gain.net, episodes),
        share=share_of(valued.gain.net, achievable, None),
    )


def random_draws(inputs: ScoringInputs, budgets: Sequence[Budget]) -> list[list[DrawResult]]:
    """Per budget, the result of each random draw (`fleetlab.oracle.valuation.random_draws`).

    One permutation of the catalogue per draw, from numpy's generator seeded
    with `inputs.seed`: its first k actions are k distinct actions drawn
    uniformly (count budgets), and `cut` of the whole permutation is the shuffle
    cut at the hours (cost budgets).
    """
    return shared.random_draws(
        inputs.oracle,
        inputs.truth,
        inputs.current,
        inputs.catalogue,
        budgets,
        seed=inputs.seed,
        draws=inputs.draws,
        costs=inputs.costs,
    )


def score_decisions(inputs: ScoringInputs) -> tuple[DecisionScore, NeedsLog, list[str]]:
    """Claim 3 at every budget of `SCORING_BUDGETS`; the replays it waits for; notes."""
    notes: list[str] = []
    needs: NeedsLog = []
    if inputs.replay_missing:
        reason = inputs.replay_missing
        return (
            DecisionScore(
                budgets=[
                    BudgetScore(
                        budget=b,
                        label=budget_label(b),
                        world_id=inputs.fault_table.world_id,
                        condition=inputs.condition,
                        valuation_missing=reason,
                        achievable=Achievable(missing=reason),
                        agent=FixSetValue(missing=reason, share=Metric.absent(reason)),
                        no_data=FixSetValue(missing=reason, share=Metric.absent(reason)),
                        random=RandomBaseline(missing=reason),
                        baseline_sources=inputs.baseline_sources,
                        baselines={
                            name: FixSetValue(missing=reason, share=Metric.absent(reason))
                            for name in set(inputs.baseline_sources)
                            | set(required_baselines(inputs.condition))
                        },
                    )
                    for b in SCORING_BUDGETS
                ]
            ),
            [],
            [reason],
        )
    options, options_missing = removal_options(inputs.oracle)
    if options_missing:
        needs.append(("achievable gain", options_missing))
    specs = fault_specs(inputs)
    fix_costs: dict[str, tuple[Action, float]] = {}
    for fault_id, spec in sorted(specs.items()):
        if fault_id in inputs.oracle.excluded:
            continue
        fix = cheapest_fix(spec, inputs.truth, inputs.costs)
        if fix is None:
            notes.append(
                f"fault {fault_id} ({spec.kind}) has no fixing action that applies to its target: "
                "left out of the achievable gain under cost budgets"
            )
        else:
            fix_costs[fault_id] = fix
    agent_ranked = [] if inputs.fix_set is None else [r.action for r in inputs.fix_set.actions]
    if inputs.decision_actions is not None:
        agent_ranked = inputs.decision_actions
    left_out = len(inputs.catalogue) - len(complete_actions(inputs.catalogue))
    if left_out:
        notes.append(
            f"{left_out} catalogue entries without a value are left out of the random draws"
        )
    chance_records = {c["budget"]: c for c in inputs.oracle_chance}

    budgets: list[BudgetScore] = []
    for budget in SCORING_BUDGETS:
        label = budget_label(budget)
        achievable = achievable_at(budget, inputs, options_missing, options, fix_costs)
        agent = fix_set_value(
            agent_ranked,
            budget,
            inputs,
            achievable,
            "no ranked fix set submitted",
            f"claim 3: agent at {label}",
            needs,
        )
        no_data = fix_set_value(
            inputs.no_data_ranked,
            budget,
            inputs,
            achievable,
            "no no-data ranking supplied",
            f"claim 3: no-data ranker at {label}",
            needs,
        )
        chance = chance_records.get(label, {})
        random = RandomBaseline(
            draws=chance.get("draws", 0),
            valued=chance.get("valued", 0),
            skipped_fresh_replay=chance.get("skipped_fresh_replay", 0),
            skipped_error=chance.get("skipped_error", 0),
            missing=chance.get("missing") if chance else "oracle chance report not supplied",
        )
        for key in (
            "mean_points",
            "p5_points",
            "p95_points",
            "above_zero",
            "mean_share",
            "p5_share",
            "p95_share",
        ):
            setattr(
                random,
                key,
                Metric.of(chance.get(key), chance.get("missing") or "oracle draw statistic absent"),
            )
        exact = chance.get("exact")
        accounting_problem = exact_problem(exact)
        chance_missing = (
            (accounting_problem or "oracle exact chance statistic unavailable")
            + "; produce it with: "
            + inputs.chance_command
            if accounting_problem
            or (
                exact is not None
                and exact.get("p95_points") is None
                and not (exact.get("bounds") or {}).get("p95_points")
            )
            else None
        )
        baselines = {
            name: fix_set_value(
                ranked,
                budget,
                inputs,
                achievable,
                "no ranking supplied",
                f"claim 3: {name} at {label}",
                needs,
            )
            for name, ranked in inputs.baseline_rankings.items()
        }
        for name, sets in inputs.baseline_fix_sets.items():
            submitted = sets.get(label)
            if submitted is not None:
                if submitted.budget != budget or not fits_budget(submitted, inputs.costs):
                    raise ValueError(f"baseline {name} exceeds or mislabels {label}")
                if cut(submitted.actions, budget, inputs.costs).actions != submitted.actions:
                    raise ValueError(
                        f"baseline {name} at {label} contains unsupported budget actions"
                    )
            baselines[name] = fix_set_value(
                None if submitted is None else submitted.actions,
                budget,
                inputs,
                achievable,
                f"no fix set submitted at {label}",
                f"claim 3: {name} at {label}",
                needs,
            )
        for name, source in inputs.baseline_sources.items():
            if source.get("lost_call"):
                baselines[name] = FixSetValue(
                    missing="both model call starts lost",
                    share=Metric.absent("both model call starts lost"),
                )
        for name in required_baselines(inputs.condition):
            if name not in baselines and name + " repeat 1" not in baselines:
                baselines[name] = FixSetValue(
                    missing="required baseline absent",
                    share=Metric.absent("required baseline absent"),
                )
            if name in MODEL_BASELINES and name + " repeat 2" not in baselines:
                baselines[name + " repeat 2"] = FixSetValue(
                    missing="second baseline repeat absent",
                    share=Metric.absent("second baseline repeat absent"),
                )
        budgets.append(
            BudgetScore(
                budget=budget,
                world_id=inputs.fault_table.world_id,
                condition=inputs.condition,
                analysis_binding_missing=inputs.analysis_binding_missing,
                label=label,
                achievable=achievable,
                agent=agent,
                no_data=no_data,
                random=random,
                baselines=baselines,
                baseline_sources=inputs.baseline_sources,
                exact_chance=exact,
                chance_missing=chance_missing,
                chance_command=inputs.chance_command,
                chance_bounds=chance.get("bounds") or {},
                sampling_percentile_range={
                    "p95_points": chance.get("p95_points_interval"),
                    "coverage": chance.get("p95_interval_coverage"),
                    "order_stats": chance.get("p95_interval_order_stats"),
                }
                if chance.get("p95_points_interval") is not None
                else {},
            )
        )
    return DecisionScore(budgets=budgets), needs, notes


def headline(decisions: DecisionScore, k: int = HEADLINE_K) -> Headline:
    """The share of the achievable gain at k = 3 for the agent and every baseline, and the
    ratio to the best baseline when that baseline beats chance."""
    at = next(
        (s for s in decisions.budgets if s.budget.kind == "count" and s.budget.amount == k), None
    )
    if at is None:
        return Headline(k=k, verdict=f"no budget of k = {k} was scored")
    if at.valuation_missing:
        return Headline(
            k=k, verdict=at.valuation_missing, chance_p95_share=Metric.absent(at.valuation_missing)
        )

    def pts(value: FixSetValue) -> Metric:
        return (
            Metric.of(value.points)
            if value.points is not None
            else Metric.absent(value.missing or "not valued")
        )

    random = at.random
    rows = [
        HeadlineRow(method=AGENT, share=at.agent.share, points=pts(at.agent)),
        HeadlineRow(
            method=f"{RANDOM} (mean of {random.valued} draws)",
            share=random.mean_share,
            points=random.mean_points,
        ),
        HeadlineRow(method=NO_DATA, share=at.no_data.share, points=pts(at.no_data)),
    ]
    if NO_DATA in at.baselines:
        rows = [row for row in rows if row.method != NO_DATA]
    baselines = [(NO_DATA, at.no_data.share)]
    for name, value in at.baselines.items():
        rows.append(HeadlineRow(method=name, share=value.share, points=pts(value)))
        baselines.append((name, value.share))
    chance = chance_share(at)
    out = Headline(k=k, rows=rows, chance_p95_share=chance)
    required, absent = required_values(at)
    logs = {name: value for name, value in required.items() if name in REQUIRED_BASELINES}
    if len(logs) == len(REQUIRED_BASELINES):
        name, value = max(logs.items(), key=lambda item: float(item[1].points or 0))
        out.opponents["best logs-only"] = {
            "name": name,
            "points": value.points,
            "share": value.share.value,
        }
    if at.condition == "with experiments":
        experimental = required.get(EXPERIMENTAL_BASELINE)
        out.opponents[EXPERIMENTAL_BASELINE] = {
            "points": None if experimental is None else experimental.points,
            "share": None if experimental is None else experimental.share.value,
            "missing": absent if experimental is None else None,
        }
    if absent:
        out.verdict = "no ratio: required baselines missing: " + ", ".join(absent)
        return out
    if any(v.share.value is None for v in required.values()):
        out.verdict = "no ratio: required baseline shares unavailable"
        return out
    best_name, best_value = max(required.items(), key=lambda t: float(t[1].share.value or 0))
    best_share = float(best_value.share.value or 0)
    out.best_baseline = best_name
    if chance.value is None:
        out.verdict = f"no ratio: the chance level is missing ({chance.missing})"
        return out
    tail = chance_net(at)
    if best_value.gain is None or tail is None:
        out.verdict = "no ratio: baseline net or exact chance net unavailable"
        return out
    if best_value.gain.net <= tail:
        out.verdict = "the baselines recover no more than chance"
        return out
    agent_share = at.agent.share.value
    if agent_share is None:
        out.verdict = f"no ratio: the agent's share is missing ({at.agent.share.missing})"
        return out
    if best_share <= 0:
        out.verdict = "no ratio: the best baseline's share is not positive"
        return out
    out.ratio = agent_share / best_share
    out.verdict = (
        f"the agent recovers {out.ratio:.2f} times the share of the best baseline ({best_name})"
    )
    return out
