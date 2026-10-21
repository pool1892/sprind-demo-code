"""The detectability list of a world, computed at generation.

A **cause** is first judged per effective fault. A marked interaction pair is also
judged jointly only when neither member qualifies alone. Overlapping pairs stay
separate; additive pairs receive no joint cause treatment (section 10, entry 8).
A cause is **detectable** when removing it alone gives
back at least `DETECT_POINTS` points of fleet success over the observed or the scoring
window (from the calibrated curves, `fleetlab.worlds.estimate`; the window that
qualified it is recorded), its target is identifiable in the visible data (robots,
sites, configuration keys and conditions all are; no fault here sits on a trait), and at
least `DETECT_EPISODES` observed episodes carry it. A data fault qualifies on the logged
outcome. Decoys and the inert entry of a confounder pair are no causes; their neutrality
and remaining estimated effect are reported separately for each measured suite.

On top of the rule, a cause is **confirmed** when every calibrated point its estimate
rests on is an effect beyond noise (the curves' role "effect"; an interaction's joint
point likewise). The gate's sizing rule (at least six detectable causes of two points
each, observed window) uses `observed_counted`; `counted` retains either-window
detectability. A detectable
cause on a point the curves could not tell from zero is reported, not counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.world import Patterns, Windows
from fleetlab.worlds import base, estimate
from fleetlab.worlds.truth import DETECT_EPISODES, DETECT_POINTS

WINDOWS = ("observed", "experiment", "scoring")


@dataclass(frozen=True)
class Cause:
    unit: str  # "f03", or "f01+f02" for an interaction
    fault_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    roles: tuple[str, ...]
    scopes: tuple[str, ...]
    classes: tuple[str, ...]
    data: bool
    effect: dict[str, float]  # points of the window's fleet success, positive is harm
    carriers: dict[str, int]
    observed_suites: dict[str, int]
    qualified_by: tuple[str, ...]
    identifiable: bool
    detectable: bool
    confirmed: bool
    transferred: bool  # rests on libero_spatial points applied to libero_object episodes
    assumed: bool  # rests on an assumed joint point
    sources: tuple[str, ...]
    confirmed_by: tuple[str, ...] = ()

    @property
    def counts(self) -> bool:
        """Detectable, on effect points measured for the suites it acts on."""
        return self.detectable and self.confirmed and not self.transferred and not self.assumed

    @property
    def counts_provisionally(self) -> bool:
        """Detectable on effect points, some of them transferred or assumed."""
        return self.detectable and self.confirmed

    @property
    def counts_observed(self) -> bool:
        return self.counts and "observed" in self.qualified_by and "observed" in self.confirmed_by

    @property
    def small(self) -> bool:
        return not self.detectable

    def as_json(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "fault_ids": list(self.fault_ids),
            "kinds": list(self.kinds),
            "roles": list(self.roles),
            "scopes": list(self.scopes),
            "classes": list(self.classes),
            "data_fault": self.data,
            "expected_effect_points": {k: round(v, 3) for k, v in self.effect.items()},
            "episodes": self.carriers,
            "observed_by_suite": self.observed_suites,
            "qualified_by": list(self.qualified_by),
            "identifiable": self.identifiable,
            "detectable": self.detectable,
            "confirmed": self.confirmed,
            "counts_for_gate": self.counts_observed,
            "confirmed_by": list(self.confirmed_by),
            "counts_provisionally": self.counts_provisionally,
            "transferred_from_spatial": self.transferred,
            "assumed_joint": self.assumed,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class DetectTable:
    units: list[Cause]
    neutral: list[dict[str, Any]]  # decoys and inert entries
    pairs: list[dict[str, Any]]  # each interaction pair: singles, joint, super-additive
    window_success: dict[str, dict[str, float]]
    rule: str = field(default="")

    @property
    def pairs_superadditive(self) -> bool:
        return all(p["superadditive"] for p in self.pairs if p["kind"] == "interaction")

    def counted(self, provisional: bool = False) -> list[Cause]:
        """Confirmed section-5 causes in either window (optionally provisional)."""
        return [u for u in self.units if (u.counts_provisionally if provisional else u.counts)]

    def observed_counted(self, provisional: bool = False) -> list[Cause]:
        """Section 1 sizing, distinct from section 5 detectability in either window."""
        return [
            u
            for u in self.counted(provisional)
            if "observed" in u.qualified_by and "observed" in u.confirmed_by
        ]

    def as_json(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "window_success": self.window_success,
            "detectable_counted": len(self.counted()),
            "observed_counted": len(self.observed_counted()),
            "detectable_by_rule": sum(1 for u in self.units if u.detectable),
            "causes": [u.as_json() for u in self.units],
            "neutral": self.neutral,
            "interaction_pairs": self.pairs,
        }


RULE = (
    f"each fault is judged alone; a marked interaction pair is judged jointly only if neither "
    "member qualifies alone (overlapping pairs are not merged); a cause is detectable "
    "when removing "
    f"it alone gives back at least {DETECT_POINTS:g} points of fleet success over the observed "
    f"or the scoring window by the calibrated curves (logged outcome for a data fault), its "
    f"target is identifiable in the visible data, and at least {DETECT_EPISODES} observed "
    "episodes carry it (experiment.md section 5); it counts for gate 3's sizing rule when "
    "the observed window qualifies and its estimate rests on an effect beyond noise, "
    "measured on the "
    "suite it acts on (no libero_spatial point applied to libero_object, no assumed joint)"
)


def episodes_by_window(
    schedule: Sequence[FaultSpec], slots: Sequence[base.Slot], windows: Windows
) -> dict[str, list[estimate.Episode]]:
    out: dict[str, list[estimate.Episode]] = {w: [] for w in WINDOWS}
    ranges = {
        "observed": windows.observed,
        "experiment": windows.experiment,
        "scoring": windows.scoring,
    }
    for s in slots:
        for name, days in ranges.items():
            if s.day in days:
                out[name].append(
                    estimate.Episode(
                        s.suite,
                        tuple(base.in_force(schedule, s.robot_id, s.site_id, s.day)),
                        s.task.task_index,
                    )
                )
    return out


def detectability(
    schedule: Sequence[FaultSpec],
    patterns: Patterns,
    small: frozenset[str] | set[str],
    slots: Sequence[base.Slot],
    windows: Windows,
    ctx: estimate.Context,
    roles: dict[str, str] | None = None,
    by_window: dict[str, list[estimate.Episode]] | None = None,
    members_alone: bool = False,
) -> DetectTable:
    """The detectability list of a schedule over a world's as-run episodes."""
    from fleetlab.worlds.compose import fault_class

    eps = by_window if by_window is not None else episodes_by_window(schedule, slots, windows)
    by_id = {f.fault_id: f for f in schedule}
    neutral_ids = set(patterns.excluded_from_replay())

    def cause(ids: list[str]) -> Cause:
        faults = [by_id[i] for i in ids]
        data = all(estimate.is_data(f) for f in faults)
        effect = {w: estimate.removal_effect(ids, eps[w], ctx, logged=data) for w in WINDOWS}
        carriers = {w: estimate.carriers(eps[w], ids) for w in WINDOWS}
        observed_suites = estimate.suite_mix(eps["observed"], ids)
        qualified = tuple(w for w in ("observed", "scoring") if effect[w] >= DETECT_POINTS)
        detectable = bool(qualified) and carriers["observed"] >= DETECT_EPISODES
        confirmed, transferred, assumed, sources = _support(
            faults, ids, eps, qualified, ctx, data, members_alone
        )
        confirmed_by = tuple(
            w for w in qualified if _support(faults, ids, eps, [w], ctx, data, members_alone)[0]
        )
        return Cause(
            unit="+".join(ids),
            fault_ids=tuple(ids),
            kinds=tuple(f.kind for f in faults),
            roles=tuple((roles or {}).get(i, "small" if i in small else "plain") for i in ids),
            scopes=tuple(f.target.scope for f in faults),
            classes=tuple(fault_class(f.kind).value for f in faults),
            data=data,
            effect=effect,
            carriers=carriers,
            observed_suites=observed_suites,
            qualified_by=qualified,
            identifiable=True,
            detectable=detectable,
            confirmed=confirmed,
            transferred=transferred,
            assumed=assumed,
            sources=sources,
            confirmed_by=confirmed_by,
        )

    causes = [cause([f.fault_id]) for f in schedule if f.fault_id not in neutral_ids]
    singles = {c.fault_ids[0]: c for c in causes}
    for a, b in sorted({tuple(sorted(pair)) for pair in patterns.interactions}):
        if not singles[a].detectable and not singles[b].detectable:
            pair = cause([a, b])
            if pair.detectable:
                causes.append(pair)
    causes.sort(key=lambda c: c.fault_ids)
    neutral = []
    for fid in sorted(neutral_ids):
        f = by_id[fid]
        neutral.append(
            {
                "fault_id": fid,
                "kind": f.kind,
                "role": "decoy" if fid in patterns.decoys else "inert",
                "scope": f.target.scope,
                "episodes": {w: estimate.carriers(eps[w], [fid]) for w in WINDOWS},
                "expected_effect_points": {
                    w: round(estimate.removal_effect([fid], eps[w], ctx, logged=False), 3)
                    for w in WINDOWS
                },
                "neutrality_by_suite": {
                    suite: (
                        "inert by construction"
                        if f.kind == "calibration_update"
                        else "confirmed decoy"
                        if (
                            point := ctx.curves.points.get(
                                (f.kind, suite, estimate.member_key(f)[1])
                            )
                        )
                        is not None
                        and point.role == "decoy"
                        and point.n >= 1000
                        else "not confirmed flat"
                    )
                    for suite in estimate.suite_mix(eps["observed"], [fid])
                },
            }
        )
    pairs = []
    for kind, marked in (
        ("interaction", patterns.interactions),
        ("additive_pair", patterns.additive_pairs),
    ):
        for a, b in marked:
            fa, fb = by_id[a], by_id[b]
            # Judge the actual scoring mix. A weighted interval is a conservative
            # combination of suite intervals, not a new pooled bootstrap interval.
            joint_eps = [e for e in eps["scoring"] if {a, b} <= e.fault_ids]
            mix: dict[tuple[str, int | None], int] = {}
            for e in joint_eps:
                key = e.suite, e.task_index if ctx.task_aware else None
                mix[key] = mix.get(key, 0) + 1
            evidence: list[dict[str, Any]] = []
            weighted = [0.0, 0.0, 0.0]
            supported = bool(joint_eps)
            for (suite, task_index), count in sorted(mix.items()):
                joint, moved = ctx.calibration(suite, task_index).joint(
                    estimate.member_key(fa), estimate.member_key(fb), suite
                )
                valid = joint.paired and not moved
                supported &= valid
                loss = None
                if joint.single_successes is not None:
                    ns = dict(zip(joint.members, joint.single_successes, strict=True))
                    loss = {
                        a: round(
                            100 * (joint.healthy_successes - ns[estimate.member_key(fa)]) / joint.n,
                            3,
                        ),
                        b: round(
                            100 * (joint.healthy_successes - ns[estimate.member_key(fb)]) / joint.n,
                            3,
                        ),
                        "joint": round(
                            100 * (joint.healthy_successes - joint.successes) / joint.n, 3
                        ),
                    }
                if valid and joint.interaction is not None:
                    for i in range(3):
                        weighted[i] += 100 * joint.interaction[i] * count / len(joint_eps)
                evidence.append(
                    {
                        "suite": suite,
                        **({"task_index": task_index} if ctx.task_aware else {}),
                        "scoring_episodes": count,
                        "n": joint.n,
                        "paired": valid,
                        "loss_points": loss,
                        "interaction_points": None
                        if joint.interaction is None
                        else [round(100 * v, 3) for v in joint.interaction],
                        "interaction_role": joint.interaction_role,
                        "joint_role": joint.role,
                        "joint_run": joint.run,
                    }
                )
            pairs.append(
                {
                    "kind": kind,
                    "pair": [a, b],
                    "by_suite": evidence,
                    "paired": supported,
                    "suite": "scoring task mix",
                    "loss_points": {
                        key: round(
                            sum(
                                e["loss_points"][key] * e["scoring_episodes"] / len(joint_eps)
                                for e in evidence
                            ),
                            3,
                        )
                        for key in (a, b, "joint")
                    }
                    if supported
                    else {},
                    "joint_role": kind,
                    "interval_method": "scoring-suite-weighted calibration bounds",
                    "interaction_points": [round(v, 3) for v in weighted] if supported else None,
                    "superadditive": supported and weighted[2] < 0,
                    "additive": supported and weighted[1] <= 0 <= weighted[2],
                }
            )
    window_success = {
        w: {
            "episodes": len(eps[w]),
            "true": round(estimate.mean_success(eps[w], ctx), 4),
            "logged": round(estimate.mean_success(eps[w], ctx, logged=True), 4),
        }
        for w in WINDOWS
    }
    return DetectTable(causes, neutral, pairs, window_success, RULE)


def _support(
    faults: Sequence[FaultSpec],
    ids: Sequence[str],
    eps: dict[str, list[estimate.Episode]],
    qualified: Sequence[str],
    ctx: estimate.Context,
    data: bool = False,
    members_alone: bool = False,
) -> tuple[bool, bool, bool, tuple[str, ...]]:
    """Whether the cause's effect rests on effect points; transferred; assumed; sources.

    An episode is *backed* when every point its share of the effect rests on is an
    effect beyond noise (an interaction's joint point; else each removed fault's own
    point). A cause is confirmed when, in a window that qualified it, the effect of its
    backed episodes alone reaches `DETECT_POINTS`: a batch fault whose effect comes from
    `libero_spatial` episodes is confirmed even though its `libero_object` episodes rest
    on a point that showed no effect (and add none). Transferred and assumed are judged
    on the backed episodes, whose points carry the confirmation.
    """
    wanted = set(ids)
    transferred = assumed = False
    sources: set[str] = set()
    best = 0.0
    for w in qualified:
        backed_gain = 0.0
        for e in eps[w]:
            present = wanted & e.fault_ids
            if not present:
                continue
            here = {f.fault_id: f for f in e.faults}
            joined: set[str] = set()
            backed = True
            moved_here = assumed_here = False
            for a, b in ctx.pairs:
                if a in here and b in here and {a, b} & present and not {a, b} & joined:
                    joint, moved = ctx.calibration(e.suite, e.task_index).joint(
                        estimate.member_key(here[a]), estimate.member_key(here[b]), e.suite
                    )
                    # an assumed joint point is no calibration: `assumed` says so, and the
                    # cause then counts only provisionally
                    backed &= joint.role in ("effect", "assumed")
                    assumed_here |= joint.role == "assumed"
                    moved_here |= moved
                    joined |= {a, b}
                    sources.add(
                        f"joint {joint.members[0][1]} with {joint.members[1][1]} "
                        f"on {joint.suite} ({joint.role})"
                        + (f"; task_index={e.task_index}" if ctx.task_aware else "")
                    )
            for fid in present - joined:
                r = ctx.ratio(here[fid], e.suite, e.task_index)
                backed &= r.effect
                moved_here |= r.transferred
                sources.add(r.source)
            if backed:
                kept = [f for f in e.faults if f.fault_id not in wanted]
                gain = estimate.probability(
                    kept, e.suite, ctx, data, e.task_index
                ) - estimate.probability(e.faults, e.suite, ctx, data, e.task_index)
                backed_gain += gain
                transferred |= moved_here
                assumed |= assumed_here
        if eps[w]:
            best = max(best, 100.0 * backed_gain / len(eps[w]))
    return best >= DETECT_POINTS, transferred, assumed, tuple(sorted(sources))
