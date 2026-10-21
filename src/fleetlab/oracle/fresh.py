"""Fresh replays: the lookups the planned replays do not cover, run on demand.

Two kinds of valuation miss rows (contracts sections 8 and 9), and `value_fix_set`
raises `MissingReplays` for both: an action that **adds** a change (a `set` to a value
no fault had, a `change_condition` to a new value), and an episode **above the cap**
of four toggled faults. A fresh set collects what a list of fix sets needs (each fix set,
and for a ranked list each action alone and the cut at every scoring budget, as the
scorer values them), optionally what every entry of the world's filled catalogue needs
alone, and what the oracle's own items need (fault effects, the achievable gains and the
all-faults-removed row of episodes above the cap). Each missing lookup becomes a
"replay" job from the reset with `remove` and `add` (`fresh_jobs`), written to
`oracle/fresh/<name>/jobs.jsonl` with proof replays (every tenth episode of the set, at
least 20, unmodified) and run like the planned replays into `oracle/fresh/<name>/run`.
After it, the table holds those rows and the same call values the fix sets
(`values.json`).

A change the engine would refuse is left out and listed: a change of the precision
(fixed per run, the model is loaded once) or of the image views (the pinned convolution
database holds the fleet's images per call only).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import SCORING_BUDGETS, Action, Budget, FixSet, cut
from fleetlab.contracts.faults import TruthTable
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import (
    CurrentValues,
    MissingReplays,
    OracleTable,
    ReplayNeed,
    fault_list_hash,
    replays_needed,
    value_fix_set,
)
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.contracts.vocabulary import ConfigKey
from fleetlab.oracle.chance import budget_label
from fleetlab.oracle.gains import Gains
from fleetlab.oracle.mechanism import mechanism_additions, mechanism_table
from fleetlab.oracle.plan import proof_jobs
from fleetlab.oracle.valuation import random_fix_sets

REFUSED_KEYS = frozenset({ConfigKey.policy_precision, ConfigKey.policy_image_views})


@dataclass
class Needs:
    """Missing lookups by what waits for them, and what could not be resolved at all."""

    by_label: dict[str, list[ReplayNeed]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def add(self, label: str, needs: Sequence[ReplayNeed]) -> None:
        if needs:
            self.by_label.setdefault(label, []).extend(needs)

    def unique(self) -> list[ReplayNeed]:
        """Every missing lookup once, in the order first asked for."""
        seen: dict[tuple[str, str], ReplayNeed] = {}
        for needs in self.by_label.values():
            for need in needs:
                seen.setdefault((need.episode_id, fault_list_hash(need.faults)), need)
        return list(seen.values())


def labelled_fix_sets(name: str, data: Any) -> list[tuple[str, FixSet]]:
    """The fix sets a file stands for: a `FixSet`, a `RankedFixSet` (each action alone and
    the cut at every scoring budget, as the scorer values them) or a list of actions (one
    fix set of all of them)."""
    if isinstance(data, dict) and "budget" in data:
        return [(name, FixSet.model_validate(data))]
    if isinstance(data, dict) and "actions" in data:
        # A ranked list as the agents submit it (`RankedFixSet`), read leniently: the
        # scorer's ten-action minimum is the submission's rule, not the valuation's.
        actions = [Action.model_validate(r["action"]) for r in data["actions"]]
        out: list[tuple[str, FixSet]] = []
        for rank, action in enumerate(actions, start=1):
            if action.has_replay and action.is_complete:
                single = FixSet(actions=[action], budget=Budget(kind="count", amount=1))
                out.append((f"{name}: ranked action {rank}", single))
        for budget in SCORING_BUDGETS:
            try:
                out.append((f"{name}: at {budget_label(budget)}", cut(actions, budget)))
            except (ValueError, KeyError):
                continue
        return out
    if isinstance(data, list):
        actions = [Action.model_validate(a) for a in data]
        return [(name, FixSet(actions=actions, budget=Budget(kind="count", amount=len(actions))))]
    raise ValueError(f"{name}: neither a fix set, a ranked fix set nor a list of actions")


def read_fix_sets(paths: Sequence[Path]) -> list[tuple[str, FixSet]]:
    out: list[tuple[str, FixSet]] = []
    for path in paths:
        out += labelled_fix_sets(Path(path).name, json.loads(Path(path).read_text()))
    return out


def read_requests(paths: Sequence[Path]) -> list[ReplayNeed]:
    """Scorer request records, directly or in its needs_fresh_replay envelope.

    Extra scorer annotations are ignored; episode_id and the full effective faults
    list are required. In particular faults=[] requests a healthy replay in any window.
    """
    out: list[ReplayNeed] = []
    for path in paths:
        data = json.loads(path.read_text())
        records = data.get("needs_fresh_replay") if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a replay request list or needs_fresh_replay")
        from fleetlab.contracts.faults import FaultSpec

        for record in records:
            faults = tuple(FaultSpec.model_validate(f) for f in record["faults"])
            if "fault_ids" in record and set(record["fault_ids"]) != {f.fault_id for f in faults}:
                raise ValueError(f"{path}: fault_ids differ from faults")
            out.append(ReplayNeed(str(record["episode_id"]), faults))
    return out


def needs_of_fix_sets(
    fix_sets: Sequence[tuple[str, FixSet]],
    table: OracleTable,
    truth: TruthTable,
    current: CurrentValues | None,
    into: Needs | None = None,
) -> Needs:
    needs = into or Needs()
    table = mechanism_table(table)
    for label, fix_set in fix_sets:
        try:
            missing = replays_needed(fix_set, table, truth, current)
        except (ValueError, KeyError) as exc:
            needs.errors[label] = str(exc)
            continue
        needs.add(label, [n for n in missing if table.success(n.episode_id, n.faults) is None])
    return needs


def catalogue_fix_sets(entries: Sequence[Action]) -> list[tuple[str, FixSet]]:
    """Every complete catalogue entry with a replay, alone."""
    return [
        (
            f"catalogue: {a.verb.value} {a.target.scope} {a.target.thing}",
            FixSet(actions=[a], budget=Budget(kind="count", amount=1)),
        )
        for a in entries
        if a.is_complete and a.has_replay
    ]


def chance_fix_sets(
    entries: Sequence[Action], seed: int, draws: int = 200
) -> list[tuple[str, FixSet]]:
    """All seeded draws at every scoring budget, with stable labels for review."""
    if draws < 1:
        raise ValueError("chance replay planning needs at least one draw")
    return [
        (f"chance: {budget_label(budget)} draw {number}", fix_set)
        for number, sets in enumerate(
            random_fix_sets(entries, SCORING_BUDGETS, seed=seed, draws=draws), start=1
        )
        for budget, fix_set in zip(SCORING_BUDGETS, sets, strict=True)
    ]


def needs_of_oracle(table: OracleTable, into: Needs | None = None) -> Needs:
    """What the oracle's own items lack: fault effects, achievable sets, all removed."""
    needs = into or Needs()
    gains = Gains(table, boot=0)
    for fault_id in table.fault_ids():
        _, missing = gains.removal({fault_id})
        needs.add(f"oracle: fault_effect {fault_id}", missing)
    _, missing = gains.subset_changes(gains.candidates())
    needs.add("oracle: achievable", missing)
    every: list[ReplayNeed] = []
    for index, identity in enumerate(table.identities):
        if identity.faults:
            gains.change(index, frozenset(identity.fault_ids), every)
        elif table.success(identity.episode_id, ()) is None:
            every.append(ReplayNeed(identity.episode_id, ()))
    needs.add("oracle: all faults removed", every)
    return needs


def fresh_jobs(
    scoring: dict[str, EpisodeIdentity], needs: Sequence[ReplayNeed]
) -> tuple[list[Job], list[str]]:
    """Replay jobs for the needed fault lists, and the ones the engine would refuse."""
    jobs: dict[str, Job] = {}
    refused: list[str] = []
    for need in needs:
        identity = scoring.get(need.episode_id)
        if identity is None:
            refused.append(f"{need.episode_id}: not an as-run episode of this world")
            continue
        original = {f.fault_id: f for f in identity.faults}
        if any(f.fault_id in original and f != original[f.fault_id] for f in need.faults):
            refused.append(f"{need.episode_id}: a retained fault differs from the as-run identity")
            continue
        wanted = {
            f.fault_id for f in need.faults if f.fault_id in original or mechanism_additions((f,))
        }
        added = list(
            mechanism_additions(f for f in need.faults if f.fault_id not in identity.fault_ids)
        )
        if identity.faults and not added and wanted == identity.fault_ids:
            refused.append(f"{need.episode_id}: the episode as run has no row (its run is missing)")
            continue
        bad = [f for f in added if f.target.config_key in REFUSED_KEYS]
        if bad:
            keys = sorted({str(f.target.config_key) for f in bad})
            refused.append(f"{need.episode_id}: a change of {keys} cannot run in this engine")
            continue
        remove = sorted(identity.fault_ids - wanted)
        job = Job(identity=identity, mode=JobMode.replay, remove=remove, add=added)
        jobs.setdefault(job.key, job)
    return list(jobs.values()), refused


def fresh_proofs(jobs: Sequence[Job]) -> list[Job]:
    """Proof replays for a fresh set: every tenth of its episodes, at least 20."""
    seen: dict[str, EpisodeIdentity] = {}
    for job in jobs:
        seen.setdefault(job.identity.episode_id, job.identity)
    return proof_jobs(list(seen.values()))


def value_all(
    fix_sets: Sequence[tuple[str, FixSet]],
    table: OracleTable,
    truth: TruthTable,
    current: CurrentValues | None,
) -> list[dict[str, Any]]:
    """Each fix set's gain by `value_fix_set`, or what it still misses."""
    n = len(table.identities)
    table = mechanism_table(table)
    out: list[dict[str, Any]] = []
    for label, fix_set in fix_sets:
        row: dict[str, Any] = {
            "label": label,
            "actions": [a.model_dump(mode="json") for a in fix_set.actions],
            "budget": fix_set.budget.model_dump(mode="json"),
        }
        try:
            gain = value_fix_set(fix_set, table, truth, current)
        except MissingReplays as exc:
            row["missing"] = f"{len(exc.needs)} replays missing"
        except (ValueError, KeyError) as exc:
            row["error"] = str(exc)
        else:
            row.update(
                rescued=gain.rescued,
                broken=gain.broken,
                net=gain.net,
                points=round(100.0 * gain.net / n, 3) if n else None,
            )
        out.append(row)
    return out
