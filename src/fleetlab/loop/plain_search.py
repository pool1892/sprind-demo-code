"""The plain search: the comparator for adaptation (the plan's comparisons: "for adaptation,
a plain search over the adaptation setting, which tries values in turn on the affected
site").

**What it is told,** and the page says so: the scope (the affected site) and the setting
(a configuration key). Nothing else: no delay, no telemetry, no outcome before its own.

**What it does,** fixed here before any result, as an engineer's default for an integer
setting, not read off this simulation's calibration:

1. **One experiment** at the scope's robots, from the fleet's current state, through the
   agent's own experiment path: the tool server's `request_experiment`, the round's queue
   and engine (`fleetlab.loop.experiments.RoundEngine`), the same allocations, the same
   attempt budget and accounting. Arm 0 is the current state (no action); each further
   arm sets one value of the grid at the scope (`set KEY = v at SCOPE`), the current value
   left out. Every arm has the same attempts, with paired starts.
2. **The grid** (`GRID`): 0, 5, 10, 15, 20 and 25, the range of a setting whose values run
   from 0 to one less than the policy's chunk of 30 actions, in equal steps of five.
3. **The attempts** (`PER_ARM`): 24 per arm, six for each robot of a four-robot site.
4. **The choice,** on the experiment's visible outcomes only (`choose`): the arm with the
   most successes; a tie goes to the value nearest the current one, then the smaller.
5. **The stopping rule:** the chosen value is adopted at the scope only if it is not the
   current one and beats the current state's arm on the paired attempts (an attempt of
   one arm and the same slot of the other share robot, day, task and start state) by an
   exact one-sided sign test at 5% divided by the number of values tried (Bonferroni over
   the arms it picked from). Otherwise nothing is adopted. Either way the search decides
   once and stops.

It runs as a one-round engagement of the driver (`fleetlab.loop.driver.run_rounds`) with
a scripted stand-in for the agent (`fleetlab.loop.scripted.ScriptedAgent`): session 1
requests the experiment and waits, session 2 reads the results and submits the round's
decision (`submit_decision`, with `stop`). The driver adopts the decision into the
ledger and evaluates the states on the reserved panel (`fleetlab.loop.evaluate`), whose
identities are minted from the world seed and layout alone, so they are the agent's
panel when the world, layout and panel size are the agent's.

**Its cost receipt** (`receipt`) is in the agent's cost format (`fleetlab.loop.costs`):
the requests, the attempts per arm and the heavy queue exactly as the driver counted
them, and its sessions as what they were: scripted, computed on the CPU, with no model.
Model starts, model seconds and tokens are 0 and the usage is complete; the sessions'
own time is `compute_seconds`. The driver's raw record of the round, which counts each
scripted session as a model start without a usage record, is kept beside it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.ids import ScopeKind, scope_covers, scope_kind
from fleetlab.contracts.vocabulary import ConfigKey

__all__ = [
    "ALPHA",
    "GRID",
    "PER_ARM",
    "Choice",
    "PlainSearch",
    "choose",
    "current_value",
    "decide_step",
    "main",
    "receipt",
    "request_for",
    "request_step",
    "run_plain_search",
    "script",
]

GRID: tuple[int, ...] = (0, 5, 10, 15, 20, 25)
PER_ARM = 24
ALPHA = 0.05


@dataclass(frozen=True)
class PlainSearch:
    """What the search is told (the scope and the key) and its fixed design."""

    scope: str
    key: ConfigKey
    grid: tuple[int, ...] = GRID
    per_arm: int = PER_ARM
    alpha: float = ALPHA

    def __post_init__(self) -> None:
        if scope_kind(self.scope) is ScopeKind.fleet:
            raise ValueError("the plain search is told one site or robot, not the fleet")
        if len(set(self.grid)) != len(self.grid) or len(self.grid) < 2:
            raise ValueError("a grid of at least two distinct values")
        if self.per_arm < 1 or not 0 < self.alpha < 1:
            raise ValueError(f"an impossible design: {self}")


@dataclass(frozen=True)
class Choice:
    """The search's decision from its experiment: per arm its value and successes, the arm
    chosen, the paired comparison with the current state's arm, and whether to adopt."""

    values: list[ConfigValue]  # arm 0 is the current value
    successes: list[int]
    attempts: list[int]
    chosen: ConfigValue
    rescued: int  # chosen arm succeeded where the current state's failed, same slot
    broken: int
    p_value: float  # one-sided exact sign test
    threshold: float  # alpha over the values tried
    adopt: bool
    reason: str

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _sign_test(rescued: int, broken: int) -> float:
    """One-sided exact sign test: the chance of at least `rescued` of the discordant pairs
    favouring the chosen arm if neither arm were better."""
    n = rescued + broken
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(rescued, n + 1)) / 2.0**n


def choose(
    values: Sequence[ConfigValue], outcomes: Sequence[Sequence[bool]], alpha: float = ALPHA
) -> Choice:
    """The fixed rule over the experiment's outcomes: `values[0]` is the current state's arm,
    `outcomes[i][slot]` arm i's attempt in that slot (paired across arms)."""
    if len(values) != len(outcomes) or len(values) < 2:
        raise ValueError("one outcome list per arm, the current state's first")
    slots = len(outcomes[0])
    if any(len(o) != slots for o in outcomes):
        raise ValueError("every arm has the same paired slots")
    successes = [sum(bool(x) for x in o) for o in outcomes]
    current = values[0]

    def distance(i: int) -> float:
        a, b = values[i], current
        if isinstance(a, int | float) and isinstance(b, int | float):
            return abs(float(a) - float(b))
        return 0.0 if a == b else math.inf

    def order(i: int) -> tuple[int, float, float]:
        v = values[i]
        return (-successes[i], distance(i), float(v) if isinstance(v, int | float) else 0.0)

    best = min(range(len(values)), key=order)
    rescued = sum(bool(outcomes[best][s]) and not outcomes[0][s] for s in range(slots))
    broken = sum(bool(outcomes[0][s]) and not outcomes[best][s] for s in range(slots))
    threshold = alpha / (len(values) - 1)
    p = _sign_test(rescued, broken) if best != 0 else 1.0
    if best == 0:
        adopt, reason = False, "the current value had the most successes"
    elif rescued <= broken or p >= threshold:
        adopt = False
        reason = (
            f"value {values[best]} had the most successes, but its paired sign test against "
            f"the current state gave p = {p:.3g}, not below {threshold:.3g}"
        )
    else:
        adopt = True
        reason = (
            f"value {values[best]}: {successes[best]} of {slots} against {successes[0]} of "
            f"{slots} with the current value {current}; paired sign test p = {p:.3g} below "
            f"{threshold:.3g}"
        )
    return Choice(
        values=list(values),
        successes=successes,
        attempts=[slots] * len(values),
        chosen=values[best],
        rescued=rescued,
        broken=broken,
        p_value=p,
        threshold=threshold,
        adopt=adopt,
        reason=reason,
    )


# ---- the public view the search reads -------------------------------------------------------


def current_value(fleet: Mapping[str, Any], scope: str, key: ConfigKey) -> ConfigValue:
    """The value in force at `scope` in the public description's fleet: a robot's own, else
    its site's, else the fleet's; for a site, the site's, else the fleet's."""
    layers: list[Mapping[str, Any]] = [fleet["config"]["values"]]
    for site in fleet["sites"]:
        robots = {r["id"]: r for r in site["robots"]}
        if site["id"] == scope:
            layers.append(site["config"]["values"])
        elif scope in robots:
            layers += [site["config"]["values"], robots[scope]["config"]["values"]]
    for values in reversed(layers):
        if key.value in values:
            value: ConfigValue = values[key.value]
            return value
    raise KeyError(f"the description has no value of {key.value} at {scope}")


def robots_of(fleet: Mapping[str, Any], scope: str) -> list[str]:
    """The robots a scope covers, in the description's order."""
    return [
        r["id"]
        for site in fleet["sites"]
        for r in site["robots"]
        if scope_covers(scope, r["id"], site["id"])
    ]


def arm_values(search: PlainSearch, current: ConfigValue) -> list[ConfigValue]:
    """Arm 0 the current value, then the grid without it, in the grid's order."""
    return [current, *(v for v in search.grid if v != current)]


def request_for(search: PlainSearch, fleet: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """The experiment request (`contracts.tools.ExperimentRequest` as JSON) and its arms'
    values, from the public description's fleet."""
    values = arm_values(search, current_value(fleet, search.scope, search.key))
    target = {
        "scope": search.scope,
        "component": None,
        "config_key": search.key.value,
        "condition_key": None,
    }
    arms: list[dict[str, Any]] = [{"actions": []}] + [
        {"actions": [{"verb": "set", "target": target, "value": v}]} for v in values[1:]
    ]
    request: dict[str, Any] = {
        "arms": arms,
        "robots": robots_of(fleet, search.scope),
        "episodes_per_arm": search.per_arm,
        "paired_starts": True,
        "power_note": (
            f"plain search: {len(arms)} arms of {search.per_arm} paired attempts, the current "
            f"value against the grid {list(search.grid)}"
        ),
    }
    return request, values


# ---- the scripted sessions --------------------------------------------------------------------


def _data(result: Any) -> Any:
    text = "\n".join(p["text"] for p in result.content if p.get("type") == "text")
    if result.is_error:
        raise RuntimeError(f"tool error: {text}")
    return json.loads(text)


def _fleet(box: Any) -> dict[str, Any]:
    described: dict[str, Any] = _data(
        box.call("describe", {"part": "fleet", "include_logs": False})
    )
    fleet: dict[str, Any] = described["fleet"]
    return fleet


def request_step(search: PlainSearch) -> Any:
    """Session 1: request the grid experiment and wait for it."""

    def step(box: Any) -> str:
        request, values = request_for(search, _fleet(box))
        receipt = _data(box.call("request_experiment", request))
        if not receipt["accepted"]:
            raise RuntimeError(f"plain search request refused: {receipt['problems']}")
        note = {"plain_search": {"receipt": receipt["receipt_id"], "values": values}}
        _data(box.call("memory_write", {"notes": json.dumps(note)}))
        return "waiting"

    return step


def _outcomes(box: Any, episode_ids: Sequence[str]) -> list[bool]:
    """One arm's visible outcomes, in slot order."""
    listed = ", ".join(f"'{e}'" for e in episode_ids)
    rows = _data(
        box.call(
            "query",
            {"sql": f"SELECT episode_id, outcome FROM episodes WHERE episode_id IN ({listed})"},
        )
    )
    found = {eid: outcome == "success" for eid, outcome in rows["rows"]}
    missing = [e for e in episode_ids if e not in found]
    if missing:
        raise RuntimeError(f"plain search: no visible outcome for {len(missing)} attempts")
    return [found[e] for e in episode_ids]


def decide_step(search: PlainSearch, record: list[Choice] | None = None) -> Any:
    """Session 2: read the experiment's results, apply the rule, submit the decision."""

    def step(box: Any) -> str:
        results = [r for r in _data(box.call("experiment_results", {})) if r["status"] == "done"]
        if len(results) != 1:
            raise RuntimeError(f"plain search: {len(results)} finished experiments, not 1")
        _, values = request_for(search, _fleet(box))
        arms = sorted(results[0]["arms"], key=lambda a: a["arm_index"])
        outcomes = [_outcomes(box, arm["episode_ids"]) for arm in arms]
        choice = choose(values, outcomes, search.alpha)
        if record is not None:
            record.append(choice)
        target = {
            "scope": search.scope,
            "component": None,
            "config_key": search.key.value,
            "condition_key": None,
        }
        slots = len(outcomes[0])
        share = len(robots_of(_fleet(box), search.scope)) / max(
            1, len(robots_of(_fleet(box), "fleet"))
        )
        best = choice.values.index(choice.chosen)
        measured = 100.0 * (choice.successes[best] - choice.successes[0]) / slots
        gain = max(0.0, measured * share) if choice.adopt else 0.0
        decision = {
            "adopt": (
                [
                    {
                        "action": {"verb": "set", "target": target, "value": choice.chosen},
                        "rationale": f"plain search over {list(search.grid)}: {choice.reason}",
                    }
                ]
                if choice.adopt
                else []
            ),
            "replace": [],
            "undo": [],
            # The site's measured difference over its share of the fleet's robots.
            "predicted_set_gain": {"low": 0.0, "mid": gain, "high": 2.0 * gain},
            "expectations": [
                f"plain search: {choice.reason}",
            ],
            "stop": True,
            "ranked_candidates": [],
        }
        receipt = _data(box.call("submit_decision", decision))
        if not receipt["accepted"]:
            raise RuntimeError(f"plain search decision refused: {receipt['problems']}")
        _data(box.call("memory_write", {"notes": json.dumps({"plain_search": choice.as_json()})}))
        return "done"

    return step


def script(search: PlainSearch, run_root: Path, record: list[Choice] | None = None) -> Any:
    """The scripted stand-in for the agent: request, then decide."""
    from fleetlab.loop.scripted import ScriptedAgent

    return ScriptedAgent([request_step(search), decide_step(search, record)], run_root=run_root)


def receipt(round_costs: Mapping[str, Any]) -> dict[str, Any]:
    """The search's cost receipt from the driver's cost record of its round: the same keys,
    the attempts, requests, experiments and heavy queue as counted, the sessions as
    scripted ones with no model (model starts, seconds and tokens 0, usage complete), and
    their own time as `compute_seconds`."""
    from fleetlab.loop.costs import TOKEN_KEYS

    sessions = list(round_costs.get("per_session") or [])
    zero = dict.fromkeys(TOKEN_KEYS, 0)
    out = dict(round_costs)
    out.update(
        model_starts=0,
        failed_starts=0,
        starts_without_usage=0,
        usage_complete=True,
        usage_note=None,
        model_seconds=0.0,
        tokens=dict(zero),
        cost_usd_list_price=0.0,
        cost_usd_list_price_status="complete",
        # The agent's price format, bounded at nothing (`cost_delta` reads `low_usd`).
        api_equivalent_usd={
            "status": "bounded_api_equivalent",
            "low_usd": 0.0,
            "high_usd": 0.0,
            "basis": "no model was called",
        },
        scripted_sessions=len(sessions),
        compute_seconds=sum(float(s.get("seconds") or 0) for s in sessions),
        per_session=[
            {
                **s,
                "starts": 0,
                "seconds": 0.0,
                "tokens": dict(zero),
                "usage_complete": True,
                "compute_seconds": float(s.get("seconds") or 0),
            }
            for s in sessions
        ],
    )
    return out


@dataclass
class SearchRun:
    """What a run of the search reports: its choice, its ledger, its cost receipt (and the
    driver's raw record of the round), its evaluation."""

    search: dict[str, Any]
    choice: dict[str, Any] | None
    adopted: list[dict[str, Any]]
    ledger: str | None
    costs: dict[str, Any]
    evaluation: dict[str, Any] | None
    engagement: str
    raw_costs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def run_plain_search(
    world_root: Path | Any,
    out: Path,
    search: PlainSearch,
    *,
    budget: Any = None,
    heavy: bool = True,
    setup: Any = None,
    layout: Any = None,
    env: Any = None,
    panel_per_robot_day: int | None = None,
    workers: int = 4,
    queue_log: Path | None = None,
    panel_video_indices: Sequence[int] = (),
) -> SearchRun:
    """The search as a one-round engagement of the driver; `out/plain_search.json` holds
    the report.

    `budget` is the round's allocated ceiling. By default it is one arm more than the
    grid has, times `per_arm`, at `per_arm` per arm (168 attempts with the default grid)
    and no count of requests of its own: room for the one request, which takes an arm for
    the current state and one for every other grid value (144 attempts when the current
    value is on the grid). What is charged is what the request takes. Pass the agent's
    budget to give both the same ceiling."""
    from fleetlab.agents.sessions import AgentSpec
    from fleetlab.loop.driver import AttemptBudget, run_rounds

    out = out.expanduser().resolve()
    record: list[Choice] = []
    agent = script(search, out / "scripted", record)  # created once the driver made `out`
    arms = len(search.grid) + 1
    budget = budget or AttemptBudget(attempts=arms * search.per_arm, per_arm=search.per_arm)
    engagement = run_rounds(
        world_root,
        out,
        agent=AgentSpec("codex", "plain-search"),
        budget=budget,
        max_sessions=2,
        workers=workers,
        queue_log=queue_log,
        label="plain-search",
        scripted=agent,
        heavy=heavy,
        setup=setup,
        layout=layout,
        env=env,
        panel_per_robot_day=panel_per_robot_day,
        panel_video_indices=panel_video_indices,
    )
    first = engagement["rounds"][0] if engagement.get("rounds") else {}
    decision = first.get("decision") or {}
    ledger = Path(engagement["loop"]) / "ledgers" / "ledger-r1.json"
    report = SearchRun(
        search={**asdict(search), "key": search.key.value, "grid": list(search.grid)},
        choice=record[-1].as_json() if record else None,
        adopted=[c["action"] for c in decision.get("adopt", [])],
        ledger=str(ledger) if ledger.is_file() else None,
        costs=receipt(first.get("costs", {})),
        evaluation=(engagement.get("evaluation") or {}).get("summary"),
        engagement=str(out / "engagement" / "engagement.json"),
        raw_costs=first.get("costs", {}),
        extra={
            "outcome": engagement.get("outcome"),
            "ended_because": engagement.get("ended_because"),
        },
    )
    (out / "plain_search.json").write_text(json.dumps(asdict(report), indent=1, default=str) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """The search on a stored world; every simulation step is a heavy job of its own."""
    import argparse

    from fleetlab.loop.driver import AttemptBudget

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.plain_search")
    parser.add_argument("--world", type=Path, required=True, help="the stored world's root")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--scope", required=True, help="the affected site (disclosed)")
    parser.add_argument("--key", default=ConfigKey.control_latency_compensation_steps.value)
    parser.add_argument("--per-arm", type=int, default=PER_ARM)
    parser.add_argument("--attempts", type=int, default=None, help="the round's budget")
    parser.add_argument("--panel-per-robot-day", type=int, default=None)
    parser.add_argument(
        "--panel-video-index",
        type=int,
        action="append",
        default=[],
        help="panel attempts with this index within their robot and day record video",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queue-log", type=Path, default=None)
    args = parser.parse_args(argv)
    search = PlainSearch(args.scope, ConfigKey(args.key), per_arm=args.per_arm)
    budget = AttemptBudget(attempts=args.attempts, per_arm=args.per_arm) if args.attempts else None
    report = run_plain_search(
        args.world,
        args.out,
        search,
        budget=budget,
        panel_per_robot_day=args.panel_per_robot_day,
        workers=args.workers,
        queue_log=args.queue_log,
        panel_video_indices=tuple(args.panel_video_index),
    )
    print(json.dumps({"choice": report.choice, "adopted": report.adopted}, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
