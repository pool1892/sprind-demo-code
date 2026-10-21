"""A scripted stand-in for the analysis agent: the driver's mechanics without a model.

`ScriptedAgent` replaces `fleetlab.agents.sessions.run_agent`. Each call is one
session: it opens the tool server's `ToolBox` over exactly what the session would
mount (the export, memory, outbox and results, with the session's budget limits),
runs the step of the script for that call, logs every tool call as the real server
does, and answers with a `SessionReport`. The runtime then decides the session's state
from the outbox as for a model, so the queue, the engine, the public results and the
next session's view all run for real. No model is called and nothing is audited at
launch (the launch audit belongs to `run_agent`, which the script replaces).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.agents.run import AgentResult, ToolAccess
from fleetlab.toolserver.server import Limits, Paths, ToolBox

__all__ = [
    "ScriptedAgent",
    "Step",
    "catalogue_fix_set",
    "decide_step",
    "request_step",
    "scripted_from_json",
    "submit_analysis_step",
]

# One session's work: it uses the tools and returns the session status it reports.
Step = Callable[[ToolBox], str]


def _data(result: Any) -> Any:
    text = "\n".join(p["text"] for p in result.content if p.get("type") == "text")
    if result.is_error:
        raise RuntimeError(f"tool error: {text}")
    return json.loads(text)


@dataclass
class ScriptedAgent:
    """`run_agent`'s stand-in: session k runs `steps[k - 1]` (the last step repeats)."""

    steps: Sequence[Step]
    run_root: Path
    calls: int = 0
    seen: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, backend: str, model: str, system: str, prompt: str, **kwargs: Any
    ) -> AgentResult:
        tools: ToolAccess = kwargs["tools"]
        self.calls += 1
        run_dir = self.run_root / f"scripted-{time.strftime('%Y%m%dT%H%M%S')}-{self.calls:03d}"
        log = run_dir / "box" / "toolbox" / "tool_calls.jsonl"
        log.parent.mkdir(parents=True)
        (run_dir / "prompt.json").write_text(json.dumps({"system": system, "prompt": prompt}))
        on_start = kwargs.get("on_start")
        if on_start is not None:
            on_start(run_dir)
        chosen = {
            name: value
            for name in (
                "max_experiment_episodes",
                "max_experiment_requests",
                "max_episodes_per_arm",
                "budget_first_session",
            )
            if (value := getattr(tools, name)) is not None
        }
        box = ToolBox(
            Paths(
                export=tools.export,
                memory=tools.memory,
                outbox=tools.outbox,
                results=tools.results,
                skills=tools.env.skills,
                log=log,
                learned=tools.learned,
            ),
            tools.tools,
            session=tools.session,
            limits=Limits(**chosen),
        )
        started = time.perf_counter()
        try:
            step = self.steps[min(self.calls, len(self.steps)) - 1]
            status = step(box)
        finally:
            box.close()
        tool_log = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        self.seen.append({"session": tools.session, "prompt": prompt, "status": status})
        report = {"status": status, "summary": f"scripted session {tools.session}"}
        result = AgentResult(
            ok=True,
            result=report,
            usage={},
            seconds=time.perf_counter() - started,
            tool_log=tool_log,
            model="scripted",
            run_dir=str(run_dir),
        )
        (run_dir / "result.json").write_text(
            json.dumps({"ok": True, "result": report, "usage": {}, "seconds": result.seconds})
        )
        return result


def request_step(*requests: Mapping[str, Any]) -> Step:
    """Request the experiments, write a note, and wait for them."""

    def step(box: ToolBox) -> str:
        ids = []
        for request in requests:
            receipt = _data(box.call("request_experiment", dict(request)))
            if not receipt["accepted"]:
                raise RuntimeError(f"scripted request refused: {receipt['problems']}")
            ids.append(receipt["receipt_id"])
        _data(box.call("memory_write", {"notes": f"requested {', '.join(ids)}"}))
        return "waiting"

    return step


def catalogue_fix_set(
    box: ToolBox,
    first: Sequence[Mapping[str, Any]] = (),
    verbs: Sequence[str] = ("recalibrate", "clean", "service", "replace", "revert"),
) -> dict[str, Any]:
    """A ranked list of ten catalogue actions without values, `first` at its head."""
    listed = ", ".join(f"'{v}'" for v in verbs)
    rows = _data(
        box.call(
            "query",
            {
                "sql": "SELECT verb, scope, component, config_key, condition_key FROM catalogue "
                f"WHERE verb IN ({listed}) "
                "ORDER BY hours, verb, scope, component, config_key LIMIT 30"
            },
        )
    )
    actions = [dict(a) for a in first]
    for verb, scope, component, config_key, condition_key in rows["rows"]:
        action = {
            "verb": verb,
            "target": {
                "scope": scope,
                "component": component,
                "config_key": config_key,
                "condition_key": condition_key,
            },
            "value": None,
        }
        if action not in actions:
            actions.append(action)
        if len(actions) == 10:
            break
    return {
        "actions": [
            {
                "action": action,
                "expected_gain": float(10 - rank),
                "uncertainty": {"low": 0.0, "high": float(10 - rank) + 1.0},
                "rationale": "scripted",
            }
            for rank, action in enumerate(actions)
        ]
    }


def submit_analysis_step(cause: Mapping[str, Any]) -> Step:
    """Read the results and the arms' outcomes, then submit a diagnosis naming `cause`
    (a target) and a ranked list headed by the action that fixes it."""

    def step(box: ToolBox) -> str:
        _data(box.call("memory_read", {}))
        _data(box.call("experiment_results", {}))
        _data(
            box.call(
                "query",
                {
                    "sql": "SELECT experiment_id, count(*) AS n, "
                    "sum(CAST(outcome = 'success' AS INT)) AS ok FROM episodes "
                    "WHERE experiment_id IS NOT NULL GROUP BY experiment_id"
                },
            )
        )
        diagnosis = {
            "causes": [
                {
                    "target": dict(cause),
                    "direction": "degraded",
                    "confidence": 0.5,
                    "evidence": [
                        {
                            "episode_id": None,
                            "step": None,
                            "table": "episodes",
                            "query": "scripted",
                        }
                    ],
                    "failure_mode": "scripted",
                }
            ],
            "mode_summaries": ["scripted"],
            "warnings": [],
            "retrain": [],
        }
        if not _data(box.call("submit_diagnosis", diagnosis))["accepted"]:
            raise RuntimeError("scripted diagnosis refused")
        fix = {"verb": "recalibrate", "target": dict(cause), "value": None}
        if not _data(box.call("submit_fix_set", catalogue_fix_set(box, [fix])))["accepted"]:
            raise RuntimeError("scripted fix set refused")
        return "done"

    return step


def decide_step(
    adopt: Sequence[Mapping[str, Any]] = (),
    *,
    replace: Sequence[Mapping[str, Any]] = (),
    undo: Sequence[Mapping[str, Any]] = (),
    stop: bool = False,
    gain: tuple[float, float, float] = (1.0, 2.0, 4.0),
    request_first: Mapping[str, Any] | None = None,
    alternatives: int = 4,
    learned: Mapping[str, Any] | None = None,
) -> Step:
    """A round's deciding session: read memory, results and the budget, then submit a
    decision adopting `adopt` (actions), replacing and undoing ledger entries
    (`{"entry_id", "action"}`, `{"entry_id"}`), with `alternatives` further catalogue
    actions as ranked candidates (none with 0). With `request_first` the session only
    requests that experiment and waits (the decision comes in the next session)."""

    def step(box: ToolBox) -> str:
        notes = _data(box.call("memory_read", {}))["notes"]
        _data(box.call("experiment_results", {}))
        _data(box.call("experiment_budget", {}))
        if request_first is not None and "requested" not in notes:
            return request_step(request_first)(box)
        chosen = [dict(a) for a in adopt] + [dict(r["action"]) for r in replace]
        # Alternatives from component work, which every robot can take; the tool checks
        # every alternative like an adoption.
        components = ("recalibrate", "clean", "service", "replace")
        ranked = (
            catalogue_fix_set(box, chosen, components)["actions"][
                len(chosen) : len(chosen) + alternatives
            ]
            if alternatives
            else []
        )
        decision = {
            "adopt": [{"action": dict(a), "rationale": "scripted"} for a in adopt],
            "replace": [
                {"entry_id": r["entry_id"], "action": dict(r["action"]), "rationale": "scripted"}
                for r in replace
            ],
            "undo": [{"entry_id": u["entry_id"], "rationale": "scripted"} for u in undo],
            "predicted_set_gain": {"low": gain[0], "mid": gain[1], "high": gain[2]},
            "expectations": ["scripted: the adopted robots succeed more often"],
            "stop": stop,
            "ranked_candidates": ranked,
        }
        if learned is not None:
            kept = _data(box.call("write_learned_skill", dict(learned)))
            if not kept["accepted"]:
                raise RuntimeError(f"scripted learned skill refused: {kept['problems']}")
        receipt = _data(box.call("submit_decision", decision))
        if not receipt["accepted"]:
            raise RuntimeError(f"scripted decision refused: {receipt['problems']}")
        _data(box.call("memory_write", {"notes": f"{notes}\ndecided {receipt['receipt_id']}"}))
        return "done"

    return step


def scripted_from_json(script: Sequence[Mapping[str, Any]], run_root: Path) -> ScriptedAgent:
    """A script from JSON, one entry per session:
    `{"kind": "request", "requests": [ExperimentRequest, ...]}` or
    `{"kind": "decide", "adopt": [Action, ...], "replace": [{"entry_id", "action"}],
    "undo": [{"entry_id"}], "stop": bool, "request_first": ExperimentRequest | null}`."""
    steps: list[Step] = []
    for entry in script:
        if entry["kind"] == "request":
            steps.append(request_step(*entry["requests"]))
        elif entry["kind"] == "decide":
            steps.append(
                decide_step(
                    entry.get("adopt", ()),
                    replace=entry.get("replace", ()),
                    undo=entry.get("undo", ()),
                    stop=bool(entry.get("stop", False)),
                    request_first=entry.get("request_first"),
                    alternatives=int(entry.get("alternatives", 4)),
                    learned=entry.get("learned"),
                )
            )
        else:
            raise ValueError(f"unknown script step {entry['kind']!r}")
    return ScriptedAgent(steps, run_root=run_root)
