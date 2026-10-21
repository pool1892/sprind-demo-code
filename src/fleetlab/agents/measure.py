"""`make check-agents`: real calls to both vendors on the toy world, written into README.md.

    python -m fleetlab.agents.measure [--models opus,haiku,gpt-6.1-sol,gpt-6-luna] [--no-readme]

1. The gate (`checks.run_gate`) on the toy world of `tests/log_toy.py`; stop if it fails.
2. Per model, one at a time so that the latencies do not disturb each other:
   a trivial call; four camera frames with a printed code each; and an
   analysis in sessions with tools and the toy skill: session 1 follows the
   skill, fetches frames, writes memory and requests an experiment (state
   "waiting"); a toy engine answers the queue; session 2 reads memory and
   the result and submits a diagnosis (state "done").
3. Codex's trivial and frames calls again with `service_tier="fast"`.
4. Eight calls at once per vendor, on the cheap model.
5. The blinding verdict: every run directory searched for the toy run's
   canary, fault kinds and trait names (`fleetlab.log.leaks.code_terms`).

The records go to `check_agents.json` beside this file, and the tables into
README.md between the `check-agents` markers. Calls the subscriptions about
forty times.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fleetlab.agents.batch import Batch
from fleetlab.agents.checks import CODE_EXEMPT_TERMS, run_gate
from fleetlab.agents.cli import BACKENDS, Backend, agent_root, pinned_version
from fleetlab.agents.run import AgentResult, call_log, run_agent
from fleetlab.agents.sessions import AgentSpec, Analysis, EngineReply, ExperimentQueue, run_analysis
from fleetlab.agents.toolenv import ToolEnv, build_tool_env
from fleetlab.agents.toy import ToyWorld, toy_world
from fleetlab.contracts.tools import Diagnosis, ExperimentResult
from fleetlab.log import LogReader, leak_search
from fleetlab.log.leaks import code_terms

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
RECORDS = HERE / "check_agents.json"
BEGIN, END = "<!-- check-agents:begin -->", "<!-- check-agents:end -->"

MODELS: dict[str, tuple[Backend, str]] = {
    "opus": ("claude", "strong"),
    "haiku": ("claude", "cheap"),
    "gpt-6.1-sol": ("codex", "strong"),
    "gpt-6-luna": ("codex", "cheap"),
}
CHEAP = {"claude": "haiku", "codex": "gpt-6-luna"}

SYSTEM = (
    "You are an analysis agent for a fleet of robot arms. You work only through the tools you "
    "are given. Answer briefly and only with what you measured."
)
PING = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
CODES = {
    "type": "object",
    "properties": {"codes": {"type": "array", "items": {"type": "integer"}}},
    "required": ["codes"],
    "additionalProperties": False,
}

FRAMES_PROMPT = (
    "Each attached frame shows a three-digit code in large type. "
    "Return the codes in the order the frames were given."
)

TASK = """Analyse this fleet's log with your tools, in this order.
1. Call list_skills, then read_skill for the skill named triage, and follow it.
2. Call get_frames for episode e000001 at steps 0 and 2 from the scene camera.
3. Write your findings so far with memory_write: the robot with the lowest success rate,
   its rate, and how many frames get_frames returned.
4. Request one experiment with request_experiment: two arms, the first with no actions, the
   second recalibrating the scene_camera of the robot with the lowest success rate (scope:
   that robot); robots: that robot; 4 episodes per arm; paired starts; a one-line power note.
The runtime test fixture refuses this request without simulating it. In the later session,
read your notes and experiment_results, report the refusal, then submit a diagnosis with
submit_diagnosis: one cause whose target is the scene_camera of that
robot, direction "degraded", confidence 0.5, evidence one table reference (table "episodes"
with the query you ran), a one-line failure_mode; one line in mode_summaries; no warnings and
no retrain indicators."""


class ToyEngine:
    """Refuse each synthetic request to measure the session/memory/reply protocol.

    The fixture has too few logged episodes to fill new arms and no simulated
    experiment outcomes. Successful A/B replies are covered by the engine tests.
    """

    def __init__(self, export: Path) -> None:
        self.export = export
        self.queues: list[ExperimentQueue] = []

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        self.queues.append(queue)
        results = [
            ExperimentResult(
                experiment_id=item.experiment_id,
                status="refused",
                arms=[],
                message="request_not_runnable",
            )
            for item in queue.experiments
        ]
        return EngineReply(results=results, export=self.export)


def code_frames(directory: Path, count: int = 4, seed: int = 7) -> tuple[list[Path], list[int]]:
    """PNG frames of 256x192 with a printed three-digit code each."""
    rng = random.Random(seed)
    font = ImageFont.load_default(size=96)
    paths, codes = [], []
    for i in range(count):
        code = rng.randint(100, 999)
        image = Image.new("RGB", (256, 192), (30 + 20 * i, 60, 90))
        ImageDraw.Draw(image).text((40, 40), str(code), font=font, fill=(255, 255, 255))
        path = directory / f"frame-{i}.png"
        image.save(path)
        paths.append(path)
        codes.append(code)
    return paths, codes


def _row(task: str, model: str, res: AgentResult, correct: bool, **extra: Any) -> dict[str, Any]:
    backend, tier = MODELS.get(model, (res.meta.get("backend"), extra.pop("tier", "?")))
    return {
        "task": task,
        "backend": backend,
        "model": model,
        "model_reported": res.model,
        "tier": tier,
        "service_tier": res.meta.get("service_tier"),
        "ok": res.ok,
        "error": res.error,
        "correct": correct,
        "seconds": res.seconds,
        "answer_s": (res.meta.get("phases") or {}).get("answer_s"),
        "api_ms": res.meta.get("api_ms"),
        "input_tokens": res.usage.get("input_tokens"),
        "cached_input_tokens": res.usage.get("cached_input_tokens"),
        "output_tokens": res.usage.get("output_tokens"),
        "reasoning_tokens": res.usage.get("reasoning_tokens"),
        "tool_server_calls": len(res.tool_log),
        "window": res.meta.get("window"),
        "run_dir": res.run_dir,
        **extra,
    }


def lowest_robot(export: Path) -> str:
    with LogReader(export) as reader:
        rows = reader.query(
            "SELECT robot_id, avg((outcome = 'success')::int) AS rate FROM episodes "
            "GROUP BY robot_id ORDER BY rate, robot_id LIMIT 1"
        ).to_pylist()
    return str(rows[0]["robot_id"])


def protocol_kept(states: Sequence[str]) -> bool:
    """At least one waiting session (the experiment), every later one waiting until a done."""
    return len(states) >= 2 and states[-1] == "done" and all(s == "waiting" for s in states[:-1])


def measure_session(model: str, toy: ToyWorld, env: ToolEnv, base: Path) -> dict[str, Any]:
    from fleetlab.agents.audit import AuditContext

    backend, _ = MODELS[model]
    analysis = Analysis.create("w01", f"check-agents {model}", base=base / "analyses")
    engine = ToyEngine(toy.export)
    started = time.perf_counter()
    record = run_analysis(
        analysis,
        toy.export,
        AgentSpec(backend, model, timeout=600),
        SYSTEM,
        TASK,
        env,
        engine,
        max_sessions=3,
        run_root=base / "runs",
        audit=AuditContext((toy.run_dir,), toy.fleet),
    )
    seconds = time.perf_counter() - started
    target = lowest_robot(toy.export)
    diagnosis: Diagnosis | None = analysis.diagnosis()
    logs = []
    for session in record.sessions:
        path = call_log(Path(session.run_dir))
        logs += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tools_used = sorted({entry["tool"] for entry in logs})
    skill_read = any(
        e["tool"] == "read_skill" and e["arguments"].get("name") == "triage" and e["ok"]
        for e in logs
    )
    frames = sum(len(e["images"]) for e in logs if e["tool"] == "get_frames" and e["ok"])
    states = [s.state for s in record.sessions]
    correct = (
        protocol_kept(states)
        and skill_read
        and frames >= 2
        and bool(analysis.notes())
        and diagnosis is not None
        and any(c.target.scope == target for c in diagnosis.causes)
    )
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for s in record.sessions:
        for key in usage:
            usage[key] += int(s.usage.get(key) or 0)
    return {
        "task": "session",
        "backend": backend,
        "model": model,
        "tier": MODELS[model][1],
        "ok": all(s.ok for s in record.sessions),
        "correct": correct,
        "states": states,
        "session_seconds": [s.seconds for s in record.sessions],
        "seconds": round(seconds, 2),
        "skill_read": skill_read,
        "frames_returned": frames,
        "tools_used": tools_used,
        "tool_server_calls": len(logs),
        "notes_bytes": len(analysis.notes().encode()),
        "diagnosis_scope": [c.target.scope for c in diagnosis.causes] if diagnosis else None,
        "expected_scope": target,
        "queues": len(engine.queues),
        "analysis": str(analysis.root),
        "run_dirs": [s.run_dir for s in record.sessions],
        **usage,
    }


def measure_concurrency(backend: Backend, env: ToolEnv, base: Path, n: int = 8) -> dict[str, Any]:
    model = CHEAP[backend]
    batch = Batch(env, [backend], concurrency=n, run_root=base / "runs", stop_at_window=1.01)
    started = time.perf_counter()
    results = batch.map(
        lambda i: run_agent(
            backend,
            model,
            "You answer briefly.",
            "Reply with the single word: pong",
            schema=PING,
            timeout=180,
            run_root=base / "runs",
            label=f"concurrency {i}",
        ),
        list(range(n)),
    )
    wall = time.perf_counter() - started
    seconds = [r.seconds for r in results if r is not None]
    return {
        "task": "concurrency",
        "backend": backend,
        "model": model,
        "at_once": n,
        "ok": sum(1 for r in results if r is not None and r.ok),
        "errors": sorted({r.error for r in results if r is not None and r.error}),
        "batch_seconds": round(wall, 2),
        "fastest": min(seconds),
        "slowest": max(seconds),
        "calls_per_minute": round(60 * n / wall, 1),
        "window": next((r.meta.get("window") for r in results if r and r.meta.get("window")), None),
        "run_dirs": [r.run_dir for r in results if r is not None],
    }


def blinding(rows: Sequence[dict[str, Any]], toy: ToyWorld) -> dict[str, Any]:
    terms = [t for t in code_terms([toy.run_dir], fleet=toy.fleet) if t not in CODE_EXEMPT_TERMS]
    dirs: list[str] = []
    for row in rows:
        dirs += row.get("run_dirs") or ([row["run_dir"]] if row.get("run_dir") else [])
        if row.get("analysis"):
            dirs.append(row["analysis"])
    hits = []
    for directory in dirs:
        # Measurement directories mix prompts, schemas and outputs. Keep the
        # entire tree strict; none of these mixed records is agent_texts.
        report = leak_search(Path(directory), terms)
        hits += [f"{directory}: {h.where}: {h.term}" for h in report.hits]
    return {"terms": len(terms), "directories": len(dirs), "hits": hits, "held": not hits}


TASKS = ("trivial", "frames", "session", "concurrency", "names")
# Models whose names are only checked to be accepted by the pinned CLIs: one trivial call.
NAME_CHECKS: dict[str, Backend] = {"fable": "claude", "gpt-6-astra": "codex"}


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["task"], row["model"], row.get("service_tier"))


def run(models: Sequence[str], tasks: Sequence[str], write_readme: bool, merge: bool) -> int:
    root = agent_root()
    toy = toy_world(root / "toy")
    env = build_tool_env(skills=toy.skills)
    backends: list[Backend] = sorted({MODELS[m][0] for m in models})
    print("gate ...", flush=True)
    gate = run_gate(backends, env, toy.export, [toy.run_dir], fleet=toy.fleet)
    for backend, record in gate.items():
        print(f"  {backend} {record['version']}: {'passed' if record['passed'] else 'FAILED'}")
    if not all(r["passed"] for r in gate.values()):
        return 1
    base = root / "measure" / time.strftime("%Y%m%dT%H%M%S")
    base.mkdir(parents=True)
    Batch(env, backends)  # the same gate a batch passes
    rows: list[dict[str, Any]] = []
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with tempfile.TemporaryDirectory(prefix="frames-", dir=base) as tmp:
        frames, codes = code_frames(Path(tmp))
        for model in models:
            backend, _ = MODELS[model]
            tiers = ["default", "fast"] if backend == "codex" else ["default"]
            for tier in tiers:
                common: dict[str, Any] = {"run_root": base / "runs", "service_tier": tier}
                if "trivial" in tasks:
                    print(f"{model} {tier}: trivial", flush=True)
                    res = run_agent(
                        backend, model, "You answer briefly.", "Reply with the single word: pong",
                        schema=PING, timeout=180, **common,
                    )  # fmt: skip
                    rows.append(_row("trivial", model, res, _pong(res)))
                if "frames" in tasks:
                    print(f"{model} {tier}: frames", flush=True)
                    res = run_agent(
                        backend, model, "You read camera frames.", FRAMES_PROMPT,
                        images=frames, schema=CODES, timeout=180, **common,
                    )  # fmt: skip
                    correct = res.ok and res.result.get("codes") == codes
                    rows.append(_row("frames", model, res, correct))
            if "session" in tasks:
                print(f"{model}: session with tools and a skill", flush=True)
                rows.append(measure_session(model, toy, env, base))
        if "concurrency" in tasks:
            for backend in sorted({MODELS[m][0] for m in models}):
                print(f"{backend}: eight at once", flush=True)
                rows.append(measure_concurrency(backend, env, base))
        if "names" in tasks:
            for model, backend in NAME_CHECKS.items():
                if backend not in backends:  # the names of the vendors measured only
                    continue
                print(f"{model}: is the name accepted", flush=True)
                res = run_agent(
                    backend, model, "You answer briefly.", "Reply with the single word: pong",
                    schema=PING, timeout=180, run_root=base / "runs",
                )  # fmt: skip
                rows.append(_row("names", model, res, _pong(res), tier="alternative"))
    for row in rows:
        row["at"] = stamp
    failed = [r for r in rows if r["task"] != "concurrency" and not (r["ok"] and r["correct"])]
    failed += [r for r in rows if r["task"] == "concurrency" and r["ok"] != r["at_once"]]
    gates = {b: r["passed"] for b, r in gate.items()}
    if merge and RECORDS.is_file():
        old = json.loads(RECORDS.read_text())
        gates = {**old.get("gate", {}), **gates}
        fresh = {_key(r) for r in rows}
        rows = [r for r in old["rows"] if _key(r) not in fresh] + rows
        backends = sorted(set(backends) | {b for b in BACKENDS if b in old.get("versions", {})})
    verdict = blinding(rows, toy)
    data = {
        "at": sorted({r.get("at", stamp) for r in rows}),
        "versions": {b: pinned_version(b) for b in backends},
        "gate": gates,
        "blinding": verdict,
        "rows": rows,
    }
    RECORDS.write_text(json.dumps(data, indent=1, default=str) + "\n")
    if write_readme:
        update_readme(data)
    print(tables(data))
    return 0 if verdict["held"] and not failed else 1


def _pong(res: AgentResult) -> bool:
    answer = res.result.get("answer", "") if res.ok and isinstance(res.result, dict) else ""
    return str(answer).strip().lower().strip(".") == "pong"


def _n(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def tables(data: dict[str, Any]) -> str:
    rows = data["rows"]
    versions = ", ".join(f"{b} {v}" for b, v in data["versions"].items())
    at = data["at"] if isinstance(data["at"], list) else [data["at"]]
    when = at[0] if len(at) == 1 else f"{at[0]} to {at[-1]}"
    out = [
        f"Measured {when} with {versions}, by `make check-agents`; raw records in "
        "`check_agents.json`. Every call in the bwrap sandboxes, one at a time except the last "
        'table. "Process" is the wall clock of the whole CLI process; "answer" is when the '
        "event with the answer arrived.",
        "",
        "| Model | Tier | Codex service tier | Trivial: answer / process (s) "
        "| Four frames: answer / process (s) | Correct | Tokens in / out (trivial; frames) |",
        "|---|---|---|---|---|---|---|",
    ]
    seen: list[tuple[str, Any]] = []
    for r in rows:
        key = (r["model"], r.get("service_tier"))
        if r["task"] in ("trivial", "frames") and key not in seen:
            seen.append(key)
    for model, tier in seen:
        t = next(
            r
            for r in rows
            if r["task"] == "trivial" and (r["model"], r.get("service_tier")) == (model, tier)
        )
        f = next(
            r
            for r in rows
            if r["task"] == "frames" and (r["model"], r.get("service_tier")) == (model, tier)
        )
        correct = f"{'yes' if t['correct'] else 'NO'} / {'yes' if f['correct'] else 'NO'}"
        tokens = (
            f"{_n(t['input_tokens'])} / {_n(t['output_tokens'])}; "
            f"{_n(f['input_tokens'])} / {_n(f['output_tokens'])}"
        )
        out.append(
            f"| `{model}` ({t['model_reported'] or '?'}) | {t['tier']} | {tier or '-'} | "
            f"{_n(t['answer_s'])} / {_n(t['seconds'])} | "
            f"{_n(f['answer_s'])} / {_n(f['seconds'])} | {correct} | {tokens} |"
        )
    out += [
        "",
        'Sessions with tools and the toy skill (session 1 ends "waiting" with an experiment '
        "queued, the toy engine answers, a later session submits a diagnosis and ends "
        '"done"; an agent may request more experiments on the way):',
        "",
        "| Model | States | Seconds per session | Tool calls | Skill read | Frames "
        "| Diagnosis names the right robot | Tokens in / out (all sessions) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["task"] != "session":
            continue
        right = r["diagnosis_scope"] is not None and r["expected_scope"] in r["diagnosis_scope"]
        out.append(
            f"| `{r['model']}` | {', '.join(r['states'])} | "
            f"{', '.join(_n(s) for s in r['session_seconds'])} | {r['tool_server_calls']} | "
            f"{'yes' if r['skill_read'] else 'no'} | {r['frames_returned']} | "
            f"{'yes' if right else 'no'} | {_n(r['input_tokens'])} / {_n(r['output_tokens'])} |"
        )
    out += [
        "",
        "Eight trivial calls started at once:",
        "",
        "| Model | When | Answered | Whole batch (s) | Fastest / slowest call (s) "
        "| Calls per minute | Errors | Subscription window (claude) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["task"] == "concurrency":
            out.append(
                f"| `{r['model']}` | {r.get('at', '-')} | {r['ok']} of {r['at_once']} | "
                f"{_n(r['batch_seconds'])} | {_n(r['fastest'])} / {_n(r['slowest'])} | "
                f"{_n(r['calls_per_minute'])} | {', '.join(r['errors']) or 'none'} | "
                f"{_window(r.get('window'))} |"
            )
    names = [r for r in rows if r["task"] == "names"]
    if names:
        out += [
            "",
            "The other model names, one trivial call each:",
            "",
            "| Model | Reported as | Accepted and answered | Answer / process (s) "
            "| Tokens in / out |",
            "|---|---|---|---|---|",
        ]
        for r in names:
            out.append(
                f"| `{r['model']}` | {r['model_reported'] or '-'} | "
                f"{'yes' if r['correct'] else 'NO: ' + str(r['error'])} | "
                f"{_n(r['answer_s'])} / {_n(r['seconds'])} | "
                f"{_n(r['input_tokens'])} / {_n(r['output_tokens'])} |"
            )
    v = data["blinding"]
    out += [
        "",
        f"Blinding verdict: **{'held' if v['held'] else 'LEAKED'}**. The gate passed for "
        f"{', '.join(b for b, ok in data['gate'].items() if ok) or 'nothing'}; {v['directories']} "
        f"run and analysis directories searched for {v['terms']} terms of the toy run's truth "
        f"(canary, fault kinds, trait names): {len(v['hits'])} hits.",
    ]
    return "\n".join(out)


def _window(window: dict[str, Any] | None) -> str:
    if not window:
        return "-"
    used = ", ".join(
        f"{k} {v:.0%}" for k, v in (window.get("utilization") or {}).items() if v is not None
    )
    return f"{window.get('status')}: {used}"


def update_readme(data: dict[str, Any]) -> None:
    text = README.read_text()
    if BEGIN not in text or END not in text:
        raise ValueError(f"{README} has no {BEGIN} ... {END} block")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{BEGIN}\n{tables(data)}\n{END}{tail}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.agents.measure")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--tasks", default=",".join(TASKS), help=f"some of {','.join(TASKS)}")
    parser.add_argument(
        "--merge", action="store_true", help="replace only these rows in check_agents.json"
    )
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    tasks = [t for t in args.tasks.split(",") if t]
    unknown = [m for m in models if m not in MODELS] + [t for t in tasks if t not in TASKS]
    if unknown:
        parser.error(f"unknown {unknown}; models: {list(MODELS)}, tasks: {list(TASKS)}")
    return run(models, tasks, write_readme=not args.no_readme, merge=args.merge)


if __name__ == "__main__":
    sys.exit(main())
