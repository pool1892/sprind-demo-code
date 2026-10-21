"""Replay the 92 stored D2 prompts, byte-identical, without a robot or GPU.

Exactly 276 calls, no automatic retries. Agreement is with xhigh, not truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fleetlab.agents.batch import Batch
from fleetlab.agents.checks import require_verified
from fleetlab.agents.run import AgentResult, run_agent
from fleetlab.live.vlm.measure import dump, gate
from fleetlab.live.vlm.supervisor import Answer, instructions, model_time, supervisor_env
from fleetlab.log.runfiles import read_table

VARIANTS = (("gpt-6-astra", "medium"), ("gpt-6-astra", "low"), ("gpt-6.1-sol", "medium"))


def inputs(source: Path, measurement: Path) -> list[dict[str, Any]]:
    rows = []
    for receipt in sorted(source.glob("*/supervisor.json")):
        run = receipt.parent
        supervisor = json.loads(receipt.read_text())
        prompt = json.loads((run / "prompt.json").read_text())
        if prompt["system"] != instructions():
            raise ValueError("stored system instructions differ from the reviewed live supervisor")
        images = sorted((run / "box").glob("image-*.png"))
        if len(images) != len(prompt["images"]) or not images:
            raise ValueError("stored images are missing")
        rows.append(
            {
                "source": str(run),
                "supervisor": supervisor,
                "prompt": prompt,
                "images": [str(p) for p in images],
                "sha256": {
                    str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [receipt, run / "prompt.json", *images]
                },
            }
        )
    decisions = [
        r
        for r in read_table(measurement / "run", "decisions").to_pylist()
        if r["call_status"] == "ok"
    ]

    def key(r: dict[str, Any]) -> tuple[float, str, str]:
        return r["decision_ms"], r["decision"], r["rationale"]

    actual = Counter(key({**r["supervisor"], **r["supervisor"]["answer"]}) for r in rows)
    if len(rows) != 92 or actual != Counter(key(r) for r in decisions):
        raise ValueError("runtime inputs do not match the 92 successful measured calls")
    return rows


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for model, effort in VARIANTS:
        group = [r for r in rows if (r["model"], r["effort"]) == (model, effort)]
        times = [r["decision_ms"] / 1000 for r in group]
        result.append(
            {
                "model": model,
                "effort": effort,
                "calls": len(group),
                "valid": sum(r["valid"] for r in group),
                "agreement": sum(r["valid"] and r["decision"] == r["original"] for r in group),
                "decision_s_median": float(np.median(times)) if times else None,
                "decision_s_p95": float(np.percentile(times, 95)) if times else None,
                "decision_s_total": sum(times),
                "transitions": dict(Counter(f"{r['original']}->{r['decision']}" for r in group)),
            }
        )
    return result


def measure(source: Path, measurement: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError("use a new offline output directory; never silently repeat calls")
    original = inputs(source, measurement)
    out.mkdir(parents=True)
    dump(out / "inputs.json", original)
    # The same gate as the measured live route, including its source/instruction
    # audit and login check. Dynamic SupervisorContext payloads are live inputs,
    # not analysis texts (whose hidden-field scanner rejects "nominal" etc.).
    # Never weaken a gate or alter the recorded payload to make that scan pass.
    gate(["codex"])
    batch = Batch(supervisor_env(), ["codex"], concurrency=2)
    # Rotate order per input to reduce confounding from time on the shared service.
    tasks = [(i, *VARIANTS[(i + k) % 3]) for i in range(92) for k in range(3)]
    dump(
        out / "protocol.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "gate": "fleetlab.live.vlm.measure.gate (unchanged live route)",
            "calls": len(tasks),
            "concurrency": 2,
            "tasks": tasks,
            "timing": "wall from gate recheck through image encoding, CLI and answer validation",
        },
    )

    def call(task: tuple[int, str, str]) -> AgentResult:
        index, model, effort = task
        row = original[index]
        key = f"{index:03d}-{model}-{effort}"
        dump(out / "started" / f"{key}.json", {"source": row["source"]})
        started = time.perf_counter()
        require_verified("codex", supervisor_env())
        with tempfile.TemporaryDirectory(prefix="supervisor-offline-") as directory:
            images = []
            for n, file in enumerate(row["images"]):
                path = Path(directory) / f"image-{n:02d}.png"
                with Image.open(file) as picture:
                    picture.save(path)
                images.append(path)
            response = run_agent(
                "codex",
                model,
                row["prompt"]["system"],
                row["prompt"]["prompt"],
                images=images,
                schema=Answer,
                auth="subscription",
                effort=effort,
                service_tier="fast",
                timeout=120,
                label="supervisor-offline",
            )
        answer = Answer.model_validate(response.result) if response.ok else None
        valid = (
            answer is not None
            and answer.decision.value in json.loads(row["prompt"]["prompt"])["options"]
            and response.model in (None, model)
        )
        elapsed = 1000 * (time.perf_counter() - started)
        model_ms, basis = model_time(response) if response.ok else (None, "unavailable")
        dump(
            out / "calls" / f"{key}.json",
            {
                "index": index,
                "source": row["source"],
                "model": model,
                "reported_model": response.model,
                "effort": effort,
                "original": row["supervisor"]["answer"]["decision"],
                "decision": answer.decision.value if answer else None,
                "valid": valid,
                "decision_ms": elapsed,
                "model_ms": model_ms,
                "model_ms_basis": basis,
                "run_dir": response.run_dir,
                "error": response.error,
            },
        )
        print(
            f"{key}: {answer.decision.value if answer else response.error} {elapsed / 1000:.2f}s",
            flush=True,
        )
        return response

    batch.map(call, tasks)
    rows = [json.loads(p.read_text()) for p in sorted((out / "calls").glob("*.json"))]
    result = {"stopped": batch.stopped, "complete": len(rows) == 276, "table": summary(rows)}
    dump(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if batch.stopped or len(rows) != 276:
        raise RuntimeError("offline comparison incomplete; no automatic retries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("measurement", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    measure(args.source, args.measurement, args.out)


if __name__ == "__main__":
    main()
