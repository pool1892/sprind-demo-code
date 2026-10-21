"""Host-only repeat reservations and analysis cost accounting; no evaluator calls."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import time
from pathlib import Path
from typing import Any

from fleetlab.agents.audit import tree_manifest

# Snapshot of https://developers.openai.com/api/docs/pricing, checked 2026-10-10.
# USD per million tokens, Fast processing, short context. These are API-price
# equivalents, not charges against a ChatGPT subscription. Long-context and
# cache-write uncertainty is retained as a range when the CLI only reports totals.
PRICES = {
    "gpt-6-astra": {"input": 20.0, "cached_input": 2.0, "output": 100.0},
    "gpt-6.1-sol": {"input": 4.0, "cached_input": 0.2, "output": 20.0},
    "gpt-6-luna": {"input": 0.2, "cached_input": 0.02, "output": 1.0},
}
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"


def library_revision(root: Path) -> str:
    return hashlib.sha256(json.dumps(tree_manifest(root), sort_keys=True).encode()).hexdigest()


# Independent of ROBOTS_AGENT_ROOT, HOME, XDG_CACHE_HOME and output paths. This
# registry is host-only; no CLI flag or environment variable selects a new one.
REPEAT_REGISTER = (
    Path(pwd.getpwuid(os.getuid()).pw_dir) / ".cache/robots-probes/analysis-repeats-v2"
)


def repeat_group(
    world_hash: str, condition: str, library: str, model: str, effort: str
) -> dict[str, str]:
    return {
        "world_hash": world_hash,
        "condition": condition,
        "library_revision": library,
        "model": model,
        "effort": effort,
    }


def _repeat_key(group: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()


def reserve_repeat(
    group: dict[str, str],
    output: Path,
    first_call: Path,
) -> dict[str, Any]:
    """Reserve when the first model process starts, after all launch checks.

    The receipt is immutable and survives failed calls and continuations. A
    scoring reader can verify the group, ordinal, receipt bytes and call path.
    """
    key = _repeat_key(group)
    root = REPEAT_REGISTER / key
    root.mkdir(parents=True, exist_ok=True)
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        number = max((int(p.stem) for p in root.glob("*.json")), default=0) + 1
        path = root / f"{number:06d}.json"
        receipt = {
            **group,
            "format": "fleetlab.repeat/2",
            "key": key,
            "repeat": number,
            "output": str(output.resolve()),
            "first_call": str(first_call.resolve()),
            "started_at": time.time(),
        }
        with path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return {
        **receipt,
        "reservation": str(path),
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def verify_repeat(receipt: dict[str, Any], group: dict[str, str]) -> None:
    """Reject relabelled or caller-selected reservations, including on resume."""
    key = _repeat_key(group)
    path = REPEAT_REGISTER / key / f"{receipt['repeat']:06d}.json"
    if receipt.get("reservation") != str(path):
        raise ValueError("repeat reservation is not in the fixed register")
    data = path.read_bytes()
    saved = json.loads(data)
    expected = {k: v for k, v in receipt.items() if k not in ("reservation", "receipt_sha256")}
    if (
        hashlib.sha256(data).hexdigest() != receipt.get("receipt_sha256")
        or saved != expected
        or saved.get("key") != key
        or any(saved.get(k) != value for k, value in group.items())
    ):
        raise ValueError("repeat reservation differs from the registered first call")


def price_equivalent(usage: dict[str, Any], prices: dict[str, float]) -> dict[str, Any]:
    """A transparent bound from CLI totals, never an invented exact API bill."""
    keys = ("input_tokens", "cached_input_tokens", "output_tokens")
    if any(usage.get(key) is None for key in keys):
        return {"status": "usage_unavailable", "low_usd": None, "high_usd": None}
    total, cached, output = (int(usage[key]) for key in keys)
    if min(total, cached, output) < 0 or cached > total:
        raise ValueError("invalid token accounting")
    fresh = total - cached
    low = (
        fresh * prices["input"] + cached * prices["cached_input"] + output * prices["output"]
    ) / 1e6
    high = (
        fresh * prices["input"] * 2.5
        + cached * prices["cached_input"] * 2
        + output * prices["output"] * 1.5
    ) / 1e6
    return {
        "status": "bounded_api_equivalent",
        "low_usd": low,
        "high_usd": high,
        "basis": "Fast API rates; CLI totals omit per-request context tier and cache writes. "
        "Lower: short context, no cache-write premium. Upper: all long context, "
        "all fresh input charged as cache writes. Reasoning is included in output.",
    }


def collect_accounting(output: Path, metadata: dict[str, Any], episodes: int) -> dict[str, Any]:
    """Include failed CLI attempts and prior continuation segments exactly once."""
    roots: list[Path] = []
    records: dict[Path, dict[str, Any]] = {}
    seen: set[Path] = set()
    root = output
    while root not in seen:
        seen.add(root)
        roots.append(root)
        record = metadata if root == output else json.loads((root / "run.json").read_text())
        records[root] = record
        parent = record.get("resume_from")
        if not parent:
            break
        root = Path(parent)
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    calls = []
    audit_seconds = 0.0
    for root in roots:
        directories = {
            p.parent
            for name in ("result.json", "prompt.json")
            for p in (root / "agent-runs").rglob(name)
        }
        for directory in sorted(directories):
            path = directory / "result.json"
            result: dict[str, Any] = (
                json.loads(path.read_text())
                if path.is_file()
                else {"error": "missing_result", "seconds": None}
            )
            used = result.get("usage", {})
            for key in usage:
                usage[key] += int(used.get(key) or 0)
            calls.append(
                {
                    "run_dir": str(directory),
                    "model": records[root]["model"],
                    "reported_model": result.get("model"),
                    "usage": used,
                    "seconds": result.get("seconds", 0),
                    "error": result.get("error"),
                    "price": price_equivalent(used, records[root]["prices"]),
                }
            )
        for path in (root / "agent-runs").rglob("audit-time.json"):
            audit_seconds += json.loads(path.read_text())["seconds"]
    # Copied engine/queue history carries paths of its original timings. Every
    # queue timing has a unique dispatch id, so continuation copies don't double it.
    queues: dict[str, dict[str, Any]] = {}
    for root in reversed(roots):
        for path in root.glob("*/queues/*.timing.json"):
            record = json.loads(path.read_text())
            queues[record["dispatch_id"]] = record
    cost: dict[str, Any] = {
        "status": "bounded_api_equivalent",
        "low_usd": sum(c["price"]["low_usd"] or 0 for c in calls),
        "high_usd": sum(c["price"]["high_usd"] or 0 for c in calls),
        "basis": "Sum of per-call bounds using each segment's recorded price snapshot.",
    }
    if any(c["price"]["status"] == "usage_unavailable" for c in calls):
        cost = {"status": "incomplete_usage", "low_usd": None, "high_usd": None}
    cost["per_1000_observed_episodes"] = {
        key: cost[key] * 1000 / episodes if cost[key] is not None and episodes else None
        for key in ("low_usd", "high_usd")
    }

    def total_time(key: str) -> float | None:
        values = [q.get(key) for q in queues.values()]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values if value is not None)

    return {
        "model": metadata["model"],
        "usage": usage,
        "calls": calls,
        "model_seconds": (
            sum(c["seconds"] for c in calls)
            if all(c["seconds"] is not None for c in calls)
            else None
        ),
        "launch_audit_seconds": audit_seconds,
        "queue_seconds": total_time("queue_seconds"),
        "simulation_seconds": total_time("simulation_seconds"),
        "queue_audit_seconds": sum(q.get("audit_seconds", 0) for q in queues.values()),
        "queue_attempts": list(queues.values()),
        "price": cost,
        "observed_episodes": episodes,
        "wall_seconds": metadata["ended"] - metadata["started"],
    }
