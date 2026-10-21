"""Read-only preflight and comparison receipts for the two deferred GPU commands.

Uses the existing pilot and nightly outputs; does not run or score a new experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from fleetlab.engine import manifests
from fleetlab.engine.ladder import compare_nights
from fleetlab.live.vlm.measure import dump


def check_sample(sample: Path) -> None:
    selection = json.loads((sample / "selection.json").read_text())
    manifest = sample / "sample.jsonl"
    jobs = manifests.read(manifest)
    ids = [j.identity.episode_id for j in jobs]
    baseline = selection["episodes"]
    if (
        len(ids) != 80
        or len(set(ids)) != 80
        or set(ids) != set(baseline)
        or sum(not v["success"] for v in baseline.values()) != 60
        or sum(bool(v["success"]) for v in baseline.values()) != 20
        or hashlib.sha256(manifest.read_bytes()).hexdigest() != selection["sample_sha256"]
    ):
        raise ValueError("repeat D2 requires the unchanged frozen 60-failure/20-success sample")


def check_software(sample: Path) -> dict[str, str]:
    """Validate the actual pinned database before occupying a GPU queue slot."""
    from fleetlab.engine import miopen, software

    jobs = manifests.read(sample / "sample.jsonl")
    hashes = set()
    for job in jobs:
        identity = job.identity
        digest = miopen.require(identity.batch_size, identity.policy.precision)
        if identity.software != software.current(digest):
            raise ValueError("frozen D2 identities do not match this software/database")
        hashes.add(digest)
    if len(hashes) != 1:
        raise ValueError("sample databases differ")
    return {"miopen_directory": str(miopen.root()), "miopen_hash": hashes.pop()}


def night_settings(before: Path) -> tuple[str, int]:
    date = json.loads((before / "scores.json").read_text())["date"]
    per_day = {
        json.loads((before / world / "files/spec.json").read_text())["episodes_per_robot_day"]
        for world in ("deterministic", "randomized")
    }
    if len(per_day) != 1:
        raise ValueError("nightly worlds have different day lengths")
    return date, per_day.pop()


def compare_reference(before: Path, after: Path) -> dict[str, Any]:
    """Write a complete candidate-reference inventory; refuse unexpected unsupervised drift."""
    old_file, new_file = before / "scores.json", after / "scores.json"
    old, new = json.loads(old_file.read_text()), json.loads(new_file.read_text())
    if old["date"] != new["date"]:
        raise ValueError("reference refresh must retain the date-seeded randomized world")
    comparisons = {}
    for world in ("deterministic", "randomized"):
        for run in ("world", "live"):
            a, b = old[world]["digests"][run], new[world]["digests"][run]
            keys = sorted(set(a) | set(b))
            comparisons[f"{world}/{run}"] = {
                "compared": len(keys),
                "identical": sum(a.get(k) == b.get(k) for k in keys),
                "changed": {
                    k: {"before": a.get(k), "after": b.get(k)} for k in keys if a.get(k) != b.get(k)
                },
            }
    checks_ok = all(
        new[world]["runs"][run]["chain_ok"]
        for world in ("deterministic", "randomized")
        for run in ("world", "live")
    )
    canary = new["canary"]
    expected = (
        checks_ok
        and canary["compared"] > 0
        and canary["identical"] == canary["compared"]
        and all(
            not comparisons[f"{world}/world"]["changed"]
            for world in ("deterministic", "randomized")
        )
    )
    receipt = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "before": str(old_file),
        "before_sha256": hashlib.sha256(old_file.read_bytes()).hexdigest(),
        "after": str(new_file),
        "after_sha256": hashlib.sha256(new_file.read_bytes()).hexdigest(),
        "checks_ok": checks_ok,
        "only_live_chains_changed": expected,
        "comparisons": comparisons,
        "canary": canary,
    }
    dump(after / "reference-transition.json", receipt)
    existing = compare_nights(new, old, before)
    dump(after / "reference-comparison.json", existing["json"])
    (after / "reference-comparison.md").write_text(existing["markdown"])
    if not expected:
        raise RuntimeError("reference refresh failed: unsupervised drift or replay/canary failure")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("path", type=Path)
    software = sub.add_parser("software")
    software.add_argument("path", type=Path)
    settings = sub.add_parser("night-settings")
    settings.add_argument("path", type=Path)
    compare = sub.add_parser("night-compare")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        check_sample(args.path)
    elif args.command == "software":
        print(json.dumps(check_software(args.path)))
    elif args.command == "night-settings":
        print(*night_settings(args.path))
    else:
        compare_reference(args.before, args.after)


if __name__ == "__main__":
    main()
