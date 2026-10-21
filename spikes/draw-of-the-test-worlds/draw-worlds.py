#!/usr/bin/env python3
"""The draw of test (or reserve) worlds: design note, entries 16 and 21.

This script decides nothing. Everything it uses is named on the tracker before
the beacon's round: the checkout and its commit, the round, the ledger's path,
the places and the world ids. It gives the commands of
`src/fleetlab/worlds/README.md`, "Drawing a test world", in their order, and
stops at anything that is not one of the documented outcomes. It can be started
again after a stop: the ledger says where the draw is.

It prints, and appends to `TRUTH/status`, exit codes, durations, the ledger's
summary and head, and the one line of `ledger check`. It opens no file of a
world. What the commands print goes to `TRUTH/logs/`, which nobody on the
development side reads unless the draw has stopped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RELAYS = ("https://api.drand.sh/public/{}", "https://drand.cloudflare.com/public/{}")
BEACON = re.compile(r"drand:default:[1-9][0-9]*:[0-9a-f]{64}")
TRIES, PAUSE = 20, 15  # a relay that does not give the round's value is asked again for five minutes

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=("test", "reserve"), required=True)
parser.add_argument("--round", type=int, required=True, help="the drand round on the tracker")
parser.add_argument("--checkout", type=Path, required=True, help="frozen, clean, at the commit")
parser.add_argument("--truth", type=Path, required=True, help="ledger/, recipes/, logs/, status")
parser.add_argument("--worlds", type=Path, default=Path.home() / ".cache/robots-probes/worlds")
parser.add_argument("--places", nargs="+", required=True, help="in the order they are drawn")
parser.add_argument("--ids", nargs="+", required=True, help="world ids in the order they are used")
parser.add_argument("--fleet-id", default="w01")
parser.add_argument("--workers", type=int, default=6)
args = parser.parse_args()

truth: Path = args.truth.expanduser().resolve()
checkout: Path = args.checkout.expanduser().resolve()
ledger = truth / "ledger" / "ledger.jsonl"
logs = truth / "logs"
for folder in (truth, truth / "ledger", truth / "recipes", logs):
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)


def say(text: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + text
    print(line, flush=True)
    with (truth / "status").open("a") as handle:
        handle.write(line + "\n")


def stop(text: str) -> None:
    say("STOPPED: " + text)
    sys.exit(2)


def command(name: str, argv: list[str]) -> tuple[int, str, str, float]:
    """One command from the checkout; its output goes to the logs, not to the screen."""
    started = time.time()
    done = subprocess.run(argv, cwd=checkout, capture_output=True, text=True)
    seconds = time.time() - started
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(started))
    (logs / f"{stamp}-{name}.out").write_text(done.stdout)
    (logs / f"{stamp}-{name}.err").write_text(done.stderr)
    return done.returncode, done.stdout, done.stderr, seconds


def worlds(name: str, *rest: str) -> tuple[int, str, str, float]:
    return command(name, ["uv", "run", "--locked", "python", "-m", "fleetlab.worlds", *rest])


def summary() -> dict[str, Any]:
    code, out, err, _ = worlds("summary", "ledger", "summary", "--ledger", str(ledger))
    if code != 0:
        stop(f"ledger summary exit {code}: {err.strip().splitlines()[-1:] or out[:200]}")
    return json.loads(last_json(out))


def last_json(text: str) -> str:
    """The JSON a command prints last: the simulator's notices can stand before it (rehearsal, 10-11)."""
    start = text.rfind("\n{")
    return text[start + 1 :] if start >= 0 else text


def head() -> str:
    code, out, _, _ = worlds("head", "ledger", "head", "--ledger", str(ledger))
    if code != 0:
        stop(f"ledger head exit {code}")
    return " ".join(last_json(out).split())


def fetch(url: str) -> str:
    """One relay's value for the round. The value cannot change, so asking again is no choice."""
    for attempt in range(1, TRIES + 1):
        # curl, not urllib: the relays answer 403 to Python's default client (rehearsal, 10-11).
        fetched = subprocess.run(["curl", "-sf", "-m", "30", url], capture_output=True, text=True)
        if fetched.returncode != 0:
            problem = f"curl exit {fetched.returncode}"
        else:
            try:
                data = json.loads(fetched.stdout)
            except ValueError:
                problem = "the answer is not JSON"
            else:
                if isinstance(data, dict) and data.get("round") == args.round:
                    return str(data.get("randomness"))
                problem = "the answer is for another round"
        say(f"{url}: {problem} (try {attempt} of {TRIES})")
        if attempt < TRIES:
            time.sleep(PAUSE)
    stop(f"{url}: no value for the round after {TRIES} tries")
    raise AssertionError


def beacon() -> str:
    """The round's randomness from two relays that have to agree, in the one writing."""
    seen = [fetch(relay.format(args.round)) for relay in RELAYS]
    if len(set(seen)) != 1:
        stop("the two relays give different values for the round")
    text = f"drand:default:{args.round}:{seen[0]}"
    if not BEACON.fullmatch(text):
        stop("the round's value is not 64 lower-case hexadecimal characters")
    return text


code, out, err, _ = command("commit", ["git", "status", "--porcelain"])
if code != 0 or out.strip():
    stop("the checkout is not a clean tree")
commit = command("commit", ["git", "rev-parse", "HEAD"])[1].strip()
say(f"draw of kind {args.kind}: checkout {checkout} at {commit}, round {args.round}, "
    f"places {' '.join(args.places)}, ids {' '.join(args.ids)}, ledger {ledger}")

expected = beacon()
if not ledger.exists():
    code, out, err, _ = worlds("open", "ledger", "open", "--ledger", str(ledger), "--beacon", expected)
    if code != 0:
        stop(f"ledger open exit {code}: {err.strip()[:300]}")
    say(f"ledger opened with {expected}")
state = summary()
if state.get("beacon") != expected:
    stop(f"the ledger's beacon is not the round's value from the relays: {state.get('beacon')!r}")
say("ledger " + head())

for place in args.places:
    while True:
        state = summary()
        if state.get("stopped"):
            stop(f"the ledger says the draw has stopped: {state['stopped']}")
        candidates = state.get("places", {}).get(place, {}).get("candidates", [])
        last = candidates[-1] if candidates else None
        if last and last["state"] == "kept":
            say(f"{place}: filled by {last['world_id']}")
            break
        used = {c["world_id"] for p in state.get("places", {}).values() for c in p["candidates"]}
        if last and last["state"] in ("open", "standing"):
            world_id = last["world_id"]
        else:
            free = [i for i in args.ids if i not in used]
            if not free:
                stop(f"{place}: no world id left on the list")
            world_id = free[0]
        recipe = truth / "recipes" / f"{place}-{world_id}.json"
        root = args.worlds.expanduser() / world_id
        if not recipe.exists():
            extra = ["--kind", "reserve"] if args.kind == "reserve" else []
            code, out, err, _ = worlds(
                f"{place}-{world_id}-recipe-file", "recipe-file", "--world-id", world_id,
                "--public-fleet-id", args.fleet_id, "--name", place, *extra, "--out", str(recipe),
            )
            if code != 0:
                stop(f"{place} {world_id}: recipe-file exit {code}: {err.strip()[:300]}")
        if not (last and last["state"] == "standing"):
            failed = 0
            while True:
                code, out, err, seconds = worlds(
                    f"{place}-{world_id}-generate", "generate", "--recipe", str(recipe),
                    "--root", str(root), "--ledger", str(ledger), "--place", place,
                )
                say(f"{place} {world_id}: draw exit {code} after {seconds:.0f} s; ledger {head()}")
                if code == 0:
                    break
                if code == 3:
                    continue
                reason = (err.strip().splitlines() or [""])[-1][:300]
                if code == 2 and "after the draw" in reason and failed == 0:
                    failed = 1
                    say(f"{place} {world_id}: the attempt failed after the draw; the same command once more")
                    continue
                stop(f"{place} {world_id}: generate exit {code}: {reason}")
        code, out, err, seconds = command(
            f"{place}-{world_id}-run",
            ["make", "world", f"WORLD=--recipe {recipe} --root {root}",
             f"WORLD_FLAGS=--workers {args.workers}"],
        )
        say(f"{place} {world_id}: run exit {code} after {seconds / 60:.0f} min")
        if code != 0:
            stop(f"{place} {world_id}: the run did not exit 0")
        code, out, err, _ = worlds(
            f"{place}-{world_id}-check", "ledger", "check", "--ledger", str(ledger), "--place", place
        )
        said = " ".join(last_json(out).split())[:400]
        say(f"{place} {world_id}: check exit {code}: {said}; ledger {head()}")
        if code == 0:
            break
        if code != 4:
            stop(f"{place} {world_id}: ledger check exit {code}: {err.strip()[:300]}")

code, out, err, seconds = worlds("verify", "ledger", "verify", "--ledger", str(ledger))
say(f"ledger verify exit {code} after {seconds:.0f} s: {' '.join(last_json(out).split())[:400]}")
say("final summary " + json.dumps(summary(), sort_keys=True))
say("final ledger " + head())
sys.exit(0 if code == 0 else 2)
