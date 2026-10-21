"""Print the event logs of selected episodes from a results file.

    run.sh show.py NAME [--suite S] [--task T] [--failed] [--retried] [--limit N]
"""
import argparse
import json

from common import OUT

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--suite")
ap.add_argument("--task", type=int)
ap.add_argument("--failed", action="store_true")
ap.add_argument("--retried", action="store_true")
ap.add_argument("--limit", type=int, default=10)
args = ap.parse_args()
n = 0
for line in open(OUT / "results" / f"{args.name}.jsonl"):
    r = json.loads(line)
    if args.suite and r["suite"] != args.suite:
        continue
    if args.task is not None and r["task"] != args.task:
        continue
    if args.failed and r["success"]:
        continue
    if args.retried and r.get("attempts", 0) <= 1:
        continue
    print(f"== {r['suite']} task {r['task']} init {r['init']} {r['scenario']} {r['fault']}: {r['status']} steps {r['steps']} attempts {r['attempts']} setup {r.get('setup')}")
    for e in r.get("events", []):
        print("    ", e)
    if r.get("error"):
        print("    ERROR", r["error"][-600:])
    n += 1
    if n >= args.limit:
        break
