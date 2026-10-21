"""Print the summary table of one or more result files again.

    run.sh summarize.py NAME [NAME ...] [--by suite,task] [--total-only]
"""
import argparse
import json

from common import OUT
from evaluate import summarize

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--by", default="suite,task")
ap.add_argument("--total-only", action="store_true")
args = ap.parse_args()
for name in args.names:
    rows = [json.loads(line) for line in open(OUT / "results" / f"{name}.jsonl")]
    for r in rows:
        r.setdefault("kind", (r.get("setup") or {}).get("kind", ""))
    text = summarize(rows, by=tuple(args.by.split(",")))
    skipped = sum(1 for r in rows if r["status"].startswith("skipped"))
    print(f"# {name}: {len(rows)} episodes" + (f", {skipped} skipped because the start state could not be set up" if skipped else ""))
    print(text.splitlines()[-1] if args.total_only else text)
