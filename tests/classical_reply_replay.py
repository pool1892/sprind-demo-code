"""Replay stored visible arms without fitting, scoring, simulation or model calls.

Run with either source tree on PYTHONPATH to compare reply-contract versions.
The analysis directory must contain contract-valid results for that version.
"""

import argparse
import hashlib
import json
from pathlib import Path

from fleetlab.baselines.classical.experiments import read_answer, update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("logs_answer", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    update(args.export, read_answer(args.logs_answer), args.analysis).write(args.out)
    hashes = {
        str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.out.rglob("*.json"))
    }
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
