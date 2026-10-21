"""uv run python -m fleetlab.baselines.classical EXPORT --out NEW_DIRECTORY"""

import argparse
from pathlib import Path

from fleetlab.baselines.classical import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--catalogue", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new result directory")
    answer = run(args.export, catalogue=args.catalogue, bootstraps=args.bootstraps, seed=args.seed)
    answer.write(args.out)
    print(
        f"{len(answer.diagnosis.causes)} causes; {len(answer.fix_set.actions)} ranked actions; "
        f"four budget cuts written to {args.out}"
    )


if __name__ == "__main__":
    main()
