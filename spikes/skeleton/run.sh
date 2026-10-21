#!/usr/bin/env bash
# One fleet run behind the machine's heavy-job lock.
#
#   spikes/skeleton/run.sh NAME MANIFEST [fleet.py flags...]
#
# Writes ~/.cache/robots-probes/skeleton/runs/NAME/ and logs/NAME.log.
# The exit status is the fleet runner's own.
set -euo pipefail

name="$1"; manifest="$2"; shift 2
repo="$(cd "$(dirname "$0")/../.." && pwd)"
base="$HOME/.cache/robots-probes/skeleton"
py="$HOME/.cache/robots-probes/baseline-venv-mj337/bin/python"
out="$base/runs/$name"
mkdir -p "$base/logs" "$base/runs"
if [ -e "$out" ]; then
    echo "run directory $out exists already; choose another name" >&2
    exit 2
fi
export PYTHONUNBUFFERED=1
cd "$repo"
status=0
tools/heavy-run "$py" "$repo/spikes/skeleton/fleet.py" --manifest "$manifest" --out "$out" "$@" \
    </dev/null >"$base/logs/$name.log" 2>&1 || status=$?
grep -v "WARN\|robosuite\|macro\|Assets\|Local assets\|task orders\|Loading weights\|torch_dtype\|unauthenticated" \
    "$base/logs/$name.log" | tail -n 12
exit $status
