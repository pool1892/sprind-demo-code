#!/usr/bin/env bash
# Deferred GPU run. A fresh candidate reference; never overwrite the earlier night's record.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo 'usage: bash src/fleetlab/live/vlm/refresh_nightly.sh COMMIT PREVIOUS_NIGHT NEW_ROOT' >&2
    exit 2
fi
expected=$(git rev-parse "$1^{commit}")
[[ $(git rev-parse HEAD) == "$expected" ]] || { echo 'wrong measured commit' >&2; exit 2; }
[[ $(git symbolic-ref -q HEAD || true) == '' ]] || { echo 'use a detached run checkout' >&2; exit 2; }
[[ -z $(git status --porcelain --untracked-files=all -- src skills) ]] || { echo 'measured source is modified' >&2; exit 2; }
before=$(realpath "$2")
out=$(realpath -m "$3")
[[ ! -e "$out" ]] || { echo 'preserve prior results: output must be new' >&2; exit 2; }
settings=$(uv run --locked python -m fleetlab.live.vlm.deferred night-settings "$before")
read -r date per_day <<< "$settings"
tools/heavy-run uv run --locked python -m fleetlab.engine.ladder nightly \
    --root "$out" --date "$date" --per-day "$per_day" --workers 4 \
    --miopen-dir /home/cs/.cache/robots-probes/engine/miopen
uv run --locked python -m fleetlab.live.vlm.deferred night-compare "$before" "$out/$date"
