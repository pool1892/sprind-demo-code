#!/usr/bin/env bash
# Paired experiment. --dry-start reaches both GPU launch boundaries without loading a model.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo 'usage: bash src/fleetlab/live/vlm/repeat_d2.sh COMMIT FROZEN_SAMPLE NEW_OUTPUT [SETTINGS_JSON] [--dry-start]' >&2
    exit 2
fi
expected=$(git rev-parse "$1^{commit}")
[[ $(git rev-parse HEAD) == "$expected" ]] || { echo 'wrong measured commit' >&2; exit 2; }
[[ $(git symbolic-ref -q HEAD || true) == '' ]] || { echo 'use a detached run checkout' >&2; exit 2; }
[[ -z $(git status --porcelain --untracked-files=all -- src skills) ]] || { echo 'measured source is modified' >&2; exit 2; }
sample=$(realpath "$2")
out=$(realpath -m "$3")
[[ -f "$sample/sample.jsonl" && -f "$sample/selection.json" ]] || exit 2
[[ ! -e "$out" ]] || { echo 'preserve prior results: output must be new' >&2; exit 2; }
uv run --locked python -m fleetlab.live.vlm.deferred sample "$sample"
settings=()
dry=()
for argument in "${@:4}"; do
    if [[ "$argument" == --dry-start ]]; then dry=(--dry-start)
    else settings=(--settings "$(realpath "$argument")"); fi
done
# Independent of result placement: sandbox assets must be outside home/repository.
export ROBOTS_AGENT_ROOT="${ROBOTS_AGENT_ROOT:-/var/tmp/robots-supervisor-repeat-$(id -u)}"
export FLEETLAB_MIOPEN_DIR="${FLEETLAB_MIOPEN_DIR:-$HOME/.cache/robots-probes/engine/miopen}"
uv run --locked python -m fleetlab.live.vlm.deferred software "$sample"
mkdir -p "$out"
printf '%s\n' "$ROBOTS_AGENT_ROOT" > "$out/agent-runtime-root.txt"
# Free checks before waiting for either GPU arm (also on a dry start).
uv run --locked python -m fleetlab.live.vlm.measure gate codex
launcher=(tools/heavy-run)
if [[ ${#dry[@]} != 0 ]]; then launcher=(); fi
# Same frozen identities/outcomes in both arms; no re-selection or model tuning.
"${launcher[@]}" uv run --locked python -m fleetlab.live.vlm.measure run \
    "$sample" "$out/timeout" --method timeout --workers 2 "${dry[@]}"
"${launcher[@]}" uv run --locked python -m fleetlab.live.vlm.measure run \
    "$sample" "$out/vlm" --method vlm --backend codex --workers 2 --effort xhigh "${settings[@]}" "${dry[@]}"
