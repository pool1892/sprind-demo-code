#!/usr/bin/env bash
# Create a frozen checkout and its own disk-backed environment, then run the grid.
set -euo pipefail
if [[ $# != 3 && $# != 5 ]]; then
    echo 'usage: run_grid.sh REVISION WORLD NEW_OUT [SUPERVISOR_SPEC FACTORY]' >&2
    exit 2
fi
revision=$(git rev-parse "$1^{commit}")
world=$(realpath -- "$2")
result=$(realpath -m -- "$3")
checkout="$result/checkout"
if [[ ! -d $checkout ]]; then
    mkdir -p "$result"
    git worktree add --detach "$checkout" "$revision"
fi
[[ $(git -C "$checkout" rev-parse HEAD) == "$revision" ]]
[[ -z $(git -C "$checkout" status --porcelain --untracked-files=no) ]]
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export TMPDIR="${TMPDIR:-/var/tmp/robots-live}"
export ROBOTS_AGENT_ROOT="${ROBOTS_AGENT_ROOT:-/var/tmp/robots-stage5-grid-$(id -u)}"
export FLEETLAB_MIOPEN_DIR="${FLEETLAB_MIOPEN_DIR:-$HOME/.cache/robots-probes/engine/miopen}"
mkdir -p "$TMPDIR"
args=("$world" "$result/grid")
if [[ $# == 5 ]]; then
    source_root=$(git rev-parse --show-toplevel)
    caller_profile=$(realpath -m -- "$4")
    if [[ $caller_profile != "$source_root/"* ]]; then
        echo 'SUPERVISOR_SPEC must name a tracked file inside this repository.' >&2
        exit 2
    fi
    relative_profile=${caller_profile#"$source_root/"}
    git -C "$checkout" --literal-pathspecs ls-files --error-unmatch \
        -- "$relative_profile" > /dev/null
    frozen_profile=$(realpath -- "$checkout/$relative_profile")
    if [[ $frozen_profile != "$checkout/"* || ! -f $frozen_profile ]]; then
        echo 'SUPERVISOR_SPEC must resolve to a file inside the frozen checkout.' >&2
        exit 2
    fi
    args+=("$frozen_profile" "$5")
fi
cd "$checkout"
make setup
if [[ ${GRID_DRY_START:-0} == 1 ]]; then
    exec bash src/fleetlab/live/grid.sh "${args[@]}"
fi
exec tools/heavy-run bash src/fleetlab/live/grid.sh "${args[@]}"
