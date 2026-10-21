#!/bin/sh
# The measurement campaign behind README.md: spikes/teleop-absolute/campaign.sh [STEP...]
#
# Runs from a frozen copy of this directory at the current commit, so the
# directory must have no uncommitted changes. Results land in
# ~/.cache/robots-probes/teleop-absolute/results/c_*.jsonl, logs in .../logs/.
# Steps (default: all, in this order): init_abs init_rel kinds inverted xvla_end xvla_mid xvla_mid_rel
# Each batch is one light job with four worker processes.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out="$HOME/.cache/robots-probes/teleop-absolute"
if [ -n "$(git -C "$root" status --porcelain -- spikes/teleop-absolute)" ]; then
    echo "spikes/teleop-absolute has uncommitted changes; commit first" >&2
    exit 1
fi
commit=$(git -C "$root" rev-parse --short HEAD)
code="$out/code/$commit"
if [ ! -d "$code" ]; then
    mkdir -p "$out/code"
    git -C "$root" archive "$commit" spikes/teleop-absolute | tar -x -C "$out/code" --one-top-level="$commit.tmp"
    mv "$out/code/$commit.tmp/spikes/teleop-absolute" "$code"
    rm -r "$out/code/$commit.tmp"
fi
mkdir -p "$out/logs"
handovers="$out/handovers"

run() {
    name="$1"; shift
    echo "$(date '+%F %T') start $name ($commit)" >> "$out/logs/campaign.log"
    if SPIKE_CODE="$code" "$here/run.sh" evaluate.py --name "$name" "$@" > "$out/logs/$name.log" 2>&1; then
        echo "$(date '+%F %T') done  $name" >> "$out/logs/campaign.log"
    else
        echo "$(date '+%F %T') FAILED $name" >> "$out/logs/campaign.log"
    fi
}

steps="${*:-init_abs init_rel kinds inverted xvla_end xvla_mid xvla_mid_rel}"
for step in $steps; do
    case "$step" in
        # LIBERO's 50 stored start states of all 20 tasks, absolute control
        init_abs) run c_init_abs --scenario init --inits 0-49 ;;
        # the first ten of them per task in relative control, the fallback
        init_rel) run c_init_rel --scenario init --inits 0-9 --mode relative ;;
        # the six kinds of disturbed state, ten start states per task, through a snapshot
        kinds) for k in displaced dropped on_side closed_empty held awkward; do
                   run "c_$k" --scenario "$k" --inits 0-9
               done ;;
        # the bowl upside down, five start states per spatial task
        inverted) run c_inverted --scenario inverted --suites libero_spatial --inits 0-4 ;;
        # X-VLA's recorded failures, handed over after their 280 steps (xvla_select.py)
        xvla_end) run c_xvla_end --scenario xvla --handovers "$handovers/xvla_end.jsonl" ;;
        # X-VLA episodes handed over in the middle of their motion
        xvla_mid) run c_xvla_mid --scenario xvla --handovers "$handovers/xvla_mid.jsonl" ;;
        # the same hand-overs into relative control: the mode switched in mid-episode
        xvla_mid_rel) run c_xvla_mid_rel --scenario xvla --handovers "$handovers/xvla_mid.jsonl" --mode relative ;;
        *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
