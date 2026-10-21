#!/usr/bin/env bash
# The policy-side comparison: X-VLA keeps acting for TAIL steps after LIBERO's check, then
# every definition is scored on the replayed states. Needs the GPU (heavy job).
#
#   spikes/success/policy_tail.sh [TAIL]        # default 60
#
# 1. Three fleet runs with X-VLA in the recommended 16-bit setting, the manifests of
#    spikes/xvla-speed (healthy200: libero_spatial; object200: libero_object; mix200:
#    the faulted mix), through fleet_tail.py, one after the other behind the heavy lock.
# 2. A light replay of their successful episodes (bit for bit, checked through the hashes).
# 3. The tables (analyze.py --policy-run) to stdout and analysis_*.json.
#
# Writes ~/.cache/robots-probes/success/policy/<kind>_tail<TAIL>/ and refuses to overwrite.
set -euo pipefail

tail_steps="${1:-60}"
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
base="$HOME/.cache/robots-probes/success"
manifests="$HOME/.cache/robots-probes/xvla-speed/manifests"
py="$HOME/.cache/robots-probes/baseline-venv-mj337/bin/python"
mkdir -p "$base/policy" "$base/logs"
cd "$repo"
runs=()
for kind in healthy200 object200 mix200; do
    out="$base/policy/${kind}_tail${tail_steps}"
    if [ -e "$out" ]; then
        echo "run directory $out exists already; remove it or choose another tail" >&2
        exit 2
    fi
    echo "# $(date '+%F %T') $kind, tail $tail_steps" | tee "$base/logs/${kind}_tail${tail_steps}.log"
    PYTHONUNBUFFERED=1 tools/heavy-run "$py" "$here/fleet_tail.py" --manifest "$manifests/$kind.jsonl" --out "$out" \
        --rows 4 --workers 10 --policy fp16_res32 --success-tail "$tail_steps" \
        </dev/null >>"$base/logs/${kind}_tail${tail_steps}.log" 2>&1
    tail -n 3 "$base/logs/${kind}_tail${tail_steps}.log"
    runs+=(--run "$out")
done
"$here/run.sh" --baseline "replay_tail${tail_steps}" replay_policy.py --name "tail${tail_steps}" --tail 0 \
    --only-success --workers 4 "${runs[@]}"
"$repo/.venv/bin/python" "$here/analyze.py" --expert expert1000 --policy-run "tail${tail_steps}" --window 40
