#!/usr/bin/env bash
# One job of this spike behind the machine's heavy-job lock (or as a light job).
#
#   spikes/pi05/run.sh [--light] NAME SCRIPT.py [ARGS...]
#
# Runs SCRIPT.py from this directory with this spike's Python
# (~/.cache/robots-probes/pi05/venv), writes the full output to
# ~/.cache/robots-probes/pi05/logs/NAME.log and prints it without the simulator's
# start-up chatter. Environment variables such as MIOPEN_DB are passed through.
# The exit status is the job's own. Copied from spikes/xvla-speed/run.sh.
set -euo pipefail

light=()
if [ "${1:-}" = "--light" ]; then light=(--light); shift; fi
name="$1"; script="$2"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
base="$HOME/.cache/robots-probes/pi05"
py="$base/venv/bin/python"
mkdir -p "$base/logs"
export PYTHONUNBUFFERED=1
cd "$repo"
echo "# $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg) gtt $(awk '{printf "%.1f", $1/2^30}' /sys/class/drm/card1/device/mem_info_gtt_used) GiB: $script $*" >"$base/logs/$name.log"
status=0
tools/heavy-run "${light[@]}" "$py" "$here/$script" "$@" </dev/null >>"$base/logs/$name.log" 2>&1 || status=$?
grep -v "WARN\|robosuite\|macro\|Assets\|Local assets\|task orders\|Loading weights\|torch_dtype\|unauthenticated\|^$" \
    "$base/logs/$name.log" | tail -n "${P5_TAIL:-60}"
echo "# exit status $status, load $(cut -d' ' -f1-3 /proc/loadavg)"
exit $status
