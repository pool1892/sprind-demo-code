#!/usr/bin/env bash
# One fleet run behind the machine's heavy-job lock.
#
#   spikes/xvla-speed/fleet.sh NAME MANIFEST_KIND --rows N --workers W --policy P [fleet.py flags...]
#
# MANIFEST_KIND is one of manifests.py's kinds (spatial100, healthy200, mix200, smoke).
# Writes ~/.cache/robots-probes/xvla-speed/runs/NAME/ and logs/NAME.log, and refuses
# to overwrite a run. The exit status is the fleet runner's own.
set -euo pipefail

name="$1"; kind="$2"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
base="$HOME/.cache/robots-probes/xvla-speed"
py="$HOME/.cache/robots-probes/baseline-venv-mj337/bin/python"
out="$base/runs/$name"
mkdir -p "$base/logs" "$base/runs" "$base/manifests"
if [ -e "$out" ]; then
    echo "run directory $out exists already; choose another name" >&2
    exit 2
fi
[ -e "$base/manifests/$kind.jsonl" ] || "$py" "$here/manifests.py" "$kind" "$base/manifests/$kind.jsonl"
export PYTHONUNBUFFERED=1
cd "$repo"
echo "# $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg): fleet.py $kind $*" >"$base/logs/$name.log"
status=0
tools/heavy-run "$py" "$here/fleet.py" --manifest "$base/manifests/$kind.jsonl" --out "$out" "$@" \
    </dev/null >>"$base/logs/$name.log" 2>&1 || status=$?
grep -v "WARN\|robosuite\|macro\|Assets\|Local assets\|task orders\|Loading weights\|torch_dtype\|unauthenticated\|Warning\|warn(\|^$" \
    "$base/logs/$name.log" | tail -n "${XS_TAIL:-9}"
echo "# exit status $status"
exit $status
