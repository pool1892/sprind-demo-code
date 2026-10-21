#!/usr/bin/env bash
# One job of this spike as a light job (no model, at most four simulator processes).
#
#   spikes/success/run.sh [--baseline] LOGNAME SCRIPT.py [ARGS...]
#
# Runs SCRIPT.py from this directory with this checkout's environment (.venv, built by
# `make setup`), or with --baseline in the environment the X-VLA runs were recorded in
# (~/.cache/robots-probes/baseline-venv-mj337), which replay_policy.py needs to reproduce
# them bit for bit. Full output goes to ~/.cache/robots-probes/success/logs/LOGNAME.log;
# the tail is printed without the simulator's start-up chatter. The exit status is the job's.
set -euo pipefail

py_kind=repo
if [ "${1:-}" = "--baseline" ]; then py_kind=baseline; shift; fi
name="$1"; script="$2"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
base="$HOME/.cache/robots-probes/success"
if [ "$py_kind" = baseline ]; then
    py="$HOME/.cache/robots-probes/baseline-venv-mj337/bin/python"
else
    py="$repo/.venv/bin/python"
fi
mkdir -p "$base/logs"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$here"
cd "$repo"
echo "# $(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg): $py_kind $script $*" >"$base/logs/$name.log"
status=0
tools/heavy-run --light -m "${SUCCESS_MEM:-16G}" "$py" "$here/$script" "$@" </dev/null >>"$base/logs/$name.log" 2>&1 || status=$?
grep -v "WARN\|robosuite\|macro\|Assets\|Local assets\|task orders\|Loading weights\|torch_dtype\|unauthenticated\|^$\|/\[_\]\|\[~\]\|SyntaxWarning\|___\|^ *|" \
    "$base/logs/$name.log" | tail -n "${SUCCESS_TAIL:-40}"
echo "# exit status $status, load $(cut -d' ' -f1-3 /proc/loadavg)"
exit $status
