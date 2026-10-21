#!/usr/bin/env bash
# One suite, one measured run per task, so that each run holds the heavy-job
# lock for a few minutes only and a stopped run costs one task, not the suite.
#
#   spikes/baseline/run_suite.sh NAME SUITE [lerobot-eval flags...]
#
# Runs NAME_t0 .. NAME_t9 through run.sh (same environment variables: MJ,
# BASELINE_ABS_WORLD, BASELINE_EXPECT_GIB, SEED). A task whose eval_info.json
# already exists is skipped, so the script can simply be started again after an
# interruption. A run that was refused or stopped for GPU memory is retried up
# to five times, one minute apart.
#
# Every task starts from the same seed, so a task's result does not depend on
# which other tasks ran before it.
#
#   TASKS="0 3 5"   run only these task indices (default: 0..9)
set -uo pipefail

name="$1"; suite="$2"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
runs="$HOME/.cache/robots-probes/baseline/runs"

for t in ${TASKS:-0 1 2 3 4 5 6 7 8 9}; do
    run="${name}_t${t}"
    for attempt in 1 2 3 4 5; do
        [ -f "$runs/$run/eval_info.json" ] && break
        "$here/run.sh" "$run" --env.task="$suite" --env.task_ids="[$t]" "$@"
        [ -f "$runs/$run/eval_info.json" ] && break
        echo "$(date +%H:%M:%S) $run attempt $attempt did not finish: $(tr '\r' '\n' <"$runs/$run.log" | grep -a -E 'MEASURE-ABORT|heavy-run: GPU|Error' | tail -1 | cut -c1-200)"
        sleep 60
    done
    if [ -f "$runs/$run/eval_info.json" ]; then
        echo "$(date +%H:%M:%S) $run done"
    else
        echo "$(date +%H:%M:%S) $run GAVE UP"
    fi
done
