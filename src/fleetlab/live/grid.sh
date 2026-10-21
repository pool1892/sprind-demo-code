#!/usr/bin/env bash
# WORLD OUT [SUPERVISOR_SPEC SUPERVISOR_FACTORY]; run in one heavy scope.
set -euo pipefail
if [[ $# != 2 && $# != 4 ]]; then
    echo 'usage: grid.sh WORLD OUT [SUPERVISOR_SPEC SUPERVISOR_FACTORY]' >&2
    exit 2
fi
world=$(realpath -- "$1")
result=$(realpath -m -- "$2")
mode=${GRID_MODE:-development}
workers=${GRID_WORKERS:-2}
operators=(2 6)
timeouts=(1 174 240)
ratios=(1 2 4)
if [[ $mode == test ]]; then
    operators=(2)
    timeouts=(1 240)
    ratios=(4)
elif [[ $mode != development ]]; then
    echo 'GRID_MODE must be development or test' >&2
    exit 2
fi
mkdir -p "$result/logs"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$result/plot-cache}"
export TMPDIR="${TMPDIR:-/var/tmp/robots-live}"
mkdir -p "$TMPDIR"
uv run --locked python -m fleetlab.live.pass_cli cohort "$world" > "$result/cohort-preview.json"
uv run --locked python - "$result/cohort-preview.json" "$mode" "$#" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
n = p['window_failures'] + p['sampled_successes']
passes = (10 if sys.argv[3] == '4' else 4) if sys.argv[2] == 'test' else (17 if sys.argv[3] == '4' else 11)
print(f'{passes} full-cohort passes, at most {n * passes} attempts; snapshot continuations are additional.')
if sys.argv[2] == 'test' and n * passes > 1400:
    print('Budget estimate exceeded openly; full cohort retained (test-grid budget decision).')
PY
if [[ $# == 4 ]]; then
    dry_args=()
    [[ ${GRID_DRY_START:-0} == 1 ]] && dry_args=(--dry-start)
    uv run --locked python -m fleetlab.live.preflight "$world" "$result" \
        --supervisor "$3" --factory "$4" --workers "$workers" --mode "$mode" \
        "${dry_args[@]}"
fi
if [[ ${GRID_DRY_START:-0} == 1 ]]; then
    first="timeout at step ${timeouts[0]} (${operators[0]} operators)"
    [[ $mode == development ]] && first="no-supervisor replay"
    echo "DRY START: preflight complete; first GPU stage would be $first."
    exit 0
fi
records=()
stage() {
    local name=$1
    shift
    local log_path
    log_path=$(mktemp "$result/logs/$name.XXXXXX.log")
    echo "$name: $log_path"
    uv run --locked python -m fleetlab.live.pass_cli "$@" > "$log_path" 2>&1
}
if [[ $mode == development ]]; then
    stage none pass --world "$world" --rule none --workers "$workers" --out "$result/none"
    records+=("$result/none/record.json")
fi
for operator_count in "${operators[@]}"; do
    stage "simple-$operator_count" sweep --world "$world" --rule timeout \
        --timeouts "${timeouts[@]}" --operators "$operator_count" --workers "$workers" \
        --out "$result/simple-$operator_count"
    for after in "${timeouts[@]}"; do
        records+=("$result/simple-$operator_count/after-$after/record.json")
    done
    stage "random-$operator_count" pass --world "$world" --rule random \
        --reference "$result/simple-$operator_count/after-240" --operators "$operator_count" \
        --workers "$workers" --out "$result/random-$operator_count"
    records+=("$result/random-$operator_count/record.json")
done
stage costs-2 sweep --world "$world" --rule decision --ratios "${ratios[@]}" \
    --operators 2 --workers "$workers" --out "$result/costs-2"
for ratio in "${ratios[@]}"; do
    [[ $ratio == 1 ]] && continue  # Strict risk * ratio > 1 cannot fire.
    records+=("$result/costs-2/cost_ratio-$ratio/record.json")
done
if [[ $# == 4 ]]; then
    stage supervisor-2 sweep --world "$world" --rule supervisor --ratios 1 2 4 \
        --operators 2 --workers "$workers" --supervisor "$3" --factory "$4" \
        --out "$result/supervisor-2"
    for ratio in 1 2 4; do
        records+=("$result/supervisor-2/cost_ratio-$ratio/record.json")
        control="$result/supervisor-2/cost_ratio-$ratio/hold-control/record.json"
        if [[ -f $control ]]; then
            records+=("$control")
        fi
    done
fi
uv run --locked python -m fleetlab.live.pass_cli score "${records[@]}" --out "$result/scores"
uv run --locked python -m fleetlab.live.figure "$result/scores/table.json" --out "$result/scores/success.png"
echo "Stored table and figure: $result/scores"
