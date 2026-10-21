#!/bin/sh
# Run one stage of the host screen through the runtime's job runner.
#
#   spikes/jobs-reliability/run_stage.sh STAGE [xvla|hold]
#
# STAGE is a folder under spikes/jobs-reliability/batches (screen-01, hold-01, confirm-01).
# Each batch runs as its own `python -m fleetlab.jobs.runner` process, in the manifest's
# order: with X-VLA through tools/heavy-run (one heavy job per batch), with the still
# policy as a light job. Results go to $RELIABILITY_ROOT/STAGE/<batch>/ (default root
# ~/.cache/robots-probes/demo/2026-10-20/reliability), never into the checkout. A batch
# whose run directory holds episodes already is continued with --resume, so the script can
# be started again after an interruption. The batches are validated first and copied
# beside the results; screen_report.{md,json} is written at the end.
#
# HOST-AUTHORED screen: its programs and results are never an engineering agent's input.
set -u

stage=${1:?usage: run_stage.sh STAGE [xvla|hold]}
policy=${2:-xvla}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
batches="$here/batches/$stage"
root=${RELIABILITY_ROOT:-$HOME/.cache/robots-probes/demo/2026-10-20/reliability}
out="$root/$stage"
case "$policy:$stage" in hold:hold-*) ;; hold:*) out="$root/$stage-hold" ;; esac
heavy=${RELIABILITY_HEAVY_RUN:-tools/heavy-run}  # a stand-in only for test_screen.py
case "$policy" in
    xvla) run="$heavy" ;;
    hold) run="$heavy --light -m 6G" ;;
    *) echo "policy is xvla or hold, not $policy" >&2; exit 2 ;;
esac

cd "$repo" || exit 2
export FLEETLAB_MIOPEN_DIR="$HOME/.cache/robots-probes/engine/miopen"
export TMPDIR=/var/tmp/robots-reliability
mkdir -p "$TMPDIR" "$out/batches"

uv run --locked python "$here/screen.py" validate "$batches" || exit 1
for f in "$batches"/*; do
    cp -n "$f" "$out/batches/"
    cmp -s "$f" "$out/batches/$(basename "$f")" || {
        echo "$out/batches holds another $(basename "$f"): a stage's batches do not change" >&2
        exit 1
    }
done
order=$(uv run --locked python -I -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["order"]))' "$batches/manifest.json")

failed=0
for b in $order; do
    dir="$out/$b"
    resume=
    [ -f "$dir/episodes.jsonl" ] && resume=--resume
    echo "$(date '+%F %T') $stage $b: start ($policy${resume:+, resumed})"
    # shellcheck disable=SC2086  # $run and $resume are word lists on purpose
    $run uv run --locked python -m fleetlab.jobs.runner --batch "$batches/$b.batch.json" \
        --out "$dir" --policy "$policy" $resume >>"$out/$b.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') $stage $b: exit $rc; $(grep '^runner: {' "$out/$b.log" | tail -n 1)"
    [ "$rc" -eq 0 ] || failed=$((failed + 1))
done

echo "$stage ($policy): $failed batch(es) failed"
if uv run --locked python "$here/screen.py" report "$out" >/dev/null; then
    echo "report: $out/screen_report.md"
else
    echo "$stage ($policy): the report could not be written" >&2
    exit 1
fi
[ "$failed" -eq 0 ]
