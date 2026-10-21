#!/bin/bash
# Everything between a world's run and its first model call, for one world:
# certify once per reading checkout, the oracle chain, the model-free baselines,
# the build rule, and the frozen analysis checkout's preflight.
#
# usage: world-chain.sh WORLD_ID STATUS_FILE LOG_DIR
#   INTEG  (env): the integration checkout (main with oracle round 9): oracle, baselines, scoring
#   SKILLS (env): the frozen analysis checkout
#   OUT    (env): where baselines and the build rule's answer go (new directories under it)
#
# The status file gets exit codes and times only. What the commands print goes to LOG_DIR:
# certify prints the truth terms and the oracle's logs name faults, so for a test world
# LOG_DIR lies on the truth side and is not read before all analyses of the world are complete.
# After anything but the known good exit codes the chain halts WITHOUT calling the oracle's
# report: the review of oracle round 9 found that reading the report after a stop blocks the recovery.
#
# Exit code: 1 after a halt (or 9 without a checkout); otherwise the worst exit code of the
# steps that follow the oracle's report, which all run whatever the one before returned
# (0 when each of them returned 0 and all four source audits of the preflight are clean).
set -u
ID=$1; S=$2; LOG=$3
INTEG=${INTEG:-/home/cs/code/robots-integ8}
SKILLS=${SKILLS:-$HOME/.cache/robots-skills-run-r8}
OUT=${OUT:-$HOME/.cache/robots-probes/campaign}
HERE=$(dirname "$(readlink -f "$0")")
W=$HOME/.cache/robots-probes/worlds/$ID
mkdir -p "$LOG" "$OUT"
stamp() { date -u '+%Y-%m-%dT%H:%MZ'; }
say() { echo "$ID: $*" >> "$S"; }
halt() { say "HALT: $*"; exit 1; }
worst=0
step() { say "$1 exit $2 at $(stamp)"; [ "$2" -eq 0 ] || worst=$2; }

say "chain started $(stamp); integration checkout $(git -C "$INTEG" rev-parse --short HEAD), analysis checkout $(git -C "$SKILLS" rev-parse --short HEAD)"

cd "$INTEG" || exit 9
uv run --locked python -m fleetlab.log certify "$W" > "$LOG/$ID-certify-integ.out" 2> "$LOG/$ID-certify-integ.err"; c=$?
say "certify (integration checkout) exit $c at $(stamp)"; [ $c -eq 0 ] || halt "certify"
cd "$SKILLS" || exit 9
.venv/bin/python -m fleetlab.log certify "$W" > "$LOG/$ID-certify-skills.out" 2> "$LOG/$ID-certify-skills.err"; c=$?
say "certify (analysis checkout) exit $c at $(stamp)"; [ $c -eq 0 ] || halt "certify"

cd "$INTEG" || exit 9
uv run --locked python -m fleetlab.oracle all "$W" --workers 8 > "$LOG/$ID-oracle-all.log" 2>&1; e=$?
say "oracle all exit $e at $(stamp)"
if [ ! -f "$W/oracle/summary.md" ] || [ $e -gt 2 ]; then halt "first pass (no summary, or exit above 2)"; fi
uv run --locked python -m fleetlab.oracle verify "$W" --workers 8 > "$LOG/$ID-oracle-verify.log" 2>&1; v=$?
say "oracle verify exit $v at $(stamp)"; [ $v -eq 0 ] || halt "verify (report not called)"
uv run --locked python -m fleetlab.oracle causes "$W" --workers 8 > "$LOG/$ID-oracle-causes.log" 2>&1; k=$?
say "oracle causes exit $k at $(stamp)"; [ $k -eq 0 ] || halt "causes (report not called)"
uv run --locked python -m fleetlab.oracle report "$W" > "$LOG/$ID-oracle-report.log" 2>&1; r=$?
say "oracle report exit $r at $(stamp)"; [ $r -eq 0 ] || halt "report"

uv run --locked python -m fleetlab.baselines.classical "$W/export" --out "$OUT/$ID-classical" > "$LOG/$ID-classical.log" 2>&1; step "classical baseline" $?
uv run --locked python -m fleetlab.baselines --export "$W/export" --out "$OUT/$ID-offline" --offline > "$LOG/$ID-offline.log" 2>&1; step "offline baselines" $?
uv run --locked python -m fleetlab.baselines.build_rule --export "$W/export" --out "$OUT/$ID-build-rule" > "$LOG/$ID-build-rule.log" 2>&1; step "build rule" $?
uv run --locked python -m fleetlab.baselines.build_rule --evaluate "$OUT/$ID-build-rule" --world-root "$W" --out "$OUT/$ID-build-rule.eval.json" > "$LOG/$ID-build-rule-eval.log" 2>&1; step "build rule evaluation" $?

cd "$SKILLS" || exit 9
rm -rf "/var/tmp/robots-preflight/$ID"; mkdir -p /var/tmp/robots-preflight
TMPDIR=/var/tmp/robots-preflight OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python "$HERE/w50-launch-audit.py" \
  "$W" "$W/export" "/var/tmp/robots-preflight/$ID" > "$LOG/$ID-preflight.log" 2>&1; p=$?
clean=$(grep -c 'passed, 0 failures' "$LOG/$ID-preflight.log")
say "preflight of the analysis checkout exit $p at $(stamp): $clean of 4 source audits clean"
[ $p -eq 0 ] || worst=$p
[ "$clean" -eq 4 ] || [ $worst -ne 0 ] || worst=1
say "chain ended $(stamp), worst exit after the oracle's report: $worst"
exit $worst
