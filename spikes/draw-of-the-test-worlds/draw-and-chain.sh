#!/bin/bash
# The draw and everything after it that needs no model, as one unattended job:
# wait for the beacon's round, draw (draw-worlds.py), then for every kept world, in the
# order of the places, world-chain.sh (certify, oracle, model-free baselines, build rule,
# preflight of the frozen analysis checkout).
#
# usage: draw-and-chain.sh EPOCH KIND ROUND TRUTH "PLACES" "IDS"
#   EPOCH   start no earlier than this (the round's time plus a few seconds)
#   TRUTH   the truth-side directory: ledger/, recipes/, logs/, chain-logs/, campaign/, status
#
# TRUTH/status gets exit codes, times (UTC) and the ledger's counts. Everything else under TRUTH
# is not read by the development side before all analyses of the world are complete.
#
# The same command line can be started again after the job was killed: the ledger says where
# the draw is. A run that an interrupted start left behind keeps running in its own scope
# (tools/heavy-run), so this job first waits until no heavy job is left on the machine.
set -u
EPOCH=$1; KIND=$2; ROUND=$3; TRUTH=$4; PLACES=$5; IDS=$6
HERE=$(dirname "$(readlink -f "$0")")
RUN=${RUN:-$HOME/.cache/robots-worlds-run-test}
mkdir -p "$TRUTH"
stamp() { date -u '+%Y-%m-%dT%H:%MZ'; }
until [ "$(date +%s)" -ge "$EPOCH" ]; do sleep 5; done
while systemctl --user list-units --state=active --no-legend 'robots-heavy-*.scope' | grep -q .; do
  echo "a heavy job is still running at $(stamp); waiting for it" >> "$TRUTH/status"; sleep 60
done
python3 "$HERE/draw-worlds.py" --kind "$KIND" --round "$ROUND" --checkout "$RUN" \
  --truth "$TRUTH" --places $PLACES --ids $IDS > "$TRUTH/draw.out" 2>&1; d=$?
echo "draw script exit $d at $(stamp)" >> "$TRUTH/status"
[ $d -eq 0 ] || exit $d
kept=$(cd "$RUN" && uv run --locked python -m fleetlab.worlds ledger summary --ledger "$TRUTH/ledger/ledger.jsonl" \
  | python3 -c '
import json, sys
text = sys.stdin.read()
start = text.rfind("\n{")
state = json.loads(text[start + 1:] if start >= 0 else text)
for place in sys.argv[1].split():
    last = state["places"][place]["candidates"][-1]
    assert last["state"] == "kept", (place, last["state"])
    print(last["world_id"])
' "$PLACES") || { echo "no list of kept worlds from the ledger" >> "$TRUTH/status"; exit 3; }
echo "kept worlds, in the order of the places: $(echo $kept)" >> "$TRUTH/status"
worst=0
for id in $kept; do
  OUT="$TRUTH/campaign" bash "$HERE/world-chain.sh" "$id" "$TRUTH/status" "$TRUTH/chain-logs"; c=$?
  echo "$id: chain script exit $c at $(stamp)" >> "$TRUTH/status"
  [ $c -eq 0 ] || worst=$c
done
echo "all chains ended $(stamp), worst exit $worst" >> "$TRUTH/status"
exit $worst
