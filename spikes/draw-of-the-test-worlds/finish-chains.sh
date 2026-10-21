#!/bin/bash
# Only for a job that ended while the steps after the draw were running (the draw is
# complete by then): the chains still owed, world by world, with new output directories,
# because a baseline refuses to write where something was written before.
#
# usage: finish-chains.sh TRUTH N "IDS"
#   N    a number for the new directories: TRUTH/campaign-N and TRUTH/chain-logs-N
#   IDS  the kept worlds whose chain has not ended, in the order of the places
#
# It starts nothing unless the job's own status file says that the draw is complete
# ("draw script exit 0") and names every given id among the kept worlds, and unless no job
# of the draw is alive: no world gets an oracle or a baseline before all places are filled,
# and a discarded candidate never does (design entry 16).
set -u
TRUTH=$1; N=$2; IDS=$3
HERE=$(dirname "$(readlink -f "$0")")
stamp() { date -u '+%Y-%m-%dT%H:%MZ'; }
refuse() { echo "finish-chains not started at $(stamp): $*" >> "$TRUTH/status"; echo "not started: $*" >&2; exit 3; }
grep -q '^draw script exit 0 ' "$TRUTH/status" || refuse "the status file does not say that the draw is complete"
kept=$(grep '^kept worlds, in the order of the places: ' "$TRUTH/status" | tail -1 | sed 's/^[^:]*: //')
[ -n "$kept" ] || refuse "the status file names no kept worlds"
for id in $IDS; do
  case " $kept " in *" $id "*) ;; *) refuse "$id is not among the kept worlds ($kept)" ;; esac
done
if pgrep -f '^([^ ]*/)?(bash|python3) [^ ]*/(draw-and-chain\.sh|world-chain\.sh|draw-worlds\.py) ' > /dev/null; then refuse "a job of the draw is still alive"; fi
while systemctl --user list-units --state=active --no-legend 'robots-heavy-*.scope' | grep -q .; do
  echo "a heavy job is still running at $(stamp); waiting for it" >> "$TRUTH/status"; sleep 60
done
echo "chains still owed, started by hand at $(stamp): $IDS (output in campaign-$N, logs in chain-logs-$N)" >> "$TRUTH/status"
worst=0
for id in $IDS; do
  OUT="$TRUTH/campaign-$N" bash "$HERE/world-chain.sh" "$id" "$TRUTH/status" "$TRUTH/chain-logs-$N"; c=$?
  echo "$id: chain script exit $c at $(stamp)" >> "$TRUTH/status"
  [ $c -eq 0 ] || worst=$c
done
echo "all chains ended $(stamp), worst exit $worst" >> "$TRUTH/status"
exit $worst
