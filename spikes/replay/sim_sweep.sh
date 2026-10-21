#!/usr/bin/env bash
# Question 1 across tasks: for each (suite, task) make a contact-rich action script,
# then play it twice in each of two separate processes that run at the same time.
# All four state traces, image streams and fixture layouts per task must agree.
#   spikes/replay/sim_sweep.sh [--no-clear]     (from the repository root)
# --no-clear leaves LIBERO's accumulating sampler list alone; the two tasks with
# drawers or doors that start open or closed then differ from reset to reset.
set -u
PY=$HOME/.cache/robots-probes/libero-port/bin/python
OUT=$HOME/.cache/robots-probes/replay
HERE=$(dirname "$0")
CLEAR=--clear; [ "${1:-}" = "--no-clear" ] && CLEAR=
for st in libero_spatial:3 libero_spatial:5 libero_spatial:8 libero_10:0 libero_10:3 libero_10:6 libero_10:9 libero_goal:2 libero_goal:7 libero_object:4; do
  suite=${st%%:*}; task=${st##*:}; tag=${suite}_${task}
  tools/heavy-run --light $PY $HERE/scripted_actions.py --suite $suite --task $task --out sweep_$tag 2>&1 | grep "^task\|^steps\|^object"
  pids=""
  for p in a b; do
    tools/heavy-run --light $PY $HERE/sim_trace.py --suite $suite --task $task --actions $OUT/sweep_$tag.npy \
      --out sweep_${tag}_$p --repeats 2 --seeds 1000 --np-seed $CLEAR > $OUT/sweep_${tag}_$p.log 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  n_runs=$(grep -h "^repeat" $OUT/sweep_${tag}_?.log | wc -l)
  n_state=$(grep -h "^repeat" $OUT/sweep_${tag}_?.log | awk '{print $8}' | sort -u | wc -l)
  n_img=$(grep -h "^repeat" $OUT/sweep_${tag}_?.log | awk '{print $12 $13}' | sort -u | wc -l)
  n_fix=$(grep -h "^repeat" $OUT/sweep_${tag}_?.log | awk '{print $15}' | sort -u | wc -l)
  echo "RESULT $tag: $n_runs runs; distinct state traces $n_state, image streams $n_img, fixture layouts $n_fix"
  rm -f $OUT/sweep_${tag}_a.npz $OUT/sweep_${tag}_b.npz
done
