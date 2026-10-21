#!/usr/bin/env bash
# The GPU runs of this spike, one lock acquisition each:
#
#   tools/heavy-run spikes/replay/gpu_jobs.sh JOB        (from the repository root)
#
# Every job writes its recording to ~/.cache/robots-probes/replay/ and prints a
# one-line summary per repeat. Compare recordings with compare_runs.py.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
py="$HOME/.cache/robots-probes/libero-port/bin/python"
all=0,1,2,3,4,5,6,7,8,9
five=0,1,2,3,4
run() {
  "$py" "$here/$1.py" "${@:2}" 2>&1 |
    grep -v "WARN\|robosuite\|macro\|Assets\|Local assets\|task orders\|Loading weights\|torch_dtype\|unauthenticated" || true
}

case "$1" in
  probe)      # questions 4 and 5: the policy alone
    run policy_probe --obs obs_sim --out probe_1 --k 4 ;;
  probe2)     # question 4, fresh process: the same outputs again
    run policy_probe --obs obs_sim --out probe_2 --reference-only ;;
  order)      # side finding: mujoco imported before torch breaks loading onto the GPU
    "$py" "$here/import_order.py" torch-first 2>&1 | tail -1
    "$py" "$here/import_order.py" mujoco-first 2>&1 | tail -1 ;;
  q6)         # question 6: six episodes, twice in one process
    run closed_loop --out cl_same --tasks 0,1,2 --inits 0,1 --repeats 2 ;;
  restore)    # question 3 with the policy in the loop
    run closed_loop_restore --out cl_restore --tasks 0,1,2 --k 37,130 ;;
  ref)        # 50 episodes, each inferred alone: the reference
    run closed_loop --out n50_ref --tasks $all --inits $five --group 5 --one-env ;;
  rep)        # the same 50 again in a fresh process and in reverse order
    run closed_loop --out n50_rep --tasks $all --inits $five --order reverse --group 5 --one-env ;;
  together)   # the same 50, five at a time in one batch that shrinks as episodes end
    run closed_loop --out n50_together --tasks $all --inits $five --batching together --group 5 --one-env ;;
  slots)      # 20 episodes in batches of exactly 5 rows, each episode in a row of its own
    run closed_loop --out s20_a --tasks 0,1,2,3 --inits $five --batching slots5 --group 5 --one-env ;;
  slots2)     # the same 20 with other episodes in the other rows, and 5 of them with none
    run closed_loop --out s20_b --batching slots5 --group 5 --one-env \
      --pairs 0:0,1:1,2:2,3:3,0:4,1:0,2:1,3:2,0:3,1:4,2:0,3:1,0:2,1:3,2:4,3:0,0:1,1:2,2:3,3:4
    run closed_loop --out s5_c --pairs 1:0,1:1,1:2,1:3,1:4 --batching slots5 --group 1 --one-env ;;
  noise)      # the same 50 with fresh sampler noise: the statistical noise floor
    run closed_loop --out n50_noise --tasks $all --inits $five --noise-offset 50000 --group 5 --one-env ;;
  bf16)       # the same 50 in bfloat16, alone
    run closed_loop --out n50_bf16 --tasks $all --inits $five --amp bf16 --group 5 --one-env ;;
  bf16rep)
    run closed_loop --out n50_bf16_rep --tasks $all --inits $five --amp bf16 --order reverse --group 5 --one-env ;;
  slot3)      # one lock acquisition: checks first, then the reference set next to the shrinking-batch set
    "$0" order; "$0" q6; "$0" restore
    "$0" ref & "$0" together & wait ;;
  slot4)      # one lock acquisition, one process at a time (GPU memory): the repeat of the
              # reference set, then the slotted sets
    "$0" rep; "$0" slots; "$0" slots2 ;;
  *) echo "unknown job $1" >&2; exit 2 ;;
esac
