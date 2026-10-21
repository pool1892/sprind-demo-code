#!/usr/bin/env bash
# The pi0.5 spike step by step, as it was run on 2026-10-07.
#
#   spikes/pi05/pipeline.sh STEP      (from the repository root)
#
# The README's tables name the step behind every number. Every step that loads
# a model or starts workers takes the machine's heavy-job lock (run.sh,
# fleet.sh). fleet.sh refuses to overwrite a finished run.
#
# I ran these commands one by one as the work went along and wrote the script
# as the record; it has not been run from top to bottom.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="$HOME/.cache/robots-probes/pi05"
py="$base/venv/bin/python"
xv="$HOME/.cache/robots-probes/xvla-speed/runs"
res="$base/results"

case "$1" in
env)         # the probe environment: a copy of the baseline spike's, and a LIBERO configuration pointing into it
    cp -a --reflink=always "$HOME/.cache/robots-probes/baseline-venv-mj337" "$base/venv"
    mkdir -p "$base/libero-config"
    sed "s#$HOME/.cache/robots-probes/baseline-venv-mj337#$base/venv#g" \
        "$HOME/.cache/robots-probes/baseline/libero-config-mj337/config.yaml" >"$base/libero-config/config.yaml"
    VIRTUAL_ENV="$base/venv" uv pip freeze --python "$py" >"$base/venv.freeze.txt" ;;
checkpoints) # pi0.5 is the revision the baseline spike had downloaded (same commit, same file hashes); pi0 from the Hub
    #   lerobot/pi05_libero_finetuned_v044 @ 7152f13e: copied by reflink from ~/.cache/robots-probes/baseline/pi05_libero_finetuned_local
    #   and checked: sha256 of model.safetensors = the Hub's LFS oid (ckpt/pi05_sha256.txt, ckpt/pi05_hub_tree.json)
    "$py" -c "from huggingface_hub import snapshot_download as s; s('lerobot/pi0_libero_finetuned_v044', revision='45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f', local_dir='$base/ckpt/pi0_libero_finetuned_v044')"
    curl -s -o "$base/tokenizer/src/paligemma_tokenizer.model" https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
    "$py" "$here/make_tokenizer.py" ;;
profile)     # question 1: load, memory, latency by batch size, stages; pins the MIOpen database first
    "$here/run.sh" profile_bf16 profile_pi05.py --tag bf16 --opts '{"dtype": "bfloat16"}' --batches 1 2 4 6 8 --reps 5 --pin
    "$here/run.sh" profile_fp32 profile_pi05.py --tag fp32 --opts '{"dtype": "float32"}' --batches 1 4 8 --reps 5 --against bf16 ;;
rows)        # question 3, policy alone: rows, batch mates, batch size, other processes, and which layers
    "$here/run.sh" rows_bf16 rows_pi05.py --tag bf16 --sizes 4 8 --opts '{"dtype": "bfloat16"}'
    "$here/run.sh" rows_bf16_lt rows_pi05.py --tag bf16_lt --sizes 4 --effect-sizes --opts '{"dtype": "bfloat16", "linear_t": true}'
    "$here/run.sh" rows_diag_bf16 rows_diag_pi05.py --tag bf16 --opts '{"dtype": "bfloat16"}'
    "$here/run.sh" rows_bf16_rocblas rows_pi05.py --tag bf16_rocblas --sizes 4 --effect-sizes 1 4 --opts '{"dtype": "bfloat16", "blas": "cublas"}'
    "$here/run.sh" rows_bf16_p2 rows_pi05.py --tag bf16_p2 --sizes 1 4 --effect-sizes --opts '{"dtype": "bfloat16"}' --against bf16
    "$here/run.sh" rows_bf16_p3 rows_pi05.py --tag bf16_p3 --sizes 1 --effect-sizes --opts '{"dtype": "bfloat16"}' --against bf16_p2 ;;
smoke)       # the runner on 24 episodes of all three kinds, 4 free rows (not replayable: see rows)
    "$here/fleet.sh" smoke_bf16_r4w6 smoke --model pi05 --rows 4 --workers 6 --policy bf16 --video ;;
main)        # questions 2, 4 and 5: batch size 1 (replayable), three workers, both videos
    # nominal: the X-VLA spike's 200 healthy libero_spatial episodes and its 200 libero_object ones
    "$here/fleet.sh" h200v_pi05_r1w3 healthy200 --model pi05 --rows 1 --workers 3 --policy bf16 --video
    "$here/fleet.sh" o200v_pi05_r1w3 object200 --model pi05 --rows 1 --workers 3 --policy bf16 --video
    # faults: exactly the episodes behind X-VLA's 55.2% and 72.5%
    "$here/fleet.sh" c6av_pi05_r1w3 cam6_a --model pi05 --rows 1 --workers 3 --policy bf16 --video
    "$here/fleet.sh" c6bv_pi05_r1w3 cam6_b --model pi05 --rows 1 --workers 3 --policy bf16 --video
    "$here/fleet.sh" d17v_pi05_r1w3 delay17 --model pi05 --rows 1 --workers 3 --policy bf16 --video
    # closed-loop replay: 24 of those episodes again, two workers, no video
    "$here/fleet.sh" smoke_r1w2 smoke --model pi05 --rows 1 --workers 2 --policy bf16 ;;
pi0)         # pi0 alongside: question 1 and 50 nominal episodes (10 tasks x start states 0 to 4)
    "$here/run.sh" profile_pi0_bf16 profile_pi05.py --tag pi0_bf16 --opts '{"model": "pi0", "dtype": "bfloat16"}' --batches 1 4 8 --reps 5
    "$here/fleet.sh" s50v_pi0_r1w3 spatial50 --model pi0 --rows 1 --workers 3 --policy bf16 --video ;;
delays)      # planned, not run: pi0.5 at lower delays, robot 3's scenes as in delay17, start states 0 to 4
             # (dropped because other jobs were queueing for the heavy-job lock)
    "$here/fleet.sh" d5v_pi05_r1w3 delay5_i5 --model pi05 --rows 1 --workers 3 --policy bf16 --video
    "$here/fleet.sh" d2v_pi05_r1w3 delay2_i5 --model pi05 --rows 1 --workers 3 --policy bf16 --video ;;
tables)      # the comparison tables, written to results/
    runs="h200v_pi05_r1w3 o200v_pi05_r1w3 c6av_pi05_r1w3 c6bv_pi05_r1w3 d17v_pi05_r1w3"
    "$py" "$here/analyze.py" summary smoke_bf16_r4w6 $runs smoke_r1w2 s50v_pi0_r1w3 | tee "$res/summary.md"
    "$py" "$here/analyze.py" tasks $runs s50v_pi0_r1w3 \
        "$xv/h200v_fp32_r4w10" "$xv/p_obj_fp32" "$xv/p_cam6a_fp32" "$xv/p_cam6b_fp32" "$xv/p_delay17_fp32" "$xv/s100_fp32_r8w10" | tee "$res/tasks.md"
    "$py" "$here/analyze.py" paired \
        "libero_spatial, no fault (the skeleton's 200)" "$xv/h200v_fp32_r4w10" h200v_pi05_r1w3 \
        "libero_object, no fault" "$xv/p_obj_fp32" o200v_pi05_r1w3 \
        "camera turned 6 degrees" "$xv/p_cam6a_fp32,$xv/p_cam6b_fp32" c6av_pi05_r1w3,c6bv_pi05_r1w3 \
        "actions 17 steps late" "$xv/p_delay17_fp32" d17v_pi05_r1w3 | tee "$res/paired_xvla32.md"
    "$py" "$here/analyze.py" paired \
        "libero_spatial, no fault (the skeleton's 200)" "$xv/h200v_fp16res32_r4w10" h200v_pi05_r1w3 \
        "libero_object, no fault" "$xv/p_obj_res32" o200v_pi05_r1w3 \
        "camera turned 6 degrees" "$xv/p_cam6a_res32,$xv/p_cam6b_res32" c6av_pi05_r1w3,c6bv_pi05_r1w3 \
        "actions 17 steps late" "$xv/p_delay17_res32" d17v_pi05_r1w3 | tee "$res/paired_xvla16.md"
    "$py" "$here/analyze.py" bytask "$xv/p_cam6a_fp32,$xv/p_cam6b_fp32" c6av_pi05_r1w3,c6bv_pi05_r1w3 | tee "$res/bytask_cam6.md"
    "$py" "$here/analyze.py" bytask "$xv/p_delay17_fp32" d17v_pi05_r1w3 | tee "$res/bytask_delay17.md"
    A_NAME=pi0.5 B_NAME=pi0 "$py" "$here/analyze.py" paired "libero_spatial, 10 tasks x start states 0 to 4" h200v_pi05_r1w3 s50v_pi0_r1w3 | tee "$res/paired_pi0.md"
    "$py" "$here/analyze.py" failures $runs | tee "$res/failures.md"
    "$py" "$here/analyze.py" compare smoke_r1w2 h200v_pi05_r1w3,c6av_pi05_r1w3,d17v_pi05_r1w3 | tee "$res/replay_smoke.md"
    "$py" "$here/analyze.py" motion $runs s50v_pi0_r1w3 smoke_bf16_r4w6 | tee "$res/gripper.md"
    "$py" "$here/make_tokenizer.py" --compare-only ;;
figures)     # the pictures and the camera-fault check (light; reads run outputs only)
    for e in h1-t2-i3 h2-t3-i2 h3-t4-i3 h2-t5-i1 h1-t8-i1; do
        "$py" "$here/figures.py" sheet h200v_pi05_r1w3 "$e" "$res/frames/h200_$e.png"
    done
    "$py" "$here/figures.py" sheet smoke_bf16_r4w6 k3-t0-i0 "$res/frames/smoke_k3-t0-i0.png"
    "$py" "$here/figures.py" pair h200v_pi05_r1w3 h2-t0-i0 c6av_pi05_r1w3 c6-t0-i0 "$res/frames/scene_healthy_vs_cam6_t0_i0.png"
    "$py" "$here/figures.py" camcheck ;;
*) echo "unknown step $1" >&2; exit 2 ;;
esac
