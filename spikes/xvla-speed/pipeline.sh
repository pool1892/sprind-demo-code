#!/usr/bin/env bash
# The X-VLA speed spike step by step, as it was run on 2026-10-06.
#
#   spikes/xvla-speed/pipeline.sh STEP      (from the repository root)
#
# The README's tables name the step behind every number. Every step that loads
# the model or starts workers takes the machine's heavy-job lock one job at a
# time (run.sh, fleet.sh), so other jobs can go in between. Nothing is skipped or
# resumed: fleet.sh refuses to overwrite a finished run.
#
# I ran these commands one by one as the work went along and wrote the script
# afterwards as the record; it has not been run from top to bottom.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="$HOME/.cache/robots-probes/xvla-speed"
py="$HOME/.cache/robots-probes/baseline-venv-mj337/bin/python"
FP16='{"half": {"transformer": "fp16", "vision": "fp16", "encoder": "fp16"}, "linear_t": true, "hoist": true, "fast_post": true, "benchmark": false}'
FP16_PLAIN='{"half": {"transformer": "fp16", "vision": "fp16", "encoder": "fp16"}, "hoist": true, "fast_post": true, "benchmark": false}'
FP16_RES32='{"half": {"transformer": "fp16", "vision": "fp16", "encoder": "fp16"}, "residual32": true, "linear_t": true, "hoist": true, "fast_post": true, "benchmark": false}'
FP32='{"hoist": true, "fast_post": true, "benchmark": false}'

case "$1" in
obs)         # 80 recorded observations for every policy-only check (light: one simulator, no model)
    "$here/run.sh" --light make_obs make_obs.py ;;
miopen-copy) # a private copy of the shared convolution database as it stood, for the variants named "today"
    mkdir -p "$base/miopen/today/db" "$base/miopen/today/cache"
    cp -n ~/.config/miopen/gfx1151_20.HIP.3_6_0_8d1ae90e.u*db.txt "$base/miopen/today/db/"
    cp -n ~/.cache/miopen/3.6.0.8d1ae90e/gfx1151_20.ukdb "$base/miopen/today/cache/" ;;
profile)     # question 1: where a call's time goes, 32-bit as today and 16-bit as recommended
    MIOPEN_DB=today "$here/run.sh" profile_a profile_call.py --tag a
    "$here/run.sh" profile_fp16_lt profile_call.py --tag fp16_lt --batches 1 2 3 4 5 6 8 10 12 16 --opts "$FP16" ;;
micro)       # matrix products and attention alone, in three number formats
    MIOPEN_DB=micro "$here/run.sh" micro_a micro_bench.py --tag a ;;
variants)    # question 2: every change, one fresh process each; then the table
    "$here/run.sh" variants_1 variants.py --suite today,today@2,hoist,fast_post,t_fp16,t_bf16
    "$here/run.sh" variants_2 variants.py --suite t_fp16_res32,t_bf16_res32,all_fp16,all_bf16,all_fp16_res32,all_bf16_res32,amp_bf16,amp_fp16
    "$here/run.sh" variants_3 variants.py --reps 3 --suite miopen_empty_db,miopen_empty_db@2,miopen_no_winograd,miopen_search,miopen_tuned
    "$here/run.sh" variants_4 variants.py --reps 3 --suite benchmark_off,benchmark_off_empty_db,benchmark_off_no_winograd_db,channels_last,no_grad_only,attn_manual
    "$here/run.sh" variants_5 variants.py --suite all_fp16@2,all_fp16_benchmark_off,all_fp16_benchmark_off_empty_db,all_fp16_channels_last,all_fp16_attn_manual,all_fp16_hoist_fast_post,all_fp16_no_winograd,all_fp16_search,all_fp16_tuned
    "$here/run.sh" variants_6 variants.py --suite all_fp16_lt,all_fp16_lt@2,all_fp16_res32_lt,fp32_pinned
    "$here/run.sh" variants_7 variants.py --suite all_bf16_lt,fp32_lt,all_fp16_lt_steps_5,all_fp16_lt_steps_20,all_fp16_lt_views_3
    "$here/run.sh" variants_8 variants.py --suite all_fp16_lt_compile
    "$here/run.sh" variants_9 variants.py --reps 3 --suite compile
    "$py" "$here/variants.py" --table
    "$py" "$here/variants.py" --table --ref all_fp16_lt ;;
miopen-pin)  # the convolution database every later run reads, searched once on a quiet machine
    MIOPEN_DB=pinned "$here/run.sh" miopen_pin miopen_pin.py --rows 1 2 3 4 5 6 7 8 10 12 16 --fp32-rows 1 2 4 6 8 12 16 ;;
rows)        # question 3, policy alone: row position, batch mates, batch size
    "$here/run.sh" rows_fp16 rows_check.py --tag fp16 --sizes 8 4 6 12 --opts "$FP16_PLAIN"
    "$here/run.sh" rows_diag rows_diag.py
    "$here/run.sh" rows_diag2 rows_diag2.py
    "$here/run.sh" rows_diag2_forms rows_diag2.py --forms --tag _forms
    "$here/run.sh" rows_fp16_lt rows_check.py --tag fp16_lt --sizes 8 4 6 --opts "$FP16"
    "$here/run.sh" rows_fp32 rows_check.py --tag fp32 --sizes 8 4 --effect-sizes 1 2 4 6 8 12 16 --opts "$FP32"
    "$here/run.sh" rows_fp16res32 rows_check.py --tag fp16res32 --sizes 4 8 --effect-sizes 4 8 --opts "$FP16_RES32" ;;
scan)        # question 3, end to end: batch size and worker count on the 200 healthy episodes, 16-bit
    for cfg in "4 10" "4 8" "6 10" "8 10" "5 10" "3 10"; do
        set -- $cfg
        "$here/fleet.sh" "h200_fp16_r$1w$2" healthy200 --rows "$1" --workers "$2" --policy fp16
    done ;;
exact)       # the same episodes from runs with other worker counts: identical?
    "$here/fleet.sh" m200_fp16_r4w8 mix200 --rows 4 --workers 8 --policy fp16
    "$here/fleet.sh" m200_fp16res32_r4w10 mix200 --rows 4 --workers 10 --policy fp16_res32
    "$here/fleet.sh" m200_fp16res32_r4w8 mix200 --rows 4 --workers 8 --policy fp16_res32
    "$py" "$here/analyze.py" compare h200_fp16_r4w10 h200_fp16_r4w8
    "$py" "$here/analyze.py" compare h200_fp16_r4w10 h200v_fp16_r4w10
    "$py" "$here/analyze.py" compare s100_fp16_r4w10 h200_fp16_r4w10
    "$py" "$here/analyze.py" compare m200v_fp16_r4w10 m200_fp16_r4w8
    "$py" "$here/analyze.py" compare m200_fp16res32_r4w10 m200_fp16res32_r4w8
    "$py" "$here/analyze.py" compare m200v_fp16res32_r4w10 m200_fp16res32_r4w8
    "$py" "$here/analyze.py" compare h200v_fp16res32_r4w10 s100_fp16res32_r4w10
    "$py" "$here/analyze.py" compare s100_fp32_r8w10 "$HOME/.cache/robots-probes/skeleton/runs/healthy200" --ignore-policy ;;
success)     # 100 episodes per inference setting on the same start states, next to today's
    "$here/fleet.sh" s100_fp32_r8w10 spatial100 --rows 8 --workers 10 --policy fp32
    "$here/fleet.sh" s100_fp16_r4w10 spatial100 --rows 4 --workers 10 --policy fp16
    "$here/fleet.sh" s100_fp16res32_r4w10 spatial100 --rows 4 --workers 10 --policy fp16_res32
    "$here/fleet.sh" s100_bf16_r4w10 spatial100 --rows 4 --workers 10 --policy bf16
    for r in s100_fp16_r4w10 s100_fp16res32_r4w10 s100_bf16_r4w10; do "$py" "$here/analyze.py" pairs s100_fp32_r8w10 "$r"; done
    "$py" "$here/analyze.py" tasks s100_fp32_r8w10 s100_fp16_r4w10 s100_fp16res32_r4w10 s100_bf16_r4w10
    # the 200 healthy and the 200 faulted-mix episodes, pair by pair (runs of the steps "scan", "headline", "exact")
    "$py" "$here/analyze.py" pairs "$HOME/.cache/robots-probes/skeleton/runs/healthy200" h200_fp16_r4w10
    "$py" "$here/analyze.py" pairs "$HOME/.cache/robots-probes/skeleton/runs/healthy200" h200v_fp16res32_r4w10
    "$py" "$here/analyze.py" pairs m200v_fp32_r4w10 m200v_fp16_r4w10
    "$py" "$here/analyze.py" pairs m200v_fp32_r4w10 m200v_fp16res32_r4w10
    "$py" "$here/analyze.py" pairs m200v_fp32_r4w10 m200_fp16res32_r4w10 ;;
headline)    # the numbers of the verdict: healthy and faulted mix, with video and all tables written
    "$here/fleet.sh" h200v_fp16res32_r4w10 healthy200 --rows 4 --workers 10 --policy fp16_res32 --video
    "$here/fleet.sh" m200v_fp16res32_r4w10 mix200 --rows 4 --workers 10 --policy fp16_res32 --video
    "$here/fleet.sh" h200v_fp16_r4w10 healthy200 --rows 4 --workers 10 --policy fp16 --video
    "$here/fleet.sh" m200v_fp16_r4w10 mix200 --rows 4 --workers 10 --policy fp16 --video
    "$here/fleet.sh" h200v_fp32_r4w10 healthy200 --rows 4 --workers 10 --policy fp32 --video
    "$here/fleet.sh" m200v_fp32_r4w10 mix200 --rows 4 --workers 10 --policy fp32 --video
    "$py" "$here/analyze.py" summary h200v_fp16res32_r4w10 m200v_fp16res32_r4w10 h200v_fp16_r4w10 m200v_fp16_r4w10 \
        h200v_fp32_r4w10 m200v_fp32_r4w10 ;;
knobs)       # question 4: settings that change what the policy does (16-bit, 100 episodes each)
    "$here/fleet.sh" k100_steps5 spatial100 --rows 4 --workers 10 --policy fp16 --steps 5
    "$here/fleet.sh" k100_steps20 spatial100 --rows 4 --workers 10 --policy fp16 --steps 20
    "$here/fleet.sh" k100_nas20 spatial100 --rows 4 --workers 10 --policy fp16 --n-action-steps 20
    "$here/fleet.sh" k100_nas10 spatial100 --rows 4 --workers 10 --policy fp16 --n-action-steps 10
    "$here/fleet.sh" k100_views3 spatial100 --rows 4 --workers 10 --policy fp16 --views 3
    "$py" "$here/analyze.py" tasks s100_fp16_r4w10 k100_steps5 k100_steps20 k100_nas20 k100_nas10 k100_views3
    for r in k100_steps5 k100_steps20 k100_nas20 k100_nas10 k100_views3; do "$py" "$here/analyze.py" pairs s100_fp16_r4w10 "$r"; done ;;
smolvla)     # question 5: the second policy, policy alone and end to end
    "$here/run.sh" rows_smolvla_fp32 rows_check.py --model smolvla --tag smolvla_fp32 --sizes 4 8 --effect-sizes 1 2 4 8 16 --opts '{"n_action_steps": 10}'
    "$here/run.sh" rows_smolvla_fp32_lt rows_check.py --model smolvla --tag smolvla_fp32_lt --sizes 4 8 --effect-sizes 1 2 4 8 16 --opts '{"n_action_steps": 10, "linear_t": true}'
    "$here/run.sh" rows_smolvla_fp16_lt rows_check.py --model smolvla --tag smolvla_fp16_lt --sizes 4 8 --opts '{"n_action_steps": 10, "half": "fp16", "linear_t": true}'   # fails
    "$here/fleet.sh" h200v_smolvla_r4w10 healthy200 --model smolvla --rows 4 --workers 10 --policy fp32 --video
    "$here/fleet.sh" m200v_smolvla_r4w10 mix200 --model smolvla --rows 4 --workers 10 --policy fp32 --video
    "$py" "$here/analyze.py" summary h200v_smolvla_r4w10 m200v_smolvla_r4w10 ;;
paired)      # section 8: the same episodes in 32-bit and in the recommended 16-bit variant, outcome by outcome
    for kind in cam6_a cam6_b object200 delay17 cam3 cam9; do
        name="${kind/_/}"; name="${name/object200/obj}"
        "$here/fleet.sh" "p_${name}_res32" "$kind" --rows 4 --workers 10 --policy fp16_res32
        "$here/fleet.sh" "p_${name}_fp32" "$kind" --rows 4 --workers 10 --policy fp32
    done
    "$py" "$here/analyze.py" paired \
        "libero_object, no fault" p_obj_fp32 p_obj_res32 \
        "camera 6 degrees" p_cam6a_fp32,p_cam6b_fp32 p_cam6a_res32,p_cam6b_res32 \
        "delay 17 steps" p_delay17_fp32 p_delay17_res32 \
        "camera 3 degrees" p_cam3_fp32 p_cam3_res32 \
        "camera 9 degrees" p_cam9_fp32 p_cam9_res32
    "$py" "$here/analyze.py" bytask p_cam6a_fp32,p_cam6b_fp32 p_cam6a_res32,p_cam6b_res32
    # the episodes these runs share with the earlier faulted mix: replayed identically?
    "$py" "$here/analyze.py" compare p_cam6a_fp32,p_delay17_fp32 m200v_fp32_r4w10
    "$py" "$here/analyze.py" compare p_cam6a_res32,p_delay17_res32 m200v_fp16res32_r4w10 ;;
*) echo "unknown step $1" >&2; exit 2 ;;
esac
