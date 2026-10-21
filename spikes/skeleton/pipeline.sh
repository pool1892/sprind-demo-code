#!/usr/bin/env bash
# The walking skeleton from start to finish, as it was run on 2026-10-06.
#
#   spikes/skeleton/pipeline.sh STEP      (from the repository root)
#
# Each step is one command or a few; the README's tables name the step behind
# every number. Steps that use the GPU take the machine's heavy-job lock one
# run at a time (run.sh), so other jobs can go in between. Nothing is skipped
# or resumed automatically: run.sh refuses to overwrite a finished run.
#
#   CAM=6 DELAY=17   the fault levels of robots 2 and 3, chosen from the dose runs
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
base="$HOME/.cache/robots-probes/skeleton"
sim="$HOME/.cache/robots-probes/baseline-venv-mj337/bin/python"   # simulator and policy (read-only, from the baseline spike)
py="$base/venv/bin/python"                                        # tools, export, agent (built by the step "venv")
runs="$base/runs"; man="$base/manifests"
CAM="${CAM:-6}"; DELAY="${DELAY:-17}"

case "$1" in
venv)        # the small environment for the tool server, the export and the agent runtime
    UV_PYTHON_PREFERENCE=only-managed uv venv --python 3.12 "$base/venv"
    uv pip install --python "$py" mcp==2.3.0 duckdb==1.5.6 pillow==12.3.0 jsonschema==4.26.0 av==15.1.0 numpy pyarrow ;;
probe)       # policy alone: call time by batch size, row independence, repeat in a second process
    tools/heavy-run "$sim" "$here/probe_policy.py" a
    tools/heavy-run "$sim" "$here/probe_policy.py" b ;;
smoke)       # 24 episodes as a first test of the runner (the day's first one used --life 2 and an
             # earlier server); run once more at the very end as NAME=smoke_final
    "$sim" "$here/manifests.py" smoke "$man/smoke.jsonl"
    "$here/run.sh" "${NAME:-smoke}" "$man/smoke.jsonl" --video ;;
dose)        # dose-response of both faults, 40 episodes per level, in two halves each
    "$sim" "$here/manifests.py" dose "$man/dose_cam.jsonl" --robot 2 --kind scene_camera_pan --param degrees --levels 0 3 6 9 12
    "$sim" "$here/manifests.py" dose "$man/dose_delay.jsonl" --robot 3 --kind action_delay --param steps --levels 0 8 14 17 20
    "$here/run.sh" dose_cam_p1 "$man/dose_cam.jsonl" --tasks 0-4
    "$here/run.sh" dose_cam_p2 "$man/dose_cam.jsonl" --tasks 5-9
    "$here/run.sh" dose_delay_p1 "$man/dose_delay.jsonl" --tasks 0-4
    "$here/run.sh" dose_delay_p2 "$man/dose_delay.jsonl" --tasks 5-9
    "$sim" "$here/analyze.py" dose "$runs"/dose_cam_p1 "$runs"/dose_cam_p2 "$runs"/dose_delay_p1 "$runs"/dose_delay_p2 ;;
fleet)       # the 120 episodes of the three robots, with video
    "$sim" "$here/manifests.py" fleet "$man/fleet.jsonl" --cam-degrees "$CAM" --delay-steps "$DELAY"
    "$here/run.sh" fleet "$man/fleet.jsonl" --video
    "$py" "$here/score.py" table "$runs/fleet" ;;
replay)      # every episode again with its fault removed, read from the run's truth records
    "$sim" "$here/manifests.py" replay "$man/replay.jsonl" --from "$runs/fleet" --keep-faults-every 5
    "$here/run.sh" replay "$man/replay.jsonl" ;;
healthy)     # 200 episodes with no fault anywhere, with video: the throughput batch and the clean reference
    "$sim" "$here/manifests.py" healthy "$man/healthy200.jsonl"
    "$here/run.sh" healthy200 "$man/healthy200.jsonl" --video
    "$sim" "$here/analyze.py" summary "$runs/healthy200" ;;
checks)      # exactness, and the truth per episode
    "$sim" "$here/analyze.py" compare "$runs/fleet" "$runs/replay"
    "$sim" "$here/analyze.py" compare "$runs/replay" "$runs/healthy200"
    "$sim" "$here/analyze.py" compare "$runs/fleet" "$runs/healthy200"
    "$sim" "$here/analyze.py" compare "$runs/fleet" "$runs/dose_cam_p1,$runs/dose_cam_p2,$runs/dose_delay_p1,$runs/dose_delay_p2"
    "$sim" "$here/analyze.py" compare "$runs/healthy200" "$runs/dose_cam_p1,$runs/dose_cam_p2,$runs/dose_delay_p1,$runs/dose_delay_p2"
    "$sim" "$here/analyze.py" compare "$runs/smoke" "$runs/smoke_final"
    "$sim" "$here/analyze.py" attribute "$runs/fleet" "$runs/replay" ;;
export)      # the agent's workspace, the search for anything that names the truth, the sandbox check
    "$py" "$here/export.py" "$runs/fleet" "$base/workspace"
    "$py" "$here/leak_check.py" "$runs/fleet" "$base/workspace"
    "$py" "$here/sandbox_check.py" "$base/workspace" "$runs/fleet" ;;
diagnose)    # three blind diagnoses, scored
    for i in 1 2 3; do
        "$py" "$here/diagnose.py" "$base/workspace" "$base/agent/diagnosis_$i.json" || true
    done
    "$py" "$here/score.py" score "$runs/fleet" "$base"/agent/diagnosis_*.json
    "$py" "$here/leak_check.py" "$runs/fleet" "$base/workspace" --after ;;
*) echo "unknown step $1" >&2; exit 2 ;;
esac
