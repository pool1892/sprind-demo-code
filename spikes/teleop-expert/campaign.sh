#!/bin/sh
# The measurement campaign behind README.md. Run from anywhere: spikes/teleop-expert/campaign.sh
# Each line is one batch; results land in ~/.cache/robots-probes/teleop/results/final_*.jsonl.
# About 3800 episodes, roughly 40 minutes with four worker processes.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="$HOME/.cache/robots-probes/teleop/results"
mkdir -p "$out"
{ date; cd "$here" && sha256sum common.py scene.py grasp.py expert.py runner.py disturb.py evaluate.py; } > "$out/final_manifest.txt"
run() { name="$1"; shift; "$here/run.sh" evaluate.py --name "$name" "$@" > "$out/$name.log" 2>&1 || echo "$name FAILED" >> "$out/final_manifest.txt"; }

# 1. From LIBERO's own initial states: all 50 per task.
run final_init --scenario init --inits 0-49
# 2. Take-over from disturbed states: 10 per task and kind.
for s in displaced dropped on_side inverted closed_empty held awkward; do
    run "final_$s" --scenario "$s" --inits 0-9
done
# 3. Faulted robot and object, from the initial states: 10 per task and level.
for f in friction_0.5 friction_0.1 friction_0.02 mass_10 mass_50 mass_200 friction_0.1+mass_50; do
    run "final_fault_$f" --scenario init --fault "$f" --inits 0-9
done
date >> "$out/final_manifest.txt"
