#!/bin/sh
# Run one of the spike scripts in the probe environment.
#
#   spikes/throughput/run.sh SCRIPT.py [ARGS...]
#
# Wrap the call in tools/heavy-run (model on the GPU, or more than four
# simulation processes) or tools/heavy-run --light (everything else).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
probes="$HOME/.cache/robots-probes"
export LIBERO_CONFIG_PATH="$probes/libero-config-libero-port"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MIOPEN_FIND_MODE=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
script="$1"; shift
exec "$probes/libero-port/bin/python" "$here/$script" "$@"
