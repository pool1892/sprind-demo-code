#!/bin/sh
# Run one of the scripts in this directory inside the probe venv, under the
# light memory ceiling: spikes/teleop-expert/run.sh SCRIPT.py [ARGS...]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
script="$1"; shift
exec "$root/tools/heavy-run" --light env \
    MUJOCO_GL=egl \
    LIBERO_CONFIG_PATH="$HOME/.cache/robots-probes/libero-config-libero-port" \
    PYTHONPATH="$here" PYTHONWARNINGS=ignore \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$HOME/.cache/robots-probes/libero-port/bin/python" "$here/$script" "$@"
