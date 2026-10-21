#!/bin/sh
# Run one script of this spike in the locked environment, as a light job
# (8 GB ceiling, no heavy lock, no model): spikes/teleop-absolute/run.sh SCRIPT.py [ARGS...]
#
# SPIKE_CODE=DIR runs the scripts from a copy of this directory instead (campaign.sh
# freezes the code that way, so that editing here cannot reach a running campaign).
# numpy and BLAS get one thread per process: the expert's geometry is small and
# four worker processes share the machine with other agents' jobs.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
code="${SPIKE_CODE:-$here}"
script="$1"; shift
cd "$root"
exec "$root/tools/heavy-run" --light ${HEAVY_RUN_MEM:+-m "$HEAVY_RUN_MEM"} env \
    PYTHONPATH="$code" PYTHONWARNINGS=ignore \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run --locked python "$code/$script" "$@"
