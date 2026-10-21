#!/usr/bin/env bash
# One measured evaluation run, queued behind the machine's heavy-job lock.
#
#   spikes/baseline/run.sh NAME [lerobot-eval flags...]
#
# Writes ~/.cache/robots-probes/baseline/runs/NAME/{eval_info.json,measure.json}
# and NAME.log beside that directory. Defaults that every run shares are set
# here; any flag given on the command line is appended and so wins.
#
# Environment:
#   MJ=381|337             MuJoCo version: 381 = shared venv libero-port (default),
#                          337 = my copy of it with mujoco 3.3.7 (see README, "Environments")
#   BASELINE_NO_VIDEO=0    keep LeRobot's episode videos (default here: 1, no videos)
#   SEED=<n>               evaluation seed (default 1000)
set -euo pipefail

name="$1"; shift
repo="$(cd "$(dirname "$0")/../.." && pwd)"
probes="$HOME/.cache/robots-probes"
case "${MJ:-381}" in
    381) venv="$probes/libero-port";          cfg="$probes/libero-config-libero-port" ;;
    337) venv="$probes/baseline-venv-mj337";  cfg="$probes/baseline/libero-config-mj337" ;;
    *) echo "MJ must be 381 or 337" >&2; exit 2 ;;
esac
out="$probes/baseline/runs/$name"
mkdir -p "$probes/baseline/runs"
rm -rf "$out"
mkdir -p "$out"   # without videos LeRobot does not create it and then fails to write eval_info.json

export LIBERO_CONFIG_PATH="$cfg"   # must be set, or LIBERO asks a question on first import
export MIOPEN_FIND_MODE=2      # ROCm: skip the slow convolution auto-tuning
export MUJOCO_GL=egl           # headless rendering on the Radeon
export BASELINE_NO_VIDEO="${BASELINE_NO_VIDEO:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

cd "$repo"
exec tools/heavy-run "$venv/bin/python" "$repo/spikes/baseline/measured_eval.py" \
    --env.type=libero \
    --eval.n_episodes=10 --eval.batch_size=10 --eval.use_async_envs=false \
    --policy.device=cuda --seed="${SEED:-1000}" \
    --output_dir="$out" \
    "$@" </dev/null >"$out.log" 2>&1
