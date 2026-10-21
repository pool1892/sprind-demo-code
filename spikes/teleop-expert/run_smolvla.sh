#!/bin/sh
# Take over from SmolVLA's failures: spikes/teleop-expert/run_smolvla.sh SUITE [EPISODES_PER_TASK]
# Loads the policy onto the GPU, so it waits for the machine-wide heavy lock.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
suite="${1:-libero_spatial}"
n="${2:-5}"
P="$HOME/.cache/robots-probes"
out="$P/teleop/smolvla_$suite"
rm -rf "$out"
exec "$root/tools/heavy-run" env \
    MUJOCO_GL=egl MIOPEN_FIND_MODE=2 LIBERO_CONFIG_PATH="$P/libero-config-libero-port" \
    PYTHONPATH="$here" PYTHONWARNINGS=ignore TAKEOVER_NAME="smolvla_takeover_$suite" \
    "$P/libero-port/bin/python" "$here/takeover_smolvla.py" \
    --policy.path=lerobot/smolvla_libero --env.type=libero --env.task="$suite" \
    --eval.n_episodes="$n" --eval.batch_size="$n" --eval.use_async_envs=false --policy.device=cuda --seed=1000 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --output_dir="$out"
