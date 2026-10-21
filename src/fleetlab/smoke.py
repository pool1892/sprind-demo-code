"""Smoke test: one LIBERO episode, headless, with the baseline policy on the GPU.

Proves that simulator, renderer, policy and GPU work together and that a video
comes out. It does not judge the policy: a failed task still passes.

The policy is X-VLA, which commands target poses of the gripper, so the
episode runs in absolute control. Policy and processors are built the way
`lerobot-eval` builds them.

Loads a model onto the GPU, so run it through `tools/heavy-run` (`make smoke`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

from fleetlab.runtime import cap_gpu_memory, configure_process, data_dir

# The setup measured in spikes/baseline: the checkpoint as stored, absolute control.
POLICY = "lerobot/xvla-libero"
CONTROL_MODE: Literal["relative", "absolute"] = "absolute"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda", help="cuda (the Radeon, through ROCm) or cpu")
    parser.add_argument("--max-steps", type=int, help="default: the suite's episode limit")
    parser.add_argument("--out", type=Path, default=data_dir() / "smoke" / "episode.mp4")
    args = parser.parse_args()

    configure_process()  # before the imports below, which read the environment

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LeRobotLiberoConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed

    from fleetlab.sim.libero_env import CONTROL_HZ, LiberoEnv, as_batch_of_one, camera_frame
    from fleetlab.video import write_mp4

    if args.device == "cuda" and not torch.cuda.is_available():
        print("smoke: FAILED, PyTorch does not see the GPU", file=sys.stderr)
        return 1
    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"smoke: torch {torch.__version__} on {device_name}", flush=True)
    cap_gpu_memory()
    set_seed(args.seed)

    started = time.monotonic()
    config = PreTrainedConfig.from_pretrained(POLICY)
    config.pretrained_path = Path(POLICY)
    config.device = args.device
    env_config = LeRobotLiberoConfig(task=args.suite, control_mode=CONTROL_MODE)
    policy = make_policy(config, env_cfg=env_config)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        config,
        pretrained_path=POLICY,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    # The part that depends on the environment: images upright, the robot state as
    # one vector, and the policy's action in the form the arm controller takes.
    env_preprocess, env_postprocess = make_env_pre_post_processors(env_config, config)
    print(f"smoke: loaded {POLICY} in {time.monotonic() - started:.1f} s", flush=True)

    started = time.monotonic()
    env = LiberoEnv(args.suite, args.task_id, control_mode=CONTROL_MODE)
    max_steps = args.max_steps or env.max_steps
    observation = env.reset(init_state=0, seed=args.seed)
    print(
        f"smoke: built {args.suite} task {args.task_id} in {time.monotonic() - started:.1f} s: "
        f"{env.task_description!r}",
        flush=True,
    )

    frames = [camera_frame(observation)]
    policy.reset()
    steps, done, success = 0, False, False
    started = time.monotonic()
    while not done and steps < max_steps:
        batch = preprocess_observation(as_batch_of_one(observation))
        batch["task"] = [env.task_description]
        batch = preprocess(env_preprocess(batch))
        with torch.inference_mode():
            action = postprocess(policy.select_action(batch))
        action = env_postprocess({ACTION: action})[ACTION]
        observation, done, success = env.step(action[0].to("cpu").numpy())
        frames.append(camera_frame(observation))
        steps += 1
    seconds = time.monotonic() - started
    env.close()

    write_mp4(args.out, frames, fps=CONTROL_HZ)
    size = args.out.stat().st_size
    print(
        f"smoke: episode ran {steps} of at most {max_steps} steps in {seconds:.1f} s "
        f"({steps / seconds:.0f} steps/s), task success: {success}"
    )
    print(f"smoke: wrote {args.out} ({len(frames)} frames, {size / 1024:.0f} KiB)")
    if size == 0:
        print("smoke: FAILED, the video file is empty", file=sys.stderr)
        return 1
    print("smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
