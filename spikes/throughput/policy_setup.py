"""SmolVLA for LIBERO, built the way lerobot-eval builds it (see lerobot_eval.py, eval_main)."""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

POLICY = "lerobot/smolvla_libero"
# The mapping used in the known 58-61% runs: two of the checkpoint's three cameras.
RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def autocast(amp: str, device: str = "cuda"):
    if amp == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[amp])


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


class Policy:
    def __init__(self, device: str = "cuda", suite: str = "libero_spatial"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        t0 = time.perf_counter()
        cfg = PreTrainedConfig.from_pretrained(POLICY, cli_overrides=[f"--device={device}"])
        cfg.pretrained_path = POLICY
        self.cfg, self.device = cfg, device
        self.env_cfg = LiberoEnvConfig(task=suite)
        self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg, rename_map=RENAME)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": RENAME},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        sync(device)
        self.load_s = time.perf_counter() - t0

    def preprocess(self, stacked_obs: dict, tasks: list[str]) -> dict:
        """Vector-env style observation (numpy, batch first) to the policy's input batch on the device."""
        from lerobot.envs.utils import preprocess_observation

        o = preprocess_observation(stacked_obs)
        o["task"] = list(tasks)
        o = self.env_pre(o)
        return self.pre(o)

    @torch.no_grad()
    def chunks(self, batch: dict, amp: str = "fp32", noise: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized action chunks, shape (batch, chunk_size, action_dim), still on the device."""
        self.policy.reset()
        with torch.inference_mode(), autocast(amp, self.device):
            return self.policy.predict_action_chunk(dict(batch), noise=noise)

    @torch.no_grad()
    def to_env_actions(self, chunk: torch.Tensor) -> np.ndarray:
        """Unnormalize a whole chunk at once: (batch, steps, 7) numpy actions for env.step."""
        b, t, d = chunk.shape
        a = self.post(chunk.reshape(b * t, d).float())
        a = self.env_post({"action": a})["action"]
        return a.reshape(b, t, -1).to("cpu").numpy()


def stack_obs(obs_list: list[dict]):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: stack_obs([o[k] for o in obs_list]) for k in first}
    return np.stack([np.asarray(o) for o in obs_list])


def sample_obs(res: int = 360, task_id: int = 0):
    """One real observation and its instruction, from a freshly reset environment."""
    import common

    env = common.make_env(task_id=task_id, res=res)
    obs, _ = env.reset(seed=0)
    task = env.task_description
    env.close()
    return obs, task
