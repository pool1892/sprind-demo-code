"""SmolVLA on LIBERO, set up as lerobot-eval sets it up, but with the noise under our control.

lerobot-eval lets the policy draw its sampler noise from torch's global
generator on the GPU, in one call for the whole batch. Here the noise is drawn
per episode from a generator seeded with (noise seed, chunk index) and handed
to the policy, so an episode's noise does not depend on batch size, row
position or call order.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

import common as C  # noqa: F401  (sets the environment variables)

POLICY = "lerobot/smolvla_libero"
RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


class Policy:
    def __init__(self, device: str = "cuda", suite: str = "libero_spatial"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        # lerobot-eval sets these two; kept so the numbers are those of the real evaluation path.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        cfg = PreTrainedConfig.from_pretrained(POLICY, cli_overrides=[f"--device={device}"])
        cfg.pretrained_path = POLICY
        self.cfg = cfg
        self.device = device
        self.env_cfg = LiberoEnvConfig(task=suite)
        if device == "cuda":
            # GPU memory is system memory on this machine and the machine hangs when it fills
            torch.cuda.set_per_process_memory_fraction(0.3)
            free, total = torch.cuda.mem_get_info()
            print(f"GPU memory free {free / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)
        for attempt in range(4):
            try:
                self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg, rename_map=RENAME)
                break
            except torch.AcceleratorError as e:
                # Seen twice on this machine right after another job had released the GPU:
                # "CUDA error: invalid argument" while the weights were copied to the device.
                print(f"loading the policy onto the GPU failed ({str(e).splitlines()[0]}), attempt {attempt + 1}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(10)
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
        self.chunk = cfg.chunk_size
        self.n_action_steps = cfg.n_action_steps
        self.noise_shape = (cfg.chunk_size, cfg.max_action_dim)

    def noise(self, noise_seed: int, chunk_index: int) -> torch.Tensor:
        """Noise for one episode's chunk: depends only on (noise_seed, chunk_index)."""
        g = torch.Generator(device="cpu")
        g.manual_seed(int(np.random.SeedSequence([noise_seed, chunk_index]).generate_state(1)[0]))
        return torch.randn(self.noise_shape, generator=g, dtype=torch.float32)

    def batch(self, obs_list: list[dict], tasks: list[str]) -> dict:
        """Stack per-environment observations the way a vector env would and preprocess them."""
        from lerobot.envs.utils import preprocess_observation

        stacked = _stack(obs_list)
        o = preprocess_observation(stacked)
        o["task"] = list(tasks)
        o = self.env_pre(o)
        return self.pre(o)

    @torch.no_grad()
    def chunks(self, batch: dict, noise: torch.Tensor | None, amp: str = "fp32") -> torch.Tensor:
        """Normalized action chunks (B, chunk, action_dim) on the CPU as float32."""
        ctx = {
            "fp32": contextlib.nullcontext(),
            "bf16": torch.autocast("cuda", dtype=torch.bfloat16),
            "fp16": torch.autocast("cuda", dtype=torch.float16),
        }[amp]
        if noise is not None:
            noise = noise.to(self.device)
        self.policy.reset()
        with torch.inference_mode(), ctx:
            out = self.policy.predict_action_chunk(dict(batch), noise=noise)
        return out.float().cpu()

    @torch.no_grad()
    def to_env_actions(self, chunk: torch.Tensor) -> np.ndarray:
        """Unnormalize (B, chunk, action_dim) step by step, as lerobot-eval does, to numpy."""
        steps = []
        for t in range(self.n_action_steps):
            a = self.post(chunk[:, t].to(self.device))
            a = self.env_post({"action": a})["action"]
            steps.append(a.to("cpu").numpy())
        return np.stack(steps, axis=1)


def _stack(obs_list: list[dict]):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: _stack([o[k] for o in obs_list]) for k in first}
    return np.stack([np.asarray(o) for o in obs_list])


def bits_equal(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()
