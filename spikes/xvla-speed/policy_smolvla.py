"""SmolVLA on LIBERO as the second policy, with the same interface as policy_fast.Policy.

Adapted from spikes/replay/policy_lib.py. `lerobot/smolvla_libero` is used as the
baseline spike corrected it: 10 of the 50 predicted actions are executed before the
policy looks again (`n_action_steps`), cameras `image` and `image2` are fed as
`camera1` and `camera2`, relative control. SmolVLA honours the `noise` argument,
so LeRobot's own `predict_action_chunk` is called.

Options:
  n_action_steps  actions executed per chunk (default 10, the corrected value)
  linear_t        store every linear layer's weight transposed (see policy_fast.py)
  half            None, "fp16" or "bf16": convert the weights to 16-bit floats
  amp             None, "fp16" or "bf16": automatic mixed precision instead
  benchmark       torch.backends.cudnn.benchmark (default False here: pinned database)
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

import xs_common as C  # noqa: F401  (sets the environment variables)
from policy_fast import DTYPES, _stack, replace_linears

POLICY = "lerobot/smolvla_libero"
RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
DEFAULT_OPTS = {"n_action_steps": 10, "linear_t": False, "half": None, "amp": None, "benchmark": False}


class Policy:
    def __init__(self, device: str = "cuda", suite: str = C.SUITE, gpu_cap_gib: float = 8.0, opts: dict | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        self.opts = {**DEFAULT_OPTS, **(opts or {})}
        assert not set(self.opts) - set(DEFAULT_OPTS), self.opts
        torch.backends.cudnn.benchmark = bool(self.opts["benchmark"])
        torch.backends.cuda.matmul.allow_tf32 = True
        t0 = time.perf_counter()
        cfg = PreTrainedConfig.from_pretrained(POLICY, cli_overrides=[f"--device={device}"])
        cfg.pretrained_path = POLICY
        self.cfg, self.device = cfg, device
        self.env_cfg = LiberoEnvConfig(task=suite)
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_cap_gib * 2**30 / total))
        for attempt in range(4):
            try:
                self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg, rename_map=RENAME)
                break
            except torch.AcceleratorError as e:
                print(f"loading onto the GPU failed ({str(e).splitlines()[0]}), attempt {attempt + 1}", flush=True)
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
        self.n_action_steps = int(self.opts["n_action_steps"])
        self.noise_shape = (cfg.chunk_size, cfg.max_action_dim)
        if self.opts["half"]:
            self.policy.model.to(dtype=DTYPES[self.opts["half"]])
        if self.opts["linear_t"]:
            self.n_linear_t = replace_linears(self.policy.model)
        if device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0

    def noise(self, noise_seed: int, chunk_index: int) -> torch.Tensor:
        """Noise for one episode's chunk: depends only on (noise_seed, chunk_index)."""
        g = torch.Generator(device="cpu")
        g.manual_seed(int(np.random.SeedSequence([noise_seed, chunk_index]).generate_state(1)[0]))
        return torch.randn(self.noise_shape, generator=g, dtype=torch.float32)

    def batch(self, obs_list: list[dict], tasks: list[str]) -> dict:
        from lerobot.envs.utils import preprocess_observation

        o = preprocess_observation(_stack(obs_list))
        o["task"] = list(tasks)
        o = self.env_pre(o)
        return self.pre(o)

    @torch.no_grad()
    def chunks_staged(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        """Normalised action chunks (B, 50, action_dim) on the CPU as float32."""
        amp = self.opts["amp"]
        ctx = torch.autocast("cuda", dtype=DTYPES[amp]) if amp else contextlib.nullcontext()
        noise = noise.to(self.device)
        if self.opts["half"]:
            noise = noise.to(DTYPES[self.opts["half"]])
        self.policy.reset()
        with torch.inference_mode(), ctx:
            out = self.policy.predict_action_chunk(dict(batch), noise=noise)
        return out.float().cpu()

    @torch.no_grad()
    def to_env_actions(self, chunk: torch.Tensor) -> np.ndarray:
        """Unnormalise step by step, as lerobot-eval does: (B, n_action_steps, 7)."""
        steps = []
        for t in range(self.n_action_steps):
            a = self.post(chunk[:, t].to(self.device))
            a = self.env_post({"action": a})["action"]
            steps.append(a.to("cpu").numpy())
        return np.stack(steps, axis=1).astype(np.float32)
