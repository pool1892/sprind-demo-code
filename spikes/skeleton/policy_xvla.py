"""X-VLA on LIBERO, built the way lerobot-eval builds it, with the sampler noise under our control.

Adapted from spikes/replay/policy_lib.py, which did the same for SmolVLA.

One difference that matters for exact replay (replay rule 2): LeRobot's X-VLA
wrapper accepts a `noise` argument and ignores it. `XVLAModel.generate_actions`
draws its starting noise from PyTorch's global generator on the GPU, one call
for the whole batch, so an episode's noise depends on what was computed before
and beside it. `chunks()` below repeats that method line for line and takes the
noise as an input, drawn per episode on the CPU from (noise seed, chunk number).
"""

from __future__ import annotations

import time

import numpy as np
import torch

import sk_common as C  # noqa: F401  (sets the environment variables)

POLICY = "lerobot/xvla-libero"


class Policy:
    def __init__(self, device: str = "cuda", suite: str = C.SUITE, gpu_cap_gib: float = 8.0):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        # lerobot-eval sets these two; kept so the numbers are those of the baseline's evaluation path.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        t0 = time.perf_counter()
        cfg = PreTrainedConfig.from_pretrained(POLICY, cli_overrides=[f"--device={device}"])
        cfg.pretrained_path = POLICY
        self.cfg, self.device = cfg, device
        self.env_cfg = LiberoEnvConfig(task=suite, control_mode="absolute")
        if device == "cuda":
            # GPU memory is system memory on this machine and the machine hangs when it fills.
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_cap_gib * 2**30 / total))
        for attempt in range(4):
            try:
                self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg)
                break
            except torch.AcceleratorError as e:
                # Seen in the replay spike right after another job had released the GPU.
                print(f"loading onto the GPU failed ({str(e).splitlines()[0]}), attempt {attempt + 1}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(10)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        self.chunk = cfg.chunk_size
        self.n_action_steps = cfg.n_action_steps
        self.model = self.policy.model
        self.noise_shape = (cfg.chunk_size, self.model.dim_action)
        if device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0

    def noise(self, noise_seed: int, chunk_index: int) -> torch.Tensor:
        """Starting noise for one episode's chunk: depends only on (noise_seed, chunk_index)."""
        g = torch.Generator(device="cpu")
        g.manual_seed(int(np.random.SeedSequence([noise_seed, chunk_index]).generate_state(1)[0]))
        return torch.randn(self.noise_shape, generator=g, dtype=torch.float32)

    def batch(self, obs_list: list[dict], tasks: list[str]) -> dict:
        """Stack per-environment observations the way a vector env would and preprocess them."""
        from lerobot.envs.utils import preprocess_observation

        o = preprocess_observation(_stack(obs_list))
        o["task"] = list(tasks)
        o = self.env_pre(o)
        return self.pre(o)

    @torch.no_grad()
    def chunks(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        """Action chunks (B, chunk, 20) on the CPU as float32.

        The body is XVLAModel.generate_actions with `x1` passed in instead of drawn.
        """
        m = self.model
        with torch.inference_mode():
            inputs = self.policy._build_model_inputs(dict(batch))
            dtype = m._get_target_dtype()
            image_input = inputs["image_input"].to(dtype=dtype)
            proprio = inputs["proprio"].to(dtype=dtype)
            enc = m.forward_vlm(inputs["input_ids"], image_input, inputs["image_mask"])
            b = inputs["input_ids"].shape[0]
            x1 = noise.to(device=proprio.device, dtype=dtype)
            assert x1.shape == (b, m.chunk_size, m.dim_action), x1.shape
            action = torch.zeros_like(x1)
            steps = max(1, int(self.cfg.num_denoising_steps))
            for i in range(steps, 0, -1):
                t = torch.full((b,), i / steps, device=proprio.device, dtype=dtype)
                x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
                proprio_m, x_t_m = m.action_space.preprocess(proprio, x_t)
                action = m.transformer(
                    domain_id=inputs["domain_id"], action_with_noise=x_t_m, proprio=proprio_m, t=t, **enc
                )
            out = m.action_space.postprocess(action)
        return out.float().cpu()

    @torch.no_grad()
    def to_env_actions(self, chunk: torch.Tensor) -> np.ndarray:
        """Turn (B, chunk, 20) into (B, n_action_steps, 7) environment actions, step by step as lerobot-eval does."""
        steps = []
        for t in range(self.n_action_steps):
            a = self.post(chunk[:, t].to(self.device))
            a = self.env_post({"action": a})["action"]
            steps.append(a.to("cpu").numpy())
        return np.stack(steps, axis=1).astype(np.float32)


def _stack(obs_list: list[dict]):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: _stack([o[k] for o in obs_list]) for k in first}
    return np.stack([np.asarray(o) for o in obs_list])


def bits_equal(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()
