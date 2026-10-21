"""pi0.5 (and pi0) on LIBERO with the interface of spikes/xvla-speed/policy_fast.Policy.

`lerobot/pi05_libero_finetuned_v044` (and `lerobot/pi0_libero_finetuned_v044`)
from the local copies named in p5_common.CKPT, whose preprocessor points at the
rebuilt tokenizer (make_tokenizer.py). Relative control, as LeRobot's LIBERO
environment gives it by default. LeRobot's own `predict_action_chunk` is called
with the starting noise passed in (`noise=`, which LeRobot's pi0 and pi0.5 accept
and forward to their sampling loop unchanged), so a chunk depends only on the
observation, the instruction, the noise and the batch it is computed in.

Options:
  model           "pi05" (default) or "pi0"
  dtype           "bfloat16" (as the checkpoint is configured and was trained) or "float32".
                  LeRobot's bfloat16 keeps SigLIP, the image projector and every RMS norm in
                  float32 and the two Gemma stacks and the projections in bfloat16.
  n_action_steps  actions executed per chunk (default 10, what LeRobot's LIBERO page uses for
                  pi0.5; the checkpoint says 50)
  steps           denoising steps (default: the checkpoint's 10)
  linear_t        store every linear layer's weight transposed (spikes/xvla-speed/policy_fast.py)
  benchmark       torch.backends.cudnn.benchmark (default False: the pinned MIOpen database decides)
  blas            None (PyTorch's choice) or a torch.backends.cuda.preferred_blas_library value:
                  "cublas" means rocBLAS on this machine, "cublaslt" hipBLASLt
  rowwise         compute a batch one row at a time (batch size 1 inside), so that a row's chunk
                  cannot depend on its position or its batch mates
  gpu_cap_gib     cap on PyTorch's GPU memory (default 12 for bfloat16, 20 for float32)

`chunks_timed` repeats LeRobot's PI05Pytorch.sample_actions line for line with the
GPU's queue flushed between the stages, to show where a call's time goes; it must
give the same bits as `chunks_staged` (profile_pi05.py checks that).
"""

from __future__ import annotations

import time

import numpy as np
import torch

import p5_common as C

DEFAULT_OPTS = {"model": "pi05", "dtype": "bfloat16", "n_action_steps": 10, "steps": None, "linear_t": False,
                "benchmark": False, "gpu_cap_gib": None, "blas": None, "rowwise": False}


class LinearT(torch.nn.Module):
    """nn.Linear with the weight stored transposed: y = x @ Wt + b (copied from policy_fast.py)."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.wt = torch.nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
        self.bias = lin.bias
        self.in_features, self.out_features = lin.in_features, lin.out_features

    @property
    def weight(self) -> torch.Tensor:  # some callers read .weight.dtype
        return self.wt.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.wt)
        return y + self.bias if self.bias is not None else y


def replace_linears(root: torch.nn.Module) -> int:
    """Swap every nn.Linear below `root` for a LinearT, one at a time so that each old weight is freed.

    The language model's output layer (`lm_head`, 0.5 B parameters) is left alone: pi0 and pi0.5
    never call it, and it shares its weight with the token embedding.
    """
    targets = [(pname, name) for pname, parent in root.named_modules() for name, child in parent.named_children()
               if type(child) is torch.nn.Linear and name != "lm_head"]
    for pname, name in targets:
        parent = root.get_submodule(pname)
        setattr(parent, name, LinearT(getattr(parent, name)))
    return len(targets)


def _stack(obs_list: list[dict]):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: _stack([o[k] for o in obs_list]) for k in first}
    return np.stack([np.asarray(o) for o in obs_list])


def bits_equal(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def load_obs(n: int | None = None):
    """The 80 observations spikes/xvla-speed/make_obs.py recorded: (list of obs, list of task descriptions)."""
    import pickle

    with open(C.OBS_PKL, "rb") as fh:
        recs = pickle.load(fh)
    recs = recs if n is None else recs[:n]
    return [r["obs"] for r in recs], [r["task_description"] for r in recs]


class Policy:
    def __init__(self, device: str = "cuda", suite: str = C.SUITE, gpu_cap_gib: float | None = None,
                 opts: dict | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        self.opts = {**DEFAULT_OPTS, **(opts or {})}
        unknown = set(self.opts) - set(DEFAULT_OPTS)
        assert not unknown, f"unknown options {unknown}"
        assert self.opts["dtype"] in ("bfloat16", "float32"), self.opts
        torch.backends.cudnn.benchmark = bool(self.opts["benchmark"])
        if self.opts["blas"]:
            torch.backends.cuda.preferred_blas_library(self.opts["blas"])
        self.blas_library = str(torch.backends.cuda.preferred_blas_library())
        torch.backends.cuda.matmul.allow_tf32 = True  # lerobot-eval sets it; no effect on this GPU's float32 that I know of
        path = str(C.CKPT[self.opts["model"]])
        overrides = [f"--device={device}", f"--dtype={self.opts['dtype']}", "--compile_model=false",
                     f"--n_action_steps={int(self.opts['n_action_steps'])}"]
        if self.opts["steps"]:
            overrides.append(f"--num_inference_steps={int(self.opts['steps'])}")
        t0 = time.perf_counter()
        cfg = PreTrainedConfig.from_pretrained(path, cli_overrides=overrides)
        cfg.pretrained_path = path
        self.cfg, self.device = cfg, device
        self.env_cfg = LiberoEnvConfig(task=suite)  # control_mode "relative"
        cap = gpu_cap_gib or self.opts["gpu_cap_gib"] or (12.0 if self.opts["dtype"] == "bfloat16" else 20.0)
        if device == "cuda":
            # GPU memory is system memory on this machine and the machine hangs when it fills.
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / total))
        for attempt in range(4):
            try:
                self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg)
                break
            except torch.AcceleratorError as e:
                print(f"loading onto the GPU failed ({str(e).splitlines()[0]}), attempt {attempt + 1}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(10)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        self.chunk = cfg.chunk_size
        self.n_action_steps = int(self.opts["n_action_steps"])
        assert self.policy.config.n_action_steps == self.n_action_steps
        self.noise_shape = (cfg.chunk_size, cfg.max_action_dim)
        self.n_linear_t = replace_linears(self.policy.model) if self.opts["linear_t"] else 0
        if device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.n_params = sum(p.numel() for p in self.policy.parameters())
        self.params_by_dtype: dict[str, int] = {}
        for p in self.policy.parameters():
            self.params_by_dtype[str(p.dtype)] = self.params_by_dtype.get(str(p.dtype), 0) + p.numel()

    def noise(self, noise_seed: int, chunk_index: int) -> torch.Tensor:
        """Noise for one episode's chunk: depends only on (noise_seed, chunk_index). As in policy_smolvla.py."""
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
        """Normalised action chunks (B, 50, 7) on the CPU as float32, through LeRobot's own call."""
        self.policy.reset()
        noise = noise.to(self.device)
        with torch.inference_mode():
            if self.opts["rowwise"]:
                n = noise.shape[0]
                rows = [{k: (v[i:i + 1] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == n
                             else v[i:i + 1] if isinstance(v, list) and len(v) == n else v)
                         for k, v in batch.items()} for i in range(n)]
                out = torch.cat([self.policy.predict_action_chunk(r, noise=noise[i:i + 1]) for i, r in enumerate(rows)])
            else:
                out = self.policy.predict_action_chunk(dict(batch), noise=noise)
        return out.float().cpu()

    @torch.no_grad()
    def chunks_timed(self, batch: dict, noise: torch.Tensor, times: dict) -> torch.Tensor:
        """As chunks_staged for pi0.5, with the time of every stage added to `times` (seconds)."""
        from lerobot.policies.common.flow_matching import euler_integrate
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        assert self.opts["model"] == "pi05"
        pol, m = self.policy, self.policy.model

        def mark(name, s):
            torch.cuda.synchronize()
            now = time.perf_counter()
            times[name] = times.get(name, 0.0) + now - s
            return now

        self.policy.reset()
        with torch.inference_mode():
            s = time.perf_counter()
            noise = noise.to(self.device)
            images, img_masks = pol._preprocess_images(dict(batch))
            tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
            s = mark("images_resize_normalise", s)
            # embed_prefix, split into the image part (SigLIP and projector) and the rest
            embs, pad_masks, att_masks = [], [], []
            for img, img_mask in zip(images, img_masks, strict=True):
                img_emb = m.paligemma_with_expert.embed_image(img)
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs
            s = mark("prefix_vision_siglip", s)
            lang_emb = m.paligemma_with_expert.embed_language_tokens(tokens)
            embs.append(lang_emb)
            pad_masks.append(masks)
            att_masks += [0] * lang_emb.shape[1]
            prefix_embs = torch.cat(embs, dim=1)
            prefix_pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
            prefix_att_masks = att_masks[None, :].expand(prefix_pad_masks.shape[0], len(att_masks))
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
            m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = m.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True,
            )
            s = mark("prefix_language_model", s)
            num_steps = m.config.num_inference_steps
            actions = euler_integrate(
                lambda x_t, t: m.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                                              x_t=x_t, timestep=t),
                noise, num_steps, rtc_processor=m.rtc_processor, rtc_enabled=m._rtc_enabled(),
            )
            s = mark("denoise_loop", s)
            actions = actions[:, :, : pol.config.output_features["action"].shape[0]]
            out = actions.float().cpu()
            mark("copy_out", s)
        times["prefix_tokens"] = int(prefix_pad_masks.shape[1])
        times["prefix_tokens_real_mean"] = float(prefix_pad_masks.float().sum(1).mean())
        return out

    @torch.no_grad()
    def to_env_actions(self, chunk: torch.Tensor) -> np.ndarray:
        """Unnormalise step by step, as lerobot-eval does: (B, n_action_steps, 7)."""
        steps = []
        for t in range(self.n_action_steps):
            a = self.post(chunk[:, t].to(self.device))
            a = self.env_post({"action": a})["action"]
            steps.append(a.to("cpu").numpy())
        return np.stack(steps, axis=1).astype(np.float32)
