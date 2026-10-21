"""X-VLA on LIBERO with the sampler noise passed in, split into stages, with switches for every speed-up tried.

Copied from spikes/skeleton/policy_xvla.py and extended. `chunks()` is that file's
method unchanged: it repeats LeRobot's XVLAModel.generate_actions line for line and
takes the starting noise as an input (replay rule 2). It is the reference every
variant is compared with ("today's output").

`Policy(opts=...)` switches, all off by default so that the default object is today's:

  benchmark      torch.backends.cudnn.benchmark (default True, as lerobot-eval sets it)
  channels_last  vision tower weights and images in channels-last memory format
  attn           attention in the action transformer: "sdpa" (PyTorch's fused call, today) or "manual"
  hoist          compute once per call what the sampling loop recomputes at every step
  half           which parts hold their weights and compute in 16-bit floats, for example
                 {"transformer": "fp16", "vision": "bf16"}; parts are "transformer" (the
                 sampling loop), "vision" (tower and projector) and "encoder" (language)
  residual32     with half["transformer"]: only the attention and MLP sub-layers run in 16
                 bits; the stream they add to, the layer norms and everything outside the
                 24 blocks stay in 32 bits
  amp            None, "bf16" or "fp16": PyTorch's automatic mixed precision instead, which
                 leaves the weights in 32 bits and converts them at every use ...
  amp_parts      ... for these parts
  linear_t       store every linear layer's weight transposed and multiply without the
                 "transpose" flag. In 16-bit floats PyTorch's usual call gives a row another
                 result in another row of the batch for some layer shapes; this form does not
                 (rows_diag2.py). Same arithmetic, other bits than the usual call.
  compile        None or a torch.compile mode for the action transformer's blocks
  steps          sampling steps (default: the checkpoint's 10)
  views          None (today) or 3. The checkpoint's configuration says three image views:
                 scene camera, wrist camera, one empty. LeRobot 0.6.1 builds four, because it
                 counts the empty view twice, so the action transformer carries 50 tokens of
                 zeros more than the configuration asks for. 3 removes them. This changes
                 what the policy computes.
  n_action_steps actions executed per chunk (default: the checkpoint's 30)
  fast_post      turn a chunk into environment actions in one pass instead of 30

The stage methods (`model_inputs`, `vision`, `encode`, `sample`, `finish`) exist so
that a call can be timed stage by stage; `chunks_staged` chains them.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

import xs_common as C  # noqa: F401  (sets the environment variables)

POLICY = "lerobot/xvla-libero"
DEFAULT_OPTS = {
    "benchmark": True, "channels_last": False, "attn": "sdpa", "hoist": False, "amp": None,
    "amp_parts": (), "half": None, "residual32": False, "linear_t": False, "compile": None, "steps": None,
    "views": None,
    "n_action_steps": None, "fast_post": False,
}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class Policy:
    def __init__(self, device: str = "cuda", suite: str = C.SUITE, gpu_cap_gib: float = 8.0, opts: dict | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        self.opts = {**DEFAULT_OPTS, **(opts or {})}
        unknown = set(self.opts) - set(DEFAULT_OPTS)
        assert not unknown, f"unknown options {unknown}"
        # lerobot-eval sets these two; kept so the numbers are those of the baseline's evaluation path.
        torch.backends.cudnn.benchmark = bool(self.opts["benchmark"])
        torch.backends.cuda.matmul.allow_tf32 = True

        t0 = time.perf_counter()
        overrides = [f"--device={device}"]
        if self.opts["views"] is not None:
            assert self.opts["views"] == 3, "only the configured three views can be selected"
            overrides.append("--empty_cameras=0")
        cfg = PreTrainedConfig.from_pretrained(POLICY, cli_overrides=overrides)
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
        self.n_action_steps = int(self.opts["n_action_steps"] or cfg.n_action_steps)
        assert 1 <= self.n_action_steps <= cfg.chunk_size
        self.steps = max(1, int(self.opts["steps"] or cfg.num_denoising_steps))
        self.model = self.policy.model
        self.noise_shape = (cfg.chunk_size, self.model.dim_action)

        m = self.model
        if self.opts["channels_last"]:
            m.vlm.vision_tower.to(memory_format=torch.channels_last)
        if self.opts["attn"] == "manual":
            for blk in m.transformer.blocks:
                blk.attn.fused_attn = False
        half = dict(self.opts["half"] or {})
        assert set(half) <= {"transformer", "vision", "encoder"}, half
        self.dt = {part: DTYPES[half[part]] if part in half else torch.float32
                   for part in ("transformer", "vision", "encoder")}
        if "vision" in half:
            m.vlm.vision_tower.to(dtype=self.dt["vision"])
            m.vlm.multi_modal_projector.to(dtype=self.dt["vision"])
        if "encoder" in half:
            m.vlm.language_model.to(dtype=self.dt["encoder"])
        if "transformer" in half:
            if self.opts["residual32"]:
                for blk in m.transformer.blocks:
                    blk.attn.to(dtype=self.dt["transformer"])
                    blk.mlp.to(dtype=self.dt["transformer"])
            else:
                m.transformer.to(dtype=self.dt["transformer"])
        if self.opts["linear_t"]:
            self.n_linear_t = sum(replace_linears(root) for root in (m.vlm, m.transformer))
        self._blocks = list(m.transformer.blocks)
        if self.opts["compile"]:
            self._blocks = [torch.compile(b, mode=self.opts["compile"]) for b in self._blocks]
        if device == "cuda":
            torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0

    # ---- inputs -------------------------------------------------------------------------------

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

    # ---- today's call, unchanged ------------------------------------------------------------------

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
        if self.opts["fast_post"]:
            return self.to_env_actions_fast(chunk)
        steps = []
        for t in range(self.n_action_steps):
            a = self.post(chunk[:, t].to(self.device))
            a = self.env_post({"action": a})["action"]
            steps.append(a.to("cpu").numpy())
        return np.stack(steps, axis=1).astype(np.float32)

    @torch.no_grad()
    def to_env_actions_fast(self, chunk: torch.Tensor) -> np.ndarray:
        """The same conversion in one pass on the CPU.

        The stored postprocessor is the identity on actions (no normalisation) followed
        by a move to the CPU, so only the environment's conversion (6D rotation to axis
        angle, gripper threshold) is left; it works row by row, so all steps go in at once.
        """
        b = chunk.shape[0]
        flat = chunk[:, : self.n_action_steps].reshape(b * self.n_action_steps, -1).contiguous()
        a = self.env_post({"action": flat})["action"]
        return a.numpy().reshape(b, self.n_action_steps, -1).astype(np.float32)

    # ---- the same call in stages ------------------------------------------------------------------

    def _amp(self, part: str):
        on = self.opts["amp"] is not None and part in self.opts["amp_parts"]
        return torch.autocast("cuda", dtype=DTYPES[self.opts["amp"]] if on else None, enabled=on)

    def model_inputs(self, batch: dict) -> dict:
        """Resize both images to 224x224 on the GPU, add the empty third view, pad the state."""
        return self.policy._build_model_inputs(dict(batch))

    def vision(self, inputs: dict) -> torch.Tensor:
        """Florence-2's vision tower and projector on the real views: (B, views, 50, 1024)."""
        m = self.model
        pixel_values = inputs["image_input"].to(dtype=self.dt["vision"])
        image_mask = inputs["image_mask"]
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        valid_images = flat_images[flat_mask]
        if self.opts["channels_last"]:
            valid_images = valid_images.contiguous(memory_format=torch.channels_last)
        with self._amp("vision"):
            valid_feats = m.vlm.get_image_features(valid_images).pooler_output
        valid_feats = valid_feats.float()
        tokens_per_view, hidden_dim = valid_feats.shape[1:]
        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        return image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)

    def encode(self, inputs: dict, image_features: torch.Tensor) -> dict:
        """Token embeddings and Florence-2's language encoder over the scene view and the instruction."""
        m = self.model
        batch_size = image_features.shape[0]
        hidden_dim = image_features.shape[-1]
        inputs_embeds = m.vlm.get_input_embeddings()(inputs["input_ids"])
        merged_embeds = torch.cat([image_features[:, 0].to(inputs_embeds.dtype), inputs_embeds], dim=1)
        attention_mask = torch.ones(merged_embeds.shape[:2], dtype=torch.long, device=merged_embeds.device)
        with self._amp("encoder"):
            enc_out = m.vlm.language_model.encoder(attention_mask=attention_mask, inputs_embeds=merged_embeds)[0]
        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out.float(), "aux_visual_inputs": aux_visual_inputs}

    def sample(self, inputs: dict, enc: dict, noise: torch.Tensor, step_times: list | None = None) -> torch.Tensor:
        """The sampling loop: `steps` passes of the action transformer. Returns (B, chunk, 20) on the GPU."""
        m = self.model
        dtype = m._get_target_dtype()
        proprio = inputs["proprio"].to(dtype=dtype)
        b = inputs["input_ids"].shape[0]
        x1 = noise.to(device=proprio.device, dtype=dtype)
        assert x1.shape == (b, m.chunk_size, m.dim_action), x1.shape
        action = torch.zeros_like(x1)
        steps = self.steps
        hoisted = self._context(inputs, enc) if self.opts["hoist"] else None
        for i in range(steps, 0, -1):
            if step_times is not None:
                torch.cuda.synchronize()
                s = time.perf_counter()
            t = torch.full((b,), i / steps, device=proprio.device, dtype=dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = m.action_space.preprocess(proprio, x_t)
            if hoisted is None:
                if self._plain_transformer():
                    action = m.transformer(
                        domain_id=inputs["domain_id"], action_with_noise=x_t_m, proprio=proprio_m, t=t, **enc
                    )
                else:
                    action = self._transformer(inputs["domain_id"], x_t_m, proprio_m, t, enc, None)
            else:
                action = self._transformer(inputs["domain_id"], x_t_m, proprio_m, t, enc, hoisted)
            if step_times is not None:
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - s)
        return m.action_space.postprocess(action)

    def finish(self, action: torch.Tensor) -> torch.Tensor:
        return action.float().cpu()

    @torch.no_grad()
    def chunks_staged(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            inputs = self.model_inputs(batch)
            feats = self.vision(inputs)
            enc = self.encode(inputs, feats)
            return self.finish(self.sample(inputs, enc, noise))

    # ---- the action transformer, re-stated so that parts of it can be changed -----------------------

    def _plain_transformer(self) -> bool:
        o = self.opts
        return not (o["compile"] or "transformer" in (o["half"] or {}) or ("transformer" in o["amp_parts"] and o["amp"]))

    def _context(self, inputs: dict, enc: dict) -> dict:
        """Everything in SoftPromptedTransformer.forward that does not depend on the sampling step."""
        tr = self.model.transformer
        assert not tr.use_hetero_proj
        b = inputs["input_ids"].shape[0]
        n_act = self.model.chunk_size
        with self._amp("transformer"):
            ctx = torch.cat([tr.vlm_proj(self._cast(enc["vlm_features"])),
                             tr.aux_visual_proj(self._cast(enc["aux_visual_inputs"]))], dim=1)
            seq_len = n_act + ctx.shape[1]
            ctx = ctx + tr.pos_emb[:, n_act:seq_len, :]
            soft = tr.soft_prompt_hub(inputs["domain_id"]).view(b, tr.len_soft_prompts, tr.hidden_size)
        return {"ctx": ctx, "soft": soft, "pos_act": tr.pos_emb[:, :n_act, :]}

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        """To the action transformer's number format, unless only its sub-layers are in 16 bits."""
        return x if self.opts["residual32"] else x.to(self.dt["transformer"])

    def _block(self, block, x: torch.Tensor) -> torch.Tensor:
        if not self.opts["residual32"] or self.dt["transformer"] is torch.float32:
            return block(x)
        hd = self.dt["transformer"]
        x = x + block.attn(block.norm1(x).to(hd)).float()
        return x + block.mlp(block.norm2(x).to(hd)).float()

    def _transformer(self, domain_id, action_with_noise, proprio, t, enc, hoisted) -> torch.Tensor:
        """SoftPromptedTransformer.forward, with the step-independent part optionally passed in."""
        from lerobot.policies.xvla.soft_transformer import timestep_embedding

        tr = self.model.transformer
        b, n_act = action_with_noise.shape[:2]
        with self._amp("transformer"):
            time_emb = timestep_embedding(t, tr.dim_time)
            time_tokens = time_emb.unsqueeze(1).expand(b, n_act, tr.dim_time)
            proprio_tokens = proprio.unsqueeze(1).expand(b, n_act, proprio.shape[-1])
            action_tokens = self._cast(torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1))
            x = tr.action_encoder(action_tokens, domain_id)
            if hoisted is None:
                x = torch.cat([x, tr.vlm_proj(self._cast(enc["vlm_features"])),
                               tr.aux_visual_proj(self._cast(enc["aux_visual_inputs"]))], dim=1)
                x = x + tr.pos_emb[:, : x.shape[1], :]
                soft = tr.soft_prompt_hub(domain_id).view(b, tr.len_soft_prompts, tr.hidden_size)
                x = torch.cat([x, soft], dim=1)
            else:
                x = torch.cat([x + hoisted["pos_act"], hoisted["ctx"], hoisted["soft"]], dim=1)
            for block in self._blocks:
                x = self._block(block, x)
            out = tr.action_decoder(tr.norm(x[:, :n_act]), domain_id)
        return out.float()


class LinearT(torch.nn.Module):
    """nn.Linear with the weight stored transposed: y = x @ Wt + b, as one product over all leading dimensions."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.wt = torch.nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().clone(), requires_grad=False)

    @property
    def weight(self) -> torch.Tensor:
        """The weight in nn.Linear's layout, for code that asks a layer for its number format or device."""
        return self.wt.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features)
        y = torch.mm(flat, self.wt) if self.bias is None else torch.addmm(self.bias, flat, self.wt)
        return y.view(*x.shape[:-1], self.out_features)


def replace_linears(root: torch.nn.Module) -> int:
    """Swap every nn.Linear below `root` for a LinearT. Returns how many."""
    n = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is torch.nn.Linear:
                setattr(parent, name, LinearT(child))
                n += 1
    return n


def _stack(obs_list: list[dict]):
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: _stack([o[k] for o in obs_list]) for k in first}
    return np.stack([np.asarray(o) for o in obs_list])


def bits_equal(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def load_obs(n: int | None = None):
    """The observations make_obs.py recorded: (list of obs, list of task descriptions)."""
    import pickle

    with open(C.OUT / "obs" / "obs.pkl", "rb") as fh:
        recs = pickle.load(fh)
    recs = recs if n is None else recs[:n]
    return [r["obs"] for r in recs], [r["task_description"] for r in recs]


__all__ = ["Policy", "bits_equal", "load_obs", "F"]
