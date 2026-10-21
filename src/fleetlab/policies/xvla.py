"""X-VLA (`lerobot/xvla-libero`) behind `contracts.policy.Policy`, with our own sampling loop.

LeRobot 0.6.1's X-VLA accepts a noise argument and ignores it: its sampler
draws the starting noise from PyTorch's global generator, once for the whole
batch, so a row's noise depends on what ran before and beside it.
`sample` below repeats the body of `XVLAModel.generate_actions` with the noise
handed in, one row per episode, from `contracts.identity.noise_for` (replay
rule 2). The code is the X-VLA speed spike's (`spikes/xvla-speed/
policy_fast.py`), cut to the settings the fleet runs:

- `fp32`: the checkpoint's arithmetic in 32-bit floats; the context of the
  sampling loop computed once per call instead of once per step ("hoist").
- `fp16_res32`: every weight in float16; in the action transformer only the
  attention and MLP sub-layers compute in 16 bits, each token's running sum
  and the layer norms stay in 32 bits; every linear weight stored transposed,
  because in 16 bits PyTorch's usual product gives a row other bits in other
  rows of a batch for some layer shapes, and the transposed form does not
  (xvla-speed sections 2 and 3). 3.3 times faster per call, no difference in
  success found on 1,200 paired episodes.
- `fp16`: as `fp16_res32` with the whole action transformer in 16 bits.

`row_independent` is True for all three, verified by the rows check
(`fleetlab.policies.rows_check`, `make check-model`), not assumed. The
batch size changes the bits, so it is fixed per run (rule 3); convolution
algorithms come from the pinned database with `cudnn.benchmark` off (rule 7,
`fleetlab.engine.miopen`).

Knobs read from the effective spec the call carries (the group's, after the
policy input hooks): `sampling_steps` (the checkpoint has 10)
and `image_views` (LeRobot builds 4 views, scene, wrist and two empty ones,
where the checkpoint's configuration asks for 3; 3 drops one empty view,
which changes what the policy computes; xvla-speed section 6). `convert`
turns a chunk into absolute target poses (position, axis-angle, gripper)
whatever the control mode: X-VLA only commands target poses, so under a
relative control mode the controller reads them as changes of pose, which is
what a misconfigured controller would do.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from fleetlab.contracts.env import Observation
from fleetlab.contracts.identity import PolicySpec
from fleetlab.contracts.policy import Action7, ActionChunk, ControlMode, NoiseSample

CHECKPOINT = "lerobot/xvla-libero"
REVISION = "12e8783e996944f5c97e490d37d4c145484ed70a"
PRECISIONS = ("fp32", "fp16", "fp16_res32")
CHUNK = 30
ACTION_WIDTH = 20
VIEWS = 4  # what LeRobot 0.6.1 builds for this checkpoint
SAMPLING_STEPS = 10


def nominal_spec(precision: str = "fp16_res32") -> PolicySpec:
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; known: {PRECISIONS}")
    return PolicySpec(
        name="xvla-libero",
        revision=REVISION[:12],
        precision=precision,
        control_mode=ControlMode.absolute,
        actions_executed_per_chunk=CHUNK,
        sampling_steps=SAMPLING_STEPS,
        image_views=VIEWS,
    )


class XVLAPolicy:
    """X-VLA with the noise handed in, in one of `PRECISIONS`. A server policy."""

    chunk_length = CHUNK
    needs_noise = True
    noise_shape: tuple[int, ...] | None = (CHUNK, ACTION_WIDTH)
    noise_dtype: str | None = "float32"
    row_independent = True  # verified by the rows check (make check-model)
    in_worker = False
    image_views = VIEWS

    def __init__(self) -> None:
        self.spec = nominal_spec()
        self.load_s = 0.0
        self._loaded = False

    # ---- loading -----------------------------------------------------------------------------

    def load(self, device: str, precision: str, *, benchmark: bool = False) -> None:
        """Build the model as lerobot-eval does, then apply the precision's changes.

        `benchmark=True` lets MIOpen time its convolution algorithms; only the
        pinning search (`engine.miopen.pin`) asks for it.
        """
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the checkpoint is cached
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        from fleetlab.runtime import cap_gpu_memory

        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision {precision!r}; known: {PRECISIONS}")
        t0 = time.perf_counter()
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = True  # as lerobot-eval sets it
        if device == "cuda":
            cap_gpu_memory()
        cfg = PreTrainedConfig.from_pretrained(CHECKPOINT, cli_overrides=[f"--device={device}"])
        cfg.pretrained_path = CHECKPOINT
        env_cfg = LiberoEnvConfig(task="libero_spatial", control_mode="absolute")
        for attempt in range(4):
            try:
                self._policy = make_policy(cfg=cfg, env_cfg=env_cfg)
                break
            except torch.AcceleratorError:  # seen right after another job released the GPU
                if attempt == 3:
                    raise
                time.sleep(10)
        self._policy.eval()
        self._pre, _ = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self._env_pre, self._env_post = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=cfg
        )
        self._cfg, self._device, self._torch = cfg, device, torch
        model = self._policy.model
        self._model = model
        if (cfg.chunk_size, model.dim_action) != (CHUNK, ACTION_WIDTH):
            raise RuntimeError(f"unexpected chunk shape {(cfg.chunk_size, model.dim_action)}")
        if int(cfg.num_denoising_steps) != SAMPLING_STEPS:
            raise RuntimeError(f"the checkpoint samples in {cfg.num_denoising_steps} steps")
        half = precision != "fp32"
        self._residual32 = precision == "fp16_res32"
        f16, f32 = torch.float16, torch.float32
        self._dt = {
            "transformer": f16 if half else f32,
            "vision": f16 if half else f32,
            "encoder": f16 if half else f32,
        }
        if half:
            model.vlm.vision_tower.to(dtype=f16)
            model.vlm.multi_modal_projector.to(dtype=f16)
            model.vlm.language_model.to(dtype=f16)
            if self._residual32:
                for block in model.transformer.blocks:
                    block.attn.to(dtype=f16)
                    block.mlp.to(dtype=f16)
            else:
                model.transformer.to(dtype=f16)
            self.linear_t = sum(replace_linears(root) for root in (model.vlm, model.transformer))
        self._blocks = list(model.transformer.blocks)
        if device == "cuda":
            torch.cuda.synchronize()
        self.spec = nominal_spec(precision)
        self._nominal = self.spec
        self._loaded = True
        self.load_s = time.perf_counter() - t0

    # ---- Policy ------------------------------------------------------------------------------

    def act(
        self,
        batch: Sequence[Observation],
        instructions: Sequence[str],
        noise: Sequence[NoiseSample | None],
        spec: PolicySpec,
    ) -> list[ActionChunk]:
        torch = self._torch
        if any(n is None for n in noise):
            raise ValueError("X-VLA samples: every row needs its noise (replay rule 2)")
        self._check_spec(spec)
        x1 = torch.stack([torch.from_numpy(np.asarray(n.values, np.float32)) for n in noise if n])
        with torch.inference_mode():
            prepared = self._prepare(batch, instructions)
            inputs = self._policy._build_model_inputs(dict(prepared))
            if spec.image_views < inputs["image_input"].shape[1]:
                views = spec.image_views
                inputs["image_input"] = inputs["image_input"][:, :views].contiguous()
                inputs["image_mask"] = inputs["image_mask"][:, :views].contiguous()
            feats = self._vision(inputs)
            enc = self._encode(inputs, feats)
            out = self._sample(inputs, enc, x1, int(spec.sampling_steps or SAMPLING_STEPS))
            actions = out.float().cpu().numpy()
        return [
            ActionChunk(chunk_index=n.chunk_index if n else 0, actions=actions[i])
            for i, n in enumerate(noise)
        ]

    def act_one(self, observation: Observation, env: Any) -> Action7:
        raise NotImplementedError("X-VLA is a server policy")

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        """Target poses: position, axis-angle (from the 6D rotation), gripper -1 or 1."""
        torch = self._torch
        with torch.inference_mode():
            flat = torch.from_numpy(np.ascontiguousarray(chunk.actions, np.float32))
            converted = self._env_post({"action": flat})["action"]
        array = np.asarray(converted.numpy() if hasattr(converted, "numpy") else converted)
        return [row.astype(np.float32) for row in array.reshape(len(chunk.actions), -1)]

    # ---- for the pinning search --------------------------------------------------------------

    def time_vision(self, observations: Sequence[tuple[Observation, str]]) -> float:
        """Run the vision tower once on these rows; returns the seconds the first call took."""
        torch = self._torch
        with torch.inference_mode():
            prepared = self._prepare([o for o, _ in observations], [t for _, t in observations])
            inputs = self._policy._build_model_inputs(dict(prepared))
            torch.cuda.synchronize()
            s = time.perf_counter()
            self._vision(inputs)
            torch.cuda.synchronize()
        return time.perf_counter() - s

    # ---- inside ------------------------------------------------------------------------------

    def _check_spec(self, spec: PolicySpec) -> None:
        nominal = self._nominal
        if (spec.name, spec.revision, spec.precision) != (
            nominal.name,
            nominal.revision,
            nominal.precision,
        ):
            raise ValueError(f"this policy is {nominal.name} {nominal.precision}, asked for {spec}")
        if not 1 <= spec.image_views <= VIEWS:
            raise ValueError(f"image views must be 1 to {VIEWS}, not {spec.image_views}")

    def _prepare(self, batch: Sequence[Observation], instructions: Sequence[str]) -> Any:
        from lerobot.envs.utils import preprocess_observation

        stacked = _stack([{"pixels": o["pixels"], "robot_state": o["robot_state"]} for o in batch])
        prepared = preprocess_observation(stacked)
        prepared["task"] = list(instructions)
        return self._pre(self._env_pre(prepared))

    def _vision(self, inputs: dict[str, Any]) -> Any:
        """Florence-2's vision tower and projector on the real views: (B, views, 50, 1024)."""
        model = self._model
        pixel_values = inputs["image_input"].to(dtype=self._dt["vision"])
        image_mask = inputs["image_mask"]
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=self._torch.bool)
        valid = pixel_values.flatten(0, 1)[flat_mask]
        feats = model.vlm.get_image_features(valid).pooler_output.float()
        tokens, hidden = feats.shape[1:]
        image_features = feats.new_zeros((batch_size * num_views, tokens, hidden))
        image_features[flat_mask] = feats
        return image_features.view(batch_size, num_views, tokens, hidden)

    def _encode(self, inputs: dict[str, Any], image_features: Any) -> dict[str, Any]:
        """Token embeddings and the language encoder over the scene view and the instruction."""
        model = self._model
        torch = self._torch
        batch_size, hidden = image_features.shape[0], image_features.shape[-1]
        embeds = model.vlm.get_input_embeddings()(inputs["input_ids"])
        merged = torch.cat([image_features[:, 0].to(embeds.dtype), embeds], dim=1)
        mask = torch.ones(merged.shape[:2], dtype=torch.long, device=merged.device)
        out = model.vlm.language_model.encoder(attention_mask=mask, inputs_embeds=merged)[0]
        aux = image_features[:, 1:].reshape(batch_size, -1, hidden)
        return {"vlm_features": out.float(), "aux_visual_inputs": aux}

    def _sample(self, inputs: dict[str, Any], enc: dict[str, Any], noise: Any, steps: int) -> Any:
        """The sampling loop of `XVLAModel.generate_actions`, with the starting noise handed in."""
        model = self._model
        torch = self._torch
        dtype = model._get_target_dtype()
        proprio = inputs["proprio"].to(dtype=dtype)
        b = inputs["input_ids"].shape[0]
        x1 = noise.to(device=proprio.device, dtype=dtype)
        if tuple(x1.shape) != (b, model.chunk_size, model.dim_action):
            raise ValueError(f"noise of shape {tuple(x1.shape)} for a batch of {b}")
        action = torch.zeros_like(x1)
        context = self._context(inputs, enc)
        for i in range(steps, 0, -1):
            t = torch.full((b,), i / steps, device=proprio.device, dtype=dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = model.action_space.preprocess(proprio, x_t)
            action = self._transformer(inputs["domain_id"], x_t_m, proprio_m, t, context)
        return model.action_space.postprocess(action)

    def _cast(self, x: Any) -> Any:
        return x if self._residual32 else x.to(self._dt["transformer"])

    def _context(self, inputs: dict[str, Any], enc: dict[str, Any]) -> dict[str, Any]:
        """What `SoftPromptedTransformer.forward` computes the same at every sampling step."""
        tr = self._model.transformer
        torch = self._torch
        if tr.use_hetero_proj:
            raise RuntimeError("heterogeneous projections are not supported")
        b = inputs["input_ids"].shape[0]
        n_act = self._model.chunk_size
        ctx = torch.cat(
            [
                tr.vlm_proj(self._cast(enc["vlm_features"])),
                tr.aux_visual_proj(self._cast(enc["aux_visual_inputs"])),
            ],
            dim=1,
        )
        ctx = ctx + tr.pos_emb[:, n_act : n_act + ctx.shape[1], :]
        soft = tr.soft_prompt_hub(inputs["domain_id"]).view(b, tr.len_soft_prompts, tr.hidden_size)
        return {"ctx": ctx, "soft": soft, "pos_act": tr.pos_emb[:, :n_act, :]}

    def _block(self, block: Any, x: Any) -> Any:
        if not self._residual32 or self._dt["transformer"] is self._torch.float32:
            return block(x)
        half = self._dt["transformer"]
        x = x + block.attn(block.norm1(x).to(half)).float()
        return x + block.mlp(block.norm2(x).to(half)).float()

    def _transformer(
        self, domain_id: Any, action_with_noise: Any, proprio: Any, t: Any, ctx: dict[str, Any]
    ) -> Any:
        from lerobot.policies.xvla.soft_transformer import timestep_embedding

        torch = self._torch
        tr = self._model.transformer
        b, n_act = action_with_noise.shape[:2]
        time_tokens = timestep_embedding(t, tr.dim_time).unsqueeze(1).expand(b, n_act, tr.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(b, n_act, proprio.shape[-1])
        tokens = self._cast(torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1))
        x = tr.action_encoder(tokens, domain_id)
        x = torch.cat([x + ctx["pos_act"], ctx["ctx"], ctx["soft"]], dim=1)
        for block in self._blocks:
            x = self._block(block, x)
        return tr.action_decoder(tr.norm(x[:, :n_act]), domain_id).float()


def replace_linears(root: Any) -> int:
    """Swap every `nn.Linear` below `root` for one that stores its weight transposed."""
    import torch

    class LinearT(torch.nn.Module):
        """y = x @ Wt + b as one product over all leading dimensions."""

        def __init__(self, linear: torch.nn.Linear) -> None:
            super().__init__()
            self.in_features, self.out_features = linear.in_features, linear.out_features
            weight = linear.weight.detach().t().contiguous()
            self.wt = torch.nn.Parameter(weight, requires_grad=False)
            bias = linear.bias
            self.bias = None if bias is None else torch.nn.Parameter(bias.detach().clone(), False)

        @property
        def weight(self) -> torch.Tensor:
            return self.wt.t()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(-1, self.in_features)
            if self.bias is None:
                y = torch.mm(flat, self.wt)
            else:
                y = torch.addmm(self.bias, flat, self.wt)
            return y.view(*x.shape[:-1], self.out_features)

    n = 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is torch.nn.Linear:
                setattr(parent, name, LinearT(child))
                n += 1
    return n


def _stack(observations: Sequence[Any]) -> Any:
    first = observations[0]
    if isinstance(first, dict):
        return {k: _stack([o[k] for o in observations]) for k in first}
    return np.stack([np.asarray(o) for o in observations])


def sample_observations(n: int, image_size: int = 360) -> list[tuple[Observation, str]]:
    """`n` observations of different scenes (four tasks, start states in turn), with their task.

    For the rows check and the pinning search: real first observations after a
    reset, no model needed to make them.
    """
    from fleetlab.contracts.rollout import Job
    from fleetlab.engine import manifests, software
    from fleetlab.sim.libero_sim import LiberoSimEnv

    out: list[tuple[Observation, str]] = []
    tasks = (0, 3, 6, 9)
    envs = {t: LiberoSimEnv("libero_spatial", t, image_size) for t in tasks}
    soft = software.current("none")
    for k in range(n):
        task = tasks[k % len(tasks)]
        ident = manifests.identity(
            world_seed=99,
            episode_number=k + 1,
            robot=1,
            site=1,
            day=0,
            episode_index=k,
            suite="libero_spatial",
            task_index=task,
            start_state=k // len(tasks),
            policy=nominal_spec(),
            batch_size=1,
            software=soft,
            image_size=image_size,
        )
        env = envs[task]
        obs = env.reset(Job(identity=ident).identity, [])
        out.append((obs, env.task_description))
    for env in envs.values():
        env.close()
    return out
