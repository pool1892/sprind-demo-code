"""Do a checkpoint's weights match the model this LeRobot version builds? CPU only.

Builds the policy from the checkpoint's config without loading weights, then
compares the names and shapes of its parameters with those in model.safetensors.

    python check_checkpoint_keys.py lerobot/smolvla_libero [more repo ids or local dirs]
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

for repo in sys.argv[1:]:
    path = Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo))
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = "cpu"
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = False  # only the layout matters here
    policy = get_policy_class(cfg.type)(cfg)
    model = {k: tuple(v.shape) for k, v in policy.state_dict().items()}
    with safe_open(path / "model.safetensors", "pt") as f:
        ckpt = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    missing = sorted(set(model) - set(ckpt))
    unexpected = sorted(set(ckpt) - set(model))
    shape = sorted(k for k in set(model) & set(ckpt) if model[k] != ckpt[k])
    print(
        f"KEYS {repo}: model has {len(model)} tensors, checkpoint {len(ckpt)}; "
        f"missing in checkpoint {len(missing)}, not used by model {len(unexpected)}, shape mismatches {len(shape)}"
    )
    for label, keys in (("missing", missing), ("unexpected", unexpected), ("shape", shape)):
        for k in keys[:8]:
            print(f"   {label}: {k} model={model.get(k)} checkpoint={ckpt.get(k)}")
