"""Side finding: on this ROCm setup the import order of mujoco and torch decides whether
weights can be copied to the GPU.

    python import_order.py torch-first | mujoco-first
"""

import glob
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
if sys.argv[1] == "torch-first":
    import torch
    import mujoco  # noqa: F401
else:
    import mujoco  # noqa: F401
    import torch
from safetensors.torch import load_file

f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lerobot--smolvla_libero/snapshots/*/model.safetensors"))[0]
try:
    sd = load_file(f, device="cuda")
    print(f"{sys.argv[1]}: weights copied to the GPU ({len(sd)} tensors)")
except Exception as e:  # noqa: BLE001
    print(f"{sys.argv[1]}: FAILED: {type(e).__name__}: {str(e).splitlines()[0]}")
