"""The software an episode runs on (`contracts.identity.SoftwareVersion`).

Exactness holds on one machine with this software. The engine computes it once
per run and refuses a job whose identity names other software: a replay on
changed software is a new measurement, not a replay.

- `lock_hash`: the hash of `uv.lock` (every Python package, pinned).
- `libero_patch_set`: the hash of our LIBERO patches in `third_party/`.
- `mujoco_version`: read from the installed package's metadata.
- `miopen_db_hash`: the pinned convolution database (`engine.miopen`), or
  `"none"` for a run without convolutions on the GPU (the toy policy).
"""

from __future__ import annotations

from importlib import metadata

from fleetlab.contracts.hashing import hash_bytes
from fleetlab.contracts.identity import SoftwareVersion
from fleetlab.runtime import repo_root


def lock_hash() -> str:
    return hash_bytes((repo_root() / "uv.lock").read_bytes())


def libero_patch_set() -> str:
    patches = sorted((repo_root() / "third_party").glob("*.patch"))
    return hash_bytes(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in patches))


def current(miopen_db_hash: str = "none") -> SoftwareVersion:
    return SoftwareVersion(
        lock_hash=lock_hash(),
        libero_patch_set=libero_patch_set(),
        mujoco_version=metadata.version("mujoco"),
        miopen_db_hash=miopen_db_hash,
    )
