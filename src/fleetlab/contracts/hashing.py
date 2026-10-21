"""The one hash the test bed uses (docs/design/contracts.md, section 4, "Proof of replay").

The hash of anything is the first 8 bytes of SHA-256 over its contiguous
bytes, written as 16 hexadecimal digits. The engine and the comparison tool
both call these functions, so they agree: the simulator state is hashed as the
float64 array `mj_getState(mjSTATE_INTEGRATION)` gives, each camera image as
rendered (rotated, uint8), never a video file.

`hash_json` hashes a canonical JSON encoding (sorted keys, no spaces), for
records such as an effective fault list (`contracts.oracle`).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import numpy.typing as npt

HASH_BYTES = 8


def hash_digest(data: bytes | bytearray | memoryview) -> bytes:
    """The first 8 bytes of SHA-256 over `data`, raw (what `SimEnv.state_hash` returns)."""
    return hashlib.sha256(data).digest()[:HASH_BYTES]


def hash_bytes(data: bytes | bytearray | memoryview) -> str:
    """The first 8 bytes of SHA-256 over `data`, as 16 hexadecimal digits (as the chain stores)."""
    return hash_digest(data).hex()


def hash_array(array: npt.NDArray[Any]) -> str:
    """The hash of an array's contiguous bytes (C order). The dtype is the caller's choice.

    Shape and dtype are not part of the hash: hash the state as float64 and
    images as uint8, as section 4 fixes.
    """
    return hash_bytes(np.ascontiguousarray(array).tobytes())


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal values give equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_json(value: Any) -> str:
    """The hash of a value's canonical JSON encoding."""
    return hash_bytes(canonical_json(value).encode())
