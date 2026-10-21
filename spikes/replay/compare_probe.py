"""Compare the outputs two policy_probe.py runs stored (question 4, fresh process).

    python compare_probe.py A B
"""

from __future__ import annotations

import sys

import numpy as np

import common as C

a, b = (np.load(C.OUT / f"{n}.npz") for n in sys.argv[1:3])
for k in sorted(set(a.files) & set(b.files)):
    x, y = a[k], b[k]
    same = x.shape == y.shape and x.tobytes() == y.tobytes()
    rows = x.reshape(len(x), -1) if x.ndim > 2 else x.reshape(1, -1)
    rows_y = y.reshape(len(y), -1) if y.ndim > 2 else y.reshape(1, -1)
    n_same = int(sum(r.tobytes() == s.tobytes() for r, s in zip(rows, rows_y, strict=True)))
    print(
        f"{k}: {'bit-identical' if same else 'DIFFER'} across the two processes "
        f"({n_same}/{len(rows)} outputs identical, largest difference {np.abs(x - y).max():.3e})"
    )
