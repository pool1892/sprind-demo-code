"""Turn a video into one image with N evenly spaced frames.

    run.sh strip.py VIDEO.mp4 [N] [OUT.png] [FIRST_FRAME] [LAST_FRAME]
"""
import sys

import imageio
import numpy as np

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else path.rsplit(".", 1)[0] + "_strip.png"
frames = [f for f in imageio.get_reader(path).iter_data()]
a = int(sys.argv[4]) if len(sys.argv) > 4 else 0
b = int(sys.argv[5]) if len(sys.argv) > 5 else len(frames) - 1
idx = np.linspace(a, min(b, len(frames) - 1), n).round().astype(int)
rows = [np.concatenate([frames[i] for i in idx[r:r + 4]], axis=1) for r in range(0, n, 4)]
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
imageio.imwrite(out, np.concatenate(rows, axis=0))
print(out, "frames", len(frames), "shown", idx.tolist())
