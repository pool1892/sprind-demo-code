"""Top view of the heights around the target object, as the planner sees them.

    run.sh probes/heightmap.py SUITE TASK INIT   -> PNG under ~/.cache/robots-probes/teleop/debug/
Colour is height (dark = low, bright = high) relative to the object's rim; the object is tinted red.
The robot is to the left (-x), +y is up.
"""
import sys

import imageio
import numpy as np

from common import OUT, make_env, reset_to_init
from expert import ScriptedExpert

suite, tid, iid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env, task, inits = make_env(suite, tid, use_camera_obs=False)
reset_to_init(env, inits[iid])
ex = ScriptedExpert(env)
sc = ex.sc
P = sc.points(ex.obj, 0.002)
Q = sc.obstacle_points(exclude=(ex.obj,), near=sc.body_pos(ex.obj), radius=0.4)
c = sc.body_pos(ex.obj)[:2]
half, cell = 0.16, 0.002
n = int(2 * half / cell)
top = P[:, 2].max()


def raster(pts):
    img = np.full((n, n), -np.inf)
    ij = np.floor((pts[:, :2] - (c - half)) / cell).astype(int)
    ok = np.all((ij >= 0) & (ij < n), axis=1)
    np.maximum.at(img, (ij[ok, 0], ij[ok, 1]), pts[ok, 2])
    return img


hp, hq = raster(P), raster(sc.obstacle_points(exclude=(ex.obj,), near=sc.body_pos(ex.obj), radius=0.4) if True else Q)
sc._spacing = (0.003, 0.002)
hq = raster(sc.obstacle_points(exclude=(ex.obj,), near=sc.body_pos(ex.obj), radius=0.4))
lo, hi = top - 0.08, top + 0.04
g = lambda h: np.clip((h - lo) / (hi - lo), 0, 1)  # noqa: E731
rgb = np.zeros((n, n, 3))
rgb[..., 1] = rgb[..., 2] = rgb[..., 0] = np.where(np.isfinite(hq), 0.15 + 0.85 * g(hq), 0.0)
objm = np.isfinite(hp) & (hp >= np.where(np.isfinite(hq), hq, -1))
rgb[objm] = np.stack([0.3 + 0.7 * g(hp[objm]), 0.2 * g(hp[objm]), 0.2 * g(hp[objm])], axis=1)
img = (255 * rgb.transpose(1, 0, 2)[::-1]).astype(np.uint8)   # x to the right, y up
img = np.kron(img, np.ones((4, 4, 1), dtype=np.uint8))
out = OUT / "debug" / f"heightmap_{suite}_{tid}_{iid}.png"
imageio.imwrite(out, img)
print(out, "object rim", round(top, 3), "centre", np.round(c, 3), "scale: image is", 2 * half, "m wide")
for name in sc.names:
    if name != ex.obj:
        q = sc.points(name, 0.006)
        if np.linalg.norm(q[:, :2] - c, axis=1).min() < 0.15:
            near = q[np.linalg.norm(q[:, :2] - c, axis=1) < 0.15]
            print(f"  near: {name} max height within 15 cm {near[:,2].max():.3f}")
