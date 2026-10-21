"""Step two same-seed robosuite two-arm envs side by side in one process and report
the first step and observation key that differ. Usage: python robosuite_obs_diff.py TASK CONFIG
"""
import sys, numpy as np, hashlib
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from robosuite_twoarm import make
task = sys.argv[1]; cfg = sys.argv[2]
envs = [make(task, cfg, seed=7, cams=("frontview",)) for _ in range(2)]
o = [e.reset() for e in envs]
rng = np.random.default_rng(3)
low, high = envs[0].action_spec
first = None; ndiff = 0; keys = {}
for t in range(200):
    a = rng.uniform(low, high)
    o = [e.step(a)[0] for e in envs]
    for k in o[0]:
        if not np.array_equal(o[0][k], o[1][k]):
            keys[k] = keys.get(k, 0) + 1
            if first is None:
                first = (t, k)
                if k.endswith("image"):
                    d = np.abs(o[0][k].astype(int) - o[1][k].astype(int))
                    print("image diff: max", d.max(), "pixels differing", (d.sum(-1) > 0).sum())
                else:
                    print(k, o[0][k], o[1][k])
print("first diff", first, "keys", keys)
