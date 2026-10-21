"""Does env.seed() reach numpy's global generator on this stack?  run.sh probes/seed_reaches.py"""
import numpy as np

from common import make_env

env, task, inits = make_env("libero_spatial", 4, renderer=False)
inner = env.env
print("type(inner).seed is a method of the class:", callable(getattr(type(inner), "seed", None)))
print("inner.seed on the instance:", repr(inner.__dict__.get("seed", "<not set on the instance>")))
np.random.seed(123)
a = np.random.get_state()[1][:3].copy()
env.seed(0)
b = np.random.get_state()[1][:3].copy()
print("numpy's global generator changed by env.seed(0):", not np.array_equal(a, b))
print("env.rng exists:", getattr(inner, "rng", None) is not None)
