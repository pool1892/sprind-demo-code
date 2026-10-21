"""How much memory does one environment process need? run.sh probes/memory.py"""
import resource, os
def rss():
    with open(f"/proc/{os.getpid()}/status") as f:
        d = dict(l.split(":", 1) for l in f if ":" in l)
    return d["VmRSS"].strip(), d.get("VmHWM", "").strip()
print("start", rss())
from common import make_env, reset_to_init
print("after imports", rss())
for suite, tid in (("libero_spatial", 0), ("libero_spatial", 4), ("libero_object", 0)):
    env, task, inits = make_env(suite, tid, use_camera_obs=False)
    print(suite, tid, "after make_env", rss())
    for i in range(4):
        reset_to_init(env, inits[i])
    print(suite, tid, "after 4 resets", rss())
    env.close()
    print(suite, tid, "after close", rss())
print("peak", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "GB")
