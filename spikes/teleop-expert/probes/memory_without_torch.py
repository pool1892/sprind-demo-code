import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
def rss():
    with open(f"/proc/{os.getpid()}/status") as f:
        d = dict(l.split(":", 1) for l in f if ":" in l)
    return d["VmRSS"].strip(), "threads " + d["Threads"].strip()
print("start", rss())
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
print("after libero.envs import", rss(), "torch loaded:", "torch" in sys.modules)
import glob
bddl = sorted(glob.glob(os.path.expanduser("~/.cache/robots-probes/libero-port/lib/python3.12/site-packages/libero/libero/bddl_files/libero_spatial/*.bddl")))[0]
off = sys.argv[1] == "offscreen"
if off:
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, control_freq=20, hard_reset=True, use_camera_obs=False)
else:
    env = ControlEnv(bddl_file_name=bddl, control_freq=20, hard_reset=True, use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
env.reset()
print("offscreen" if off else "no renderer", "after make+reset", rss())
for i in range(3):
    env.reset()
print("after 3 more resets", rss())
