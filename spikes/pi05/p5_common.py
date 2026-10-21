"""Shared helpers of the pi0.5 spike.

Environment set-up, the pinned reset and the small /proc readers. Copied from
spikes/xvla-speed/xs_common.py (which took them from the skeleton, replay and
throughput spikes); those directories are not imported. Changed: the output
area, the LIBERO configuration and the MIOpen directories are this spike's own.

Runs in this spike's environment ~/.cache/robots-probes/pi05/venv, a copy of
the baseline spike's ~/.cache/robots-probes/baseline-venv-mj337 (MuJoCo 3.3.7,
robosuite 1.5.2 port, LeRobot 0.6.1, transformers 5.5.4, PyTorch 2.11.0 on ROCm
10.0.0), and writes only below ~/.cache/robots-probes/pi05.

MIOpen (the library PyTorch uses for convolutions on the Radeon) keeps two
files per user: a database of which convolution algorithm was fastest, and a
cache of compiled kernels. Both normally live in ~/.config/miopen and
~/.cache/miopen and are shared by every project on this machine. This spike
points both at its own directory, so that it neither changes the shared files
nor is changed by them; `MIOPEN_DB` selects which copy ("pinned" by default,
built by `profile_pi05.py --pin`).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PROBES = Path.home() / ".cache" / "robots-probes"
OUT = PROBES / "pi05"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(OUT / "libero-config"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
# One directory per named database. "pinned" is searched once and then read-only.
MIOPEN_DB = os.environ.get("MIOPEN_DB", "pinned")
_miopen = OUT / "miopen" / MIOPEN_DB
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(_miopen / "db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(_miopen / "cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the checkpoints are local
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The checkpoints, as local copies whose preprocessor names the rebuilt tokenizer (make_tokenizer.py).
CKPT = {
    "pi05": OUT / "ckpt" / "pi05_libero_finetuned_v044",  # lerobot/pi05_libero_finetuned_v044 @ 7152f13e
    "pi0": OUT / "ckpt" / "pi0_libero_finetuned_v044",  # lerobot/pi0_libero_finetuned_v044 @ 45dcc8fc
}
OBS_PKL = PROBES / "xvla-speed" / "obs" / "obs.pkl"  # 80 recorded libero_spatial observations (read only)

# torch before mujoco: with mujoco first, copying weights to the Radeon fails.
import torch  # noqa: E402, F401, I001
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

SUITE = "libero_spatial"
RES = 360  # LeRobot's default render size for LIBERO, as in the baseline runs
CAMERAS = ("image", "image2")  # LeRobot's names: the scene camera and the wrist camera
SCENE_CAMERA = "agentview"  # MuJoCo name of the camera behind "image"
CONTROL_DT = 0.05  # 20 control steps per second
SPEC_INTEGRATION = int(mujoco.mjtState.mjSTATE_INTEGRATION)

GTT_USED = "/sys/class/drm/card1/device/mem_info_gtt_used"


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------

_patched = False


def patch_absolute_world() -> None:
    """Make absolute pose commands mean world coordinates, as on robosuite 1.4.

    The robosuite 1.5.2 port reads absolute commands in the robot-base frame;
    X-VLA commands world poses. Same patch as
    spikes/baseline/measured_eval.py applies under BASELINE_ABS_WORLD=1.
    """
    global _patched
    if _patched:
        return
    import lerobot.envs.libero as ll

    orig = ll._set_delta

    def _set_delta(robot, use_delta: bool) -> None:
        orig(robot, use_delta)
        ctrl = ll._arm_controller(robot)
        if not use_delta and hasattr(ctrl, "input_ref_frame"):
            ctrl.input_ref_frame = "world"

    ll._set_delta = _set_delta
    _patched = True


def make_env(task_id: int, suite_name: str = SUITE, res: int = RES, control_mode: str = "absolute"):
    """One LeRobot LIBERO environment with soft resets.

    X-VLA commands target poses ("absolute"); SmolVLA commands small movements ("relative").
    """
    from lerobot.envs.libero import LiberoEnv, _get_suite

    if control_mode == "absolute":
        patch_absolute_world()
    return LiberoEnv(
        task_suite=_get_suite(suite_name),
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        observation_height=res,
        observation_width=res,
        episode_index=0,
        n_envs=1,
        control_mode=control_mode,
        hard_reset=False,
    )


def sim_of(env):
    return env._env.env.sim


def rs_env_of(env):
    return env._env.env


def make_current(env) -> None:
    """Make this environment's OpenGL context current (replay rule 6)."""
    if getattr(env, "_env", None) is not None:
        ctx = getattr(sim_of(env), "_render_context_offscreen", None)
        if ctx is not None:
            ctx.gl_ctx.make_current()


def reset_pinned(env, init: int, scene_seed: int, before_reset=None) -> dict:
    """Reset so that the scene depends only on (task, init, scene seed, cell faults).

    Replay rule 1: numpy's global generator is seeded right before the reset,
    because LIBERO draws the fixture positions from it and the seed passed to
    reset() does not reach it on robosuite 1.5. `before_reset(env)` runs after
    the simulator exists and before the reset renders anything; that is where
    a fault that changes the cell is applied.
    """
    env._ensure_env()
    make_current(env)
    rs = rs_env_of(env)
    if rs.hard_reset and hasattr(rs, "object_property_initializers"):
        rs.object_property_initializers.clear()
    if before_reset is not None:
        before_reset(env)
    np.random.seed(scene_seed)
    env.init_state_id = init
    obs, _ = env.reset(seed=scene_seed)
    return obs


def mj_state(env) -> np.ndarray:
    sim = sim_of(env)
    m, d = sim.model._model, sim.data._data
    out = np.empty(mujoco.mj_stateSize(m, SPEC_INTEGRATION), np.float64)
    mujoco.mj_getState(m, d, out, SPEC_INTEGRATION)
    return out


def h64(a) -> np.uint64:
    """First eight bytes of the SHA-256 of an array's bytes."""
    digest = hashlib.sha256(np.ascontiguousarray(a).tobytes()).digest()[:8]
    return np.frombuffer(digest, np.uint64)[0]


def fixture_hash(env) -> np.uint64:
    """Hash of every body's pose in the model: the scene as the reset placed it."""
    m = sim_of(env).model._model
    return h64(np.concatenate([m.body_pos.ravel(), m.body_quat.ravel()]))


# ----------------------------------------------------------------------------
# /proc readers
# ----------------------------------------------------------------------------


def read_int(path: str) -> int:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except OSError:
        return -1


def gtt_gib() -> float:
    return read_int(GTT_USED) / 2**30


def loadavg() -> str:
    with open("/proc/loadavg") as f:
        return " ".join(f.read().split()[:3])


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return -1.0


def pss_mib(pid: int | str = "self") -> float:
    """Proportional memory of one process in MiB; Pss of several processes adds up."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def gpu_mib_of(pid: int | str = "self") -> float:
    """GPU memory the kernel accounts to one process (drm fdinfo), in MiB."""
    seen, total = set(), 0
    try:
        for f in os.listdir(f"/proc/{pid}/fdinfo"):
            try:
                text = Path(f"/proc/{pid}/fdinfo/{f}").read_text()
            except OSError:
                continue
            if "drm-client-id" not in text:
                continue
            fields = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip()
            cid = fields.get("drm-client-id")
            if cid in seen:
                continue
            seen.add(cid)
            for k, v in fields.items():
                if k.startswith("drm-memory-") and v.endswith(("KiB", "MiB", "GiB")):
                    num, unit = v.split()
                    total += int(num) * {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}[unit]
    except OSError:
        pass
    return total / 2**20
