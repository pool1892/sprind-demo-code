"""Shared helpers for the throughput spike.

Everything here runs against the read-only probe venv
~/.cache/robots-probes/libero-port; see run.sh for the environment variables.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

OUT = Path.home() / ".cache/robots-probes/throughput"
GTT_USED = "/sys/class/drm/card1/device/mem_info_gtt_used"
VRAM_USED = "/sys/class/drm/card1/device/mem_info_vram_used"
GPU_BUSY = "/sys/class/drm/card1/device/gpu_busy_percent"

# What lerobot-eval uses for LIBERO by default (lerobot/envs/configs.py, LiberoEnv).
DEFAULT_RES = 360
CAMERAS = "agentview_image,robot0_eye_in_hand_image"


def read_int(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip())


def gpu_mem_gib() -> float:
    """GPU memory in use, system-wide: GTT (shared system RAM) plus the small fixed VRAM."""
    return (read_int(GTT_USED) + read_int(VRAM_USED)) / 2**30


def gtt_gib() -> float:
    return read_int(GTT_USED) / 2**30


def gpu_busy() -> int:
    try:
        return read_int(GPU_BUSY)
    except OSError:
        return -1


def loadavg() -> str:
    with open("/proc/loadavg") as f:
        return f.read().strip()


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return -1.0


def proc_mem_mib(pid: int | str = "self") -> dict[str, float]:
    """Resident and proportional memory of one process, in MiB.

    Pss (proportional set size) splits shared pages between the processes that
    map them, so summing Pss over workers gives the real total.
    """
    out = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                key = line.split(":")[0]
                if key in ("Rss", "Pss", "Pss_Anon", "Pss_Shmem"):
                    out[key] = int(line.split()[1]) / 1024
    except OSError:
        pass
    return out


def cpu_seconds(pid: int | str = "self") -> float:
    """User plus system CPU time consumed so far by one process, all threads."""
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    tick = os.sysconf("SC_CLK_TCK")
    return (int(fields[11]) + int(fields[12])) / tick


def make_env(task_id: int = 0, res: int = DEFAULT_RES, suite_name: str = "libero_spatial",
             cameras: str = CAMERAS, hard_reset: bool = True, episode_index: int = 0):
    """One LIBERO environment through LeRobot's own wrapper, as lerobot-eval builds it."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(suite_name)
    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name=cameras,
        obs_type="pixels_agent_pos",
        observation_width=res,
        observation_height=res,
        episode_index=episode_index,
        n_envs=1,
        hard_reset=hard_reset,
    )


def save(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, default=float))
    return path
