"""Who holds the GPU memory right now?

The kernel's total (GTT, /sys/class/drm/card1/device/mem_info_gtt_used) says
how full the GPU's 31 GiB are but not who filled them. Each process's share is
readable without root from /proc/PID/fdinfo, one entry per open handle on the
graphics device. Rendering contexts and PyTorch-on-ROCm both show up here.

    python spikes/throughput/gpu_mem_by_process.py
"""

from __future__ import annotations

import os
import re

UNIT = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}


def gtt_bytes(pid: str) -> int:
    seen, total = set(), 0
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return 0
    for fd in fds:
        try:
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                text = f.read()
        except OSError:
            continue
        client = re.search(r"drm-client-id:\s*(\d+)", text)
        if not client or client.group(1) in seen:  # several handles can point at the same client
            continue
        seen.add(client.group(1))
        mem = re.search(r"drm-memory-gtt:\s*(\d+)\s*(\w+)", text)
        if mem:
            total += int(mem.group(1)) * UNIT.get(mem.group(2), 1)
    return total


def main():
    rows = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        b = gtt_bytes(pid)
        if b > 50 * 2**20:
            try:
                with open(f"/proc/{pid}/cmdline") as f:
                    cmd = f.read().replace("\0", " ").strip()[:120]
            except OSError:
                cmd = "?"
            rows.append((b, pid, cmd))
    for b, pid, cmd in sorted(rows, reverse=True):
        print(f"{b / 2**30:6.2f} GiB  pid {pid:>7}  {cmd}")
    with open("/sys/class/drm/card1/device/mem_info_gtt_used") as f:
        total = int(f.read())
    print(f"listed {sum(r[0] for r in rows) / 2**30:.1f} GiB, kernel total {total / 2**30:.1f} GiB of 31 GiB")


if __name__ == "__main__":
    main()
