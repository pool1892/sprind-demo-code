"""Check two reports about ROCm on this GPU family.

Report 1: every process that loads the ROCm runtime keeps one CPU core spinning.
Report 2: two or more such processes keep the GPU pinned (busy).

Method: start K child processes that initialise the GPU through PyTorch, run
one small computation, and then do nothing at all for --idle seconds. During
that idle time the parent, which never touches the GPU itself, reads each
child's consumed CPU time from /proc and the kernel's GPU busy percentage.
K = 0 is the baseline with no child. A plain Python child that never imports
torch is the control for report 1.

    tools/heavy-run spikes/throughput/run.sh rocm_idle.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import common


def child(kind: str, ready, stop):
    if kind == "rocm":
        import torch

        x = torch.randn(1024, 1024, device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
    elif kind == "torch_cpu":
        import torch

        x = torch.randn(1024, 1024)
        (x @ x).sum().item()
    ready.set()
    stop.wait()


def thread_cpu(pid: int) -> dict[str, float]:
    """CPU seconds per thread name, to see which thread is the busy one."""
    tick = os.sysconf("SC_CLK_TCK")
    out: dict[str, float] = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                raw = f.read()
            name = raw[raw.index("(") + 1: raw.rindex(")")]
            fields = raw.rsplit(")", 1)[1].split()
            out[name] = out.get(name, 0.0) + (int(fields[11]) + int(fields[12])) / tick
        except OSError:
            pass
    return out


def measure(kind: str, k: int, idle: float, env: dict | None = None) -> dict:
    ctx = mp.get_context("spawn")
    old = {}
    for key, val in (env or {}).items():
        old[key] = os.environ.get(key)
        os.environ[key] = val
    procs, stop = [], ctx.Event()
    for _ in range(k):
        ready = ctx.Event()
        p = ctx.Process(target=child, args=(kind, ready, stop))
        p.start()
        ready.wait(120)
        procs.append(p)
    for key, val in old.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    time.sleep(2)  # let start-up work settle
    cpu0 = [common.cpu_seconds(p.pid) for p in procs]
    thr0 = [thread_cpu(p.pid) for p in procs]
    busy, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < idle:
        busy.append(common.gpu_busy())
        time.sleep(0.25)
    wall = time.perf_counter() - t0
    cores = [(common.cpu_seconds(p.pid) - c) / wall for p, c in zip(procs, cpu0)]
    hot = []
    for p, before in zip(procs, thr0):
        after = thread_cpu(p.pid)
        d = {n: (after[n] - before.get(n, 0.0)) / wall for n in after}
        hot.append({n: round(v, 2) for n, v in sorted(d.items(), key=lambda kv: -kv[1])[:3] if v > 0.01})
    row = {"kind": kind, "processes": k, "env": env or {}, "idle_s": wall,
           "cpu_cores_per_process": cores, "busiest_threads": hot,
           "gpu_busy_pct_mean": sum(busy) / len(busy), "gpu_busy_pct_max": max(busy),
           "gtt_gib": common.gtt_gib(), "loadavg": common.loadavg()}
    stop.set()
    for p in procs:
        p.join(30)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=12.0)
    args = ap.parse_args()
    plan = [("none", 0, None), ("python", 1, None), ("torch_cpu", 1, None),
            ("rocm", 1, None), ("rocm", 2, None), ("rocm", 3, None), ("none", 0, None)]
    rows = []
    for kind, k, env in plan:
        r = measure(kind, k, args.idle, env)
        rows.append(r)
        print(f"{kind:10s} x{k}: idle CPU per process {[round(c, 2) for c in r['cpu_cores_per_process']]} cores, "
              f"busiest threads {r['busiest_threads']}, GPU busy mean {r['gpu_busy_pct_mean']:.0f}% "
              f"max {r['gpu_busy_pct_max']}%, GTT {r['gtt_gib']:.2f} GiB, load {r['loadavg'].split()[0]}",
              flush=True)
        time.sleep(2)
    print("saved", common.save("rocm_idle.json", {"rows": rows}))


if __name__ == "__main__":
    main()
