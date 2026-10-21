"""Does a worker's memory grow over many episodes?

One environment, many resets with a few steps in between, memory read every
ten resets. A hard reset (LeRobot's default) rebuilds the MuJoCo model and the
rendering context each time; a soft reset keeps them.

    tools/heavy-run --light spikes/throughput/run.sh reset_memory.py --resets 150
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resets", type=int, default=150)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--res", type=int, default=common.DEFAULT_RES)
    args = ap.parse_args()

    gtt0 = common.gtt_gib()
    env = common.make_env(res=args.res, hard_reset=not args.soft)
    a = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    rows, t_reset = [], []
    for i in range(args.resets + 1):
        s = time.perf_counter()
        env.reset(seed=i)
        t_reset.append(time.perf_counter() - s)
        for _ in range(args.steps):
            env.step(a)
        if i % 10 == 0:
            m = common.proc_mem_mib()
            rows.append({"resets": i, "pss_mib": m.get("Pss"), "anon_mib": m.get("Pss_Anon"),
                         "gtt_delta_gib": common.gtt_gib() - gtt0})
            print(f"after {i:4d} resets: Pss {m.get('Pss'):6.0f} MiB, GTT change since start "
                  f"{rows[-1]['gtt_delta_gib']:+.2f} GiB, last reset {t_reset[-1]:.2f} s", flush=True)
    env.close()
    first, last = rows[1], rows[-1]
    slope = (last["pss_mib"] - first["pss_mib"]) / max(last["resets"] - first["resets"], 1)
    print(f"growth between reset {first['resets']} and {last['resets']}: {slope:+.2f} MiB per reset; "
          f"mean reset {sum(t_reset[1:]) / len(t_reset[1:]):.2f} s; load {common.loadavg()}")
    print("saved", common.save(f"reset_memory_{'soft' if args.soft else 'hard'}.json",
                               {"args": vars(args), "rows": rows, "mib_per_reset": slope,
                                "reset_s_mean": sum(t_reset[1:]) / len(t_reset[1:]), "loadavg": common.loadavg()}))


if __name__ == "__main__":
    main()
