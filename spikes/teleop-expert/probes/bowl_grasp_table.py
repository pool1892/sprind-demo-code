"""Tabulate bowl_grasp_study results: run.sh probes/bowl_grasp_table.py FILE.jsonl"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
print(len(rows), "trials; held", sum(r["held"] for r in rows))
def table(key, bins):
    print(f"-- by {key}")
    for a, b in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if a <= r[key] < b]
        if sel:
            print(f"   {a:8.4f} .. {b:8.4f}: held {sum(r['held'] for r in sel):3d}/{len(sel):3d}  lifted {sum(r['lifted'] for r in sel):3d}")
table("below_rim", [0, 0.006, 0.010, 0.014, 0.018, 0.022, 0.026, 0.030, 0.05])
table("overlap", [0, 0.008, 0.011, 0.014, 0.017])
table("width", [0, 0.004, 0.006, 0.008, 0.010, 0.02, 0.08])
table("gap", [0, 0.003, 0.005, 0.007, 0.009, 0.02, 0.08])
table("bulge", [0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.05])
table("slant", [0, 0.001, 0.002, 0.003, 0.004])
table("radius", [0, 0.03, 0.038, 0.042, 0.046, 0.05, 0.06])
for r in rows:
    r["abs_yaw_vs_radial"] = abs(r["yaw_vs_radial"])
    r["phi_mod"] = r["phi"] % 22.5
table("abs_yaw_vs_radial", [0, 4, 8, 12, 20, 45, 91])
table("phi_mod", [0, 3, 6, 9, 12, 15, 18, 21, 22.6])
