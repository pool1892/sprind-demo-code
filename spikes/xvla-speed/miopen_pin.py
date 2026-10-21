"""Build the convolution database this spike's runs are pinned to.

    MIOPEN_DB=pinned tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/xvla-speed/miopen_pin.py --rows 1 2 3 4 5 6 7 8 10 12 16 [--fp32-rows 1 2 4 6 8 12 16]

Why: X-VLA's vision tower has eight kinds of convolution, and MIOpen has several
algorithms for each. Which one runs decides the last bits of the result (measured:
variants `miopen_no_winograd`, `benchmark_off_empty_db`). With
`torch.backends.cudnn.benchmark = True`, which is what lerobot-eval and the walking
skeleton set, every process times the algorithms again when it first meets a
convolution and takes the fastest at that moment. With `benchmark = False` PyTorch
asks MIOpen's database instead, takes the fastest one stored there and measures
nothing; with nothing stored it falls back to a slower default, and gives yet
other bits.

So this script runs the search once per batch size on a quiet machine, into a
database of its own, and makes the database files read-only. Every later run reads
it with `benchmark = False`. The search is keyed by the number of images in the
call (two per row) and by the number format, so every batch size that will be used
has to be listed here.

Only the vision tower runs: nothing else in the model is a convolution.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path

import xs_common as C
from policy_fast import Policy, load_obs

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=[], help="batch sizes for the 16-bit vision tower")
    ap.add_argument("--fp32-rows", type=int, nargs="*", default=[], help="batch sizes for the 32-bit vision tower")
    ap.add_argument("--half", default="fp16")
    args = ap.parse_args()
    db = Path(os.environ["MIOPEN_USER_DB_PATH"])
    db.mkdir(parents=True, exist_ok=True)
    for f in db.glob("*.txt"):
        f.chmod(f.stat().st_mode | stat.S_IWUSR)  # a database pinned earlier is extended, not replaced
    obs, tasks = load_obs()
    report = {"db": str(db), "load_start": C.loadavg(), "searched": []}
    for fmt, rows_list in ((args.half, args.rows), ("fp32", args.fp32_rows)):
        if not rows_list:
            continue
        opts = {"benchmark": True}
        if fmt != "fp32":
            opts["half"] = {"vision": fmt, "encoder": fmt, "transformer": fmt}
        pol = Policy("cuda", opts=opts)
        for b in rows_list:
            batch = pol.batch(obs[:b], tasks[:b])
            with torch.inference_mode():
                inputs = pol.model_inputs(batch)
                torch.cuda.synchronize()
                s = time.perf_counter()
                pol.vision(inputs)
                torch.cuda.synchronize()
                first = time.perf_counter() - s
                ts = []
                for _ in range(3):
                    s = time.perf_counter()
                    pol.vision(inputs)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - s)
            report["searched"].append({"format": fmt, "rows": b, "search_s": round(first, 1),
                                       "vision_ms_after": round(1000 * sorted(ts)[1], 1)})
            print(report["searched"][-1], C.loadavg(), flush=True)
        del pol
        torch.cuda.empty_cache()
    for f in db.glob("*.txt"):
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    report["load_end"] = C.loadavg()
    report["files"] = {f.name: f.stat().st_size for f in sorted(db.glob("*"))}
    (C.OUT / "results" / f"miopen_pin_{C.MIOPEN_DB}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["files"]), "read-only now; load", report["load_start"], "->", report["load_end"])


if __name__ == "__main__":
    main()
