"""Model-free rehearsal of the analysis pipeline's preflight on one world.

The same calls `fleetlab.integration.analyse.analyse` makes before its first
session, without the backend gate, the login check or any model call. Prints
counts and pass/fail; a failure's text is printed cut to 200 characters.
"""
import fleetlab  # noqa: F401  torch before mujoco

import json
import sys
import time
from pathlib import Path

from fleetlab.agents.checks import source_audit
from fleetlab.agents.toolenv import build_tool_env
from fleetlab.contracts.model import Fleet
from fleetlab.integration.analyse import (
    check_base_runs,
    check_observed_export,
    inputs,
    task_prompt,
)
from fleetlab.integration.chain import SYSTEM

world_root, export, out = (Path(p).expanduser().resolve() for p in sys.argv[1:4])
files, spec, fleet, runs = inputs(world_root)
n = check_observed_export(export, spec, runs, fleet)
print(f"observed export bound to the base runs: {n} episodes", flush=True)
out.mkdir(parents=True, exist_ok=False)
truth = json.loads((files / "world_truth.json").read_text())
t = time.time()
check_base_runs(
    runs, export, out, fleet, Fleet.model_validate(truth["fleet"]), truth["task_classes"]
)
print(f"base runs republish the cleared export byte for byte: yes ({time.time() - t:.0f} s)", flush=True)
failed = 0
for condition in ("logs", "experiments"):
    for ablated in (False, True):
        texts = {"system": SYSTEM, "task": task_prompt(spec, condition)}
        env = build_tool_env(role="analysis", ablate_fault_classes=ablated)
        t = time.time()
        report = source_audit(export, runs, env, fleet=fleet, texts=texts)
        problems = report.failures()
        print(
            f"source audit, {condition}, {'ablated' if ablated else 'full'} library: "
            f"{'passed' if report.passed else 'FAILED'}, {len(problems)} failures "
            f"({time.time() - t:.0f} s)",
            flush=True,
        )
        for line in problems[:12]:
            print("   ", line[:200], flush=True)
        failed += not report.passed
sys.exit(1 if failed else 0)
