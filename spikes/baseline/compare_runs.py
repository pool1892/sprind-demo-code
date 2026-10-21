"""Paired comparison of two runs that used the same tasks, seeds and start states.

    python spikes/baseline/compare_runs.py EVAL_INFO_A EVAL_INFO_B

Counts the episodes that only A solved and that only B solved, and gives the
exact two-sided sign-test (McNemar) probability that such a split arises when
the two setups are in truth equally good.
"""

from __future__ import annotations

import json
import sys
from math import comb


def load(path):
    info = json.load(open(path))
    return {(t["task_group"], t["task_id"]): t["metrics"]["successes"] for t in info["per_task"]}


a, b = load(sys.argv[1]), load(sys.argv[2])
only_a = only_b = both = neither = 0
for key in sorted(set(a) & set(b)):
    for x, y in zip(a[key], b[key]):
        both += x and y
        neither += (not x) and (not y)
        only_a += x and not y
        only_b += y and not x
n = only_a + only_b
p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2**n) if n else 1.0
tot = both + neither + n
print(f"episodes compared: {tot}; A solved {both + only_a}, B solved {both + only_b}")
print(f"both {both}, neither {neither}, only A {only_a}, only B {only_b}; exact two-sided p = {p:.2g}")
