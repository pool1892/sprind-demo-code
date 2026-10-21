"""Which part of pi0.5 gives a row other bits when it sits in another row of the batch?

    tools/heavy-run ~/.cache/robots-probes/pi05/venv/bin/python spikes/pi05/rows_diag_pi05.py \
        --tag bf16 --opts '{"dtype": "bfloat16"}' [--rows 4]

The idea of spikes/xvla-speed/rows_diag2.py: one real call (4 recorded
observations, their noise, LeRobot's predict_action_chunk) runs with a hook on
every leaf module and on the attention functions. Each hook takes the inputs the
module just received, rotates the rows of the batch by one, calls the module
again on them, and compares the result row by row with the rotated output of the
real call. A module is position-free if every row comes out with the same bits.
Since every module gets the very same input twice, a difference can only come
from the module itself, not from upstream.

Attention is covered by wrapping transformers' eager attention functions (Gemma's
and SigLIP's) and its SDPA function; RMS norms, linear layers, activations, the
convolution and the embedding are modules. Writes results/rows_diag_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

import p5_common as C
from policy_pi05 import Policy, load_obs

import torch  # noqa: E402

STATS: dict = defaultdict(lambda: {"calls": 0, "differ": 0, "max_abs": 0.0, "shapes": set()})
BUSY = {"on": False}


def rot(x, b, perm):
    if isinstance(x, torch.Tensor) and x.dim() > 0 and x.shape[0] == b:
        return x[perm].clone()
    if isinstance(x, tuple):
        return tuple(rot(y, b, perm) for y in x)
    if isinstance(x, list):
        return [rot(y, b, perm) for y in x]
    return x


def first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            t = first_tensor(y)
            if t is not None:
                return t
    return None


def record(kind: str, out, out2, b, perm):
    a, c = first_tensor(out), first_tensor(out2)
    if a is None or c is None or a.dim() == 0 or a.shape[0] != b:
        return
    a = a[perm]
    s = STATS[kind]
    s["calls"] += 1
    s["shapes"].add(tuple(a.shape[1:]))
    same = a.shape == c.shape and torch.equal(a.contiguous().view(torch.uint8) if a.dtype != torch.bool else a,
                                              c.contiguous().view(torch.uint8) if c.dtype != torch.bool else c)
    if not same:
        s["differ"] += 1
        s["max_abs"] = max(s["max_abs"], float((a.float() - c.float()).abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--opts", default='{"dtype": "bfloat16"}')
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()
    b = args.rows
    perm = torch.tensor([(i + 1) % b for i in range(b)])
    obs, tasks = load_obs(b)
    pol = Policy("cuda", opts=json.loads(args.opts))
    perm = perm.to("cuda")

    def hook(name):
        kind = re.sub(r"\.\d+\.", ".N.", name)

        def h(mod, hargs, hkwargs, out):
            if BUSY["on"]:
                return
            BUSY["on"] = True
            try:
                with torch.inference_mode():
                    out2 = mod(*rot(hargs, b, perm), **{k: rot(v, b, perm) for k, v in hkwargs.items()})
                record(f"{type(mod).__name__} {kind}", out, out2, b, perm)
            finally:
                BUSY["on"] = False
        return h

    n_hooks = 0
    for name, mod in pol.policy.named_modules():
        if len(list(mod.children())) == 0:
            mod.register_forward_hook(hook(name), with_kwargs=True)
            n_hooks += 1

    import transformers.models.gemma.modeling_gemma as mg
    import transformers.models.siglip.modeling_siglip as ms
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def wrap(fn, kind):
        def w(module, query, key, value, attention_mask, *a, **k):
            out = fn(module, query, key, value, attention_mask, *a, **k)
            if not BUSY["on"]:
                BUSY["on"] = True
                try:
                    mask = attention_mask
                    out2 = fn(module, rot(query, b, perm), rot(key, b, perm), rot(value, b, perm),
                              rot(mask, b, perm), *a, **k)
                    record(kind + f" q{tuple(query.shape[1:])} k{tuple(key.shape[1:])}", out, out2, b, perm)
                finally:
                    BUSY["on"] = False
            return out
        return w

    mg.eager_attention_forward = wrap(mg.eager_attention_forward, "gemma eager attention")
    ms.eager_attention_forward = wrap(ms.eager_attention_forward, "siglip eager attention")
    import lerobot.policies.pi_gemma as pg

    if hasattr(pg, "eager_attention_forward"):
        pg.eager_attention_forward = wrap(pg.eager_attention_forward, "pi_gemma eager attention")
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrap(sdpa, "sdpa attention")
    impl = {"siglip": pol.policy.model.paligemma_with_expert.paligemma.config.vision_config._attn_implementation,
            "paligemma_text": pol.policy.model.paligemma_with_expert.paligemma.config.text_config._attn_implementation,
            "expert": pol.policy.model.paligemma_with_expert.gemma_expert.config._attn_implementation}

    batch = pol.batch(obs, tasks)
    noise = torch.stack([pol.noise(9000 + c, 0) for c in range(b)])
    out = pol.chunks_staged(batch, noise)
    rows = []
    for kind, s in sorted(STATS.items()):
        rows.append({"kind": kind, "calls": s["calls"], "differ": s["differ"], "max_abs": s["max_abs"],
                     "shapes": sorted(str(x) for x in s["shapes"])[:4]})
    bad = [r for r in rows if r["differ"]]
    for r in rows:
        flag = "DIFFERS" if r["differ"] else "same"
        print(f"{flag:8s} {r['differ']:4d}/{r['calls']:<4d} up to {r['max_abs']:.1e}  {r['kind']}  {r['shapes'][:2]}")
    report = {"tag": args.tag, "opts": json.loads(args.opts), "rows": b, "hooks": n_hooks, "attention_impl": impl,
              "kinds": rows, "position_dependent": [r["kind"] for r in bad], "finite": bool(np.isfinite(out.numpy()).all()),
              "load": C.loadavg()}
    (C.OUT / "results" / f"rows_diag_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"{len(bad)} of {len(rows)} kinds give a row other bits in another position; attention: {impl}")


if __name__ == "__main__":
    main()
