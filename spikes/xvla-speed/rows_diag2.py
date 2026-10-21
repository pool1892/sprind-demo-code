"""Which layers of the real 16-bit model give a row another result in another position?

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python spikes/xvla-speed/rows_diag2.py [--half fp16]

rows_diag.py showed that the building blocks are position-free on random numbers
and the real 16-bit model is not. This script tests every layer of the real model
on the numbers it really sees: while one call runs, each leaf layer (and each
attention call) is run a second time on its own input with the rows rotated, and
the result, rotated back, is compared with the first. Nothing propagates, so a
layer listed here misbehaves by itself.

Layers whose first dimension is not "rows" (the vision tower works on 2 images per
row, its window attention on 25 windows per image) are rotated by the matching
number of entries.

Writes ~/.cache/robots-probes/xvla-speed/results/rows_diag2_<half>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import xs_common as C
from policy_fast import Policy, load_obs

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SHIFT = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--half", default="fp16", help="fp16, bf16 or fp32")
    ap.add_argument("--forms", action="store_true", help="also test other ways of writing each linear layer")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    b = args.rows
    opts = {"benchmark": False, "steps": 2}
    if args.half != "fp32":
        opts["half"] = {"transformer": args.half, "vision": args.half, "encoder": args.half}
    pol = Policy("cuda", opts=opts)
    m = pol.model
    obs, tasks = load_obs(b)
    stats: dict = defaultdict(lambda: {"calls": 0, "calls_position_free": 0, "largest_abs_difference": 0.0,
                                       "largest_abs_value": 0.0, "shapes": set()})
    state = {"busy": False, "part": ""}

    def compare(key: str, y: torch.Tensor, y2: torch.Tensor, s: int, x: torch.Tensor):
        y2 = torch.roll(y2, -s, 0)
        same = torch.equal(y, y2)
        st = stats[key]
        st["calls"] += 1
        st["calls_position_free"] += int(same)
        st["shapes"].add(str(tuple(x.shape)))
        st["largest_abs_value"] = max(st["largest_abs_value"], float(y.float().abs().max()))
        if not same:
            st["largest_abs_difference"] = max(st["largest_abs_difference"], float((y.float() - y2.float()).abs().max()))

    def make_hook(name: str, mod: nn.Module):
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Conv2d):
            label = f"convolution {'depthwise' if mod.groups > 1 else 'patch'} {mod.in_channels}->{mod.out_channels}"
        elif isinstance(mod, nn.Linear):
            label = f"linear {leaf} {mod.in_features}->{mod.out_features}"
        else:
            label = type(mod).__name__

        def hook(_mod, inp, out):
            if state["busy"] or not isinstance(out, torch.Tensor):
                return
            x = inp[0]
            if x.shape[0] % b:
                return
            s = SHIFT * (x.shape[0] // b)
            state["busy"] = True
            try:
                extra = [torch.roll(a, s, 0).contiguous() if isinstance(a, torch.Tensor) and a.shape[:1] == x.shape[:1] else a
                         for a in inp[1:]]
                y2 = _mod(torch.roll(x, s, 0).contiguous(), *extra)
            finally:
                state["busy"] = False
            compare(f"{state['part']}: {label}", out, y2, s, x)
            if isinstance(_mod, nn.Linear) and x.dim() == 3 and args.forms:
                # the same product written in other ways: is any of them position-free for this shape?
                w, bias = _mod.weight, _mod.bias
                wt = w.t().contiguous()
                xr = torch.roll(x, s, 0).contiguous()
                per = x.shape[0] // b
                forms = {
                    "weight stored transposed": lambda z: (torch.addmm(bias, z.reshape(-1, z.shape[-1]), wt) if bias is not None
                                                           else torch.mm(z.reshape(-1, z.shape[-1]), wt)).view(*z.shape[:-1], -1),
                    "one product per entry (bmm)": lambda z: (torch.baddbmm(bias, z, wt.unsqueeze(0).expand(z.shape[0], -1, -1))
                                                              if bias is not None else torch.bmm(z, wt.unsqueeze(0).expand(z.shape[0], -1, -1))),
                    "one call per row": lambda z: torch.cat([F.linear(c, w, bias) for c in z.split(per)]),
                }
                for fname, fn in forms.items():
                    compare(f"{state['part']}: {label} | {fname}", fn(x), fn(xr), s, x)

        return hook

    for part, root in (("vision", m.vlm.vision_tower), ("vision", m.vlm.multi_modal_projector),
                       ("encoder", m.vlm.language_model), ("transformer", m.transformer)):
        for name, mod in root.named_modules():
            if list(mod.children()) or isinstance(mod, (nn.Dropout, nn.Identity, nn.Embedding)):
                continue
            mod.register_forward_hook(make_hook(name, mod))

    orig_sdpa = F.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        y = orig_sdpa(q, k, v, *a, **kw)
        if not state["busy"] and q.shape[0] % b == 0 and not a and not {k_ for k_, v_ in kw.items() if isinstance(v_, torch.Tensor)}:
            s = SHIFT * (q.shape[0] // b)
            y2 = orig_sdpa(*(torch.roll(t, s, 0) for t in (q, k, v)), **kw)
            compare(f"{state['part']}: attention, inputs as the model passes them", y, y2, s, q)
            y3 = orig_sdpa(*(torch.roll(t, s, 0).contiguous() for t in (q, k, v)), **kw)
            y0 = orig_sdpa(*(t.contiguous() for t in (q, k, v)), **kw)
            compare(f"{state['part']}: attention, inputs copied to plain memory first", y0, y3, s, q)
        return y

    F.scaled_dot_product_attention = sdpa
    batch = pol.batch(obs, tasks)
    noise = torch.stack([pol.noise(9000 + c, 0) for c in range(b)])
    with torch.inference_mode():
        inputs = pol.model_inputs(batch)
        state["part"] = "vision"
        feats = pol.vision(inputs)
        state["part"] = "encoder"
        enc = pol.encode(inputs, feats)
        state["part"] = "transformer"
        pol.sample(inputs, enc, noise)
    F.scaled_dot_product_attention = orig_sdpa

    out = {}
    width = 104 if args.forms else 72
    print(f"{'layer':{width}s} {'calls':>5s} {'position-free':>13s} {'largest diff':>12s} {'largest value':>13s}  input shapes")
    for key, st in sorted(stats.items(), key=lambda kv: (kv[1]["calls_position_free"] == kv[1]["calls"], kv[0])):
        st["shapes"] = sorted(st["shapes"])
        out[key] = st
        print(f"{key:{width}s} {st['calls']:5d} {st['calls_position_free']:13d} {st['largest_abs_difference']:12.1e} "
              f"{st['largest_abs_value']:13.1f}  {', '.join(st['shapes'][:3])}")
    (C.OUT / "results" / f"rows_diag2_{args.half}{args.tag}.json").write_text(json.dumps(
        {"rows": b, "half": args.half, "load": C.loadavg(), "layers": out}, indent=1))
    bad = [k for k, st in out.items() if st["calls_position_free"] != st["calls"] and " | " not in k]
    print(f"\n{len(bad)} of {sum(' | ' not in k for k in out)} kinds of layer depend on the row position: {bad}")
    if args.forms:
        for fname in ("weight stored transposed", "one product per entry (bmm)", "one call per row"):
            ks = [k for k in out if k.endswith("| " + fname)]
            free = [k for k in ks if out[k]["calls_position_free"] == out[k]["calls"]]
            print(f"written as '{fname}': {len(free)} of {len(ks)} kinds of linear layer are position-free")


if __name__ == "__main__":
    main()
