"""Which operation makes a row's result depend on its position in the batch?

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python spikes/xvla-speed/rows_diag.py [--rows 8]

rows_check.py found that with the whole model in float16 a chunk changes when its
content moves to another row, and does not change when the other rows' contents
change. This script takes the model apart. For each operation it computes a batch,
then the same batch rotated by three rows, rotates the result back and counts the
rows whose bits are the same. An operation is "position-free" if all rows are.

Tested in 32-bit floats, float16 and bfloat16:

  each stage of the real model on recorded observations (vision tower, language
  encoder, one pass of the action transformer),
  and the building blocks on random tensors of the action transformer's shapes:
  linear layers written in several ways, attention, layer norm, GELU.

Writes ~/.cache/robots-probes/xvla-speed/results/rows_diag.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import xs_common as C
from policy_fast import Policy, load_obs

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SHIFT = 3


def rot(x: torch.Tensor, s: int = SHIFT) -> torch.Tensor:
    return torch.roll(x, shifts=s, dims=0)


def rows_same(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, float]:
    """(rows with identical bits, rows, largest absolute difference)."""
    a, b = a.float().cpu().numpy(), b.float().cpu().numpy()
    same = sum(a[i].tobytes() == b[i].tobytes() for i in range(a.shape[0]))
    return same, a.shape[0], float(np.abs(a.astype(np.float64) - b).max())


def check(fn, x: torch.Tensor) -> dict:
    """fn works on a batch (first dimension = rows). Compare fn(x) with fn(rotated x) rotated back."""
    y = fn(x)
    y2 = rot(fn(rot(x).contiguous()), -SHIFT)
    same, n, worst = rows_same(y, y2)
    return {"rows_identical": same, "rows": n, "largest_abs_difference": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()
    b = args.rows
    dts = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    report: dict = {"rows": b, "load_start": C.loadavg(), "miopen_db": C.MIOPEN_DB, "building_blocks": {}, "stages": {}}
    g = torch.Generator(device="cpu").manual_seed(1)
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    n, d = 312, 1024

    with torch.inference_mode():
        x32 = torch.randn(b, n, d, generator=g).to(dev)
        h32 = torch.randn(b, n, 4 * d, generator=g).to(dev)
        w1 = (torch.randn(4 * d, d, generator=g) / d**0.5).to(dev)
        w2 = (torch.randn(d, 4 * d, generator=g) / (4 * d) ** 0.5).to(dev)
        bias1 = torch.randn(4 * d, generator=g).to(dev)
        q32 = torch.randn(b, 16, n, 64, generator=g).to(dev)
        for name, dt in dts.items():
            x, h, wa, wb, ba, q = (t.to(dt) for t in (x32, h32, w1, w2, bias1, q32))
            ln = torch.nn.LayerNorm(d).to(dev, dt)
            pad = (-n) % 64

            def lin_padded(z, wa=wa, ba=ba, pad=pad):
                zp = F.pad(z, (0, 0, 0, pad))  # every row's block of tokens starts at a multiple of 64
                return F.linear(zp, wa, ba)[:, : z.shape[1]]

            tests = {
                "linear 1024->4096, as PyTorch does it (one product over all rows' tokens)": lambda z, wa=wa, ba=ba: F.linear(z, wa, ba),
                "linear 4096->1024, the same way": None,
                "linear 1024->4096, tokens per row padded to a multiple of 64": lin_padded,
                "linear 1024->4096 as one product per row (bmm)": lambda z, wa=wa, ba=ba: torch.baddbmm(
                    ba, z, wa.t().unsqueeze(0).expand(z.shape[0], -1, -1)),
                "linear 1024->4096, each row in a call of its own": lambda z, wa=wa, ba=ba: torch.cat(
                    [F.linear(z[i : i + 1], wa, ba) for i in range(z.shape[0])]),
                "layer norm": lambda z, ln=ln: ln(z),
                "GELU (tanh form)": lambda z: F.gelu(z, approximate="tanh"),
                "residual addition": lambda z: z + z * 0.5,
            }
            res = {}
            for tname, fn in tests.items():
                try:
                    if fn is None:
                        res[tname] = check(lambda z, wb=wb: F.linear(z, wb), h)
                    else:
                        res[tname] = check(fn, x)
                except Exception as e:  # noqa: BLE001
                    res[tname] = {"error": str(e).splitlines()[0][:160]}
            try:
                res["attention (PyTorch's fused call)"] = check(lambda z: F.scaled_dot_product_attention(z, z, z), q)
                res["attention written out (matmul, softmax, matmul)"] = check(
                    lambda z: ((z * 0.125) @ z.transpose(-2, -1)).softmax(dim=-1) @ z, q)
            except Exception as e:  # noqa: BLE001
                res["attention"] = {"error": str(e).splitlines()[0][:160]}
            report["building_blocks"][name] = res
            print(f"\n{name}:")
            for k, v in res.items():
                print(f"   {v.get('rows_identical', '-')}/{v.get('rows', '-')} rows identical, up to "
                      f"{v.get('largest_abs_difference', float('nan')):.1e}   {k}   {v.get('error', '')}", flush=True)

    # ---- the real model, stage by stage ---------------------------------------------------------
    obs, tasks = load_obs(b)
    for name in ("float32", "float16"):
        opts = {"benchmark": False, "hoist": True}
        if name != "float32":
            opts["half"] = {"transformer": "fp16", "vision": "fp16", "encoder": "fp16"}
        pol = Policy("cuda", opts=opts)
        m = pol.model
        batch = pol.batch(obs, tasks)
        noise = torch.stack([pol.noise(9000 + c, 0) for c in range(b)])
        res = {}
        with torch.inference_mode():
            inputs = pol.model_inputs(batch)
            inputs_r = {k: rot(v).contiguous() for k, v in inputs.items()}
            feats = pol.vision(inputs)
            feats_r = pol.vision(inputs_r)
            s, n_, w = rows_same(feats, rot(feats_r, -SHIFT))
            res["vision tower and projector"] = {"rows_identical": s, "rows": n_, "largest_abs_difference": w}
            # each later stage gets the unrotated stage's output, rotated, so that only this stage is tested
            enc = pol.encode(inputs, feats)
            enc_r = pol.encode(inputs_r, rot(feats).contiguous())
            s, n_, w = rows_same(enc["vlm_features"], rot(enc_r["vlm_features"], -SHIFT))
            res["language encoder"] = {"rows_identical": s, "rows": n_, "largest_abs_difference": w}
            enc_rot = {k: rot(v).contiguous() for k, v in enc.items()}
            a = pol.sample(inputs, enc, noise)
            a_r = pol.sample(inputs_r, enc_rot, rot(noise).contiguous())
            s, n_, w = rows_same(a, rot(a_r, -SHIFT))
            res["sampling loop (10 passes of the action transformer)"] = {"rows_identical": s, "rows": n_, "largest_abs_difference": w}
            # inside the vision tower: after each of its four stages
            px = inputs["image_input"][:, :2].flatten(0, 1).to(pol.dt["vision"])  # 2 real views per row
            vt = m.vlm.vision_tower

            def tower_upto(z, upto):
                for si, (conv, blocks) in enumerate(zip(vt.convs, vt.blocks)):
                    z = conv(z)
                    if si == upto and upto_conv_only[0]:
                        return z
                    for layer in blocks:
                        z = layer(z)
                    if si == upto:
                        return z
                return z

            upto_conv_only = [False]
            for si in range(4):
                for conv_only in (True, False):
                    upto_conv_only[0] = conv_only
                    y = tower_upto(px, si)
                    y2 = torch.roll(tower_upto(torch.roll(px, 2 * SHIFT, 0).contiguous(), si), -2 * SHIFT, 0)
                    s, n_, w = rows_same(y, y2)
                    res[f"vision stage {si + 1} " + ("patch convolution only" if conv_only else "complete")] = {
                        "rows_identical": s, "rows": n_, "largest_abs_difference": w}
        report["stages"][name] = res
        print(f"\nreal model, {name}:")
        for k, v in res.items():
            print(f"   {v['rows_identical']}/{v['rows']} rows identical, up to {v['largest_abs_difference']:.1e}   {k}", flush=True)
        del pol, m
        torch.cuda.empty_cache()
    report["load_end"] = C.loadavg()
    (C.OUT / "results" / "rows_diag.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote results/rows_diag.json; load {report['load_start']} -> {report['load_end']}")


if __name__ == "__main__":
    main()
