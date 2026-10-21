"""The two operations that make up the sampling loop, alone: matrix products and attention.

    MIOPEN_DB=micro tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/xvla-speed/micro_bench.py [--rows 8] [--tokens 312]

No policy is loaded; the tensors have the shapes the action transformer uses for
`rows` episodes (rows x 312 tokens of width 1024, 16 heads of width 64). Each
variant runs `--reps` times in a row with the GPU's queue flushed once at the end,
after a warm-up. "TFLOPS" is 2 x (multiply-adds) per second, in units of 10^12.

1. Linear layers in 32-bit floats, float16 and bfloat16, as PyTorch calls them
   (rocBLAS), and the same product written as a 1x1 convolution (MIOpen).
2. Attention for one layer, written in several ways, with the largest difference
   of each result from today's call.

Writes ~/.cache/robots-probes/xvla-speed/results/micro_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import time

import xs_common as C

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def timed(fn, reps: int, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=312)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--no-conv", action="store_true")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6.0 * 2**30 / total))
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    b, n = args.rows, args.tokens
    report: dict = {"rows": b, "tokens": n, "load_start": C.loadavg(), "gtt_start_gib": C.gtt_gib(),
                    "miopen_db": C.MIOPEN_DB, "device": torch.cuda.get_device_name(0)}
    print(report, flush=True)

    # ---- 1. linear layers ---------------------------------------------------------------------
    lin = {}
    with torch.inference_mode():
        for name, k_in, k_out in (("MLP fc1 1024->4096", 1024, 4096), ("MLP fc2 4096->1024", 4096, 1024),
                                  ("attention inputs 1024->3072", 1024, 3072), ("attention output 1024->1024", 1024, 1024)):
            x32 = torch.randn(b, n, k_in, generator=g).to(dev)
            w32 = (torch.randn(k_out, k_in, generator=g) / k_in**0.5).to(dev)
            bias32 = torch.zeros(k_out, device=dev)
            flops = 2.0 * b * n * k_in * k_out
            row = {}
            ref = F.linear(x32, w32, bias32)
            for dt_name, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
                x, w, bias = x32.to(dt), w32.to(dt), bias32.to(dt)
                try:
                    t = timed(lambda: F.linear(x, w, bias), args.reps)
                    err = float((F.linear(x, w, bias).float() - ref).abs().max())
                    row[dt_name] = {"ms": round(1000 * t, 3), "tflops": round(flops / t / 1e12, 2), "max_abs_diff": err}
                except Exception as e:  # noqa: BLE001
                    row[dt_name] = {"error": str(e).splitlines()[0][:200]}
            # the same product as a 1x1 convolution: (B, C, tokens, 1)
            if not args.no_conv:
                for dt_name, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
                    xc = x32.to(dt).transpose(1, 2).unsqueeze(-1).contiguous()
                    wc, bias = w32.to(dt)[:, :, None, None].contiguous(), bias32.to(dt)
                    for fmt_name, fmt in (("", torch.contiguous_format), (" channels-last", torch.channels_last)):
                        xcf, wcf = xc.contiguous(memory_format=fmt), wc.contiguous(memory_format=fmt)
                        key = f"1x1 convolution {dt_name}{fmt_name}"
                        try:
                            s = time.perf_counter()
                            out = F.conv2d(xcf, wcf, bias)
                            torch.cuda.synchronize()
                            first = time.perf_counter() - s
                            t = timed(lambda: F.conv2d(xcf, wcf, bias), args.reps)
                            err = float((out.float().squeeze(-1).transpose(1, 2) - ref).abs().max())
                            row[key] = {"ms": round(1000 * t, 3), "tflops": round(flops / t / 1e12, 2),
                                        "max_abs_diff": err, "first_call_s": round(first, 1)}
                        except Exception as e:  # noqa: BLE001
                            row[key] = {"error": str(e).splitlines()[0][:200]}
            lin[name] = row
            print(name, json.dumps(row), flush=True)
        # how fast can this GPU multiply at all: one large square product
        a32 = torch.randn(4096, 4096, generator=g).to(dev)
        sq = {}
        for dt_name, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
            a = a32.to(dt)
            try:
                t = timed(lambda: a @ a, 10)
                sq[dt_name] = {"ms": round(1000 * t, 2), "tflops": round(2.0 * 4096**3 / t / 1e12, 2)}
            except Exception as e:  # noqa: BLE001
                sq[dt_name] = {"error": str(e).splitlines()[0][:200]}
        lin["square 4096x4096 product"] = sq
        print("square 4096:", json.dumps(sq), flush=True)
    report["linear"] = lin

    # ---- 2. attention -------------------------------------------------------------------------
    heads, hd = 16, 64
    att = {}
    with torch.inference_mode():
        q = torch.randn(b, heads, n, hd, generator=g).to(dev)
        k = torch.randn(b, heads, n, hd, generator=g).to(dev)
        v = torch.randn(b, heads, n, hd, generator=g).to(dev)
        scale = hd**-0.5
        ref = F.scaled_dot_product_attention(q, k, v)

        def manual4d():
            return ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1) @ v

        def manual4d_scale_after():
            return ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1) @ v

        q3, k3, v3 = (z.reshape(b * heads, n, hd) for z in (q, k, v))
        kt3 = k3.transpose(1, 2).contiguous()

        def bmm3d():
            return torch.bmm(torch.bmm(q3 * scale, k3.transpose(1, 2)).softmax(dim=-1), v3)

        def bmm3d_contig():
            return torch.bmm(torch.bmm(q3 * scale, kt3).softmax(dim=-1), v3)

        def baddbmm3d():
            w = torch.baddbmm(torch.empty(1, device=dev), q3, k3.transpose(1, 2), beta=0, alpha=scale)
            return torch.bmm(w.softmax(dim=-1), v3)

        def einsum():
            w = torch.einsum("bhqd,bhkd->bhqk", q * scale, k).softmax(dim=-1)
            return torch.einsum("bhqk,bhkd->bhqd", w, v)

        def loop_heads():
            # one ordinary matrix product per (row, head) pair is far too many calls; one per head over rows
            outs = []
            for h in range(heads):
                w = torch.bmm(q[:, h] * scale, k[:, h].transpose(1, 2)).softmax(dim=-1)
                outs.append(torch.bmm(w, v[:, h]))
            return torch.stack(outs, dim=1)

        variants = {
            "today: F.scaled_dot_product_attention": lambda: F.scaled_dot_product_attention(q, k, v),
            "written out, 4D matmul (LeRobot's fallback)": manual4d,
            "written out, scale after the product": manual4d_scale_after,
            "written out, 3D bmm": bmm3d,
            "written out, 3D bmm, k transposed in memory": bmm3d_contig,
            "written out, baddbmm": baddbmm3d,
            "einsum": einsum,
            "one bmm per head": loop_heads,
        }
        for name, fn in variants.items():
            try:
                t = timed(fn, args.reps)
                out = fn().reshape(ref.shape)
                att[name] = {"ms": round(1000 * t, 3), "max_abs_diff": float((out - ref).abs().max()),
                             "bit_identical": bool(torch.equal(out, ref))}
            except Exception as e:  # noqa: BLE001
                att[name] = {"error": str(e).splitlines()[0][:200]}
            print(name, att[name], flush=True)
        # which of PyTorch's three fused back ends does this GPU run?
        for flags in ((True, False, False), (False, True, False), (False, False, True)):
            nm = "sdpa back end " + ("flash", "mem_efficient", "math")[flags.index(True)]
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=flags[0], enable_mem_efficient=flags[1], enable_math=flags[2]):
                    t = timed(lambda: F.scaled_dot_product_attention(q, k, v), args.reps)
                    out = F.scaled_dot_product_attention(q, k, v)
                att[nm] = {"ms": round(1000 * t, 3), "max_abs_diff": float((out - ref).abs().max()),
                           "bit_identical": bool(torch.equal(out, ref))}
            except Exception as e:  # noqa: BLE001
                att[nm] = {"error": str(e).splitlines()[0][:200]}
            print(nm, att[nm], flush=True)
        # the parts of the written-out form
        w_ = (q * scale) @ k.transpose(-2, -1)
        p_ = w_.softmax(dim=-1)
        parts = {
            "q k^T (128 products of 312x64 by 64x312)": timed(lambda: (q * scale) @ k.transpose(-2, -1), args.reps),
            "softmax over 8x16x312x312": timed(lambda: w_.softmax(dim=-1), args.reps),
            "weights times v": timed(lambda: p_ @ v, args.reps),
        }
        att["parts_ms"] = {kk: round(1000 * vv, 3) for kk, vv in parts.items()}
        print("parts:", att["parts_ms"], flush=True)
        # half precision
        for dt_name, dt in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
            qh, kh, vh = q.to(dt), k.to(dt), v.to(dt)
            try:
                t = timed(lambda: F.scaled_dot_product_attention(qh, kh, vh), args.reps)
                out = F.scaled_dot_product_attention(qh, kh, vh).float()
                att[f"sdpa in {dt_name}"] = {"ms": round(1000 * t, 3), "max_abs_diff": float((out - ref).abs().max())}
            except Exception as e:  # noqa: BLE001
                att[f"sdpa in {dt_name}"] = {"error": str(e).splitlines()[0][:200]}
            print(f"sdpa in {dt_name}", att[f"sdpa in {dt_name}"], flush=True)
        # GELU and layer norm, for scale
        h = torch.randn(b, n, 4096, generator=g).to(dev)
        x = torch.randn(b, n, 1024, generator=g).to(dev)
        ln = torch.nn.LayerNorm(1024).to(dev)
        att["other_ms"] = {
            "GELU (tanh form) on 8x312x4096": round(1000 * timed(lambda: F.gelu(h, approximate="tanh"), args.reps), 3),
            "layer norm on 8x312x1024": round(1000 * timed(lambda: ln(x), args.reps), 3),
            "residual addition on 8x312x1024": round(1000 * timed(lambda: x + x, args.reps), 3),
        }
        print("other:", att["other_ms"], flush=True)
    report["attention"] = att
    report["load_end"] = C.loadavg()
    report["torch_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out = C.OUT / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"micro_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out / f'micro_{args.tag}.json'}; load {report['load_start']} -> {report['load_end']}")


if __name__ == "__main__":
    main()
