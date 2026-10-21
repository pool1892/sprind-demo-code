"""Where does one X-VLA call spend its time? One heavy job, about three minutes.

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/xvla-speed/profile_call.py [--batches 1 2 4 8 12 16] [--tag a]

Uses the observations make_obs.py recorded. Three measurements:

1. Stage by stage, for each batch size: every stage of a call with the GPU's queue
   flushed before and after it (torch.cuda.synchronize), median of three calls
   after one warm-up. Also the whole call timed as the fleet runner times it, to
   show that the stage timers add nothing.
2. Layer by layer at one batch size (--detail-batch): a timer around every leaf
   layer and around every attention call. These timers flush the GPU several
   thousand times per call, so their sum is larger than the undisturbed call; read
   the shares, not the milliseconds.
3. Checks that come for free here: the staged call returns the same bits as
   today's call, and the one-pass action conversion returns the same bits as the
   30-pass one.

Writes ~/.cache/robots-probes/xvla-speed/results/profile_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

import xs_common as C
from policy_fast import Policy, bits_equal, load_obs

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def sync():
    torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = defaultdict(list)

    def run(self, name, fn, *a, **k):
        sync()
        s = time.perf_counter()
        out = fn(*a, **k)
        sync()
        self.t[name].append(time.perf_counter() - s)
        return out


def staged_call(pol: Policy, obs, tasks, seeds, tm: Timer):
    from lerobot.envs.utils import preprocess_observation

    from policy_fast import _stack

    stacked = tm.run("prep: stack observations", _stack, obs)
    o = tm.run("prep: images to float tensors (CPU)", preprocess_observation, stacked)
    o["task"] = list(tasks)
    o = tm.run("prep: LIBERO step (flip image, build state)", pol.env_pre, o)
    batch = tm.run("prep: tokenizer, copy to the GPU", pol.pre, o)
    noise = tm.run("prep: draw noise (CPU)", lambda: torch.stack([pol.noise(s, 0) for s in seeds]))
    with torch.inference_mode():
        inputs = tm.run("resize images to 224x224 (GPU)", pol.model_inputs, batch)
        feats = tm.run("vision tower and projector", pol.vision, inputs)
        enc = tm.run("language encoder (12 layers, 100 tokens)", pol.encode, inputs, feats)
        steps: list = []
        action = tm.run("sampling loop (10 steps x 24 layers, 312 tokens)", pol.sample, inputs, enc, noise, steps)
        tm.t["sampling loop: one step"].append(float(np.median(steps)))
        tm.t["sampling loop: first step"].append(steps[0])
        chunk = tm.run("copy the chunk to the CPU", pol.finish, action)
    acts = tm.run("chunk to environment actions (30 passes)", pol.to_env_actions, chunk)
    return chunk, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    ap.add_argument("--detail-batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--opts", default="{}", help="Policy options as JSON, e.g. '{\"half\": {\"transformer\": \"fp16\"}}'")
    ap.add_argument("--no-detail", action="store_true")
    args = ap.parse_args()

    obs, tasks = load_obs()
    report: dict = {"tag": args.tag, "gtt_start_gib": C.gtt_gib(), "load_start": C.loadavg(),
                    "miopen_db": C.MIOPEN_DB, "torch": torch.__version__}
    opts = json.loads(args.opts)
    pol = Policy("cuda", opts=opts)
    report["opts"] = opts
    report["policy_load_s"] = pol.load_s
    m = pol.model
    n_params = {
        "vision tower and projector": sum(p.numel() for p in m.vlm.vision_tower.parameters())
        + sum(p.numel() for p in m.vlm.multi_modal_projector.parameters()),
        "language encoder": sum(p.numel() for p in m.vlm.language_model.parameters()),
        "action transformer": sum(p.numel() for p in m.transformer.parameters()),
    }
    report["parameters_million"] = {k: round(v / 1e6, 1) for k, v in n_params.items()}
    report["florence_attn_implementation"] = {
        "vision": getattr(m.vlm.vision_tower.config, "_attn_implementation", None),
        "language": getattr(m.vlm.language_model.config, "_attn_implementation", None),
    }
    report["sdp_backends"] = {
        "flash": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    print(f"policy loaded in {pol.load_s:.1f} s; parameters (million): {report['parameters_million']}; "
          f"attention: {report['florence_attn_implementation']}, {report['sdp_backends']}", flush=True)

    # today's call where the options leave the model as it is, the staged call otherwise
    whole_call = pol.chunks if not (opts.get("half") or opts.get("amp") or opts.get("compile")) else pol.chunks_staged

    # ---- 1. stage by stage --------------------------------------------------------------------
    report["stages_ms"], report["whole_call_ms"], report["torch_peak_gib"] = {}, {}, {}
    shapes = {}
    for b in args.batches:
        rows = list(range(b))
        ob, tk = [obs[r] for r in rows], [tasks[r] for r in rows]
        staged_call(pol, ob, tk, rows, Timer())  # warm-up for this batch size
        tm = Timer()
        for _ in range(args.reps):
            chunk, acts = staged_call(pol, ob, tk, rows, tm)
        report["stages_ms"][b] = {k: round(1000 * float(np.median(v)), 2) for k, v in tm.t.items()}
        # the whole call as fleet.py times it: preparation, then chunks() + to_env_actions()
        whole = []
        for _ in range(args.reps):
            sync()
            s0 = time.perf_counter()
            batch = pol.batch(ob, tk)
            noise = torch.stack([pol.noise(s, 0) for s in rows])
            sync()
            s1 = time.perf_counter()
            ref = whole_call(batch, noise)
            sync()
            s2 = time.perf_counter()
            ref_acts = pol.to_env_actions(ref)
            sync()
            s3 = time.perf_counter()
            whole.append((s1 - s0, s2 - s1, s3 - s2))
        w = np.median(np.array(whole), axis=0)
        report["whole_call_ms"][b] = {"prepare": round(1000 * w[0], 1), "chunks": round(1000 * w[1], 1),
                                      "to_env_actions": round(1000 * w[2], 1)}
        report["torch_peak_gib"][b] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        same = bits_equal(chunk.numpy(), ref.numpy()) and bits_equal(acts, ref_acts)  # False by design with 16-bit options
        fast = pol.to_env_actions_fast(ref)
        shapes[b] = {"staged_equals_reference": same, "fast_post_equals_30_pass": bits_equal(fast, ref_acts)}
        s = report["stages_ms"][b]
        print(f"batch {b:2d}: chunks() {1000 * w[1]:7.1f} ms | vision {s['vision tower and projector']:6.1f}, "
              f"encoder {s['language encoder (12 layers, 100 tokens)']:6.1f}, "
              f"loop {s['sampling loop (10 steps x 24 layers, 312 tokens)']:7.1f}, "
              f"prepare {1000 * w[0]:5.1f}, to actions {1000 * w[2]:5.1f} | {shapes[b]} | load {C.loadavg()}", flush=True)
    report["checks"] = shapes

    out = C.OUT / "results"
    out.mkdir(parents=True, exist_ok=True)
    if args.no_detail:
        report["load_end"] = C.loadavg()
        (out / f"profile_{args.tag}.json").write_text(json.dumps(report, indent=1))
        print(f"wrote {out / f'profile_{args.tag}.json'}; load {report['load_start']} -> {report['load_end']}")
        return

    # ---- 2. layer by layer ----------------------------------------------------------------------
    b = args.detail_batch
    rows = list(range(b))
    ob, tk = [obs[r] for r in rows], [tasks[r] for r in rows]
    acc: dict = defaultdict(float)
    state = {"part": None, "t": 0.0}
    handles = []

    def hook(part: str, label: str):
        def pre(_mod, _inp):
            sync()
            state["t"] = time.perf_counter()

        def post(_mod, _inp, _out):
            sync()
            acc[(part, label)] += time.perf_counter() - state["t"]

        return pre, post

    def label_of(part: str, name: str, mod: nn.Module) -> str:
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Conv2d):
            return "convolution, depthwise 3x3" if mod.groups > 1 else "convolution, patch embedding"
        if isinstance(mod, nn.LayerNorm):
            return "layer norm"
        if isinstance(mod, nn.Embedding):
            return "embedding lookup"
        if isinstance(mod, nn.Linear):
            if leaf in ("qkv", "q_proj", "k_proj", "v_proj"):
                return "linear: attention inputs (q, k, v)"
            if leaf in ("proj", "out_proj"):
                return "linear: attention output"
            if leaf in ("fc1", "fc2"):
                return f"linear: MLP {leaf}"
            return f"linear: {leaf}"
        return type(mod).__name__

    roots = {"vision": m.vlm.vision_tower, "projector": m.vlm.multi_modal_projector,
             "encoder": m.vlm.language_model, "transformer": m.transformer}
    for part, root in roots.items():
        for name, mod in root.named_modules():
            if list(mod.children()) or isinstance(mod, (nn.Dropout, nn.Identity)):
                continue
            pre, post = hook(part, label_of(part, name, mod))
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))

    import torch.nn.functional as F

    orig_sdpa = F.scaled_dot_product_attention

    def timed_sdpa(*a, **k):
        sync()
        s = time.perf_counter()
        out = orig_sdpa(*a, **k)
        sync()
        acc[(state["part"], "attention itself (softmax of q k^T, times v)")] += time.perf_counter() - s
        return out

    F.scaled_dot_product_attention = timed_sdpa
    totals: dict = defaultdict(float)
    reps = 2
    with torch.inference_mode():
        batch = pol.batch(ob, tk)
        noise = torch.stack([pol.noise(s, 0) for s in rows])
        for _ in range(reps):
            inputs = pol.model_inputs(batch)
            for part, fn in (("vision", lambda: pol.vision(inputs)),):
                state["part"] = part
                sync()
                s = time.perf_counter()
                feats = fn()
                sync()
                totals[part] += time.perf_counter() - s
            state["part"] = "encoder"
            sync()
            s = time.perf_counter()
            enc = pol.encode(inputs, feats)
            sync()
            totals["encoder"] += time.perf_counter() - s
            state["part"] = "transformer"
            sync()
            s = time.perf_counter()
            pol.sample(inputs, enc, noise)
            sync()
            totals["transformer"] += time.perf_counter() - s
    F.scaled_dot_product_attention = orig_sdpa
    for h in handles:
        h.remove()
    detail: dict = {}
    for (part, label), v in acc.items():
        part = "vision" if part == "projector" else part
        detail.setdefault(part, {})
        detail[part][label] = detail[part].get(label, 0.0) + 1000 * v / reps
    for part in detail:
        tot = 1000 * totals[part] / reps
        detail[part]["everything else (reshapes, additions, timer overhead)"] = tot - sum(detail[part].values())
        detail[part] = {k: round(v, 1) for k, v in sorted(detail[part].items(), key=lambda kv: -kv[1])}
        detail[part]["TOTAL with timers"] = round(tot, 1)
    report["detail_batch"] = b
    report["detail_ms"] = detail
    for part, d in detail.items():
        print(f"\n{part}, batch {b}, ms per call with layer timers:")
        for k, v in d.items():
            print(f"  {v:8.1f}  {k}")

    # token counts, for the README
    with torch.inference_mode():
        inputs = pol.model_inputs(batch)
        feats = pol.vision(inputs)
        enc = pol.encode(inputs, feats)
    report["shapes"] = {
        "image_input": list(inputs["image_input"].shape), "image_mask_row0": inputs["image_mask"][0].tolist(),
        "image_features": list(feats.shape), "vlm_features": list(enc["vlm_features"].shape),
        "aux_visual_inputs": list(enc["aux_visual_inputs"].shape),
        "input_ids": list(inputs["input_ids"].shape),
        "real_text_tokens_row0": int((inputs["input_ids"][0] != 1).sum()),
        "transformer_sequence": pol.chunk + enc["vlm_features"].shape[1] + enc["aux_visual_inputs"].shape[1]
        + m.transformer.len_soft_prompts,
    }
    print("\nshapes:", report["shapes"])
    report["torch_peak_gib_all"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["own_gpu_mib"] = C.gpu_mib_of()
    report["pss_mib"] = C.pss_mib()
    report["gtt_end_gib"] = C.gtt_gib()
    report["load_end"] = C.loadavg()
    (out / f"profile_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out / f'profile_{args.tag}.json'}; load {report['load_start']} -> {report['load_end']}")


if __name__ == "__main__":
    main()
