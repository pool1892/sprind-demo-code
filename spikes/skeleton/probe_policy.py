"""Checks of the policy side before the fleet is built on it. One heavy job, about two minutes.

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python spikes/skeleton/probe_policy.py TAG

1. Eight real observations (tasks 0 and 5, start states 0 to 3) from one process.
2. Time of one X-VLA call for batches of 1, 2, 4 and 8 rows.
3. Replay rule 3 for X-VLA on the Radeon: is a row's chunk independent of what the
   other rows contain, at fixed batch size and row?
4. The same batch again in this process; and the output is saved, so a second run
   of this script (another TAG) shows whether a fresh process repeats it.
5. One closed-loop episode in a padded batch of eight, as a smoke test of absolute
   control, the noise override and the action conversion.
6. The scene-camera fault on a live model: the image changes, and undoing it gives
   back the identical image after the next reset.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import sk_common as C
from policy_xvla import Policy, bits_equal

tag = sys.argv[1] if len(sys.argv) > 1 else "a"
out_dir = C.OUT / "probe"
out_dir.mkdir(parents=True, exist_ok=True)
report: dict = {"tag": tag, "gtt_start_gib": C.gtt_gib(), "load": C.loadavg()}

import faults as F  # noqa: E402

obs, tasks = [], []
cam_check = {}
for task in (0, 5):
    env = C.make_env(task)
    for init in range(4):
        o = C.reset_pinned(env, init, 1000 + 100 * task + init)
        obs.append(o)
        tasks.append(env.task_description)
    if task == 0:
        # scene-camera fault on the live model
        ref = C.reset_pinned(env, 0, 1000)
        ref_state = C.mj_state(env)
        f = F.make({"kind": "scene_camera_pan", "degrees": 10.0})
        faulted = C.reset_pinned(env, 0, 1000, before_reset=f.apply_cell)
        faulted_state = C.mj_state(env)
        f.undo_cell(env)
        back = C.reset_pinned(env, 0, 1000)
        cam_check = {
            "scene_image_changed": not bits_equal(ref["pixels"]["image"], faulted["pixels"]["image"]),
            "wrist_image_unchanged": bits_equal(ref["pixels"]["image2"], faulted["pixels"]["image2"]),
            "share_of_scene_pixel_values_changed": float(
                (ref["pixels"]["image"] != faulted["pixels"]["image"]).mean()
            ),
            "identical_after_undo": bits_equal(ref["pixels"]["image"], back["pixels"]["image"])
            and bits_equal(ref["pixels"]["image2"], back["pixels"]["image2"]),
            "physics_state_unchanged_by_fault": bits_equal(ref_state, faulted_state),
            "physics_state_identical_after_undo": bits_equal(ref_state, C.mj_state(env)),
        }
        import imageio.v2 as imageio

        imageio.imwrite(out_dir / "scene_ref.png", ref["pixels"]["image"][::-1, ::-1])
        imageio.imwrite(out_dir / "scene_pan10.png", faulted["pixels"]["image"][::-1, ::-1])
        imageio.imwrite(out_dir / "wrist_ref.png", ref["pixels"]["image2"])
    C.make_current(env)
    env.close()
report["camera_fault"] = cam_check
print("camera fault:", cam_check, flush=True)

import torch  # noqa: E402

pol = Policy("cuda")
report["policy_load_s"] = pol.load_s
print(f"policy loaded in {pol.load_s:.1f} s, chunk {pol.chunk}, executes {pol.n_action_steps}, noise {pol.noise_shape}", flush=True)


def call(rows: list[int], seeds: list[int] | None = None):
    seeds = seeds or rows
    batch = pol.batch([obs[r] for r in rows], [tasks[r] for r in rows])
    noise = torch.stack([pol.noise(7000 + s, 0) for s in seeds])
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = pol.chunks(batch, noise)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


call(list(range(8)))  # warm-up
timing = {}
for b in (1, 2, 4, 8):
    ts = [call(list(range(b)))[1] for _ in range(3)]
    timing[b] = round(1000 * float(np.median(ts)), 1)
report["ms_per_call_by_batch"] = timing
print("ms per call by batch size:", timing, flush=True)

a, _ = call(list(range(8)))
a2, _ = call(list(range(8)))
report["same_batch_again_identical"] = bits_equal(a.numpy(), a2.numpy())
# rows 0 and 5 keep their content, row and noise; every other row gets other content
b_rows = [0, 0, 0, 0, 0, 5, 5, 5]
b, _ = call(b_rows, seeds=[0, 100, 101, 102, 103, 5, 104, 105])
report["row_independent_of_other_rows"] = bits_equal(a[0].numpy(), b[0].numpy()) and bits_equal(
    a[5].numpy(), b[5].numpy()
)
# the same content and noise in another row
c, _ = call([1, 0, 2, 3, 4, 5, 6, 7])
report["row_position_matters"] = not bits_equal(a[0].numpy(), c[1].numpy())
report["max_abs_diff_other_row"] = float((a[0] - c[1]).abs().max())
# the same content alone
d, _ = call([0])
report["batch_size_matters"] = not bits_equal(a[0].numpy(), d[0].numpy())
report["max_abs_diff_batch_1"] = float((a[0] - d[0]).abs().max())
np.savez(out_dir / f"chunk_{tag}.npz", a=a.numpy(), env_actions=pol.to_env_actions(a))
print({k: v for k, v in report.items() if "row" in k or "batch" in k or "identical" in k}, flush=True)

# one closed-loop episode: task 0, start state 0, in row 0 of a batch of eight
env = C.make_env(0)
o = C.reset_pinned(env, 0, 1000)
t0, steps, success, queue, chunk_index = time.perf_counter(), 0, False, [], 0
while steps < 280:
    if not queue:
        batch = pol.batch([o] * 8, [env.task_description] * 8)
        noise = torch.stack([pol.noise(42, chunk_index)] * 8)
        queue = list(pol.to_env_actions(pol.chunks(batch, noise))[0])
        chunk_index += 1
    o, _, terminated, _, info = env.step(queue.pop(0))
    steps += 1
    if terminated:
        success = bool(info["is_success"])
        break
report["episode"] = {"success": success, "steps": steps, "chunks": chunk_index, "s": time.perf_counter() - t0}
print("episode:", report["episode"], flush=True)
env.close()

report["torch_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
report["own_gpu_mib"] = C.gpu_mib_of()
report["pss_mib"] = C.pss_mib()
report["gtt_end_gib"] = C.gtt_gib()
(out_dir / f"probe_{tag}.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
