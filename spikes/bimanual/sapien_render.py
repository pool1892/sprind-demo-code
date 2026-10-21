"""Probe SAPIEN 3 offscreen rendering on this machine (Vulkan, no display, no CUDA).

Usage: python sapien_render.py --shader default|minimal|rt --out DIR [--spp 32] [--denoiser none|oidn]

Builds a small table-top scene (ground, table, two boxes, a sphere), renders RGB at
256x256 and 480x640, times frames, writes PNGs and a JSON summary. It also checks that
the same scene rendered twice gives identical pixels and that CPU PhysX stepped twice
from the same state gives identical poses.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import sapien
from PIL import Image


def build_scene():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    scene.add_ground(0)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0.3, 0.6, -1], [0.8, 0.8, 0.8], shadow=True)
    scene.add_point_light([0.5, -0.5, 1.5], [1.0, 1.0, 1.0])

    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[0.4, 0.6, 0.02])
    b.add_box_visual(half_size=[0.4, 0.6, 0.02], material=[0.6, 0.45, 0.3])
    table = b.build_kinematic(name="table")
    table.set_pose(sapien.Pose([0, 0, 0.4]))

    actors = []
    for i, (color, pos) in enumerate(
        [([0.9, 0.1, 0.1], [0.0, -0.2, 0.6]), ([0.1, 0.3, 0.9], [0.1, 0.2, 0.7])]
    ):
        b = scene.create_actor_builder()
        b.add_box_collision(half_size=[0.03, 0.03, 0.03])
        b.add_box_visual(half_size=[0.03, 0.03, 0.03], material=color)
        a = b.build(name=f"box{i}")
        a.set_pose(sapien.Pose(pos))
        actors.append(a)
    b = scene.create_actor_builder()
    b.add_sphere_collision(radius=0.04)
    b.add_sphere_visual(radius=0.04, material=[0.2, 0.8, 0.2])
    s = b.build(name="ball")
    s.set_pose(sapien.Pose([-0.15, 0.0, 0.8]))
    actors.append(s)
    return scene, actors


def add_camera(scene, name, w, h):
    cam = scene.add_camera(name=name, width=w, height=h, fovy=np.deg2rad(55), near=0.01, far=10)
    # Look from the front and above at the table centre.
    cam.entity.set_pose(sapien.Pose([-1.0, 0.0, 1.1], [0.9659258, 0, 0.258819, 0]))
    return cam


def grab(scene, cam):
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")  # float32 in [0, 1], H x W x 4
    return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader", default="default")
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--denoiser", default="none")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "sapien": sapien.__version__,
        "shader": args.shader,
        "DISPLAY": os.environ.get("DISPLAY"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loadavg_start": os.getloadavg(),
    }
    sapien.render.set_camera_shader_dir(args.shader)
    if args.shader == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(args.spp)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser(args.denoiser)
        result["spp"] = args.spp
        result["denoiser"] = args.denoiser
    result["device_summary"] = sapien.render.get_device_summary()

    for w, h in [(256, 256), (640, 480)]:
        scene, actors = build_scene()
        for _ in range(50):
            scene.step()
        cam = add_camera(scene, f"cam{w}x{h}", w, h)
        t0 = time.perf_counter()
        first = grab(scene, cam)
        t_first = time.perf_counter() - t0
        Image.fromarray(first).save(out / f"frame_{w}x{h}.png")
        again = grab(scene, cam)
        for _ in range(20):  # warm-up, not timed
            grab(scene, cam)
        # Render-only time: same scene state, picture after picture.
        t0 = time.perf_counter()
        for _ in range(args.frames):
            grab(scene, cam)
        t_render = (time.perf_counter() - t0) / args.frames
        # Physics step plus picture, as a simulator loop would do it.
        t0 = time.perf_counter()
        for _ in range(args.frames):
            scene.step()
            grab(scene, cam)
        t_step_render = (time.perf_counter() - t0) / args.frames
        result[f"{w}x{h}"] = {
            "first_frame_seconds": t_first,
            "ms_per_frame_render_only": t_render * 1000,
            "ms_per_frame_with_physics_step": t_step_render * 1000,
            "same_state_rendered_twice_identical": bool(np.array_equal(first, again)),
            "frame_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "mean_pixel": float(first.mean()),
        }

    # Physics determinism: two fresh scenes, same steps, compare poses bit for bit.
    poses = []
    for _ in range(2):
        scene, actors = build_scene()
        for k in range(300):
            if k == 100:
                actors[0].set_pose(sapien.Pose([0.0, -0.2, 0.9]))
            scene.step()
        poses.append(np.concatenate([np.concatenate([a.pose.p, a.pose.q]) for a in actors]))
    result["physx_cpu_two_runs_identical"] = bool(np.array_equal(poses[0], poses[1]))
    result["loadavg_end"] = os.getloadavg()
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
