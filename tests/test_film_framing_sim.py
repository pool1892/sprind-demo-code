"""Framed re-renders for the film on the renderer and the simulator (`make check-sim`).

On a synthetic MuJoCo scene: the frames a `Renderer` draws put each body where its `View`
projects it (upright and not mirrored), for a placed camera and a named one, and a named
camera drawn as a placed pose with the same frame gives the same picture. On LIBERO: a batch
of two framed clips of one played episode is accepted after a single replay, its tracks
follow the drawn steps with the logged objects exactly where the log has them, a second run
skips what is drawn, and a record the replay does not reproduce keeps no video.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fleetlab.contracts.jobs import JobProgram, RequestState
from fleetlab.film import framing
from fleetlab.film.framing import CameraSpec, Pose, Renderer

pytestmark = pytest.mark.sim

SCENE_XML = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <camera name="side" pos="1.6 0.4 0.9" xyaxes="-0.24 0.97 0 -0.3 -0.07 0.95"/>
    <body name="red" pos="0 0 0.2"><geom type="sphere" size="0.05" rgba="1 0 0 1"/></body>
    <body name="green" pos="0.15 0.35 0.12"><geom type="sphere" size="0.05" rgba="0 1 0 1"/></body>
    <body name="blue" pos="-0.2 -0.3 0.45"><geom type="sphere" size="0.06" rgba="0 0 1 1"/></body>
  </worldbody>
</mujoco>
"""
BALLS = ("red", "green", "blue")


@pytest.fixture
def scene() -> Iterator[Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gl = mujoco.GLContext(640, 480)
    gl.make_current()
    context = SimpleNamespace(
        con=mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150),
        scn=mujoco.MjvScene(model, maxgeom=100),
        vopt=mujoco.MjvOption(),
        pert=mujoco.MjvPerturb(),
        update_offscreen_size=None,  # the frames here fit the model's buffer
    )
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context.con)
    try:
        yield SimpleNamespace(model=model, data=data, context=context, gl=gl)
    finally:
        gl.make_current()
        context.con.free()
        gl.free()


def renderer(scene: Any, camera: CameraSpec, width: int = 320, height: int = 240) -> Renderer:
    return Renderer(
        scene.model, scene.data, scene.context, camera, width, height, scene.gl.make_current
    )


def centroids(scene: Any, r: Renderer) -> dict[str, tuple[float, float]]:
    import mujoco

    bodies = r.segment()
    out = {}
    for name in BALLS:
        body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
        rows, cols = np.nonzero(bodies == body)
        out[name] = (cols.mean() + 0.5, rows.mean() + 0.5)  # pixel centres
    return out


@pytest.mark.parametrize(
    "camera",
    [
        CameraSpec("placed", pose=Pose(position=(1.2, -0.6, 1.1), look_at=(0, 0, 0.25), fovy=40)),
        CameraSpec("rolled", pose=Pose((1.2, 0.5, 0.9), (0, 0, 0.25), fovy=45, roll=20)),
        CameraSpec("side", name="side"),
    ],
    ids=lambda c: c.label,
)
def test_each_body_appears_where_the_view_projects_it(scene: Any, camera: CameraSpec) -> None:
    r = renderer(scene, camera)
    image, view = r.render()
    assert image.shape == (240, 320, 3) and image.any()
    seen = centroids(scene, r)
    for name in BALLS:
        uv, depth = view.project(scene.data.xpos[scene.model.body(name).id])
        assert depth[0] > 0
        np.testing.assert_allclose(uv[0], seen[name], atol=1.0, err_msg=name)


def test_a_named_camera_drawn_as_a_pose_with_its_frame_gives_the_same_picture(
    scene: Any,
) -> None:
    cam = scene.model.camera("side").id
    position = np.asarray(scene.data.cam_xpos[cam])
    axes = np.asarray(scene.data.cam_xmat[cam]).reshape(3, 3)  # columns: right, up, back
    pose = Pose(
        position=tuple(position),
        look_at=tuple(position - axes[:, 2]),
        fovy=float(scene.model.cam_fovy[cam]),
        up=tuple(axes[:, 1]),
    )
    named, named_view = renderer(scene, CameraSpec("side", name="side")).render()
    placed, placed_view = renderer(scene, CameraSpec("copy", pose=pose)).render()
    diff = np.abs(named.astype(int) - placed.astype(int))
    assert diff.mean() < 0.05 and (diff.max(axis=-1) > 8).mean() < 0.002
    np.testing.assert_allclose(named_view.matrix(), placed_view.matrix(), rtol=1e-6)
    np.testing.assert_allclose(
        named_view.world_to_camera(), placed_view.world_to_camera(), atol=1e-6
    )


def test_an_unknown_camera_is_refused(scene: Any) -> None:
    with pytest.raises(framing.ClipsError, match="no camera"):
        renderer(scene, CameraSpec("nope", name="nope"))


# ---- a framed replay of a played episode -----------------------------------------------------

STATE = RequestState.model_validate(
    {
        "request_text": "test",
        "requested_goals": [
            {"predicate": "On", "objects": ["plate_1", "main_table_stove_front_region"]},
            {"predicate": "On", "objects": ["akita_black_bowl_1", "wooden_cabinet_1_top_side"]},
        ],
        "base_task": 3,
        "order_rules": [],
        "budget": {
            "starts_per_request": 20,
            "requests_per_session": 4,
            "sessions": 6,
            "starts_per_investigation": 400,
        },
        "acceptance": None,
        "message_log": [],
    }
)
LIFT = JobProgram.model_validate(
    {
        "name": "lift-and-wait",
        "purpose": "a short episode to frame",
        "notes": "",
        "steps": [
            {
                "kind": "motion",
                "instruction": None,
                "until": None,
                "max_steps": None,
                "segment": {
                    "primitive": "ramp",
                    "axis": "z",
                    "amplitude_m": 0.04,
                    "duration_steps": 10,
                    "period_steps": None,
                },
                "steps": None,
            },
            {
                "kind": "wait",
                "instruction": None,
                "until": None,
                "max_steps": None,
                "segment": None,
                "steps": 4,
            },
        ],
    }
)
STEPS = 14


@pytest.fixture(scope="module")
def played(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from fleetlab.engine import software
    from fleetlab.jobs.runner import HoldSource, JobRunner
    from fleetlab.jobs.starts import development_starts

    root = tmp_path_factory.mktemp("framing") / "run"
    runner = JobRunner(HoldSource(), software.current("none"), image_size=64)
    try:
        runner.run_starts(
            root,
            "j0001",
            LIFT,
            STATE,
            development_starts(3, 1),
            record_video=False,
            first_episode=101,
        )
    finally:
        runner.close()
    return root


def clips(run: Path) -> framing.ClipsManifest:
    data = {
        "format": framing.CLIPS_FORMAT,
        "clips": [
            {"id": "placed", "run": str(run), "episode": "e000101", "camera": "hero-plate",
             "width": 320, "height": 180, "frame_every": 2, "bodies": ["flat_stove_1"]},
            {"id": "scene", "run": str(run), "episode": "e000101", "camera": "scene",
             "width": 96, "height": 96, "from_step": 4, "to_step": 9},
            {"id": "wrist", "run": str(run), "episode": "e000101", "camera": "wrist",
             "width": 64, "height": 64, "frame_every": 5},
        ],
    }  # fmt: skip
    return framing.parse_clips(data, run)


def test_a_batch_of_framed_clips_is_verified_tracked_and_skipped_when_drawn(
    played: Path, tmp_path: Path
) -> None:
    import imageio.v2 as imageio

    manifest = clips(played)
    index = framing.frame_clips(manifest, tmp_path, say=lambda _: None)
    assert index["accepted"] == 3 and index["statuses"] == {"rendered": 3}
    assert index["hash_mismatches"] == 0
    for clip in manifest.clips:
        report = json.loads((tmp_path / f"{clip.id}.rerender.json").read_text())
        assert report["accepted"] and report["verdict_matches"]
        assert report["hashes_compared"] == report["recorded_hashes"] == STEPS + 1
        assert report["largest_position_difference"]["objects_m"] == 0.0
        track = json.loads((tmp_path / report["track"]).read_text())
        steps = [row["step"] for row in track["rows"]]
        assert steps == clip.steps(STEPS) and track["frames"] == len(steps) == report["frames"]
        assert len(imageio.mimread(tmp_path / report["video"], memtest=False)) == len(steps)
        row = track["rows"][0]
        assert set(row["targets"]) >= {
            "akita_black_bowl_1", "plate_1", "gripper", "main_table_stove_front_region",
            "wooden_cabinet_1_top_side",
        }  # fmt: skip
        assert len(row["targets"]["main_table_stove_front_region"]["outline"]) == 4
        assert len(row["goals"]) == 2 and row["instruction"] is None
        assert (tmp_path / report["debug_still"]).is_file()
    placed = json.loads((tmp_path / "placed.track.json").read_text())
    assert placed["rows"][0]["targets"]["flat_stove_1"]["visible_px"] > 0
    assert placed["camera"]["view"]["fovy_deg"] == pytest.approx(37.0)
    wrist = json.loads((tmp_path / "wrist.track.json").read_text())
    assert wrist["camera"]["moves"] and all("camera" in row for row in wrist["rows"])
    hand = [row["camera"]["position"] for row in wrist["rows"]]
    assert hand[0] != hand[-1]  # the hand rose, and its camera with it
    written = (tmp_path / "index.json").read_bytes()
    again = framing.frame_clips(manifest, tmp_path, say=lambda _: None)
    assert again["accepted"] == 3 and again == index  # nothing drawn: the index stands
    assert (tmp_path / "index.json").read_bytes() == written
    redraw = framing.frame_clips(manifest, tmp_path, only={"scene"}, force=True, say=print)
    assert redraw["statuses"] == {"skipped": 2, "rendered": 1} and redraw["accepted"] == 3


def test_a_record_the_replay_does_not_reproduce_keeps_no_video(
    played: Path, tmp_path: Path
) -> None:
    from fleetlab.jobs import tables

    run = tmp_path / "tampered"
    shutil.copytree(played, run)
    executed, hashes, identity = tables.load_replay_record(run, "e000101")
    executed[5, 2] += 0.02
    tables.save_replay_record(run, "e000101", executed, hashes, identity)
    out = tmp_path / "out"
    index = framing.frame_clips(clips(run), out, say=lambda _: None)
    assert index["accepted"] == 0 and index["statuses"] == {"rejected": 3}
    report = json.loads((out / "placed.rejected-rerender.json").read_text())
    assert report["first_mismatch_step"] == 6 and "after step 6" in report["reason"]
    assert report["video"] is None and report["track"] is None
    assert not list(out.glob("*.mp4")) and not list(out.glob("*.track.json"))
    # Review finding 3 (461f987): a rejected stills run left earlier stills in place.
    stills = tmp_path / "stills"
    stills.mkdir()
    earlier = stills / "top.e000101.s0003.png"
    earlier.write_bytes(b"a still from an earlier, accepted run")
    top = framing.resolve_camera("top")
    rejected = framing.render_stills(run, "e000101", [top], [3], stills, width=64, height=64)
    assert not rejected["accepted"] and rejected["removed_earlier_stills"] == [earlier.name]
    assert not earlier.exists() and rejected["stills"] == []
