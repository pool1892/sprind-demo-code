"""Framed re-renders for the film (`fleetlab.film.framing`) without the simulator: the pinhole
projection on synthetic cameras, placed poses, the clips manifest, the frames a clip draws and
the check that lets a batch skip a clip already drawn."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fleetlab.film import framing
from fleetlab.film.framing import POSES, CameraSpec, Clip, ClipsError, Pose, View


def view(pose: Pose, width: int = 200, height: int = 100) -> View:
    return View.from_pose(pose, width, height, near=0.01, far=100.0)


# A camera at the origin looking along +x with +z up; with a vertical field of view of 90
# degrees on a 200 x 100 picture, fx = fy = 50 and the centre is (100, 50). Its right is -y.
ALONG_X = Pose(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), fovy=90.0)


def test_a_synthetic_camera_projects_points_where_a_pinhole_puts_them() -> None:
    v = view(ALONG_X)
    assert (v.fx, v.fy, v.cx, v.cy) == pytest.approx((50.0, 50.0, 100.0, 50.0))
    points = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, -0.5, 0.0), (2.0, 0.4, -0.2)]
    uv, depth = v.project(points)
    expected = [(100.0, 50.0), (100.0, 25.0), (125.0, 50.0), (90.0, 55.0)]
    np.testing.assert_allclose(uv, expected, atol=1e-9)
    np.testing.assert_allclose(depth, [1.0, 1.0, 1.0, 2.0])
    assert v.in_frame(uv[0, 0], uv[0, 1], depth[0])


def test_a_point_behind_the_camera_has_no_pixel_and_is_not_in_frame() -> None:
    v = view(ALONG_X)
    uv, depth = v.project([(-1.0, 0.0, 0.0), (1.0, 3.0, 0.0)])
    assert np.isnan(uv[0]).all() and depth[0] == pytest.approx(-1.0)
    assert not v.in_frame(uv[0, 0], uv[0, 1], depth[0])
    assert not v.in_frame(uv[1, 0], uv[1, 1], depth[1])  # in front, but off the picture's left edge


def test_the_recorded_intrinsics_and_extrinsics_reproduce_the_projection() -> None:
    pose = Pose(position=(0.6, 0.26, 1.53), look_at=(-0.03, -0.14, 1.09), fovy=35.0)
    v = view(pose, 1920, 1080)
    record = json.loads(json.dumps(v.as_json()))
    k = np.asarray(record["K"])
    world_to_camera = np.asarray(record["world_to_camera"])
    rng = np.random.default_rng(3)
    points = rng.uniform([-0.4, -0.4, 0.9], [0.2, 0.3, 1.3], size=(20, 3))
    uv, depth = v.project(points)
    camera = (world_to_camera @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    pixels = (k @ camera.T).T
    np.testing.assert_allclose(pixels[:, :2] / pixels[:, 2:], uv, atol=0.01)  # rounded record
    np.testing.assert_allclose(camera[:, 2], depth, atol=1e-5)
    assert record["fovy_deg"] == pytest.approx(35.0)


@pytest.mark.parametrize("name", sorted(POSES))
def test_every_proposed_pose_looks_at_its_target_with_orthonormal_axes(name: str) -> None:
    pose = POSES[name]
    forward, up, right = pose.axes()
    axes = np.stack([forward, up, right])
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(right, up), -forward, atol=1e-12)  # right-handed
    v = view(pose, 1280, 720)
    uv, depth = v.project([pose.look_at])
    np.testing.assert_allclose(uv[0], (640.0, 360.0), atol=1e-6)
    assert depth[0] > 0


def test_the_top_view_puts_the_robot_at_the_top_and_plus_y_to_the_right() -> None:
    v = view(POSES["top"], 1280, 720)
    centre = np.asarray(POSES["top"].look_at)
    uv, _ = v.project([centre + np.array([-0.1, 0, 0]), centre + np.array([0, 0.1, 0])])
    assert uv[0, 1] < 360 and uv[0, 0] == pytest.approx(640)  # towards the robot: up
    assert uv[1, 0] > 640 and uv[1, 1] == pytest.approx(360)  # +y: right


def test_roll_turns_the_picture_counterclockwise() -> None:
    above = (1.0, 0.0, 0.3)  # straight above the line of sight
    uv, _ = view(ALONG_X).project([above])
    assert uv[0, 0] == pytest.approx(100) and uv[0, 1] < 50
    rolled = Pose(position=ALONG_X.position, look_at=ALONG_X.look_at, fovy=90.0, roll=90.0)
    uv, _ = view(rolled).project([above])
    assert uv[0, 0] < 100 and uv[0, 1] == pytest.approx(50)  # up has turned to the left


def test_a_pose_without_a_usable_up_or_line_of_sight_is_refused() -> None:
    with pytest.raises(ValueError, match="parallel"):
        Pose(position=(0.0, 0.0, 2.0), look_at=(0.0, 0.0, 0.9), fovy=30.0)
    with pytest.raises(ValueError, match="no direction"):
        Pose(position=(0.0, 0.0, 1.0), look_at=(0.0, 0.0, 1.0), fovy=30.0)
    with pytest.raises(ValueError, match="field of view"):
        Pose(position=(0.0, 0.0, 1.0), look_at=(1.0, 0.0, 1.0), fovy=0.0)


def test_an_asymmetric_frustum_moves_the_principal_point() -> None:
    v = View.from_frustum(
        200, 100, (0, 0, 0), (1, 0, 0), (0, 0, 1), near=1.0, far=10.0, top=0.5, bottom=-0.5,
        center=0.25,
    )  # fmt: skip
    assert v.cx == pytest.approx(100 - 0.25 * v.fx)
    assert v.cy == pytest.approx(50)


# ---- cameras by name and the manifest -------------------------------------------------------


def test_cameras_resolve_to_the_manifest_s_poses_then_the_proposed_ones_then_names() -> None:
    own = Pose(position=(1.0, 1.0, 2.0), look_at=(0.0, 0.0, 1.0), fovy=30.0)
    assert framing.resolve_camera("hero-cabinet", {"hero-cabinet": own}).pose == own
    assert framing.resolve_camera("hero-plate").pose == POSES["hero-plate"]
    assert framing.resolve_camera("scene") == CameraSpec("scene", name="agentview")
    assert framing.resolve_camera("wrist").name == "robot0_eye_in_hand"
    assert framing.resolve_camera("frontview") == CameraSpec("frontview", name="frontview")
    placed = framing.resolve_camera(
        {"label": "low", "position": [1, 0, 1], "look_at": [0, 0, 1], "fovy": 40}
    )
    assert placed.label == "low" and placed.pose is not None and placed.pose.fovy == 40.0
    with pytest.raises(ClipsError, match="unknown field"):
        framing.resolve_camera({"position": [1, 0, 1], "look_at": [0, 0, 1], "fov": 40})


def manifest(clips: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"format": framing.CLIPS_FORMAT, "clips": clips, **extra}


def test_a_manifest_fills_defaults_and_reads_runs_from_its_folder(tmp_path: Path) -> None:
    data = manifest(
        [
            {"id": "a", "run": "runs/one", "episode": "e000101", "camera": "low"},
            {
                "id": "b",
                "run": "/abs/run",
                "episode": "e000102",
                "camera": "scene",
                "width": 480,
                "height": 480,
                "from_step": 10,
                "to_step": 20,
                "bodies": ["x"],
            },
        ],
        defaults={"width": 1280, "height": 720, "frame_every": 2},
        poses={"low": {"position": [1, 0, 1], "look_at": [0, 0, 1], "fovy": 40}},
    )
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(data))
    m = framing.load_clips(path)
    a, b = m.clips
    assert a.run_dir == (tmp_path / "runs" / "one").resolve() and a.camera.label == "low"
    assert (a.width, a.height, a.frame_every) == (1280, 720, 2)
    assert (b.width, b.height, b.from_step, b.to_step, b.bodies) == (480, 480, 10, 20, ("x",))
    assert b.run_dir == Path("/abs/run") and m.poses["low"].fovy == 40.0


@pytest.mark.parametrize(
    ("clip", "message"),
    [
        ({"id": "a b", "run": "r", "episode": "e000101", "camera": "scene"}, "letters"),
        ({"id": "a", "run": "r", "episode": "101", "camera": "scene"}, "episode id"),
        ({"id": "a", "run": "r", "episode": "e000101"}, "camera"),
        ({"id": "a", "run": "r", "episode": "e000101", "camera": "scene", "fps": 5}, "unknown"),
        (
            {"id": "a", "run": "r", "episode": "e000101", "camera": "scene", "frame_every": 3},
            "divides",
        ),
        ({"id": "a", "run": "r", "episode": "e000101", "camera": "scene", "width": 641}, "even"),
        (
            {
                "id": "a",
                "run": "r",
                "episode": "e000101",
                "camera": "scene",
                "from_step": 9,
                "to_step": 3,
            },
            "from_step",
        ),
    ],
)
def test_a_manifest_refuses_clips_that_cannot_be_drawn(
    tmp_path: Path, clip: dict[str, Any], message: str
) -> None:
    with pytest.raises(ClipsError, match=message):
        framing.parse_clips(manifest([clip]), tmp_path)


def test_a_manifest_refuses_a_repeated_id_and_another_format(tmp_path: Path) -> None:
    clip = {"id": "a", "run": "r", "episode": "e000101", "camera": "scene"}
    with pytest.raises(ClipsError, match="used twice"):
        framing.parse_clips(manifest([clip, clip]), tmp_path)
    with pytest.raises(ClipsError, match="format"):
        framing.parse_clips({**manifest([clip]), "format": "fleetlab.film-manifest/1"}, tmp_path)


def test_the_committed_manifests_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "site" / "explainer"
    for name in ("clips.example.json", "clips-batch-1.json"):
        m = framing.load_clips(root / name)
        assert m.clips and all(c.camera.label for c in m.clips)


# ---- the frames a clip draws, its spec, and skipping what is drawn ---------------------------


def clip(**fields: Any) -> Clip:
    base: dict[str, Any] = {
        "id": "c",
        "run_dir": Path("/runs/r"),
        "episode_id": "e000101",
        "camera": framing.resolve_camera("hero-cabinet"),
    }
    return Clip(**{**base, **fields})


def test_a_clip_draws_every_nth_step_of_its_range_and_its_last() -> None:
    assert clip().steps(5) == [0, 1, 2, 3, 4, 5]
    assert clip(frame_every=4).steps(10) == [0, 4, 8, 10]
    assert clip(frame_every=2, from_step=3, to_step=8).steps(100) == [3, 5, 7, 8]
    assert clip(from_step=8, to_step=50).steps(10) == [8, 9, 10]
    assert clip(from_step=20).steps(10) == []
    assert clip(frame_every=4).fps == 5


def test_a_clip_s_spec_survives_its_report_s_json() -> None:
    c = clip(bodies=("wooden_cabinet_1_cabinet_middle",), from_step=3)
    spec = c.spec()
    assert json.loads(json.dumps(spec)) == spec
    assert spec["camera"]["pose"]["fovy"] == POSES["hero-cabinet"].fovy
    assert clip(width=1280, height=720).spec() != spec


def test_a_batch_skips_only_a_clip_whose_outputs_are_intact_for_the_same_spec(
    tmp_path: Path,
) -> None:
    _, c, _, out = accepted_clip(tmp_path)
    names = framing._names(c.id)
    assert framing._accepted_output(c, out) is not None
    assert framing._accepted_output(replace(c, frame_every=2), out) is None  # another spec
    (out / names["video"]).write_bytes(b"other")
    assert framing._accepted_output(c, out) is None  # not the verified bytes
    (out / names["video"]).unlink()
    assert framing._accepted_output(c, out) is None


def test_a_batch_names_clips_it_does_not_know(tmp_path: Path) -> None:
    m = framing.parse_clips(
        manifest([{"id": "a", "run": "r", "episode": "e000101", "camera": "scene"}]), tmp_path
    )
    with pytest.raises(ClipsError, match="no clip"):
        framing.frame_clips(m, tmp_path / "out", only={"b"})


def test_the_fovy_of_a_pose_is_the_vertical_field_of_view() -> None:
    v = view(POSES["hero-plate"], 1920, 1080)
    assert 2 * math.degrees(math.atan(540 / v.fy)) == pytest.approx(POSES["hero-plate"].fovy)


def test_the_frame_command_needs_a_manifest_or_a_whole_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fleetlab.film.__main__ import main

    assert main(["frame", "--out", str(tmp_path)]) == 2
    assert main(["frame", "--run", str(tmp_path), "--episode", "e000101", "--out", "x"]) == 2
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps(manifest([])))
    assert main(["frame", "--clips", str(clips), "--run", str(tmp_path), "--out", "x"]) == 2
    assert "--clips, or --run, --episode and --camera" in capsys.readouterr().err
    with pytest.raises(ClipsError, match="a list of clips"):
        main(["frame", "--clips", str(clips), "--out", str(tmp_path / "out")])


def accepted_clip(tmp_path: Path) -> tuple[framing.ClipsManifest, Clip, Path, Path]:
    """A clip whose accepted outputs exist, drawn from sources that are all readable."""
    run, out = tmp_path / "run", tmp_path / "out"
    truth = run / "truth" / "jobs"
    truth.mkdir(parents=True)
    np.savez(truth / "e000101.npz", executed=np.zeros((2, 7)), state_hash=np.array(["a", "b"]))
    (truth / "e000101.json").write_text('{"episode_id": "e000101"}\n')
    (run / "episodes.jsonl").write_text('{"episode_id":"e000101","delivered":true}\n')
    (run / "run.json").write_text(json.dumps({"state": {"requested_goals": []}}))
    for table in ("object_steps", "goal_steps", "robot_steps"):
        (run / "log" / table).mkdir(parents=True)
        (run / "log" / table / "e000101.parquet").write_bytes(table.encode())
    m = framing.parse_clips(
        manifest([{"id": "c", "run": str(run), "episode": "e000101", "camera": "top"}]), tmp_path
    )
    (c,) = m.clips
    names = framing._names(c.id)
    out.mkdir()
    for kind in ("video", "track"):
        (out / names[kind]).write_bytes(kind.encode())
    report = {
        "accepted": True,
        "spec": c.spec(),
        "source_sha256": framing.source_digest(run, "e000101"),
        "video": names["video"],
        "video_sha256": hashlib.sha256(b"video").hexdigest(),
        "track": names["track"],
        "track_sha256": hashlib.sha256(b"track").hexdigest(),
    }
    (out / names["report"]).write_text(json.dumps(report))
    return m, c, run, out


def test_a_clip_is_skipped_only_while_its_sources_are_unchanged(tmp_path: Path) -> None:
    # Review finding 2 (461f987): the skip trusted paths, not contents.
    _, c, run, out = accepted_clip(tmp_path)
    assert framing._accepted_output(c, out) is not None
    table = run / "log" / "robot_steps" / "e000101.parquet"
    table.write_bytes(b"corrected")
    assert framing._accepted_output(c, out) is None


def test_an_unreadable_source_withdraws_an_earlier_acceptance(tmp_path: Path) -> None:
    # Review finding 1 (461f987): a rejection left the earlier acceptance standing,
    # and the next run skipped the clip as accepted.
    m, _, run, out = accepted_clip(tmp_path)
    say: list[str] = []
    assert framing.frame_clips(m, out, say=say.append)["statuses"] == {"skipped": 1}
    (run / "truth" / "jobs" / "e000101.npz").unlink()
    for attempt in ("first", "retry"):
        index = framing.frame_clips(m, out, force=attempt == "first", say=say.append)
        assert index["statuses"] == {"rejected": 1} and index["accepted"] == 0, attempt
        assert sorted(p.name for p in out.iterdir()) == ["c.rejected-rerender.json", "index.json"]
