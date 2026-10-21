"""Framed, verified re-renders of recorded job episodes, with overlay tracks, for the film.

A framed re-render re-executes a recorded job episode exactly as `fleetlab.jobs.replay`'s
`rerender` does (`replay.load_record` and `replay.replay`: a record that does not stand for
its episode is refused, and the MuJoCo state hash after the reset and after every step and
the final delivery verdict are compared with the record) and draws it at any width x height
from a named camera of the LIBERO scene or from a camera placed by hand (`Pose`). Only an
accepted replay keeps its video, overlay track and debug still; a rejected one keeps its
report, which says where the replay left the record. The replay always runs and is checked
over the whole episode; a step range only chooses the frames drawn.

Rendering leaves the physics alone. A placed camera is written into MuJoCo's scene after
the scene is updated (`mjvScene.camera`, and the headlight that MuJoCo puts at the camera),
never into the model, and the scene is drawn in the replay's own offscreen context, which is
enlarged once when a frame is bigger than its buffer. The scene's own observation cameras
are built at `OBSERVATION_SIZE`, since nothing here reads them. The hashes are the proof.

Frames are upright and not mirrored: OpenGL reads rows bottom-up, so the rows are flipped
and nothing else. LIBERO's convention, which the recorded videos and `rerender` follow,
turns the read-back image by 180 degrees; for the scene's upright cameras that shows the
kitchen mirrored left to right (the stove, at +y, lands on the left of the scene camera's
picture although the camera's right is +y). A framed `scene` clip is the true view, the
mirror image of the recorded scene video.

**The overlay track** (`<id>.track.json`, `TRACK_FORMAT`) has a row for every frame of the
video: its control step and time; for every target (each object of the episode's
`object_steps`, the gripper, any extra body named in the clip) where the camera sees it:
`u`, `v` (pixels) and `depth` (metres along the line of sight) of its point (an object's
body origin, which `object_steps.position` records exactly; the gripper's grip site, which
`robot_steps.eef_pos` samples one 2 ms physics substep before the end of the control step,
so it trails the position the frame shows by up to about 2 mm when the hand moves fast),
`cu`, `cv` for the centre of
the target's visible geometry, `in_frame` (in front of the camera and inside the picture),
and from a segmentation pass
`visible_px` (pixels showing the target, occlusion counted) and `bbox` ([u0, v0, u1, v1]
around them, or null); for each requested goal's region (a site, such as the stove front or
the cabinet top) and any site named in the clip, its centre and the `outline` of the face of
its box that looks most upward (four [u, v]); and the run's own recorded values for that step
from its log tables:
each object's `in_contact_with_robot` and `in_contact_with`, each requested goal's `holds`,
the gripper's opening and command, the program step, the instruction and the phase. The
projected positions are those of the replayed state, which the hashes tie to the record;
the report gives their largest coordinate difference from the logged positions, in metres
(`largest_position_difference`: 0 for the objects). The header carries the
camera (intrinsics and extrinsics), the requested goals, the program's steps with their
control steps, and the recorded delivery.

Pixel coordinates have their origin at the top-left corner of the picture, `u` to the right
and `v` down; the pixel in column j and row i has its centre at (j + 0.5, i + 0.5). The
camera is a pinhole in OpenCV's convention: `K` (fx, fy, cx, cy) and `world_to_camera`, a
4 x 4 matrix taking world points (metres) to camera coordinates with x right, y down and z
along the line of sight, so u = fx x / z + cx and v = fy y / z + cy, with depth z.

Each clip's outputs, in the batch's folder: `<id>.rerender.mp4` (H.264, 20 / frame_every
frames a second, repeatable encode), `<id>.rerender.json` (the report: the clip's spec, the
verification, the camera, the SHA-256 of the video and the track, and `source_sha256`, the
digest of the episode's sources, `source_digest`, without which a later run does not skip
the clip), `<id>.track.json` and
`<id>.debug.png` (four frames with every target's projected point, centre and box drawn);
a rejected clip leaves `<id>.rejected-rerender.json` alone, and its earlier outputs are
removed, also when its record cannot be read. `index.json` lists every clip of
the manifest with its status and hash counts; it is written once, when a run is complete (a
run that draws removes the old one first; a run that draws nothing leaves it as it is).

Commands, each one light simulation job (one at a time):

    tools/heavy-run --light -m 4G uv run --locked python -m fleetlab.film frame \\
        --clips site/explainer/clips-batch-1.json --out DIR [--only ID,ID] [--force]
    tools/heavy-run --light -m 4G uv run --locked python -m fleetlab.film frame \\
        --run RUN_DIR --episode e000202 --camera hero-cabinet --out DIR \\
        [--width 1920 --height 1080 --frame-every 1 --from-step 60 --to-step 160 --id ID]
    tools/heavy-run --light -m 4G uv run --locked python -m fleetlab.film stills \\
        --run RUN_DIR --episode e000202 --steps 0,82,140 --camera hero-cabinet \\
        --camera top [--poses CLIPS.json] [--width 1280 --height 720] --out DIR

The clips manifest is described at `load_clips`; `site/explainer/README-footage.md` walks
through it, and `site/explainer/clips.example.json` is a documented example.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.jobs import replay as jobs_replay
from fleetlab.jobs import tables

__all__ = [
    "CLIPS_FORMAT",
    "INDEX_FORMAT",
    "POSES",
    "REPORT_FORMAT",
    "TRACK_FORMAT",
    "CameraSpec",
    "Clip",
    "ClipsError",
    "ClipsManifest",
    "Pose",
    "Renderer",
    "View",
    "frame_clips",
    "load_clips",
    "load_poses",
    "parse_clips",
    "render_stills",
    "resolve_camera",
]

CLIPS_FORMAT = "fleetlab.film-clips/1"
REPORT_FORMAT = "fleetlab.film-frame/1"
TRACK_FORMAT = "fleetlab.film-track/1"
INDEX_FORMAT = "fleetlab.film-frame-index/1"
STILLS_FORMAT = "fleetlab.film-stills/1"
RENDERER_VERSION = 4  # part of every clip's spec: raise it when the same spec draws otherwise
CONTROL_HZ = jobs_replay.CONTROL_HZ
OBSERVATION_SIZE = 64  # the scene's own two cameras, drawn every step and not read here
CAMERA_ALIASES = {"scene": "agentview", "wrist": "robot0_eye_in_hand"}
GRIPPER = "gripper"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SEGMENT_WIDTH = 960  # the segmentation pass is drawn at most about this wide
CRF, PRESET = 18, "slow"  # as `rerender` encodes

Vec3 = tuple[float, float, float]
Array = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]


class ClipsError(ValueError):
    """A clips manifest or a clip that cannot be used as it stands."""


def _unit(vector: npt.ArrayLike, what: str) -> Array:
    v = np.asarray(vector, np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise ValueError(f"{what} has no direction")
    return v / norm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, indent: int | None = 1) -> None:
    """Written beside and renamed into place, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=indent) + "\n")
    os.replace(tmp, path)


# ---- cameras -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """A camera placed by hand, in world coordinates (metres and degrees).

    It stands at `position` and looks at `look_at`; `fovy` is its vertical field of view;
    `up` is the world direction that points up in the picture (+z suits every view except
    one looking straight down, which names another, such as -x to put the robot at the
    top); `roll` then turns the camera about its line of sight, positive values turning the
    picture counterclockwise.
    """

    position: Vec3
    look_at: Vec3
    fovy: float
    roll: float = 0.0
    up: Vec3 = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not 1.0 <= self.fovy <= 150.0:
            raise ValueError(f"fovy {self.fovy} is not a field of view in degrees")
        self.axes()  # refuses a pose without a line of sight or with up along it

    def axes(self) -> tuple[Array, Array, Array]:
        """The camera's forward, up and right directions in the world, orthonormal."""
        forward = _unit(np.subtract(self.look_at, self.position), "the line of sight")
        right = np.cross(forward, np.asarray(self.up, np.float64))
        if float(np.linalg.norm(right)) < 1e-6:
            raise ValueError("up is parallel to the line of sight; give another up direction")
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        if self.roll:
            a = math.radians(self.roll)
            up, right = (
                math.cos(a) * up + math.sin(a) * right,
                math.cos(a) * right - math.sin(a) * up,
            )
        return forward, up, right

    def as_json(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "look_at": list(self.look_at),
            "fovy": self.fovy,
            "roll": self.roll,
            "up": list(self.up),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any], where: str = "pose") -> Pose:
        unknown = sorted(set(data) - {"position", "look_at", "fovy", "roll", "up"})
        if unknown:
            raise ClipsError(f"{where}: unknown field(s) {', '.join(unknown)}")

        def vec(key: str, default: Vec3 | None = None) -> Vec3:
            value = data.get(key, default)
            if (
                not isinstance(value, list | tuple)
                or len(value) != 3
                or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
            ):
                raise ClipsError(f"{where}.{key}: three numbers expected")
            return (float(value[0]), float(value[1]), float(value[2]))

        def number(key: str, default: float | None = None) -> float:
            value = data.get(key, default)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ClipsError(f"{where}.{key}: a number expected")
            return float(value)

        try:
            return cls(
                position=vec("position"),
                look_at=vec("look_at"),
                fovy=number("fovy"),
                roll=number("roll", 0.0),
                up=vec("up", (0.0, 0.0, 1.0)),
            )
        except ValueError as e:
            if isinstance(e, ClipsError):
                raise
            raise ClipsError(f"{where}: {e}") from e


# The poses proposed for the kitchen of LIBERO's goal suite (every job here runs in it), at
# 16:9. The scene: the robot's base at x = -0.66; the table top at z = 0.90; the cabinet at
# x -0.11..0.15, y -0.36..-0.13, its top at z = 1.13 and its drawers facing +y; the stove at
# x -0.47..-0.17, y 0.11..0.30, with the stove-front region at (-0.05, 0.21); the bowl
# starts near (-0.09, 0.0) and the plate near (0.05, -0.02).
POSES: dict[str, Pose] = {
    # job 1, close three-quarter from the front right: the cabinet's top (where the bowl
    # lands), its middle drawer (front faces +y) and the hand large; the bowl's start and
    # the plate below; the drawer stays readable when the arm comes in to open it
    "hero-cabinet": Pose(position=(0.60, 0.26, 1.53), look_at=(-0.03, -0.14, 1.09), fovy=35.0),
    # job 2, steeper three-quarter from the front right: cream cheese, bowl and plate, the
    # push of the plate to the stove front (left to right, +y), the cabinet top where the
    # bowl goes last; the plate's rim stays in sight beside the hand while it pushes
    "hero-plate": Pose(position=(0.50, 0.30, 1.62), look_at=(-0.09, -0.03, 1.00), fovy=37.0),
    # the whole station and the robot, three-quarter from the front right, table legs in
    "establishing": Pose(position=(1.45, 1.05, 1.80), look_at=(-0.30, -0.05, 1.02), fovy=38.0),
    # straight down on the working area (x -0.51..0.19, the stove to the cabinet's front),
    # the robot at the top and +y to the right, as the scene camera has them; 24 degrees
    # from 1.65 m keeps the furniture's lean small for diagrams drawn on top
    "top": Pose(position=(-0.16, -0.08, 2.55), look_at=(-0.16, -0.08, 0.90), fovy=24.0,
                up=(-1.0, 0.0, 0.0)),
    # the same working area from 25 degrees off vertical, from the front left (the cabinet's
    # side), the robot still at the top: straight down, the arm hides the cream cheese while
    # the hand works it (0 visible pixels through the slide of session 5's e001501); from
    # here it stays in sight beside the fingers in every touch of the four reference episodes
    "top-tilt": Pose(position=(0.41, -0.48, 2.40), look_at=(-0.16, -0.08, 0.90), fovy=26.0,
                     up=(-1.0, 0.0, 0.0)),
}  # fmt: skip


@dataclass(frozen=True)
class CameraSpec:
    """A clip's camera: a named camera of the scene (`name`) or a placed one (`pose`)."""

    label: str
    name: str | None = None
    pose: Pose | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "name": self.name,
            "pose": self.pose.as_json() if self.pose is not None else None,
        }


def resolve_camera(
    value: object, poses: Mapping[str, Pose] | None = None, where: str = "camera"
) -> CameraSpec:
    """A camera from a manifest: a pose object, the name of a pose (the manifest's own, then
    `POSES`), `scene` or `wrist` (the observation cameras `agentview` and
    `robot0_eye_in_hand`), or any other camera of the scene by its MuJoCo name, which the
    replay checks."""
    if isinstance(value, Mapping):
        data = dict(value)
        label = data.pop("label", "custom")
        if not isinstance(label, str) or not ID_PATTERN.match(label):
            raise ClipsError(f"{where}.label: a short name expected")
        return CameraSpec(label=label, pose=Pose.from_json(data, where))
    if not isinstance(value, str) or not value:
        raise ClipsError(f"{where}: a camera name or a pose expected")
    table = {**POSES, **(poses or {})}
    if value in table:
        return CameraSpec(label=value, pose=table[value])
    return CameraSpec(label=value, name=CAMERA_ALIASES.get(value, value))


@dataclass(frozen=True)
class View:
    """The pinhole camera a frame was drawn with; `project` puts world points on its picture."""

    width: int
    height: int
    position: Array
    rotation: Array  # world to camera: its rows are right, down and forward
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float

    @classmethod
    def from_frustum(
        cls,
        width: int,
        height: int,
        position: npt.ArrayLike,
        forward: npt.ArrayLike,
        up: npt.ArrayLike,
        *,
        near: float,
        far: float,
        top: float,
        bottom: float,
        center: float = 0.0,
        half_width: float = 0.0,
    ) -> View:
        """From an OpenGL frustum as MuJoCo draws it: the half width follows the picture's
        aspect unless given (`mjvGLCamera.frustum_width`)."""
        f = _unit(forward, "forward")
        right = _unit(np.cross(f, np.asarray(up, np.float64)), "right")
        u = np.cross(right, f)
        half = half_width if half_width > 0 else 0.5 * width / height * (top - bottom)
        left, right_edge = center - half, center + half
        return cls(
            width=width,
            height=height,
            position=np.asarray(position, np.float64).copy(),
            rotation=np.stack([right, -u, f]),
            fx=near * width / (right_edge - left),
            fy=near * height / (top - bottom),
            cx=-width * left / (right_edge - left),
            cy=height * top / (top - bottom),
            near=float(near),
            far=float(far),
        )

    @classmethod
    def from_gl(cls, cameras: Sequence[Any], width: int, height: int) -> View:
        """MuJoCo's two eyes (`mjvScene.camera`) averaged, as its mono rendering does."""

        def mean(attr: str) -> Array:
            stacked = np.stack([np.asarray(getattr(c, attr), np.float64) for c in cameras])
            out: Array = stacked.mean(axis=0)
            return out

        def scalar(attr: str) -> float:
            return float(np.mean([float(getattr(c, attr)) for c in cameras]))

        return cls.from_frustum(
            width,
            height,
            mean("pos"),
            mean("forward"),
            mean("up"),
            near=scalar("frustum_near"),
            far=scalar("frustum_far"),
            top=scalar("frustum_top"),
            bottom=scalar("frustum_bottom"),
            center=scalar("frustum_center"),
            half_width=scalar("frustum_width"),
        )

    @classmethod
    def from_pose(cls, pose: Pose, width: int, height: int, near: float, far: float) -> View:
        forward, up, _ = pose.axes()
        top = near * math.tan(math.radians(pose.fovy) / 2)
        return cls.from_frustum(
            width, height, pose.position, forward, up, near=near, far=far, top=top, bottom=-top
        )

    @property
    def forward(self) -> Array:
        out: Array = self.rotation[2]
        return out

    @property
    def up(self) -> Array:
        out: Array = -self.rotation[1]
        return out

    @property
    def right(self) -> Array:
        out: Array = self.rotation[0]
        return out

    def project(self, points: npt.ArrayLike) -> tuple[Array, Array]:
        """Pixel positions (n x 2; NaN behind the camera) and depths (n) of world points."""
        p = (np.atleast_2d(np.asarray(points, np.float64)) - self.position) @ self.rotation.T
        z = p[:, 2]
        uv = np.full((len(p), 2), np.nan)
        front = z > 1e-9
        uv[front, 0] = self.fx * p[front, 0] / z[front] + self.cx
        uv[front, 1] = self.fy * p[front, 1] / z[front] + self.cy
        return uv, z

    def in_frame(self, u: float, v: float, depth: float) -> bool:
        return bool(
            depth > self.near
            and math.isfinite(u)
            and math.isfinite(v)
            and 0.0 <= u < self.width
            and 0.0 <= v < self.height
        )

    def matrix(self) -> Array:
        """K, the 3 x 3 intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def world_to_camera(self) -> Array:
        """4 x 4: world points to OpenCV camera coordinates (x right, y down, z forward)."""
        out = np.eye(4)
        out[:3, :3] = self.rotation
        out[:3, 3] = -self.rotation @ self.position
        return out

    def as_json(self) -> dict[str, Any]:
        def r(values: Iterable[float], digits: int = 6) -> list[float]:
            return [round(float(x), digits) for x in values]

        return {
            "width": self.width,
            "height": self.height,
            "fovy_deg": round(math.degrees(2 * math.atan(self.height / (2 * self.fy))), 4),
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "K": [r(row, 4) for row in self.matrix()],
            "position": r(self.position),
            "forward": r(self.forward),
            "up": r(self.up),
            "right": r(self.right),
            "world_to_camera": [r(row) for row in self.world_to_camera()],
            "near": round(self.near, 6),
            "far": round(self.far, 3),
        }


class Renderer:
    """Draws one camera of a running MuJoCo scene at width x height, upright and not mirrored.

    `render` returns the picture and the `View` it was drawn with; for a placed camera the
    view is written into the updated scene, its two eyes and the headlight, and for a named
    camera it is MuJoCo's own. `segment` draws the same view once more with MuJoCo's
    segmentation colours and returns the body seen at each pixel. Neither touches the model
    or the simulation state.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        context: Any,
        camera: CameraSpec,
        width: int,
        height: int,
        make_current: Callable[[], None] | None = None,
    ) -> None:
        import mujoco

        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ClipsError(f"{width} x {height}: even sizes are needed for H.264")
        self.model, self.data, self.context = model, data, context
        self.camera, self.width, self.height = camera, width, height
        self._make_current = make_current or (lambda: None)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.moving = False
        if camera.name is not None:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
            if cam_id < 0:
                names = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(model.ncam)
                ]
                raise ClipsError(f"the scene has no camera {camera.name!r}; it has {names}")
            self._cam.fixedcamid = cam_id
            self.moving = bool(model.cam_bodyid[cam_id] != 0)
        elif camera.pose is not None:
            self._cam.fixedcamid = -1
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        else:
            raise ClipsError(f"camera {camera.label!r} has neither a name nor a pose")
        extent = float(model.stat.extent)
        self._near = float(model.vis.map.znear) * extent
        self._far = float(model.vis.map.zfar) * extent

    @classmethod
    def of_env(cls, env: Any, camera: CameraSpec, width: int, height: int) -> Renderer:
        """For a `LiberoSimEnv`: its model, data and offscreen context."""
        sim = env._sim
        return cls(
            sim.model._model,
            sim.data._data,
            sim._render_context_offscreen,
            camera,
            width,
            height,
            make_current=env._inner._make_current,
        )

    def _ensure_buffer(self) -> None:
        con = self.context.con
        if self.width > con.offWidth or self.height > con.offHeight:
            # robosuite's own way: the model's offscreen size and a new MuJoCo context
            self.context.update_offscreen_size(
                max(self.width, con.offWidth), max(self.height, con.offHeight)
            )

    def _update(self) -> View:
        import mujoco

        ctx = self.context
        scn = ctx.scn
        mujoco.mjv_updateScene(
            self.model, self.data, ctx.vopt, ctx.pert, self._cam, mujoco.mjtCatBit.mjCAT_ALL, scn
        )
        if self.camera.pose is None:
            return View.from_gl([scn.camera[0], scn.camera[1]], self.width, self.height)
        view = View.from_pose(self.camera.pose, self.width, self.height, self._near, self._far)
        top = self._near * math.tan(math.radians(self.camera.pose.fovy) / 2)
        for i in range(2):
            eye = scn.camera[i]
            eye.pos[:] = view.position
            eye.forward[:] = view.forward
            eye.up[:] = view.up
            eye.frustum_near, eye.frustum_far = self._near, self._far
            eye.frustum_top, eye.frustum_bottom = top, -top
            eye.frustum_center, eye.frustum_width = 0.0, 0.0
        for i in range(scn.nlight):
            light = scn.lights[i]
            if light.headlight:  # MuJoCo puts it at the camera when it updates the scene
                light.pos[:] = view.position
                light.dir[:] = view.forward
        return view

    def render(self) -> tuple[Image, View]:
        import mujoco

        self._make_current()
        self._ensure_buffer()
        view = self._update()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.context.scn, self.context.con)
        rgb = np.empty((self.height, self.width, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context.con)
        return np.ascontiguousarray(rgb[::-1]), view

    def segment(self, scale: int = 1) -> npt.NDArray[np.int32]:
        """The body seen at each pixel (-1 where none), of the scene as last rendered, drawn
        at 1 / `scale` of the size. Call right after `render`."""
        import mujoco

        width, height = self.width // scale, self.height // scale
        scn, con = self.context.scn, self.context.con
        segment, idcolor = mujoco.mjtRndFlag.mjRND_SEGMENT, mujoco.mjtRndFlag.mjRND_IDCOLOR
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), np.uint8)
        self._make_current()
        before = scn.flags[segment], scn.flags[idcolor]
        scn.flags[segment], scn.flags[idcolor] = 1, 1
        try:
            mujoco.mjr_render(viewport, scn, con)
            mujoco.mjr_readPixels(rgb, None, viewport, con)
        finally:
            scn.flags[segment], scn.flags[idcolor] = before
        code = rgb[::-1].astype(np.int32)
        ids = code[..., 0] | (code[..., 1] << 8) | (code[..., 2] << 16)  # segid + 1; 0: none
        lut = np.full(scn.ngeom + 1, -1, np.int32)
        for i in range(scn.ngeom):
            geom = scn.geoms[i]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and 0 <= geom.segid < scn.ngeom:
                lut[geom.segid + 1] = int(self.model.geom_bodyid[geom.objid])
        ids[ids > scn.ngeom] = 0
        out: npt.NDArray[np.int32] = lut[ids]
        return out


def _segment_scale(width: int, height: int) -> int:
    scale = max(1, width // SEGMENT_WIDTH)
    return scale if width % scale == 0 and height % scale == 0 else 1


# ---- what is tracked -----------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """Something the track follows: its point (a body's origin or a site), the bodies that
    make it up (for its visible pixels) and their visible geoms (for its centre). A region
    (`kind` site: a goal's region, such as the stove front) has no bodies of its own; the
    track gives the outline of its box's top face instead."""

    name: str
    kind: str  # object | gripper | body | site
    body: int
    site: int | None
    bodies: tuple[int, ...]
    geoms: tuple[int, ...]

    def describe(self, model: Any) -> dict[str, Any]:
        import mujoco

        def body_name(b: int) -> str | None:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            return None if name is None else str(name)

        point = (
            f"site {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, self.site)}"
            if self.site is not None
            else f"origin of body {body_name(self.body)}"
        )
        return {
            "kind": self.kind,
            "point": point,
            "body": body_name(self.body),
            "bodies": len(self.bodies),
        }


def _subtree(model: Any, root: int) -> tuple[int, ...]:
    parents = np.asarray(model.body_parentid)
    out = []
    for b in range(int(model.nbody)):
        walk = b
        while walk != root and walk != 0:
            walk = int(parents[walk])
        if walk == root:
            out.append(b)
    return tuple(out)


def _visible_geoms(model: Any, bodies: Sequence[int], vopt: Any) -> tuple[int, ...]:
    shown = np.asarray(vopt.geomgroup)
    members = set(bodies)
    out = []
    for g in range(int(model.ngeom)):
        group = int(model.geom_group[g])
        if (
            int(model.geom_bodyid[g]) in members
            and group < len(shown)
            and shown[group]
            and float(model.geom_rgba[g][3]) > 0
        ):
            out.append(g)
    return tuple(out)


def targets_of(
    env: Any, objects: Sequence[str], extra: Sequence[str] = (), regions: Sequence[str] = ()
) -> list[Target]:
    """The logged objects (root bodies of the scene's objects), the gripper (the robot's
    end-effector site, the gripper's bodies), extra targets by object, body or site name, and
    the goals' regions (sites; those that are not sites are left out)."""
    import mujoco

    model = env._sim.model._model
    rs = env._rs
    vopt = env._sim._render_context_offscreen.vopt
    roots: dict[str, int] = {name: int(body) for name, body in rs.obj_body_id.items()}
    out = []
    for name in objects:
        if name not in roots:
            raise ClipsError(f"the scene has no object {name!r}")
        bodies = _subtree(model, roots[name])
        out.append(
            Target(name, "object", roots[name], None, bodies, _visible_geoms(model, bodies, vopt))
        )
    robot = rs.robots[0]
    site = int(robot.eef_site_id["right"])
    gripper_root = getattr(robot.gripper["right"], "root_body", None)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_root) if gripper_root else -1
    if root < 0:
        root = int(model.body_parentid[int(model.site_bodyid[site])])
    bodies = _subtree(model, root)
    out.append(Target(GRIPPER, "gripper", root, site, bodies, _visible_geoms(model, bodies, vopt)))
    names = {t.name for t in out}
    for name in [*extra, *regions]:
        if name in names:
            continue
        names.add(name)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if name in roots:
            body = roots[name]
        elif (body := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) < 0:
            if site < 0:
                if name in regions:
                    continue
                raise ClipsError(f"the scene has no object, body or site {name!r}")
            out.append(Target(name, "site", int(model.site_bodyid[site]), site, (), ()))
            continue
        bodies = _subtree(model, body)
        out.append(Target(name, "body", body, None, bodies, _visible_geoms(model, bodies, vopt)))
    return out


def _site_outline(model: Any, data: Any, site: int) -> Array | None:
    """The four corners of the face of a site's box that looks most upward in the world (a
    region's footprint or opening), in order around it; None for a site that is no box."""
    import mujoco

    if int(model.site_type[site]) != mujoco.mjtGeom.mjGEOM_BOX:
        return None
    half = np.asarray(model.site_size[site], np.float64)
    axes = np.asarray(data.site_xmat[site], np.float64).reshape(3, 3)  # columns: local axes
    up = int(np.argmax(np.abs(axes[2])))
    a, b = (i for i in range(3) if i != up)
    face = (
        np.asarray(data.site_xpos[site], np.float64) + np.sign(axes[2, up]) * half[up] * axes[:, up]
    )
    signs = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    corners = [face + i * half[a] * axes[:, a] + j * half[b] * axes[:, b] for i, j in signs]
    return np.asarray(corners, np.float64)


_CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _centre(model: Any, data: Any, geoms: Sequence[int]) -> Array | None:
    """The centre of the world-aligned box around the geoms' own bounding boxes."""
    if not geoms:
        return None
    index = np.asarray(geoms)
    aabb = np.asarray(model.geom_aabb)[index]
    local = aabb[:, None, :3] + _CORNERS[None] * aabb[:, None, 3:]
    xmat = np.asarray(data.geom_xmat)[index].reshape(-1, 3, 3)
    world = np.einsum("nij,nkj->nki", xmat, local) + np.asarray(data.geom_xpos)[index][:, None]
    flat = world.reshape(-1, 3)
    centre: Array = (flat.min(axis=0) + flat.max(axis=0)) / 2
    return centre


# ---- the run's own log for one episode -----------------------------------------------------


@dataclass(frozen=True)
class EpisodeLog:
    """An episode's recorded per-step values, read from the run's log tables."""

    objects: tuple[str, ...]
    object_rows: dict[int, dict[str, dict[str, Any]]]  # step -> object -> row
    goal_rows: dict[int, list[bool]]  # step -> holds by goal
    robot_rows: dict[int, dict[str, Any]]  # step -> row
    goals: list[dict[str, Any]]
    program: list[dict[str, Any]]

    @classmethod
    def read(cls, run_dir: Path, episode_id: str) -> EpisodeLog:
        import pyarrow.parquet as pq

        def rows(table: str) -> list[dict[str, Any]]:
            path = tables.table_path(run_dir, table, episode_id)
            return list(pq.read_table(path).to_pylist())

        object_rows: dict[int, dict[str, dict[str, Any]]] = {}
        objects: list[str] = []
        for row in rows("object_steps"):
            object_rows.setdefault(int(row["step"]), {})[row["object"]] = row
            if row["object"] not in objects:
                objects.append(row["object"])
        goal_rows: dict[int, list[bool]] = {}
        for row in sorted(rows("goal_steps"), key=lambda r: (r["step"], r["goal"])):
            goal_rows.setdefault(int(row["step"]), []).append(bool(row["holds"]))
        robot_rows = {int(row["step"]): row for row in rows("robot_steps")}
        state = json.loads((run_dir / tables.RUN_RECORD).read_text())["state"]
        goals = [
            {
                "index": i,
                "predicate": g["predicate"],
                "objects": list(g["objects"]),
                "text": " ".join([g["predicate"], *g["objects"]]),
            }
            for i, g in enumerate(state["requested_goals"])
        ]
        events_path = tables.table_path(run_dir, "job_step_events")
        program = []
        if events_path.is_file():
            for row in pq.read_table(events_path).to_pylist():
                if row["episode_id"] == episode_id:
                    program.append({k: v for k, v in row.items() if k != "episode_id"})
        return cls(tuple(objects), object_rows, goal_rows, robot_rows, goals, program)

    def flags(self, step: int) -> dict[str, Any]:
        objects = self.object_rows.get(step, {})
        robot = self.robot_rows.get(step, {})
        return {
            "contact": {n: bool(r["in_contact_with_robot"]) for n, r in objects.items()},
            "touching": {n: list(r["in_contact_with"]) for n, r in objects.items()},
            "goals": list(self.goal_rows.get(step, [])),
            "gripper_opening": robot.get("gripper_opening"),
            "gripper_command": robot.get("gripper_command"),
            "program_step": robot.get("program_step"),
            "instruction": robot.get("instruction"),
            "phase": robot.get("phase"),
        }

    def first_contact(self) -> int | None:
        """The first step at which the robot touches a logged object."""
        touched = [
            step
            for step, rows in self.object_rows.items()
            if any(r["in_contact_with_robot"] for r in rows.values())
        ]
        return min(touched) if touched else None


# ---- clips and the manifest ----------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """One framed re-render: an episode of a run, a camera, a size, a cadence and a range."""

    id: str
    run_dir: Path
    episode_id: str
    camera: CameraSpec
    width: int = 1920
    height: int = 1080
    frame_every: int = 1
    from_step: int | None = None
    to_step: int | None = None
    bodies: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        if not ID_PATTERN.match(self.id):
            raise ClipsError(f"clip id {self.id!r}: letters, digits, '.', '_' and '-' only")
        if self.width < 2 or self.height < 2 or self.width % 2 or self.height % 2:
            raise ClipsError(f"{self.id}: {self.width} x {self.height}; even sizes needed")
        if self.frame_every < 1 or CONTROL_HZ % self.frame_every:
            raise ClipsError(f"{self.id}: frame_every divides {CONTROL_HZ}, not {self.frame_every}")
        if (
            self.from_step is not None
            and self.to_step is not None
            and self.from_step > self.to_step
        ):
            raise ClipsError(f"{self.id}: from_step {self.from_step} > to_step {self.to_step}")

    @property
    def fps(self) -> int:
        return CONTROL_HZ // self.frame_every

    def steps(self, episode_steps: int) -> list[int]:
        """The control steps drawn: every `frame_every`-th from `from_step`, and the last."""
        first = 0 if self.from_step is None else self.from_step
        last = episode_steps if self.to_step is None else min(self.to_step, episode_steps)
        if first > last:
            return []
        out = list(range(first, last + 1, self.frame_every))
        if out[-1] != last:
            out.append(last)
        return out

    def spec(self) -> dict[str, Any]:
        """Everything that decides the outputs; equal specs give the same outputs."""
        return {
            "format": REPORT_FORMAT,
            "renderer": RENDERER_VERSION,
            "run_dir": str(self.run_dir),
            "episode_id": self.episode_id,
            "camera": self.camera.as_json(),
            "width": self.width,
            "height": self.height,
            "frame_every": self.frame_every,
            "from_step": self.from_step,
            "to_step": self.to_step,
            "bodies": list(self.bodies),
            "observation_size": OBSERVATION_SIZE,
        }


@dataclass(frozen=True)
class ClipsManifest:
    clips: tuple[Clip, ...]
    poses: dict[str, Pose] = field(default_factory=dict)
    path: Path | None = None


CLIP_KEYS = {
    "id", "run", "episode", "camera", "width", "height", "frame_every", "from_step", "to_step",
    "bodies", "note",
}  # fmt: skip
DEFAULT_KEYS = {"run", "camera", "width", "height", "frame_every", "bodies"}


def _whole(data: Mapping[str, Any], key: str, where: str, default: int | None) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ClipsError(f"{where}.{key}: a whole number expected")
    return value


def load_clips(path: Path) -> ClipsManifest:
    """Read a clips manifest (`CLIPS_FORMAT`), JSON:

        {"format": "fleetlab.film-clips/1",
         "about": "free text",
         "poses": {"name": {"position": [x, y, z], "look_at": [x, y, z], "fovy": 36,
                            "roll": 0, "up": [0, 0, 1]}},          # optional, beside POSES
         "defaults": {"width": 1920, "height": 1080, "frame_every": 1,
                      "camera": ..., "run": ..., "bodies": [...]},  # optional
         "clips": [{"id": "j1-chain-e000202-hero",                # file-name safe, unique
                    "run": "~/.cache/robots-probes/.../run",      # a job run directory
                    "episode": "e000202",
                    "camera": "hero-cabinet" | "scene" | "frontview" | {pose ...},
                    "width": 1920, "height": 1080, "frame_every": 1,  # divides 20
                    "from_step": null, "to_step": null,           # the frames drawn
                    "bodies": ["wooden_cabinet_1_cabinet_middle"],  # also tracked
                    "note": "free text"}]}

    A relative `run` is taken from the manifest's folder; `~` is expanded. A camera name is
    the manifest's pose, else one of `POSES`, else `scene`, `wrist` or a camera of the scene.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ClipsError(f"{path}: not JSON ({e})") from e
    return parse_clips(data, path.parent, path)


def load_poses(path: Path) -> dict[str, Pose]:
    """The named poses of a clips manifest (its `poses`), for a command that takes poses."""
    data = json.loads(Path(path).read_text())
    raw = data.get("poses") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ClipsError(f"{path}: no poses")
    return {name: Pose.from_json(pose, f"poses.{name}") for name, pose in raw.items()}


def parse_clips(data: Any, base: Path, path: Path | None = None) -> ClipsManifest:
    if not isinstance(data, dict):
        raise ClipsError("a clips manifest is a JSON object")
    if data.get("format") != CLIPS_FORMAT:
        raise ClipsError(f"format {data.get('format')!r}, not {CLIPS_FORMAT!r}")
    unknown = sorted(set(data) - {"format", "about", "poses", "defaults", "clips"})
    if unknown:
        raise ClipsError(f"unknown field(s) {', '.join(unknown)}")
    raw_poses = data.get("poses") or {}
    if not isinstance(raw_poses, dict):
        raise ClipsError("poses: an object of named poses expected")
    poses = {}
    for name, pose in raw_poses.items():
        if not ID_PATTERN.match(name) or not isinstance(pose, dict):
            raise ClipsError(f"poses.{name}: a short name and a pose object expected")
        poses[name] = Pose.from_json(pose, f"poses.{name}")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict) or set(defaults) - DEFAULT_KEYS:
        raise ClipsError(f"defaults: an object with some of {', '.join(sorted(DEFAULT_KEYS))}")
    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ClipsError("clips: a list of clips expected")
    clips: list[Clip] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_clips):
        where = f"clips[{i}]"
        if not isinstance(raw, dict):
            raise ClipsError(f"{where}: an object expected")
        unknown = sorted(set(raw) - CLIP_KEYS)
        if unknown:
            raise ClipsError(f"{where}: unknown field(s) {', '.join(unknown)}")
        item = {**defaults, **raw}
        clip_id = item.get("id")
        if not isinstance(clip_id, str):
            raise ClipsError(f"{where}.id: a name expected")
        if clip_id in seen:
            raise ClipsError(f"{where}: the id {clip_id!r} is used twice")
        seen.add(clip_id)
        where = f"clips[{i}] ({clip_id})"
        run = item.get("run")
        if not isinstance(run, str) or not run:
            raise ClipsError(f"{where}.run: a run directory expected")
        run_dir = Path(run).expanduser()
        if not run_dir.is_absolute():
            run_dir = base / run_dir
        episode = item.get("episode")
        if not isinstance(episode, str) or not re.match(r"^e\d+$", episode):
            raise ClipsError(f"{where}.episode: an episode id such as e000202 expected")
        if "camera" not in item:
            raise ClipsError(f"{where}.camera: required")
        bodies = item.get("bodies") or []
        if not isinstance(bodies, list) or not all(isinstance(b, str) for b in bodies):
            raise ClipsError(f"{where}.bodies: a list of names expected")
        note = item.get("note")
        if note is not None and not isinstance(note, str):
            raise ClipsError(f"{where}.note: text expected")
        clips.append(
            Clip(
                id=clip_id,
                run_dir=run_dir.resolve(),
                episode_id=episode,
                camera=resolve_camera(item["camera"], poses, f"{where}.camera"),
                width=_whole(item, "width", where, 1920) or 0,
                height=_whole(item, "height", where, 1080) or 0,
                frame_every=_whole(item, "frame_every", where, 1) or 0,
                from_step=_whole(item, "from_step", where, None),
                to_step=_whole(item, "to_step", where, None),
                bodies=tuple(bodies),
                note=note,
            )
        )
    return ClipsManifest(tuple(clips), poses, path)


# ---- one clip's outputs --------------------------------------------------------------------


def _names(clip_id: str) -> dict[str, str]:
    return {
        "video": f"{clip_id}.rerender.mp4",
        "report": f"{clip_id}.rerender.json",
        "track": f"{clip_id}.track.json",
        "debug": f"{clip_id}.debug.png",
        "rejected": f"{clip_id}.rejected-rerender.json",
        "partial": f"{clip_id}.partial.mp4",
    }


class _VideoWriter:
    """Frames streamed to ffmpeg as `fleetlab.video.write_mp4` encodes them."""

    def __init__(self, path: Path, fps: int) -> None:
        import imageio.v2 as imageio

        from fleetlab.video import REPEATABLE_PARAMS

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_params=["-crf", str(CRF), "-preset", PRESET, *REPEATABLE_PARAMS],
        )
        self.frames = 0

    def write(self, frame: Image) -> None:
        self._writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        self._writer.close()

    def abort(self) -> None:
        with contextlib.suppress(Exception):  # the partial file goes either way
            self._writer.close()
        self.path.unlink(missing_ok=True)


def _r(value: float, digits: int) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


class _ClipJob:
    """One clip drawn during a replay: its renderer, encoder and track, then its outputs."""

    def __init__(
        self, clip: Clip, out_dir: Path, record: jobs_replay.ReplayRecord, source_sha256: str
    ) -> None:
        self.clip, self.out_dir, self.record = clip, out_dir, record
        self.source_sha256 = source_sha256
        self.names = _names(clip.id)
        self.error: str | None = None
        self.steps = clip.steps(len(record.executed))
        self._wanted = set(self.steps)
        self.log: EpisodeLog | None = None
        self.renderer: Renderer | None = None
        self.targets: list[Target] = []
        self.writer: _VideoWriter | None = None
        self.rows: list[dict[str, Any]] = []
        self.first_view: View | None = None
        self.debug: dict[int, tuple[Image, dict[str, Any]]] = {}
        self.debug_steps: set[int] = set()
        self.position_dev = {"objects_m": 0.0, "gripper_m": 0.0}
        self.seg_scale = _segment_scale(clip.width, clip.height)
        self.started = time.monotonic()
        if not self.steps:
            self.error = f"no control step of the episode is in {clip.from_step}..{clip.to_step}"

    def prepare(self, env: Any) -> None:
        if self.error is not None:
            return
        try:
            self.log = EpisodeLog.read(self.record.run_dir, self.record.episode_id)
            self.renderer = Renderer.of_env(
                env, self.clip.camera, self.clip.width, self.clip.height
            )
            regions = [g["objects"][-1] for g in self.log.goals]
            self.targets = targets_of(env, self.log.objects, self.clip.bodies, regions)
            self.debug_steps = self._choose_debug_steps(self.log)
            self.writer = _VideoWriter(self.out_dir / self.names["partial"], self.clip.fps)
        except Exception as e:
            self.fail(e)

    def _choose_debug_steps(self, log: EpisodeLog) -> set[int]:
        """Four drawn steps: the first, the first touch, the last, then the middle ones."""
        steps = self.steps
        contact = log.first_contact()
        touched = [i for i, s in enumerate(steps) if contact is not None and s >= contact][:1]
        wanted: list[int] = []
        for i in [0, *touched, len(steps) - 1, len(steps) // 2]:
            if steps[i] not in wanted:
                wanted.append(steps[i])
        for fraction in (1 / 4, 3 / 4, 1 / 8, 7 / 8):
            step = steps[int(fraction * (len(steps) - 1))]
            if len(wanted) < 4 and step not in wanted:
                wanted.append(step)
        return set(wanted[:4])

    def fail(self, error: BaseException) -> None:
        self.error = f"{type(error).__name__}: {error}"
        if self.writer is not None:
            self.writer.abort()
            self.writer = None

    def observe(self, env: Any, step: int) -> None:
        if self.error is not None or step not in self._wanted:
            return
        assert self.renderer is not None and self.writer is not None and self.log is not None
        try:
            image, view = self.renderer.render()
            body_image = self.renderer.segment(self.seg_scale)
            self.writer.write(image)
            row = self._row(step, view, body_image)
            self.rows.append(row)
            if self.first_view is None:
                self.first_view = view
            if self.renderer.moving:  # a camera on the robot: its pose in every row
                pose = view.as_json()
                row["camera"] = {
                    k: pose[k] for k in ("position", "forward", "up", "world_to_camera")
                }
            if step in self.debug_steps:
                self.debug[step] = (image.copy(), row)
        except Exception as e:
            self.fail(e)

    def _row(self, step: int, view: View, body_image: npt.NDArray[np.int32]) -> dict[str, Any]:
        assert self.log is not None and self.renderer is not None
        model, data = self.renderer.model, self.renderer.data
        index = body_image + 1  # -1 (nothing) -> 0
        nbody = int(model.nbody)
        scale = self.seg_scale
        targets: dict[str, Any] = {}
        logged = self.log.object_rows.get(step, {})
        robot_row = self.log.robot_rows.get(step, {})
        for target in self.targets:
            if target.site is not None:
                point = np.asarray(data.site_xpos[target.site], np.float64).copy()
            else:
                point = np.asarray(data.xpos[target.body], np.float64).copy()
            if target.kind == "object" and target.name in logged:
                dev = float(np.abs(point - np.asarray(logged[target.name]["position"])).max())
                self.position_dev["objects_m"] = max(self.position_dev["objects_m"], dev)
            if target.kind == "gripper" and robot_row.get("eef_pos") is not None:
                dev = float(np.abs(point - np.asarray(robot_row["eef_pos"])).max())
                self.position_dev["gripper_m"] = max(self.position_dev["gripper_m"], dev)
            if target.kind == "site":
                assert target.site is not None
                corners = _site_outline(model, data, target.site)
                outline = view.project(corners)[0] if corners is not None else np.empty((0, 2))
                uv, depth = view.project(point)
                targets[target.name] = {
                    "u": _r(uv[0, 0], 1),
                    "v": _r(uv[0, 1], 1),
                    "depth": _r(depth[0], 4),
                    "in_frame": view.in_frame(float(uv[0, 0]), float(uv[0, 1]), float(depth[0])),
                    "outline": [[_r(u, 1), _r(v, 1)] for u, v in outline] or None,
                }
                continue
            centre = _centre(model, data, target.geoms)
            pts = np.stack([point, centre if centre is not None else point])
            uv, depth = view.project(pts)
            member = np.zeros(nbody + 1, bool)
            member[np.asarray(target.bodies, np.int64) + 1] = True
            mask = member[index]
            count = int(mask.sum())
            bbox = None
            if count:
                rows = np.flatnonzero(mask.any(axis=1))
                cols = np.flatnonzero(mask.any(axis=0))
                bbox = [
                    int(cols[0]) * scale,
                    int(rows[0]) * scale,
                    (int(cols[-1]) + 1) * scale,
                    (int(rows[-1]) + 1) * scale,
                ]
            targets[target.name] = {
                "u": _r(uv[0, 0], 1),
                "v": _r(uv[0, 1], 1),
                "depth": _r(depth[0], 4),
                "cu": _r(uv[1, 0], 1),
                "cv": _r(uv[1, 1], 1),
                "in_frame": view.in_frame(float(uv[0, 0]), float(uv[0, 1]), float(depth[0])),
                "visible_px": count * scale * scale,
                "bbox": bbox,
            }
        return {
            "frame": len(self.rows),
            "step": step,
            "t": round(step / CONTROL_HZ, 3),
            "targets": targets,
            **self.log.flags(step),
        }

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.abort()
            self.writer = None

    # -- after the replay --

    def finish(self, check: jobs_replay.Verification) -> dict[str, Any]:
        names, out = self.names, self.out_dir
        if self.writer is not None and self.error is None:
            try:
                self.writer.close()
            except Exception as e:
                self.fail(e)
        frames_ok = self.error is None and len(self.rows) == len(self.steps)
        accepted = check.accepted and frames_ok
        for stale in ("video", "report", "track", "debug", "rejected"):
            (out / names[stale]).unlink(missing_ok=True)  # the partial is moved or dropped
        partial = out / names["partial"]
        video_sha = track_sha = None
        if accepted:
            os.replace(partial, out / names["video"])
            video_sha = _sha256(out / names["video"])
            _write_json(out / names["track"], self._track(check, video_sha), indent=None)
            track_sha = _sha256(out / names["track"])
            _debug_still(out / names["debug"], self.clip, self.debug, self.targets)
        else:
            partial.unlink(missing_ok=True)
        report = self._report(check, accepted, video_sha, track_sha)
        _write_json(out / (names["report"] if accepted else names["rejected"]), report)
        return report

    def _reason(self, check: jobs_replay.Verification) -> str | None:
        record = check.record
        if record.problems:
            return "; ".join(record.problems)
        if self.error is not None:
            return self.error
        if check.mismatches:
            return (
                f"{len(check.mismatches)} state hash(es) differ from the record, the first "
                f"after step {check.mismatches[0]}"
            )
        if check.compared != len(record.hashes):
            return f"{check.compared} of {len(record.hashes)} recorded hashes compared"
        if not check.verdict_matches:
            return "the replayed delivery verdict is not the recorded one"
        if len(self.rows) != len(self.steps):
            return f"{len(self.rows)} of {len(self.steps)} frames drawn"
        return None

    def _report(
        self,
        check: jobs_replay.Verification,
        accepted: bool,
        video_sha: str | None,
        track_sha: str | None,
    ) -> dict[str, Any]:
        record, recorded = check.record, check.record.recorded
        names = self.names
        return {
            "format": REPORT_FORMAT,
            "label": "re-render" if accepted else "rejected re-render",
            "accepted": accepted,
            "reason": None if accepted else self._reason(check),
            "clip": self.clip.id,
            "note": self.clip.note,
            "spec": self.clip.spec(),
            "problems": record.problems,
            "error": self.error,
            "episode_id": record.episode_id,
            "job_id": recorded.job_id if recorded is not None else None,
            "run_dir": str(record.run_dir),
            "source_sha256": self.source_sha256,
            "video": names["video"] if accepted else None,
            "video_sha256": video_sha,
            "track": names["track"] if accepted else None,
            "track_sha256": track_sha,
            "debug_still": names["debug"] if accepted else None,
            "camera": self.clip.camera.as_json(),
            "view": self.first_view.as_json() if self.first_view is not None else None,
            "camera_moves": bool(self.renderer is not None and self.renderer.moving),
            "width": self.clip.width,
            "height": self.clip.height,
            "fps": self.clip.fps,
            "frame_every": self.clip.frame_every,
            "from_step": self.clip.from_step,
            "to_step": self.clip.to_step,
            "frames": len(self.rows) if accepted else 0,
            "first_frame_step": self.steps[0] if self.steps else None,
            "last_frame_step": self.steps[-1] if self.steps else None,
            "steps": len(record.executed),
            "recorded_hashes": len(record.hashes),
            "hashes_compared": check.compared,
            "hash_mismatches": len(check.mismatches),
            "first_mismatch_step": check.mismatches[0] if check.mismatches else None,
            "recorded_end_state_hash": recorded.end_state_hash if recorded is not None else None,
            "end_state_hash": check.replayed_end,
            "verdict_matches": check.verdict_matches,
            "recorded_verdict": recorded.delivery.model_dump(mode="json") if recorded else None,
            "replayed_verdict": (
                check.verdict.model_dump(mode="json") if check.verdict is not None else None
            ),
            "largest_position_difference": {
                k: float(f"{v:.3g}") for k, v in self.position_dev.items()
            },
            "targets": [t.name for t in self.targets],
            "observation_size": OBSERVATION_SIZE,
            "seconds": round(time.monotonic() - self.started, 1),
        }

    def _track(self, check: jobs_replay.Verification, video_sha: str) -> dict[str, Any]:
        record, recorded = check.record, check.record.recorded
        assert self.log is not None and self.renderer is not None and self.first_view is not None
        model = self.renderer.model
        return {
            "format": TRACK_FORMAT,
            "clip": self.clip.id,
            "run_dir": str(record.run_dir),
            "episode_id": record.episode_id,
            "job_id": recorded.job_id if recorded is not None else None,
            "video": self.names["video"],
            "video_sha256": video_sha,
            "fps": self.clip.fps,
            "frame_every": self.clip.frame_every,
            "width": self.clip.width,
            "height": self.clip.height,
            "frames": len(self.rows),
            "steps": len(record.executed),
            "control_hz": CONTROL_HZ,
            "conventions": {
                "pixels": "u right, v down from the top-left corner; pixel (column j, row i) "
                "has its centre at (j + 0.5, i + 0.5)",
                "time": "t: the control step's time in the episode (step / control_hz "
                "seconds); a row's time in the video is frame / fps",
                "camera": "OpenCV pinhole: world_to_camera takes world metres to x right, "
                "y down, z along the line of sight; u = fx x / z + cx, v = fy y / z + cy",
                "depth": "metres along the line of sight (camera z); u, v are null behind "
                "the camera",
                "point": "u, v, depth: the target's point at the end of the row's control "
                "step, as the frame shows it (an object's body origin, which "
                "object_steps.position records exactly; the gripper's grip site, which "
                "robot_steps.eef_pos samples one 2 ms physics substep earlier, so it "
                "trails by up to about 2 mm when the hand moves fast: the report's "
                "largest_position_difference.gripper_m); cu, cv: the centre of its visible "
                "geometry",
                "visibility": "in_frame: in front of the camera and inside the picture; "
                "visible_px, bbox: the pixels showing the target in a segmentation pass, "
                "occlusion counted",
                "regions": "a site target (a requested goal's region, or a site named in "
                "the clip): u, v, depth of its centre and outline, the four corners of the face "
                "of its box that looks most upward in the world (a footprint on the table, the "
                "opening of a drawer), in pixels; null for a site that is no box",
                "flags": "contact, touching, goals, gripper, program step, instruction, "
                "phase: the run's own recorded values for the row's control step",
            },
            "camera": {
                **self.clip.camera.as_json(),
                "view": self.first_view.as_json(),
                "moves": self.renderer.moving,
            },
            "targets": {t.name: t.describe(model) for t in self.targets},
            "goals": self.log.goals,
            "program": self.log.program,
            "delivered": recorded.delivered if recorded is not None else None,
            "delivery_reason": recorded.delivery.reason if recorded is not None else None,
            "rows": self.rows,
        }


# ---- the debug still -----------------------------------------------------------------------

PALETTE = [
    (42, 120, 214), (27, 175, 122), (237, 161, 0), (180, 90, 220), (0, 190, 210),
    (240, 240, 240), (150, 110, 60),
]  # fmt: skip
GRIPPER_COLOUR = (235, 64, 52)
REGION_COLOUR = (255, 225, 60)


def _debug_still(
    path: Path,
    clip: Clip,
    frames: Mapping[int, tuple[Image, dict[str, Any]]],
    targets: Sequence[Target],
) -> None:
    """Up to four frames, each with every target's logged point (dot), centre (cross) and
    visible box drawn, so a person can see the projections sit on the objects."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    from fleetlab.film.compose import font

    steps = sorted(frames)[:4]
    if not steps:
        return
    scale = 0.5 if clip.width > 960 else 1.0
    w, h = round(clip.width * scale), round(clip.height * scale)
    cols = 2 if len(steps) > 1 else 1
    rows_n = math.ceil(len(steps) / cols)
    sheet = PILImage.new("RGB", (w * cols, h * rows_n), (17, 19, 22))
    colours = {}
    for i, t in enumerate(targets):
        colours[t.name] = {"gripper": GRIPPER_COLOUR, "site": REGION_COLOUR}.get(
            t.kind, PALETTE[i % len(PALETTE)]
        )
    size = max(10, round(h / 34))
    label_font, title_font = font(size), font(size, "bold")
    radius = max(3, round(h / 120))
    for k, step in enumerate(steps):
        image, row = frames[step]
        tile = PILImage.fromarray(image).resize((w, h), PILImage.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        for name, values in row["targets"].items():
            colour = colours.get(name, (255, 255, 255))
            outline = [
                (u * scale, v * scale)
                for u, v in values.get("outline") or []
                if u is not None and v is not None
            ]
            if len(outline) == 4:
                draw.polygon(outline, outline=REGION_COLOUR, width=2)
            if values.get("bbox") is not None:
                x0, y0, x1, y1 = (c * scale for c in values["bbox"])
                draw.rectangle([x0, y0, x1, y1], outline=colour, width=1)
            if values.get("cu") is not None and values.get("cv") is not None:
                cx, cy = values["cu"] * scale, values["cv"] * scale
                draw.line([cx - radius * 2, cy, cx + radius * 2, cy], fill=colour, width=2)
                draw.line([cx, cy - radius * 2, cx, cy + radius * 2], fill=colour, width=2)
            if values["u"] is not None and values["v"] is not None:
                x, y = values["u"] * scale, values["v"] * scale
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=colour,
                    outline=(0, 0, 0),
                )
                touching = row["contact"].get(name, False)
                text = f"{name}{' *contact*' if touching else ''}"
                draw.text((x + radius + 3, y - size // 2), text, fill=colour, font=label_font)
        goals = "".join("1" if g else "0" for g in row["goals"])
        title = f"{clip.id}  step {step}  frame {row['frame']}  goals {goals}"
        draw.rectangle([0, 0, w, size + 8], fill=(0, 0, 0))
        draw.text((6, 3), title, fill=(238, 236, 230), font=title_font)
        sheet.paste(tile, ((k % cols) * w, (k // cols) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


# ---- batches -------------------------------------------------------------------------------


def source_digest(run_dir: Path, episode_id: str) -> str:
    """SHA-256 over what a clip of the episode is drawn from: its replay record (the actions
    and state hashes, the identity), its row of `episodes.jsonl`, its per-step log tables,
    its rows of `job_step_events` and the run's request state. Raises when they cannot be
    read."""
    import pyarrow.parquet as pq

    digest = hashlib.sha256()

    def add(label: str, data: bytes) -> None:
        digest.update(f"{label}\0{len(data)}\0".encode())
        digest.update(data)

    run_dir = Path(run_dir)
    npz, identity = tables.truth_paths(run_dir, episode_id)
    add("truth", npz.read_bytes())
    add("identity", identity.read_bytes())
    marker = f'"episode_id":"{episode_id}"'
    rows = [ln for ln in (run_dir / tables.EPISODES_FILE).read_text().splitlines() if marker in ln]
    add("episode", "\n".join(rows).encode())
    for table in ("object_steps", "goal_steps", "robot_steps"):
        add(table, tables.table_path(run_dir, table, episode_id).read_bytes())
    events = tables.table_path(run_dir, "job_step_events")
    mine = (
        [r for r in pq.read_table(events).to_pylist() if r["episode_id"] == episode_id]
        if events.is_file()
        else []
    )
    add("events", json.dumps(mine, sort_keys=True, default=str).encode())
    state = json.loads((run_dir / tables.RUN_RECORD).read_text())["state"]
    add("state", json.dumps(state, sort_keys=True).encode())
    return digest.hexdigest()


def _accepted_output(clip: Clip, out_dir: Path) -> dict[str, Any] | None:
    """The clip's report when its accepted outputs exist for the same spec, intact, and were
    drawn from the episode's sources as they are now (`source_digest`)."""
    names = _names(clip.id)
    path = out_dir / names["report"]
    if not path.is_file():
        return None
    try:
        report: dict[str, Any] = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not report.get("accepted") or report.get("spec") != clip.spec():
        return None
    for name_key, sha_key in (("video", "video_sha256"), ("track", "track_sha256")):
        target = out_dir / str(report.get(name_key))
        if not target.is_file() or _sha256(target) != report.get(sha_key):
            return None
    try:
        current = source_digest(clip.run_dir, clip.episode_id)
    except (OSError, ValueError, KeyError):
        return None  # a source that cannot be read is drawn again, and rejected
    return report if report.get("source_sha256") == current else None


def _entry(clip: Clip, status: str, report: Mapping[str, Any] | None) -> dict[str, Any]:
    names = _names(clip.id)
    entry: dict[str, Any] = {
        "id": clip.id,
        "status": status,
        "accepted": bool(report and report.get("accepted")),
        "run_dir": str(clip.run_dir),
        "episode_id": clip.episode_id,
        "camera": clip.camera.label,
        "width": clip.width,
        "height": clip.height,
        "frame_every": clip.frame_every,
        "from_step": clip.from_step,
        "to_step": clip.to_step,
        "note": clip.note,
    }
    if report is None:
        return entry
    accepted = bool(report.get("accepted"))
    entry.update(
        {
            "frames": report.get("frames"),
            "video": report.get("video"),
            "video_sha256": report.get("video_sha256"),
            "track": report.get("track"),
            "track_sha256": report.get("track_sha256"),
            "debug_still": report.get("debug_still"),
            "report": names["report"] if accepted else names["rejected"],
            "steps": report.get("steps"),
            "recorded_hashes": report.get("recorded_hashes"),
            "hashes_compared": report.get("hashes_compared"),
            "hash_mismatches": report.get("hash_mismatches"),
            "first_mismatch_step": report.get("first_mismatch_step"),
            "verdict_matches": report.get("verdict_matches"),
            "delivered": (report.get("recorded_verdict") or {}).get("delivered"),
            "reason": report.get("reason"),
        }
    )
    return entry


def _index(manifest: ClipsManifest, entries: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    listed = [entries[c.id] for c in manifest.clips if c.id in entries]
    counts: dict[str, int] = {}
    for e in listed:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {
        "format": INDEX_FORMAT,
        "manifest": str(manifest.path.resolve()) if manifest.path is not None else None,
        "clips": len(listed),
        "accepted": sum(1 for e in listed if e["accepted"]),
        "statuses": counts,
        "hashes_compared": sum(int(e.get("hashes_compared") or 0) for e in listed),
        "hash_mismatches": sum(int(e.get("hash_mismatches") or 0) for e in listed),
        "entries": listed,
    }


def _same_outputs(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """Whether an index lists the same clips with the same accepted outputs as another."""
    keys = ("id", "accepted", "video_sha256", "track_sha256", "hashes_compared")
    pick = [[e.get(k) for k in keys] for e in old.get("entries", [])]
    return old.get("manifest") == new.get("manifest") and pick == [
        [e.get(k) for k in keys] for e in new["entries"]
    ]


def frame_clips(
    manifest: ClipsManifest,
    out_dir: Path,
    *,
    only: set[str] | None = None,
    force: bool = False,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Draw every clip of the manifest (or those in `only`) into `out_dir`, one replay per
    episode for all of its clips, skipping a clip whose accepted outputs exist for the same
    spec (unless `force`). Returns the index, `out_dir/index.json`, which is written once,
    when the run is complete: a run that draws removes the old one first, so an index is
    there only after a complete run, and a run that draws nothing leaves an index that lists
    the same outputs as it is."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if only is not None and (missing := only - {c.id for c in manifest.clips}):
        raise ClipsError(f"no clip(s) {', '.join(sorted(missing))} in the manifest")
    entries: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[Path, str], list[Clip]] = {}
    for clip in manifest.clips:
        existing = _accepted_output(clip, out_dir)
        chosen = only is None or clip.id in only
        if existing is not None and not (force and chosen):
            entries[clip.id] = _entry(clip, "skipped", existing)
        elif chosen:
            groups.setdefault((clip.run_dir, clip.episode_id), []).append(clip)
            entries[clip.id] = _entry(clip, "pending", None)
        else:
            entries[clip.id] = _entry(clip, "not run", None)
    skipped = sum(1 for e in entries.values() if e["status"] == "skipped")
    if skipped:
        say(f"{skipped} clip(s) already accepted for the same spec and sources: skipped")
    path = out_dir / "index.json"
    if groups:
        path.unlink(missing_ok=True)  # it would no longer describe the folder
    for n, ((run_dir, episode_id), clips) in enumerate(groups.items(), 1):
        say(f"[{n}/{len(groups)}] {episode_id} of {run_dir}: {', '.join(c.id for c in clips)}")
        for clip, report in zip(
            clips, _render_group(run_dir, episode_id, clips, out_dir), strict=True
        ):
            status = (
                "rendered"
                if report.get("accepted")
                else ("error" if report.get("error") and not report.get("problems") else "rejected")
            )
            entries[clip.id] = _entry(clip, status, report)
            say(
                f"    {clip.id}: {status}, {report.get('frames')} frames, "
                f"{report.get('hashes_compared')}/{report.get('recorded_hashes')} hashes, "
                f"{report.get('hash_mismatches')} mismatches, {report.get('seconds')} s"
                + (f" ({report.get('reason')})" if not report.get("accepted") else "")
            )
    index = _index(manifest, entries)
    if not groups and path.is_file():
        try:
            old: dict[str, Any] = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = {}
        if _same_outputs(old, index):
            return old
    _write_json(path, index)
    return index


def _render_group(
    run_dir: Path, episode_id: str, clips: Sequence[Clip], out_dir: Path
) -> list[dict[str, Any]]:
    """One replay for every clip of the same episode; a report per clip."""
    try:
        record = jobs_replay.load_record(run_dir, episode_id)
        digest = source_digest(run_dir, episode_id)
    except (OSError, ValueError, KeyError) as e:
        return [_unreadable(clip, out_dir, e) for clip in clips]
    jobs = [_ClipJob(clip, out_dir, record, digest) for clip in clips]

    def prepare(env: Any) -> None:
        for job in jobs:
            job.prepare(env)

    def observe(env: Any, observation: Any, step: int) -> None:
        for job in jobs:
            job.observe(env, step)

    try:
        check = jobs_replay.replay(
            record, image_size=OBSERVATION_SIZE, prepare=prepare, observe=observe
        )
    except BaseException:
        for job in jobs:
            job.abort()
        raise
    return [job.finish(check) for job in jobs]


def _clear_outputs(out_dir: Path, clip_id: str) -> None:
    """Remove a clip's earlier outputs, accepted or rejected, before its new verdict."""
    names = _names(clip_id)
    for stale in ("video", "report", "track", "debug", "rejected", "partial"):
        (out_dir / names[stale]).unlink(missing_ok=True)


def _unreadable(clip: Clip, out_dir: Path, error: BaseException) -> dict[str, Any]:
    _clear_outputs(out_dir, clip.id)  # an earlier acceptance no longer stands
    report = {
        "format": REPORT_FORMAT,
        "label": "rejected re-render",
        "accepted": False,
        "reason": f"the run's replay record cannot be read: {type(error).__name__}: {error}",
        "clip": clip.id,
        "spec": clip.spec(),
        "problems": [str(error)],
        "error": None,
        "episode_id": clip.episode_id,
        "run_dir": str(clip.run_dir),
        "frames": 0,
        "hashes_compared": 0,
        "hash_mismatches": 0,
    }
    _write_json(out_dir / _names(clip.id)["rejected"], report)
    return report


# ---- stills --------------------------------------------------------------------------------


def render_stills(
    run_dir: Path,
    episode_id: str,
    cameras: Sequence[CameraSpec],
    steps: Sequence[int],
    out_dir: Path,
    *,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    """PNG stills of several cameras at chosen control steps, from one verified replay
    (`<camera>.<episode>.s<step>.png`, kept only when the replay is accepted: a rejected or
    unreadable record removes earlier stills of those names), and a report
    `stills.<episode>.json` with the verification and each camera's view."""
    from PIL import Image as PILImage

    out_dir = Path(out_dir)
    names = sorted({f"{c.label}.{episode_id}.s{s:04d}.png" for c in cameras for s in steps})

    def remove_earlier() -> list[str]:
        gone = [name for name in names if (out_dir / name).is_file()]
        for name in gone:
            (out_dir / name).unlink()
        return gone

    try:
        record = jobs_replay.load_record(Path(run_dir), episode_id)
    except (OSError, ValueError, KeyError):
        remove_earlier()
        raise
    wanted = sorted({s for s in steps if 0 <= s <= len(record.executed)})
    renderers: list[Renderer] = []
    pictures: dict[tuple[str, int], Image] = {}
    views: dict[str, dict[str, Any]] = {}

    def prepare(env: Any) -> None:
        renderers.extend(Renderer.of_env(env, c, width, height) for c in cameras)

    def observe(env: Any, observation: Any, step: int) -> None:
        if step in wanted:
            for renderer in renderers:
                image, view = renderer.render()
                pictures[renderer.camera.label, step] = image
                views.setdefault(renderer.camera.label, view.as_json())

    check = jobs_replay.replay(
        record, image_size=OBSERVATION_SIZE, prepare=prepare, observe=observe
    )
    written: list[str] = []
    removed: list[str] = []
    if check.accepted:
        out_dir.mkdir(parents=True, exist_ok=True)
        for (label, step), image in sorted(pictures.items()):
            name = f"{label}.{episode_id}.s{step:04d}.png"
            PILImage.fromarray(image).save(out_dir / name)
            written.append(name)
    else:
        removed = remove_earlier()
    report = {
        "format": STILLS_FORMAT,
        "accepted": check.accepted,
        "run_dir": str(record.run_dir),
        "episode_id": episode_id,
        "steps": wanted,
        "width": width,
        "height": height,
        "cameras": {c.label: c.as_json() for c in cameras},
        "views": views,
        "stills": written,
        "removed_earlier_stills": removed,
        "problems": record.problems,
        "recorded_hashes": len(record.hashes),
        "hashes_compared": check.compared,
        "hash_mismatches": len(check.mismatches),
        "first_mismatch_step": check.mismatches[0] if check.mismatches else None,
        "verdict_matches": check.verdict_matches,
    }
    _write_json(out_dir / f"stills.{episode_id}.json", report)
    return report
