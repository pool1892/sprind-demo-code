"""A camera knocked out of its pose: a sensor fault, as a cell hook on the live model.

The walking skeleton's fault (spikes/skeleton/faults.py), generalised to both
cameras and to tilt, roll and shifts. It writes no `Change`:
`camera.scene.calibration` keeps the pose the fleet believes, and
`recalibrate` brings the real pose back to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from fleetlab.contracts.faults import FaultSpec, FaultState, FaultTypeBase
from fleetlab.contracts.vocabulary import Component, FaultClass, Verb
from fleetlab.faults import register
from fleetlab.faults._common import (
    CAMERA_NAMES,
    DosePoint,
    FaultParams,
    ModelPatch,
    camera_of,
    component_truth,
    default_points,
    mj_id,
)

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Undo

# The image size the health residual refers to (LeRobot's default render size), and each
# camera's vertical field of view (degrees, LIBERO's) and working distance (metres).
REFERENCE_IMAGE_SIZE = 360
CAMERA_OPTICS = {Component.scene_camera: (45.0, 1.0), Component.wrist_camera: (75.0, 0.2)}


class PoseParams(FaultParams):
    """How the camera moved. Rotations take the level in degrees, shifts in millimetres.

    `pan` turns a camera that hangs in the world about the world's vertical
    axis (the skeleton's fault) and a camera on the robot about its own up
    axis; `tilt` and `roll` turn about the camera's own horizontal and optical
    axes; the shifts move it along its own right, up and forward directions.
    `negative` turns or shifts the other way.
    """

    motion: Literal["pan", "tilt", "roll", "shift_right", "shift_up", "shift_forward"] = "pan"
    negative: bool = False


@register("camera_pose_offset")
class CameraPoseOffset(FaultTypeBase):
    """A camera turned or shifted on its mount, as if knocked (cell hook on the live model).

    Level: degrees for a turn, millimetres for a shift. Measured on X-VLA in
    stage 1 for the scene camera's pan: 3, 6 and 9 degrees gave 89.5, 54.5 and
    27% success against 97% healthy.
    """

    kind = "camera_pose_offset"
    fault_class = FaultClass.sensor
    Params = PoseParams
    dose_levels = (0.0, 2.0, 3.0, 4.0, 6.0, 9.0)
    truth = component_truth({Verb.recalibrate, Verb.replace_})
    calibration: tuple[DosePoint, ...] = (
        *default_points((3.0, 6.0, 9.0), Component.scene_camera.value, unit=" deg scene pan"),
        *default_points((5.0, 10.0), Component.wrist_camera.value, unit=" deg wrist pan"),
        # A mount that creeps: the scene camera sags downward a little more every day
        # (stage 3, the drift with a known strong curve beside `lens_contamination`).
        *default_points(
            (2.0, 4.0, 6.0),
            Component.scene_camera.value,
            {"motion": "tilt", "negative": True},
            unit=" deg scene tilt down",
        ),
    )

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        import mujoco

        params = PoseParams.model_validate(state.spec.params)
        model = handle.model
        cam = mj_id(model, "camera", CAMERA_NAMES[camera_of(state.spec)])
        sign = -1.0 if params.negative else 1.0
        patch = ModelPatch()
        quat = np.array(model.cam_quat[cam], dtype=np.float64)
        if params.motion.startswith("shift"):
            local = {"shift_right": (1.0, 0.0, 0.0), "shift_up": (0.0, 1.0, 0.0)}.get(
                params.motion,
                (0.0, 0.0, -1.0),  # a camera looks along its own -z
            )
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quat)
            offset = rotation.reshape(3, 3) @ np.array(local) * sign * state.spec.level / 1000.0
            patch.set(model, "cam_pos", cam, model.cam_pos[cam] + offset)
            return patch.undo
        turn = np.zeros(4)
        new = np.zeros(4)
        radians = sign * np.deg2rad(state.spec.level)
        if params.motion == "pan" and int(model.cam_bodyid[cam]) == 0:
            # About the world's vertical, applied in the world frame (the skeleton's fault).
            mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), radians)
            mujoco.mju_mulQuat(new, turn, quat)
        else:
            axis = {"pan": (0.0, 1.0, 0.0), "tilt": (1.0, 0.0, 0.0), "roll": (0.0, 0.0, 1.0)}
            mujoco.mju_axisAngle2Quat(turn, np.array(axis[params.motion]), radians)
            mujoco.mju_mulQuat(new, quat, turn)
        patch.set(model, "cam_quat", cam, new)
        return patch.undo

    @staticmethod
    def health_shift(spec: FaultSpec) -> dict[str, float]:
        """What the fleet's calibration check would see: the reprojection residual in pixels.

        For a turn, the focal length of a 360-pixel image at the camera's field
        of view (45 degrees for the scene camera, 75 for the wrist camera)
        times the tangent of the angle; for a shift, the same focal length
        times the shift over the camera's working distance (1 m for the scene
        camera, 0.2 m for the wrist camera). Noise and benign variation are
        the world generator's to add.
        """
        params = PoseParams.model_validate(spec.params)
        fovy, distance = CAMERA_OPTICS[camera_of(spec)]
        focal = (REFERENCE_IMAGE_SIZE / 2) / np.tan(np.deg2rad(fovy) / 2)
        if params.motion.startswith("shift"):
            residual = focal * spec.level / 1000.0 / distance
        else:
            residual = focal * np.tan(np.deg2rad(spec.level))
        return {"calibration_residual": float(residual)}
