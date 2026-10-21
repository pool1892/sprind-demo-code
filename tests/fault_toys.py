"""A toy work cell for the fault tests: a small MuJoCo model with LIBERO's names.

`make check` builds no LIBERO scene, but a cell hook needs a live model to
change and undo. This one has what the hooks reach for: the scene and wrist
cameras, two lights and the headlight, textured walls, floor and table, a
robot base with a hand on seven joints with torque motors, two fingers with
position actuators and pads, and two free objects. `toy_handle()` wraps it in a
`CellHandle` that records texture uploads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt

import fleetlab  # noqa: F401  (torch before mujoco)
from fleetlab.contracts.env import CellHandle, Observation
from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.policy import ControlMode
from fleetlab.faults._common import DosePoint, target_of

TOY_XML = """
<mujoco model="toy_cell">
  <asset>
    <texture name="tex-wall" type="2d" builtin="checker" width="16" height="16"
        rgb1="0.8 0.7 0.6" rgb2="0.6 0.5 0.4"/>
    <texture name="texplane" type="2d" builtin="checker" width="8" height="8" rgb1="0.3 0.3 0.3"
        rgb2="0.2 0.2 0.2"/>
    <texture name="tex-table" type="2d" builtin="gradient" width="8" height="8"
        rgb1="0.9 0.8 0.7" rgb2="0.5 0.4 0.3"/>
    <material name="walls_mat" texture="tex-wall"/>
    <material name="floorplane" texture="texplane"/>
    <material name="table_texture" texture="tex-table"/>
  </asset>
  <visual>
    <headlight ambient="0.1 0.1 0.1" diffuse="0.4 0.4 0.4" specular="0.5 0.5 0.5"/>
  </visual>
  <worldbody>
    <light name="light1" pos="1 1 4" dir="0 -0.15 -1" diffuse="0.8 0.8 0.8"
        specular="0.3 0.3 0.3"/>
    <light name="light2" pos="-3 -3 4" dir="0 -0.15 -1" diffuse="0.8 0.8 0.8"
        specular="0.3 0.3 0.3"/>
    <camera name="agentview" pos="0.66 0 1.61" quat="0.638 0.305 0.305 0.638"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floorplane"/>
    <geom name="wall" type="box" pos="-2 0 1" size="0.01 3 1" material="walls_mat" contype="0"
        conaffinity="0"/>
    <body name="table" pos="0 0 0.875">
      <geom name="table_visual" type="box" size="0.5 0.6 0.025" material="table_texture"/>
    </body>
    <body name="robot0_base" pos="-0.66 0 0.912">
      <geom name="robot0_link0" type="cylinder" size="0.06 0.05"/>
      <body name="robot0_link1">
        <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
        <body name="robot0_link2">
          <joint name="robot0_joint2" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
          <body name="robot0_link3">
            <joint name="robot0_joint3" type="hinge" axis="1 0 0"/>
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
            <body name="robot0_link4">
              <joint name="robot0_joint4" type="hinge" axis="0 1 0"/>
              <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
              <body name="robot0_link5">
                <joint name="robot0_joint5" type="hinge" axis="0 0 1"/>
                <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                <body name="robot0_link6">
                  <joint name="robot0_joint6" type="hinge" axis="0 1 0"/>
                  <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                  <body name="robot0_link7">
                    <joint name="robot0_joint7" type="hinge" axis="0 0 1"/>
                    <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                    <body name="robot0_right_hand" pos="0.4 0 0.3">
                      <camera name="robot0_eye_in_hand" pos="0.05 0 0"
                          quat="0 0.707107 0.707107 0" fovy="75"/>
                      <body name="gripper0_right_leftfinger" pos="0 0.02 -0.05">
                        <joint name="gripper0_right_finger_joint1" type="slide"
                            axis="0 1 0" range="0 0.04"/>
                        <geom name="gripper0_right_finger1_pad_collision" type="box"
                            size="0.008 0.004 0.008"
                            friction="2 0.05 0.0001"/>
                      </body>
                      <body name="gripper0_right_rightfinger" pos="0 -0.02 -0.05">
                        <joint name="gripper0_right_finger_joint2" type="slide"
                            axis="0 1 0" range="-0.04 0"/>
                        <geom name="gripper0_right_finger2_pad_collision" type="box"
                            size="0.008 0.004 0.008"
                            friction="2 0.05 0.0001"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="bowl_main" pos="0 -0.1 0.95">
      <freejoint name="bowl_joint"/>
      <geom name="bowl_g0" type="cylinder" size="0.05 0.02" mass="0.0056"
          friction="0.95 0.3 0.1"/>
      <geom name="bowl_visual" type="cylinder" size="0.05 0.02" contype="0" conaffinity="0"
          mass="0"/>
    </body>
    <body name="box_main" pos="0.1 0.1 0.95">
      <freejoint name="box_joint"/>
      <geom name="box_g0" type="box" size="0.03 0.04 0.05" mass="0.0096" friction="0.95 0.3 0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="robot0_torq_j1" joint="robot0_joint1" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="robot0_torq_j2" joint="robot0_joint2" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="robot0_torq_j3" joint="robot0_joint3" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="robot0_torq_j4" joint="robot0_joint4" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="robot0_torq_j5" joint="robot0_joint5" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="robot0_torq_j6" joint="robot0_joint6" ctrlrange="-12 12" ctrllimited="true"/>
    <motor name="robot0_torq_j7" joint="robot0_joint7" ctrlrange="-12 12" ctrllimited="true"/>
    <position name="gripper0_right_gripper_finger_joint1" joint="gripper0_right_finger_joint1"
        kp="1000" ctrlrange="0 0.04" forcerange="-20 20"/>
    <position name="gripper0_right_gripper_finger_joint2" joint="gripper0_right_finger_joint2"
        kp="1000" ctrlrange="-0.04 0" forcerange="-20 20"/>
  </actuator>
</mujoco>
"""


@dataclass
class Uploads:
    """Records the texture uploads a hook asks for."""

    calls: list[int] = field(default_factory=list)

    def __call__(self, tex_id: int) -> None:
        self.calls.append(tex_id)


def toy_handle() -> tuple[CellHandle, Uploads]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(TOY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(
        robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"))],
        objects_dict={
            "bowl_1": SimpleNamespace(root_body="bowl_main"),
            "box_1": SimpleNamespace(root_body="box_main"),
        },
    )
    uploads = Uploads()
    return CellHandle(model=model, data=data, env=env, upload_texture=uploads), uploads


def model_arrays(model: Any) -> dict[str, npt.NDArray[Any]]:
    """Copies of every array of the model and of the headlight: what an undo must restore."""
    out: dict[str, npt.NDArray[Any]] = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray):
            out[name] = value.copy()
    for name in ("ambient", "diffuse", "specular"):
        out[f"vis.headlight.{name}"] = np.array(getattr(model.vis.headlight, name), copy=True)
    return out


def changed_arrays(before: dict[str, npt.NDArray[Any]], model: Any) -> list[str]:
    after = model_arrays(model)
    return sorted(k for k in before if before[k].tobytes() != after[k].tobytes())


POLICY = PolicySpec(
    name="xvla-libero",
    revision="12e8783e",
    precision="fp16_res32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=30,
    sampling_steps=10,
    image_views=4,
)
SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="p1", mujoco_version="3.3.7", miopen_db_hash="0" * 16
)


def scope_for(kind_scopes: frozenset[Any]) -> str:
    """The narrowest scope a truth mapping allows, as the toy fleet's id: r01, s1 or fleet."""
    from fleetlab.contracts.ids import ScopeKind

    if ScopeKind.robot in kind_scopes:
        return "r01"
    if ScopeKind.site in kind_scopes:
        return "s1"
    return "fleet"


def spec_for(kind: str, point: DosePoint, fault_id: str = "f01") -> FaultSpec:
    """The fault spec of a calibration point, at the narrowest scope its type allows."""
    from fleetlab.faults import get

    truth = get(kind).truth
    return FaultSpec(
        fault_id=fault_id,
        kind=kind,
        params=point.params,
        level=point.level,
        target=target_of(scope_for(truth.scopes), point.target, truth.target_kind),
        schedule=Schedule(onset_day=0),
        nominal=point.nominal,
    )


def identity(
    faults: list[FaultSpec] | None = None,
    index: int = 0,
    control_mode: ControlMode = ControlMode.absolute,
) -> EpisodeIdentity:
    scene, noise = seeds_for(7, "r01", 1, index)
    policy = POLICY.model_copy(update={"control_mode": control_mode})
    return EpisodeIdentity(
        world_id="w90",
        episode_id=f"e{index:06d}",
        robot_id="r01",
        site_id="s1",
        day=1,
        episode_index=index,
        suite="libero_spatial",
        task_index=0,
        start_state=0,
        scene_seed=scene,
        noise_seed=noise,
        policy=policy,
        batch_size=4,
        row=None,
        image_size=128,
        max_steps=280,
        faults=faults or [],
        software=SOFTWARE,
    )


def observation(step: int = 0, size: int = 64, seed: int = 0) -> Observation:
    """A synthetic observation: textured images and a robot state."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    pattern = ((xx // 4 + yy // 4) % 2 * 120 + 60).astype(np.uint8)
    scene = np.stack([pattern, np.roll(pattern, 3, axis=1), pattern // 2], axis=-1)
    wrist = rng.integers(40, 200, size=(size, size, 3), dtype=np.uint8)
    angle = 0.3
    mat = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    ) @ np.diag([1.0, -1.0, -1.0])
    return {
        "pixels": {"image": scene, "image2": wrist},
        "robot_state": {
            "eef": {
                "pos": np.array([-0.2, 0.01 * step, 1.17]),
                "quat": np.array([0.0, 1.0, 0.0, 0.0]),
                "mat": mat,
            },
            "gripper": {"qpos": np.array([0.02, -0.02]), "qvel": np.zeros(2)},
            "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
        },
        "task_description": "pick up the bowl",
        "step": step,
    }


def toy_target(scope: str, thing: str, kind: Any) -> Target:
    return target_of(scope, thing, kind)
