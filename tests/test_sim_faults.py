"""The fault types on a real LIBERO scene, no model loaded (`make check-sim`).

Each cell hook changes the rendered images or the simulator as intended, and
its undo restores the cell exactly: the fingerprint of a healthy reset (the
simulator state, the fixture placement, both images) after a faulted episode,
and after a healthy episode run in between, equals that of a healthy reset in
a fresh process. The observation hooks are checked on rendered images.

The reset follows the engine's adapter (`LiberoSimEnv.reset` on
stage2/engine): every fault's `new_episode`, then the cell hooks through the
handle, then LIBERO's seeded reset, start state and settle steps.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fault_toys import identity, spec_for
from sim_helpers import SEED, fingerprint, in_fresh_process

import fleetlab.faults as faults
from fleetlab.contracts.env import CellHandle, Observation
from fleetlab.contracts.faults import FaultSpec, FaultState, Schedule, Target
from fleetlab.contracts.vocabulary import Component
from fleetlab.faults._common import DosePoint
from fleetlab.sim.libero_env import LiberoEnv

pytestmark = pytest.mark.sim

SUITE, TASK = "libero_spatial", 4  # a cabinet with a drawer, a stove, two bowls, a plate
IMAGE_SIZE = 128
faults.load_all()


def handle(env: LiberoEnv) -> CellHandle:
    """The handle the engine's adapter gives cell hooks."""
    import mujoco

    sim = env._env.sim

    def upload(texture_id: int) -> None:
        env._make_current()
        mujoco.mjr_uploadTexture(sim.model._model, sim._render_context_offscreen.con, texture_id)

    return CellHandle(
        model=sim.model._model, data=sim.data._data, env=env._env.env, upload_texture=upload
    )


class Episode:
    """One episode with faults, as the engine runs it, without a policy."""

    def __init__(self, env: LiberoEnv, specs: list[FaultSpec]) -> None:
        self.env = env
        self.identity = identity(specs)
        self.types = [faults.get(s.kind) for s in specs]
        self.states: list[FaultState] = [
            t.new_episode(s, self.identity) for t, s in zip(self.types, specs, strict=True)
        ]
        self.undos = []
        for t, st in zip(self.types, self.states, strict=True):
            undo = t.cell_hook(handle(env), st)
            if undo is not None:
                self.undos.append(undo)
        self.raw = env.reset(init_state=0, seed=SEED)
        self.observation = self.hooked(self.raw)

    def hooked(self, observation: Any) -> Observation:
        for t, st in zip(self.types, self.states, strict=True):
            observation = t.observation_hook(observation, st)
        out: Observation = observation
        return out

    def step(self, command: Any) -> Observation:
        for t, st in zip(self.types, self.states, strict=True):
            command = t.action_hook(command, st)
        self.raw, _, _ = self.env.step(np.asarray(command, dtype=np.float32))
        self.observation = self.hooked(self.raw)
        return self.observation

    def hold(self, steps: int, gripper: float = -1.0) -> None:
        """Hold the end effector where it is (absolute control), gripper open or closed."""
        from fleetlab.faults._common import matrix_to_rotvec

        eef = self.observation["robot_state"]["eef"]
        command = np.concatenate([eef["pos"], matrix_to_rotvec(eef["mat"]), [gripper]])
        for _ in range(steps):
            self.step(command)

    def end(self) -> None:
        for undo in reversed(self.undos):
            undo()
        self.undos = []


def healthy_fingerprint(steps: int = 0) -> dict[str, str]:
    env = LiberoEnv(SUITE, TASK, image_size=IMAGE_SIZE, control_mode="absolute", hard_reset=False)
    episode = Episode(env, [])
    episode.hold(steps)
    episode = Episode(env, [])
    out = fingerprint(env, episode.raw)
    env.close()
    return out


@pytest.fixture(scope="module")
def fresh() -> dict[str, str]:
    """A healthy reset's fingerprint in a process that never saw a fault."""
    result: dict[str, str] = in_fresh_process(healthy_fingerprint)
    return result


@pytest.fixture(scope="module")
def env() -> Iterator[LiberoEnv]:
    built = LiberoEnv(SUITE, TASK, image_size=IMAGE_SIZE, control_mode="absolute", hard_reset=False)
    yield built
    built.close()


def point(kind: str, index: int = -1) -> FaultSpec:
    calibration: tuple[DosePoint, ...] = getattr(faults.get(kind), "calibration", ())
    return spec_for(kind, calibration[index])


def mean(observation: Observation, key: str) -> float:
    return float(observation["pixels"][key].astype(np.float64).mean())


def changed_share(a: Observation, b: Observation, key: str) -> float:
    diff = np.abs(a["pixels"][key].astype(int) - b["pixels"][key].astype(int)).sum(axis=-1)
    return float((diff > 0).mean())


def contact_force_on(env: LiberoEnv, body_name: str) -> float:
    """The vertical contact force on one body, in newtons."""
    import mujoco

    model, data = env._env.sim.model._model, env._env.sim.data._data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
        if body not in bodies:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        world = contact.frame.reshape(3, 3).T @ force[:3]
        total += world[2] if bodies[1] == body else -world[2]
    return total


# Every cell-hook type at its strongest calibration point, and what it must change.
CELL_CASES = {
    "camera_pose_offset scene": (point("camera_pose_offset", 2), {"image"}),
    "camera_pose_offset wrist": (point("camera_pose_offset", 4), {"image2"}),
    "gripper_wear": (point("gripper_wear"), set()),
    "object_batch_variant heavier": (point("object_batch_variant", 2), {"state"}),
    "object_batch_variant slipperier": (point("object_batch_variant", 3), set()),
    "furniture_shift": (point("furniture_shift"), {"image", "image2"}),
    "illumination_change": (point("illumination_change", 1), {"image", "image2"}),
    "backdrop_change walls": (point("backdrop_change", 1), {"image"}),
    "backdrop_change table": (point("backdrop_change", 2), {"image", "image2"}),
    # stage 3
    "camera_pose_offset tilt": (point("camera_pose_offset", -1), {"image"}),
    "gripper_force_loss": (point("gripper_force_loss"), set()),
    "arm_torque_loss": (point("arm_torque_loss"), {"state"}),
    "table_height_change lower": (point("table_height_change", 2), {"image", "image2"}),
    "table_height_change higher": (point("table_height_change", 3), {"image", "image2"}),
    "light_color_shift warm": (point("light_color_shift", 1), {"image", "image2"}),
    "light_color_shift cool": (point("light_color_shift", 2), {"image", "image2"}),
}


@pytest.mark.parametrize("case", list(CELL_CASES))
def test_a_cell_hook_changes_the_cell_and_its_undo_restores_it_exactly(
    case: str, env: LiberoEnv, fresh: dict[str, str]
) -> None:
    spec, must_change = CELL_CASES[case]
    healthy = Episode(env, [])
    assert fingerprint(env, healthy.raw) == fresh

    faulted = Episode(env, [spec])
    print_ = fingerprint(env, faulted.raw)
    differs = {k for k in ("state", "image", "image2") if print_[k] != fresh[k]}
    assert must_change <= differs, f"{case}: {differs}"
    faulted.hold(15)
    faulted.end()

    assert fingerprint(env, Episode(env, []).raw) == fresh
    between = Episode(env, [])
    between.hold(15, gripper=1.0)
    assert fingerprint(env, Episode(env, []).raw) == fresh


def test_a_turned_scene_camera_moves_only_the_scene_image(env: LiberoEnv) -> None:
    healthy = Episode(env, []).observation
    turned = Episode(env, [point("camera_pose_offset", 1)])  # 6 degrees
    assert changed_share(healthy, turned.observation, "image") > 0.3
    assert changed_share(healthy, turned.observation, "image2") == 0.0
    turned.end()


def test_a_worn_gripper_closes_short(env: LiberoEnv) -> None:
    import mujoco

    model, data = env._env.sim.model._model, env._env.sim.data._data
    pads = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")
    ]

    def closed_gap(specs: list[FaultSpec]) -> float:
        episode = Episode(env, specs)
        episode.hold(30, gripper=1.0)
        gap = float(np.linalg.norm(data.geom_xpos[pads[0]] - data.geom_xpos[pads[1]]))
        episode.end()
        return gap - 2 * float(model.geom_size[pads[0]][1])

    healthy = closed_gap([])
    worn = closed_gap([point("gripper_wear", 1)])  # level 0.5: a 1.5 mm gap
    assert healthy < 0.0005
    assert worn - healthy == pytest.approx(0.0015, abs=0.0002)


def test_heavier_objects_press_harder_and_slipperier_ones_slide(env: LiberoEnv) -> None:
    import mujoco

    model, data = env._env.sim.model._model, env._env.sim.data._data
    healthy = Episode(env, [])
    healthy.hold(10)
    weight = contact_force_on(env, "plate_1_main")
    heavy = Episode(env, [point("object_batch_variant", 1)])  # 4 doublings: 16 times
    heavy.hold(10)
    assert contact_force_on(env, "plate_1_main") == pytest.approx(16 * weight, rel=0.05)
    heavy.end()

    plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_1_main")
    own = max(
        float(model.geom_friction[g][0])
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == plate and (model.geom_contype[g] or model.geom_conaffinity[g])
    )
    slippery = Episode(env, [point("object_batch_variant", 3)])  # 2 halvings
    frictions = {
        round(float(data.contact[i].friction[0]), 6)
        for i in range(data.ncon)
        if plate
        in (model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2])
    }
    slippery.end()
    assert frictions == {round(own / 4, 6)}


def test_heavier_objects_get_the_solver_constants_mj_setconst_computes(env: LiberoEnv) -> None:
    import copy

    import mujoco

    model = env._env.sim.model._model
    reference = copy.deepcopy(model)
    episode = Episode(env, [point("object_batch_variant", 0)])  # 2 doublings
    roots = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o.root_body)
        for o in env._env.env.objects_dict.values()
    ]
    for body in roots:
        reference.body_mass[body] *= 4
        reference.body_inertia[body] *= 4
    mujoco.mj_setConst(reference, mujoco.MjData(reference))
    dofs = [d for d in range(model.nv) if model.dof_bodyid[d] in roots]
    for name, index in (
        ("body_mass", roots),
        ("body_inertia", roots),
        ("body_invweight0", roots),
        ("body_subtreemass", [*roots, 0]),
        ("dof_invweight0", dofs),
        ("dof_M0", dofs),
    ):
        ours, theirs = getattr(model, name)[index], getattr(reference, name)[index]
        assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-12), name
    episode.end()


def test_a_shifted_table_looks_moved_and_the_robot_reports_its_own_frame(env: LiberoEnv) -> None:
    healthy = Episode(env, [])
    eef = healthy.observation["robot_state"]["eef"]["pos"].copy()
    shifted = Episode(env, [point("furniture_shift")])  # 4 cm to the robot's left
    reported = shifted.observation["robot_state"]["eef"]["pos"]
    true = shifted.raw["robot_state"]["eef"]["pos"]
    assert np.allclose(reported, eef, atol=1e-6)
    assert np.allclose(true, eef - [0.0, 0.04, 0.0], atol=1e-6)
    # An absolute target in the robot's frame is reached in that frame.
    target = eef + np.array([0.03, 0.02, 0.0])
    rot = healthy.observation["robot_state"]["eef"]["mat"]
    from fleetlab.faults._common import matrix_to_rotvec

    command = np.concatenate([target, matrix_to_rotvec(rot), [-1.0]])
    for _ in range(40):
        shifted.step(command)
    assert np.allclose(shifted.observation["robot_state"]["eef"]["pos"], target, atol=0.003)
    shifted.end()


def test_dimmer_lights_darken_both_images(env: LiberoEnv) -> None:
    healthy = Episode(env, []).observation
    dim = Episode(env, [point("illumination_change", 1)])  # 2 stops
    for key in ("image", "image2"):
        assert mean(dim.observation, key) < 0.75 * mean(healthy, key)
    dim.end()


def test_a_repainted_backdrop_changes_part_of_the_scene_image(env: LiberoEnv) -> None:
    healthy = Episode(env, []).observation
    painted = Episode(env, [point("backdrop_change", 1)])  # walls and floor, full blend
    share = changed_share(healthy, painted.observation, "image")
    assert 0.1 < share < 0.5
    painted.end()


def degradation(mode: str, level: float, **params: Any) -> FaultSpec:
    return FaultSpec(
        fault_id="f02",
        kind="camera_degradation",
        params={"mode": mode, **params},
        level=level,
        target=Target(scope="r01", component=Component.wrist_camera),
        schedule=Schedule(onset_day=0),
    )


def test_observation_hooks_on_rendered_images(env: LiberoEnv) -> None:
    raw = Episode(env, []).raw
    original = raw["pixels"]["image2"].astype(np.float64)

    def apply(spec: FaultSpec) -> Any:
        episode = Episode(env, [spec])
        assert np.array_equal(episode.raw["pixels"]["image2"], raw["pixels"]["image2"])
        assert np.array_equal(episode.observation["pixels"]["image"], raw["pixels"]["image"])
        return episode.observation["pixels"]["image2"].astype(np.float64)

    def sharpness(image: Any) -> float:
        g = image.mean(axis=-1)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return float(lap.var())

    assert sharpness(apply(degradation("blur", 4.0))) < 0.5 * sharpness(original)
    noise = apply(degradation("noise", 10.0)) - original
    assert 8.0 < noise.std() < 10.5  # clipped at 0 and 255, so a little under 10
    darker = apply(degradation("exposure", 1.0, darker=True))
    assert darker.mean() == pytest.approx(original.mean() / 2, rel=0.02)


# ---------------------------------------------------------------------------------------- stage 3


def finger_opening(env: LiberoEnv) -> float:
    qpos = env._env.robots[0].sim.data.qpos
    return float(sum(abs(float(qpos[i])) for i in (7, 8)))


def test_a_sagging_camera_looks_further_down(env: LiberoEnv) -> None:
    import mujoco

    data, model = env._env.sim.data._data, env._env.sim.model._model
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    Episode(env, [])
    healthy = -float(
        data.cam_xmat[cam].reshape(3, 3)[2, 2]
    )  # the view's z: the camera looks along -z
    sagged = Episode(env, [point("camera_pose_offset", -1)])  # 6 degrees down
    down = -float(data.cam_xmat[cam].reshape(3, 3)[2, 2])
    sagged.end()
    assert down < healthy  # more of the view points at the floor
    assert np.degrees(np.arcsin(-down)) - np.degrees(np.arcsin(-healthy)) == pytest.approx(6, abs=1)


def test_a_weak_arm_sags_under_its_own_weight(env: LiberoEnv) -> None:
    def height(specs: list[FaultSpec]) -> float:
        episode = Episode(env, specs)
        start = episode.observation["robot_state"]["eef"]["pos"].copy()
        episode.hold(1)
        command = episode.observation["robot_state"]["eef"]
        from fleetlab.faults._common import matrix_to_rotvec

        target = np.concatenate([start, matrix_to_rotvec(command["mat"]), [-1.0]])
        for _ in range(40):
            episode.step(target)
        z = float(episode.raw["robot_state"]["eef"]["pos"][2] - start[2])
        episode.end()
        return z

    healthy = height([])
    sags = [height([point("arm_torque_loss", i)]) for i in (0, 1, 2)]  # 0.1, 0.2, 0.3 lost
    assert abs(healthy) < 0.002
    assert sags[0] > sags[1] > sags[2]
    assert healthy - sags[2] > 0.005  # the weakest arm holds at least 5 mm low


def test_a_weak_gripper_closes_slower_and_stalls_short(env: LiberoEnv) -> None:
    def openings(specs: list[FaultSpec]) -> tuple[float, float]:
        episode = Episode(env, specs)
        episode.hold(5, gripper=1.0)
        early = finger_opening(env)
        episode.hold(40, gripper=1.0)
        late = finger_opening(env)
        episode.end()
        return early, late

    healthy = openings([])
    weak = openings([point("gripper_force_loss", 1)])  # 0.5 lost
    assert weak[0] > healthy[0] + 0.003  # slower to close (the fingers' own friction)
    assert healthy[1] < 0.0015 and weak[1] > healthy[1] + 0.0015  # and stalls short of closed


def test_a_sticky_gripper_opens_late_in_the_simulator(env: LiberoEnv) -> None:
    def open_after(specs: list[FaultSpec]) -> int:
        episode = Episode(env, specs)
        episode.hold(30, gripper=1.0)
        closed = finger_opening(env)
        for step in range(60):
            episode.hold(1, gripper=-1.0)
            if finger_opening(env) > closed + 0.005:
                episode.end()
                return step
        episode.end()
        return 60

    healthy = open_after([])
    late = open_after([point("gripper_sticky_release", 1)])  # 10 steps
    assert late - healthy == pytest.approx(10, abs=1)


def test_a_lower_table_is_reached_in_the_robots_own_frame(env: LiberoEnv) -> None:
    healthy = Episode(env, [])
    eef = healthy.observation["robot_state"]["eef"]["pos"].copy()
    lower = Episode(env, [point("table_height_change", 2)])  # 4 cm lower
    assert np.allclose(lower.observation["robot_state"]["eef"]["pos"], eef, atol=1e-6)
    assert np.allclose(
        lower.raw["robot_state"]["eef"]["pos"], eef + np.array([0.0, 0.0, 0.04]), atol=1e-6
    )
    lower.end()


def test_warm_lamps_change_the_colour_not_the_brightness(env: LiberoEnv) -> None:
    healthy = Episode(env, []).observation["pixels"]["image"].astype(np.float64)
    warm = Episode(env, [point("light_color_shift", 1)])
    image = warm.observation["pixels"]["image"].astype(np.float64)
    warm.end()
    ratio = image[..., 0].mean() / image[..., 2].mean()
    assert ratio > 1.3 * healthy[..., 0].mean() / healthy[..., 2].mean()
    luma = np.array([0.299, 0.587, 0.114])
    assert (image @ luma).mean() == pytest.approx((healthy @ luma).mean(), rel=0.15)


def test_the_new_observation_hooks_on_rendered_images(env: LiberoEnv) -> None:
    raw = Episode(env, []).raw
    dirty = Episode(env, [point("lens_contamination", -1)])  # wrist, full density
    assert np.array_equal(dirty.raw["pixels"]["image2"], raw["pixels"]["image2"])
    softer = dirty.observation["pixels"]["image2"].astype(np.float64)
    original = raw["pixels"]["image2"].astype(np.float64)

    def sharpness(image: Any) -> float:
        g = image.mean(axis=-1)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return float(lap.var())

    assert sharpness(softer) < 0.5 * sharpness(original)
    swapped = Episode(env, [point("image_pipeline_release", 0)])
    assert np.array_equal(swapped.observation["pixels"]["image"], raw["pixels"]["image"][..., ::-1])
    late = Episode(env, [point("camera_latency", 0)])  # 6 steps, scene camera
    first = late.observation["pixels"]["image"].copy()
    late.hold(2)
    assert np.array_equal(late.observation["pixels"]["image"], first)  # still the first frame
    late.hold(1)
    assert not np.array_equal(late.raw["pixels"]["image"], late.observation["pixels"]["image"])


def test_a_placement_motion_makes_a_failure_eligible_for_the_detector(env: LiberoEnv) -> None:
    from fleetlab.contracts.env import SuccessVerdict

    spec = point("success_false_positive")  # level 1
    episode = Episode(env, [spec])
    episode.hold(30, gripper=1.0)
    episode.hold(30, gripper=-1.0)
    fault_type, state = episode.types[0], episode.states[0]
    flipped = fault_type.data_hook(SuccessVerdict(False, True, "step limit"), state)
    episode.end()
    assert flipped.success
    idle = Episode(env, [spec])
    idle.hold(30, gripper=-1.0)
    kept = idle.types[0].data_hook(SuccessVerdict(False, True, "step limit"), idle.states[0])
    idle.end()
    assert not kept.success
