"""A recorded job episode replayed without the policy, verified, and re-rendered.

A re-render re-executes the episode's recorded actions (`truth/jobs/<episode>.npz`, as they
reached the simulator) from its identity (`truth/jobs/<episode>.json`: the base task's
scene, the stored initial state, the scene seed) in a freshly built scene. The hash of
MuJoCo's integration state after the reset and after every step is compared with the
recorded one, and the delivery check at the end with the recorded verdict. Only when every
hash and the verdict match is the re-rendered video kept; it is labelled a re-render in its
file name (`<episode>.rerender.<camera>.<size>px.mp4`) and in the report beside it, and the
original clip stays where it is. A mismatch keeps no video and says where it began. A record
that does not stand for its episode (`record_problems`: lengths, the end hash, the identity)
is refused before anything is simulated, and the report counts the hashes it did compare. An
accepted report carries the SHA-256 of the re-rendered MP4 (`video_sha256`) and of the
original clip of the same camera (`original_camera`, `original_video`), so the video a film
uses can be bound to this verification. A run records only the scene and wrist cameras: a
re-render from any other camera has no original, and its report says so with
`original_camera`, `original_video` and `original_video_sha256` all null.

Rendering does not touch the physics, so the scene may be built at another image size than
the run's (the memory `xvla-recordings-replay`: fleet episodes replay bit for bit at any
size). The camera is the scene or the wrist camera of the observation, or any other camera
of the LIBERO scene by its MuJoCo name (`frontview`, `birdview`, `sideview`, ...), rendered
beside them at the same size.

    tools/heavy-run --light uv run --locked python -m fleetlab.jobs replay --run RUN_DIR \\
        --episode e000101 --out DIR [--size 720] [--camera scene] [--frame-every 1]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import PIXEL_KEYS, Camera
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.jobs import DeliveryVerdict, JobEpisode, RequestState
from fleetlab.jobs import tables
from fleetlab.jobs.check import DeliveryState, SceneGoals, delivery_verdict

__all__ = ["REPORT_FORMAT", "ReplayMismatch", "record_problems", "rerender"]

REPORT_FORMAT = "fleetlab.job-rerender/1"
CONTROL_HZ = 20


class ReplayMismatch(RuntimeError):
    """The re-execution left the recorded episode: no re-render is kept."""


def _state_of(run_dir: Path) -> RequestState:
    record = json.loads((run_dir / tables.RUN_RECORD).read_text())
    return RequestState.model_validate(record["state"])


def _frame(env: Any, observation: Any, camera: str, size: int) -> npt.NDArray[np.uint8]:
    if camera in (Camera.scene.value, Camera.wrist.value):
        image = observation["pixels"][PIXEL_KEYS[Camera(camera)]]
    else:
        env._inner._make_current()
        image = env._sim.render(camera_name=camera, width=size, height=size)
    return np.ascontiguousarray(np.asarray(image, np.uint8)[::-1, ::-1])


def record_problems(
    episode_id: str,
    executed: npt.NDArray[np.float32],
    hashes: list[str],
    identity: EpisodeIdentity,
    recorded: JobEpisode | None,
) -> list[str]:
    """What keeps a replay record from standing for its episode; empty when it does.

    The record must be the episode's own (its identity names the episode and the recorded
    start), hold exactly one action per control step the episode ran and exactly one state
    hash more (after the reset and after every action), and end on the episode's recorded
    end state hash. Only then is every recorded hash one the replay compares.
    """
    if recorded is None:
        return [f"{episode_id} is not among the run's recorded episodes"]
    problems = []
    if identity.episode_id != episode_id or recorded.episode_id != episode_id:
        problems.append(f"the record's identity names {identity.episode_id}, not {episode_id}")
    start = recorded.start
    if (identity.start_state, identity.scene_seed, identity.noise_seed) != (
        start.init_state,
        start.scene_seed,
        start.noise_seed,
    ):
        problems.append("the record's identity is not the episode's recorded start")
    if len(executed) != recorded.control_steps:
        problems.append(
            f"{len(executed)} recorded actions for an episode of {recorded.control_steps} steps"
        )
    if len(hashes) != len(executed) + 1:
        problems.append(
            f"{len(hashes)} recorded state hashes for {len(executed)} actions; one more is due"
        )
    if not hashes or hashes[-1] != recorded.end_state_hash:
        problems.append("the record's last state hash is not the episode's end state hash")
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rerender(
    run_dir: Path,
    episode_id: str,
    out_dir: Path,
    *,
    image_size: int | None = None,
    camera: str = Camera.scene.value,
    frame_every: int = 1,
    keep_on_mismatch: bool = False,
) -> dict[str, Any]:
    """Replay one episode, verify it, and write the re-render and its report to `out_dir`.

    Returns the report. A record that does not stand for its episode is refused before
    anything is simulated; a mismatch writes the report with `accepted: false` and no video
    (unless `keep_on_mismatch`, for diagnosis, which names it a rejected re-render). An
    accepted report carries `video_sha256`, the hash of the MP4's bytes, so that whoever uses
    the video can check it is the one this verification produced.
    """
    from fleetlab.jobs.runner import GoalSuite, make_env
    from fleetlab.video import write_mp4

    if frame_every < 1 or CONTROL_HZ % frame_every:
        raise ValueError(f"frame_every divides {CONTROL_HZ}, not {frame_every}")
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    executed, hashes, identity_json = tables.load_replay_record(run_dir, episode_id)
    identity = EpisodeIdentity.model_validate_json(identity_json)
    recorded = {e.episode_id: e for e in tables.read_episodes(run_dir)}.get(episode_id)
    problems = record_problems(episode_id, executed, hashes, identity, recorded)
    if not problems:
        assert recorded is not None
        if GoalSuite().bench[recorded.start.base_task] != identity.task_index:
            problems.append("the record's scene is not the episode's base task")
    size = int(image_size or identity.image_size)
    compared = 0
    mismatches: list[int] = []
    frames: list[npt.NDArray[np.uint8]] = []
    verdict: DeliveryVerdict | None = None
    replayed_end: str | None = None
    if not problems:
        assert recorded is not None
        state = _state_of(run_dir)
        env = make_env(identity.task_index, size)
        try:
            if camera not in (Camera.scene.value, Camera.wrist.value):
                model = env._sim.model
                names = [model.camera_id2name(i) for i in range(model.ncam)]
                if camera not in names:
                    raise ValueError(f"the scene has no camera {camera!r}; it has {names}")
            observation = env.reset(identity.model_copy(update={"image_size": size}), [])
            goals = SceneGoals(env._rs, state.requested_goals)
            frames.append(_frame(env, observation, camera, size))
            replayed_end = env.state_hash().hex()
            compared += 1
            if replayed_end != hashes[0]:
                mismatches.append(0)
            for k, action in enumerate(executed):
                observation = env.step(np.asarray(action, np.float32)).observation
                replayed_end = env.state_hash().hex()
                compared += 1
                if replayed_end != hashes[k + 1]:
                    mismatches.append(k + 1)
                if (k + 1) % frame_every == 0 or k + 1 == len(executed):
                    frames.append(_frame(env, observation, camera, size))
            priv = env.privileged_state()
            scene = DeliveryState(
                goals_hold=goals.holds(),
                contacts=tuple(priv.contacts),
                velocities=priv.object_velocities,
                gripper_opening=priv.gripper_opening,
                gripper_command=float(executed[-1][6]) if len(executed) else None,
            )
            verdict = delivery_verdict(scene, state.requested_goals, goals.named)
        finally:
            env.close()
    verdict_matches = verdict is not None and recorded is not None and verdict == recorded.delivery
    accepted = not problems and compared == len(hashes) and not mismatches and verdict_matches
    stem = f"{episode_id}.rerender.{camera}.{size}px"
    if not accepted:
        stem = f"{episode_id}.rejected-rerender.{camera}.{size}px"
    out_dir.mkdir(parents=True, exist_ok=True)
    video: str | None = None
    video_sha256: str | None = None
    if frames and (accepted or keep_on_mismatch):
        path = out_dir / f"{stem}.mp4"
        write_mp4(path, frames, CONTROL_HZ // frame_every, crf=18, preset="slow", repeatable=True)
        video, video_sha256 = path.name, _sha256(path)
    # the original clip of the same camera; a camera the run never recorded has none
    recorded_camera = camera in (Camera.scene.value, Camera.wrist.value)
    original = (
        run_dir / "log" / tables.video_relative(episode_id, Camera(camera))
        if recorded_camera
        else None
    )
    if original is not None and not original.is_file():
        original = None
    report = {
        "format": REPORT_FORMAT,
        "label": "re-render" if accepted else "rejected re-render",
        "accepted": accepted,
        "problems": problems,
        "episode_id": episode_id,
        "job_id": recorded.job_id if recorded is not None else None,
        "run_dir": str(run_dir),
        "original_camera": camera if recorded_camera else None,
        "original_video": str(original) if original is not None else None,
        "original_video_sha256": _sha256(original) if original is not None else None,
        "video": video,
        "video_sha256": video_sha256,
        "camera": camera,
        "image_size": size,
        "recorded_image_size": identity.image_size,
        "frame_every": frame_every,
        "frames": len(frames),
        "steps": len(executed),
        "recorded_hashes": len(hashes),
        "hashes_compared": compared,
        "hash_mismatches": len(mismatches),
        "first_mismatch_step": mismatches[0] if mismatches else None,
        "recorded_end_state_hash": recorded.end_state_hash if recorded is not None else None,
        "end_state_hash": replayed_end,
        "verdict_matches": verdict_matches,
        "recorded_verdict": recorded.delivery.model_dump(mode="json") if recorded else None,
        "replayed_verdict": verdict.model_dump(mode="json") if verdict is not None else None,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
