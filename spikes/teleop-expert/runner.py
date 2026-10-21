"""Run the scripted expert in a LIBERO environment and report what happened."""
from __future__ import annotations

import numpy as np

from common import DUMMY, frame
from expert import ScriptedExpert


def run_expert(env, max_steps=600, video=None, cameras=("agentview", "frontview"), hw=256, every=2,
               verbose=False, frames=None, expert_kwargs=None):
    """Let a fresh expert act until it stops. Returns a result dict.

    `success` is the strict outcome: LIBERO's own success check is true after
    the gripper has let go and the scene has had 15 more steps to settle.
    `libero_success_step` is the first step at which LIBERO's check was true at
    all, which is where LeRobot's evaluation loop would have ended the episode.
    """
    expert = ScriptedExpert(env, max_steps=max_steps, verbose=verbose, **(expert_kwargs or {}))
    frames = frames if frames is not None else ([] if video else None)

    def grab():
        if frames is not None:
            frames.append(np.concatenate([frame(env, c, hw) for c in cameras], axis=1))

    grab()
    steps = 0
    error = None
    while not expert.done:
        try:
            env.step(expert.act())
        except Exception as ex:  # noqa: BLE001  (simulation instability and the like)
            error = f"{type(ex).__name__}: {ex}"
            break
        steps += 1
        if steps % every == 0:
            grab()
    if error is None:
        for i in range(15):
            env.step(DUMMY)
            if i % every == 0:
                grab()
    sc = expert.sc
    success = bool(env.check_success()) and not any(sc.fingers_touching(expert.obj)) and error is None
    status = expert.status if error is None else "fail:exception"
    if success and status != "success":
        status = "success_late"       # satisfied although the script had given up
    if not success and status == "success":
        status = "fail:unstable_after_release"
    result = dict(
        success=success,
        status=status,
        steps=steps,
        attempts=expert.attempt,
        libero_success_step=expert.first_success_step,
        events=[f"{t}: {msg}" for t, msg in expert.events],
        error=error,
    )
    if video and frames:
        import imageio

        imageio.mimsave(video, frames, fps=20 // every, macro_block_size=1)
        result["video"] = str(video)
    return result


def failure_reason(result):
    """One short label per failed run, from the status and the last event."""
    if result["success"]:
        return ""
    ev = " | ".join(result["events"])
    for key, label in [
        ("no top-down grasp", "no_grasp_found"),
        ("object_off_table", "object_off_table"),
        ("exception", "sim_exception"),
    ]:
        if key in ev or key in result["status"]:
            return label
    counts = {
        "grasp_missed": ev.count("grasp missed"),
        "slipped_in_lift": ev.count("slipped out while lifting"),
        "dropped_in_transit": ev.count("dropped in transit") + ev.count("dropped while"),
        "unreachable": ev.count("could not reach"),
        "placed_wrong": ev.count("not satisfied"),
    }
    top = max(counts, key=counts.get)
    return top if counts[top] else result["status"].replace("fail:", "")
