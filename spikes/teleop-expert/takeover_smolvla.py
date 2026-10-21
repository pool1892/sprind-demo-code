"""Hand real failures of a pretrained policy to the scripted expert.

This runs LeRobot's normal evaluation (`lerobot-eval`) unchanged, with one addition: after
each batch of episodes, every environment in which the policy did not reach success within
its step limit is given, exactly as the policy left it, to a fresh ScriptedExpert. One line
per episode goes to ~/.cache/robots-probes/teleop/results/$TAKEOVER_NAME.jsonl.

It loads the policy onto the GPU, so it must go through the heavy lock. Use run_smolvla.sh.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import lerobot.scripts.lerobot_eval as le

from runner import failure_reason, run_expert
from scene import Scene

OUT = Path.home() / ".cache/robots-probes/teleop/results"
OUT.mkdir(parents=True, exist_ok=True)
PATH = OUT / f"{os.environ.get('TAKEOVER_NAME', 'smolvla_takeover')}.jsonl"
PATH.write_text("")
_rollout = le.rollout
_count = [0]


def _state(inner):
    """A short description of the scene the policy left behind."""
    e = inner.env
    sc = Scene(inner)
    pred, obj, target = e.parsed_problem["goal_state"][0]
    goal = sc.site_pos(target)[0] if target in e.object_sites_dict else sc.body_pos(target)
    pos = sc.body_pos(obj)
    return dict(
        object_height_m=round(float(pos[2] - sc.support_height()), 3),
        object_to_target_m=round(float(np.linalg.norm(pos[:2] - goal[:2])), 3),
        gripper_gap_mm=round(1000 * sc.finger_gap(), 1),
        touching=[bool(t) for t in sc.fingers_touching(obj)],
        gripper_height_m=round(float(sc.eef_pos()[2] - sc.support_height()), 3),
        joints=[round(float(v), 2) for v in sc.d.qpos[sc.arm_qpos_idx]],
    )


def rollout_with_takeover(env, *args, **kwargs):
    ret = _rollout(env, *args, **kwargs)
    success = ret["success"].any(dim=1).cpu().numpy()
    envs = getattr(env, "envs", None)
    if envs is None:
        print("take-over needs --eval.use_async_envs=false", file=sys.stderr)
        return ret
    for i, wrapped in enumerate(envs):
        lib = wrapped.unwrapped
        row = dict(batch=_count[0], env_index=i, task=int(lib.task_id), language=lib.task_description,
                   policy_success=bool(success[i]))
        if not success[i]:
            try:
                row["setup"] = _state(lib._env)
                res = run_expert(lib._env, max_steps=600)
                res["reason"] = failure_reason(res)
                row.update(res)
            except Exception as ex:  # noqa: BLE001
                row.update(success=False, status="fail:harness_exception", reason="harness_exception",
                           steps=0, attempts=0, error=f"{type(ex).__name__}: {ex}")
        with open(PATH, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    _count[0] += 1
    return ret


le.rollout = rollout_with_takeover

if __name__ == "__main__":
    le.main()
