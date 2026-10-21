"""One blind diagnosis of the three-robot fleet, through the sandboxed agent runtime.

    python diagnose.py WORKSPACE OUT.json [--model opus] [--check-only]

The agent gets a description of the system, the question and two tools. It runs
through `run_agent` (agent_runtime.py, copied from spikes/agent-runtime) in the
`bwrap` sandbox: the command-line tool in one sandbox with the login and the
network but no data, the tool server in a second sandbox inside it with the
workspace and no network. The answer is checked against ANSWER_SCHEMA.

Nothing in this file knows which robot carries which fault. --check-only prints
the prompt and searches it for the truth's words without calling a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path.home() / ".cache" / "robots-probes" / "skeleton"
os.environ.setdefault("ROBOTS_AGENT_STATE", str(BASE / "agent-state"))

from agent_runtime import ToolServer, run_agent  # noqa: E402

COMPONENTS = ["scene_camera", "wrist_camera", "arm", "gripper", "controller", "policy_computer", "workcell"]
VERBS = ["recalibrate", "clean", "service", "replace", "revert_setting", "restart", "none"]

SYSTEM = (
    "You are a diagnostic analyst for a small fleet of robot arms. You cannot see or touch the robots; "
    "you work only through the two tools you are given, which read the fleet's logs. Every number you "
    "report must come from a tool result, not from your own estimate. Be economical: aggregate in SQL "
    "and look at frames only where they can settle a question."
)

PROMPT = """\
# The fleet

One site with three robot cells, `robot_1`, `robot_2` and `robot_3`, built to the same design and
configured identically. Each cell runs the same pretrained vision-language-action policy on the same
ten table-top pick-and-place tasks. Every task was attempted from the same four stored object layouts
(`start_state_id` 0 to 3), so each robot has 40 episodes. An episode ends when the task is done, or
after 280 control steps (14 seconds), which counts as a failure.

What is normal: in acceptance tests healthy cells succeeded in 192 of 200 episodes on these tasks,
and no single task fell below 16 of 20. The policy samples its actions, so two episodes with the same
task and layout differ in detail even on healthy robots. Furniture (cabinet, stove) stands up to about
2 cm differently from cell to cell and from episode to episode.

# What a cell consists of

Components, one set per robot:

- `scene_camera`: a fixed camera on a mount beside the table that looks at the whole work area.
- `wrist_camera`: a camera on the gripper.
- `arm`: seven joints with motors and position sensors.
- `gripper`: two parallel fingers.
- `controller`: the real-time computer that receives one command per control step (20 per second)
  and drives the arm and the gripper. A command is a target pose of the gripper in the world frame
  plus open or close.
- `policy_computer`: runs the policy. Once every 30 control steps it takes one image from each camera
  and the measured gripper pose, and produces the next 30 commands.
- `workcell`: table, objects, furniture, lighting.

Settings, the same nominal value on every robot:

| Setting | Nominal value |
|---|---|
| `scene_camera.mount_pose` | as calibrated at installation |
| `scene_camera.resolution` | 360 x 360 |
| `scene_camera.exposure` | automatic |
| `wrist_camera.resolution` | 360 x 360 |
| `wrist_camera.exposure` | automatic |
| `arm.joint_speed_limit` | factory default |
| `gripper.max_force` | 20 N |
| `controller.rate_hz` | 20 |
| `controller.command_frame` | world |
| `controller.command_buffer_steps` | 0 (a command is executed in the step it arrives) |
| `controller.stiffness` | factory default |
| `policy.checkpoint` | lerobot/xvla-libero |
| `policy.actions_per_call` | 30 |
| `policy.denoising_steps` | 10 |

# The logs

Per episode: task, object layout, length, outcome. Per control step: measured joint and gripper
state, measured gripper pose, and the command the policy computer issued at that step. Video from
both cameras at 5 frames per second, as the robot's own cameras recorded it. Time in the logs is
robot time: the arm holds still while the policy computes, and that pause is not in the log. The
tools' descriptions list every column.

# Actions that can be ordered

- `recalibrate` a camera: measure its pose again and correct it.
- `clean` a camera.
- `service` a component: inspect and repair it mechanically.
- `replace` a component.
- `revert_setting`: put one setting back to its nominal value. The target is the setting's name.
- `restart` the controller or the policy computer.
- `none`.

# Your task

Decide for each of the three robots whether it has a problem. A robot may be healthy; several may
have problems; the problems, if any, need not be alike. For a robot with a problem say which one
component is responsible, which setting if you believe a setting is off (otherwise an empty
string), what exactly is wrong, how sure you are (0 to 1), the evidence (episode ids and the numbers
or frames you relied on), and the one action you would order. For a healthy robot use component
`none`, action `none` and say briefly why you consider it healthy.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "robots": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "robot_id": {"type": "string", "enum": ["robot_1", "robot_2", "robot_3"]},
                    "has_problem": {"type": "boolean"},
                    "component": {"type": "string", "enum": COMPONENTS + ["none"]},
                    "setting": {"type": "string"},
                    "what_is_wrong": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "action": {
                        "type": "object",
                        "properties": {
                            "verb": {"type": "string", "enum": VERBS},
                            "target": {"type": "string"},
                        },
                        "required": ["verb", "target"],
                        "additionalProperties": False,
                    },
                },
                "required": ["robot_id", "has_problem", "component", "setting", "what_is_wrong", "confidence",
                             "evidence", "action"],
                "additionalProperties": False,
            },
        },
        "method": {"type": "string"},
    },
    "required": ["robots", "method"],
    "additionalProperties": False,
}


def tool_server(workspace: Path) -> ToolServer:
    return ToolServer(
        script=HERE / "fleet_tools.py",
        python=BASE / "venv" / "bin" / "python",
        workspace=workspace,
        tools=["query", "get_frames"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("out")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if args.check_only:
        print(SYSTEM + "\n\n" + PROMPT)
        return
    res = run_agent("claude", args.model, SYSTEM, PROMPT, images=[], tools=tool_server(Path(args.workspace)),
                    schema=ANSWER_SCHEMA, sandbox="bwrap", timeout=args.timeout)
    saved = {
        "ok": res.ok, "error": res.error, "error_detail": res.error_detail, "result": res.result,
        "usage": res.usage, "seconds": res.seconds, "model": res.model, "meta": res.meta,
        "tool_calls": res.tool_calls, "run_dir": res.run_dir,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(saved, indent=1, default=str))
    print(f"ok={res.ok} error={res.error} {res.seconds:.0f} s, model {res.model}, {len(res.tool_calls)} tool calls, "
          f"usage {res.usage}")
    if res.ok:
        for r in res.result["robots"]:
            print(f"  {r['robot_id']}: problem={r['has_problem']} component={r['component']} "
                  f"setting={r['setting']!r} confidence={r['confidence']} action={r['action']}")
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
