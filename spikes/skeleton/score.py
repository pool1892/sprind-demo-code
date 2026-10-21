"""The hidden fault table of a fleet run, and the mechanical score of a diagnosis against it.

    python score.py table MAIN_RUN_DIR                 write truth/fault_table.json (before any agent runs)
    python score.py score MAIN_RUN_DIR ANSWER.json...  score saved agent answers

The table maps each fault kind to the entry of the system description it
belongs to. It is fixed here, in code, before the first agent call, so that the
score cannot be bent toward what an agent happened to say.

What is scored per answer:

  right robot       a robot that carries a fault is reported as having a problem
  right component   ... and the reported component is the fault's component
  right setting     for a fault that is a setting: the reported setting is that key
  right action      the recommended verb and target are the fix in the table
  false alarm       a robot without a fault is reported as having a problem

Nothing here is a judgment: every cell is a comparison of two strings.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

# fault kind -> where it lives in the system description, and the generic action that removes it
TRUTH_OF_KIND = {
    "scene_camera_pan": {
        "component": "scene_camera",
        "setting": None,
        "action": {"verb": "recalibrate", "target": "scene_camera"},
    },
    "action_delay": {
        "component": "controller",
        "setting": "controller.command_buffer_steps",
        "action": {"verb": "revert_setting", "target": "controller.command_buffer_steps"},
    },
}


def write_table(run_dir: Path) -> dict:
    robots: dict[str, dict] = {}
    for line in (run_dir / "truth" / "episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        entry = robots.setdefault(e["robot"], {"faults": []})
        for f in e.get("faults") or []:
            item = {"fault": f, **TRUTH_OF_KIND[f["kind"]]}
            if item not in entry["faults"]:
                entry["faults"].append(item)
    table = {"robots": robots}
    (run_dir / "truth" / "fault_table.json").write_text(json.dumps(table, indent=1))
    canary = run_dir / "truth" / "canary.txt"
    if not canary.exists():
        # A marker that exists only on the truth side. It must never turn up in a workspace
        # or in anything an agent run leaves behind.
        canary.write_text("CANARY-" + secrets.token_hex(12) + "\n")
    return table


def score(table: dict, answer: dict) -> dict:
    said = {r["robot_id"]: r for r in answer["robots"]}
    rows, false_alarms = [], 0
    for robot, entry in sorted(table["robots"].items()):
        a = said.get(robot, {"has_problem": False, "component": "none", "setting": "", "action": {"verb": "none", "target": ""}})
        if not entry["faults"]:
            alarm = bool(a["has_problem"])
            false_alarms += alarm
            rows.append({"robot": robot, "faulted": False, "false_alarm": alarm,
                         "said_component": a["component"], "confidence": a.get("confidence")})
            continue
        t = entry["faults"][0]
        right_robot = bool(a["has_problem"])
        rows.append({
            "robot": robot, "faulted": True,
            "right_robot": right_robot,
            "right_component": right_robot and a["component"] == t["component"],
            "right_setting": None if t["setting"] is None else (right_robot and a.get("setting", "") == t["setting"]),
            "right_action": right_robot and a["action"]["verb"] == t["action"]["verb"]
            and a["action"]["target"] == t["action"]["target"],
            "said_component": a["component"], "said_setting": a.get("setting", ""),
            "said_action": a["action"], "confidence": a.get("confidence"),
            "truth_component": t["component"], "truth_setting": t["setting"], "truth_action": t["action"],
        })
    faulted = [r for r in rows if r["faulted"]]
    return {
        "rows": rows,
        "faults": len(faulted),
        "right_robot": sum(r["right_robot"] for r in faulted),
        "right_component": sum(r["right_component"] for r in faulted),
        "right_action": sum(r["right_action"] for r in faulted),
        "false_alarms": false_alarms,
    }


if __name__ == "__main__":
    cmd, run_dir = sys.argv[1], Path(sys.argv[2])
    if cmd == "table":
        print(json.dumps(write_table(run_dir), indent=1))
    elif cmd == "score":
        table = json.loads((run_dir / "truth" / "fault_table.json").read_text())
        for path in sys.argv[3:]:
            saved = json.loads(Path(path).read_text())
            if not saved.get("ok"):
                print(f"{path}: no valid answer ({saved.get('error')})")
                continue
            s = score(table, saved["result"])
            saved["score"] = s
            Path(path).write_text(json.dumps(saved, indent=1))
            u = saved["usage"]
            print(f"{Path(path).name}: right robot {s['right_robot']}/{s['faults']}, right component "
                  f"{s['right_component']}/{s['faults']}, right action {s['right_action']}/{s['faults']}, "
                  f"false alarms {s['false_alarms']}; {saved['seconds']:.0f} s, "
                  f"{u.get('input_tokens')} input and {u.get('output_tokens')} output tokens, "
                  f"{len(saved.get('tool_calls', []))} tool calls")
            for r in s["rows"]:
                print("   ", json.dumps({k: v for k, v in r.items() if not k.startswith("truth_")}))
    else:
        raise SystemExit(__doc__)
