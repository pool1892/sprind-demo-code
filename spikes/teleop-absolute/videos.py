"""Record thirteen take-overs from the campaign as videos, and check they are the measured episodes.

    run.sh videos.py

Picks episodes from the campaign's result files (fixed choices below, first
match in file order), runs each again with 256-pixel images and writes
`<name>.mp4` and `<name>_strip.png` to ~/.cache/robots-probes/teleop-absolute/videos/.
Physics does not depend on the image size, so every re-run must reproduce the
campaign row's chain (every action and every state) bit for bit; the script
says so per video and writes videos/index.json.

Each video shows the scene camera and the wrist camera side by side at 10
frames per second. The bar on top is orange while the set-up or X-VLA acts and
blue once the expert has taken over (after the snapshot was restored).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fleetlab  # noqa: F401  (torch before mujoco)

from determinism import SPEC_FIELDS
from episode import OUT, EpisodeSpec, Recorder, make_env, run_episode
from fleetlab.video import write_mp4
from tables import counted, handover_state, load, xvla_group
from try_one import write_strip
from xvla_record import load_episode, load_index

Pick = tuple[str, str, Callable[[dict[str, Any]], bool]]

PICKS: list[Pick] = [
    ("xvla_end_healthy_held", "c_xvla_end",
     lambda r: xvla_group(r) == "no fault" and handover_state(r) == "object between the fingers" and r["success"]),
    ("xvla_end_camera9_elsewhere", "c_xvla_end",
     lambda r: xvla_group(r) == "camera turned 9 deg" and handover_state(r) == "object elsewhere" and r["success"]),
    ("xvla_end_delay17", "c_xvla_end",
     lambda r: xvla_group(r).startswith("actions 17") and r["success"]),
    ("xvla_end_at_target", "c_xvla_end",
     lambda r: handover_state(r) == "object at the target, not counted" and r["success"]),
    ("xvla_end_failure", "c_xvla_end", lambda r: not r["success"]),
    ("xvla_end_drawer_pushed_in", "c_xvla_end", lambda r: r["xvla_episode"] == "c6-t4-i13"),
    ("xvla_mid_spatial_held", "c_xvla_mid",
     lambda r: r["suite"] == "libero_spatial" and handover_state(r) == "object between the fingers" and r["success"]),
    ("xvla_mid_object", "c_xvla_mid",
     lambda r: r["suite"] == "libero_object" and handover_state(r) == "object between the fingers" and r["success"]),
    ("dropped_spatial", "c_dropped", lambda r: r["suite"] == "libero_spatial" and r["success"]),
    ("on_side_object", "c_on_side", lambda r: r["suite"] == "libero_object" and r["success"]),
    ("closed_empty_spatial", "c_closed_empty", lambda r: r["suite"] == "libero_spatial" and r["success"]),
    ("awkward_twisted", "c_awkward", lambda r: r.get("setup", {}).get("kind") == "twisted" and r["success"]),
    ("inverted_bowl_failure", "c_inverted", lambda r: not r["success"]),
]


def main() -> None:
    out = OUT / "videos"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for name, source, want in PICKS:
        rows = [r for r in counted(load(source)) if want(r)]
        if not rows:
            print(f"{name}: no episode in {source} matches", flush=True)
            continue
        ref = rows[0]
        spec = EpisodeSpec(**{k: ref[k] for k in SPEC_FIELDS})
        recording = None
        if spec.scenario == "xvla":
            run_dir = Path(spec.xvla_run)
            recording = load_episode(run_dir, spec.xvla_episode, load_index(run_dir))
        env = make_env(spec.suite, spec.task, image_size=256)
        recorder = Recorder(every=2)
        row = run_episode(env, spec, recording, recorder)
        env.close()
        video = out / f"{name}.mp4"
        write_mp4(video, recorder.frames, fps=10)
        write_strip(recorder.frames, out / f"{name}_strip.png")
        same = row["chain"] == ref["chain"] and row["final_state_hash"] == ref["final_state_hash"]
        entry = dict(video=str(video), source=source, suite=spec.suite, task=spec.task, init=spec.init,
                     scenario=spec.scenario, xvla_episode=spec.xvla_episode, handover_step=spec.handover_step,
                     kind=ref.get("setup", {}).get("kind"), success=row["success"], status=row["status"],
                     expert_steps=row["steps"], same_as_campaign=same, events=row.get("events", []))
        index.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "events"}), flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{sum(e['same_as_campaign'] for e in index)} of {len(index)} videos show exactly the measured episode")


if __name__ == "__main__":
    main()
