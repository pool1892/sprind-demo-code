"""A learned library with a probe recipe for the miniature fleet w10-v2 (lane P's evidence).

    uv run --locked python spikes/instrument/recipe_dev_library.py OUT

Freezes, as the host freezes an investigation's skills (`loop.learned.freeze`, with the
leak audit against w10-v2's runs and fleet), one learned skill that carries the step probe
of the instrument spike and the query that reads it. The skill is written here by the
host, not by an agent: it is a fixture for running the recipe command on a development
world. Prints the library's root, for

    tools/heavy-run --light -m 10G uv run --locked python -m fleetlab.loop.probes \
        --library LIB --skill command-lag-probe --robot r05 --robot r01 \
        --world ~/.cache/robots-probes/mini/w10-v2 --out DIR --workers 3
"""

from __future__ import annotations

import sys
from pathlib import Path

WORLD = Path.home() / ".cache" / "robots-probes" / "mini" / "w10-v2"
FIRST_MOVES = (
    "WITH s AS (SELECT e.robot_id, e.repeat, st.episode_id, st.step, st.target_pos[1] AS "
    "target_x, st.eef_pos[1] AS hand_x, lag(st.target_pos[1]) OVER (PARTITION BY "
    "st.episode_id ORDER BY st.step) AS target_before FROM probe_steps AS st JOIN "
    "probe_episodes AS e USING (episode_id) WHERE e.probe_id = $probe AND e.robot_id IN "
    "($robot, $reference)), changes AS (SELECT episode_id, robot_id, repeat, step AS issued, "
    "target_x - target_before AS change, hand_x AS hand_at_issue FROM s WHERE target_before "
    "IS NOT NULL AND abs(target_x - target_before) > 1e-6) SELECT c.robot_id, c.repeat, "
    "c.issued, round(c.change * 1000, 1) AS change_mm, min(s.step) FILTER (WHERE "
    "sign(c.change) * (s.hand_x - c.hand_at_issue) >= 0.001) - c.issued AS first_move_rows, "
    "min(s.step) FILTER (WHERE sign(c.change) * (s.hand_x - c.hand_at_issue) >= "
    "abs(c.change) / 2) - c.issued AS half_way_rows FROM changes AS c JOIN s ON s.episode_id "
    "= c.episode_id AND s.step > c.issued GROUP BY c.robot_id, c.repeat, c.issued, c.change "
    "ORDER BY c.robot_id, c.repeat, c.issued"
)
STEP = [
    {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 10,
     "period_steps": None},
    {"primitive": "step", "axis": "x", "amplitude_m": 0.03, "duration_steps": 40,
     "period_steps": None},
    {"primitive": "step", "axis": "x", "amplitude_m": 0.0, "duration_steps": 40,
     "period_steps": None},
    {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 10,
     "period_steps": None},
]  # fmt: skip
DRAFT = {
    "name": "command-lag-probe",
    "description": "Tell a command delay apart from a slow or weak arm and from a motion "
    "path that is fine, with a step probe on a suspect robot and on one that does well.",
    "applies_when": "Robots at one place fail more often in motions that need timely "
    "corrections, and their motion records show the hand trailing its commands.",
    "parameters": [
        {"name": "probe", "meaning": "the id of the step probe below, once it has run"},
        {"name": "robot", "meaning": "a robot under suspicion"},
        {"name": "reference", "meaning": "a robot elsewhere that does well"},
    ],
    "evidence": [
        {
            "claim": "rows from each commanded step to the first movement and to half-way, "
            "on {robot} and on {reference}",
            "sql": FIRST_MOVES,
            "result": "{robot} moved many rows later than {reference} at both steps, the "
            "same in every repetition",
        }
    ],
    "alternatives": [
        {
            "explanation": "a controller that falls behind rather than a fixed delay",
            "how_to_tell": "its delay is longer at the second step than at the first and "
            "differs between repetitions",
        },
        {
            "explanation": "a weak arm",
            "how_to_tell": "it moves on time but settles short of the target and low",
        },
    ],
    "next_experiment": "Run the probe below on {robot} and {reference}, read it with the "
    "evidence query, then test the remedy its shape suggests in a task experiment.",
    "check_with": [],
    "probe": {
        "name": "step-lag-x",
        "purpose": "how long the hand of {robot} takes to follow a commanded step, against "
        "{reference}",
        "expectation": "a robot whose commands arrive at once moves in the next row",
        "robot_parameters": ["robot", "reference"],
        "repeats": 3,
        "segments": STEP,
    },
}

if __name__ == "__main__":
    from fleetlab.agents.sessions import Analysis
    from fleetlab.contracts.learned import LearnedSkillDraft
    from fleetlab.loop.learned import Source, freeze
    from fleetlab.loop.world import LoopWorld
    from fleetlab.toolserver.files import LEARNED_FILE, append_line

    out = Path(sys.argv[1])
    world = LoopWorld.load(WORLD)
    draft = LearnedSkillDraft.model_validate(DRAFT)
    analysis = Analysis.create(world.world_id, base=out / "analyses")
    record = {"receipt_id": "learned-1", "session": 1, "draft": draft.model_dump(mode="json")}
    append_line(analysis.outbox / LEARNED_FILE, record)
    source = Source(analysis.root, "the miniature fleet", 1, tuple(world.runs), world.model)
    report = freeze([source], out / "libraries", kept_on="2026-10-19")
    if report.library is None:
        raise SystemExit(f"refused: {report.refused}")
    print(report.library.root)
